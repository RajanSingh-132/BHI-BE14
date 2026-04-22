"""
Analytics routes — returns frontend-compatible metric shapes.

GET  /api/analytics?type=leads|revenue|ads|summary
POST /api/submit-report

Every GET response is shaped as:
    { "metrics": { <FrontendShape> }, "warnings": [...] }

The summary type returns:
    { "leads": {...}, "revenue": {...}, "ads": {...}, "warnings": [...] }
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from mongo_client import mongo_client as _mongo

logger = logging.getLogger(__name__)
router = APIRouter()

# ─── Session helpers ──────────────────────────────────────────────────────────

def _get_session_id(request: Request) -> str:
    sid = request.headers.get("x-session-id", "").strip()
    if not sid:
        sid = str(uuid.uuid4())
        logger.warning(f"[ANALYTICS] No X-Session-ID → fallback {sid!r}")
    return sid


# ─── Numeric helpers ──────────────────────────────────────────────────────────

def _round(value: Any, digits: int = 2) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[,\$₹€£%\s]", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


# ─── Column detection ─────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find the first DataFrame column that matches any candidate name.
    Tries exact → normalized (underscore/dash/space folded) → substring match.
    """
    def norm(s: str) -> str:
        return s.lower().replace("_", " ").replace("-", " ").strip()

    norm_to_orig = {norm(c): c for c in df.columns}
    lower_to_orig = {c.lower(): c for c in df.columns}

    for cand in candidates:
        found = lower_to_orig.get(cand.lower()) or norm_to_orig.get(norm(cand))
        if found:
            return found

    # substring partial match
    for cand in candidates:
        nc = norm(cand)
        for col in df.columns:
            nc_col = norm(col)
            if nc in nc_col or nc_col in nc:
                return col

    return None


def _schema_role_col(schema: Dict[str, Any], *roles: str) -> Optional[str]:
    columns = (schema or {}).get("columns", {})
    for role in roles:
        for col_name, meta in columns.items():
            if (meta or {}).get("semantic_role") == role:
                return col_name
    return None


def _schema_dim_col(schema: Dict[str, Any], *dims: str) -> Optional[str]:
    dim_map = (schema or {}).get("dimension_map", {})
    for dim in dims:
        col = dim_map.get(dim)
        if col:
            return col
    return None


# ─── Aggregation helpers ──────────────────────────────────────────────────────

def _group_by_col(
    df: pd.DataFrame,
    group_col: str,
    value_col: Optional[str] = None,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Sum value_col (or count rows) grouped by group_col. Returns top_n desc."""
    if group_col not in df.columns:
        return []

    temp = pd.DataFrame({"group": df[group_col].astype(str).str.strip()})

    if value_col and value_col in df.columns:
        temp["val"] = _to_numeric(df[value_col])
    else:
        temp["val"] = 1.0

    temp = temp[
        temp["group"].notna()
        & (temp["group"] != "")
        & (temp["group"].str.lower() != "nan")
        & (temp["group"].str.lower() != "none")
    ]
    if temp.empty:
        return []

    grouped = (
        temp.groupby("group")["val"]
        .sum()
        .dropna()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return [
        {"name": str(k), "value": _round(v)}
        for k, v in grouped.items()
        if pd.notna(v)
    ]


def _monthly_trend(
    df: pd.DataFrame,
    date_col: str,
    value_col: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Group by calendar month. Returns list of {"sort_key":..., "month":..., "value":...}."""
    if date_col not in df.columns:
        return []

    temp = df.copy()
    # Robust date parsing: handles DD/MM/YYYY, MM/DD/YYYY, and YYYY/MM/DD
    # dayfirst=True tells pandas to prefer DD/MM/YYYY when ambiguous (e.g. 01/02/2023)
    temp["_date"] = pd.to_datetime(temp[date_col], dayfirst=True, errors="coerce")
    
    # If some rows failed to parse (e.g. they were MM/DD/YYYY but dayfirst=True caused an issue or they are YYYY/MM/DD), 
    # pandas usually handles them anyway, but we ensure maximum coverage.
    if temp["_date"].isna().any():
        fallback = pd.to_datetime(temp[date_col], errors="coerce")
        temp["_date"] = temp["_date"].fillna(fallback)

    temp = temp.dropna(subset=["_date"])
    if temp.empty:
        return []

    temp["_month"] = temp["_date"].dt.to_period("M")

    if value_col and value_col in df.columns:
        temp["_val"] = _to_numeric(temp[value_col])
        grouped = temp.groupby("_month")["_val"].sum()
    else:
        grouped = temp.groupby("_month")["_date"].count()

    result = []
    for period, val in grouped.sort_index().tail(12).items():
        result.append({
            "sort_key": str(period),          # "2025-01" — used for cross-dataset merge
            "month": period.strftime("%b '%y"),  # "Jan '25"
            "value": _round(val),
        })
    return result


def _agg_named_groups(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse {"name":..., "value":...} groups from multiple datasets by summing."""
    agg: Dict[str, float] = {}
    for g in groups:
        agg[g["name"]] = agg.get(g["name"], 0.0) + float(g["value"])
    return sorted(
        [{"name": k, "value": v} for k, v in agg.items()],
        key=lambda x: -x["value"],
    )


def _agg_months(all_monthly: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate monthly items from multiple datasets by sort_key."""
    agg: Dict[str, float] = {}
    labels: Dict[str, str] = {}
    for m in all_monthly:
        sk = m["sort_key"]
        agg[sk] = agg.get(sk, 0.0) + float(m["value"])
        labels[sk] = m["month"]
    return {"agg": agg, "labels": labels}


# ─── MongoDB data loading ─────────────────────────────────────────────────────

def _load_session_datasets(
    db: Any,
    active_datasets: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    docs = list(
        db["documents"].find(
            {"type": "dataset", "file_name": {"$in": active_datasets}},
            {"_id": 0, "file_name": 1, "data": 1, "rows": 1},
        )
    )
    schemas = list(
        db["schema_profiles"].find(
            {"file_name": {"$in": active_datasets}},
            {"_id": 0},
        )
    )
    docs_by_file = {d["file_name"]: d for d in docs}
    schemas_by_file = {s["file_name"]: s for s in schemas}

    entries: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for fn in active_datasets:
        doc = docs_by_file.get(fn)
        if not doc:
            warnings.append(
                f"Dataset '{fn}' is in session but not found in documents collection."
            )
            continue
        schema = schemas_by_file.get(fn, {})
        if not schema:
            warnings.append(
                f"Schema profile missing for '{fn}'. Column detection may be reduced."
            )
        entries.append({
            "file_name": fn,
            "dataset_type": schema.get("dataset_type", "unknown"),
            "row_count": int(doc.get("rows") or len(doc.get("data") or [])),
            "data": doc.get("data") or [],
            "schema": schema,
        })

    return entries, warnings


# ─── LEADS builder ────────────────────────────────────────────────────────────

def _build_leads_metrics(dataset_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return LeadMetrics-compatible dict."""
    total_leads = 0.0
    qualified_leads = 0.0
    converted_leads = 0.0
    total_spend = 0.0
    all_scores: List[float] = []
    all_sources: List[Dict[str, Any]] = []
    all_statuses: List[Dict[str, Any]] = []
    all_monthly: List[Dict[str, Any]] = []

    for entry in dataset_entries:
        df = pd.DataFrame(entry["data"])
        if df.empty:
            continue
        schema = entry.get("schema", {})

        leads_col = _schema_role_col(schema, "leads_total") or _find_col(
            df, ["leads", "total_leads", "lead_count", "new_leads", "num_leads", "lead_volume"]
        )
        qual_col = _schema_role_col(schema, "leads_qualified") or _find_col(
            df, ["qualified_leads", "qualified", "mql", "sql", "marketing_qualified"]
        )
        conv_col = _schema_role_col(schema, "leads_converted", "conversions") or _find_col(
            df, ["converted_leads", "conversions", "converted", "closed_won", "deals_closed", "won"]
        )
        spend_col = _schema_role_col(schema, "cost_total") or _find_col(
            df, ["spend", "cost", "total_spend", "budget", "ad_spend", "marketing_spend"]
        )
        score_col = _find_col(
            df, ["lead_score", "score", "rating", "quality_score", "priority_score"]
        )
        source_col = _schema_dim_col(schema, "source", "channel") or _find_col(
            df,
            [
                "source", "lead_source", "channel", "medium", "origin",
                "referral", "utm_source", "acquisition_source", "traffic_source",
            ],
        )
        status_col = _schema_dim_col(schema, "status") or _find_col(
            df,
            [
                "status", "stage", "lead_status", "pipeline_stage",
                "state", "phase", "lead_stage", "funnel_stage",
            ],
        )
        date_col = _schema_dim_col(schema, "date") or _find_col(
            df,
            [
                "date", "created_at", "created_date", "month",
                "timestamp", "period", "time", "week", "record_date",
            ],
        )

        # ── Totals ──────────────────────────────────────────────────────────
        if leads_col:
            v = _to_numeric(df[leads_col]).sum()
            if pd.notna(v):
                total_leads += float(v)
        else:
            total_leads += len(df)

        if qual_col:
            v = _to_numeric(df[qual_col]).sum()
            if pd.notna(v):
                qualified_leads += float(v)

        if conv_col:
            v = _to_numeric(df[conv_col]).sum()
            if pd.notna(v):
                converted_leads += float(v)

        if spend_col:
            v = _to_numeric(df[spend_col]).sum()
            if pd.notna(v):
                total_spend += float(v)

        if score_col:
            all_scores.extend(
                _to_numeric(df[score_col]).dropna().tolist()
            )

        # ── Breakdowns ──────────────────────────────────────────────────────
        if source_col:
            all_sources.extend(_group_by_col(df, source_col, leads_col))

        if status_col:
            all_statuses.extend(_group_by_col(df, status_col, leads_col))

        if date_col:
            all_monthly.extend(_monthly_trend(df, date_col, leads_col))

    # ── Aggregate across datasets ────────────────────────────────────────────
    sources_agg = _agg_named_groups(all_sources)[:10]
    statuses_agg = _agg_named_groups(all_statuses)[:10]

    source_total = sum(s["value"] for s in sources_agg) or 1.0
    top_sources = [
        {
            "source": s["name"],
            "count": int(s["value"]),
            "percentage": _round((s["value"] / source_total) * 100),
        }
        for s in sources_agg
    ]
    by_status = [{"status": s["name"], "count": int(s["value"])} for s in statuses_agg]

    month_data = _agg_months(all_monthly)
    monthly_trend = [
        {"month": month_data["labels"][k], "leads": int(month_data["agg"][k])}
        for k in sorted(month_data["agg"].keys())
    ][-12:]

    total_leads = int(total_leads)
    qualified_leads = int(qualified_leads)
    converted_leads = int(converted_leads)

    conversion_rate = _round((converted_leads / total_leads * 100) if total_leads > 0 else 0)
    avg_lead_score = _round(sum(all_scores) / len(all_scores)) if all_scores else 0.0
    cost_per_lead = _round(total_spend / total_leads) if total_leads > 0 and total_spend > 0 else 0.0

    best_lead = top_sources[0] if top_sources else None
    worst_lead = top_sources[-1] if len(top_sources) > 1 else None

    return {
        "totalLeads": total_leads,
        "qualifiedLeads": qualified_leads,
        "convertedLeads": converted_leads,
        "conversionRate": conversion_rate,
        "avgLeadScore": avg_lead_score,
        "costPerLead": cost_per_lead,
        "topSources": top_sources,
        "byStatus": by_status,
        "monthlyTrend": monthly_trend,
        "bestLead": best_lead,
        "worstLead": worst_lead,
    }


# ─── REVENUE builder ──────────────────────────────────────────────────────────

def _build_revenue_metrics(dataset_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return RevenueMetrics-compatible dict."""
    total_revenue = 0.0
    expected_revenue = 0.0
    total_spend = 0.0
    closed_deals = 0
    all_deal_amounts: List[float] = []
    all_regions: List[Dict[str, Any]] = []
    all_monthly: List[Dict[str, Any]] = []

    for entry in dataset_entries:
        df = pd.DataFrame(entry["data"])
        if df.empty:
            continue
        schema = entry.get("schema", {})

        revenue_col = _schema_role_col(schema, "revenue_actual") or _find_col(
            df,
            [
                "revenue", "actual_revenue", "total_revenue", "sales",
                "income", "amount", "deal_value", "revenue_actual",
            ],
        )
        expected_col = _schema_role_col(schema, "revenue_expected") or _find_col(
            df,
            [
                "expected_revenue", "pipeline_value", "projected_revenue",
                "forecast", "expected_value", "potential_revenue",
            ],
        )
        deal_col = _schema_role_col(schema, "deal_amount") or _find_col(
            df,
            [
                "deal_amount", "deal_size", "deal_value", "contract_value",
                "opportunity_value", "deal_revenue",
            ],
        )
        spend_col = _schema_role_col(schema, "cost_total") or _find_col(
            df, ["spend", "cost", "total_spend", "budget", "expense", "cost_total"]
        )
        closed_col = _find_col(
            df,
            ["closed_deals", "won_deals", "closed_won", "deals_closed", "num_won"],
        )
        region_col = _schema_dim_col(schema, "region", "territory") or _find_col(
            df,
            [
                "region", "territory", "area", "country", "state",
                "city", "location", "zone", "market", "geo",
            ],
        )
        date_col = _schema_dim_col(schema, "date", "period") or _find_col(
            df,
            [
                "date", "close_date", "month", "period", "quarter",
                "timestamp", "created_at", "record_date",
            ],
        )

        primary_rev_col = revenue_col or deal_col

        if primary_rev_col:
            v = _to_numeric(df[primary_rev_col]).sum()
            if pd.notna(v):
                total_revenue += float(v)
            all_deal_amounts.extend(
                _to_numeric(df[primary_rev_col]).dropna().tolist()
            )

        # ── Pipeline Revenue (New Formula) ───────────────────────────────────
        # Formula: Σ (Deal Value × Status Probability) WHERE Converted=Yes, Value>0, Close Date exists
        prob_col = _schema_role_col(schema, "status_probability") or _find_col(
            df, ["probability", "deal_probability", "win_probability", "status_probability"]
        )
        is_conv_col = _schema_role_col(schema, "is_converted") or _find_col(
            df, ["converted", "is_converted", "is converted"]
        )

        if date_col and (deal_col or expected_col) and prob_col and is_conv_col:
            # Use deal_col if available, otherwise expected_col as the base value
            val_base = deal_col or expected_col
            df_p = df.copy()
            df_p["_val"] = _to_numeric(df_p[val_base]).fillna(0)
            df_p["_prob"] = _to_numeric(df_p[prob_col]).fillna(0)
            # Normalize probability (assume 0-100 if max > 1.1)
            if df_p["_prob"].max() > 1.1:
                df_p["_prob"] = df_p["_prob"] / 100.0
            
            mask = (
                df_p[date_col].notna() &
                (df_p[date_col].astype(str).str.strip() != "") &
                (df_p["_val"] > 0) &
                (df_p[is_conv_col].astype(str).str.lower().str.contains("yes|true|1", regex=True))
            )
            expected_revenue += float((df_p[mask]["_val"] * df_p[mask]["_prob"]).sum())

        if spend_col:
            v = _to_numeric(df[spend_col]).sum()
            if pd.notna(v):
                total_spend += float(v)

        if closed_col:
            v = _to_numeric(df[closed_col]).sum()
            if pd.notna(v):
                closed_deals += int(v)
        else:
            # Count rows where status indicates closed
            status_col = _find_col(df, ["status", "stage", "deal_status"])
            if status_col:
                mask = df[status_col].astype(str).str.lower().str.contains(
                    r"clos|won|complet|success", regex=True
                )
                closed_deals += int(mask.sum())
            else:
                closed_deals += len(df)

        if region_col:
            all_regions.extend(_group_by_col(df, region_col, primary_rev_col))

        if date_col:
            all_monthly.extend(_monthly_trend(df, date_col, primary_rev_col))

    regions_agg = _agg_named_groups(all_regions)[:10]
    by_region = [
        {"region": r["name"], "revenue": _round(r["value"])}
        for r in regions_agg
    ]

    month_data = _agg_months(all_monthly)
    monthly_revenue = [
        {"month": month_data["labels"][k], "revenue": _round(month_data["agg"][k])}
        for k in sorted(month_data["agg"].keys())
    ][-12:]

    total_revenue = _round(total_revenue)
    avg_deal_size = (
        _round(sum(all_deal_amounts) / len(all_deal_amounts))
        if all_deal_amounts
        else 0.0
    )

    growth_rate = 0.0
    if len(monthly_revenue) >= 2:
        first = monthly_revenue[0]["revenue"]
        last = monthly_revenue[-1]["revenue"]
        if first > 0:
            growth_rate = _round(((last - first) / first) * 100)

    # Pipeline Revenue = Σ (Deal Value × Status Probability) WHERE Converted=Yes, Value>0, Close Date exists
    # Note: Using expected_revenue as the accumulator for the specific formula calculated during dataset loops
    pipeline_value = _round(expected_revenue) if expected_revenue > 0 else 0.0
    roi = (
        _round(((total_revenue - total_spend) / total_spend) * 100)
        if total_spend > 0
        else 0.0
    )

    best_revenue = by_region[0] if by_region else None
    worst_revenue = by_region[-1] if len(by_region) > 1 else None

    return {
        "totalRevenue": total_revenue,
        "avgDealSize": avg_deal_size,
        "closedDeals": closed_deals,
        "pipelineValue": pipeline_value,
        "growthRate": growth_rate,
        "totalSpend": _round(total_spend),
        "roi": roi,
        "byRegion": by_region,
        "monthlyRevenue": monthly_revenue,
        "bestRevenue": best_revenue,
        "worstRevenue": worst_revenue,
    }


# ─── ADS builder ─────────────────────────────────────────────────────────────

def _build_ads_metrics(dataset_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return AdsMetrics-compatible dict."""
    total_impressions = 0.0
    total_clicks = 0.0
    total_conversions = 0.0
    total_spend = 0.0
    total_revenue = 0.0
    camp_agg: Dict[str, Dict[str, float]] = {}
    ch_agg: Dict[str, Dict[str, float]] = {}

    for entry in dataset_entries:
        df = pd.DataFrame(entry["data"])
        if df.empty:
            continue
        schema = entry.get("schema", {})

        imp_col = _schema_role_col(schema, "impressions") or _find_col(
            df, ["impressions", "total_impressions", "views", "reach", "ad_impressions"]
        )
        clicks_col = _schema_role_col(schema, "clicks") or _find_col(
            df, ["clicks", "total_clicks", "link_clicks", "tap", "ad_clicks"]
        )
        conv_col = _schema_role_col(schema, "conversions") or _find_col(
            df,
            [
                "conversions", "total_conversions", "leads", "sign_ups",
                "purchases", "goals", "actions",
            ],
        )
        spend_col = _schema_role_col(schema, "cost_total") or _find_col(
            df,
            ["spend", "cost", "total_spend", "budget", "amount_spent", "ad_spend"],
        )
        revenue_col = _schema_role_col(schema, "revenue_actual") or _find_col(
            df, ["revenue", "total_revenue", "sales_revenue", "income", "returns"]
        )
        campaign_col = _schema_dim_col(schema, "campaign") or _find_col(
            df,
            [
                "campaign", "campaign_name", "ad_campaign", "utm_campaign",
                "ad_name", "ad_set", "campaign_id",
            ],
        )
        channel_col = _schema_dim_col(schema, "channel", "source") or _find_col(
            df,
            [
                "channel", "platform", "network", "publisher",
                "ad_channel", "source", "medium", "ad_platform",
            ],
        )

        def safe_sum(col: Optional[str]) -> float:
            if not col or col not in df.columns:
                return 0.0
            v = _to_numeric(df[col]).sum()
            return float(v) if pd.notna(v) else 0.0

        total_impressions += safe_sum(imp_col)
        total_clicks += safe_sum(clicks_col)
        total_conversions += safe_sum(conv_col)
        total_spend += safe_sum(spend_col)
        total_revenue += safe_sum(revenue_col)

        # ── Campaign breakdown ───────────────────────────────────────────────
        if campaign_col and campaign_col in df.columns:
            temp_camp = pd.DataFrame({
                "campaign": df[campaign_col].astype(str).str.strip(),
                "spend":   _to_numeric(df[spend_col]).fillna(0) if spend_col else pd.Series([0.0] * len(df)),
                "clicks":  _to_numeric(df[clicks_col]).fillna(0) if clicks_col else pd.Series([0.0] * len(df)),
                "conv":    _to_numeric(df[conv_col]).fillna(0) if conv_col else pd.Series([0.0] * len(df)),
            })
            temp_camp = temp_camp[
                temp_camp["campaign"].notna()
                & (temp_camp["campaign"] != "")
                & (temp_camp["campaign"].str.lower() != "nan")
            ]
            for _, row in temp_camp.iterrows():
                k = row["campaign"]
                if k not in camp_agg:
                    camp_agg[k] = {"spend": 0.0, "clicks": 0.0, "conv": 0.0}
                camp_agg[k]["spend"] += float(row["spend"])
                camp_agg[k]["clicks"] += float(row["clicks"])
                camp_agg[k]["conv"] += float(row["conv"])

        # ── Channel breakdown ────────────────────────────────────────────────
        if channel_col and channel_col in df.columns:
            temp_ch = pd.DataFrame({
                "channel": df[channel_col].astype(str).str.strip(),
                "spend":   _to_numeric(df[spend_col]).fillna(0) if spend_col else pd.Series([0.0] * len(df)),
                "revenue": _to_numeric(df[revenue_col]).fillna(0) if revenue_col else pd.Series([0.0] * len(df)),
                "clicks":  _to_numeric(df[clicks_col]).fillna(0) if clicks_col else pd.Series([0.0] * len(df)),
            })
            temp_ch = temp_ch[
                temp_ch["channel"].notna()
                & (temp_ch["channel"] != "")
                & (temp_ch["channel"].str.lower() != "nan")
            ]
            for _, row in temp_ch.iterrows():
                k = row["channel"]
                if k not in ch_agg:
                    ch_agg[k] = {"spend": 0.0, "revenue": 0.0, "clicks": 0.0}
                ch_agg[k]["spend"] += float(row["spend"])
                ch_agg[k]["revenue"] += float(row["revenue"])
                ch_agg[k]["clicks"] += float(row["clicks"])

    # ── Finalise campaign list ───────────────────────────────────────────────
    by_campaign = sorted(
        [
            {
                "campaign": k,
                "spend": _round(v["spend"]),
                "clicks": int(v["clicks"]),
                "conversions": int(v["conv"]),
            }
            for k, v in camp_agg.items()
        ],
        key=lambda x: -x["clicks"],
    )[:10]

    # Best = highest clicks, Worst = lowest clicks
    best_campaign = by_campaign[0] if by_campaign else None
    worst_campaign = by_campaign[-1] if len(by_campaign) > 1 else None

    # ── Finalise channel list ────────────────────────────────────────────────
    by_channel = sorted(
        [
            {
                "channel": k,
                "spend": _round(v["spend"]),
                "roas": _round(v["revenue"] / v["spend"]) if v["spend"] > 0 else 0.0,
            }
            for k, v in ch_agg.items()
        ],
        key=lambda x: -x["spend"],
    )[:10]

    avg_ctr = _round((total_clicks / total_impressions * 100) if total_impressions > 0 else 0, 4)
    avg_cpc = _round(total_spend / total_clicks if total_clicks > 0 else 0)
    roas = _round(total_revenue / total_spend if total_spend > 0 else 0)
    cost_per_conv = _round(total_spend / total_conversions if total_conversions > 0 else 0)

    return {
        "totalSpend": _round(total_spend),
        "totalImpressions": _round(total_impressions),
        "totalClicks": _round(total_clicks),
        "totalConversions": _round(total_conversions),
        "avgCTR": avg_ctr,
        "avgCPC": avg_cpc,
        "roas": roas,
        "costPerConversion": cost_per_conv,
        "byCampaign": by_campaign,
        "byChannel": by_channel,
        "bestCampaign": best_campaign,
        "worstCampaign": worst_campaign,
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/analytics")
async def get_analytics(
    request: Request,
    analytics_type: Literal["leads", "revenue", "ads", "summary"] = Query(
        ...,
        alias="type",
        description="Analytics type: leads | revenue | ads | summary",
    ),
):
    """
    Return analytics for uploaded dataset(s) in current session.
    Response shape: { "metrics": <FrontendMetrics>, "warnings": [...] }
    Summary shape : { "leads": {...}, "revenue": {...}, "ads": {...}, "warnings": [...] }
    """
    session_id = _get_session_id(request)
    session_state = _mongo.get_session_state(session_id)
    active_datasets = session_state.get("active_datasets", []) or []

    if not active_datasets:
        raise HTTPException(
            status_code=404,
            detail="No dataset found for this session. Please upload a dataset first.",
        )

    db = request.app.state.mongo.db
    dataset_entries, warnings = _load_session_datasets(db, active_datasets)

    if not dataset_entries:
        raise HTTPException(
            status_code=404,
            detail="Session dataset records could not be read. Please re-upload the dataset.",
        )

    if analytics_type == "leads":
        return {
            "metrics": _build_leads_metrics(dataset_entries),
            "warnings": warnings,
        }

    if analytics_type == "revenue":
        return {
            "metrics": _build_revenue_metrics(dataset_entries),
            "warnings": warnings,
        }

    if analytics_type == "ads":
        return {
            "metrics": _build_ads_metrics(dataset_entries),
            "warnings": warnings,
        }

    # summary
    return {
        "leads":   {"metrics": _build_leads_metrics(dataset_entries)},
        "revenue": {"metrics": _build_revenue_metrics(dataset_entries)},
        "ads":     {"metrics": _build_ads_metrics(dataset_entries)},
        "warnings": warnings,
    }


@router.post("/submit-report")
async def submit_report(request: Request):
    """
    Persist the completed analysis report to MongoDB.
    Called from the Summary page when the user clicks 'Submit Analysis'.
    """
    session_id = _get_session_id(request)
    session_state = _mongo.get_session_state(session_id)
    active_datasets = session_state.get("active_datasets", []) or []

    if not active_datasets:
        raise HTTPException(
            status_code=404,
            detail="No dataset found for this session.",
        )

    db = request.app.state.mongo.db
    dataset_entries, warnings = _load_session_datasets(db, active_datasets)

    if not dataset_entries:
        raise HTTPException(
            status_code=404,
            detail="Cannot find uploaded dataset records for report generation.",
        )

    submitted_at = datetime.datetime.utcnow().isoformat() + "Z"

    report: Dict[str, Any] = {
        "session_id": session_id,
        "datasets": active_datasets,
        "submitted_at": submitted_at,
        "leads": _build_leads_metrics(dataset_entries),
        "revenue": _build_revenue_metrics(dataset_entries),
        "ads": _build_ads_metrics(dataset_entries),
        "warnings": warnings,
    }

    db["reports"].insert_one(report)
    logger.info(f"[SUBMIT] Report saved for session={session_id!r} datasets={active_datasets}")

    return {
        "status": "success",
        "message": "Analysis report submitted and saved successfully.",
        "session_id": session_id,
        "datasets": active_datasets,
        "submitted_at": submitted_at,
    }
