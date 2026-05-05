"""
Analytics routes — returns frontend-compatible metric shapes.

GET  /api/analytics?type=leads|Sales|ads|summary
POST /api/submit-report

Every GET response is shaped as:
    { "metrics": { <FrontendShape> }, "warnings": [...] }

The summary type returns:
    { "leads": {...}, "Sales": {...}, "ads": {...}, "warnings": [...] }
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from mongo_client import mongo_client as _mongo
from prompts.Lead_prompt import classify_lead_status, BUCKET_ORDER
from .analytics_utils import (
    _round, _to_numeric, _find_col, _schema_role_col, _schema_dim_col,
    _group_by_col, _monthly_trend, _agg_named_groups, _agg_months,
    _grouped_monthly_trend, _agg_grouped_months
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ─── Session helpers ──────────────────────────────────────────────────────────

def _get_session_id(request: Request) -> str:
    sid = request.headers.get("x-session-id", "").strip()
    if not sid:
        sid = str(uuid.uuid4())
        logger.warning(f"[ANALYTICS] No X-Session-ID → fallback {sid!r}")
    return sid




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

from .analysLead import calculate_lead_metrics
from .analysSales import calculate_revenue_metrics
from .analysProductivity import calculate_productivity_metrics



# ─── ADS builder ─────────────────────────────────────────────────────────────

def _build_ads_metrics(dataset_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return AdsMetrics-compatible dict."""
    total_impressions = 0.0
    total_clicks      = 0.0
    total_conversions = 0.0
    total_spend       = 0.0
    total_revenue     = 0.0
    camp_agg: Dict[str, Dict[str, float]] = {}
    ch_agg:   Dict[str, Dict[str, float]] = {}

    for entry in dataset_entries:
        df = pd.DataFrame(entry["data"])
        if df.empty:
            continue
        schema = entry.get("schema", {})

        imp_col      = _schema_role_col(schema, "impressions") or _find_col(df, ["impressions", "total_impressions", "views", "reach", "ad_impressions"])
        clicks_col   = _schema_role_col(schema, "clicks")      or _find_col(df, ["clicks", "total_clicks", "link_clicks", "tap", "ad_clicks"])
        conv_col     = _schema_role_col(schema, "conversions") or _find_col(df, ["conversions", "total_conversions", "leads", "sign_ups", "purchases", "goals", "actions"])
        spend_col    = _schema_role_col(schema, "cost_total")  or _find_col(df, ["spend", "cost", "total_spend", "budget", "amount_spent", "ad_spend"])
        revenue_col  = _schema_role_col(schema, "revenue_actual") or _find_col(df, ["revenue", "total_revenue", "sales_revenue", "income", "returns"])
        campaign_col = _schema_dim_col(schema, "campaign")     or _find_col(df, ["campaign", "campaign_name", "ad_campaign", "utm_campaign", "ad_name", "ad_set", "campaign_id"])
        channel_col  = _schema_dim_col(schema, "channel", "source") or _find_col(df, ["channel", "platform", "network", "publisher", "ad_channel", "source", "medium", "ad_platform"])

        def safe_sum(col: Optional[str]) -> float:
            if not col or col not in df.columns:
                return 0.0
            v = _to_numeric(df[col]).sum()
            return float(v) if pd.notna(v) else 0.0

        total_impressions += safe_sum(imp_col)
        total_clicks      += safe_sum(clicks_col)
        total_conversions += safe_sum(conv_col)
        total_spend       += safe_sum(spend_col)
        total_revenue     += safe_sum(revenue_col)

        if campaign_col and campaign_col in df.columns:
            temp_camp = pd.DataFrame({
                "campaign": df[campaign_col].astype(str).str.strip(),
                "spend":    _to_numeric(df[spend_col]).fillna(0)   if spend_col  else pd.Series([0.0] * len(df)),
                "clicks":   _to_numeric(df[clicks_col]).fillna(0)  if clicks_col else pd.Series([0.0] * len(df)),
                "conv":     _to_numeric(df[conv_col]).fillna(0)    if conv_col   else pd.Series([0.0] * len(df)),
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
                camp_agg[k]["spend"]  += float(row["spend"])
                camp_agg[k]["clicks"] += float(row["clicks"])
                camp_agg[k]["conv"]   += float(row["conv"])

        if channel_col and channel_col in df.columns:
            temp_ch = pd.DataFrame({
                "channel": df[channel_col].astype(str).str.strip(),
                "spend":   _to_numeric(df[spend_col]).fillna(0)   if spend_col   else pd.Series([0.0] * len(df)),
                "revenue": _to_numeric(df[revenue_col]).fillna(0) if revenue_col else pd.Series([0.0] * len(df)),
                "clicks":  _to_numeric(df[clicks_col]).fillna(0)  if clicks_col  else pd.Series([0.0] * len(df)),
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
                ch_agg[k]["spend"]   += float(row["spend"])
                ch_agg[k]["revenue"] += float(row["revenue"])
                ch_agg[k]["clicks"]  += float(row["clicks"])

    by_campaign = sorted(
        [
            {
                "campaign":    k,
                "spend":       _round(v["spend"]),
                "clicks":      int(v["clicks"]),
                "conversions": int(v["conv"]),
            }
            for k, v in camp_agg.items()
        ],
        key=lambda x: -x["clicks"],
    )[:10]

    best_campaign  = by_campaign[0]  if by_campaign          else None
    worst_campaign = by_campaign[-1] if len(by_campaign) > 1 else None

    by_channel = sorted(
        [
            {
                "channel": k,
                "spend":   _round(v["spend"]),
                "roas":    _round(v["revenue"] / v["spend"]) if v["spend"] > 0 else 0.0,
            }
            for k, v in ch_agg.items()
        ],
        key=lambda x: -x["spend"],
    )[:10]

    avg_ctr       = _round((total_clicks / total_impressions * 100) if total_impressions > 0 else 0, 4)
    avg_cpc       = _round(total_spend / total_clicks               if total_clicks        > 0 else 0)
    roas          = _round(total_revenue / total_spend              if total_spend         > 0 else 0)
    cost_per_conv = _round(total_spend / total_conversions          if total_conversions   > 0 else 0)

    return {
        "totalSpend":        _round(total_spend),
        "totalImpressions":  _round(total_impressions),
        "totalClicks":       _round(total_clicks),
        "totalConversions":  _round(total_conversions),
        "avgCTR":            avg_ctr,
        "avgCPC":            avg_cpc,
        "roas":              roas,
        "costPerConversion": cost_per_conv,
        "byCampaign":        by_campaign,
        "byChannel":         by_channel,
        "bestCampaign":      best_campaign,
        "worstCampaign":     worst_campaign,
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/analytics")
async def get_analytics(
    request: Request,
    analytics_type: Literal["leads", "Sales", "revenue", "Productivity", "ads", "summary"] = Query(
        ...,
        alias="type",
        description="Analytics type: leads | Sales | Productivity | summary",
    ),
    file_name: Optional[str] = Query(None, description="Optional specific dataset to analyze"),
):
    """
    Return analytics for uploaded dataset(s) in current session.
    Response shape: { "metrics": <FrontendMetrics>, "warnings": [...] }
    Summary shape : { "leads": {...}, "revenue": {...}, "ads": {...}, "warnings": [...] }
    """
    session_id = _get_session_id(request)
    session_state = _mongo.get_session_state(session_id)
    active_datasets = session_state.get("active_datasets", []) or []

    if file_name and file_name in active_datasets:
        active_datasets = [file_name]


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
        return {"metrics": calculate_lead_metrics(dataset_entries), "warnings": warnings}

    if analytics_type in ["revenue", "Sales"]:
        return {"metrics": calculate_revenue_metrics(dataset_entries), "warnings": warnings}

    if analytics_type == "ads":
        return {"metrics": _build_ads_metrics(dataset_entries), "warnings": warnings}

    if analytics_type == "Productivity":
        return {"metrics": calculate_productivity_metrics(dataset_entries), "warnings": warnings}

    # summary
    return {
        "leads":        {"metrics": calculate_lead_metrics(dataset_entries)},
        "Sales":        {"metrics": calculate_revenue_metrics(dataset_entries)},
        "Productivity": {"metrics": calculate_productivity_metrics(dataset_entries)},
        "warnings":     warnings,
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
        raise HTTPException(status_code=404, detail="No dataset found for this session.")

    db = request.app.state.mongo.db
    dataset_entries, warnings = _load_session_datasets(db, active_datasets)

    if not dataset_entries:
        raise HTTPException(
            status_code=404,
            detail="Cannot find uploaded dataset records for report generation.",
        )

    submitted_at = datetime.datetime.utcnow().isoformat() + "Z"

    report: Dict[str, Any] = {
        "session_id":   session_id,
        "datasets":     active_datasets,
        "submitted_at": submitted_at,
        "leads":        calculate_lead_metrics(dataset_entries),
        "Sales":        calculate_revenue_metrics(dataset_entries),
        "Productivity": calculate_productivity_metrics(dataset_entries),
        "warnings":     warnings,
    }

    db["reports"].insert_one(report)
    logger.info(f"[SUBMIT] Report saved for session={session_id!r} datasets={active_datasets}")

    return {
        "status":       "success",
        "message":      "Analysis report submitted and saved successfully.",
        "session_id":   session_id,
        "datasets":     active_datasets,
        "submitted_at": submitted_at,
    }