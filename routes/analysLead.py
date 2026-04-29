import pandas as pd
from typing import Any, Dict, List
from prompts.Lead_prompt import classify_lead_status, BUCKET_ORDER
from .analytics_utils import (
    _schema_role_col, _find_col, _to_numeric, _schema_dim_col,
    _group_by_col, _monthly_trend, _agg_months, _round,
    _grouped_monthly_trend, _agg_grouped_months
)

def calculate_lead_metrics(dataset_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return LeadMetrics-compatible dict."""
    total_leads = 0.0
    qualified_leads = 0.0
    converted_leads = 0.0
    total_spend = 0.0
    all_scores: List[float] = []
    all_sources: List[Dict[str, Any]] = []
    all_source_revenue: List[Dict[str, Any]] = []
    total_revenue_sum: float = 0.0
    won_revenue_sum: float = 0.0
    qual_revenue_sum: float = 0.0
    source_stats: Dict[str, Dict[str, float]] = {}
    status_stats: Dict[str, Dict[str, float]] = {}
    source_labels: Dict[str, str] = {"won": "Won", "qualified": "Qualified"}
    all_monthly: List[Dict[str, Any]] = []
    all_grouped_monthly: List[List[Dict[str, Any]]] = []
    user_stats: Dict[str, Dict[str, float]] = {}

    global_won_count: float = 0.0
    global_qual_count: float = 0.0

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
            df, ["converted_leads", "conversions", "converted", "closed_won", "deals_closed", "won", "conv_status", "conv. status", "conversion_status", "active leads converted"]
        )
        spend_col = _schema_role_col(schema, "cost_total") or _find_col(
            df, ["spend", "cost", "total_spend", "budget", "ad_spend", "marketing_spend"]
        )
        score_col = _find_col(
            df, ["lead_score", "score", "rating", "quality_score", "priority_score"]
        )
        source_col = _schema_dim_col(schema, "source", "channel") or _find_col(
            df, [
                "source", "lead_source", "channel", "medium", "origin",
                "referral", "utm_source", "acquisition_source", "traffic_source",
            ],
        )
        status_col = _schema_dim_col(schema, "status") or _find_col(
            df, [
                "status", "stage", "lead_status", "pipeline_stage",
                "state", "phase", "lead_stage", "funnel_stage",
                "conv_status", "conversion_status"
            ],
        )
        rev_col = _schema_role_col(schema, "revenue_actual") or _find_col(
            df, [
                "revenue", "sales", "amount", "deal_value", "revenue_actual", 
                "income", "deal_size", "value", "deal size", "deal value", 
                "deal amount", "forecast_amount", "forecast amount (inr)"
            ],
        )
        owner_col = _schema_dim_col(schema, "owner") or _find_col(
            df, ["owner", "assigned_to", "sales_rep", "agent", "user", "owner_name", "created_by"]
        )
        date_col = _schema_dim_col(schema, "date") or _find_col(
            df, [
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

        if spend_col:
            v = _to_numeric(df[spend_col]).sum()
            if pd.notna(v):
                total_spend += float(v)

        if score_col:
            all_scores.extend(_to_numeric(df[score_col]).dropna().tolist())

        # ── Global Revenues ──────────────────────────────────────────────────
        if rev_col and rev_col in df.columns:
            total_revenue_sum += float(_to_numeric(df[rev_col]).sum())
            
            is_won_global = pd.Series([False] * len(df), index=df.index)
            if conv_col and conv_col in df.columns and conv_col != status_col:
                v_num = _to_numeric(df[conv_col])
                if v_num.isna().mean() < 0.8:
                    is_won_global = v_num > 0
                else:
                    is_won_global = df[conv_col].astype(str).str.lower().str.contains(r"won|convert|success|purchased", regex=True)
            elif status_col and status_col in df.columns:
                is_won_global = df[status_col].astype(str).apply(lambda v: classify_lead_status(v) == "Won")
            else:
                is_won_global = pd.Series([True] * len(df), index=df.index)
            
            if is_won_global.any():
                won_revenue_sum += float(_to_numeric(df[is_won_global][rev_col]).sum())

            is_qual_global = pd.Series([False] * len(df), index=df.index)
            if qual_col and qual_col in df.columns and qual_col != status_col:
                v_num = _to_numeric(df[qual_col])
                is_qual_global = v_num > 0
            elif status_col and status_col in df.columns:
                is_qual_global = df[status_col].astype(str).apply(lambda v: classify_lead_status(v) == "Qualified")

            if is_qual_global.any():
                qual_revenue_sum += float(_to_numeric(df[is_qual_global][rev_col]).sum())

        # ── Breakdowns ──────────────────────────────────────────────────────
        if source_col:
            all_sources.extend(_group_by_col(df, source_col, leads_col))
            if rev_col:
                all_source_revenue.extend(_group_by_col(df, source_col, rev_col))

            for src, group in df.groupby(source_col):
                src_name = str(src).strip()
                if not src_name or src_name.lower() in ["nan", "none"]:
                    continue

                if src_name not in source_stats:
                    source_stats[src_name] = {
                        "won": 0, "qualified": 0, "total": 0,
                        "revenue": 0, "cost": 0, "profit": 0, "totalRevenue": 0,
                    }

                if leads_col and leads_col in group.columns:
                    source_stats[src_name]["total"] += _to_numeric(group[leads_col]).sum()
                else:
                    source_stats[src_name]["total"] += len(group)

                if rev_col and rev_col in group.columns:
                    source_stats[src_name]["totalRevenue"] += _to_numeric(group[rev_col]).sum()

                won_added = False
                if conv_col and conv_col in group.columns and conv_col != status_col:
                    v_num = _to_numeric(group[conv_col])
                    if v_num.isna().mean() < 0.8:
                        v = v_num.sum()
                    else:
                        v = group[conv_col].astype(str).str.lower().str.contains(r"won|convert|success|purchased", regex=True).sum()
                    if v > 0:
                        source_stats[src_name]["won"] += v
                        won_added = True
                        source_labels["won"] = conv_col

                qual_added = False
                if qual_col and qual_col in group.columns and qual_col != status_col:
                    v = _to_numeric(group[qual_col]).sum()
                    if v > 0:
                        source_stats[src_name]["qualified"] += v
                        qual_added = True
                        source_labels["qualified"] = qual_col

                if status_col and (not won_added or not qual_added):
                    st_counts = group[status_col].astype(str).value_counts()
                    for st_orig, count in st_counts.items():
                        bucket = classify_lead_status(str(st_orig))
                        if not won_added and bucket == "Won":
                            source_stats[src_name]["won"] += count
                        elif not qual_added and bucket == "Qualified":
                            source_stats[src_name]["qualified"] += count

                if status_col and status_col in group.columns:
                    is_won = group[status_col].astype(str).apply(
                        lambda v: classify_lead_status(v) == "Won"
                    )
                else:
                    is_won = pd.Series([True] * len(group), index=group.index)
                won_data = group[is_won]

                won_rev = _to_numeric(won_data[rev_col]).sum() if rev_col and rev_col in won_data.columns else 0.0
                won_cost = _to_numeric(won_data[spend_col]).sum() if spend_col and spend_col in won_data.columns else 0.0

                source_stats[src_name]["revenue"] += won_rev
                source_stats[src_name]["cost"] += won_cost

                profit_col = _find_col(group, ["profit", "net_profit", "earnings", "net_income", "margin"])
                if profit_col:
                    source_stats[src_name]["profit"] += _to_numeric(group[profit_col]).sum()
                else:
                    source_stats[src_name]["profit"] += (won_rev - won_cost)



                if status_col and status_col in group.columns:
                    is_qual = group[status_col].astype(str).apply(
                        lambda v: classify_lead_status(v) == "Qualified"
                    )
                else:
                    is_qual = pd.Series([False] * len(group), index=group.index)

        if owner_col:
            for owner, group in df.groupby(owner_col):
                owner_name = str(owner).strip()
                if not owner_name or owner_name.lower() in ["nan", "none"]:
                    continue
                
                if owner_name not in user_stats:
                    user_stats[owner_name] = {"leads": 0.0, "revenue": 0.0}
                
                if leads_col and leads_col in group.columns:
                    user_stats[owner_name]["leads"] += _to_numeric(group[leads_col]).sum()
                else:
                    user_stats[owner_name]["leads"] += len(group)
                
                if rev_col and rev_col in group.columns:
                    user_stats[owner_name]["revenue"] += _to_numeric(group[rev_col]).sum()

        won_added_global = False
        if conv_col and conv_col in df.columns and conv_col != status_col:
            v_num = _to_numeric(df[conv_col])
            if v_num.isna().mean() < 0.8:
                v = v_num.sum()
            else:
                v = df[conv_col].astype(str).str.lower().str.contains(r"won|convert|success|purchased", regex=True).sum()
            if v > 0:
                global_won_count += v
                won_added_global = True
                source_labels["won"] = conv_col

        qual_added_global = False
        if qual_col and qual_col in df.columns and qual_col != status_col:
            v = _to_numeric(df[qual_col]).sum()
            if v > 0:
                global_qual_count += v
                qual_added_global = True
                source_labels["qualified"] = qual_col

        if status_col:
            for st, group in df.groupby(status_col):
                st_name = str(st).strip()
                if not st_name or st_name.lower() in ["nan", "none"]:
                    continue

                bucket = classify_lead_status(st_name)

                if not won_added_global and bucket == "Won":
                    global_won_count += len(group)
                elif not qual_added_global and bucket == "Qualified":
                    global_qual_count += len(group)

                if bucket not in status_stats:
                    status_stats[bucket] = {"count": 0.0, "revenue": 0.0}

                if leads_col and leads_col in group.columns:
                    status_stats[bucket]["count"] += _to_numeric(group[leads_col]).sum()
                else:
                    status_stats[bucket]["count"] += len(group)

                if rev_col and rev_col in group.columns:
                    status_stats[bucket]["revenue"] += _to_numeric(group[rev_col]).sum()

        if date_col:
            all_monthly.extend(_monthly_trend(df, date_col, leads_col))
            all_grouped_monthly.append(_grouped_monthly_trend(df, date_col, status_col, classify_lead_status))

    # ── Aggregate across datasets ────────────────────────────────────────────
    top_sources_list = sorted(
        source_stats.items(),
        key=lambda x: x[1]["total"],
        reverse=True,
    )[:10]

    # If we found explicit won/qual columns, they override the generic status column grouping
    if won_added_global and global_won_count > 0:
        if "Won" not in status_stats:
            status_stats["Won"] = {"count": 0.0, "revenue": 0.0}
        status_stats["Won"]["count"] = max(status_stats["Won"]["count"], global_won_count)

    if qual_added_global and global_qual_count > 0:
        if "Qualified" not in status_stats:
            status_stats["Qualified"] = {"count": 0.0, "revenue": 0.0}
        status_stats["Qualified"]["count"] = max(status_stats["Qualified"]["count"], global_qual_count)

    if status_stats:
        converted_leads = status_stats.get("Won", {}).get("count", 0.0)
        qualified_leads = status_stats.get("Qualified", {}).get("count", 0.0)
        won_revenue_sum  = max(won_revenue_sum,  status_stats.get("Won",       {}).get("revenue", 0.0))
        qual_revenue_sum = max(qual_revenue_sum, status_stats.get("Qualified", {}).get("revenue", 0.0))
    elif source_stats:
        qualified_leads = sum(s["qualified"] for s in source_stats.values())
        converted_leads = sum(s["won"] for s in source_stats.values())
    else:
        qualified_leads = global_qual_count
        converted_leads = global_won_count

    top_sources = []
    for name, stats in top_sources_list:
        total   = int(stats["total"])
        won     = int(stats["won"])
        qual    = int(stats["qualified"])
        contacted = max(0, total - (won + qual))
        top_sources.append({
            "source":       name,
            "won":          won,
            "qualified":    qual,
            "contacted":    contacted,
            "count":        total,
            "revenue":      _round(stats["revenue"]),
            "cost":         _round(stats["cost"]),
            "profit":       _round(stats["profit"]),
            "totalRevenue": _round(stats["totalRevenue"]),
            "percentage":   _round((total / total_leads * 100) if total_leads > 0 else 0),
        })

    # Emit by_status in canonical order driven by BUCKET_ORDER
    by_status = []
    for bucket in BUCKET_ORDER:
        if bucket in status_stats and status_stats[bucket]["count"] > 0:
            by_status.append({
                "status":  bucket,
                "count":   int(status_stats[bucket]["count"]),
                "revenue": _round(status_stats[bucket]["revenue"]),
            })

    contacted_revenue = max(0, total_revenue_sum - (won_revenue_sum + qual_revenue_sum))
    if not by_status:
        contacted = max(0, total_leads - (global_qual_count + global_won_count))
        if global_won_count > 0:
            by_status.append({"status": source_labels.get("won", "Won"),            "count": int(global_won_count),  "revenue": _round(won_revenue_sum)})
        if global_qual_count > 0:
            by_status.append({"status": source_labels.get("qualified", "Qualified"), "count": int(global_qual_count), "revenue": _round(qual_revenue_sum)})
        if contacted > 0:
            by_status.append({"status": "Contacted", "count": int(contacted), "revenue": _round(contacted_revenue)})

    month_data = _agg_months(all_monthly)
    # Generic trend for backward compatibility (if needed by other views)
    monthly_trend_legacy = [
        {"month": month_data["labels"][k], "leads": int(month_data["agg"][k])}
        for k in sorted(month_data["agg"].keys())
    ][-12:]

    # New grouped trend (Jan-Dec full year)
    monthly_trend = _agg_grouped_months(all_grouped_monthly)
    if not monthly_trend:
        monthly_trend = monthly_trend_legacy

    total_leads     = int(total_leads)
    qualified_leads = int(qualified_leads)
    converted_leads = int(converted_leads)

    conversion_rate = _round((converted_leads / total_leads * 100) if total_leads > 0 else 0)
    avg_lead_score  = _round(sum(all_scores) / len(all_scores)) if all_scores else 0.0
    cost_per_lead   = _round(total_spend / total_leads) if total_leads > 0 and total_spend > 0 else 0.0
    contacted_leads = max(0, total_leads - (qualified_leads + converted_leads))

    won_sorted = sorted([s for s in top_sources if s["count"] > 0], key=lambda x: x["won"], reverse=True)
    best_lead  = won_sorted[0]  if won_sorted          else None
    worst_lead = won_sorted[-1] if len(won_sorted) > 1 else None

    rev_sorted = sorted([s for s in top_sources if s["count"] > 0], key=lambda x: x["revenue"], reverse=True)
    best_revenue_lead = rev_sorted[0] if rev_sorted else None
    worst_revenue_lead = rev_sorted[-1] if len(rev_sorted) > 1 else None

    # Best User metrics
    best_user_lead = None
    best_user_revenue = None
    if user_stats:
        u_lead_sorted = sorted(user_stats.items(), key=lambda x: x[1]["leads"], reverse=True)
        if u_lead_sorted:
            best_user_lead = {"userName": u_lead_sorted[0][0], "leads": int(u_lead_sorted[0][1]["leads"])}
        
        u_rev_sorted = sorted(user_stats.items(), key=lambda x: x[1]["revenue"], reverse=True)
        if u_rev_sorted:
            best_user_revenue = {"userName": u_rev_sorted[0][0], "revenue": _round(u_rev_sorted[0][1]["revenue"])}

    return {
        "totalLeads":       total_leads,
        "qualifiedLeads":   qualified_leads,
        "convertedLeads":   converted_leads,
        "contactedLeads":   int(contacted_leads),
        "totalRevenue":     _round(total_revenue_sum),
        "wonRevenue":       _round(won_revenue_sum),
        "qualifiedRevenue": _round(qual_revenue_sum),
        "contactedRevenue": _round(contacted_revenue),
        "conversionRate":   conversion_rate,
        "avgLeadScore":     avg_lead_score,
        "costPerLead":      cost_per_lead,
        "topSources":       top_sources,
        "sourceLabels":     source_labels,
        "byStatus":         by_status,
        "monthlyTrend":     monthly_trend,
        "bestLead":         best_lead,
        "worstLead":        worst_lead,
        "bestRevenueLead":  best_revenue_lead,
        "worstRevenueLead": worst_revenue_lead,
        "bestUserLead":     best_user_lead,
        "bestUserRevenue":  best_user_revenue,
    }

