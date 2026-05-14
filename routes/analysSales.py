import pandas as pd
from typing import Any, Dict, List
from .analytics_utils import (
    _schema_role_col, _find_col, _to_numeric, _schema_dim_col,
    _group_by_col, _monthly_trend, _agg_months, _round,
    _agg_named_groups
)
from prompts.Sales_prompt import classify_sales_status

def calculate_revenue_metrics(datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_revenue = 0.0
    total_spend = 0.0
    closed_deals = 0.0
    expected_revenue = 0.0
    all_deal_amounts = []
    all_regions = []
    all_monthly_indexed = []
    
    won_revenue = 0.0
    qualified_revenue = 0.0

    for entry in datasets:
        df = pd.DataFrame(entry.get("data") or [])
        if df.empty:
            continue
            
        schema = entry.get("schema", {})

        revenue_col = _schema_role_col(schema, "revenue_actual") or _find_col(
            df, [
                "forecast_amount", "forecast amount (inr)", "revenue", 
                "actual_revenue", "total_revenue", "sales", "income", 
                "amount", "deal_value", "revenue_actual"
            ],
        )
        expected_col = _schema_role_col(schema, "revenue_expected") or _find_col(
            df, [
                "expected_revenue", "pipeline_value", "projected_revenue",
                "forecast", "expected_value", "potential_revenue",
            ],
        )
        deal_col = _schema_role_col(schema, "deal_amount") or _find_col(
            df, [
                "deal_amount", "deal_size", "deal_value", "contract_value",
                "opportunity_value", "deal_revenue",
            ],
        )
        spend_col = _schema_role_col(schema, "cost_total") or _find_col(
            df, ["spend", "cost", "total_spend", "budget", "expense", "cost_total"]
        )
        closed_col = _find_col(
            df, ["closed_deals", "won_deals", "closed_won", "deals_closed", "num_won"],
        )
        region_col = _schema_dim_col(schema, "region", "territory") or _find_col(
            df, [
                "region", "territory", "area", "country", "state",
                "city", "location", "zone", "market", "geo",
            ],
        )
        date_col = _schema_dim_col(schema, "date", "period") or _find_col(
            df, [
                "date", "close_date", "month", "period", "quarter",
                "timestamp", "created_at", "record_date",
            ],
        )

        primary_rev_col = revenue_col or deal_col

        if primary_rev_col:
            v = _to_numeric(df[primary_rev_col]).sum()
            if pd.notna(v):
                total_revenue += float(v)
            all_deal_amounts.extend(_to_numeric(df[primary_rev_col]).dropna().tolist())

        prob_col = _schema_role_col(schema, "status_probability") or _find_col(
            df, ["probability", "deal_probability", "win_probability", "status_probability"]
        )
        is_conv_col = _schema_role_col(schema, "is_converted") or _find_col(
            df, ["converted", "is_converted", "is converted"]
        )

        if date_col and (deal_col or expected_col) and prob_col and is_conv_col:
            val_base = deal_col or expected_col
            df_p = df.copy()
            df_p["_val"]  = _to_numeric(df_p[val_base]).fillna(0)
            df_p["_prob"] = _to_numeric(df_p[prob_col]).fillna(0)
            if df_p["_prob"].max() > 1.1:
                df_p["_prob"] = df_p["_prob"] / 100.0
            mask = (
                df_p[date_col].notna()
                & (df_p[date_col].astype(str).str.strip() != "")
                & (df_p["_val"] > 0)
                & (df_p[is_conv_col].astype(str).str.lower().str.contains("yes|true|1", regex=True))
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
            status_col = _find_col(df, ["status", "stage", "deal_status"])
            if status_col:
                mask = df[status_col].astype(str).str.lower().str.contains(
                    r"closed|won|complet|success", regex=True
                )
                closed_deals += int(mask.sum())
            else:
                closed_deals += len(df)

        if region_col:
            if primary_rev_col is None:
                return {
                    "totalRevenue": 0, "avgDealSize": 0, "closedDeals": 0, "growthRate": 0, "roi": 0,
                    "byRegion": [], "monthlyRevenue": [], "bestRevenue": None, "worstRevenue": None,
                    "bestWonRegion": None, "worstWonRegion": None, "pipelineValue": 0, "wonRevenue": 0,
                    "qualifiedRevenue": 0
                }
            temp_reg = df.copy()
            temp_reg["_rev"] = _to_numeric(temp_reg[primary_rev_col]).fillna(0)
            temp_reg["_status"] = temp_reg[status_col].astype(str).apply(classify_sales_status) if status_col else "Contacted"
            temp_reg["_won_rev"] = temp_reg.apply(lambda r: r["_rev"] if r["_status"] == "Won" else 0, axis=1)
            
            for reg, group in temp_reg.groupby(region_col):
                reg_name = str(reg).strip()
                if not reg_name or reg_name.lower() in ["nan", "none"]:
                    continue
                all_regions.append({
                    "name": reg_name,
                    "value": float(group["_rev"].sum()),
                    "wonValue": float(group["_won_rev"].sum())
                })


        # Revenue buckets (Won, Qualified, Contacted)
        status_col = _find_col(df, ["status", "stage", "deal_status", "phase"])
        if status_col and primary_rev_col:
            temp_df = df.copy()
            temp_df["_rev"] = _to_numeric(temp_df[primary_rev_col]).fillna(0)
            temp_df["_status_bucket"] = temp_df[status_col].astype(str).apply(classify_sales_status)
            
            won_revenue += float(temp_df[temp_df["_status_bucket"] == "Won"]["_rev"].sum())
            qualified_revenue += float(temp_df[temp_df["_status_bucket"] == "Qualified"]["_rev"].sum())
            if date_col:
                temp_trend = df.copy()
                temp_trend["_date"] = pd.to_datetime(temp_trend[date_col], dayfirst=True, errors="coerce")
                if temp_trend["_date"].isna().any():
                    temp_trend["_date"] = temp_trend["_date"].fillna(pd.to_datetime(temp_trend[date_col], errors="coerce"))
                
                temp_trend = temp_trend.dropna(subset=["_date"])
                if not temp_trend.empty:
                    temp_trend["_m_idx"] = temp_trend["_date"].dt.month
                    temp_trend["_rev"] = _to_numeric(temp_trend[primary_rev_col]).fillna(0)
                    temp_trend["_status"] = temp_trend[status_col].astype(str).apply(classify_sales_status)
                    temp_trend["_is_won"] = temp_trend.apply(lambda r: r["_rev"] if r["_status"] == "Won" else 0, axis=1)
                    
                    m_sums = temp_trend.groupby("_m_idx").agg({
                        "_rev": "sum",
                        "_is_won": "sum"
                    })
                    for m_idx, row in m_sums.iterrows():
                        all_monthly_indexed.append({
                            "idx": int(m_idx), 
                            "revenue": float(row["_rev"]),
                            "wonRevenue": float(row["_is_won"])
                        })

    # Aggregate regional data across datasets
    reg_agg: Dict[str, Dict[str, float]] = {}
    for r in all_regions:
        name = r["name"]
        if name not in reg_agg:
            reg_agg[name] = {"total": 0.0, "won": 0.0}
        reg_agg[name]["total"] += r["value"]
        reg_agg[name]["won"] += r["wonValue"]

    by_region = sorted([
        {"region": name, "revenue": _round(stats["total"]), "wonRevenue": _round(stats["won"])}
        for name, stats in reg_agg.items()
    ], key=lambda x: -x["revenue"])[:10]

    # Best/Worst Won Region
    won_sorted = sorted([r for r in by_region if r["wonRevenue"] > 0], key=lambda x: x["wonRevenue"], reverse=True)
    best_won_region = won_sorted[0] if won_sorted else None
    worst_won_region = won_sorted[-1] if len(won_sorted) > 1 else None

    # Final aggregation for 12 months
    full_months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    monthly_revenue = []
    
    # Sum up all datasets by month index
    agg_rev = {i: 0.0 for i in range(1, 13)}
    agg_won = {i: 0.0 for i in range(1, 13)}
    for entry in all_monthly_indexed:
        agg_rev[entry["idx"]] += entry["revenue"]
        agg_won[entry["idx"]] += entry["wonRevenue"]

    for i in range(1, 13):
        monthly_revenue.append({
            "month": full_months[i-1],
            "revenue": _round(agg_rev[i]),
            "wonRevenue": _round(agg_won[i])
        })

    total_revenue = _round(total_revenue)
    avg_deal_size = (
        _round(sum(all_deal_amounts) / len(all_deal_amounts))
        if all_deal_amounts else 0.0
    )

    growth_rate = 0.0
    # Find last month with data and previous month to calculate growth
    active_months = [m for m in monthly_revenue if m["revenue"] > 0]
    if len(active_months) >= 2:
        prev = active_months[-2]["revenue"]
        curr = active_months[-1]["revenue"]
        if prev > 0:
            growth_rate = _round(((curr - prev) / prev) * 100)

    roi = (
        _round(((total_revenue - total_spend) / total_spend) * 100)
        if total_spend > 0 else 0.0
    )

    return {
        "totalRevenue":  total_revenue,
        "avgDealSize":   avg_deal_size,
        "closedDeals":   int(closed_deals),
        "growthRate":    growth_rate,
        "roi":           roi,
        "byRegion":      by_region,
        "monthlyRevenue": monthly_revenue,
        "bestRevenue":   by_region[0] if by_region else None,
        "worstRevenue":  by_region[-1] if len(by_region) > 1 else None,
        "bestWonRegion":  best_won_region,
        "worstWonRegion": worst_won_region,
        "pipelineValue": _round(expected_revenue),
        "wonRevenue":    _round(won_revenue),
        "qualifiedRevenue": _round(qualified_revenue)
    }
