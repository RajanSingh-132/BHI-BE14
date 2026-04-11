"""
CalculationEngine — pure pandas/numpy arithmetic. Zero LLM involvement.

Receives:
  - data           : list of row dicts (the dataset)
  - query_intent   : parsed intent from the Intent Extractor LLM call
  - schema_profile : column profile from DatasetProfiler

Returns a CalculationResult dict that is passed directly to the Analysis LLM.
The Analysis LLM only receives computed numbers — it never sees raw rows.

CalculationResult shape:
{
  "metric":          str,
  "result":          float | None,      # scalar (None when breakdown is returned)
  "breakdown":       list[{group, value}],  # for group_by queries
  "lead_breakdown":  dict,              # for multi-lead queries
  "group_by_col":    str | None,
  "metric_col":      str | list | None,
  "formula":         str | None,
  "source":          "pre_computed" | "calculated" | "error",
  "unit":            str | None,
  "filter_applied":  str,
  "row_count":       int,
  "warnings":        list[str],
  "error":           str | None,
}
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from services.column_mapper import (
    resolve as resolve_column,
    get_dimension_col,
    find_value_dimension,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def calculate(
    data: List[Dict[str, Any]],
    query_intent: Dict,
    schema_profile: Dict,
) -> Dict:
    """
    Main calculation entry point.

    query_intent shape (from Intent Extractor LLM):
    {
      "metric":               str,    e.g. "roi", "revenue_actual", "leads_total"
      "filters":              list[{"field": str, "value": str}],
      "aggregation":          str,    "sum" | "avg" | "count" | "max" | "min" | "group_by" | "trend"
      "group_by":             str | None,   dimension type key
      "time_period":          str | None,
      "return_all_lead_types": bool,
    }
    """
    try:
        df = pd.DataFrame(data)
        if df.empty:
            return _error_result("Dataset is empty")

        metric             = (query_intent.get("metric") or "").strip()
        filters            = query_intent.get("filters") or []
        aggregation        = (query_intent.get("aggregation") or "sum").lower()
        group_by           = query_intent.get("group_by")
        return_all_leads   = bool(query_intent.get("return_all_lead_types", False))

        # 1. Apply filters first — reduces the working dataframe
        df, filter_desc = _apply_filters(df, filters, schema_profile)
        if df.empty:
            return _error_result(
                f"No data matches the applied filters: {filter_desc or 'unknown'}"
            )

        # 2. Resolve the column(s) for the requested metric
        col_res = resolve_column(metric, schema_profile, return_all_lead_types=return_all_leads)

        if col_res["missing"]:
            return _error_result(col_res.get("warning") or f"Cannot resolve metric: '{metric}'")

        # 3. Route to the correct calculation path
        if col_res["role"] == "leads_multi":
            return _calc_multi_leads(df, col_res, filter_desc, schema_profile, group_by)

        if col_res["derivable"]:
            return _calc_derived(df, col_res, metric, aggregation, filter_desc, schema_profile, group_by)

        return _calc_direct(df, col_res, metric, aggregation, filter_desc, schema_profile, group_by)

    except Exception as e:
        logger.error("[CALC_ENGINE] Unexpected error", exc_info=True)
        return _error_result(f"Calculation error: {str(e)}")


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

def _apply_filters(
    df: pd.DataFrame,
    filters: List[Dict],
    schema_profile: Dict,
) -> tuple:  # (filtered_df, description_str)
    desc_parts: List[str] = []

    for f in filters:
        field_type = (f.get("field") or "").strip()
        value      = (f.get("value") or "").strip()
        if not field_type or not value:
            continue

        # Try dimension_map first (fast path)
        col_name = get_dimension_col(field_type, schema_profile)

        # Fallback: scan all dimension_values for the value itself
        if not col_name:
            result = find_value_dimension(value, schema_profile)
            if result:
                col_name, value = result

        if col_name and col_name in df.columns:
            mask = df[col_name].astype(str).str.strip().str.lower() == value.lower()
            df = df[mask].copy()
            desc_parts.append(f"{col_name} = '{value}'")
        else:
            logger.warning(
                f"[CALC_ENGINE] Filter ignored — no column for field_type='{field_type}', value='{value}'"
            )

    return df, ", ".join(desc_parts) if desc_parts else "none"


# ---------------------------------------------------------------------------
# Direct column calculation  (column exists in dataset)
# ---------------------------------------------------------------------------

def _calc_direct(
    df:           pd.DataFrame,
    col_res:      Dict,
    metric:       str,
    aggregation:  str,
    filter_desc:  str,
    schema:       Dict,
    group_by:     Optional[str],
) -> Dict:
    col_name     = col_res["primary_col"]
    is_summable  = col_res["is_summable"]
    unit         = col_res["unit"]
    is_precomp   = col_res["is_precomputed"]
    warnings:    List[str] = []

    if col_name not in df.columns:
        return _error_result(f"Column '{col_name}' not found in dataframe")

    df = df.copy()
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    col_data = df[col_name].dropna()

    if col_data.empty:
        return _error_result(f"Column '{col_name}' has no numeric values")

    if col_data.sum() == 0 and len(col_data) > 0:
        warnings.append(
            f"'{col_name}' contains all zeros — data may not yet be available "
            f"(e.g. deals still in pipeline stage)"
        )

    # ---- Group-by path ----
    if group_by or aggregation == "group_by":
        group_col = get_dimension_col(group_by or "", schema)
        if group_col and group_col in df.columns:
            if is_summable:
                grouped = (
                    df.groupby(group_col)[col_name]
                    .sum()
                    .sort_values(ascending=False)
                )
                agg_label = "SUM"
            else:
                # Ratios (ROI %, CTR …) → mean per group
                grouped = (
                    df.groupby(group_col)[col_name]
                    .mean()
                    .round(2)
                    .sort_values(ascending=False)
                )
                agg_label = "AVG"

            breakdown = [
                {"group": str(k), "value": round(float(v), 2)}
                for k, v in grouped.items()
                if not (isinstance(v, float) and np.isnan(v))
            ]
            return {
                "metric":        metric,
                "result":        None,
                "breakdown":     breakdown,
                "lead_breakdown": {},
                "group_by_col":  group_col,
                "metric_col":    col_name,
                "formula":       f"{agg_label}({col_name}) GROUP BY {group_col}",
                "source":        "pre_computed" if is_precomp else "calculated",
                "unit":          unit,
                "filter_applied": filter_desc,
                "row_count":     len(df),
                "warnings":      warnings,
                "error":         None,
            }

    # ---- Scalar path ----
    result, formula = _aggregate_scalar(col_data, col_name, aggregation, is_summable)

    return {
        "metric":        metric,
        "result":        round(result, 2),
        "breakdown":     [],
        "lead_breakdown": {},
        "group_by_col":  None,
        "metric_col":    col_name,
        "formula":       formula,
        "source":        "pre_computed" if is_precomp else "calculated",
        "unit":          unit,
        "filter_applied": filter_desc,
        "row_count":     len(df),
        "warnings":      warnings,
        "error":         None,
    }


def _aggregate_scalar(col_data, col_name, aggregation, is_summable):
    if aggregation in ("sum", "total") and is_summable:
        return float(col_data.sum()), f"SUM({col_name})"
    elif aggregation == "avg" or not is_summable:
        return float(col_data.mean()), f"AVG({col_name})"
    elif aggregation == "max":
        return float(col_data.max()), f"MAX({col_name})"
    elif aggregation == "min":
        return float(col_data.min()), f"MIN({col_name})"
    elif aggregation == "count":
        return float(col_data.count()), f"COUNT({col_name})"
    else:
        # Default: sum if summable, average otherwise
        if is_summable:
            return float(col_data.sum()), f"SUM({col_name})"
        return float(col_data.mean()), f"AVG({col_name})"


# ---------------------------------------------------------------------------
# Derived calculation  (column must be computed from other columns)
# ---------------------------------------------------------------------------

def _calc_derived(
    df:           pd.DataFrame,
    col_res:      Dict,
    metric:       str,
    aggregation:  str,
    filter_desc:  str,
    schema:       Dict,
    group_by:     Optional[str],
) -> Dict:
    derivable     = col_res["derivable"]
    required_cols = derivable["required_cols"]   # {role: col_name}
    formula_label = derivable["formula_label"]
    warnings:     List[str] = []

    # Use the canonical semantic role for dispatch — the raw metric string may be a
    # natural-language alias (e.g. "click through rate") that won't match the formula keys.
    canonical_role = col_res["role"]

    df = df.copy()
    # Coerce all required columns to numeric
    for col_name in required_cols.values():
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        else:
            return _error_result(f"Required column '{col_name}' not found in dataframe")

    def _compute_on(sub_df: pd.DataFrame) -> Optional[float]:
        """Run the formula on a (possibly filtered) sub-dataframe."""
        try:
            if canonical_role == "roi":
                rev  = sub_df[required_cols["revenue_actual"]].sum()
                cost = sub_df[required_cols["cost_total"]].sum()
                return None if cost == 0 else round(((rev - cost) / cost) * 100, 2)

            elif canonical_role == "profit":
                rev  = sub_df[required_cols["revenue_actual"]].sum()
                cost = sub_df[required_cols["cost_total"]].sum()
                return round(float(rev - cost), 2)

            elif canonical_role == "ctr":
                clicks = sub_df[required_cols["clicks"]].sum()
                imps   = sub_df[required_cols["impressions"]].sum()
                return None if imps == 0 else round((clicks / imps) * 100, 4)

            elif canonical_role == "cpl":
                cost  = sub_df[required_cols["cost_total"]].sum()
                leads = sub_df[required_cols["leads_total"]].sum()
                return None if leads == 0 else round(float(cost / leads), 2)

            elif canonical_role in ("conversion_rate", "win_rate"):
                conv  = sub_df[required_cols["leads_converted"]].sum()
                leads = sub_df[required_cols["leads_total"]].sum()
                return None if leads == 0 else round((conv / leads) * 100, 2)

            elif canonical_role == "revenue_per_lead":
                rev   = sub_df[required_cols["revenue_actual"]].sum()
                leads = sub_df[required_cols["leads_total"]].sum()
                return None if leads == 0 else round(float(rev / leads), 2)

            elif canonical_role == "cost_per_conversion":
                cost  = sub_df[required_cols["cost_total"]].sum()
                convs = sub_df[required_cols["conversions"]].sum()
                return None if convs == 0 else round(float(cost / convs), 2)

        except Exception as e:
            logger.error(f"[CALC_ENGINE] _compute_on failed for {canonical_role}: {e}")
        return None

    # ---- Group-by path ----
    if group_by or aggregation == "group_by":
        group_col = get_dimension_col(group_by or "", schema)
        if group_col and group_col in df.columns:
            breakdown = []
            for group_val, sub_df in df.groupby(group_col):
                val = _compute_on(sub_df)
                if val is not None:
                    breakdown.append({"group": str(group_val), "value": val})
            breakdown.sort(key=lambda x: x["value"], reverse=True)
            return {
                "metric":        metric,
                "result":        None,
                "breakdown":     breakdown,
                "lead_breakdown": {},
                "group_by_col":  group_col,
                "metric_col":    list(required_cols.values()),
                "formula":       formula_label,
                "source":        "calculated",
                "unit":          col_res["unit"],
                "filter_applied": filter_desc,
                "row_count":     len(df),
                "warnings":      warnings,
                "error":         None,
            }

    # ---- Scalar path ----
    result = _compute_on(df)
    if result is None:
        return _error_result(
            f"Could not compute '{metric}' — possible division by zero or missing data"
        )

    return {
        "metric":        metric,
        "result":        result,
        "breakdown":     [],
        "lead_breakdown": {},
        "group_by_col":  None,
        "metric_col":    list(required_cols.values()),
        "formula":       formula_label,
        "source":        "calculated",
        "unit":          col_res["unit"],
        "filter_applied": filter_desc,
        "row_count":     len(df),
        "warnings":      warnings,
        "error":         None,
    }


# ---------------------------------------------------------------------------
# Multi-lead calculation
# ---------------------------------------------------------------------------

def _calc_multi_leads(
    df:          pd.DataFrame,
    col_res:     Dict,
    filter_desc: str,
    schema:      Dict,
    group_by:    Optional[str],
) -> Dict:
    lead_cols = col_res["lead_cols"]   # {role: {col, is_summable, unit}}
    df = df.copy()
    results: Dict[str, Any] = {}

    for role, info in lead_cols.items():
        col_name = info["col"]
        if col_name not in df.columns:
            continue
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        total = float(df[col_name].sum())
        results[role] = {
            "col":   col_name,
            "value": round(total, 2),
            "unit":  info["unit"],
        }

    return {
        "metric":        "leads_multi",
        "result":        None,
        "breakdown":     [],
        "lead_breakdown": results,
        "group_by_col":  None,
        "metric_col":    [v["col"] for v in results.values()],
        "formula":       "SUM per lead type",
        "source":        "calculated",
        "unit":          "count",
        "filter_applied": filter_desc,
        "row_count":     len(df),
        "warnings":      [],
        "error":         None,
    }


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

def _error_result(msg: str) -> Dict:
    logger.warning(f"[CALC_ENGINE] {msg}")
    return {
        "metric":        "error",
        "result":        None,
        "breakdown":     [],
        "lead_breakdown": {},
        "group_by_col":  None,
        "metric_col":    None,
        "formula":       None,
        "source":        "error",
        "unit":          None,
        "filter_applied": None,
        "row_count":     0,
        "warnings":      [msg],
        "error":         msg,
    }
