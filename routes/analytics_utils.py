"""
analytics_utils.py — Shared pure-Python helpers.

Imported by both analytics.py and analysLead.py.
Contains NO FastAPI, NO MongoDB, NO cross-route imports.
This breaks the circular-import chain.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


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
    Find the first DataFrame column matching any candidate name.
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
    """Group by calendar month. Returns list of {sort_key, month, value}."""
    if date_col not in df.columns:
        return []

    temp = df.copy()
    temp["_date"] = pd.to_datetime(temp[date_col], dayfirst=True, errors="coerce")
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
            "sort_key": str(period),
            "month": period.strftime("%b '%y"),
            "value": _round(val),
        })
    return result


def _agg_named_groups(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse {name, value} groups from multiple datasets by summing."""
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
