"""
ColumnMapper — deterministic resolution of metric_intent → actual column name.

Uses the schema_profile produced by DatasetProfiler.
No LLM involved. No guessing.

Key contract: given a metric string (e.g. "roi", "revenue_actual", "leads_total")
and a schema_profile, return the exact column(s) to query and whether they
are pre-computed in the dataset or must be derived.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Derivable metrics — these can be computed from other columns even when
# no direct column exists in the dataset.
# ---------------------------------------------------------------------------
DERIVABLE_METRICS: Dict[str, Dict] = {
    "roi": {
        "requires": ["revenue_actual", "cost_total"],
        "formula_label": "(Revenue - Cost) / Cost × 100",
        "unit": "percentage",
        "summable": False,
    },
    "profit": {
        "requires": ["revenue_actual", "cost_total"],
        "formula_label": "Revenue - Cost",
        "unit": "currency",
        "summable": True,
    },
    "ctr": {
        "requires": ["clicks", "impressions"],
        "formula_label": "(Clicks / Impressions) × 100",
        "unit": "percentage",
        "summable": False,
    },
    "cpl": {
        "requires": ["cost_total", "leads_total"],
        "formula_label": "Total Cost / Total Leads",
        "unit": "currency_rate",
        "summable": False,
    },
    "conversion_rate": {
        "requires": ["leads_converted", "leads_total"],
        "formula_label": "(Converted Leads / Total Leads) × 100",
        "unit": "percentage",
        "summable": False,
    },
    "win_rate": {
        "requires": ["leads_converted", "leads_total"],
        "formula_label": "(Deals Won / Total Leads) × 100",
        "unit": "percentage",
        "summable": False,
    },
    "revenue_per_lead": {
        "requires": ["revenue_actual", "leads_total"],
        "formula_label": "Total Revenue / Total Leads",
        "unit": "currency_rate",
        "summable": False,
    },
    "cost_per_conversion": {
        "requires": ["cost_total", "conversions"],
        "formula_label": "Total Cost / Total Conversions",
        "unit": "currency_rate",
        "summable": False,
    },
}

# All lead-type roles — returned together for generic "leads" queries
ALL_LEAD_ROLES = ["leads_total", "leads_qualified", "leads_converted", "opportunities"]

# Metric aliases — normalises what the LLM returns to canonical role names
METRIC_ALIASES: Dict[str, str] = {
    "revenue":            "revenue_actual",
    "total revenue":      "revenue_actual",
    "sales":              "revenue_actual",
    "sales revenue":      "revenue_actual",
    "earned revenue":     "revenue_actual",
    "earnings":           "revenue_actual",
    "income":             "revenue_actual",
    "expected revenue":   "revenue_expected",
    "pipeline revenue":   "revenue_expected",
    "projected revenue":  "revenue_expected",
    "cost":               "cost_total",
    "total cost":         "cost_total",
    "spend":              "cost_total",
    "marketing spend":    "cost_total",
    "campaign cost":      "cost_total",
    "cpl":                "cpl",
    "cost per lead":      "cpl",
    "leads":              "leads_total",
    "total leads":        "leads_total",
    "lead count":         "leads_total",
    "qualified leads":    "leads_qualified",
    "converted leads":    "leads_converted",
    "deals won":          "leads_converted",
    "closed won":         "leads_converted",
    "win rate":           "win_rate",
    "conversion rate":    "conversion_rate",
    "return on investment": "roi",
    "return on ad spend": "roi",
    "roas":               "roi",
    "click through rate": "ctr",
    "click-through rate": "ctr",
    "impressions":        "impressions",
    "clicks":             "clicks",
    "conversions":        "conversions",
    "profit":             "profit",
    "net profit":         "profit",
    "deal amount":        "deal_amount",
    "deal value":         "deal_amount",
    "opportunities":      "opportunities",
    "opps":               "opportunities",
    "revenue per lead":   "revenue_per_lead",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve(
    metric_intent: str,
    schema_profile: Dict,
    return_all_lead_types: bool = False,
) -> Dict:
    """
    Map a metric_intent string to one or more actual columns in schema_profile.

    Returns a resolution dict:
    {
      "primary_col":    str | None,       # direct column name (or None if derived)
      "role":           str,              # canonical semantic role
      "is_summable":    bool,
      "unit":           str | None,
      "is_precomputed": bool,             # True = column exists in dataset
      "derivable":      dict | None,      # not None = must be calculated
        → {"required_cols": {role: col_name}, "formula_label": str}
      "lead_cols":      dict,             # filled for leads_multi queries
      "missing":        bool,             # True = cannot resolve at all
      "warning":        str | None,
    }
    """
    # Normalise incoming intent
    canonical = _canonicalise(metric_intent)
    columns = schema_profile.get("columns", {})

    # ---- Generic "leads" → return all lead-type columns together ----
    if canonical in ("leads_total", "leads") and return_all_lead_types:
        return _resolve_all_leads(columns)

    # ---- Direct column match by semantic_role ----
    direct = _find_col_by_role(canonical, columns)
    if direct:
        meta = columns[direct]
        return {
            "primary_col":    direct,
            "role":           canonical,
            "is_summable":    meta.get("is_summable", True),
            "unit":           meta.get("unit"),
            "is_precomputed": meta.get("dtype") == "pre_computed_ratio",
            "derivable":      None,
            "lead_cols":      {},
            "missing":        False,
            "warning":        None,
        }

    # ---- CPL special case: "cpl" canonical may exist as a direct cost_per_unit column ----
    # The profiler assigns semantic_role="cost_per_unit" to pre-computed CPL columns
    # (e.g. "Cost Per Lead (₹)"). If the direct "cpl" role wasn't found above,
    # check for a cost_per_unit column before falling through to the derivable path.
    if canonical == "cpl":
        direct_cpu = _find_col_by_role("cost_per_unit", columns)
        if direct_cpu:
            meta = columns[direct_cpu]
            return {
                "primary_col":    direct_cpu,
                "role":           "cost_per_unit",
                "is_summable":    meta.get("is_summable", False),
                "unit":           meta.get("unit", "currency_rate"),
                "is_precomputed": False,
                "derivable":      None,
                "lead_cols":      {},
                "missing":        False,
                "warning":        None,
            }

    # ---- Derivable: build from other columns ----
    if canonical in DERIVABLE_METRICS:
        return _resolve_derivable(canonical, columns)

    # ---- Not found ----
    logger.warning(f"[COLUMN_MAPPER] Cannot resolve metric: '{metric_intent}' (canonical: '{canonical}')")
    return {
        "primary_col":    None,
        "role":           canonical,
        "is_summable":    False,
        "unit":           None,
        "is_precomputed": False,
        "derivable":      None,
        "lead_cols":      {},
        "missing":        True,
        "warning":        f"No column found for metric: '{metric_intent}'",
    }


def get_dimension_col(field_type: str, schema_profile: Dict) -> Optional[str]:
    """Resolve a dimension type label to the actual column name."""
    return schema_profile.get("dimension_map", {}).get(field_type)


def find_value_dimension(
    value: str, schema_profile: Dict
) -> Optional[Tuple[str, str]]:
    """
    Search all dimension_values to find which column contains a given filter value.

    Returns (col_name, matched_value_original_casing) or None.
    Uses rapidfuzz when available, falls back to case-insensitive exact match.
    """
    try:
        from rapidfuzz import process, fuzz as _fuzz
        use_fuzzy = True
    except ImportError:
        use_fuzzy = False

    for col_name, values in schema_profile.get("dimension_values", {}).items():
        str_values = [str(v) for v in values]
        if use_fuzzy:
            result = process.extractOne(
                value,
                str_values,
                scorer=_fuzz.ratio,
                score_cutoff=75,
            )
            if result:
                return col_name, result[0]
        else:
            for v in str_values:
                if v.lower() == value.lower():
                    return col_name, v

    return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _canonicalise(metric: str) -> str:
    """Lowercase + strip, then map through METRIC_ALIASES."""
    norm = metric.lower().strip()
    return METRIC_ALIASES.get(norm, norm)


def _find_col_by_role(role: str, columns: Dict) -> Optional[str]:
    """Return the first column name whose semantic_role matches `role`."""
    for col_name, meta in columns.items():
        if meta.get("semantic_role") == role:
            return col_name
    return None


def _resolve_derivable(canonical: str, columns: Dict) -> Dict:
    """Build a derivable resolution for metrics that need to be computed."""
    derivable_cfg = DERIVABLE_METRICS[canonical]
    required_cols: Dict[str, str] = {}
    missing_roles: List[str] = []

    for required_role in derivable_cfg["requires"]:
        found = _find_col_by_role(required_role, columns)
        if found:
            required_cols[required_role] = found
        else:
            missing_roles.append(required_role)

    if missing_roles:
        return {
            "primary_col":    None,
            "role":           canonical,
            "is_summable":    False,
            "unit":           None,
            "is_precomputed": False,
            "derivable":      None,
            "lead_cols":      {},
            "missing":        True,
            "warning":        f"Cannot compute '{canonical}': missing columns for {missing_roles}",
        }

    return {
        "primary_col":    None,
        "role":           canonical,
        "is_summable":    derivable_cfg["summable"],
        "unit":           derivable_cfg["unit"],
        "is_precomputed": False,
        "derivable": {
            "required_cols": required_cols,
            "formula_label": derivable_cfg["formula_label"],
        },
        "lead_cols":      {},
        "missing":        False,
        "warning":        None,
    }


def _resolve_all_leads(columns: Dict) -> Dict:
    """Build a multi-lead resolution for generic 'show me leads' queries."""
    lead_cols: Dict[str, Dict] = {}
    for col_name, meta in columns.items():
        if meta.get("semantic_role") in ALL_LEAD_ROLES:
            lead_cols[meta["semantic_role"]] = {
                "col":         col_name,
                "is_summable": meta.get("is_summable", True),
                "unit":        meta.get("unit", "count"),
            }

    if not lead_cols:
        return {
            "primary_col":    None,
            "role":           "leads_multi",
            "is_summable":    False,
            "unit":           "count",
            "is_precomputed": False,
            "derivable":      None,
            "lead_cols":      {},
            "missing":        True,
            "warning":        "No lead columns found in dataset",
        }

    return {
        "primary_col":    None,
        "role":           "leads_multi",
        "is_summable":    True,
        "unit":           "count",
        "is_precomputed": False,
        "derivable":      None,
        "lead_cols":      lead_cols,
        "missing":        False,
        "warning":        None,
    }
