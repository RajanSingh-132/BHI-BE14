"""
DatasetProfiler — classifies every column at upload time.

Outputs a schema_profile dict stored in MongoDB.
This is the single source of truth for all downstream column resolution
so that the LLM never has to guess column semantics again.

Column dtype hierarchy (evaluated in order):
  identifier → contact → date → numeric_measure/pre_computed_ratio → categorical/name
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SEMANTIC ONTOLOGY
#
# Each role maps to:
#   keywords    : normalized aliases (lowercase, no currency/unit symbols)
#   exclude_if  : disqualify this role if any of these substrings appear
#                 in the normalized column name
#   summable    : whether SUM() is meaningful (False for rates/ratios)
#   unit        : currency | percentage | count | currency_rate | score
# ---------------------------------------------------------------------------
SEMANTIC_ONTOLOGY: Dict[str, Dict] = {
    # ---- Revenue ----
    "revenue_actual": {
        "keywords": [
            "revenue", "revenue generated", "sales revenue", "revenue earned",
            "actual revenue", "income", "earnings", "total revenue",
        ],
        "exclude_if": ["expected", "projected", "forecast", "potential", "pipeline", "per "],
        "summable": True,
        "unit": "currency",
    },
    "revenue_expected": {
        "keywords": [
            "expected revenue", "projected revenue", "forecast revenue",
            "potential revenue", "pipeline revenue", "estimated revenue",
        ],
        "exclude_if": [],
        "summable": True,
        "unit": "currency",
    },
    "deal_amount": {
        "keywords": [
            "deal amount", "deal value", "deal size", "opportunity value",
            "contract value", "opportunity amount",
        ],
        "exclude_if": [],
        "summable": True,
        "unit": "currency",
    },
    # ---- Cost ----
    "cost_total": {
        "keywords": [
            "campaign cost", "marketing spend", "spend", "total spend",
            "total cost", "budget spend", "cost", "ad spend", "advertising spend",
        ],
        "exclude_if": ["per lead", "per click", "per unit", " cpc", " cpl",
                       "per acquisition", "avg", "average", "daily", "per "],
        "summable": True,
        "unit": "currency",
    },
    "cost_per_unit": {
        "keywords": [
            "cost per lead", "cpl", "cpc", "avg cpc", "cost per click",
            "cost per acquisition", "average cost", "cost per conversion",
        ],
        "exclude_if": [],
        "summable": False,
        "unit": "currency_rate",
    },
    "budget": {
        "keywords": [
            "budget", "daily budget", "budget allocated", "allocated budget",
        ],
        "exclude_if": ["spend"],
        "summable": True,
        "unit": "currency",
    },
    # ---- Leads ----
    "leads_total": {
        "keywords": [
            "total leads", "leads generated", "leads", "lead count",
            "number of leads", "new leads",
        ],
        "exclude_if": ["qualified", "converted", "won", " id", "name", "source",
                       "status", "owner", "stage"],
        "summable": True,
        "unit": "count",
    },
    "leads_qualified": {
        "keywords": [
            "qualified leads", "mql", "marketing qualified lead",
            "sales qualified lead", "sql",
        ],
        "exclude_if": [],
        "summable": True,
        "unit": "count",
    },
    "leads_converted": {
        "keywords": [
            "converted leads", "closed won leads", "deals won", "won leads",
            "closed won", "conversions won",
        ],
        "exclude_if": [],
        "summable": True,
        "unit": "count",
    },
    "opportunities": {
        "keywords": [
            "opportunities created", "opportunities", "pipeline opportunities", "opps",
        ],
        "exclude_if": [],
        "summable": True,
        "unit": "count",
    },
    # ---- Ads performance ----
    "impressions": {
        "keywords": ["impressions", "views", "reach", "ad impressions"],
        "exclude_if": [],
        "summable": True,
        "unit": "count",
    },
    "clicks": {
        "keywords": ["clicks", "link clicks", "ad clicks", "total clicks"],
        "exclude_if": ["avg", "average", "cpc"],
        "summable": True,
        "unit": "count",
    },
    "conversions": {
        "keywords": ["conversions", "goals", "actions", "goal completions"],
        "exclude_if": ["rate", "converted leads", "won", "closed"],
        "summable": True,
        "unit": "count",
    },
    # ---- Financial ratios (pre-computed) ----
    "profit": {
        "keywords": ["profit", "net profit", "gross profit", "margin amount", "net income"],
        "exclude_if": ["margin %", "margin pct", "profit %", "profit pct", "pct", " %"],
        "summable": True,
        "unit": "currency",
    },
    "roi": {
        "keywords": ["roi", "return on investment", "roas", "return on ad spend", "return on spend"],
        "exclude_if": [],
        "summable": False,
        "unit": "percentage",
    },
    "ctr": {
        "keywords": ["ctr", "click through rate", "click-through rate", "click rate"],
        "exclude_if": [],
        "summable": False,
        "unit": "percentage",
    },
    "nps": {
        "keywords": ["nps", "net promoter score", "promoter score"],
        "exclude_if": [],
        "summable": False,
        "unit": "score",
    },
}

# ---------------------------------------------------------------------------
# DIMENSION ONTOLOGY — categorical columns for filtering / grouping
# ---------------------------------------------------------------------------
DIMENSION_ONTOLOGY: Dict[str, List[str]] = {
    "campaign":  ["campaign name", "campaign", "ad campaign"],
    "channel":   ["marketing channel", "channel", "source channel", "ad channel"],
    "source":    ["lead source", "source", "traffic source", "referral source"],
    "status":    ["lead status", "status", "deal stage", "stage", "pipeline stage",
                  "meeting status", "lead qualifies"],
    "owner":     ["owner", "sales rep", "assigned to", "rep", "agent name"],
    # "client" = the lead/contact person (individual)
    "client":    ["client", "user name", "client name", "account",
                  "client user name", "lead name", "prospect name"],
    # "company" = the organisation the lead belongs to (separate from individual name)
    "company":   ["company", "company name", "organisation", "organization",
                  "account name", "firm", "business"],
    "ad_group":  ["ad group", "adgroup", "audience"],
    "date":      ["month", "date", "period", "close month", "created date",
                  "deal close date", "lead generation date", "meeting date",
                  "next step email date", "follow up email date", "last follow-up"],
    "city":      ["city", "location", "region", "area"],
}

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_col(col_name: str) -> str:
    """Lowercase; strip currency symbols, parens, units; collapse whitespace."""
    s = str(col_name).lower()
    s = re.sub(r"[₹$€£%()#]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Column type classifiers
# ---------------------------------------------------------------------------

def _is_identifier(col_name: str, values: List[Any]) -> bool:
    """True when column is a surrogate key / ID field — not a measure."""
    norm = _normalize_col(col_name)

    # Never classify date / time columns as identifiers even if values look numeric
    date_keywords = ["month", "date", "year", "period", "close month",
                     "created date", "deal close date"]
    if any(kw in norm for kw in date_keywords):
        return False

    id_patterns = [r"\bid\b", r"\bids\b", r"_id$", r"^id_", r"\b(click id|google click)\b"]
    for pat in id_patterns:
        if re.search(pat, norm):
            return True

    # Values look like alphanumeric codes (L0001, SF-L1000, LID1000, CMP1000 …)
    # Guard: month strings like "Mar-2026" or "Jan-2026" must NOT match.
    clean = [v for v in values if v is not None]
    if clean and all(isinstance(v, str) for v in clean[:10]):
        # Reject if values look like "Mon-YYYY" (month abbreviation + 4-digit year)
        month_like = re.compile(r"^[A-Za-z]{3}-\d{4}$")
        if any(month_like.match(str(v).strip()) for v in clean[:3]):
            return False
        code_like = all(
            re.match(r"^[A-Za-z]{0,6}[-_]?\d+$", str(v).strip())
            for v in clean[:5]
        )
        if code_like:
            return True
    return False


def _is_contact(col_name: str) -> bool:
    norm = _normalize_col(col_name)
    return any(kw in norm for kw in [
        "email", "phone", "mobile", "address", "url",
        "contact number", "contact no", "contact num",
        "prospect contact", "tel ", "telephone",
    ])


def _is_name_field(col_name: str) -> bool:
    """
    True only for purely personal / entity name fields that carry no
    dimension meaning (e.g. 'Lead Name', 'Contact Name').

    'Campaign Name', 'Client/User Name', 'Company Name' are intentionally
    excluded here — they are dimension columns that will be caught by
    _detect_dimension_type() first.
    """
    norm = _normalize_col(col_name)
    # Only match if the word before "name" is a person/entity label,
    # not a dimension descriptor like campaign, company, client, user.
    _dimension_prefixes = [
        "campaign", "company", "client", "user", "channel",
        "source", "owner", "rep", "group", "ad",
    ]
    if any(pfx in norm for pfx in _dimension_prefixes):
        return False
    return norm.endswith(" name") or norm == "name"


def _get_raw_dtype(values: List[Any]) -> str:
    """Infer underlying dtype from sample values."""
    clean = [v for v in values if v is not None]
    if not clean:
        return "string"
    numeric_count = sum(1 for v in clean if isinstance(v, (int, float)))
    if numeric_count / len(clean) > 0.8:
        return "numeric"
    return "string"


def _is_date_col(col_name: str) -> bool:
    norm = _normalize_col(col_name)
    date_keywords = ["date", "month", "year", "day", "period",
                     "close month", "created date", "deal close date"]
    return any(kw in norm for kw in date_keywords)


# ---------------------------------------------------------------------------
# Semantic role detection
# ---------------------------------------------------------------------------

def _detect_semantic_role(
    col_name: str, values: List[Any]
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Returns (semantic_role, unit, is_summable).

    Strategy:
      1. Check exclude_if rules — disqualify roles before scoring
      2. Score normalized col name against all keywords in the ontology
      3. Take best score above threshold
    """
    try:
        from rapidfuzz import fuzz
        _fuzz = fuzz
    except ImportError:
        logger.warning("[PROFILER] rapidfuzz not available — using substring fallback")
        _fuzz = None

    norm = _normalize_col(col_name)

    best_role: Optional[str] = None
    best_score: float = 0.0

    for role, config in SEMANTIC_ONTOLOGY.items():
        # --- exclusion check ---
        if any(excl in norm for excl in config.get("exclude_if", [])):
            continue

        for keyword in config["keywords"]:
            if _fuzz:
                score = float(_fuzz.ratio(norm, keyword))
            else:
                # Substring containment fallback
                if keyword == norm:
                    score = 100.0
                elif keyword in norm or norm in keyword:
                    score = 80.0
                elif any(w in norm for w in keyword.split() if len(w) > 3):
                    score = 60.0
                else:
                    score = 0.0

            if score > best_score:
                best_score = score
                best_role = role

    THRESHOLD = 55.0
    if best_score < THRESHOLD:
        # Generic numeric — summable but no semantic role assigned
        return None, "count", True

    cfg = SEMANTIC_ONTOLOGY[best_role]
    return best_role, cfg["unit"], cfg["summable"]


def _detect_dimension_type(col_name: str) -> Optional[str]:
    """Maps a categorical column to a known dimension type."""
    try:
        from rapidfuzz import fuzz
        _fuzz = fuzz
    except ImportError:
        _fuzz = None

    norm = _normalize_col(col_name)
    best_dim: Optional[str] = None
    best_score: float = 0.0

    for dim_type, keywords in DIMENSION_ONTOLOGY.items():
        for kw in keywords:
            if _fuzz:
                score = float(_fuzz.ratio(norm, kw))
            else:
                score = 100.0 if kw == norm else (70.0 if kw in norm else 0.0)
            if score > best_score:
                best_score = score
                best_dim = dim_type

    return best_dim if best_score >= 60.0 else None


# ---------------------------------------------------------------------------
# Dataset-type inference
# ---------------------------------------------------------------------------

def _detect_dataset_type(column_roles: Dict[str, str]) -> str:
    """Infer the high-level dataset category from detected roles."""
    roles = set(column_roles.values())

    if "impressions" in roles or "clicks" in roles or "ctr" in roles:
        return "google_ads"

    if "roi" in roles and "leads_total" in roles and "cost_total" in roles:
        return "marketing_campaign"

    # Aggregate marketing/campaign datasets that have total counts + ROI
    if ("leads_total" in roles or "opportunities" in roles) and "cost_total" in roles:
        return "marketing_campaign"

    if "deal_amount" in roles or "opportunities" in roles:
        return "sales_pipeline"

    if "leads_total" in roles and "cost_total" in roles:
        return "lead_generation"

    # Row-level CRM/lead files: per-lead cost + expected or actual revenue
    # e.g. Salesforce Lead (Cost Per Lead + Expected Revenue)
    # e.g. Zoho CRM     (Cost + Expected Revenue)
    if "cost_per_unit" in roles and (
        "revenue_expected" in roles or "revenue_actual" in roles
    ):
        return "crm_leads"

    if "cost_total" in roles and "revenue_expected" in roles:
        return "crm_leads"

    if "revenue_actual" in roles and "cost_total" not in roles:
        return "crm_leads"

    if "revenue_actual" in roles and "cost_total" in roles:
        return "sales_revenue"

    return "generic"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile(data: List[Dict[str, Any]], file_name: str) -> Dict:
    """
    Profile a dataset and return a schema_profile dict.

    Called ONCE at upload time. Result stored in MongoDB `schema_profiles`
    collection and used for all subsequent query-time column resolution.

    Returns:
        {
          file_name, dataset_type, row_count,
          columns: {col_name: {dtype, semantic_role, is_summable, unit}},
          available_metrics: [list of detected semantic roles],
          dimension_map: {dim_type: col_name},
          dimension_values: {col_name: [unique values]},
        }
    """
    if not data:
        logger.warning(f"[PROFILER] Empty data for {file_name}")
        return {}

    columns_profile: Dict[str, Dict] = {}
    available_metrics: List[str] = []
    dimension_map: Dict[str, str] = {}
    dimension_values: Dict[str, List] = {}
    column_roles: Dict[str, str] = {}   # col_name → semantic_role

    for col_name in data[0].keys():
        if not col_name:
            continue

        col_str = str(col_name).strip()

        # Skip Excel artifact columns ("Unnamed: N") and whitespace-only names
        if not col_str or col_str.lower().startswith("unnamed:"):
            logger.debug(f"[PROFILER] Skipping artifact column: {col_name!r}")
            continue

        # Sample up to 20 rows for type inference
        values = [row.get(col_name) for row in data[:20]]

        # Skip fully-empty columns
        if all(v is None or (isinstance(v, float) and str(v) == "nan") for v in values):
            logger.debug(f"[PROFILER] Skipping empty column: {col_name!r}")
            continue
        clean_values = [v for v in values if v is not None]
        raw_dtype = _get_raw_dtype(clean_values)

        # ----------------------------------------------------------------
        # Priority classification order
        # ----------------------------------------------------------------

        if _is_identifier(col_name, clean_values):
            col_meta = {
                "dtype": "identifier",
                "semantic_role": None,
                "is_summable": False,
                "unit": None,
            }

        elif _is_contact(col_name):
            col_meta = {
                "dtype": "contact",
                "semantic_role": None,
                "is_summable": False,
                "unit": None,
            }

        elif _is_name_field(col_name):
            col_meta = {
                "dtype": "name",
                "semantic_role": None,
                "is_summable": False,
                "unit": None,
            }

        elif _is_date_col(col_name):
            col_meta = {
                "dtype": "date",
                "semantic_role": "time_dimension",
                "is_summable": False,
                "unit": None,
            }
            # Also register as a dimension
            dimension_map["date"] = col_name
            unique_vals = sorted(
                {str(row.get(col_name)) for row in data if row.get(col_name) is not None}
            )
            dimension_values[col_name] = unique_vals[:30]

        elif raw_dtype == "numeric":
            semantic_role, unit, is_summable = _detect_semantic_role(col_name, clean_values)

            # A column is a pre_computed_ratio if it carries a % unit or
            # its name contains %, pct, or the semantic role is a ratio type
            is_precomputed_ratio = (
                "%" in col_name
                or "pct" in col_name.lower()
                or unit == "percentage"
                or semantic_role in ("roi", "ctr", "nps")
            )

            col_meta = {
                "dtype": "pre_computed_ratio" if is_precomputed_ratio else "numeric_measure",
                "semantic_role": semantic_role,
                "is_summable": is_summable,
                "unit": unit,
            }

            if semantic_role:
                # Avoid duplicates — first occurrence of a role wins
                if semantic_role not in available_metrics:
                    available_metrics.append(semantic_role)
                column_roles[col_name] = semantic_role

        else:
            # String column: check dimension ontology BEFORE name-field test.
            # Rationale: "Campaign Name" is a campaign dimension, not a name field.
            dim_type = _detect_dimension_type(col_name)

            if dim_type:
                col_meta = {
                    "dtype": "categorical",
                    "semantic_role": dim_type,
                    "is_summable": False,
                    "unit": None,
                }
                if dim_type not in dimension_map:
                    dimension_map[dim_type] = col_name
                unique_vals = sorted(
                    {str(row.get(col_name)) for row in data if row.get(col_name) is not None}
                )
                dimension_values[col_name] = unique_vals[:30]

            elif _is_name_field(col_name):
                col_meta = {
                    "dtype": "name",
                    "semantic_role": None,
                    "is_summable": False,
                    "unit": None,
                }

            else:
                # Generic string with no dimension mapping
                col_meta = {
                    "dtype": "categorical",
                    "semantic_role": None,
                    "is_summable": False,
                    "unit": None,
                }

        columns_profile[col_name] = col_meta

    dataset_type = _detect_dataset_type(column_roles)

    schema = {
        "file_name": file_name,
        "dataset_type": dataset_type,
        "row_count": len(data),
        "columns": columns_profile,
        "available_metrics": available_metrics,
        "dimension_map": dimension_map,
        "dimension_values": dimension_values,
    }

    logger.info(
        f"[PROFILER] '{file_name}' → type={dataset_type}, "
        f"metrics={available_metrics}, dimensions={list(dimension_map.keys())}"
    )
    return schema
