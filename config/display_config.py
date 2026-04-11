"""
display_config.py — Controls what appears in the rectangular KPI boxes
rendered inside the main chat area (AIBubble inline grid).

HOW IT WORKS
────────────
Each entry in QUERY_DISPLAY_RULES maps to a "display rule":
  - trigger_keywords : if ANY of these appear in the user query (lowercase),
                       this rule is selected.
  - dataset_types    : optional allowlist of dataset_type values this rule
                       applies to. Empty list = applies to all dataset types.
  - fields           : ordered list of boxes to show. Fields are matched
                       against the schema_profile column roles.
  - max_boxes        : hard cap on rendered boxes (default 6).

FIELD ARGUMENTS
───────────────
  column_role   (str)   Semantic role from DatasetProfiler / schema_profile.
                        Must match a key in SEMANTIC_ONTOLOGY or DIMENSION_ONTOLOGY.
                        Available metric roles:
                          revenue_actual, revenue_expected, deal_amount,
                          cost_total, cost_per_unit, budget,
                          leads_total, leads_qualified, leads_converted,
                          opportunities, impressions, clicks, conversions,
                          profit, roi, ctr, nps
                        Available dimension roles:
                          campaign, channel, source, status, owner,
                          client, company, ad_group, date, city

  label         (str)   Display label shown above the value in the box.

  required      (bool)  True  → box is always included even if value is missing.
                        False → box is omitted when the column doesn't exist
                                in the active dataset.

  format        (str)   One of:
                          "currency"    → ₹1,23,456
                          "number"      → 1,234
                          "percentage"  → 42.3%
                          "text"        → raw string value
                          "auto"        → inferred from column unit (default)

  prefix        (str)   Optional string prepended to value. e.g. "₹"
  suffix        (str)   Optional string appended to value. e.g. "%"

PRIORITY ORDER
──────────────
Rules are evaluated top-to-bottom. First matching rule wins.
Put more specific rules before more general ones.
"default_metric" fires when no other rule matches.
"""

from typing import List, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# Helper type alias
# ─────────────────────────────────────────────────────────────────────────────
Field = Dict[str, Any]


def _field(
    column_role: str,
    label: str,
    required: bool = False,
    format: str = "auto",
    prefix: str = "",
    suffix: str = "",
) -> Field:
    return {
        "column_role": column_role,
        "label":       label,
        "required":    required,
        "format":      format,
        "prefix":      prefix,
        "suffix":      suffix,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUERY DISPLAY RULES
# ─────────────────────────────────────────────────────────────────────────────
QUERY_DISPLAY_RULES: List[Dict[str, Any]] = [

    # ── Lead detail / best lead ───────────────────────────────────────────────
    # Triggered by: "best lead", "top lead", "who is the lead", etc.
    # Shows the individual lead's key attributes as boxes.
    {
        "rule_id":        "lead_detail",
        "trigger_keywords": [
            "best lead", "top lead", "highest lead", "who is the lead",
            "lead detail", "lead info", "lead profile", "which lead",
            "lead name", "lead revenue",
        ],
        "dataset_types":  ["crm_leads", "lead_generation", "sales_pipeline", "generic"],
        "max_boxes":      6,
        "fields": [
            _field("client",           "Lead Name",          required=True,  format="text"),
            _field("source",           "Lead Source",        required=False, format="text"),
            _field("company",          "Company",            required=False, format="text"),
            _field("revenue_expected", "Expected Revenue",   required=False, format="currency"),
            _field("revenue_actual",   "Revenue",            required=False, format="currency"),
            _field("cost_per_unit",    "Cost Per Lead",      required=False, format="currency"),
            _field("status",           "Lead Status",        required=False, format="text"),
            _field("campaign",         "Campaign",           required=False, format="text"),
            _field("owner",            "Owner",              required=False, format="text"),
        ],
    },

    # ── Campaign performance ──────────────────────────────────────────────────
    {
        "rule_id":        "campaign_performance",
        "trigger_keywords": [
            "best campaign", "top campaign", "campaign performance",
            "which campaign", "campaign revenue", "campaign roi",
            "campaign result", "campaign detail",
        ],
        "dataset_types":  ["marketing_campaign", "google_ads", "lead_generation"],
        "max_boxes":      6,
        "fields": [
            _field("campaign",       "Campaign",         required=True,  format="text"),
            _field("revenue_actual", "Revenue",          required=False, format="currency"),
            _field("roi",            "ROI",              required=False, format="percentage"),
            _field("leads_total",    "Leads Generated",  required=False, format="number"),
            _field("cost_total",     "Spend",            required=False, format="currency"),
            _field("profit",         "Profit",           required=False, format="currency"),
            _field("channel",        "Channel",          required=False, format="text"),
        ],
    },

    # ── Google Ads / CTR / clicks ─────────────────────────────────────────────
    {
        "rule_id":        "ads_performance",
        "trigger_keywords": [
            "ctr", "click through", "impressions", "clicks", "ad performance",
            "best ad", "top ad", "google ads", "ad group", "ad spend",
            "cost per click", "cpc",
        ],
        "dataset_types":  ["google_ads"],
        "max_boxes":      6,
        "fields": [
            _field("campaign",       "Campaign",      required=True,  format="text"),
            _field("clicks",         "Clicks",        required=False, format="number"),
            _field("impressions",    "Impressions",   required=False, format="number"),
            _field("ctr",            "CTR",           required=False, format="percentage"),
            _field("cost_per_unit",  "CPC",           required=False, format="currency"),
            _field("conversions",    "Conversions",   required=False, format="number"),
            _field("revenue_actual", "Revenue",       required=False, format="currency"),
        ],
    },

    # ── Revenue summary ───────────────────────────────────────────────────────
    {
        "rule_id":        "revenue_summary",
        "trigger_keywords": [
            "total revenue", "revenue summary", "revenue breakdown",
            "revenue by", "highest revenue", "revenue generated",
            "revenue wise", "how much revenue",
        ],
        "dataset_types":  [],          # all dataset types
        "max_boxes":      4,
        "fields": [
            _field("revenue_actual",   "Total Revenue",    required=False, format="currency"),
            _field("revenue_expected", "Expected Revenue", required=False, format="currency"),
            _field("profit",           "Profit",           required=False, format="currency"),
            _field("roi",              "ROI",              required=False, format="percentage"),
            _field("cost_total",       "Total Spend",      required=False, format="currency"),
        ],
    },

    # ── ROI / profitability ───────────────────────────────────────────────────
    {
        "rule_id":        "roi_summary",
        "trigger_keywords": [
            "roi", "return on investment", "roas", "profitability",
            "profit margin", "most profitable", "best roi",
        ],
        "dataset_types":  [],
        "max_boxes":      4,
        "fields": [
            _field("roi",            "ROI",          required=True,  format="percentage"),
            _field("profit",         "Profit",       required=False, format="currency"),
            _field("revenue_actual", "Revenue",      required=False, format="currency"),
            _field("cost_total",     "Spend",        required=False, format="currency"),
            _field("campaign",       "Campaign",     required=False, format="text"),
        ],
    },

    # ── Lead counts / volume ──────────────────────────────────────────────────
    {
        "rule_id":        "lead_volume",
        "trigger_keywords": [
            "total leads", "leads generated", "lead count", "how many leads",
            "qualified leads", "converted leads", "lead conversion",
            "conversion rate",
        ],
        "dataset_types":  ["marketing_campaign", "lead_generation", "crm_leads"],
        "max_boxes":      4,
        "fields": [
            _field("leads_total",      "Total Leads",      required=False, format="number"),
            _field("leads_qualified",  "Qualified Leads",  required=False, format="number"),
            _field("leads_converted",  "Converted Leads",  required=False, format="number"),
            _field("cost_per_unit",    "Cost Per Lead",    required=False, format="currency"),
            _field("cost_total",       "Total Spend",      required=False, format="currency"),
        ],
    },

    # ── Default fallback — used when no specific rule matches ─────────────────
    # Shows whichever metric the engine computed + key contextual fields.
    {
        "rule_id":          "default_metric",
        "trigger_keywords": [],         # matches everything — must be LAST
        "dataset_types":    [],
        "max_boxes":        4,
        "fields": [
            # The engine always populates result/metric — these are generic placeholders.
            # The analysis LLM fills in the actual kpis[] array from computed results.
            _field("revenue_actual",   "Revenue",         required=False, format="currency"),
            _field("cost_total",       "Spend",           required=False, format="currency"),
            _field("roi",              "ROI",             required=False, format="percentage"),
            _field("leads_total",      "Leads",           required=False, format="number"),
            _field("cost_per_unit",    "CPL",             required=False, format="currency"),
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Public helper — used by the analysis prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def get_display_rule(query: str, dataset_type: str = "") -> Dict[str, Any]:
    """
    Return the first matching display rule for a query.
    Falls back to "default_metric" if no trigger keywords match.

    Args:
        query        : user query string (lowercased before matching)
        dataset_type : value from schema_profile["dataset_type"]

    Returns:
        The matching rule dict from QUERY_DISPLAY_RULES.
    """
    q = query.lower()

    for rule in QUERY_DISPLAY_RULES:
        keywords = rule.get("trigger_keywords", [])
        if not keywords:
            continue   # skip default here; it's the fallback below

        # Dataset type filter (empty list = any type)
        allowed_types = rule.get("dataset_types", [])
        if allowed_types and dataset_type and dataset_type not in allowed_types:
            continue

        if any(kw in q for kw in keywords):
            return rule

    # Return the default fallback (last rule)
    return QUERY_DISPLAY_RULES[-1]


def get_fields_for_prompt(query: str, dataset_type: str = "") -> str:
    """
    Returns a formatted string injected into the analysis prompt so the
    LLM knows exactly which fields to populate in the kpis[] array.

    Example output:
        Populate kpis[] with these fields (in order, skip if unavailable):
        1. Lead Name     (column_role: client,   format: text)
        2. Lead Source   (column_role: source,   format: text)
        3. Revenue       (column_role: revenue_actual, format: currency)
    """
    rule   = get_display_rule(query, dataset_type)
    fields = rule["fields"][: rule.get("max_boxes", 6)]

    lines = ["Populate kpis[] with these fields (in order, omit if data unavailable):"]
    for i, f in enumerate(fields, 1):
        req = " [required]" if f["required"] else ""
        lines.append(
            f"  {i}. {f['label']:<22} "
            f"(column_role: {f['column_role']}, format: {f['format']}){req}"
        )
    lines.append(f"Maximum {rule.get('max_boxes', 6)} boxes. Do NOT add boxes not listed above.")
    return "\n".join(lines)
