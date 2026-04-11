"""
Unit tests — ColumnMapper

What we test:
  - METRIC_ALIASES normalises aliases to canonical roles
  - Direct column found by semantic_role
  - Derivable resolution returns correct required_cols formula
  - Missing role → missing=True
  - All-leads resolution returns lead_cols dict
  - cost_per_unit is NOT summable
  - pre_computed_ratio columns are flagged correctly
  - get_dimension_col resolves from dimension_map
  - find_value_dimension locates a filter value

Run with:  pytest -s tests/test_column_mapper.py
"""

import pytest
from services.column_mapper import (
    resolve,
    get_dimension_col,
    find_value_dimension,
    METRIC_ALIASES,
    DERIVABLE_METRICS,
)


# ---------------------------------------------------------------------------
# Helpers — build minimal schema_profiles without loading real data
# ---------------------------------------------------------------------------

def _minimal_schema(columns: dict, dimension_map: dict = None,
                    dimension_values: dict = None) -> dict:
    return {
        "columns": columns,
        "available_metrics": [
            m["semantic_role"] for m in columns.values() if m.get("semantic_role")
        ],
        "dimension_map": dimension_map or {},
        "dimension_values": dimension_values or {},
    }


# Marketing campaign schema (mirrors SF Campaign dataset structure)
CAMPAIGN_SCHEMA = _minimal_schema(
    columns={
        "Campaign ID":            {"dtype": "identifier",       "semantic_role": None,               "is_summable": False, "unit": None},
        "Campaign Name":          {"dtype": "categorical",      "semantic_role": "campaign",         "is_summable": False, "unit": None},
        "Marketing Channel":      {"dtype": "categorical",      "semantic_role": "channel",          "is_summable": False, "unit": None},
        "Month":                  {"dtype": "date",             "semantic_role": "time_dimension",   "is_summable": False, "unit": None},
        "Leads Generated":        {"dtype": "numeric_measure",  "semantic_role": "leads_total",      "is_summable": True,  "unit": "count"},
        "Qualified Leads":        {"dtype": "numeric_measure",  "semantic_role": "leads_qualified",  "is_summable": True,  "unit": "count"},
        "Converted Leads":        {"dtype": "numeric_measure",  "semantic_role": "leads_converted",  "is_summable": True,  "unit": "count"},
        "Campaign Cost (₹)":      {"dtype": "numeric_measure",  "semantic_role": "cost_total",       "is_summable": True,  "unit": "currency"},
        "Revenue Generated (₹)":  {"dtype": "numeric_measure",  "semantic_role": "revenue_actual",   "is_summable": True,  "unit": "currency"},
        "Profit (₹)":             {"dtype": "numeric_measure",  "semantic_role": "profit",           "is_summable": True,  "unit": "currency"},
        "ROI (%)":                {"dtype": "pre_computed_ratio","semantic_role": "roi",             "is_summable": False, "unit": "percentage"},
    },
    dimension_map={
        "campaign": "Campaign Name",
        "channel":  "Marketing Channel",
        "date":     "Month",
    },
    dimension_values={
        "Marketing Channel": ["Email", "Facebook Ads", "Google Ads", "Referral", "SEO"],
        "Campaign Name":     ["Spring Sale", "Holiday Push", "Q4 Launch"],
    },
)

# Google Ads schema (no pre-computed CTR column)
GOOGLE_ADS_SCHEMA = _minimal_schema(
    columns={
        "Campaign":       {"dtype": "categorical",      "semantic_role": "campaign",      "is_summable": False, "unit": None},
        "Impressions":    {"dtype": "numeric_measure",  "semantic_role": "impressions",   "is_summable": True,  "unit": "count"},
        "Clicks":         {"dtype": "numeric_measure",  "semantic_role": "clicks",        "is_summable": True,  "unit": "count"},
        "Avg CPC (₹)":    {"dtype": "numeric_measure",  "semantic_role": "cost_per_unit", "is_summable": False, "unit": "currency_rate"},
        "Cost (₹)":       {"dtype": "numeric_measure",  "semantic_role": "cost_total",    "is_summable": True,  "unit": "currency"},
        "Conversions":    {"dtype": "numeric_measure",  "semantic_role": "conversions",   "is_summable": True,  "unit": "count"},
    },
    dimension_map={"campaign": "Campaign"},
)

# SF Lead schema (cost_per_unit CPL + expected revenue)
SF_LEAD_SCHEMA = _minimal_schema(
    columns={
        "Lead ID":            {"dtype": "identifier",      "semantic_role": None,               "is_summable": False, "unit": None},
        "Campaign Name":      {"dtype": "categorical",     "semantic_role": "campaign",         "is_summable": False, "unit": None},
        "Lead Status":        {"dtype": "categorical",     "semantic_role": "status",           "is_summable": False, "unit": None},
        "Cost Per Lead (₹)":  {"dtype": "numeric_measure", "semantic_role": "cost_per_unit",    "is_summable": False, "unit": "currency_rate"},
        "Expected Revenue (₹)": {"dtype": "numeric_measure","semantic_role": "revenue_expected","is_summable": True,  "unit": "currency"},
    },
    dimension_map={"status": "Lead Status", "campaign": "Campaign Name"},
    dimension_values={"Lead Status": ["Converted", "Open - Not Contacted", "Working - Contacted"]},
)

# Schema with no lead columns at all
EMPTY_LEAD_SCHEMA = _minimal_schema(
    columns={
        "Revenue (₹)": {"dtype": "numeric_measure", "semantic_role": "revenue_actual", "is_summable": True, "unit": "currency"},
    }
)


# ---------------------------------------------------------------------------
# METRIC_ALIASES
# ---------------------------------------------------------------------------
class TestMetricAliases:
    def test_revenue_alias(self):
        assert METRIC_ALIASES["revenue"] == "revenue_actual"

    def test_total_revenue_alias(self):
        assert METRIC_ALIASES["total revenue"] == "revenue_actual"

    def test_expected_revenue_alias(self):
        assert METRIC_ALIASES["expected revenue"] == "revenue_expected"

    def test_cpl_alias(self):
        assert METRIC_ALIASES["cost per lead"] == "cpl"

    def test_cost_alias(self):
        assert METRIC_ALIASES["cost"] == "cost_total"

    def test_leads_alias(self):
        assert METRIC_ALIASES["leads"] == "leads_total"

    def test_roas_alias(self):
        assert METRIC_ALIASES["roas"] == "roi"

    def test_ctr_alias(self):
        assert METRIC_ALIASES["click through rate"] == "ctr"

    def test_win_rate_alias(self):
        assert METRIC_ALIASES["win rate"] == "win_rate"


# ---------------------------------------------------------------------------
# Direct column resolution
# ---------------------------------------------------------------------------
class TestDirectResolution:
    def test_revenue_actual_resolved(self):
        r = resolve("revenue_actual", CAMPAIGN_SCHEMA)
        assert r["missing"] is False
        assert r["primary_col"] == "Revenue Generated (₹)"
        assert r["is_summable"] is True
        assert r["unit"] == "currency"

    def test_revenue_alias_resolves_to_actual(self):
        """'revenue' must map to revenue_actual, not revenue_expected."""
        r = resolve("revenue", CAMPAIGN_SCHEMA)
        assert r["missing"] is False
        assert r["role"] == "revenue_actual"
        assert r["primary_col"] == "Revenue Generated (₹)"

    def test_leads_total_resolved(self):
        r = resolve("leads_total", CAMPAIGN_SCHEMA)
        assert r["missing"] is False
        assert r["primary_col"] == "Leads Generated"
        assert r["is_summable"] is True

    def test_cost_total_resolved(self):
        r = resolve("cost_total", CAMPAIGN_SCHEMA)
        assert r["missing"] is False
        assert r["primary_col"] == "Campaign Cost (₹)"
        assert r["is_summable"] is True

    def test_cost_alias_resolves_to_cost_total(self):
        r = resolve("cost", CAMPAIGN_SCHEMA)
        assert r["role"] == "cost_total"

    def test_roi_direct_column(self):
        """ROI(%) pre-computed column — must NOT go to derivable path."""
        r = resolve("roi", CAMPAIGN_SCHEMA)
        assert r["missing"] is False
        assert r["primary_col"] == "ROI (%)"
        assert r["derivable"] is None
        assert r["is_summable"] is False
        assert r["is_precomputed"] is True

    def test_cost_per_unit_not_summable(self):
        """Cost Per Lead is a per-unit rate — is_summable must be False."""
        r = resolve("cost_per_unit", SF_LEAD_SCHEMA)
        assert r["missing"] is False
        assert r["primary_col"] == "Cost Per Lead (₹)"
        assert r["is_summable"] is False

    def test_expected_revenue_not_actual(self):
        """expected revenue must NOT resolve to revenue_actual."""
        r = resolve("expected revenue", SF_LEAD_SCHEMA)
        assert r["role"] == "revenue_expected"
        assert r["primary_col"] == "Expected Revenue (₹)"

    def test_conversions_direct(self):
        r = resolve("conversions", GOOGLE_ADS_SCHEMA)
        assert r["missing"] is False
        assert r["primary_col"] == "Conversions"


# ---------------------------------------------------------------------------
# Derivable metric resolution
# ---------------------------------------------------------------------------
class TestDerivableResolution:
    def test_ctr_derivable_from_clicks_impressions(self):
        """CTR column does not exist → must be derived from Clicks / Impressions."""
        r = resolve("ctr", GOOGLE_ADS_SCHEMA)
        assert r["missing"] is False
        assert r["primary_col"] is None
        assert r["derivable"] is not None
        assert "clicks" in r["derivable"]["required_cols"]
        assert "impressions" in r["derivable"]["required_cols"]
        assert r["derivable"]["required_cols"]["clicks"] == "Clicks"
        assert r["derivable"]["required_cols"]["impressions"] == "Impressions"
        assert r["is_summable"] is False
        assert r["unit"] == "percentage"

    def test_profit_derivable_from_revenue_and_cost(self):
        """If no Profit column, derive from Revenue - Cost."""
        schema = _minimal_schema(columns={
            "Revenue (₹)": {"dtype": "numeric_measure", "semantic_role": "revenue_actual", "is_summable": True, "unit": "currency"},
            "Cost (₹)":    {"dtype": "numeric_measure", "semantic_role": "cost_total",    "is_summable": True, "unit": "currency"},
        })
        r = resolve("profit", schema)
        assert r["missing"] is False
        assert r["derivable"] is not None
        assert "revenue_actual" in r["derivable"]["required_cols"]
        assert "cost_total" in r["derivable"]["required_cols"]

    def test_cpl_derivable_from_cost_and_leads(self):
        """CPL = Total Cost / Total Leads — must require cost_total and leads_total."""
        r = resolve("cpl", CAMPAIGN_SCHEMA)
        assert r["missing"] is False
        assert r["derivable"] is not None
        req = r["derivable"]["required_cols"]
        assert "cost_total" in req
        assert "leads_total" in req
        assert r["unit"] == "currency_rate"

    def test_conversion_rate_derivable(self):
        r = resolve("conversion_rate", CAMPAIGN_SCHEMA)
        assert r["missing"] is False
        assert r["derivable"] is not None
        req = r["derivable"]["required_cols"]
        assert "leads_converted" in req
        assert "leads_total" in req

    def test_derivable_missing_when_required_col_absent(self):
        """CTR cannot be derived if Impressions column is missing."""
        schema = _minimal_schema(columns={
            "Clicks": {"dtype": "numeric_measure", "semantic_role": "clicks", "is_summable": True, "unit": "count"},
        })
        r = resolve("ctr", schema)
        assert r["missing"] is True
        assert "impressions" in r["warning"].lower()


# ---------------------------------------------------------------------------
# Missing metric
# ---------------------------------------------------------------------------
class TestMissingMetric:
    def test_unknown_metric_returns_missing(self):
        r = resolve("magic_unicorn_metric", CAMPAIGN_SCHEMA)
        assert r["missing"] is True
        assert r["warning"] is not None

    def test_roi_missing_when_no_columns(self):
        """If no revenue_actual OR cost_total exist, ROI cannot be derived."""
        r = resolve("roi", EMPTY_LEAD_SCHEMA)
        # EMPTY_LEAD_SCHEMA has no ROI column and no cost_total → missing
        assert r["missing"] is True


# ---------------------------------------------------------------------------
# All-leads resolution
# ---------------------------------------------------------------------------
class TestAllLeadsResolution:
    def test_all_leads_returns_multi(self):
        r = resolve("leads_total", CAMPAIGN_SCHEMA, return_all_lead_types=True)
        assert r["role"] == "leads_multi"
        assert r["missing"] is False
        assert "leads_total" in r["lead_cols"]
        assert "leads_qualified" in r["lead_cols"]
        assert "leads_converted" in r["lead_cols"]

    def test_all_leads_each_entry_has_col_key(self):
        r = resolve("leads_total", CAMPAIGN_SCHEMA, return_all_lead_types=True)
        for role, info in r["lead_cols"].items():
            assert "col" in info
            assert "is_summable" in info
            assert "unit" in info

    def test_all_leads_missing_when_no_lead_cols(self):
        r = resolve("leads_total", EMPTY_LEAD_SCHEMA, return_all_lead_types=True)
        assert r["missing"] is True
        assert "No lead columns" in r["warning"]


# ---------------------------------------------------------------------------
# get_dimension_col
# ---------------------------------------------------------------------------
class TestGetDimensionCol:
    def test_channel_resolves(self):
        col = get_dimension_col("channel", CAMPAIGN_SCHEMA)
        assert col == "Marketing Channel"

    def test_campaign_resolves(self):
        col = get_dimension_col("campaign", CAMPAIGN_SCHEMA)
        assert col == "Campaign Name"

    def test_unknown_dimension_returns_none(self):
        col = get_dimension_col("nonexistent_dim", CAMPAIGN_SCHEMA)
        assert col is None


# ---------------------------------------------------------------------------
# find_value_dimension
# ---------------------------------------------------------------------------
class TestFindValueDimension:
    def test_exact_channel_match(self):
        result = find_value_dimension("Email", CAMPAIGN_SCHEMA)
        assert result is not None
        col, val = result
        assert col == "Marketing Channel"
        assert val == "Email"

    def test_case_insensitive_match(self):
        result = find_value_dimension("email", CAMPAIGN_SCHEMA)
        assert result is not None

    def test_nonexistent_value_returns_none(self):
        result = find_value_dimension("TikTok Ads", CAMPAIGN_SCHEMA)
        assert result is None

    def test_status_value_match(self):
        result = find_value_dimension("Converted", SF_LEAD_SCHEMA)
        assert result is not None
        col, val = result
        assert col == "Lead Status"
        assert val == "Converted"
