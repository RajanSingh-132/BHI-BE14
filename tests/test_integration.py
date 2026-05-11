# """
# Integration tests — full deterministic pipeline

# Tests the three-layer deterministic pipeline end-to-end WITHOUT any LLM calls:

#   DatasetProfiler.profile()
#         ↓  (schema_profile stored at upload time)
#   ColumnMapper.resolve()          ← simulated intent from LLM
#         ↓
#   CalculationEngine.calculate()
#         ↓
#   CalculationResult dict          ← verified against ground truth

# What we test here that unit tests don't:
#   - Schema profile built from real data routes to the correct column
#   - A simulated LLM-style intent string resolves and calculates correctly
#   - The same metric keyword on DIFFERENT datasets selects the right column
#     (e.g. "revenue" picks "Revenue Generated (₹)" on SF Campaign but
#      "Revenue Earned (₹)" on Zoho Sales)
#   - Alias normalization flows through the full stack
#   - cross-dataset column disambiguation: "cost" on Google Ads vs SF Campaign
#   - "cost per lead" (cost_per_unit) is never summed on any dataset
#   - result dict has correct shape for the Analysis LLM contract

# Run with:  pytest -s tests/test_integration.py
# """

# import pytest
# from tests.fixtures import (
#     SF_CAMPAIGN, SF_SALES, GOOGLE_ADS, MARKETING_REV,
#     SUGAR_CRM, ZOHO_CRM, SF_LEAD, ZOHO_SALES,
#     EXPECTED,
# )
# from services.dataset_profiler import profile
# from services.calculation_engine import calculate
# from services.column_mapper import resolve


# # ---------------------------------------------------------------------------
# # Helper — build an intent dict exactly as the Intent Extractor LLM would
# # ---------------------------------------------------------------------------

# def lm_intent(
#     metric: str,
#     aggregation: str = "sum",
#     filters=None,
#     group_by: str = None,
#     all_leads: bool = False,
# ) -> dict:
#     """Simulate the JSON output of the Intent Extractor LLM."""
#     return {
#         "metric":              metric,
#         "filters":             filters or [],
#         "aggregation":         aggregation,
#         "group_by":            group_by,
#         "time_period":         None,
#         "return_all_lead_types": all_leads,
#     }


# ANALYSIS_LLM_REQUIRED_KEYS = {
#     "metric", "result", "breakdown", "lead_breakdown",
#     "group_by_col", "metric_col", "formula", "source",
#     "unit", "filter_applied", "row_count", "warnings", "error",
# }


# def assert_result_shape(result: dict):
#     """Verify CalculationResult contract is complete for Analysis LLM."""
#     missing = ANALYSIS_LLM_REQUIRED_KEYS - set(result.keys())
#     assert not missing, f"CalculationResult missing keys: {missing}"


# # ---------------------------------------------------------------------------
# # Fixture: pre-built schema profiles for all datasets
# # ---------------------------------------------------------------------------

# @pytest.fixture(scope="module")
# def schemas():
#     return {
#         "sf_campaign":   profile(SF_CAMPAIGN,   "sf_campaign"),
#         "sf_sales":      profile(SF_SALES,       "sf_sales"),
#         "google_ads":    profile(GOOGLE_ADS,     "google_ads"),
#         "marketing_rev": profile(MARKETING_REV,  "marketing_rev"),
#         "sugar_crm":     profile(SUGAR_CRM,      "sugar_crm"),
#         "zoho_crm":      profile(ZOHO_CRM,       "zoho_crm"),
#         "sf_lead":       profile(SF_LEAD,        "sf_lead"),
#         "zoho_sales":    profile(ZOHO_SALES,     "zoho_sales"),
#     }


# # ---------------------------------------------------------------------------
# # 1. Result shape contract (Analysis LLM never receives broken dicts)
# # ---------------------------------------------------------------------------
# class TestResultShapeContract:
#     """
#     Every possible calculation path must return a complete CalculationResult.
#     The Analysis LLM will crash or hallucinate if a key is missing.
#     """

#     def test_direct_scalar_has_all_keys(self, schemas):
#         r = calculate(SF_CAMPAIGN, lm_intent("revenue_actual"), schemas["sf_campaign"])
#         assert_result_shape(r)

#     def test_derivable_scalar_has_all_keys(self, schemas):
#         r = calculate(GOOGLE_ADS, lm_intent("ctr"), schemas["google_ads"])
#         assert_result_shape(r)

#     def test_group_by_has_all_keys(self, schemas):
#         r = calculate(SF_CAMPAIGN, lm_intent("leads_total", group_by="channel"), schemas["sf_campaign"])
#         assert_result_shape(r)

#     def test_multi_lead_has_all_keys(self, schemas):
#         r = calculate(SF_CAMPAIGN, lm_intent("leads_total", all_leads=True), schemas["sf_campaign"])
#         assert_result_shape(r)

#     def test_error_path_has_all_keys(self, schemas):
#         r = calculate(SF_CAMPAIGN, lm_intent("nonexistent_metric_xyz"), schemas["sf_campaign"])
#         assert_result_shape(r)
#         assert r["error"] is not None

#     def test_filtered_result_has_all_keys(self, schemas):
#         r = calculate(
#             SF_LEAD,
#             lm_intent("cost_per_unit", filters=[{"field": "status", "value": "Converted"}]),
#             schemas["sf_lead"],
#         )
#         assert_result_shape(r)


# # ---------------------------------------------------------------------------
# # 2. Cross-dataset: same LLM intent string → correct column per dataset
# # ---------------------------------------------------------------------------
# class TestCrossDatasetColumnDisambiguation:
#     """
#     The LLM emits "revenue_actual" for any revenue question.
#     Each dataset maps this to a DIFFERENT physical column.
#     The pipeline must resolve to the right column every time.
#     """

#     def test_revenue_sf_campaign(self, schemas):
#         r = calculate(SF_CAMPAIGN, lm_intent("revenue_actual"), schemas["sf_campaign"])
#         assert r["error"] is None
#         assert r["result"] == pytest.approx(EXPECTED["sf_campaign"]["total_revenue"], abs=1)
#         assert "Revenue Generated" in r["metric_col"]

#     def test_revenue_sf_sales(self, schemas):
#         r = calculate(SF_SALES, lm_intent("revenue_actual"), schemas["sf_sales"])
#         assert r["error"] is None
#         assert r["result"] == pytest.approx(EXPECTED["sf_sales"]["total_revenue"], abs=1)

#     def test_revenue_sugar_crm(self, schemas):
#         r = calculate(SUGAR_CRM, lm_intent("revenue_actual"), schemas["sugar_crm"])
#         assert r["error"] is None
#         assert r["result"] == pytest.approx(EXPECTED["sugar_crm"]["total_revenue"], abs=1)

#     def test_revenue_zoho_sales_distinct_from_deal_amount(self, schemas):
#         """Zoho Sales has both revenue_actual and deal_amount — must resolve separately."""
#         r_rev  = calculate(ZOHO_SALES, lm_intent("revenue_actual"), schemas["zoho_sales"])
#         r_deal = calculate(ZOHO_SALES, lm_intent("deal_amount"),     schemas["zoho_sales"])
#         assert r_rev["error"] is None
#         assert r_deal["error"] is None
#         assert r_rev["result"]  == pytest.approx(EXPECTED["zoho_sales"]["total_revenue"],    abs=1)
#         assert r_deal["result"] == pytest.approx(EXPECTED["zoho_sales"]["total_deal_amount"], abs=1)
#         assert r_rev["result"] != r_deal["result"]  # must be different columns

#     def test_cost_sf_campaign_vs_google_ads(self, schemas):
#         """
#         'cost_total' on SF Campaign = campaign spend in ₹.
#         'cost_total' on Google Ads  = ad cost in ₹.
#         Both should resolve without error and return the correct total.
#         """
#         r_camp = calculate(SF_CAMPAIGN, lm_intent("cost_total"), schemas["sf_campaign"])
#         r_gads = calculate(GOOGLE_ADS,  lm_intent("cost_total"), schemas["google_ads"])
#         assert r_camp["error"] is None
#         assert r_gads["error"] is None
#         assert r_camp["result"] == pytest.approx(EXPECTED["sf_campaign"]["total_cost"], abs=1)
#         assert r_gads["result"] == pytest.approx(EXPECTED["google_ads"]["total_cost"],  abs=1)

#     def test_expected_revenue_vs_actual_revenue(self, schemas):
#         """
#         Zoho CRM: Expected Revenue (pipeline value) ≠ actual earned revenue.
#         LLM alias 'expected revenue' must NOT map to revenue_actual.
#         """
#         r_exp = calculate(ZOHO_CRM, lm_intent("revenue_expected"), schemas["zoho_crm"])
#         assert r_exp["error"] is None
#         assert r_exp["metric"] == "revenue_expected"
#         assert r_exp["result"] == pytest.approx(EXPECTED["zoho_crm"]["expected_revenue"], abs=1)


# # ---------------------------------------------------------------------------
# # 3. Alias normalisation through the full stack
# # ---------------------------------------------------------------------------
# class TestAliasNormalisationEndToEnd:
#     """
#     These test that the LLM's natural-language output aliases reach the
#     correct columns without any intervention.
#     Intent Extractor is expected to produce any of these strings.
#     """

#     @pytest.mark.parametrize("alias", [
#         "revenue", "total revenue", "sales revenue", "earnings",
#     ])
#     def test_revenue_aliases_all_resolve(self, alias, schemas):
#         r = calculate(SF_CAMPAIGN, lm_intent(alias), schemas["sf_campaign"])
#         assert r["error"] is None
#         assert r["result"] == pytest.approx(EXPECTED["sf_campaign"]["total_revenue"], abs=1)

#     @pytest.mark.parametrize("alias", [
#         "cost", "total cost", "spend", "marketing spend", "campaign cost",
#     ])
#     def test_cost_aliases_all_resolve(self, alias, schemas):
#         r = calculate(SF_CAMPAIGN, lm_intent(alias), schemas["sf_campaign"])
#         assert r["error"] is None
#         assert r["result"] == pytest.approx(EXPECTED["sf_campaign"]["total_cost"], abs=1)

#     @pytest.mark.parametrize("alias", [
#         "leads", "total leads", "lead count",
#     ])
#     def test_leads_aliases_resolve(self, alias, schemas):
#         r = calculate(SF_CAMPAIGN, lm_intent(alias), schemas["sf_campaign"])
#         assert r["error"] is None
#         assert r["result"] == pytest.approx(EXPECTED["sf_campaign"]["total_leads"], abs=0)

#     @pytest.mark.parametrize("alias", ["roas", "return on investment"])
#     def test_roi_aliases_resolve(self, alias, schemas):
#         r = calculate(SF_CAMPAIGN, lm_intent(alias), schemas["sf_campaign"])
#         assert r["error"] is None  # ROI(%) column exists — direct resolution

#     @pytest.mark.parametrize("alias", ["cpl", "cost per lead"])
#     def test_cpl_aliases_not_summed(self, alias, schemas):
#         """cost per lead is a rate — never sum regardless of alias used."""
#         r = calculate(SF_LEAD, lm_intent(alias, aggregation="sum"), schemas["sf_lead"])
#         assert r["error"] is None
#         # If resolved as cost_per_unit (direct column), is_summable=False → AVG returned
#         # If resolved as derivable cpl, uses SUM(cost)/SUM(leads)
#         # Either way result must be within range of plausible CPL, not a summed total
#         assert r["result"] < 10_000, (
#             f"CPL result {r['result']} looks like a summed total, not a per-lead rate"
#         )

#     def test_click_through_rate_alias(self, schemas):
#         r = calculate(GOOGLE_ADS, lm_intent("click through rate"), schemas["google_ads"])
#         assert r["error"] is None
#         assert r["result"] == pytest.approx(EXPECTED["google_ads"]["aggregate_ctr"], abs=0.01)

#     def test_expected_revenue_alias(self, schemas):
#         r = calculate(ZOHO_CRM, lm_intent("expected revenue"), schemas["zoho_crm"])
#         assert r["error"] is None
#         # result["metric"] echoes the raw intent string; check the resolved column
#         assert r["result"] == pytest.approx(EXPECTED["zoho_crm"]["expected_revenue"], abs=1)
#         assert "expected" in r["metric_col"].lower()


# # ---------------------------------------------------------------------------
# # 4. Derived metric correctness across datasets
# # ---------------------------------------------------------------------------
# class TestDerivedMetricIntegration:
#     """
#     Derived metrics need the right component columns from each dataset's schema.
#     If a dataset lacks a required column, the engine must return a clear error.
#     """

#     def test_ctr_derived_google_ads(self, schemas):
#         r = calculate(GOOGLE_ADS, lm_intent("ctr"), schemas["google_ads"])
#         assert r["error"] is None
#         assert r["source"] == "calculated"
#         assert r["result"] == pytest.approx(EXPECTED["google_ads"]["aggregate_ctr"], abs=0.01)

#     def test_ctr_not_available_on_sf_campaign(self, schemas):
#         """SF Campaign has no Impressions/Clicks — CTR should return an error."""
#         r = calculate(SF_CAMPAIGN, lm_intent("ctr"), schemas["sf_campaign"])
#         assert r["error"] is not None, (
#             "CTR on a dataset with no Impressions/Clicks must return an error"
#         )

#     def test_roi_derived_when_no_roi_column(self, schemas):
#         """
#         Google Ads has Revenue + Cost but no ROI(%) column.
#         ROI must be derived from (Revenue - Cost) / Cost × 100.
#         """
#         r = calculate(GOOGLE_ADS, lm_intent("roi"), schemas["google_ads"])
#         assert r["error"] is None
#         assert r["source"] == "calculated"
#         assert r["result"] == pytest.approx(EXPECTED["google_ads"]["aggregate_roi"], abs=0.5)

#     def test_conversion_rate_sf_campaign(self, schemas):
#         r = calculate(SF_CAMPAIGN, lm_intent("conversion_rate"), schemas["sf_campaign"])
#         assert r["error"] is None
#         assert r["result"] == pytest.approx(EXPECTED["sf_campaign"]["conversion_rate"], abs=0.5)

#     def test_profit_sf_campaign(self, schemas):
#         """Profit = Revenue - Cost. Both SUM to get net profit."""
#         r = calculate(SF_CAMPAIGN, lm_intent("profit"), schemas["sf_campaign"])
#         assert r["error"] is None
#         assert r["result"] == pytest.approx(EXPECTED["sf_campaign"]["total_profit"], abs=1)

#     def test_profit_sf_sales(self, schemas):
#         r = calculate(SF_SALES, lm_intent("profit"), schemas["sf_sales"])
#         assert r["error"] is None
#         assert r["result"] == pytest.approx(EXPECTED["sf_sales"]["total_profit"], abs=1)


# # ---------------------------------------------------------------------------
# # 5. Filter → group-by combinations
# # ---------------------------------------------------------------------------
# class TestFilterAndGroupBy:
#     def test_group_by_channel_revenue(self, schemas):
#         r = calculate(
#             SF_CAMPAIGN,
#             lm_intent("revenue_actual", aggregation="group_by", group_by="channel"),
#             schemas["sf_campaign"],
#         )
#         assert r["error"] is None
#         assert r["breakdown"], "Expected non-empty breakdown for group_by"
#         by_channel = {item["group"]: item["value"] for item in r["breakdown"]}
#         assert "Email" in by_channel
#         assert by_channel["Email"] == pytest.approx(
#             EXPECTED["sf_campaign"]["revenue_by_channel"]["Email"], abs=1
#         )

#     def test_group_by_leads_by_channel_order(self, schemas):
#         r = calculate(
#             SF_CAMPAIGN,
#             lm_intent("leads_total", aggregation="group_by", group_by="channel"),
#             schemas["sf_campaign"],
#         )
#         assert r["error"] is None
#         assert r["breakdown"]
#         top_channel, top_value = EXPECTED["sf_campaign"]["leads_by_channel_top1"]
#         assert r["breakdown"][0]["group"] == top_channel
#         assert r["breakdown"][0]["value"] == pytest.approx(top_value, abs=0)

#     def test_filter_by_status_reduces_rows(self, schemas):
#         r_all  = calculate(SF_LEAD, lm_intent("cost_per_unit"), schemas["sf_lead"])
#         r_conv = calculate(
#             SF_LEAD,
#             lm_intent("cost_per_unit", filters=[{"field": "status", "value": "Converted"}]),
#             schemas["sf_lead"],
#         )
#         assert r_all["row_count"]  == EXPECTED["sf_lead"]["n_rows"]
#         assert r_conv["row_count"] <  r_all["row_count"]
#         assert r_conv["filter_applied"] != "none"

#     def test_filter_and_group_by_combination(self, schemas):
#         """Filter to Email channel, then group revenue by campaign — no crash."""
#         r = calculate(
#             SF_CAMPAIGN,
#             lm_intent(
#                 "revenue_actual",
#                 aggregation="group_by",
#                 group_by="campaign",
#                 filters=[{"field": "channel", "value": "Email"}],
#             ),
#             schemas["sf_campaign"],
#         )
#         assert r["error"] is None
#         assert r["breakdown"]
#         # Sum of filtered breakdown ≤ total Email revenue
#         total_in_breakdown = sum(item["value"] for item in r["breakdown"])
#         assert total_in_breakdown == pytest.approx(
#             EXPECTED["sf_campaign"]["revenue_by_channel"]["Email"], abs=1
#         )


# # ---------------------------------------------------------------------------
# # 6. Schema profile correctness for key disambiguation rules
# # ---------------------------------------------------------------------------
# class TestSchemaProfileRules:
#     """
#     These validate the dataset_profiler outputs that all downstream logic
#     depends on.  If the profile is wrong, every calculation is wrong.
#     """

#     def test_lead_id_not_classified_as_leads_total(self, schemas):
#         """Lead ID is an identifier — must never be classified as leads_total."""
#         sf_lead_schema = schemas["sf_lead"]
#         for col, meta in sf_lead_schema["columns"].items():
#             if "id" in col.lower():
#                 assert meta.get("semantic_role") != "leads_total", (
#                     f"Column '{col}' looks like an ID but got role 'leads_total'"
#                 )

#     def test_campaign_name_is_dimension_not_name_dtype(self, schemas):
#         """'Campaign Name' must be classified as a campaign dimension, not name dtype."""
#         for dataset_key, schema in schemas.items():
#             for col, meta in schema["columns"].items():
#                 if "campaign name" in col.lower():
#                     assert meta["dtype"] == "categorical", (
#                         f"[{dataset_key}] '{col}' should be categorical dimension, got {meta['dtype']}"
#                     )
#                     assert meta.get("semantic_role") == "campaign", (
#                         f"[{dataset_key}] '{col}' should have role 'campaign', got {meta.get('semantic_role')}"
#                     )

#     def test_month_not_classified_as_identifier(self, schemas):
#         """Month (e.g. 'Mar-2026') must be classified as date, not identifier."""
#         for dataset_key, schema in schemas.items():
#             for col, meta in schema["columns"].items():
#                 if "month" in col.lower():
#                     assert meta["dtype"] == "date", (
#                         f"[{dataset_key}] '{col}' should be dtype=date, got {meta['dtype']}"
#                     )
#                     assert meta["dtype"] != "identifier", (
#                         f"[{dataset_key}] '{col}' must not be classified as identifier"
#                     )

#     def test_roi_col_marked_not_summable(self, schemas):
#         """ROI(%) columns must never be summable — summing percentages is invalid."""
#         for dataset_key, schema in schemas.items():
#             for col, meta in schema["columns"].items():
#                 if meta.get("semantic_role") == "roi":
#                     assert meta["is_summable"] is False, (
#                         f"[{dataset_key}] ROI column '{col}' must have is_summable=False"
#                     )

#     def test_ctr_col_marked_not_summable(self, schemas):
#         for dataset_key, schema in schemas.items():
#             for col, meta in schema["columns"].items():
#                 if meta.get("semantic_role") == "ctr":
#                     assert meta["is_summable"] is False, (
#                         f"[{dataset_key}] CTR column '{col}' must have is_summable=False"
#                     )

#     def test_cost_per_unit_not_summable(self, schemas):
#         """cost_per_unit (CPC, CPL) must never be summed."""
#         for dataset_key, schema in schemas.items():
#             for col, meta in schema["columns"].items():
#                 if meta.get("semantic_role") == "cost_per_unit":
#                     assert meta["is_summable"] is False, (
#                         f"[{dataset_key}] cost_per_unit column '{col}' must have is_summable=False"
#                     )

#     def test_expected_revenue_role_distinct_from_actual(self, schemas):
#         """Expected Revenue and Revenue Actual must resolve to different columns."""
#         zoho_schema = schemas["zoho_crm"]
#         actual_cols   = [c for c, m in zoho_schema["columns"].items()
#                          if m.get("semantic_role") == "revenue_actual"]
#         expected_cols = [c for c, m in zoho_schema["columns"].items()
#                          if m.get("semantic_role") == "revenue_expected"]
#         # Zoho CRM has Expected Revenue — it must be revenue_expected, not actual
#         assert expected_cols, "Zoho CRM must have at least one revenue_expected column"
#         for col in expected_cols:
#             assert col not in actual_cols, (
#                 f"Column '{col}' is classified as BOTH revenue_actual and revenue_expected"
#             )

#     def test_all_datasets_have_at_least_one_metric(self, schemas):
#         """Every dataset must expose at least one queryable metric after profiling."""
#         for dataset_key, schema in schemas.items():
#             assert schema.get("available_metrics"), (
#                 f"Dataset '{dataset_key}' has no available_metrics — profiler failed"
#             )

#     def test_google_ads_has_impressions_and_clicks(self, schemas):
#         """Google Ads must have both impressions and clicks for CTR derivation."""
#         roles = [m.get("semantic_role") for m in schemas["google_ads"]["columns"].values()]
#         assert "impressions" in roles
#         assert "clicks" in roles

#     def test_sf_campaign_dimension_map_has_channel(self, schemas):
#         assert "channel" in schemas["sf_campaign"]["dimension_map"]

#     def test_dimension_values_populated(self, schemas):
#         """dimension_values must be non-empty for datasets with categorical columns."""
#         s = schemas["sf_campaign"]
#         assert s["dimension_values"], "SF Campaign must have populated dimension_values"


# # ---------------------------------------------------------------------------
# # 7. Row-count preservation — filters must not silently drop all rows
# # ---------------------------------------------------------------------------
# class TestRowCountPreservation:
#     def test_unfiltered_uses_all_rows(self, schemas):
#         for key, data, exp in [
#             ("sf_campaign",  SF_CAMPAIGN,  EXPECTED["sf_campaign"]),
#             ("sf_sales",     SF_SALES,     EXPECTED["sf_sales"]),
#             ("google_ads",   GOOGLE_ADS,   EXPECTED["google_ads"]),
#             ("zoho_crm",     ZOHO_CRM,     EXPECTED["zoho_crm"]),
#             ("sf_lead",      SF_LEAD,      EXPECTED["sf_lead"]),
#         ]:
#             r = calculate(data, lm_intent("revenue_actual"), schemas[key])
#             if r["error"] is None:   # only check datasets where revenue resolves
#                 assert r["row_count"] == exp["n_rows"], (
#                     f"[{key}] row_count {r['row_count']} != expected {exp['n_rows']}"
#                 )


# # ---------------------------------------------------------------------------
# # 8. Query / Response pairs — reference table for manual QA
# # ---------------------------------------------------------------------------
# #
# # This section does NOT run as a test class.
# # It documents the expected (query, response) pairs for each dataset so
# # the team can validate the full LLM pipeline manually.
# #
# # Format:  {"query": str, "dataset": str, "expected_result": ..., "notes": str}
# #
# QUERY_RESPONSE_REFERENCE = [
#     # ── Salesforce Marketing Campaign ─────────────────────────────────────
#     {
#         "query":           "What is total revenue?",
#         "dataset":         "sf_campaign",
#         "metric_intent":   "revenue_actual",
#         "aggregation":     "sum",
#         "expected_result": 52_650_707,
#         "unit":            "₹",
#         "notes":           "Maps to 'Revenue Generated (₹)' column",
#     },
#     {
#         "query":           "What is total cost?",
#         "dataset":         "sf_campaign",
#         "metric_intent":   "cost_total",
#         "aggregation":     "sum",
#         "expected_result": 8_958_211,
#         "unit":            "₹",
#         "notes":           "Maps to 'Campaign Cost (₹)' — NOT 'Cost Per Lead'",
#     },
#     {
#         "query":           "How many leads were generated?",
#         "dataset":         "sf_campaign",
#         "metric_intent":   "leads_total",
#         "aggregation":     "sum",
#         "expected_result": 24_112,
#         "unit":            "count",
#         "notes":           "Maps to 'Leads Generated' — NOT 'Lead ID'",
#     },
#     {
#         "query":           "What is the average ROI?",
#         "dataset":         "sf_campaign",
#         "metric_intent":   "roi",
#         "aggregation":     "avg",
#         "expected_result": 605.13,
#         "unit":            "%",
#         "notes":           "AVG of pre-computed ROI(%) column per row",
#     },
#     {
#         "query":           "What is the conversion rate?",
#         "dataset":         "sf_campaign",
#         "metric_intent":   "conversion_rate",
#         "aggregation":     "sum",
#         "expected_result": 33.8,
#         "unit":            "%",
#         "notes":           "Derived: SUM(Converted) / SUM(Total) × 100",
#     },
#     {
#         "query":           "Show me leads by channel",
#         "dataset":         "sf_campaign",
#         "metric_intent":   "leads_total",
#         "aggregation":     "group_by",
#         "group_by":        "channel",
#         "expected_result": [{"group": "Email", "value": 5990}],
#         "notes":           "Top channel = Email with 5990 leads",
#     },
#     {
#         "query":           "What is total profit?",
#         "dataset":         "sf_campaign",
#         "metric_intent":   "profit",
#         "aggregation":     "sum",
#         "expected_result": 43_692_496,
#         "unit":            "₹",
#         "notes":           "Either SUM(Profit ₹) directly or Revenue − Cost",
#     },

#     # ── Salesforce Sales ──────────────────────────────────────────────────
#     {
#         "query":           "What is total sales revenue?",
#         "dataset":         "sf_sales",
#         "metric_intent":   "revenue_actual",
#         "aggregation":     "sum",
#         "expected_result": 97_411_108,
#         "unit":            "₹",
#         "notes":           "Maps to actual revenue column, not expected",
#     },
#     {
#         "query":           "How many deals were won?",
#         "dataset":         "sf_sales",
#         "metric_intent":   "leads_converted",
#         "aggregation":     "sum",
#         "expected_result": 11_676,
#         "unit":            "count",
#         "notes":           "Maps to 'Closed Won Leads' or equivalent",
#     },

#     # ── Google Ads ────────────────────────────────────────────────────────
#     {
#         "query":           "What is the total number of impressions?",
#         "dataset":         "google_ads",
#         "metric_intent":   "impressions",
#         "aggregation":     "sum",
#         "expected_result": 990_190,
#         "unit":            "count",
#     },
#     {
#         "query":           "What is the click-through rate?",
#         "dataset":         "google_ads",
#         "metric_intent":   "ctr",
#         "aggregation":     "sum",
#         "expected_result": 1.8468,
#         "unit":            "%",
#         "notes":           "Derived: SUM(Clicks)/SUM(Impressions)×100 — NOT AVG(CPC)",
#     },
#     {
#         "query":           "What is average cost per click?",
#         "dataset":         "google_ads",
#         "metric_intent":   "cost_per_unit",
#         "aggregation":     "avg",
#         "expected_result": 10.12,
#         "unit":            "₹",
#         "notes":           "AVG of per-row Avg CPC — never sum this",
#     },
#     {
#         "query":           "What is the ROI?",
#         "dataset":         "google_ads",
#         "metric_intent":   "roi",
#         "aggregation":     "sum",
#         "expected_result": 1353.77,
#         "unit":            "%",
#         "notes":           "Derived from Revenue/Cost — no pre-computed ROI column",
#     },

#     # ── Zoho CRM ──────────────────────────────────────────────────────────
#     {
#         "query":           "What is expected revenue?",
#         "dataset":         "zoho_crm",
#         "metric_intent":   "revenue_expected",
#         "aggregation":     "sum",
#         "expected_result": 288_122.79,
#         "unit":            "₹",
#         "notes":           "MUST resolve to revenue_expected, NOT revenue_actual",
#     },
#     {
#         "query":           "What is total marketing cost?",
#         "dataset":         "zoho_crm",
#         "metric_intent":   "cost_total",
#         "aggregation":     "sum",
#         "expected_result": 73_294.7,
#         "unit":            "₹",
#     },

#     # ── Salesforce Lead (Google Ads) ──────────────────────────────────────
#     {
#         "query":           "What is the average cost per lead?",
#         "dataset":         "sf_lead",
#         "metric_intent":   "cost_per_unit",
#         "aggregation":     "avg",
#         "expected_result": 903.52,
#         "unit":            "₹/lead",
#         "notes":           "AVG of per-lead CPL — NEVER sum; Lead ID is not leads_total",
#     },
#     {
#         "query":           "What is CPL for converted leads?",
#         "dataset":         "sf_lead",
#         "metric_intent":   "cost_per_unit",
#         "aggregation":     "avg",
#         "filters":         [{"field": "status", "value": "Converted"}],
#         "expected_result": 867.36,
#         "unit":            "₹/lead",
#         "notes":           "Filter by Lead Status = Converted",
#     },
#     {
#         "query":           "What is total expected revenue?",
#         "dataset":         "sf_lead",
#         "metric_intent":   "revenue_expected",
#         "aggregation":     "sum",
#         "expected_result": 449_072.11,
#         "unit":            "₹",
#     },

#     # ── Zoho Sales ────────────────────────────────────────────────────────
#     {
#         "query":           "What is total deal amount?",
#         "dataset":         "zoho_sales",
#         "metric_intent":   "deal_amount",
#         "aggregation":     "sum",
#         "expected_result": 7_391_705,
#         "unit":            "₹",
#     },
#     {
#         "query":           "What is revenue earned?",
#         "dataset":         "zoho_sales",
#         "metric_intent":   "revenue_actual",
#         "aggregation":     "sum",
#         "expected_result": 1_260_445,
#         "unit":            "₹",
#         "notes":           "Revenue Earned ≠ Deal Amount — different columns",
#     },

#     # ── SugarCRM ──────────────────────────────────────────────────────────
#     {
#         "query":           "What is total revenue?",
#         "dataset":         "sugar_crm",
#         "metric_intent":   "revenue_actual",
#         "aggregation":     "sum",
#         "expected_result": 1_550_711.06,
#         "unit":            "₹",
#     },

#     # ── Marketing Revenue ─────────────────────────────────────────────────
#     {
#         "query":           "Show me revenue by marketing channel",
#         "dataset":         "marketing_rev",
#         "metric_intent":   "revenue_actual",
#         "aggregation":     "group_by",
#         "group_by":        "channel",
#         "expected_result": "breakdown list with at least 3 channels",
#         "notes":           "Verifies Channel dimension is detected",
#     },
# ]
