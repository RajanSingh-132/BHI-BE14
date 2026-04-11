"""
Unit tests — DatasetProfiler

What we test:
  - Identifier columns are NEVER classified as measures
  - Lead Source / Lead Status are always categorical, never a measure
  - Revenue columns map to the correct role (actual vs expected)
  - Cost columns distinguish total cost from per-unit cost
  - Month / Close Month are never classified as identifiers
  - ROI (%) is always pre_computed_ratio, not numeric_measure
  - Campaign Name is a dimension, not a name field
  - Client/User Name is a dimension, not a name field

Run with:  pytest -s tests/test_dataset_profiler.py
"""

import pytest
from tests.fixtures import (
    SF_CAMPAIGN, SF_SALES, GOOGLE_ADS, MARKETING_REV,
    SUGAR_CRM, ZOHO_CRM, SF_LEAD, ZOHO_SALES,
)
from services.dataset_profiler import profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(prof, col_name):
    """Return metadata dict for a specific column name."""
    return prof["columns"].get(col_name)


def _role(prof, col_name):
    return prof["columns"].get(col_name, {}).get("semantic_role")


def _dtype(prof, col_name):
    return prof["columns"].get(col_name, {}).get("dtype")


# ---------------------------------------------------------------------------
# SugarCRM — crm_leads type, single revenue column
# ---------------------------------------------------------------------------
class TestSugarCRM:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.prof = profile(SUGAR_CRM, "sugar_crm")

    def test_dataset_type(self):
        assert self.prof["dataset_type"] == "crm_leads"

    def test_lead_id_is_identifier(self):
        assert _dtype(self.prof, "Lead ID") == "identifier"

    def test_lead_source_is_categorical(self):
        """Lead Source is a filter dimension — never a measure."""
        assert _dtype(self.prof, "Lead Source") == "categorical"
        assert _role(self.prof, "Lead Source") == "source"

    def test_revenue_is_actual(self):
        assert _role(self.prof, "Revenue (₹)") == "revenue_actual"

    def test_revenue_is_summable(self):
        assert _col(self.prof, "Revenue (₹)")["is_summable"] is True

    def test_status_is_categorical(self):
        assert _dtype(self.prof, "Status") == "categorical"
        assert _role(self.prof, "Status") == "status"

    def test_available_metrics(self):
        assert "revenue_actual" in self.prof["available_metrics"]

    def test_created_date_is_date(self):
        assert _dtype(self.prof, "Created Date") == "date"


# ---------------------------------------------------------------------------
# Salesforce Campaign — full marketing_campaign dataset
# ---------------------------------------------------------------------------
class TestSalesforceCampaign:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.prof = profile(SF_CAMPAIGN, "sf_campaign")

    def test_dataset_type(self):
        assert self.prof["dataset_type"] == "marketing_campaign"

    def test_campaign_id_is_identifier(self):
        assert _dtype(self.prof, "Campaign ID") == "identifier"

    def test_campaign_name_is_campaign_dimension(self):
        """Campaign Name must be a dimension, not a 'name' field."""
        assert _dtype(self.prof, "Campaign Name") == "categorical"
        assert _role(self.prof, "Campaign Name") == "campaign"

    def test_leads_generated_is_leads_total(self):
        assert _role(self.prof, "Leads Generated") == "leads_total"
        assert _col(self.prof, "Leads Generated")["is_summable"] is True

    def test_qualified_leads_correct_role(self):
        assert _role(self.prof, "Qualified Leads") == "leads_qualified"

    def test_converted_leads_correct_role(self):
        assert _role(self.prof, "Converted Leads") == "leads_converted"

    def test_campaign_cost_is_cost_total(self):
        """Campaign Cost must map to cost_total, NOT cost_per_unit."""
        assert _role(self.prof, "Campaign Cost (₹)") == "cost_total"
        assert _col(self.prof, "Campaign Cost (₹)")["is_summable"] is True

    def test_revenue_generated_is_actual(self):
        assert _role(self.prof, "Revenue Generated (₹)") == "revenue_actual"

    def test_roi_is_precomputed_ratio(self):
        assert _dtype(self.prof, "ROI (%)") == "pre_computed_ratio"
        assert _role(self.prof, "ROI (%)") == "roi"
        assert _col(self.prof, "ROI (%)")["is_summable"] is False

    def test_month_is_not_identifier(self):
        """Month strings like 'Mar-2026' must not be classified as identifiers."""
        assert _dtype(self.prof, "Month") != "identifier"

    def test_channel_is_dimension(self):
        assert _role(self.prof, "Marketing Channel") == "channel"
        assert "channel" in self.prof["dimension_map"]

    def test_all_expected_metrics_present(self):
        expected = {
            "leads_total", "leads_qualified", "leads_converted",
            "cost_total", "revenue_actual", "profit", "roi",
        }
        assert expected.issubset(set(self.prof["available_metrics"]))


# ---------------------------------------------------------------------------
# Salesforce Lead (Google Ads) — expected revenue vs cost per lead
# ---------------------------------------------------------------------------
class TestSalesforceLeadGoogleAds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.prof = profile(SF_LEAD, "sf_lead")

    def test_cost_per_lead_is_not_cost_total(self):
        """Cost Per Lead is a unit rate — must NOT map to cost_total."""
        assert _role(self.prof, "Cost Per Lead (₹)") == "cost_per_unit"

    def test_cost_per_lead_is_not_summable(self):
        """Summing per-unit costs is meaningless — must be False."""
        assert _col(self.prof, "Cost Per Lead (₹)")["is_summable"] is False

    def test_expected_revenue_is_not_actual(self):
        """'Expected Revenue' is pipeline, not earned — must NOT be revenue_actual."""
        assert _role(self.prof, "Expected Revenue (₹)") == "revenue_expected"
        assert _role(self.prof, "Expected Revenue (₹)") != "revenue_actual"

    def test_campaign_name_is_dimension(self):
        assert _dtype(self.prof, "Campaign Name") == "categorical"
        assert _role(self.prof, "Campaign Name") == "campaign"

    def test_google_click_id_is_identifier(self):
        assert _dtype(self.prof, "Google Click ID") == "identifier"


# ---------------------------------------------------------------------------
# Google Ads — click/impression dataset
# ---------------------------------------------------------------------------
class TestGoogleAds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.prof = profile(GOOGLE_ADS, "google_ads")

    def test_dataset_type(self):
        assert self.prof["dataset_type"] == "google_ads"

    def test_impressions_role(self):
        assert _role(self.prof, "Impressions") == "impressions"
        assert _col(self.prof, "Impressions")["is_summable"] is True

    def test_clicks_role(self):
        assert _role(self.prof, "Clicks") == "clicks"

    def test_avg_cpc_is_cost_per_unit_not_cost_total(self):
        assert _role(self.prof, "Avg CPC (₹)") == "cost_per_unit"
        assert _col(self.prof, "Avg CPC (₹)")["is_summable"] is False

    def test_cost_is_cost_total(self):
        assert _role(self.prof, "Cost (₹)") == "cost_total"
        assert _col(self.prof, "Cost (₹)")["is_summable"] is True

    def test_conversions_role(self):
        assert _role(self.prof, "Conversions") == "conversions"


# ---------------------------------------------------------------------------
# Zoho CRM — cost vs expected revenue
# ---------------------------------------------------------------------------
class TestZohoCRM:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.prof = profile(ZOHO_CRM, "zoho_crm")

    def test_cost_is_cost_total(self):
        """Plain 'Cost (₹)' without 'per' qualifier = cost_total."""
        assert _role(self.prof, "Cost (₹)") == "cost_total"

    def test_expected_revenue_is_not_actual(self):
        assert _role(self.prof, "Expected Revenue (₹)") == "revenue_expected"


# ---------------------------------------------------------------------------
# Zoho Sales — deal_amount vs revenue_earned
# ---------------------------------------------------------------------------
class TestZohoSales:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.prof = profile(ZOHO_SALES, "zoho_sales")

    def test_dataset_type(self):
        assert self.prof["dataset_type"] == "sales_pipeline"

    def test_deal_amount_role(self):
        assert _role(self.prof, "Deal Amount (₹)") == "deal_amount"

    def test_revenue_earned_is_actual(self):
        assert _role(self.prof, "Revenue Earned (₹)") == "revenue_actual"

    def test_deal_stage_is_status_dimension(self):
        assert _role(self.prof, "Deal Stage") == "status"

    def test_lead_id_is_identifier(self):
        assert _dtype(self.prof, "Lead ID") == "identifier"

    def test_deal_id_is_identifier(self):
        assert _dtype(self.prof, "Deal ID") == "identifier"

    def test_deal_close_date_is_not_identifier(self):
        """Deal Close Date has date-like string values but must not be an identifier."""
        assert _dtype(self.prof, "Deal Close Date") != "identifier"


# ---------------------------------------------------------------------------
# Marketing Revenue — tests that Client/User Name is a client dimension
# ---------------------------------------------------------------------------
class TestMarketingRevenue:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.prof = profile(MARKETING_REV, "marketing_rev")

    def test_client_user_name_is_dimension(self):
        """'Client/User Name' should be a client dimension, not a name field."""
        assert _dtype(self.prof, "Client/User Name") == "categorical"
        assert _role(self.prof, "Client/User Name") == "client"

    def test_campaign_name_is_dimension(self):
        assert _dtype(self.prof, "Campaign Name") == "categorical"
        assert _role(self.prof, "Campaign Name") == "campaign"

    def test_marketing_spend_is_cost_total(self):
        assert _role(self.prof, "Marketing Spend (₹)") == "cost_total"

    def test_dimension_map_not_empty(self):
        """Marketing Rev dataset must have at least client and campaign dimensions."""
        assert len(self.prof["dimension_map"]) >= 2
