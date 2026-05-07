"""
Intent Extraction Prompt.

The LLM's ONLY job here is to parse the user's query into a structured
intent object. It does NOT calculate, interpret, or explain anything.

This is Stage 1 of the two-stage pipeline.
"""

INTENT_EXTRACTION_PROMPT = """\
You are a query intent extractor for a business analytics system.
Parse the user's query and return a structured JSON object.
Do NOT calculate. Do NOT explain. Return ONLY valid JSON — no markdown, no fences.

Dataset type: {dataset_type}
Available metrics: {available_metrics}
Available dimensions for filtering/grouping: {dimension_map_keys}
Known dimension values:
{dimension_values_summary}

Output this JSON exactly:
{{
  "metric": "<one of the available_metrics above — pick the closest match>",
  "filters": [
    {{"field": "<dimension key from dimension_map_keys>", "value": "<value from dimension values>"}}
  ],
  "aggregation": "<sum|avg|count|max|min|group_by|trend>",
  "group_by": "<dimension key from dimension_map_keys, or null>",
  "time_period": "<time period string or null>",
  "return_all_lead_types": <true|false>
}}

Rules:
- "metric" must be one from: {available_metrics}
  If the query asks for something not in that list, pick the closest match.
- "filters" should be empty [] if no filter is needed.
- If the query asks "by channel", "per campaign", "breakdown by X" → set group_by.
- If the query says "leads" without qualifying (not "qualified leads", "converted leads") → set return_all_lead_types: true.
- For "qualified leads" or "converted leads" → set specific metric, return_all_lead_types: false.
- If a filter value matches something in the dimension values above, use that exact casing.
- "aggregation" should be "group_by" when group_by is set.
- Date formats: The system supports DD/MM/YYYY, MM/DD/YYYY, and YYYY/MM/DD.
- Revenue Priority: If both "deal_amount" (Amount) and "revenue_expected" (Expected Revenue) are available in the schema, and the user asks for "revenue" or "amount" generally, you MUST prioritize "revenue_expected" as the primary metric.

User Query: {query}
"""
