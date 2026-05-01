import pandas as pd
import numpy as np
from typing import Any, Dict, List
from prompts.productivity_prompt import classify_productivity_status, BUCKET_ORDER
from .analytics_utils import _find_col, _round

def calculate_productivity_metrics(dataset_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate productivity metrics using Status and Priority overrides."""
    all_dfs = []
    for entry in dataset_entries:
        data = entry.get("data")
        if data:
            all_dfs.append(pd.DataFrame(data))
    
    if not all_dfs:
        return {}

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.dropna(how='all') # Drop empty rows
    if df.empty:
        return {}
        
    total_tasks = len(df)

    # --- 1. Identify Columns ---
    status_col = _find_col(df, ["Status", "status", "state", "Stage", "Task Status", "Progress"])
    assignee_col = _find_col(df, ["Assignee", "Owner", "assigned_to", "owner", "User", "Team Member", "Resource"])
    project_col = _find_col(df, ["Project_Name", "project", "Project", "Category", "Department"])
    priority_col = _find_col(df, ["Priority", "Severity", "priority", "severity", "Urgency", "Impact"])

    # --- 2. Classification Logic ---
    
    # Priority-First Classification: The user expects the buckets to reflect Priority levels
    if priority_col:
        df['bucket'] = df[priority_col].apply(classify_productivity_status)
    elif status_col:
        df['bucket'] = df[status_col].apply(classify_productivity_status)
    else:
        df['bucket'] = "Low"

    # Optional: Ensure "Stuck" or "Blocker" status still counts as Critical if detected in Status
    if status_col:
        stuck_mask = df[status_col].astype(str).str.lower().str.contains("stuck|block", na=False)
        df.loc[stuck_mask, 'bucket'] = "Critical"

    # --- 3. Generate Aggregates ---
    counts = df['bucket'].value_counts().to_dict()
    high_count = counts.get("High", 0)
    medium_count = counts.get("Medium", 0)
    low_count = counts.get("Low", 0)
    critical_count = counts.get("Critical", 0)

    # Performer Stats (Only if assignee detected)
    performers = []
    if assignee_col:
        # Define resolved keywords for the formula
        resolved_keywords = ["resolved", "closed", "done", "success", "finish", "completed", "close"]
        
        if status_col:
            df['is_resolved'] = df[status_col].astype(str).str.lower().apply(
                lambda x: any(kw in x for kw in resolved_keywords)
            )
        else:
            df['is_resolved'] = False

        perf_group = df.groupby(assignee_col).agg(
            tasks=(assignee_col, 'count'),
            resolved=('is_resolved', 'sum')
        ).reset_index()
        
        # Resolution Rate (%) = (Bugs Resolved by Owner / Bugs Assigned to Owner) × 100
        perf_group['rate'] = ((perf_group['resolved'] / perf_group['tasks']) * 100).apply(lambda x: _round(x))
        
        performers = perf_group.rename(columns={assignee_col: 'name'}).to_dict('records')
        # Sort for top performers (highest rate first)
        performers.sort(key=lambda x: x['rate'], reverse=True)

    # Project Stats (Dynamic breakdown)
    proj_col_name = project_col if project_col else "General"
    project_summary = []
    project_summary_cols = []
    
    if project_col:
        # If priority exists, use Priority buckets
        if priority_col:
            project_summary_cols = ["High", "Medium", "Low", "Critical"]
            proj_group = df.groupby([proj_col_name, 'bucket']).size().unstack(fill_value=0)
            for b in project_summary_cols:
                if b not in proj_group.columns: proj_group[b] = 0
            
            # Ensure Total column
            proj_group['Total'] = proj_group.sum(axis=1)
            project_summary = proj_group.reset_index().rename(columns={proj_col_name: 'project'}).to_dict('records')
        # If no priority, use Top 4 Statuses as columns
        elif status_col:
            # Re-calculate top statuses here for the summary
            raw_s = df[status_col].astype(str).value_counts().head(4).index.tolist()
            project_summary_cols = [str(s) for s in raw_s if str(s).lower() not in ["nan", "none", ""]]
            
            # Group and Pivot
            proj_group = df.groupby([proj_col_name, status_col]).size().unstack(fill_value=0)
            for s in project_summary_cols:
                if s not in proj_group.columns: proj_group[s] = 0
            
            # Keep only project_summary_cols + Total
            proj_group['Total'] = proj_group.sum(axis=1)
            project_summary = proj_group.reset_index().rename(columns={proj_col_name: 'project'}).to_dict('records')

    # --- 4. Severity Distribution ---
    severity_col = _find_col(df, ["Severity", "severity", "Impact"])
    severity_summary = []
    if severity_col:
        def standardize_severity(s: Any) -> str:
            s_low = str(s).lower()
            if "block" in s_low: return "Blocker"
            if "critical" in s_low: return "Critical"
            if "major" in s_low: return "Major"
            if "minor" in s_low: return "Minor"
            return "Other"
        
        df['severity_clean'] = df[severity_col].apply(standardize_severity)
        sev_counts = df['severity_clean'].value_counts()
        for level in ["Blocker", "Critical", "Major", "Minor"]:
            severity_summary.append({
                "name": level,
                "value": int(sev_counts.get(level, 0))
            })

    # --- 5. Raw Status Distribution (For "Status Wise Distribution" chart) ---
    status_distribution = []
    if status_col:
        raw_counts = df[status_col].astype(str).value_counts().to_dict()
        # Filter out "nan" strings and empty values
        status_distribution = [
            {"status": str(k), "count": int(v)} 
            for k, v in raw_counts.items() 
            if str(k).lower() not in ["nan", "none", ""]
        ]

    # --- 6. Dynamic KPI Cards ---
    kpi_cards = [
        {"label": "Total", "value": total_tasks, "color": "text-slate-900"}
    ]

    # If priority column exists, use Priority buckets
    if priority_col:
        kpi_cards.extend([
            {"label": "Critical", "value": int(critical_count), "color": "text-rose-600"},
            {"label": "High", "value": int(high_count), "color": "text-blue-600"},
            {"label": "Medium", "value": int(medium_count), "color": "text-amber-600"},
            {"label": "Low", "value": int(low_count), "color": "text-slate-500"}
        ])
    # Fallback: If no priority, use the Top Statuses as KPIs
    else:
        # Sort status distribution by count to get the most significant ones
        sorted_statuses = sorted(status_distribution, key=lambda x: x['count'], reverse=True)
        colors = ["text-rose-600", "text-blue-600", "text-amber-600", "text-slate-500"]
        for i, s in enumerate(sorted_statuses[:4]):
            kpi_cards.append({
                "label": str(s['status']).upper(),
                "value": s['count'],
                "color": colors[i % len(colors)]
            })

    completion_rate = _round((high_count / total_tasks * 100) if total_tasks > 0 else 0)

    return {
        "totalTasks": total_tasks,
        "highProductivity": int(high_count),
        "mediumProductivity": int(medium_count),
        "lowProductivity": int(low_count),
        "criticalProductivity": int(critical_count),
        "completionRate": completion_rate,
        "topPerformers": performers[:5],
        "lowPerformers": sorted(performers, key=lambda x: x['rate'])[:5],
        "performersDistribution": performers,
        "projectSummary": project_summary,
        "projectSummaryColumns": project_summary_cols,
        "severityDistribution": severity_summary,
        "statusDistribution": status_distribution,
        "kpiCards": kpi_cards,
        "columnLabels": {
            "assignee": assignee_col if assignee_col else "Assignee",
            "status": status_col if status_col else "Status",
            "priority": priority_col if priority_col else "Priority",
            "project": project_col if project_col else "Project"
        },
        "byStatus": [{"status": b, "count": int(counts.get(b, 0))} for b in BUCKET_ORDER]
    }
