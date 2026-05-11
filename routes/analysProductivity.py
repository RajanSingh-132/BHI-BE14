import pandas as pd
import numpy as np
from typing import Any, Dict, List
from prompts.productivity_prompt import classify_productivity_status, BUCKET_ORDER
from .analytics_utils import _find_col, _round, _to_numeric, _smart_parse_dates

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
    milestone_col = _find_col(df, ["Milestone_Name", "Milestone"])
    priority_col = _find_col(df, ["Priority", "Severity", "priority", "severity", "Urgency", "Impact"])
    progress_col = _find_col(df, ["Progress", "progress", "Percentage", "Completion", "Progres"])
    end_date_col = _find_col(df, ["End_Date", "Close_Date", "Due_Date", "Finish_Date"])

    # --- 2. Classification Logic ---
    
    # Identify Completed tasks for the Milestone table
    df['is_actually_complete'] = False
    if status_col:
        df['is_actually_complete'] = df[status_col].astype(str).str.lower().str.contains("completed|done|success", na=False)
    if progress_col:
        df['is_actually_complete'] = df['is_actually_complete'] | (_to_numeric(df[progress_col]) == 100)

    # Priority-First Classification
    if priority_col:
        df['bucket'] = df[priority_col].apply(classify_productivity_status)
    elif status_col:
        df['bucket'] = df[status_col].apply(classify_productivity_status)
    else:
        df['bucket'] = "Low"

    # Optional: Stuck/Blocker override
    if status_col:
        stuck_mask = df[status_col].astype(str).str.lower().str.contains("stuck|block", na=False)
        df.loc[stuck_mask, 'bucket'] = "Critical"

    # --- 3. Generate Aggregates ---
    counts = df['bucket'].value_counts().to_dict()
    high_count = counts.get("High", 0)
    medium_count = counts.get("Medium", 0)
    low_count = counts.get("Low", 0)
    critical_count = counts.get("Critical", 0)

    # Recent Milestones (Dates <= Today)
    recent_milestones = []
    if end_date_col and project_col:
        temp_df = df.copy()
        temp_df['_end_dt'] = _smart_parse_dates(temp_df[end_date_col])
        temp_df = temp_df.dropna(subset=['_end_dt'])
        
        import datetime
        today = pd.to_datetime(datetime.date.today())
        
        # Filter for dates <= today
        if not temp_df.empty:
            # 1. Calculate Absolute Max Date per project (Entire Dataset)
            abs_max_dates = temp_df.groupby(project_col)['_end_dt'].max()

            # 2. Filter for Current/Past milestones to get Status/Owner
            past_df = temp_df[temp_df['_end_dt'] <= today]
            
            if not past_df.empty:
                for proj_name, group in past_df.groupby(project_col):
                    # Get the most recent record that has already passed or is today
                    current_max_date = group['_end_dt'].max()
                    latest_records = group[group['_end_dt'] == current_max_date]
                    
                    # Absolute deadline for this project
                    project_deadline = abs_max_dates.get(proj_name)
                    deadline_str = project_deadline.strftime('%Y-%m-%d') if pd.notna(project_deadline) else "N/A"

                    for _, row in latest_records.iterrows():
                        owner_val = str(row[assignee_col]) if assignee_col and pd.notna(row[assignee_col]) else "Unassigned"
                        if owner_val.lower() != str(proj_name).lower():
                            recent_milestones.append({
                                "project": str(proj_name),
                                "date": deadline_str, # This is the Absolute Max Date
                                "milestone": str(row[milestone_col]) if milestone_col and pd.notna(row[milestone_col]) else "General",
                                "status": str(row[status_col]) if status_col and pd.notna(row[status_col]) else "Unknown"
                            })

    # Performer Stats
    performers = []
    if assignee_col:
        resolved_keywords = ["resolved", "closed", "done", "success", "finish", "completed", "close"]
        if status_col:
            df['is_resolved'] = df[status_col].apply(
                lambda x: any(kw in x for kw in resolved_keywords) if isinstance(x, str) else False
            )
        else:
            df['is_resolved'] = False

        perf_group = df.groupby(assignee_col).agg(
            tasks=(assignee_col, 'count'),
            resolved=('is_resolved', 'sum')
        ).reset_index()
        
        perf_group['rate'] = ((perf_group['resolved'] / perf_group['tasks']) * 100).apply(lambda x: _round(x))
        performers = perf_group.rename(columns={assignee_col: 'name'}).to_dict('records')
        performers.sort(key=lambda x: x['rate'], reverse=True)

    # Project Stats (Dynamic breakdown)
    proj_col_name = project_col if project_col else "General"
    project_summary = []
    project_summary_cols = []
    
    if project_col:
        # NEW LOGIC: If both Project and Milestone exist, pivot them with Total and Completed rows
        if milestone_col:
            raw_m = df[milestone_col].astype(str).value_counts().head(6).index.tolist()
            project_summary_cols = [str(m) for m in raw_m if str(m).lower() not in ["nan", "none", ""]]
            
            # 1. Total counts
            total_pivot = df.groupby([proj_col_name, milestone_col]).size().unstack(fill_value=0)
            
            # 2. Completed counts
            completed_df = df[df['is_actually_complete'] == True]
            completed_pivot = completed_df.groupby([proj_col_name, milestone_col]).size().unstack(fill_value=0)
            
            # Combine into a structured list
            for proj in total_pivot.index:
                # Add Total Row
                row_total = {"project": proj, "row_type": "total"}
                for m in project_summary_cols:
                    row_total[m] = int(total_pivot.loc[proj, m]) if m in total_pivot.columns else 0
                row_total["Total"] = sum(row_total[m] for m in project_summary_cols)
                project_summary.append(row_total)
                
                # Add Completed Row
                row_comp = {"project": "Completed", "row_type": "completed"}
                for m in project_summary_cols:
                    row_comp[m] = int(completed_pivot.loc[proj, m]) if proj in completed_pivot.index and m in completed_pivot.columns else 0
                row_comp["Total"] = sum(row_comp[m] for m in project_summary_cols)
                project_summary.append(row_comp)
            
        # Fallback 1: If priority exists, use Priority buckets
        elif priority_col:
            project_summary_cols = ["High", "Medium", "Low", "Critical"]
            proj_group = df.groupby([proj_col_name, 'bucket']).size().unstack(fill_value=0)
            for b in project_summary_cols:
                if b not in proj_group.columns: proj_group[b] = 0
            
            proj_group['Total'] = proj_group.sum(axis=1)
            project_summary = proj_group.reset_index().rename(columns={proj_col_name: 'project'}).to_dict('records')
            
        # Fallback 2: Use Top 4 Statuses
        elif status_col:
            raw_s = df[status_col].astype(str).value_counts().head(4).index.tolist()
            project_summary_cols = [str(s) for s in raw_s if str(s).lower() not in ["nan", "none", ""]]
            
            proj_group = df.groupby([proj_col_name, status_col]).size().unstack(fill_value=0)
            for s in project_summary_cols:
                if s not in proj_group.columns: proj_group[s] = 0
            
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
            "project": project_col if project_col else "Project",
        },
        "byStatus": [{"status": b, "count": int(counts.get(b, 0))} for b in BUCKET_ORDER],
        "recentMilestones": recent_milestones
    }
