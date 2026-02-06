import pandas as pd
import sys

def main():
    # Load data
    try:
        df_apps = pd.read_csv('analysis_csv/python_frameworks.csv')
        df_rq1 = pd.read_csv('analysis_csv/RQ1_results.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Add tech to all RQ1 projects
    df_rq1['tech'] = df_rq1['llm_full'].map(
        df_apps.set_index('app_id')['python_techs'].to_dict()
    ).fillna('Plain')
    
    # Reorder: tech first
    cols = ['tech'] + [col for col in df_rq1.columns if col != 'tech']
    df_rq1_with_tech = df_rq1[cols]

    # Weights per query type
    query_precision = {
        'find_abort_except': 1.0,
        'find_logging_except': 0.8,
        'find_pass_only_except': 0.85,
        'find_todo_except': 1.0
    }
    
    # Calculate weighted score
    df_rq1_with_tech['weighted_error'] = df_rq1_with_tech['query'].map(query_precision).fillna(0)

    # Total files per tech (unique lm_full)
    total_projects = df_rq1_with_tech.groupby('tech')['llm_full'].nunique()

    # Error per tech
    error_mask = df_rq1_with_tech['row_count'] > 0
    df_errors = df_rq1_with_tech[error_mask]
    # Sum because we want total errors per tech and a file can have multiple errors
    raw_errors_per_tech = df_errors.groupby('tech')['row_count'].sum() 

    # Weighted errors (sum per tech)
    weighted_errors_per_tech = df_errors.groupby('tech')['weighted_error'].sum()

    # Summary table
    summary_data = []
    for tech in total_projects.index:
        total = total_projects[tech]
        raw_errors = raw_errors_per_tech.get(tech, 0)
        weighted = int(weighted_errors_per_tech.get(tech, 0))  # NO VIRGOLA
        #raw_pct = round((raw_errors / total * 100), 2) if total > 0 else 0
        weighted_pct = round((weighted / total * 100), 2) if total > 0 else 0
        summary_data.append({
            'Technology': tech,
            'Files (Total)': total,
            'Findings (Raw)': raw_errors,
            'Expected True': weighted,
            'Error (%)': weighted_pct
        })
    
    df_summary = pd.DataFrame(summary_data)

    # Sort by findings
    df_summary = df_summary.sort_values('Findings (Raw)', ascending=False).reset_index(drop=True)
    
    print("\nRQ2 statistics:")
    print(df_summary.to_string(index=False, float_format='%.2f'))

    # Save to csv
    output_file = 'analysis_csv/RQ2_table.csv'
    df_summary.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

if __name__ == "__main__":
    main()
