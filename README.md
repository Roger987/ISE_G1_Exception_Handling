# ISE_G1_Exception_Handling

This project runs a CodeQL-based analysis to detect exception handling issues in vibe-coded Python web applications from the Invicti dataset.

## Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run
From the project root:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This script:

1. Downloads and extracts the dataset; 
2. Selects a subset of python projects; 
3. Creates a CodeQL database for each project;
4. Runs all custom CodeQL queries;
5. Exports results to CSV.

The results are saved in the folder analysis_csv/. 

Each CSV file corresponds to one CodeQL query applied to one project.

To run multiple analyses in parallel  run:
```bash
chmod +x run_dast.sh
JOBS=4 THREADS=2 ./run_fast.sh
```

## Ground Truth
This allows user to check if the issues raised by CodeQL are legitimate.

From the project root run:
```bash
streamlit run ground_truth.py --   --analysis-dir analysis_csv   --source-root vibe_dataset/apps   --out-dir analysis_labeling   --sample-per-query 10   --snippet-context 15   --min-lines-after 8

```

## RQ1 Analysis

1. Runs descriptive analysis, adjusts results based on the CodeQL precision obtined by ground_truth.py
2. Generates plots and charts.
3. Calculates Kruskal-Wallis H-test, along its assumptions.
4. Runs post-hoc: Dunn test with Benjamini-Hochberg correction to limit False Discovery Rate.

From the project root:
```bash
python3 rq1_statistical_analysis.py --analysis-dir analysis_csv --out-dir final_results --stats-group llm_family --use-inline-calibration --stats-level file --stats-min-n 5 --stats-fail-small-n
```

## RQ2 Analysis
From the project root:
```bash
chmod +x run_tech_scanner.sh
./run_tech_scanner.sh
python3 rq2_analysis.py
```

1. These scripts categorize the previoulsy downloaded dataset, exporting the Python frameworks found to a CSV;
2. Aggregate the results with the one from RQ1 and export table result to a CSV.

Finally, the following script create a histogram image from the previous generated table.
```bash
python3 rq2_graph.py
```
