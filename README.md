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