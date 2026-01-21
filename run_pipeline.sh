#!/bin/bash

set -e  

ROOT_DIR=$(pwd)
LIST_FILE="python_projects_subset.txt"
QUERIES_SRC="$ROOT_DIR/src"

ANALYSIS_DIR="$ROOT_DIR/analysis_csv"
mkdir -p "$ANALYSIS_DIR"


echo "======================================"
echo "Downloading and extracting dataset"
echo "======================================"

python extract_data.py

echo
echo "======================================"
echo "Selecting project subset"
echo "======================================"

find vibe_dataset/apps -name "*.py" -exec dirname {} \; \
    | sort -u \
    > python_projects.txt

python select_subset.py

echo
echo "======================================"
echo "Creating CodeQL databases + running queries"
echo "======================================"

while IFS= read -r project_dir; do
    echo
    echo "--------------------------------------"
    echo "Processing: $project_dir"

    if [ ! -d "$project_dir" ]; then
        continue
    fi

    cd "$project_dir"

    echo "Removing existing CodeQL database..."
    rm -rf codeql-db

    echo "Creating database..."
    codeql database create codeql-db \
        --language=python \
        --source-root=.
    
    echo "Copying queries to project..."
    mkdir -p queries-all
    rm -rf queries-all/src
    cp -r "$QUERIES_SRC" queries-all/

    echo "Running queries..."

    for ql_file in queries-all/src/*.ql; do
        query_name=$(basename "$ql_file" .ql)
        output_file="results-${query_name}.bqrs"

        echo "Running query: $query_name"
        codeql query run \
            --database=codeql-db \
            --output="$output_file" \
            "$ql_file"

        echo "Query results saved to: $output_file"
    done

    echo "Converting BQRS files to CSV..."

    for bqrs_file in results-*.bqrs; do
        [ -e "$bqrs_file" ] || continue

        base_name=$(basename "$bqrs_file" .bqrs)
        csv_file="${base_name}.csv"

        codeql bqrs decode \
            --format=csv \
            --output="$csv_file" \
            "$bqrs_file"

        project_tag=$(echo "$project_dir" | tr '/' '_')
        cp "$csv_file" "$ANALYSIS_DIR/${project_tag}_${csv_file}"
    done

    cd "$ROOT_DIR"

done < "$LIST_FILE"

echo
echo "Pipeline completed."
