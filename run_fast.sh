#!/bin/bash
set -euo pipefail

# ============================================================
# run_pipeline.sh  (WORKING with older CodeQL CLI syntax)
#
# Goals:
# - Create CodeQL DBs per project
# - Run your QL queries
# - Decode BQRS -> CSV
# - Parallelize safely
# - Resumable (skip per-query CSVs already produced)
#
# IMPORTANT for your CodeQL version:
# - NO "codeql pack install --dir ..."  (unsupported)
#   -> must run inside the pack dir: (cd src && codeql pack install)
#
# This script DOES NOT use `codeql database analyze --format=csv`
# (which was causing metadata/result-pattern issues).
# ============================================================

ROOT_DIR="$(pwd)"
LIST_FILE="python_projects_subset.txt"
QUERIES_DIR="$ROOT_DIR/src"          # must contain qlpack.yml + *.ql
OUT_DIR="$ROOT_DIR/analysis_csv"
mkdir -p "$OUT_DIR"

export CODEQL_ALLOW_INSTALLATION_ANYWHERE=true

# Parallel knobs (tune)
JOBS="${JOBS:-2}"        # projects in parallel (start with 1-2)
THREADS="${THREADS:-2}"  # CodeQL threads per query/db op

# -------------------------
# Preflight
# -------------------------
command -v codeql >/dev/null 2>&1 || { echo "ERROR: codeql not found in PATH"; exit 1; }
[ -f "$LIST_FILE" ] || { echo "ERROR: missing $LIST_FILE"; exit 1; }
[ -f "$QUERIES_DIR/qlpack.yml" ] || { echo "ERROR: missing $QUERIES_DIR/qlpack.yml"; exit 1; }

# Ensure we have .ql files
shopt -s nullglob
QL_FILES=("$QUERIES_DIR"/*.ql)
if [ "${#QL_FILES[@]}" -eq 0 ]; then
  echo "ERROR: no .ql files in $QUERIES_DIR"
  exit 1
fi

# If you interrupt, stop child jobs to reduce half-built DBs.
cleanup() {
  echo
  echo "[cleanup] stopping running jobs..."
  jobs -pr | xargs -r kill 2>/dev/null || true
}
trap cleanup INT TERM

echo "======================================"
echo "Downloading and extracting dataset"
echo "======================================"
#python extract_data.py

echo
echo "======================================"
echo "Selecting project subset"
echo "======================================"
find vibe_dataset/apps -name "*.py" -exec dirname {} \; | sort -u > python_projects.txt
python select_subset.py

echo
echo "======================================"
echo "Installing CodeQL pack dependencies (once)"
echo "======================================"
# Older CLI: run pack install *inside* the pack dir (or as positional arg)
(
  cd "$QUERIES_DIR"
  codeql pack install
)

analyze_one() {
  local project_dir="$1"
  [ -d "$project_dir" ] || return 0

  local project_tag
  project_tag="$(echo "$project_dir" | tr '/' '_')"

  local project_out="$OUT_DIR/$project_tag"
  mkdir -p "$project_out"

  # Recompute query list inside worker (so it works under parallel/xargs subshells)
  local ql
  local any_missing=0
  for ql in "$QUERIES_DIR"/*.ql; do
    [ -e "$ql" ] || continue
    local qname
    qname="$(basename "$ql" .ql)"
    if [ ! -f "$project_out/${qname}.csv" ]; then
      any_missing=1
      break
    fi
  done

  if [ "$any_missing" -eq 0 ]; then
    echo "Skip (already analyzed): $project_dir"
    return 0
  fi

  echo "--------------------------------------"
  echo "Processing: $project_dir"

  pushd "$project_dir" >/dev/null

  # If DB exists but is broken/unfinalized, remove it.
  if [ -d "codeql-db" ]; then
    # Older CLI may error on finalize if not finalized; treat as broken
    if ! codeql database finalize codeql-db >/dev/null 2>&1; then
      echo "DB invalid/unfinalized -> rebuilding: $project_dir"
      rm -rf codeql-db
    fi
  fi

  if [ ! -d "codeql-db" ]; then
    echo "DB create: $project_dir"
    codeql database create codeql-db \
      --language=python \
      --source-root=. \
      --command="python3 -c 'print(0)'" \
      --threads="$THREADS"
    # finalize (ignore failure if already finalized)
    codeql database finalize codeql-db >/dev/null 2>&1 || true
  else
    echo "DB reuse:  $project_dir"
  fi

  # Run each query and decode -> CSV
  for ql in "$QUERIES_DIR"/*.ql; do
    [ -e "$ql" ] || continue
    qname="$(basename "$ql" .ql)"
    csv="$project_out/${qname}.csv"
    bqrs="$project_out/${qname}.bqrs"

    # Resume per-query
    if [ -f "$csv" ]; then
      continue
    fi

    echo "  Query: $qname"
    # Always rerun to avoid stale BQRS metadata/pattern issues
    rm -f "$bqrs" "$csv"

    codeql query run "$ql" \
      --database=codeql-db \
      --output="$bqrs" \
      --threads="$THREADS"

    codeql bqrs decode "$bqrs" \
      --format=csv \
      --output="$csv"
  done

  popd >/dev/null
}

export -f analyze_one
export ROOT_DIR LIST_FILE QUERIES_DIR OUT_DIR THREADS

echo
echo "======================================"
echo "Running CodeQL (parallel projects: $JOBS, threads/project: $THREADS)"
echo "======================================"

if command -v parallel >/dev/null 2>&1; then
  # Fail fast so you see the first real error
  parallel -j "$JOBS" --line-buffer --halt soon,fail=1 analyze_one :::: "$LIST_FILE"
else
  # xargs fallback
  xargs -r -n 1 -P "$JOBS" bash -lc 'analyze_one "$0"' < "$LIST_FILE"
fi

echo
echo "Pipeline completed."
echo "Results:"
echo "  $OUT_DIR/<project_tag>/*.csv"

