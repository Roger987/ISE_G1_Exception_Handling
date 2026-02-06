#!/usr/bin/env python3
"""
ground_truth.py

Streamlit UI to manually label CodeQL findings as TRUE/FALSE and compute per-query precision
P(true | flagged). Shows a code snippet for each finding.

Works with BOTH result layouts:
A) Folder layout (your current structure):
   analysis_csv/<project_folder>/<query>.csv
   where <project_folder> is the FULL project folder name under vibe_dataset/apps/

B) Flat layout:
   analysis_csv/vibe_dataset_apps_<...>_results-<query>.csv

Key features:
- Extracts file:line from ANY row value (e.g., "app/__init__.py:13" or "initialize_db.py:93").
- Resolves paths under the correct per-project root:
    effective_root = <source_root>/<project_folder>  (when folder layout)
- Line-aware disambiguation: prefers candidate files with >= requested line.
- If multiple candidates remain, lets you SEARCH + CHOOSE:
    * filters candidate list by substring
    * if no match, searches WHOLE project folder for matches
- Snippet window ALWAYS shows at least N lines AFTER the highlighted line (default 5),
  even if context is small (if file has enough lines).

Run:
  streamlit run ground_truth.py -- \
    --analysis-dir analysis_csv \
    --source-root vibe_dataset/apps \
    --out-dir analysis_labeling \
    --sample-per-query 20 \
    --snippet-context 25 \
    --min-lines-after 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Naming config (for flat layout)
# -----------------------------
PREFIX = "vibe_dataset_apps_"
RESULTS_MARKERS = ["_results-", "_results_"]


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class FindingRef:
    query: str
    csv_path: str
    row_index: int
    finding_id: str  # stable hash


# -----------------------------
# IO helpers
# -----------------------------
def iter_csv_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.csv"):
        parts = set(p.parts)
        if ".locks" in parts or "_tmp" in parts:
            continue
        yield p


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=True, na_values=["", "NA", "NaN"], on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(
            path, dtype=str, encoding="latin-1", keep_default_na=True, na_values=["", "NA", "NaN"], on_bad_lines="skip"
        )


def infer_query_from_path(csv_path: Path, analysis_dir: Path) -> str:
    """
    Folder layout: analysis_csv/<project_folder>/<query>.csv -> query = stem
    Flat layout: analysis_csv/vibe_dataset_apps_<mid>_results-<query>.csv -> extract <query>
    """
    rel = csv_path.relative_to(analysis_dir)
    if len(rel.parts) >= 2:
        return rel.stem

    stem = rel.stem
    if stem.startswith(PREFIX):
        for m in RESULTS_MARKERS:
            pos = stem.find(m, len(PREFIX))
            if pos != -1:
                return stem[pos + len(m) :]
    m = re.match(r"^.+?_results[-_](?P<q>.+)$", stem)
    if m:
        return m.group("q")
    return stem


def stable_finding_id(query: str, csv_path: str, row_index: int, row_dict: Dict[str, str]) -> str:
    payload = {"query": query, "csv_path": csv_path, "row_index": int(row_index), "row": row_dict}
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8", errors="ignore")
    return hashlib.sha256(s).hexdigest()[:24]


# -----------------------------
# Folder-layout project root mapping
# -----------------------------
def project_folder_from_csv(csv_path: Path, analysis_dir: Path) -> Optional[str]:
    """
    Folder layout:
      analysis_csv/<project_folder>/<query>.csv
    -> project_folder is rel.parts[0]
    """
    try:
        rel = csv_path.relative_to(analysis_dir)
    except Exception:
        return None
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return None


def effective_source_root(csv_path: Path, analysis_dir: Path, global_source_root: Optional[Path]) -> Optional[Path]:
    """
    If folder layout:
      analysis_dir/<project_folder>/<query>.csv
    and global_source_root is vibe_dataset/apps,
    then effective root is:
      global_source_root/<project_folder>

    Otherwise falls back to global_source_root.
    """
    if global_source_root is None:
        return None
    proj = project_folder_from_csv(csv_path, analysis_dir)
    if proj:
        cand = global_source_root / proj
        if cand.is_dir():
            return cand
    return global_source_root


# -----------------------------
# Location extraction (AUTOMATIC)
# -----------------------------
# Matches:
#   app/__init__.py:13
#   initialize_db.py:93
RE_PY_LINE = re.compile(r'(?P<file>[^:\s"\'\}\]]+\.py)\s*:\s*(?P<line>\d+)', re.IGNORECASE)
RE_HASH_LINE = re.compile(r'(?P<file>[^#\s"\'\}\]]+\.py)#L?(?P<line>\d+)', re.IGNORECASE)

JSON_FILE_KEYS = ["file", "path", "uri", "filePath", "filepath", "filename", "source", "sourcePath"]
JSON_LINE_KEYS = ["line", "startLine", "start_line", "beginLine", "lineNumber", "linenumber", "start"]


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def _looks_like_json(s: str) -> bool:
    s = s.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))


def _extract_from_json_obj(obj: Any) -> Tuple[Optional[str], Optional[int]]:
    """
    Recursively walk JSON-like object and look for (file,line).
    """
    if isinstance(obj, dict):
        file_val = None
        line_val = None

        for fk in JSON_FILE_KEYS:
            if fk in obj and obj[fk] is not None:
                file_val = str(obj[fk])
                break
        for lk in JSON_LINE_KEYS:
            if lk in obj and obj[lk] is not None:
                line_val = _to_int(obj[lk])
                break
        if file_val and line_val:
            return file_val, line_val

        for v in obj.values():
            f2, l2 = _extract_from_json_obj(v)
            if f2:
                return f2, l2
        return None, None

    if isinstance(obj, list):
        for v in obj:
            f2, l2 = _extract_from_json_obj(v)
            if f2:
                return f2, l2
        return None, None

    if isinstance(obj, (str, int, float)):
        s = str(obj)
        m = RE_PY_LINE.search(s) or RE_HASH_LINE.search(s)
        if m:
            return m.group("file"), _to_int(m.group("line"))
    return None, None


def extract_file_line_from_row_values(row_dict: Dict[str, str]) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Scan ALL values in row_dict for a usable file+line.
    Returns (file_path_str, line_int, column_used).
    """
    for col, val in row_dict.items():
        if val is None:
            continue
        s = str(val).strip()
        if not s:
            continue

        if _looks_like_json(s):
            try:
                obj = json.loads(s)
                f, l = _extract_from_json_obj(obj)
                if f:
                    return f, l, col
            except Exception:
                pass

        m = RE_HASH_LINE.search(s)
        if m:
            return m.group("file").replace("file://", ""), _to_int(m.group("line")), col

        m = RE_PY_LINE.search(s)
        if m:
            return m.group("file"), _to_int(m.group("line")), col

    return None, None, None


# -----------------------------
# Path resolution (LINE-AWARE) + candidate choice
# -----------------------------
def _count_lines_fast(p: Path) -> Optional[int]:
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def resolve_source_file_line_aware(
    path_value: str,
    eff_root: Optional[Path],
    line_hint: Optional[int],
) -> Tuple[Optional[Path], List[Path], str]:
    """
    Resolve path_value under eff_root.
    If multiple candidates exist, choose one that contains the requested line_hint.

    Returns: (resolved_path, candidates, reason)
    """
    if not path_value:
        return None, [], "no-path"

    pv = str(path_value).strip()
    pv = pv.replace("file://", "")
    pv = pv.split("#", 1)[0]
    pv = pv.strip(' "\'')

    p = Path(pv)
    if p.is_file():
        return p, [p], "absolute"

    if eff_root is None:
        return None, [], "no-effective-root"

    direct = eff_root / pv
    if direct.is_file():
        return direct, [direct], "direct"

    pv_path = Path(pv)
    name = pv_path.name
    wants_suffix = len(pv_path.parts) >= 2

    candidates: List[Path] = []
    try:
        for hit in eff_root.rglob(name):
            if not hit.is_file():
                continue
            if wants_suffix:
                if str(hit).endswith(str(pv_path)):
                    candidates.append(hit)
            else:
                candidates.append(hit)
            if len(candidates) >= 300:
                break
    except Exception:
        pass

    if not candidates:
        return None, [], "no-candidates"

    candidates = sorted(candidates)

    if len(candidates) == 1:
        return candidates[0], candidates, "single-candidate"

    if line_hint is not None and line_hint > 0:
        viable: List[Tuple[int, Path]] = []
        for c in candidates:
            n = _count_lines_fast(c)
            if n is None:
                continue
            if n >= line_hint:
                viable.append((n, c))

        if viable:
            # smallest file that still includes the requested line (often the correct one)
            viable.sort(key=lambda t: (t[0], len(t[1].parts), str(t[1])))
            chosen = viable[0][1]
            return chosen, candidates, f"line-aware (>= {line_hint})"

    return candidates[0], candidates, "fallback-first-sorted"


def choose_candidate_ui(
    *,
    key: str,
    candidates: List[Path],
    default: Optional[Path],
    search_root: Optional[Path],
) -> Path:
    """
    Search + select among multiple candidate Paths.

    - Filters existing candidates by substring (case-insensitive).
    - If filter matches nothing, searches WHOLE project folder (search_root) for matching .py files.
    - Stores selection per finding via key.
    """
    if not candidates and search_root is None:
        raise ValueError("No candidates and no search_root provided.")

    all_opts = [str(p) for p in candidates]

    search_key = f"{key}__search"
    query = st.text_input(
        "Search filename/folder (if no match, searches whole project)",
        value=st.session_state.get(search_key, ""),
        key=search_key,
        placeholder="e.g. qrmenu, backend, initialize_db.py, app/__init__.py",
    ).strip()

    opts = all_opts
    if query:
        q = query.lower()
        opts = [s for s in all_opts if q in s.lower()]

        if not opts and search_root is not None:
            st.info(f"No candidate matches. Searching whole project under: `{search_root}`")
            found: List[str] = []
            try:
                for p in search_root.rglob("*.py"):
                    if q in str(p).lower():
                        found.append(str(p))
                    if len(found) >= 300:
                        break
            except Exception:
                pass

            if found:
                opts = sorted(found)
                st.success(f"Found {len(opts)} matches in the whole project.")
            else:
                st.warning("No matches found in the whole project. Showing original candidates.")
                opts = all_opts

    if not opts:
        opts = all_opts

    # default selection
    default_str = str(default) if default is not None else (opts[0] if opts else "")
    if default_str not in opts:
        default_str = opts[0]
    default_idx = opts.index(default_str)

    chosen = st.selectbox(
        "Choose the source file",
        options=opts,
        index=default_idx,
        key=key,
    )
    return Path(chosen)


# -----------------------------
# Snippet rendering
# -----------------------------
def _format_lines(lines: List[str], start: int, end: int, highlight_line: Optional[int] = None) -> str:
    out = []
    for i in range(start, end):
        lineno = i + 1
        prefix = ">> " if (highlight_line is not None and lineno == highlight_line) else "   "
        out.append(f"{prefix}{lineno:5d} | {lines[i]}")
    return "\n".join(out)


def read_snippet(
    file_path: Path,
    center_line_1based: Optional[int],
    context: int = 10,
    prefer_except: bool = False,
    min_lines_after: int = 5,
) -> Tuple[str, Optional[int], int]:
    """
    Return (formatted snippet, used_center_line, total_lines).

    - Centers on the flagged line by default (prefer_except=False).
    - Always shows at least `min_lines_after` lines after the highlighted line if file has them.
    """
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"[error reading file] {e}", None, 0

    total = len(lines)
    if total == 0:
        return "[empty file]", None, 0

    used_line = center_line_1based
    if used_line is None or used_line <= 0:
        start = 0
        end = min(total, 2 * context + 1)
        return _format_lines(lines, start, end, highlight_line=None), None, total

    used_line = min(max(used_line, 1), total)

    start = max(used_line - context - 1, 0)
    desired_end = used_line + context
    min_end = used_line + max(0, int(min_lines_after))
    end = min(total, max(desired_end, min_end))

    return _format_lines(lines, start, end, highlight_line=used_line), used_line, total


# -----------------------------
# Label persistence
# -----------------------------
def load_labels(labels_path: Path) -> pd.DataFrame:
    if labels_path.exists():
        return pd.read_csv(labels_path, dtype=str)
    return pd.DataFrame(columns=["finding_id", "query", "csv_path", "row_index", "label", "timestamp"])


def upsert_label(labels_path: Path, finding: FindingRef, label: str) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_labels(labels_path)
    now = pd.Timestamp.utcnow().isoformat()

    if (df["finding_id"] == finding.finding_id).any():
        df.loc[df["finding_id"] == finding.finding_id, ["label", "timestamp"]] = [label, now]
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "finding_id": finding.finding_id,
                            "query": finding.query,
                            "csv_path": finding.csv_path,
                            "row_index": str(finding.row_index),
                            "label": label,
                            "timestamp": now,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    df.to_csv(labels_path, index=False)


def label_for(df_labels: pd.DataFrame, finding_id: str) -> Optional[str]:
    m = df_labels[df_labels["finding_id"] == finding_id]
    if m.empty:
        return None
    v = m.iloc[-1]["label"]
    return str(v) if pd.notna(v) and str(v).strip() else None


# -----------------------------
# Sampling (stable list in sample.jsonl)
# -----------------------------
def build_or_load_sample(
    analysis_dir: Path,
    out_dir: Path,
    sample_per_query: int,
    seed: int,
    queries_filter: Optional[List[str]],
) -> List[FindingRef]:
    sample_path = out_dir / "sample.jsonl"
    if sample_path.exists():
        refs: List[FindingRef] = []
        with sample_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                refs.append(FindingRef(**json.loads(line)))
        return refs

    rng = np.random.default_rng(seed)
    buckets: Dict[str, List[Tuple[str, int]]] = {}

    for csv in iter_csv_files(analysis_dir):
        q = infer_query_from_path(csv, analysis_dir)
        if queries_filter and q not in queries_filter:
            continue
        try:
            df = safe_read_csv(csv)
        except Exception:
            continue
        if df.empty:
            continue
        buckets.setdefault(q, []).extend([(str(csv), i) for i in range(len(df))])

    results: List[FindingRef] = []
    for q in sorted(buckets.keys()):
        items = buckets[q]
        if not items:
            continue
        n = min(sample_per_query, len(items))
        chosen_idx = sorted(map(int, rng.choice(len(items), size=n, replace=False)))

        by_csv: Dict[str, List[int]] = {}
        for idx in chosen_idx:
            csv_path, ri = items[idx]
            by_csv.setdefault(csv_path, []).append(ri)

        for csv_path, row_indices in by_csv.items():
            df = safe_read_csv(Path(csv_path))
            for ri in row_indices:
                if ri < 0 or ri >= len(df):
                    continue
                row = df.iloc[ri].to_dict()
                row_dict = {str(k): ("" if pd.isna(v) else str(v)) for k, v in row.items()}
                fid = stable_finding_id(q, csv_path, ri, row_dict)
                results.append(FindingRef(query=q, csv_path=csv_path, row_index=int(ri), finding_id=fid))

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")

    return results


# -----------------------------
# Stats: Wilson confidence interval
# -----------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    rad = (z * ((phat * (1 - phat) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return (max(0.0, center - rad), min(1.0, center + rad))


def compute_precision_table(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame(columns=["query", "n_labeled", "n_true", "precision", "ci_low", "ci_high"])

    lab = labels.copy()
    lab["label_norm"] = lab["label"].astype(str).str.lower().str.strip()
    lab = lab[lab["label_norm"].isin(["true", "false"])]

    rows = []
    for q, sub in lab.groupby("query"):
        n = int(len(sub))
        k = int((sub["label_norm"] == "true").sum())
        prec = (k / n) if n else float("nan")
        lo, hi = wilson_ci(k, n)
        rows.append({"query": q, "n_labeled": n, "n_true": k, "precision": prec, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows).sort_values("query")


# -----------------------------
# Streamlit app
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=str, default="analysis_csv")
    ap.add_argument(
        "--source-root",
        type=str,
        default="",
        help="Root folder of ALL projects (e.g. vibe_dataset/apps)",
    )
    ap.add_argument("--out-dir", type=str, default="analysis_labeling")
    ap.add_argument("--sample-per-query", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--queries", type=str, default="", help="Comma-separated list of queries to review (default: all)")
    ap.add_argument("--snippet-context", type=int, default=25, help="Lines above/below the target line")
    ap.add_argument("--min-lines-after", type=int, default=5, help="Always show at least N lines after the target line")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    global_source_root = Path(args.source_root).resolve() if args.source_root.strip() else None
    queries_filter = [q.strip() for q in args.queries.split(",") if q.strip()] if args.queries.strip() else None

    st.set_page_config(page_title="CodeQL Ground Truth", layout="wide")
    st.title("CodeQL Ground Truth Labeling (TRUE/FALSE)")

    st.sidebar.header("Run configuration")
    st.sidebar.write(f"**analysis_dir:** `{analysis_dir}`")
    st.sidebar.write(f"**source_root:** `{global_source_root}`" if global_source_root else "**source_root:** (none)")
    st.sidebar.write(f"**out_dir:** `{out_dir}`")
    st.sidebar.write(f"**sample_per_query:** `{args.sample_per_query}`")
    st.sidebar.write(f"**snippet_context:** `{args.snippet_context}`")
    st.sidebar.write(f"**min_lines_after:** `{args.min_lines_after}`")

    labels_path = out_dir / "labels.csv"
    labels_df = load_labels(labels_path)

    sample = build_or_load_sample(
        analysis_dir=analysis_dir,
        out_dir=out_dir,
        sample_per_query=args.sample_per_query,
        seed=args.seed,
        queries_filter=queries_filter,
    )
    if not sample:
        st.error("No sampled findings. Check analysis-dir and whether your CSVs contain any rows.")
        st.stop()

    # ----- Summary stats -----
    st.subheader("Precision estimates (from your labels)")
    prec_table = compute_precision_table(labels_df)
    if prec_table.empty:
        st.info("No labels yet. Start labeling to see per-query precision.")
    else:
        disp = prec_table.copy()
        disp["precision"] = disp["precision"].map(lambda x: f"{x:.3f}")
        disp["ci_low"] = disp["ci_low"].map(lambda x: f"{x:.3f}")
        disp["ci_high"] = disp["ci_high"].map(lambda x: f"{x:.3f}")
        st.dataframe(disp, use_container_width=True)

        st.markdown("**Pasteable `INLINE_QUERY_PRECISION`**")
        lines = ["INLINE_QUERY_PRECISION = {"]
        for _, r in prec_table.iterrows():
            lines.append(f'    "{r["query"]}": {float(r["precision"]):.3f},')
        lines.append("}")
        st.code("\n".join(lines), language="python")

    st.divider()

    # ----- Filtering / navigation -----
    q_all = sorted(set(r.query for r in sample))
    st.sidebar.header("View")
    selected_q = st.sidebar.selectbox("Query filter", ["(all)"] + q_all, index=0)
    only_unlabeled = st.sidebar.checkbox("Only unlabeled", value=True)

    filtered = sample
    if selected_q != "(all)":
        filtered = [r for r in filtered if r.query == selected_q]

    if only_unlabeled:
        labeled_ids = set(labels_df["finding_id"].astype(str).tolist()) if not labels_df.empty else set()
        filtered = [r for r in filtered if r.finding_id not in labeled_ids]

    st.sidebar.write(f"Items in view: **{len(filtered)}**")

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if filtered:
        st.session_state.idx = min(st.session_state.idx, len(filtered) - 1)

    if not filtered:
        st.success("Nothing to label in this view (maybe everything is already labeled).")
        st.stop()

    current = filtered[st.session_state.idx]
    current_label = label_for(labels_df, current.finding_id)

    # Load the row
    csv_path = Path(current.csv_path)
    df = safe_read_csv(csv_path)
    if current.row_index < 0 or current.row_index >= len(df):
        st.error("Sample references a row index that doesn't exist anymore. Delete sample.jsonl and rerun.")
        st.stop()

    row = df.iloc[current.row_index].to_dict()
    row_dict = {str(k): ("" if pd.isna(v) else str(v)) for k, v in row.items()}

    # Auto-extract file+line from ANY column values
    file_value, line_int, source_col_used = extract_file_line_from_row_values(row_dict)

    # Effective project root for folder layout
    eff_root = effective_source_root(csv_path, analysis_dir, global_source_root)

    resolved_file, candidates, reason = (None, [], "n/a")
    if file_value:
        resolved_file, candidates, reason = resolve_source_file_line_aware(file_value, eff_root, line_int)

    # Layout
    left, right = st.columns([0.58, 0.42], gap="large")

    with left:
        st.subheader("Code snippet")
        st.write(f"**Query:** `{current.query}`")
        st.write(f"**CSV:** `{current.csv_path}`")
        st.write(f"**Row index:** `{current.row_index}`")
        st.write(f"**Finding ID:** `{current.finding_id}`")

        if source_col_used:
            st.write(f"**Location extracted from column:** `{source_col_used}`")
        else:
            st.warning("Could not automatically extract a file:line like `app/__init__.py:13` from this row.")

        if file_value:
            st.write(f"**Parsed location:** `{file_value}`" + (f":`{line_int}`" if line_int else ""))

        st.caption(f"Effective project root: `{eff_root}`")
        st.caption(f"Resolver: {reason}")

        # If multiple candidates, allow search+choose; if search doesn't match, search whole project
        if candidates and len(candidates) > 1:
            st.warning(f"Multiple candidates ({len(candidates)}) match `{file_value}`.")
            resolved_file = choose_candidate_ui(
                key=f"cand_{current.finding_id}",
                candidates=candidates,
                default=resolved_file,
                search_root=eff_root,
            )
            reason = reason + " + user-selected"
            st.caption(f"Resolver: {reason}")

        if resolved_file and resolved_file.exists():
            snippet, used_line, total_lines = read_snippet(
                resolved_file,
                line_int,
                context=args.snippet_context,
                prefer_except=False,          # center on flagged line
                min_lines_after=args.min_lines_after,
            )
            st.code(snippet, language="text")
            st.caption(
                f"Resolved file: {resolved_file} "
                f"(file lines: {total_lines}, requested: {line_int}, used: {used_line})"
            )
        else:
            st.warning("Could not resolve the parsed file path under the effective source root.")
            st.code(
                f"Parsed file: {file_value}\n"
                f"Line: {line_int}\n"
                f"effective_root: {eff_root}\n"
                f"resolved: {resolved_file}\n"
                f"candidates: {len(candidates)}\n"
                + ("\n".join(str(c) for c in candidates[:50]) if candidates else ""),
                language="text",
            )

    with right:
        st.subheader("Finding details (raw row)")
        st.json(row_dict)

        st.divider()
        st.subheader("Label")
        st.write(f"Current label: **{current_label or '(unlabeled)'}**")

        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            if st.button("✅ TRUE (real issue)", use_container_width=True):
                upsert_label(labels_path, current, "true")
                st.rerun()
        with b2:
            if st.button("❌ FALSE (false positive)", use_container_width=True):
                upsert_label(labels_path, current, "false")
                st.rerun()
        with b3:
            if st.button("🧹 Clear label", use_container_width=True):
                upsert_label(labels_path, current, "")
                st.rerun()

        st.divider()
        nav1, nav2, nav3 = st.columns([1, 1, 1])
        with nav1:
            if st.button("⬅ Prev", use_container_width=True):
                st.session_state.idx = max(0, st.session_state.idx - 1)
                st.rerun()
        with nav2:
            st.write(f"Item **{st.session_state.idx + 1} / {len(filtered)}**")
        with nav3:
            if st.button("Next ➡", use_container_width=True):
                st.session_state.idx = min(len(filtered) - 1, st.session_state.idx + 1)
                st.rerun()

    st.caption(f"Labels saved to: {labels_path}")
    st.caption(f"Stable sample saved to: {out_dir / 'sample.jsonl'}")


if __name__ == "__main__":
    main()

