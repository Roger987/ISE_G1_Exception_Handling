#!/usr/bin/env python3
"""
Works with BOTH output layouts:
A) Per-project folders: analysis_csv/<folder>/*.csv
B) Flat legacy layout: analysis_csv/<project_tag>_results-<query>.csv

Calibration (OPTIONAL, per-query precision only):
- expected_true = findings * precision
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import stats

# REQUIRED for Dunn
try:
    import scikit_posthocs as sp  # type: ignore
except Exception:
    sp = None


# =============================================================================
# OPTIONAL INLINE PER-QUERY PRECISION
# =============================================================================
INLINE_QUERY_PRECISION: Dict[str, float] = {
    "find_abort_except": 1.0,
    "find_logging_except": 0.8,
    "find_pass_only_except": 0.85,
    "find_todo_except": 1.0,
}

# =============================================================================
# Parsing config
# =============================================================================
PREFIX = "vibe_dataset_apps_"
RESULTS_MARKERS = ["_results-", "_results_"]

# =============================================================================
# Palette
# =============================================================================
HEATMAP_CMAP = "plasma"


def _make_discrete(n: int = 8, name: str = "viridis") -> List:
    cmap = mpl.colormaps.get_cmap(name)
    if n <= 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]


PALETTE_BARS = _make_discrete(9, HEATMAP_CMAP)


def _color_cycle():
    return cycle(PALETTE_BARS)


@dataclass(frozen=True)
class ResultFile:
    path: Path
    project: str
    query: str
    llm_full: str
    llm_family: str


# =============================================================================
# Discovery + parsing
# =============================================================================
def _iter_csv_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.csv"):
        parts = set(p.parts)
        if ".locks" in parts:
            continue
        if "_tmp" in parts:
            continue
        yield p


def _extract_between_prefix_and_results(stem: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not stem.startswith(PREFIX):
        return None, None, None

    marker_pos = None
    marker_used = None
    for m in RESULTS_MARKERS:
        pos = stem.find(m, len(PREFIX))
        if pos != -1 and (marker_pos is None or pos < marker_pos):
            marker_pos = pos
            marker_used = m
    if marker_pos is None:
        return None, None, None

    mid = stem[len(PREFIX) : marker_pos]
    query = stem[marker_pos + len(marker_used) :]
    if not mid or not query:
        return None, None, None
    return mid, marker_used, query


def _split_llm_full_to_model_and_project(mid: str) -> Tuple[str, str]:
    if "_" in mid:
        model_id, project = mid.split("_", 1)
        project = project.strip() or "(empty project suffix)"
        return model_id, project
    return mid, "(no project suffix)"


def _llm_family_from_model_id(model_id: str) -> str:
    if not model_id:
        return "unknown"
    if "-" in model_id:
        fam = model_id.split("-", 1)[0]
        return fam if fam else "unknown"
    return model_id


def _infer_project_query_llms(csv_path: Path, analysis_dir: Path) -> Tuple[str, str, str, str]:
    rel = csv_path.relative_to(analysis_dir)

    # Layout A: folder structure
    if len(rel.parts) >= 2:
        folder = rel.parts[0]
        query = rel.stem

        llm_full = "unknown_llm"
        llm_family = "unknown"
        project = folder

        if folder.startswith(PREFIX):
            mid = folder[len(PREFIX) :]
            llm_full = mid
            model_id, project_part = _split_llm_full_to_model_and_project(mid)
            llm_family = _llm_family_from_model_id(model_id)
            project = project_part

        return project, query, llm_full, llm_family

    # Layout B: flat filenames
    stem = rel.stem
    mid, _, query = _extract_between_prefix_and_results(stem)
    if mid is not None and query is not None:
        model_id, project = _split_llm_full_to_model_and_project(mid)
        llm_family = _llm_family_from_model_id(model_id)
        return project, query, mid, llm_family

    # Fallback legacy parsing
    m = re.match(r"^(?P<p>.+?)_results-(?P<q>.+)$", stem)
    if m:
        return m.group("p"), m.group("q"), "unknown_llm", "unknown"
    m = re.match(r"^(?P<p>.+?)_results_(?P<q>.+)$", stem)
    if m:
        return m.group("p"), m.group("q"), "unknown_llm", "unknown"
    if "_" in stem:
        p, q = stem.rsplit("_", 1)
        return p, q, "unknown_llm", "unknown"

    return "unknown_project", stem, "unknown_llm", "unknown"


def discover_results(analysis_dir: Path) -> List[ResultFile]:
    out: List[ResultFile] = []
    for p in _iter_csv_files(analysis_dir):
        project, query, llm_full, llm_family = _infer_project_query_llms(p, analysis_dir)
        out.append(ResultFile(path=p, project=project, query=query, llm_full=llm_full, llm_family=llm_family))
    return sorted(out, key=lambda r: (r.llm_family, r.llm_full, r.project, r.query, str(r.path)))


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=True, na_values=["", "NA", "NaN"])
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1", keep_default_na=True, na_values=["", "NA", "NaN"])
    except pd.errors.ParserError:
        return pd.read_csv(path, dtype=str, keep_default_na=True, na_values=["", "NA", "NaN"], on_bad_lines="skip")


# =============================================================================
# Column profiling (schema-agnostic)
# =============================================================================
def profile_columns(
    df: pd.DataFrame,
    llm_family: str,
    llm_full: str,
    project: str,
    query: str,
    max_top_values: int,
) -> pd.DataFrame:
    rows = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        nonmissing = n - missing

        rows.append(
            dict(
                llm_family=llm_family,
                llm_full=llm_full,
                project=project,
                query=query,
                column=col,
                metric="missing_count",
                value=str(missing),
            )
        )
        rows.append(
            dict(
                llm_family=llm_family,
                llm_full=llm_full,
                project=project,
                query=query,
                column=col,
                metric="missing_frac",
                value=f"{(missing / n) if n else 0:.6f}",
            )
        )

        sn = pd.to_numeric(s, errors="coerce")
        numeric_nonmissing = int(sn.notna().sum())
        if numeric_nonmissing > 0 and numeric_nonmissing >= max(5, int(0.2 * nonmissing)):
            rows.append(dict(llm_family=llm_family, llm_full=llm_full, project=project, query=query, column=col, metric="numeric_count", value=str(numeric_nonmissing)))
            rows.append(dict(llm_family=llm_family, llm_full=llm_full, project=project, query=query, column=col, metric="numeric_mean", value=str(sn.mean(skipna=True))))
            rows.append(dict(llm_family=llm_family, llm_full=llm_full, project=project, query=query, column=col, metric="numeric_std", value=str(sn.std(skipna=True))))
            rows.append(dict(llm_family=llm_family, llm_full=llm_full, project=project, query=query, column=col, metric="numeric_min", value=str(sn.min(skipna=True))))
            rows.append(dict(llm_family=llm_family, llm_full=llm_full, project=project, query=query, column=col, metric="numeric_max", value=str(sn.max(skipna=True))))
        else:
            vc = s.dropna().astype(str).value_counts()
            for i, (val, cnt) in enumerate(vc.head(max_top_values).items(), start=1):
                rows.append(dict(llm_family=llm_family, llm_full=llm_full, project=project, query=query, column=col, metric=f"top_value_{i}", value=f"{val} ({cnt})"))
            rows.append(dict(llm_family=llm_family, llm_full=llm_full, project=project, query=query, column=col, metric="unique_count", value=str(int(vc.shape[0]))))

    return pd.DataFrame(rows)


# =============================================================================
# Plot helpers
# =============================================================================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_bar(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path, rotate_xticks: bool = True) -> None:
    if df.empty:
        return

    ys_num = pd.to_numeric(df[y], errors="coerce")
    mask = ys_num.notna() & np.isfinite(ys_num.to_numpy())
    if mask.sum() == 0:
        return

    xs = df.loc[mask, x].astype(str).tolist()
    ys = ys_num.loc[mask].astype(float).tolist()

    plt.figure()
    c = _color_cycle()
    colors = [next(c) for _ in xs]
    plt.bar(xs, ys, color=colors)
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    if rotate_xticks:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _save_boxplot_by_group(df: pd.DataFrame, group_col: str, value_col: str, title: str, out_path: Path) -> None:
    tmp = df[[group_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        return

    groups, labels = [], []
    for g, sub in tmp.groupby(group_col):
        v = sub[value_col].astype(float).values
        if len(v) == 0:
            continue
        groups.append(v)
        labels.append(str(g))
    if not groups:
        return

    plt.figure()
    bp = plt.boxplot(groups, tick_labels=labels, patch_artist=True, showfliers=True)
    c = _color_cycle()
    for box in bp["boxes"]:
        box.set_facecolor(next(c))
        box.set_alpha(0.65)
    plt.title(title)
    plt.xlabel(group_col.replace("_", " ").title())
    plt.ylabel(value_col.replace("_", " ").title())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _save_heatmap(df: pd.DataFrame, index: str, columns: str, values: str, title: str, out_path: Path) -> None:
    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc="sum", fill_value=0)
    if pivot.empty:
        return
    mat = pivot.values.astype(float)
    plt.figure(figsize=(max(6, 0.6 * pivot.shape[1]), max(4, 0.4 * pivot.shape[0])))
    plt.imshow(mat, aspect="auto", cmap=HEATMAP_CMAP)
    plt.title(title)
    plt.xlabel(columns.replace("_", " ").title())
    plt.ylabel(index.replace("_", " ").title())
    plt.xticks(range(pivot.shape[1]), [str(c) for c in pivot.columns], rotation=45, ha="right")
    plt.yticks(range(pivot.shape[0]), [str(r) for r in pivot.index])
    plt.colorbar(label=values.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _save_stacked_bar_true_false(df: pd.DataFrame, x: str, true_col: str, false_col: str, title: str, out_path: Path) -> None:
    if df.empty:
        return

    true_vals = pd.to_numeric(df[true_col], errors="coerce")
    false_vals = pd.to_numeric(df[false_col], errors="coerce")
    mask = true_vals.notna() & false_vals.notna() & np.isfinite(true_vals.to_numpy()) & np.isfinite(false_vals.to_numpy())
    if mask.sum() == 0:
        return

    xs = df.loc[mask, x].astype(str).tolist()
    true_vals = true_vals.loc[mask].astype(float).tolist()
    false_vals = false_vals.loc[mask].astype(float).tolist()

    plt.figure()
    cmap = mpl.colormaps.get_cmap("viridis")
    col_true = cmap(0.25)
    col_false = cmap(0.85)
    plt.bar(xs, true_vals, label=true_col.replace("_", " ").title(), color=col_true)
    plt.bar(xs, false_vals, bottom=true_vals, label=false_col.replace("_", " ").title(), color=col_false)
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel("Expected Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# =============================================================================
# Table helpers (readable headers + split into multiple tables)
# =============================================================================
_HEADER_RENAMES = {
    "llm_family": "LLM Family",
    "files_total": "Files (Total)",
    "files_with_findings": "Files w/ Findings",
    "pct_files_with_findings": "% Files w/ Findings",
    "total_findings": "Findings (Total)",
    "share_of_all_findings_pct": "% of All Findings",
    "mean_findings_per_file": "Mean",
    "raw_findings": "Findings (Raw)",
    "expected_true": "Expected True",
    "expected_false": "Expected False",
    "mean_precision": "Mean Precision",
    "mean_fdr": "Mean FDR",
    "expected_false_rate_pct": "% Expected False",
}


def _prettify_colname(c: str) -> str:
    if c in _HEADER_RENAMES:
        return _HEADER_RENAMES[c]
    return c.replace("_", " ").strip().title()


def _prettify_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_prettify_colname(str(c)) for c in out.columns]
    return out


def _save_table_png_split(
    df: pd.DataFrame,
    title: str,
    out_prefix: Path,
    *,
    max_rows: int = 25,
    font_size: int = 12,
    col_chunks: int = 6,
    first_col_sticky: bool = True,
) -> None:
    if df is None or df.empty:
        return

    tdf = _prettify_headers(df.head(max_rows).copy())
    cols = list(tdf.columns)
    if not cols:
        return

    sticky = [cols[0]] if first_col_sticky else []
    rest = cols[1:] if first_col_sticky else cols[:]

    chunks: List[List[str]] = []
    for i in range(0, len(rest), col_chunks):
        chunk = rest[i : i + col_chunks]
        if sticky:
            chunk = sticky + chunk
        chunks.append(chunk)

    for idx, chunk_cols in enumerate(chunks, start=1):
        sub = tdf[chunk_cols].copy().astype(str)
        nrows, ncols = sub.shape

        fig_w = max(10.0, 1.7 * ncols)
        fig_h = max(2.5, 0.70 + 0.55 * nrows)

        plt.figure(figsize=(fig_w, fig_h))
        plt.axis("off")
        plt.title(f"{title} (Part {idx}/{len(chunks)})", pad=12)

        table = plt.table(
            cellText=sub.values,
            colLabels=sub.columns.tolist(),
            cellLoc="left",
            colLoc="left",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, 1.35)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_alpha(0.96)

        plt.tight_layout()
        out_path = Path(str(out_prefix) + f"_part{idx}.png")
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()


# =============================================================================
# Precision calibration (per-query only)
# =============================================================================
def _normalize_precision(p: float) -> float:
    if p > 1.0:
        p = p / 100.0
    if not (0.0 <= p <= 1.0):
        raise SystemExit(f"Precision must be in [0,1] or [0,100], got {p}")
    return float(p)


def load_precision_calibration_from_inline() -> pd.DataFrame:
    if not INLINE_QUERY_PRECISION:
        return pd.DataFrame(columns=["query", "precision", "fdr"])
    rows = []
    for q, p in INLINE_QUERY_PRECISION.items():
        p2 = _normalize_precision(float(p))
        rows.append({"query": str(q), "precision": p2, "fdr": 1.0 - p2})
    return pd.DataFrame(rows)


def load_precision_calibration_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "query" not in df.columns or "precision" not in df.columns:
        raise SystemExit("Calibration CSV must contain columns: query,precision")
    df = df[["query", "precision"]].copy()
    df["precision"] = df["precision"].apply(lambda x: _normalize_precision(float(x)))
    df["fdr"] = 1.0 - df["precision"]
    df["query"] = df["query"].astype(str)
    return df.drop_duplicates(subset=["query"], keep="first")


def attach_precision_calibration(per_file: pd.DataFrame, calib: pd.DataFrame, default_precision: Optional[float]) -> pd.DataFrame:
    df = per_file.copy()
    df["row_count_num"] = pd.to_numeric(df["row_count"], errors="coerce").fillna(0).astype(float)
    df = df.merge(calib[["query", "precision", "fdr"]], on="query", how="left")

    if default_precision is not None:
        dp = _normalize_precision(default_precision)
        df["precision_used"] = df["precision"].fillna(dp)
        df["fdr_used"] = df["fdr"].fillna(1.0 - dp)
        df["calib_level"] = df["precision"].notna().map(lambda x: "query" if x else "default")
    else:
        df["precision_used"] = df["precision"]
        df["fdr_used"] = df["fdr"]
        df["calib_level"] = df["precision"].notna().map(lambda x: "query" if x else "none")

    df["expected_true"] = df["row_count_num"] * df["precision_used"]
    df["expected_false"] = df["row_count_num"] * (1.0 - df["precision_used"])
    return df


# =============================================================================
# Assumption checks for Kruskal–Wallis / Dunn
# =============================================================================
def _safe_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return np.nan
    try:
        return float(stats.skew(x, bias=False, nan_policy="omit"))
    except Exception:
        return np.nan


def _shape_asymmetry_ratio(x: np.ndarray) -> float:
    """(Q3 - median) / (median - Q1); ~1 indicates symmetric around median."""
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return np.nan
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    left = med - q1
    right = q3 - med
    if left <= 0:
        return np.inf if right > 0 else np.nan
    return float(right / left)


def _save_ecdf_plot(groups: Dict[str, np.ndarray], title: str, out_path: Path) -> None:
    plt.figure()

    color_iter = _color_cycle()  # <-- use your palette

    for g, x in groups.items():
        x = np.sort(np.asarray(x, dtype=float))
        if x.size == 0:
            continue
        y = np.arange(1, x.size + 1) / x.size
        plt.step(x, y, where="post", label=str(g), color=next(color_iter))

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("ECDF")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()



def _save_hist_plot(groups: Dict[str, np.ndarray], title: str, out_path: Path) -> None:
    all_vals = np.concatenate(
        [np.asarray(v, dtype=float) for v in groups.values() if len(v) > 0],
        axis=0,
    )
    if all_vals.size == 0:
        return

    vmin = int(np.nanmin(all_vals))
    vmax = int(np.nanmax(all_vals))
    bins = np.arange(vmin, vmax + 2) - 0.5  # integer-centered bins

    plt.figure()
    color_iter = _color_cycle()  # <-- use your palette

    for g, x in groups.items():
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            continue
        plt.hist(
            x,
            bins=bins,
            alpha=0.6,
            density=True,
            label=str(g),
            color=next(color_iter),
        )

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def check_kw_dunn_assumptions(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    obs_id_col: str,
    out_dir: Path,
    charts_dir: Path,
    prefix: str,
    min_n: int = 5,
    fail_on_small_n: bool = True,
    shape_skew_tol: float = 1.0,
    shape_asym_tol: float = 1.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Practical assumption checks/diagnostics for Kruskal–Wallis + Dunn.

    What it DOES:
      - Ordinal/continuous: verifies numeric coercion works; reports % failures.
      - Independence: flags duplicate obs_id_col within group (data-level sanity).
      - Distributional shape: compares skewness and robust median-asymmetry ratios.
      - Sample size: checks n>=min_n per group (fail or warn).

    What it CANNOT prove:
      - True causal independence (requires study design); this is a *sanity check* only.
    """
    warnings: List[str] = []
    df0 = df.copy()

    if group_col not in df0.columns:
        raise SystemExit(f"Assumption check: missing group_col={group_col}")
    if value_col not in df0.columns:
        raise SystemExit(f"Assumption check: missing value_col={value_col}")
    if obs_id_col not in df0.columns:
        raise SystemExit(f"Assumption check: missing obs_id_col={obs_id_col}")

    # Ordinal/continuous check: numeric coercion success
    v = pd.to_numeric(df0[value_col], errors="coerce")
    non_missing = int(df0[value_col].notna().sum())
    numeric_ok = int(v.notna().sum())
    numeric_fail = non_missing - numeric_ok
    frac_fail = (numeric_fail / non_missing) if non_missing else 0.0
    if frac_fail > 0:
        warnings.append(
            f"[ordinal/continuous] {numeric_fail}/{non_missing} ({100*frac_fail:.2f}%) values in '{value_col}' "
            f"were non-numeric after coercion and will be dropped for KW/Dunn."
        )

    df1 = df0.copy()
    df1["_value_num"] = v
    df1 = df1.dropna(subset=["_value_num"])

    # Group-wise diagnostics
    rows = []
    groups: Dict[str, np.ndarray] = {}
    for g, sub in df1.groupby(group_col):
        gname = str(g)
        x = sub["_value_num"].astype(float).to_numpy()
        n = int(x.size)
        groups[gname] = x

        # independence sanity: duplicate obs IDs within group
        dup_count = int(sub.duplicated(subset=[obs_id_col]).sum())
        dup_frac = (dup_count / n) if n else 0.0

        rows.append(
            dict(
                group=gname,
                n=n,
                mean=float(np.mean(x)) if n else np.nan,
                median=float(np.median(x)) if n else np.nan,
                q1=float(np.percentile(x, 25)) if n else np.nan,
                q3=float(np.percentile(x, 75)) if n else np.nan,
                skew=_safe_skew(x),
                asym_ratio=_shape_asymmetry_ratio(x),
                zero_frac=float(np.mean(x == 0.0)) if n else np.nan,
                obs_duplicates=dup_count,
                obs_duplicates_frac=dup_frac,
            )
        )

        if dup_count > 0:
            warnings.append(
                f"[independence] group '{gname}' has {dup_count} duplicate '{obs_id_col}' entries "
                f"({100*dup_frac:.2f}% of n={n}). This suggests non-independent / duplicated observations."
            )

    diag = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    diag.to_csv(out_dir / f"assumptions_{prefix}_group_diagnostics.csv", index=False)

    # Sample size check
    small = diag[diag["n"] < min_n]
    if not small.empty:
        msg = f"[sample size] {len(small)} groups have n < {min_n}: " + ", ".join([f"{r.group}(n={int(r.n)})" for r in small.itertuples()])
        if fail_on_small_n:
            raise SystemExit(msg)
        warnings.append(msg)

    # Distributional shape check (heuristic, but useful)
    # Compare skewness range and asymmetry ratio range across groups.
    # If the range is large, warn that KW may reflect shape differences too.
    skew_vals = diag["skew"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    asym_vals = diag["asym_ratio"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)

    if skew_vals.size >= 2:
        skew_range = float(np.nanmax(skew_vals) - np.nanmin(skew_vals))
        if skew_range > shape_skew_tol:
            warnings.append(
                f"[distributions] skewness differs substantially across groups "
                f"(range={skew_range:.2f} > tol={shape_skew_tol:.2f}). "
                f"Interpret KW as detecting distributional differences (not purely medians)."
            )

    if asym_vals.size >= 2:
        # asym ratio around 1; so compare max/min
        asym_ratio_spread = float(np.nanmax(asym_vals) / max(np.nanmin(asym_vals), 1e-12))
        if asym_ratio_spread > (1.0 + shape_asym_tol):
            warnings.append(
                f"[distributions] median-asymmetry ratios differ across groups "
                f"(max/min={asym_ratio_spread:.2f} > 1+tol={1.0+shape_asym_tol:.2f}). "
                f"Interpret KW cautiously as shapes may differ."
            )

    # Save diagnostic plots
    if groups:
        _save_ecdf_plot(groups, f"ECDF by {group_col} — {value_col}", charts_dir / f"assumptions_{prefix}_ecdf.png")
        _save_hist_plot(groups, f"Histogram by {group_col} — {value_col}", charts_dir / f"assumptions_{prefix}_hist.png")

    # Write a short assumptions report
    lines = []
    lines.append(f"# Assumption check: {prefix}\n")
    lines.append(f"- group_col: `{group_col}`\n")
    lines.append(f"- value_col: `{value_col}`\n")
    lines.append(f"- obs_id_col (independence sanity): `{obs_id_col}`\n")
    lines.append(f"- min_n: {min_n} (fail_on_small_n={fail_on_small_n})\n")
    lines.append("\n## Ordinal / continuous\n")
    lines.append(f"- Non-missing values: {non_missing}\n")
    lines.append(f"- Numeric after coercion: {numeric_ok}\n")
    lines.append(f"- Dropped as non-numeric: {numeric_fail} ({100*frac_fail:.2f}%)\n")

    lines.append("\n## Independence (sanity)\n")
    lines.append("- This script can only detect duplicated observation IDs within groups; it cannot prove study-design independence.\n")
    if "obs_duplicates" in diag.columns and not diag.empty:
        total_dups = int(diag["obs_duplicates"].sum())
        lines.append(f"- Total duplicated '{obs_id_col}' within-group entries: {total_dups}\n")

    lines.append("\n## Distributional shape (heuristics)\n")
    lines.append("- Computed per-group skewness and median-asymmetry ratio (Q3-median)/(median-Q1).\n")
    lines.append("- Large between-group differences suggest KW may reflect shape differences, not only median shifts.\n")

    lines.append("\n## Sample size\n")
    lines.append(f"- Groups checked: {len(diag)}\n")
    if not small.empty:
        lines.append(f"- Groups with n < {min_n}: {len(small)}\n")
        lines.append("- " + ", ".join([f"{r.group}(n={int(r.n)})" for r in small.itertuples()]) + "\n")
    else:
        lines.append(f"- All groups have n >= {min_n}\n")

    lines.append("\n## Warnings\n")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}\n")
    else:
        lines.append("- (none)\n")

    (out_dir / f"assumptions_{prefix}.md").write_text("".join(lines), encoding="utf-8")

    return diag, warnings


# =============================================================================
# Statistics: Kruskal–Wallis + Dunn (required)
# =============================================================================
def run_kw_dunn_and_rank(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    obs_id_col: str,
    out_dir: Path,
    charts_dir: Path,
    prefix: str,
    min_n: int = 5,
    fail_on_small_n: bool = True,
) -> None:
    if sp is None:
        raise SystemExit(
            "Dunn's test requested, but scikit-posthocs is not installed. "
            "Install it: pip install scikit-posthocs"
        )

    # Assumption checks (writes assumptions_*.md and diagnostics CSV + plots)
    tmp0 = df[[group_col, value_col, obs_id_col]].copy()
    tmp0[value_col] = pd.to_numeric(tmp0[value_col], errors="coerce")
    tmp0 = tmp0.dropna(subset=[value_col])

    check_kw_dunn_assumptions(
        tmp0,
        group_col=group_col,
        value_col=value_col,
        obs_id_col=obs_id_col,
        out_dir=out_dir,
        charts_dir=charts_dir,
        prefix=prefix,
        min_n=min_n,
        fail_on_small_n=fail_on_small_n,
    )

    if tmp0.empty:
        raise SystemExit(f"No numeric data for stats: {value_col}")

    # Prepare group arrays for Kruskal–Wallis
    grp_arrays = []
    labels = []
    for g, sub in tmp0.groupby(group_col):
        v = sub[value_col].astype(float).to_numpy()
        if v.size == 0:
            continue
        labels.append(str(g))
        grp_arrays.append(v)

    if len(grp_arrays) < 2:
        raise SystemExit(f"Need at least 2 groups for KW: got {len(grp_arrays)}")

    # Global KW
    h_stat, p_value = stats.kruskal(*grp_arrays, nan_policy="omit")

    n = int(len(tmp0))
    k = int(len(grp_arrays))
    eps2 = (float(h_stat) - (k - 1)) / (n - 1) if n > 1 else np.nan

    kw_summary = pd.DataFrame(
        [{
            "group_col": group_col,
            "value_col": value_col,
            "n_total": n,
            "k_groups": k,
            "h_stat": float(h_stat),
            "p_value": float(p_value),
            "epsilon_squared": float(eps2),
        }]
    )
    kw_summary.to_csv(out_dir / f"{prefix}_kw_summary.csv", index=False)

    # Mean-rank ranking
    tmp0["_rank"] = stats.rankdata(tmp0[value_col].astype(float).values, method="average")
    rank_table = (
        tmp0.groupby(group_col, as_index=False)
        .agg(
            n=(value_col, "count"),
            mean=(value_col, "mean"),
            median=(value_col, "median"),
            mean_rank=("_rank", "mean"),
        )
        .sort_values(["mean_rank", "mean"], ascending=[True, True])
        .reset_index(drop=True)
    )
    rank_table["rank_position"] = np.arange(1, len(rank_table) + 1)
    rank_table.to_csv(out_dir / f"{prefix}_mean_rank_ranking.csv", index=False)

    # Dunn post-hoc with BH/FDR adjustment
    # Dunn post-hoc with BH/FDR adjustment
    dmat = sp.posthoc_dunn(tmp0, val_col=value_col, group_col=group_col, p_adjust="fdr_bh")

    # Long format robustly (reset_index column name is "index" if index.name is None)
    idx_col = dmat.index.name if dmat.index.name is not None else "index"

    dlong = (
        dmat.reset_index()
        .melt(id_vars=idx_col, var_name="group_b", value_name="p_adj_bh")
        .rename(columns={idx_col: "group_a"})
    )

    # Keep upper triangle only + drop diagonal
    dlong = dlong[dlong["group_a"] != dlong["group_b"]].copy()
    dlong["pair_key"] = dlong.apply(lambda r: "||".join(sorted([str(r["group_a"]), str(r["group_b"])])), axis=1)
    dlong = dlong.drop_duplicates(subset=["pair_key"], keep="first").drop(columns=["pair_key"])
    dlong = dlong.sort_values(["p_adj_bh", "group_a", "group_b"], ascending=[True, True, True])

    dlong.to_csv(out_dir / f"{prefix}_dunn_pairwise_fdr_bh.csv", index=False)

    # Optional plot: mean ranks
    _save_bar(
        rank_table.sort_values("mean_rank", ascending=True),
        x=group_col,
        y="mean_rank",
        title=f"Mean Rank by {group_col} (Lower = Better) — {value_col}",
        out_path=charts_dir / f"{prefix}_mean_rank_bar.png",
    )


# =============================================================================
# Main analysis
# =============================================================================
def analyze(
    analysis_dir: Path,
    out_dir: Path,
    max_profile_files: int,
    max_top_values: int,
    top_heatmap_queries: int,
    calibration_csv: str,
    use_inline_calibration: bool,
    default_precision: Optional[float],
    stats_group: str,
    stats_level: str,
    stats_min_n: int,
    stats_fail_small_n: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"
    _ensure_dir(charts_dir)

    results = discover_results(analysis_dir)
    if not results:
        raise SystemExit(f"No CSV results found under: {analysis_dir}")

    # Index of files
    files_index = pd.DataFrame(
        [
            dict(
                llm_family=r.llm_family,
                llm_full=r.llm_full,
                project=r.project,
                query=r.query,
                path=str(r.path),
                bytes=r.path.stat().st_size,
            )
            for r in results
        ]
    )
    files_index.to_csv(out_dir / "files_index.csv", index=False)

    # Per-file summaries
    per_file_rows = []
    for idx, rf in enumerate(results, start=1):
        df = safe_read_csv(rf.path)
        per_file_rows.append(
            dict(
                llm_family=rf.llm_family,
                llm_full=rf.llm_full,
                project=rf.project,
                query=rf.query,
                path=str(rf.path),
                row_count=int(len(df)),
                column_count=int(df.shape[1]),
                bytes=int(rf.path.stat().st_size),
            )
        )
        if idx % 50 == 0:
            print(f"[progress] scanned {idx}/{len(results)} files...")

    per_file = pd.DataFrame(per_file_rows)
    per_file.to_csv(out_dir / "per_file_summary.csv", index=False)

    # Column profiling for top K
    prof_candidates = per_file.sort_values(["row_count", "bytes"], ascending=False).head(max_profile_files).reset_index(drop=True)
    profile_frames = []
    for _, row in prof_candidates.iterrows():
        p = Path(row["path"])
        df = safe_read_csv(p)
        profile_frames.append(
            profile_columns(
                df,
                llm_family=str(row["llm_family"]),
                llm_full=str(row["llm_full"]),
                project=str(row["project"]),
                query=str(row["query"]),
                max_top_values=max_top_values,
            )
        )
    col_profile = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame(
        columns=["llm_family", "llm_full", "project", "query", "column", "metric", "value"]
    )
    col_profile.to_csv(out_dir / "column_profile_top_values.csv", index=False)

    # Raw aggregations
    per_project = (
        per_file.groupby("project", as_index=False)
        .agg(files=("path", "count"), total_findings=("row_count", "sum"), mean_findings=("row_count", "mean"), median_findings=("row_count", "median"), total_bytes=("bytes", "sum"))
        .sort_values(["total_findings", "files"], ascending=False)
    )
    per_project.to_csv(out_dir / "per_project_summary.csv", index=False)

    per_query = (
        per_file.groupby("query", as_index=False)
        .agg(files=("path", "count"), total_findings=("row_count", "sum"), mean_findings=("row_count", "mean"), median_findings=("row_count", "median"), total_bytes=("bytes", "sum"))
        .sort_values(["total_findings", "files"], ascending=False)
    )
    per_query.to_csv(out_dir / "per_query_summary.csv", index=False)

    per_family = (
        per_file.groupby("llm_family", as_index=False)
        .agg(files=("path", "count"), total_findings=("row_count", "sum"), mean_findings=("row_count", "mean"), median_findings=("row_count", "median"), total_bytes=("bytes", "sum"))
        .sort_values(["total_findings", "files"], ascending=False)
    )
    per_family.to_csv(out_dir / "per_llm_family_summary.csv", index=False)

    per_family_query = (
        per_file.groupby(["llm_family", "query"], as_index=False)
        .agg(files=("path", "count"), total_findings=("row_count", "sum"), mean_findings=("row_count", "mean"), median_findings=("row_count", "median"), total_bytes=("bytes", "sum"))
        .sort_values(["llm_family", "total_findings", "files"], ascending=[True, False, False])
    )
    per_family_query.to_csv(out_dir / "per_llm_family_query_summary.csv", index=False)

    per_full = (
        per_file.groupby("llm_full", as_index=False)
        .agg(files=("path", "count"), total_findings=("row_count", "sum"), mean_findings=("row_count", "mean"), median_findings=("row_count", "median"), total_bytes=("bytes", "sum"))
        .sort_values(["total_findings", "files"], ascending=False)
    )
    per_full.to_csv(out_dir / "per_llm_full_summary.csv", index=False)

    # Raw charts
    _save_bar(per_family.head(30), "llm_family", "total_findings", "Total Findings (Rows) by LLM Family — Raw", charts_dir / "family_total_findings.png")
    _save_bar(per_family.head(30), "llm_family", "files", "Result Files by LLM Family — Raw", charts_dir / "family_files_count.png")
    _save_boxplot_by_group(per_file, "llm_family", "row_count", "Per-File Findings by LLM Family — Raw", charts_dir / "findings_per_file_boxplot_by_family.png")

    if not per_query.empty and not per_family_query.empty:
        top_queries = per_query.head(top_heatmap_queries)["query"].tolist()
        heat_df = per_family_query[per_family_query["query"].isin(top_queries)].copy()
        _save_heatmap(
            heat_df,
            index="llm_family",
            columns="query",
            values="total_findings",
            title=f"LLM Family × Query (Top {min(top_heatmap_queries, len(top_queries))}) — Raw Findings",
            out_path=charts_dir / "llm_query_heatmap_family_top.png",
        )

    _save_bar(per_full.head(25), "llm_full", "total_findings", "Total Findings by LLM Full (Top 25) — Raw", charts_dir / "full_total_findings_top.png")

    # KPI table (family)
    per_file_k = per_file.copy()
    per_file_k["row_count_num"] = pd.to_numeric(per_file_k["row_count"], errors="coerce").fillna(0).astype(float)
    per_file_k["has_findings"] = (per_file_k["row_count_num"] > 0).astype(int)

    llm_kpis = (
        per_file_k.groupby("llm_family", as_index=False)
        .agg(
            files_total=("path", "count"),
            files_with_findings=("has_findings", "sum"),
            total_findings=("row_count_num", "sum"),
            mean_findings_per_file=("row_count_num", "mean"),
        )
    )
    llm_kpis["pct_files_with_findings"] = 100.0 * llm_kpis["files_with_findings"] / llm_kpis["files_total"].replace(0, pd.NA)
    total_all_findings = float(per_file_k["row_count_num"].sum())
    llm_kpis["share_of_all_findings_pct"] = 100.0 * llm_kpis["total_findings"] / (total_all_findings if total_all_findings > 0 else pd.NA)
    llm_kpis = llm_kpis.sort_values(["pct_files_with_findings", "total_findings"], ascending=False)
    llm_kpis.to_csv(out_dir / "llm_kpis.csv", index=False)

    # display formatting for tables
    llm_kpis_disp = llm_kpis.copy()
    llm_kpis_disp["pct_files_with_findings"] = llm_kpis_disp["pct_files_with_findings"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    llm_kpis_disp["share_of_all_findings_pct"] = llm_kpis_disp["share_of_all_findings_pct"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    llm_kpis_disp["total_findings"] = llm_kpis_disp["total_findings"].map(lambda x: f"{int(round(x)):,}")
    llm_kpis_disp["mean_findings_per_file"] = llm_kpis_disp["mean_findings_per_file"].map(lambda x: f"{x:.2f}")

    _save_table_png_split(
        llm_kpis_disp,
        title="LLM Family KPIs (File Has Findings := Row Count > 0)",
        out_prefix=charts_dir / "llm_kpis_table",
        max_rows=50,
        font_size=12,
        col_chunks=5,
        first_col_sticky=True,
    )

    # -----------------------
    # Stats dataset (choose unit): file-level or project-level
    # -----------------------
    if stats_group not in ("llm_family", "llm_full"):
        raise SystemExit("--stats-group must be llm_family or llm_full")

    if stats_level not in ("file", "project"):
        raise SystemExit("--stats-level must be file or project")

    if stats_level == "file":
        stats_df = per_file_k.copy()
        stats_df["obs_id"] = stats_df["path"].astype(str)  # observation identity
        value_col = "row_count_num"
        prefix_raw = f"stats_raw_{stats_group}_file"
    else:
        # Aggregate to project as observational unit: sum findings across queries/files per project
        stats_df = (
            per_file_k.groupby([stats_group, "project"], as_index=False)
            .agg(
                row_count_num=("row_count_num", "sum"),
                n_files=("path", "count"),
            )
        )
        stats_df["obs_id"] = stats_df["project"].astype(str)
        value_col = "row_count_num"
        prefix_raw = f"stats_raw_{stats_group}_project"

    # Run KW + Dunn + assumptions + ranking (RAW)
    run_kw_dunn_and_rank(
        stats_df,
        group_col=stats_group,
        value_col=value_col,
        obs_id_col="obs_id",
        out_dir=out_dir,
        charts_dir=charts_dir,
        prefix=prefix_raw,
        min_n=stats_min_n,
        fail_on_small_n=stats_fail_small_n,
    )

    # -----------------------
    # Optional per-query precision calibration
    # -----------------------
    calib_df: Optional[pd.DataFrame] = None
    if calibration_csv:
        calib_df = load_precision_calibration_from_csv(Path(calibration_csv).expanduser().resolve())
    elif use_inline_calibration:
        calib_df = load_precision_calibration_from_inline()

    calibrated: Optional[pd.DataFrame] = None
    if calib_df is not None and not calib_df.empty:
        calibrated = attach_precision_calibration(per_file, calib_df, default_precision=default_precision)

        calibrated[
            [
                "llm_family",
                "llm_full",
                "project",
                "query",
                "path",
                "row_count",
                "precision_used",
                "fdr_used",
                "expected_true",
                "expected_false",
                "calib_level",
            ]
        ].to_csv(out_dir / "calibration_used.csv", index=False)

        # Stats on calibrated expected_true (same observational level as chosen)
        cal = calibrated.copy()
        cal["expected_true_num"] = pd.to_numeric(cal["expected_true"], errors="coerce")
        cal = cal.dropna(subset=["expected_true_num"])

        if stats_level == "file":
            stats_cal = cal.copy()
            stats_cal["obs_id"] = stats_cal["path"].astype(str)
            value_col_cal = "expected_true_num"
            prefix_cal = f"stats_calibrated_{stats_group}_file_expected_true"
        else:
            stats_cal = (
                cal.groupby([stats_group, "project"], as_index=False)
                .agg(
                    expected_true_num=("expected_true_num", "sum"),
                    n_files=("path", "count"),
                )
            )
            stats_cal["obs_id"] = stats_cal["project"].astype(str)
            value_col_cal = "expected_true_num"
            prefix_cal = f"stats_calibrated_{stats_group}_project_expected_true"

        if not stats_cal.empty:
            run_kw_dunn_and_rank(
                stats_cal,
                group_col=stats_group,
                value_col=value_col_cal,
                obs_id_col="obs_id",
                out_dir=out_dir,
                charts_dir=charts_dir,
                prefix=prefix_cal,
                min_n=stats_min_n,
                fail_on_small_n=stats_fail_small_n,
            )

        # (Keep your calibrated descriptive outputs if you want; omitted here to keep focus on stats)

    # -----------------------
    # Report
    # -----------------------
    total_files = int(len(per_file))
    total_findings = int(pd.to_numeric(per_file["row_count"], errors="coerce").fillna(0).sum())

    report = []
    report.append("# CodeQL Results Report\n")
    report.append(f"- Analysis dir: `{analysis_dir}`\n")
    report.append(f"- Files analyzed: **{total_files:,}**\n")
    report.append(f"- Total findings (rows across all CSVs): **{total_findings:,}**\n")
    report.append(f"- Detailed column profiles computed for top **{len(prof_candidates)}** files by findings/size.\n")

    report.append("\n## Parsing sanity\n")
    fam_counts = per_file["llm_family"].value_counts(dropna=False).to_frame("files").reset_index().rename(columns={"index": "llm_family"})
    report.append(fam_counts.to_markdown(index=False))

    report.append("\n\n## Statistical outputs (RAW)\n")
    report.append(f"- `{prefix_raw}_kw_summary.csv`\n")
    report.append(f"- `{prefix_raw}_mean_rank_ranking.csv`\n")
    report.append(f"- `{prefix_raw}_dunn_pairwise_fdr_bh.csv`\n")
    report.append(f"- `assumptions_{prefix_raw}.md`\n")
    report.append(f"- `assumptions_{prefix_raw}_group_diagnostics.csv`\n")
    report.append(f"- `charts/{prefix_raw}_mean_rank_bar.png`\n")
    report.append(f"- `charts/assumptions_{prefix_raw}_ecdf.png`\n")
    report.append(f"- `charts/assumptions_{prefix_raw}_hist.png`\n")

    if calibrated is not None:
        report.append("\n## Statistical outputs (CALIBRATED expected_true)\n")
        report.append(f"- `{prefix_cal}_kw_summary.csv`\n")
        report.append(f"- `{prefix_cal}_mean_rank_ranking.csv`\n")
        report.append(f"- `{prefix_cal}_dunn_pairwise_fdr_bh.csv`\n")
        report.append(f"- `assumptions_{prefix_cal}.md`\n")
        report.append(f"- `assumptions_{prefix_cal}_group_diagnostics.csv`\n")
        report.append(f"- `charts/{prefix_cal}_mean_rank_bar.png`\n")
        report.append(f"- `charts/assumptions_{prefix_cal}_ecdf.png`\n")
        report.append(f"- `charts/assumptions_{prefix_cal}_hist.png`\n")

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"OK: wrote outputs to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=str, default="analysis_csv")
    ap.add_argument("--out-dir", type=str, default="analysis_stats")
    ap.add_argument("--max-profile-files", type=int, default=30)
    ap.add_argument("--max-top-values", type=int, default=10)
    ap.add_argument("--top-heatmap-queries", type=int, default=12)

    ap.add_argument("--calibration", type=str, default="", help="CSV with columns: query,precision (precision in [0,1] or [0,100])")
    ap.add_argument("--use-inline-calibration", action="store_true", help="Use INLINE_QUERY_PRECISION from this file")
    ap.add_argument(
        "--default-precision",
        type=float,
        default=float("nan"),
        help="Fallback precision if query missing (0..1 or 0..100). If omitted/NaN, missing queries remain uncalibrated.",
    )

    ap.add_argument(
        "--stats-group",
        type=str,
        default="llm_family",
        help="Which grouping to use for KW/Dunn ranking: llm_family or llm_full",
    )
    ap.add_argument(
        "--stats-level",
        type=str,
        default="file",
        help="Observational unit for KW/Dunn: file or project",
    )
    ap.add_argument(
        "--stats-min-n",
        type=int,
        default=5,
        help="Minimum sample size per group (default 5).",
    )
    ap.add_argument(
        "--stats-fail-small-n",
        action="store_true",
        help="If set, abort when any group has n < stats-min-n. Otherwise, warn and continue.",
    )

    args = ap.parse_args()
    default_precision = None if (args.default_precision != args.default_precision) else args.default_precision  # NaN check

    analyze(
        analysis_dir=Path(args.analysis_dir).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        max_profile_files=args.max_profile_files,
        max_top_values=args.max_top_values,
        top_heatmap_queries=args.top_heatmap_queries,
        calibration_csv=args.calibration,
        use_inline_calibration=args.use_inline_calibration,
        default_precision=default_precision,
        stats_group=args.stats_group,
        stats_level=args.stats_level,
        stats_min_n=args.stats_min_n,
        stats_fail_small_n=args.stats_fail_small_n,
    )


if __name__ == "__main__":
    main()
