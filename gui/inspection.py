from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from .script_metadata import StudyMetadata
except ImportError:
    from script_metadata import StudyMetadata

FACTOR_SUMMARY_FIELDS = [
    "PresentedAttribution",
    "LabelAccuracy",
    "BorderCondition",
    "StudyID",
]


def _resolve_col(columns: List[str], col: str, aliases: Dict[str, List[str]]) -> str | None:
    if col in columns:
        return col
    for alt in aliases.get(col, []):
        if alt in columns:
            return alt
    return None


def inspect_csv(file_path: Path, metadata: StudyMetadata, dataset_mode: str | None = None) -> str:
    if not file_path.exists():
        return f"Selected file does not exist: {file_path}"

    header = pd.read_csv(file_path, nrows=0)
    columns = list(header.columns)

    required = metadata.required_columns.copy()
    if metadata.study_key == "study4" and dataset_mode == "Study1_Study2":
        if "BorderCondition" not in required:
            required.append("BorderCondition")

    aliases = metadata.composite_aliases

    req_present, req_missing = [], []
    for col in required:
        resolved = _resolve_col(columns, col, aliases)
        (req_present if resolved else req_missing).append(f"{col}{'' if resolved is None else f' (as {resolved})'}")

    opt_present, opt_missing = [], []
    for col in metadata.optional_columns:
        resolved = _resolve_col(columns, col, aliases)
        (opt_present if resolved else opt_missing).append(f"{col}{'' if resolved is None else f' (as {resolved})'}")

    composite_lines = []
    for comp, comp_cols in metadata.composites.items():
        parts = []
        for col in comp_cols:
            resolved = _resolve_col(columns, col, aliases)
            parts.append(f"{col}:{'present' if resolved else 'missing'}")
        composite_lines.append(f"- {comp}: " + ", ".join(parts))

    sample = pd.read_csv(file_path, nrows=5000)
    factor_lines = []
    for fld in FACTOR_SUMMARY_FIELDS:
        resolved = _resolve_col(columns, fld, aliases)
        if resolved and resolved in sample.columns:
            vals = sorted(sample[resolved].dropna().astype(str).unique().tolist())
            preview = vals[:12]
            suffix = " ..." if len(vals) > 12 else ""
            factor_lines.append(f"- {fld}: {preview}{suffix}")

    model_lines = "\n".join([f"- {m}" for m in metadata.model_descriptions])
    return (
        f"Script: {metadata.script_path}\n"
        f"Dataset mode: {dataset_mode or 'N/A'}\n"
        f"Selected file: {file_path}\n"
        f"Detected columns ({len(columns)}): {', '.join(columns)}\n\n"
        f"Configured model family / analyses:\n{model_lines}\n\n"
        f"Required columns present ({len(req_present)}):\n- " + "\n- ".join(req_present) + "\n\n"
        f"Required columns missing ({len(req_missing)}):\n" + ("- " + "\n- ".join(req_missing) if req_missing else "- None") + "\n\n"
        f"Optional columns present ({len(opt_present)}):\n" + ("- " + "\n- ".join(opt_present) if opt_present else "- None") + "\n\n"
        f"Optional columns missing ({len(opt_missing)}):\n" + ("- " + "\n- ".join(opt_missing) if opt_missing else "- None") + "\n\n"
        f"Outcomes configured: {', '.join(metadata.outcomes)}\n"
        f"Composite component availability:\n" + "\n".join(composite_lines) + "\n\n"
        f"Factor-like value summaries (sample up to 5000 rows):\n" + ("\n".join(factor_lines) if factor_lines else "- No factor fields found")
    )
