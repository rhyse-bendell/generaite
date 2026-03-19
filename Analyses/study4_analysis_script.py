"""
Gener-AI-te — Study 4 Cross-Study Analysis Script
(Studies 1+3 for attribution/accuracy; Studies 1+2 for border robustness)

Author: Dr. Rhyse Bendell + GPT-5.1 Thinking
Date: 2026-01-26

What this script assumes
------------------------
You have two cross-study CSVs in long format
(one row per ParticipantID × ArtworkID response):

  1) C:\Post-doc Work\Gener-ai-te\Data\Generaite_Study1_Study3_Combined_1-26-2026.csv
     - Contains responses from Studies 1 and 3.
     - Uses standardized schema:

       REQUIRED IDENTIFIERS
       - ParticipantID
       - ArtworkID
       - StudyID              : {1, 3}

       CORE ATTRIBUTION / LABELLING CONSTRUCTS
       - PresentedAttribution : {NoLabel, Human, AI}
       - ActualOrigin         : {Human, AI}
       - LabelAccuracy        : {NoLabel, Accurate, Deceptive}

       VISUAL / PRESENTATION
       - BorderCondition      : may exist but NOT used for H4.1–H4.3

       PARTICIPANT MODERATOR
       - AttitudesTowardAI    : numeric; centered in this script

       OUTCOMES (DVs)
       - Same as Study 1:
           * AestheticJudgment (composite)
           * NegativeEmotion (composite)
           * PositiveEmotion (composite)
           * IsArtSlider
           * LikeThisArtSlider

  2) C:\Post-doc Work\Gener-ai-te\Data\Generaite_Study1_Study2_Combined_1-21-2026.csv
     - Contains responses from Studies 1 and 2.
     - Same standardized schema as above, with StudyID ∈ {1, 2}, and
       valid BorderCondition values (e.g., Original, Swapped, Neutral)
       for labelled-trial rows.

Cross-Study Theoretical Focus
-----------------------------
- Studies 1 + 3:
  H4.1 — Directional Consistency Relative to No Label
    - Artwork labelled Human (accurately or deceptively) will be evaluated
      more positively than NoLabel.
    - Artwork labelled AI (accurately or deceptively) will be evaluated
      more negatively than NoLabel.

  H4.2 — Accuracy as a Moderator
    - Attribution effects (Human vs AI) will be stronger under Accurate
      labelling than under Deceptive labelling.

  H4.3 — Attitudinal Moderation Across Studies
    - Attitudes toward AI will moderate attribution effects across
      Studies 1 and 3.

- Studies 1 + 2:
  H4.4 — Visual Presentation Equivalence (Border Robustness)
    - BorderCondition will have little impact on participants’ ratings
      once labelling and attitudes are accounted for.
    - Only labelled image conditions from Study 1 are included here,
      because BorderCondition is only defined for labelled trials.

What this script does (high level)
----------------------------------
For each combined dataset (1+3, 1+2):

  1) Load and validate data using the standardized schema
     (IDs, StudyID, labelling constructs, attitudes, outcomes, composites).

  2) Clean/coerce variables:
     - ParticipantID / ArtworkID:
         * cleaned as string IDs, "nan" and "" treated as missing.
     - PresentedAttribution, ActualOrigin, LabelAccuracy:
         * converted to Categorical (no imputation).
         * "NoLabel" is used as the reference for PresentedAttribution.
     - Optional columns (BorderCondition, ArtStyle, StudyID):
         * cleaned to StringDtype, keeping missing as <NA>.
     - AttitudesTowardAI:
         * coerced to numeric.
         * grand-mean centered to AttitudesTowardAI_c (per combined dataset).

  3) Construct composite outcomes:
     - AestheticJudgment = mean(CreativityRating, AestheticRating,
                               FormalExecutionRating, Curiosity/Curiousity)
     - NegativeEmotion   = mean(EmotionNegHighAvg, EmotionNegLowAvg)
     - PositiveEmotion   = mean(EmotionPosLowAvg, EmotionPosHighAvg)
     - Conservative missingness:
         * Require all 4 items for AestheticJudgment.
         * Require both items for Negative/PositiveEmotion.

  4) Save analysis-ready CSVs:
     - For each combined dataset, an analysis-ready copy is written to:
         C:\Post-doc Work\Gener-ai-te\Results\Study4\<DatasetLabel>\run_YYYYMMDD_HHMMSS\
         and to:
         C:\Post-doc Work\Gener-ai-te\Data\<input_stem>_analysis_ready.csv

  5) Descriptives & diagnostics:
     - Overall descriptives for each DV.
     - Descriptives by PresentedAttribution, PresentedAttribution × ActualOrigin,
       and LabelAccuracy (and BorderCondition where relevant).
     - Outcome distribution diagnostics (skewness, kurtosis, normality tests).
     - Composite coverage and component correlations.

  6) Cross-study mixed-effects models (LMMs; crossed random intercepts):
     For Studies 1+3 combined:
       - H4.1 + H4.3: Pooled attribution model (per outcome):
           Outcome ~ PresentedAttribution * AttitudesTowardAI_c + StudyID
         Random intercepts:
           (1 | ParticipantID) + (1 | ArtworkID)
         Notes:
           * No ActualOrigin term in this primary cross-study model.
           * StudyID is treated as a nuisance covariate (additive).

       - H4.2 + H4.3: Accuracy-moderated attribution model (per outcome),
         restricted to labelled trials with non-null, non-"NoLabel" accuracy:
           Outcome ~ PresentedAttribution * LabelAccuracy + AttitudesTowardAI_c
         Random intercepts:
           (1 | ParticipantID) + (1 | ArtworkID)
         Notes:
           * StudyID is not included here because, for labelled trials,
             LabelAccuracy is structurally confounded with study
             (Study 1 = Accurate, Study 3 = Deceptive).
             We deliberately use LabelAccuracy as the cross-study factor.

     For Studies 1+2 combined:
       - H4.4: Border robustness model (per outcome),
         restricted to labelled trials (PresentedAttribution != NoLabel):
           Outcome ~ BorderCondition + PresentedAttribution + AttitudesTowardAI_c + StudyID
         Random intercepts:
           (1 | ParticipantID) + (1 | ArtworkID)
         Notes:
           * Tests whether BorderCondition has any practically important impact
             once labelling and attitudes are accounted for.

     For each fitted LMM:
       - Extract fixed effects (estimates, SEs, Wald z, p-values, 95% CIs).
       - Compute DV-standardized coefficients and attitude-standardized betas.
       - Extract variance components (participant, artwork, residual).
       - Compute approximate Nakagawa marginal and conditional R².
       - Save model summaries to text files.

  7) Residual diagnostics & influence (per model):
       - Residual vs fitted plot (PNG).
       - Q–Q plot (PNG).
       - Residual summaries (mean, SD, skewness, kurtosis, Shapiro, JB).
       - OLS-based influence diagnostics (leverage, Cook’s distance, studentized residuals).

  8) Estimated marginal means (EMMs):
       - For attribution models (Studies 1+3), EMMs by PresentedAttribution
         at AttitudesTowardAI_c = 0, ±1 SD (within combined data).
       - For the accuracy-moderated models, EMMs by PresentedAttribution × LabelAccuracy
         at AttitudesTowardAI_c levels.
       - For the border robustness models (Studies 1+2), EMMs by BorderCondition
         at AttitudesTowardAI_c = 0.

  9) Random effects summaries:
       - Participant-level BLUPs for random intercepts.
       - Artwork-level residual summaries (mean residual and SD per artwork).

Outputs are organized under:
  C:\Post-doc Work\Gener-ai-te\Results\Study4\<DatasetLabel>\run_YYYYMMDD_HHMMSS\
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set
import traceback

import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
from patsy import dmatrix


# ============================================================
# 1) Patsy helpers
# ============================================================

def Q_(colname: str) -> str:
    """Patsy quoting helper for column names."""
    return f'Q("{colname}")'


def C_(colname: str, use_sum: bool = False) -> str:
    """
    Categorical term wrapper.

    - use_sum=False: default coding (treatment/dummy).
    - use_sum=True : sum-to-zero contrasts (JASP-style Type III).
    """
    if use_sum:
        return f"C({Q_(colname)}, Sum)"
    return f"C({Q_(colname)})"


# ============================================================
# 2) Configuration dataclass
# ============================================================

@dataclass
class Study4Config:
    """
    Configuration object for Study 4 cross-study analyses.

    This config is instantiated separately for:
      - Combined Studies 1+3 (attribution/accuracy/attitudes).
      - Combined Studies 1+2 (border robustness).
    """
    # Paths
    project_root: Path
    data_dir: Path
    input_csv: Path
    output_dir: Path
    dataset_label: str

    # Column schema (standardized)
    participant_id_col: str = "ParticipantID"
    artwork_id_col: str = "ArtworkID"

    presented_attribution_col: str = "PresentedAttribution"
    actual_origin_col: str = "ActualOrigin"
    label_accuracy_col: str = "LabelAccuracy"

    border_condition_col: str = "BorderCondition"   # required for 1+2 analyses
    art_style_col: str = "ArtStyle"                 # optional descriptives
    study_id_col: str = "StudyID"                   # required for both combined datasets

    attitudes_col: str = "AttitudesTowardAI"        # raw numeric
    attitudes_centered_suffix: str = "_c"           # for centered column

    # Outcomes
    outcomes: List[str] = None

    # Composite definitions and missing-data rules
    composites: Dict[str, List[str]] = None
    composite_min_nonmissing: Dict[str, int] = None

    # Reference labels (category ordering; not imputation)
    ref_presented_attr: str = "NoLabel"
    ref_actual_origin: str = "Human"
    ref_label_accuracy: str = "NoLabel"

    # Attitudes centering
    center_attitudes: bool = True

    # Contrast options
    use_sum_contrasts_for_factors: bool = False

    # Will be set during processing per dataset
    attitudes_col_for_model: Optional[str] = None


def build_config_for_dataset(
    dataset_label: str,
    input_filename: str,
    stamp: str,
) -> Study4Config:
    """
    Build a Study4Config for a specific combined dataset (1+3 or 1+2).

    dataset_label: string used to name the subdirectory under Results/Study4.
    input_filename: CSV filename under the Data directory.
    stamp: timestamp string shared across both configs for this run.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "Data"
    input_csv = data_dir / input_filename

    output_dir = project_root / "Results" / "Study4" / dataset_label / f"run_{stamp}"

    cfg = Study4Config(
        project_root=project_root,
        data_dir=data_dir,
        input_csv=input_csv,
        output_dir=output_dir,
        dataset_label=dataset_label,
        outcomes=[
            "AestheticJudgment",
            "NegativeEmotion",
            "PositiveEmotion",
            "IsArtSlider",
            "LikeThisArtSlider",
        ],
        composites={
            "AestheticJudgment": [
                "CreativityRating",
                "AestheticRating",
                "FormalExecutionRating",
                "CuriosityRating",  # may resolve to CuriousityRating
            ],
            "NegativeEmotion": [
                "EmotionNegHighAvg",
                "EmotionNegLowAvg",
            ],
            "PositiveEmotion": [
                "EmotionPosLowAvg",
                "EmotionPosHighAvg",
            ],
        },
        composite_min_nonmissing={
            "AestheticJudgment": 4,
            "NegativeEmotion": 2,
            "PositiveEmotion": 2,
        },
        use_sum_contrasts_for_factors=False,
    )
    return cfg


# ============================================================
# 3) IO, validation, coercion
# ============================================================

def ensure_dirs(cfg: Study4Config) -> None:
    """Ensure that the output directory exists."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV from the given path."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path)


def validate_columns(
    df: pd.DataFrame,
    cfg: Study4Config,
    require_border: bool,
) -> None:
    """
    Validate that required columns exist.

    For cross-study analyses, StudyID is required in addition to the
    Study 1 schema. For border analyses (1+2), BorderCondition is required.
    """
    required_core = [
        cfg.participant_id_col,
        cfg.artwork_id_col,
        cfg.presented_attribution_col,
        cfg.actual_origin_col,
        cfg.label_accuracy_col,
        cfg.attitudes_col,
        cfg.study_id_col,
    ]
    if require_border:
        required_core.append(cfg.border_condition_col)

    # Standalone outcomes (non-composites)
    standalone_outcomes: List[str] = []
    if cfg.outcomes:
        for y in cfg.outcomes:
            if cfg.composites and y in cfg.composites:
                continue
            standalone_outcomes.append(y)

    required: List[str] = list(required_core) + list(standalone_outcomes)

    # Composite components
    if cfg.composites:
        for _, cols in cfg.composites.items():
            for col in cols:
                if col == "CuriosityRating":
                    # Accept CuriosityRating or CuriousityRating
                    if ("CuriosityRating" not in df.columns) and ("CuriousityRating" not in df.columns):
                        required.append("CuriosityRating (or CuriousityRating)")
                else:
                    required.append(col)

    missing: List[str] = []
    for c in required:
        if c == "CuriosityRating (or CuriousityRating)":
            if ("CuriosityRating" not in df.columns) and ("CuriousityRating" not in df.columns):
                missing.append(c)
        else:
            if c not in df.columns:
                missing.append(c)

    if missing:
        raise ValueError(
            f"Missing required columns in CSV ({cfg.dataset_label}):\n"
            + "\n".join([f"  - {c}" for c in missing])
            + "\n\nYour file must use the standardized schema (or update Study4Config)."
        )


def _relevel(categories: List[str], reference: str) -> List[str]:
    """Reorder categories so that 'reference' appears first, if present."""
    categories = [c for c in categories if c is not None]
    if reference in categories:
        return [reference] + [c for c in categories if c != reference]
    return categories


def _to_clean_string_series(s: pd.Series) -> pd.Series:
    """
    Convert to StringDtype, strip whitespace, and normalize missing-like values
    ("nan", "", etc.) to <NA>.
    """
    out = s.astype("string")
    out = out.str.strip()
    out = out.mask(out.str.lower() == "nan", pd.NA)
    out = out.mask(out == "", pd.NA)
    return out


def _clean_id_series(s: pd.Series) -> pd.Series:
    """Clean ID columns; treat 'nan' and '' as missing, not literal IDs."""
    out = s.astype("string").str.strip()
    out = out.mask(out.str.lower() == "nan", pd.NA)
    out = out.mask(out == "", pd.NA)
    return out


def coerce_types(df: pd.DataFrame, cfg: Study4Config) -> pd.DataFrame:
    """
    Coerce and clean columns for cross-study analyses:
      - IDs
      - Categorical attribution variables
      - Optional columns
      - AttitudesTowardAI (and centered version)
    """
    out = df.copy()

    # IDs
    out[cfg.participant_id_col] = _clean_id_series(out[cfg.participant_id_col])
    out[cfg.artwork_id_col] = _clean_id_series(out[cfg.artwork_id_col])

    # PresentedAttribution
    pa = _to_clean_string_series(out[cfg.presented_attribution_col])
    pa_cats = _relevel([x for x in pa.dropna().unique().tolist()], cfg.ref_presented_attr)
    out[cfg.presented_attribution_col] = pd.Categorical(pa, categories=pa_cats, ordered=False)

    # ActualOrigin
    ao = _to_clean_string_series(out[cfg.actual_origin_col])
    ao_cats = _relevel([x for x in ao.dropna().unique().tolist()], cfg.ref_actual_origin)
    out[cfg.actual_origin_col] = pd.Categorical(ao, categories=ao_cats, ordered=False)

    # LabelAccuracy
    la = _to_clean_string_series(out[cfg.label_accuracy_col])
    la_cats = _relevel([x for x in la.dropna().unique().tolist()], cfg.ref_label_accuracy)
    out[cfg.label_accuracy_col] = pd.Categorical(la, categories=la_cats, ordered=False)

    # Optional columns
    for optional_col in [cfg.border_condition_col, cfg.art_style_col, cfg.study_id_col]:
        if optional_col in out.columns:
            out[optional_col] = _to_clean_string_series(out[optional_col])

    # Attitudes numeric + centering
    out[cfg.attitudes_col] = pd.to_numeric(out[cfg.attitudes_col], errors="coerce")

    centered_col = cfg.attitudes_col + cfg.attitudes_centered_suffix
    if cfg.center_attitudes:
        mean_val = out[cfg.attitudes_col].mean(skipna=True)
        out[centered_col] = out[cfg.attitudes_col] - mean_val
        out.attrs["attitudes_center_mean"] = float(mean_val)
        out.attrs["attitudes_centered_col"] = centered_col
    else:
        out.attrs["attitudes_centered_col"] = cfg.attitudes_col

    return out


def drop_missing(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Drop rows with missing values in any of the specified columns."""
    return df.dropna(subset=cols).copy()


# ============================================================
# 3B) Composite construction
# ============================================================

def _resolve_column(df: pd.DataFrame, preferred: str, fallbacks: List[str]) -> str:
    """Resolve ambiguous column name; prefer 'preferred', then fallbacks."""
    if preferred in df.columns:
        return preferred
    for fb in fallbacks:
        if fb in df.columns:
            return fb
    raise KeyError(f"None of these columns were found: {[preferred] + fallbacks}")


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def add_composites(df: pd.DataFrame, cfg: Study4Config) -> pd.DataFrame:
    """
    Add composite outcome columns for AestheticJudgment, NegativeEmotion,
    PositiveEmotion with conservative missingness rules.
    """
    out = df.copy()

    if cfg.composites is None:
        cfg.composites = {}
    if cfg.composite_min_nonmissing is None:
        cfg.composite_min_nonmissing = {}

    for comp_name, comp_cols in cfg.composites.items():
        resolved_cols: List[str] = []
        for col in comp_cols:
            if col == "CuriosityRating":
                resolved_cols.append(_resolve_column(out, "CuriosityRating", ["CuriousityRating"]))
            else:
                resolved_cols.append(col)

        # Ensure numeric
        for c in resolved_cols:
            if c not in out.columns:
                raise ValueError(f"Composite '{comp_name}' requires missing column: {c}")
            out[c] = _coerce_numeric_series(out[c])

        min_n = cfg.composite_min_nonmissing.get(comp_name, len(resolved_cols))

        comp_matrix = out[resolved_cols]
        nonmissing_counts = comp_matrix.notna().sum(axis=1)

        out[comp_name] = comp_matrix.mean(axis=1, skipna=True)
        out.loc[nonmissing_counts < min_n, comp_name] = np.nan

    return out


# ============================================================
# 3C) Composite coverage report
# ============================================================

def composite_coverage_report(
    df_raw: pd.DataFrame,
    df_with_composites: pd.DataFrame,
    cfg: Study4Config
) -> pd.DataFrame:
    """Summarize coverage and missingness for each composite and component."""
    if not cfg.composites:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    total_rows = int(df_with_composites.shape[0])

    for comp_name, comp_cols in cfg.composites.items():
        resolved_cols: List[str] = []
        for col in comp_cols:
            if col == "CuriosityRating":
                resolved = (
                    "CuriosityRating"
                    if "CuriosityRating" in df_raw.columns
                    else ("CuriousityRating" if "CuriousityRating" in df_raw.columns else "CuriosityRating")
                )
                resolved_cols.append(resolved)
            else:
                resolved_cols.append(col)

        if comp_name in df_with_composites.columns:
            comp = df_with_composites[comp_name]
        else:
            comp = pd.Series([np.nan] * total_rows)

        n_nonmissing = int(comp.notna().sum())
        pct_missing = float((total_rows - n_nonmissing) / total_rows) if total_rows > 0 else np.nan

        comp_nonmiss = comp.dropna()
        comp_mean = float(comp_nonmiss.mean()) if n_nonmissing > 0 else np.nan
        comp_sd = float(comp_nonmiss.std(ddof=1)) if n_nonmissing > 1 else np.nan
        comp_min = float(comp_nonmiss.min()) if n_nonmissing > 0 else np.nan
        comp_max = float(comp_nonmiss.max()) if n_nonmissing > 0 else np.nan

        for c in resolved_cols:
            if c not in df_with_composites.columns:
                rows.append({
                    "composite": comp_name,
                    "component": c,
                    "total_rows": total_rows,
                    "composite_nonmissing_rows": n_nonmissing,
                    "composite_pct_missing": pct_missing,
                    "composite_mean": comp_mean,
                    "composite_sd": comp_sd,
                    "composite_min": comp_min,
                    "composite_max": comp_max,
                    "component_missing_n": total_rows,
                    "component_pct_missing": 1.0,
                    "min_required_nonmissing": cfg.composite_min_nonmissing.get(comp_name, len(resolved_cols)),
                    "note": "Component column missing in dataframe",
                })
                continue

            comp_col = df_with_composites[c]
            miss_n = int(comp_col.isna().sum())
            miss_pct = float(miss_n / total_rows) if total_rows > 0 else np.nan

            rows.append({
                "composite": comp_name,
                "component": c,
                "total_rows": total_rows,
                "composite_nonmissing_rows": n_nonmissing,
                "composite_pct_missing": pct_missing,
                "composite_mean": comp_mean,
                "composite_sd": comp_sd,
                "composite_min": comp_min,
                "composite_max": comp_max,
                "component_missing_n": miss_n,
                "component_pct_missing": miss_pct,
                "min_required_nonmissing": cfg.composite_min_nonmissing.get(comp_name, len(resolved_cols)),
                "note": "",
            })

    return pd.DataFrame(rows)


# ============================================================
# 3D) Composite component correlations
# ============================================================

def composite_component_correlations(df: pd.DataFrame, cfg: Study4Config) -> pd.DataFrame:
    """Pearson correlations among component items for each composite."""
    if not cfg.composites:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for comp_name, comp_cols in cfg.composites.items():
        resolved_cols: List[str] = []
        for col in comp_cols:
            if col == "CuriosityRating":
                resolved_cols.append(_resolve_column(df, "CuriosityRating", ["CuriousityRating"]))
            else:
                resolved_cols.append(col)

        sub = df[resolved_cols].apply(pd.to_numeric, errors="coerce")
        k = len(resolved_cols)

        for i in range(k):
            for j in range(i + 1, k):
                v1 = resolved_cols[i]
                v2 = resolved_cols[j]

                x = sub[v1]
                y = sub[v2]
                mask = x.notna() & y.notna()
                n = int(mask.sum())

                if n >= 3:
                    r_val, p_val = stats.pearsonr(x[mask], y[mask])
                else:
                    r_val, p_val = np.nan, np.nan

                rows.append({
                    "composite": comp_name,
                    "var1": v1,
                    "var2": v2,
                    "n": n,
                    "r": float(r_val) if not np.isnan(r_val) else np.nan,
                    "p_value": float(p_val) if not np.isnan(p_val) else np.nan,
                })

    return pd.DataFrame(rows)


# ============================================================
# 4) Descriptive statistics & outcome diagnostics
# ============================================================

def descriptives_overall(df: pd.DataFrame, outcomes: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for y in outcomes:
        s = df[y].dropna()
        n = int(s.shape[0])
        mean = float(s.mean()) if n else np.nan
        sd = float(s.std(ddof=1)) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        ci = stats.t.ppf(0.975, df=max(n - 1, 1)) * se if n > 1 else np.nan
        rows.append({
            "outcome": y,
            "n": n,
            "mean": mean,
            "sd": sd,
            "se": se,
            "ci95_low": mean - ci if n > 1 else np.nan,
            "ci95_high": mean + ci if n > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def descriptives_by_group(df: pd.DataFrame, group_cols: List[str], outcomes: List[str]) -> pd.DataFrame:
    all_rows: List[pd.DataFrame] = []
    for y in outcomes:
        dfi = df.dropna(subset=[y])
        grp = dfi.groupby(group_cols, dropna=False, observed=False)[y]
        desc = grp.agg(["count", "mean", "std"]).reset_index()
        desc.rename(columns={"count": "n", "std": "sd"}, inplace=True)
        desc["se"] = desc["sd"] / np.sqrt(desc["n"])

        def _ci_row(r: pd.Series) -> Tuple[float, float]:
            n = int(r["n"])
            if n <= 1:
                return np.nan, np.nan
            t_crit = stats.t.ppf(0.975, df=n - 1)
            ci = t_crit * r["se"]
            return r["mean"] - ci, r["mean"] + ci

        cis = desc.apply(lambda r: _ci_row(r), axis=1, result_type="expand")
        desc["ci95_low"] = cis[0]
        desc["ci95_high"] = cis[1]
        desc["outcome"] = y
        all_rows.append(desc)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def outcome_distribution_diagnostics(df: pd.DataFrame, outcomes: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for y in outcomes:
        s = df[y].dropna()
        n = int(s.shape[0])
        if n == 0:
            rows.append({
                "outcome": y,
                "n": 0,
                "mean": np.nan,
                "sd": np.nan,
                "min": np.nan,
                "max": np.nan,
                "skew": np.nan,
                "kurtosis": np.nan,
                "shapiro_p": np.nan,
                "jarque_bera_p": np.nan,
            })
            continue

        mean = float(s.mean())
        sd = float(s.std(ddof=1)) if n > 1 else np.nan
        min_val = float(s.min())
        max_val = float(s.max())
        skew_val = float(stats.skew(s, bias=False)) if n > 2 else np.nan
        kurt_val = float(stats.kurtosis(s, fisher=True, bias=False)) if n > 3 else np.nan

        if 3 <= n <= 5000:
            try:
                shapiro_p = float(stats.shapiro(s)[1])
            except Exception:
                shapiro_p = np.nan
        else:
            shapiro_p = np.nan

        try:
            jb_stat, jb_p = stats.jarque_bera(s)
            jb_p = float(jb_p)
        except Exception:
            jb_p = np.nan

        rows.append({
            "outcome": y,
            "n": n,
            "mean": mean,
            "sd": sd,
            "min": min_val,
            "max": max_val,
            "skew": skew_val,
            "kurtosis": kurt_val,
            "shapiro_p": shapiro_p,
            "jarque_bera_p": jb_p,
        })

    return pd.DataFrame(rows)


# ============================================================
# 5) Mixed model fitting (crossed random intercepts)
# ============================================================

@dataclass
class LMMBundle:
    """
    Container for a single fitted LMM in Study 4.
    """
    outcome: str
    model_name: str
    formula: str
    rhs: str
    required_cols: List[str]
    nobs: int
    n_participants: int
    n_artworks: int
    reml: bool
    data_used: pd.DataFrame
    fitted: sm.regression.mixed_linear_model.MixedLMResults
    fixed_effects: pd.DataFrame
    variance_components: pd.DataFrame
    fit_stats: pd.DataFrame


def extract_fixed_effects(
    fitted: sm.regression.mixed_linear_model.MixedLMResults
) -> pd.DataFrame:
    params = fitted.fe_params
    bse = fitted.bse_fe
    zvals = params / bse
    pvals = 2 * (1 - stats.norm.cdf(np.abs(zvals)))
    ci_low = params - 1.96 * bse
    ci_high = params + 1.96 * bse

    return pd.DataFrame({
        "term": params.index,
        "estimate": params.values,
        "se": bse.values,
        "wald_z": zvals.values,
        "p_wald_z": np.asarray(pvals),
        "ci95_low": ci_low.values,
        "ci95_high": ci_high.values,
    })


def extract_variance_components(
    fitted: sm.regression.mixed_linear_model.MixedLMResults
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # Participant intercept variance
    try:
        rows.append({"component": "participant_intercept_var", "value": float(fitted.cov_re.iloc[0, 0])})
    except Exception:
        pass

    # Artwork variance component (first vcomp entry corresponds to vc_formula)
    if getattr(fitted, "vcomp", None) is not None and len(fitted.vcomp) > 0:
        try:
            rows.append({"component": "artwork_intercept_var", "value": float(fitted.vcomp[0])})
        except Exception:
            pass

    rows.append({"component": "residual_var", "value": float(fitted.scale)})
    return pd.DataFrame(rows)


def nakagawa_pseudo_r2(
    fitted: sm.regression.mixed_linear_model.MixedLMResults
) -> Tuple[Optional[float], Optional[float]]:
    """
    Approximate Nakagawa-style pseudo-R²:

      - marginal    : variance explained by fixed effects
      - conditional : variance explained by fixed + random effects
    """
    try:
        fe_linpred = fitted.model.exog @ fitted.fe_params.values
        var_fixed = float(np.var(fe_linpred, ddof=1))
        var_part = float(fitted.cov_re.iloc[0, 0]) if fitted.cov_re is not None else 0.0
        var_vc = float(np.sum(fitted.vcomp)) if getattr(fitted, "vcomp", None) is not None else 0.0
        var_resid = float(fitted.scale)
        var_total = var_fixed + var_part + var_vc + var_resid
        if var_total <= 0:
            return None, None
        return float(var_fixed / var_total), float((var_fixed + var_part + var_vc) / var_total)
    except Exception:
        return None, None


def compute_effect_sizes_for_fixed_effects(
    fe: pd.DataFrame,
    dfi: pd.DataFrame,
    outcome: str,
    att_col_for_model: Optional[str] = None
) -> pd.DataFrame:
    """
    Add DV-standardized coefficients and standardized betas for attitudes.
    """
    fe2 = fe.copy()

    # DV-standardization
    y = dfi[outcome].dropna()
    sd_y = float(np.std(y, ddof=1)) if y.shape[0] > 1 else np.nan

    if sd_y and not np.isnan(sd_y) and sd_y > 0:
        fe2["estimate_std_y"] = fe2["estimate"] / sd_y
        fe2["ci95_low_std_y"] = fe2["ci95_low"] / sd_y
        fe2["ci95_high_std_y"] = fe2["ci95_high"] / sd_y
    else:
        fe2["estimate_std_y"] = np.nan
        fe2["ci95_low_std_y"] = np.nan
        fe2["ci95_high_std_y"] = np.nan

    # Standardized betas for attitudes
    fe2["beta_std_xy"] = np.nan
    fe2["beta_std_xy_ci95_low"] = np.nan
    fe2["beta_std_xy_ci95_high"] = np.nan

    if att_col_for_model is not None and att_col_for_model in dfi.columns:
        x = dfi[att_col_for_model].dropna()
        sd_x = float(np.std(x, ddof=1)) if x.shape[0] > 1 else np.nan

        if (
            sd_x and not np.isnan(sd_x) and sd_x > 0
            and sd_y and not np.isnan(sd_y) and sd_y > 0
        ):
            att_pat = f'Q("{att_col_for_model}")'
            is_att_term = fe2["term"].astype(str).str.contains(att_pat, regex=False)

            fe2.loc[is_att_term, "beta_std_xy"] = (fe2.loc[is_att_term, "estimate"] * sd_x) / sd_y
            fe2.loc[is_att_term, "beta_std_xy_ci95_low"] = (fe2.loc[is_att_term, "ci95_low"] * sd_x) / sd_y
            fe2.loc[is_att_term, "beta_std_xy_ci95_high"] = (fe2.loc[is_att_term, "ci95_high"] * sd_x) / sd_y

    return fe2


def _required_cols_from_rhs(rhs_cols: List[str], base_cols: List[str]) -> List[str]:
    """Build unique ordered list of required columns."""
    seen: Set[str] = set()
    out: List[str] = []
    for c in base_cols + rhs_cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def fit_lmm_crossed_intercepts(
    df: pd.DataFrame,
    cfg: Study4Config,
    outcome: str,
    rhs: str,
    rhs_required_cols: List[str],
    model_name: str,
    reml: bool = True
) -> LMMBundle:
    """
    Fit LMM:
      Outcome ~ rhs
    with crossed random intercepts:
      (1 | ParticipantID) + (1 | ArtworkID)
    """
    base_cols = [outcome, cfg.participant_id_col, cfg.artwork_id_col]
    needed = _required_cols_from_rhs(rhs_required_cols, base_cols)
    dfi = drop_missing(df, needed)

    formula = f"{Q_(outcome)} ~ {rhs}"

    vc = {"artwork": f"0 + C({Q_(cfg.artwork_id_col)})"}

    model = smf.mixedlm(
        formula=formula,
        data=dfi,
        groups=dfi[cfg.participant_id_col],
        vc_formula=vc,
        re_formula="1",
    )

    fitted = model.fit(reml=reml, method="lbfgs")

    fe = extract_fixed_effects(fitted)
    att_col_for_model = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    fe = compute_effect_sizes_for_fixed_effects(
        fe=fe,
        dfi=dfi,
        outcome=outcome,
        att_col_for_model=att_col_for_model,
    )

    vc_tab = extract_variance_components(fitted)
    r2m, r2c = nakagawa_pseudo_r2(fitted)

    fit = pd.DataFrame([{
        "outcome": outcome,
        "model": model_name,
        "formula": formula,
        "rhs": rhs,
        "reml": bool(reml),
        "nobs": int(dfi.shape[0]),
        "n_participants": int(dfi[cfg.participant_id_col].nunique()),
        "n_artworks": int(dfi[cfg.artwork_id_col].nunique()),
        "aic": float(fitted.aic) if fitted.aic is not None else np.nan,
        "bic": float(fitted.bic) if fitted.bic is not None else np.nan,
        "loglik": float(fitted.llf) if fitted.llf is not None else np.nan,
        "pseudo_r2_marginal": r2m,
        "pseudo_r2_conditional": r2c,
        "converged": bool(getattr(fitted, "converged", True)),
    }])

    return LMMBundle(
        outcome=outcome,
        model_name=model_name,
        formula=formula,
        rhs=rhs,
        required_cols=needed,
        nobs=int(dfi.shape[0]),
        n_participants=int(dfi[cfg.participant_id_col].nunique()),
        n_artworks=int(dfi[cfg.artwork_id_col].nunique()),
        reml=bool(reml),
        data_used=dfi,
        fitted=fitted,
        fixed_effects=fe,
        variance_components=vc_tab,
        fit_stats=fit,
    )


# ============================================================
# 6) Study 4 model specifications (RHS)
# ============================================================

def rhs_cs13_attribution(cfg: Study4Config) -> str:
    """
    H4.1 + H4.3: Combined Studies 1+3 attribution model:
      Outcome ~ PresentedAttribution * AttitudesTowardAI_c + StudyID
    """
    pa = C_(cfg.presented_attribution_col, use_sum=cfg.use_sum_contrasts_for_factors)
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    att = Q_(att_col)
    study = C_(cfg.study_id_col, use_sum=cfg.use_sum_contrasts_for_factors)
    return f"{pa} * {att} + {study}"


def rhs_cs13_attribution_required_cols(cfg: Study4Config) -> List[str]:
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    return [cfg.presented_attribution_col, cfg.study_id_col, att_col]


def rhs_cs13_accuracy(cfg: Study4Config) -> str:
    """
    H4.2: Combined Studies 1+3 labelled-only accuracy effect model.

    We treat PresentedAttribution (label content) and LabelAccuracy (accurate vs deceptive)
    as interacting factors, and include AttitudesTowardAI_c as a linear covariate.

      Outcome ~ PresentedAttribution * LabelAccuracy + AttitudesTowardAI_c

    StudyID is omitted here because, for labelled trials, LabelAccuracy is structurally
    confounded with study (Study 1 = Accurate, Study 3 = Deceptive). We therefore use
    LabelAccuracy as the cross-study factor and adjust for AI attitudes rather than
    modeling a full Accuracy × Attitudes moderation.
    """
    pa = C_(cfg.presented_attribution_col, use_sum=cfg.use_sum_contrasts_for_factors)
    la = C_(cfg.label_accuracy_col, use_sum=cfg.use_sum_contrasts_for_factors)
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    att = Q_(att_col)
    # Previously: f"{pa} * {la} * {att}"
    return f"{pa} * {la} + {att}"



def rhs_cs13_accuracy_required_cols(cfg: Study4Config) -> List[str]:
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    return [cfg.presented_attribution_col, cfg.label_accuracy_col, att_col]


def rhs_cs12_border(cfg: Study4Config) -> str:
    """
    H4.4: Combined Studies 1+2 border robustness model (labelled trials only):
      Outcome ~ BorderCondition + PresentedAttribution + AttitudesTowardAI_c + StudyID
    """
    bc = C_(cfg.border_condition_col, use_sum=cfg.use_sum_contrasts_for_factors)
    pa = C_(cfg.presented_attribution_col, use_sum=cfg.use_sum_contrasts_for_factors)
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    att = Q_(att_col)
    study = C_(cfg.study_id_col, use_sum=cfg.use_sum_contrasts_for_factors)
    return f"{bc} + {pa} + {att} + {study}"


def rhs_cs12_border_required_cols(cfg: Study4Config) -> List[str]:
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    return [cfg.border_condition_col, cfg.presented_attribution_col, cfg.study_id_col, att_col]


# ============================================================
# 7) Saving helpers and key-term extraction
# ============================================================

def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract_key_terms_study4(fixed_all: pd.DataFrame, cfg: Study4Config) -> pd.DataFrame:
    """
    Extract the fixed effects most relevant to Study 4 hypotheses:
      - PresentedAttribution terms (H4.1).
      - LabelAccuracy terms (H4.2).
      - AttitudesTowardAI terms (H4.3).
      - BorderCondition terms (H4.4).
      - StudyID (optional, to see cross-study shifts).
    """
    if fixed_all.empty:
        return fixed_all

    term = fixed_all["term"].astype(str)

    pa_pat = f'C(Q("{cfg.presented_attribution_col}"))'
    la_pat = f'C(Q("{cfg.label_accuracy_col}"))'
    bc_pat = f'C(Q("{cfg.border_condition_col}"))'
    study_pat = f'C(Q("{cfg.study_id_col}"))'

    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    att_pat = f'Q("{att_col}")'

    keep = (
        term.str.contains(pa_pat, regex=False)
        | term.str.contains(la_pat, regex=False)
        | term.str.contains(bc_pat, regex=False)
        | term.str.contains(study_pat, regex=False)
        | term.str.contains(att_pat, regex=False)
    )

    return fixed_all.loc[keep].copy()


# ============================================================
# 8) Diagnostics: residuals and influence
# ============================================================

def make_residual_plots(
    bundle: LMMBundle,
    cfg: Study4Config,
    diagnostics_dir: Path
) -> Dict[str, Any]:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    fitted_vals = bundle.fitted.fittedvalues
    residuals = bundle.fitted.resid
    dfi = bundle.data_used

    res_df = pd.DataFrame({
        "row_index": dfi.index,
        "participant_id": dfi[cfg.participant_id_col].values,
        "artwork_id": dfi[cfg.artwork_id_col].values,
        "fitted": np.asarray(fitted_vals),
        "residual": np.asarray(residuals),
    })
    save_df(res_df, diagnostics_dir / "residuals.csv")

    # Residual vs fitted
    fig, ax = plt.subplots()
    ax.scatter(res_df["fitted"], res_df["residual"], alpha=0.5)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"{cfg.dataset_label}: Residuals vs Fitted ({bundle.outcome}, {bundle.model_name})")
    fig.tight_layout()
    fig.savefig(diagnostics_dir / "residuals_vs_fitted.png", dpi=150)
    plt.close(fig)

    # Q–Q plot
    fig, ax = plt.subplots()
    stats.probplot(res_df["residual"], dist="norm", plot=ax)
    ax.set_title(f"{cfg.dataset_label}: Normal Q–Q Plot ({bundle.outcome}, {bundle.model_name})")
    fig.tight_layout()
    fig.savefig(diagnostics_dir / "qqplot_residuals.png", dpi=150)
    plt.close(fig)

    r = res_df["residual"].dropna()
    n = int(r.shape[0])
    if n > 0:
        r_mean = float(r.mean())
        r_sd = float(r.std(ddof=1)) if n > 1 else np.nan
        r_skew = float(stats.skew(r, bias=False)) if n > 2 else np.nan
        r_kurt = float(stats.kurtosis(r, fisher=True, bias=False)) if n > 3 else np.nan

        r_sorted = np.sort(r.values)
        probs = (np.arange(1, n + 1) - 0.5) / n
        q_theor = stats.norm.ppf(probs)
        if n > 1:
            corr_qq = float(np.corrcoef(r_sorted, q_theor)[0, 1])
        else:
            corr_qq = np.nan

        if 3 <= n <= 5000:
            try:
                shapiro_p = float(stats.shapiro(r)[1])
            except Exception:
                shapiro_p = np.nan
        else:
            shapiro_p = np.nan

        try:
            jb_stat, jb_p = stats.jarque_bera(r)
            jb_p = float(jb_p)
        except Exception:
            jb_p = np.nan
    else:
        r_mean = r_sd = r_skew = r_kurt = corr_qq = shapiro_p = jb_p = np.nan

    summary = {
        "dataset": cfg.dataset_label,
        "outcome": bundle.outcome,
        "model": bundle.model_name,
        "n_resid": n,
        "resid_mean": r_mean,
        "resid_sd": r_sd,
        "resid_skew": r_skew,
        "resid_kurtosis": r_kurt,
        "qq_correlation": corr_qq,
        "resid_shapiro_p": shapiro_p,
        "resid_jarque_bera_p": jb_p,
    }

    return summary


def compute_influence_ols_approx(
    bundle: LMMBundle,
    cfg: Study4Config,
    diagnostics_dir: Path
) -> None:
    """
    OLS-based influence diagnostics as a proxy for MixedLM influence.
    """
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    dfi = bundle.data_used
    outcome = bundle.outcome
    formula_ols = f"{Q_(outcome)} ~ {bundle.rhs}"

    try:
        ols_model = smf.ols(formula=formula_ols, data=dfi)
        ols_fit = ols_model.fit()
        infl = ols_fit.get_influence()

        infl_df = pd.DataFrame({
            "row_index": dfi.index,
            "participant_id": dfi[cfg.participant_id_col].values,
            "artwork_id": dfi[cfg.artwork_id_col].values,
            "leverage": infl.hat_matrix_diag,
            "cooks_d": infl.cooks_distance[0],
            "studentized_resid": infl.resid_studentized_internal,
        })

        save_df(infl_df, diagnostics_dir / "influence_ols_approx.csv")
    except Exception:
        err_text = traceback.format_exc()
        save_text(err_text, diagnostics_dir / "influence_ols_approx_ERROR.txt")


def collect_residual_diagnostics(
    bundles: List[LMMBundle],
    cfg: Study4Config
) -> pd.DataFrame:
    summaries: List[Dict[str, Any]] = []

    for b in bundles:
        diag_dir = cfg.output_dir / "diagnostics" / f"{b.outcome}__{b.model_name}"
        try:
            summary = make_residual_plots(b, cfg, diag_dir)
            summaries.append(summary)
        except Exception:
            err_text = traceback.format_exc()
            save_text(err_text, diag_dir / "residual_diagnostics_ERROR.txt")

        try:
            compute_influence_ols_approx(b, cfg, diag_dir)
        except Exception:
            err_text = traceback.format_exc()
            save_text(err_text, diag_dir / "influence_ols_approx_ERROR.txt")

    return pd.DataFrame(summaries) if summaries else pd.DataFrame()


# ============================================================
# 9) EMMs for Study 4 models
# ============================================================

def compute_emm_for_study4_models(
    bundles: List[LMMBundle],
    cfg: Study4Config
) -> pd.DataFrame:
    """
    Compute EMMs for Study 4 models:

      - cs13_attribution:
          EMMs by PresentedAttribution at AttitudesTowardAI_c = 0, ±1 SD,
          with StudyID set to the first observed level.

      - cs13_accuracy_moderator:
          EMMs by PresentedAttribution × LabelAccuracy at AttitudesTowardAI_c
          = 0, ±1 SD.

      - cs12_border_effect:
          EMMs by BorderCondition at AttitudesTowardAI_c = 0, holding
          PresentedAttribution and StudyID at their first observed levels.

    Note: For models that include additional factors, the EMMs fix those
          at a single level (first observed) to provide interpretable
          marginal predictions focused on the effect of interest.
    """
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    emm_rows: List[Dict[str, Any]] = []

    for b in bundles:
        dfi = b.data_used

        # SD of attitudes in this model's data
        if att_col not in dfi.columns:
            continue

        x = dfi[att_col].dropna()
        if x.shape[0] < 2:
            sd_att = np.nan
        else:
            sd_att = float(np.std(x, ddof=1))

        levels_info: List[Tuple[str, float]] = [("mean", 0.0)]
        if sd_att and not np.isnan(sd_att) and sd_att > 0:
            levels_info.append(("plus1sd", sd_att))
            levels_info.append(("minus1sd", -sd_att))

        fe_params = b.fitted.fe_params
        cov_fe = b.fitted.cov_params()

        # -----------------------------
        # cs13_attribution (H4.1 + H4.3)
        # -----------------------------
        if b.model_name == "cs13_attribution":
            # PresentedAttribution levels
            pa_series = dfi[cfg.presented_attribution_col]
            pa_levels = [lvl for lvl in pa_series.dropna().unique().tolist()]
            if not pa_levels:
                continue

            # Choose a reference StudyID (first observed)
            if cfg.study_id_col in dfi.columns:
                study_levels = [lvl for lvl in dfi[cfg.study_id_col].dropna().unique().tolist()]
                study_reference = study_levels[0] if study_levels else None
            else:
                study_reference = None

            for pa_level in pa_levels:
                for lvl_label, lvl_val in levels_info:
                    grid_dict: Dict[str, List[Any]] = {
                        cfg.presented_attribution_col: [pa_level],
                        att_col: [lvl_val],
                    }
                    if study_reference is not None:
                        grid_dict[cfg.study_id_col] = [study_reference]

                    grid = pd.DataFrame(grid_dict)

                    try:
                        X_new = dmatrix("1 + " + b.rhs, grid, return_type="dataframe")
                    except Exception:
                        continue

                    common_cols = [c for c in X_new.columns if c in fe_params.index]
                    if not common_cols:
                        continue

                    X_use = X_new[common_cols]
                    fe_use = fe_params[common_cols]

                    try:
                        cov_use = cov_fe.loc[common_cols, common_cols]
                    except Exception:
                        cov_use = None

                    y_hat = float(np.dot(X_use.values, fe_use.values)[0])

                    if cov_use is not None:
                        var_pred_mat = X_use.values @ cov_use.values @ X_use.values.T
                        var_pred = float(var_pred_mat.ravel()[0])
                        se_pred = np.sqrt(var_pred) if var_pred >= 0 else np.nan
                    else:
                        se_pred = np.nan

                    if not np.isnan(se_pred):
                        ci_low = y_hat - 1.96 * se_pred
                        ci_high = y_hat + 1.96 * se_pred
                    else:
                        ci_low = np.nan
                        ci_high = np.nan

                    emm_rows.append({
                        "dataset": cfg.dataset_label,
                        "outcome": b.outcome,
                        "model": b.model_name,
                        "effect_focus": "PresentedAttribution",
                        "presented_attribution_level": pa_level,
                        "label_accuracy_level": None,
                        "border_condition_level": None,
                        "attitudes_level_label": lvl_label,
                        "attitudes_value": lvl_val,
                        "predicted_mean": y_hat,
                        "se_pred": se_pred,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "nobs_model": b.nobs,
                        "n_participants_model": b.n_participants,
                        "n_artworks_model": b.n_artworks,
                    })

        # -------------------------------------
        # cs13_accuracy_moderator (H4.2 + 4.3)
        # -------------------------------------
        elif b.model_name == "cs13_accuracy_moderator":
            pa_series = dfi[cfg.presented_attribution_col]
            pa_levels = [lvl for lvl in pa_series.dropna().unique().tolist()]
            la_series = dfi[cfg.label_accuracy_col]
            la_levels = [lvl for lvl in la_series.dropna().unique().tolist()]

            if not pa_levels or not la_levels:
                continue

            for pa_level in pa_levels:
                for la_level in la_levels:
                    for lvl_label, lvl_val in levels_info:
                        grid = pd.DataFrame({
                            cfg.presented_attribution_col: [pa_level],
                            cfg.label_accuracy_col: [la_level],
                            att_col: [lvl_val],
                        })

                        try:
                            X_new = dmatrix("1 + " + b.rhs, grid, return_type="dataframe")
                        except Exception:
                            continue

                        common_cols = [c for c in X_new.columns if c in fe_params.index]
                        if not common_cols:
                            continue

                        X_use = X_new[common_cols]
                        fe_use = fe_params[common_cols]

                        try:
                            cov_use = cov_fe.loc[common_cols, common_cols]
                        except Exception:
                            cov_use = None

                        y_hat = float(np.dot(X_use.values, fe_use.values)[0])

                        if cov_use is not None:
                            var_pred_mat = X_use.values @ cov_use.values @ X_use.values.T
                            var_pred = float(var_pred_mat.ravel()[0])
                            se_pred = np.sqrt(var_pred) if var_pred >= 0 else np.nan
                        else:
                            se_pred = np.nan

                        if not np.isnan(se_pred):
                            ci_low = y_hat - 1.96 * se_pred
                            ci_high = y_hat + 1.96 * se_pred
                        else:
                            ci_low = np.nan
                            ci_high = np.nan

                        emm_rows.append({
                            "dataset": cfg.dataset_label,
                            "outcome": b.outcome,
                            "model": b.model_name,
                            "effect_focus": "PresentedAttribution x LabelAccuracy",
                            "presented_attribution_level": pa_level,
                            "label_accuracy_level": la_level,
                            "border_condition_level": None,
                            "attitudes_level_label": lvl_label,
                            "attitudes_value": lvl_val,
                            "predicted_mean": y_hat,
                            "se_pred": se_pred,
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                            "nobs_model": b.nobs,
                            "n_participants_model": b.n_participants,
                            "n_artworks_model": b.n_artworks,
                        })

        # -----------------------------
        # cs12_border_effect (H4.4)
        # -----------------------------
        elif b.model_name == "cs12_border_effect":
            if cfg.border_condition_col not in dfi.columns:
                continue

            bc_series = dfi[cfg.border_condition_col]
            bc_levels = [lvl for lvl in bc_series.dropna().unique().tolist()]
            if not bc_levels:
                continue

            # Hold PresentedAttribution and StudyID at their first observed levels
            if cfg.presented_attribution_col in dfi.columns:
                pa_levels = [lvl for lvl in dfi[cfg.presented_attribution_col].dropna().unique().tolist()]
                pa_reference = pa_levels[0] if pa_levels else None
            else:
                pa_reference = None

            if cfg.study_id_col in dfi.columns:
                study_levels = [lvl for lvl in dfi[cfg.study_id_col].dropna().unique().tolist()]
                study_reference = study_levels[0] if study_levels else None
            else:
                study_reference = None

            # For border, we focus on attitudes=0 only
            border_levels_info: List[Tuple[str, float]] = [("mean", 0.0)]

            for bc_level in bc_levels:
                for lvl_label, lvl_val in border_levels_info:
                    grid_dict: Dict[str, List[Any]] = {
                        cfg.border_condition_col: [bc_level],
                        att_col: [lvl_val],
                    }
                    if pa_reference is not None:
                        grid_dict[cfg.presented_attribution_col] = [pa_reference]
                    if study_reference is not None:
                        grid_dict[cfg.study_id_col] = [study_reference]

                    grid = pd.DataFrame(grid_dict)

                    try:
                        X_new = dmatrix("1 + " + b.rhs, grid, return_type="dataframe")
                    except Exception:
                        continue

                    common_cols = [c for c in X_new.columns if c in fe_params.index]
                    if not common_cols:
                        continue

                    X_use = X_new[common_cols]
                    fe_use = fe_params[common_cols]

                    try:
                        cov_use = cov_fe.loc[common_cols, common_cols]
                    except Exception:
                        cov_use = None

                    y_hat = float(np.dot(X_use.values, fe_use.values)[0])

                    if cov_use is not None:
                        var_pred_mat = X_use.values @ cov_use.values @ X_use.values.T
                        var_pred = float(var_pred_mat.ravel()[0])
                        se_pred = np.sqrt(var_pred) if var_pred >= 0 else np.nan
                    else:
                        se_pred = np.nan

                    if not np.isnan(se_pred):
                        ci_low = y_hat - 1.96 * se_pred
                        ci_high = y_hat + 1.96 * se_pred
                    else:
                        ci_low = np.nan
                        ci_high = np.nan

                    emm_rows.append({
                        "dataset": cfg.dataset_label,
                        "outcome": b.outcome,
                        "model": b.model_name,
                        "effect_focus": "BorderCondition",
                        "presented_attribution_level": pa_reference,
                        "label_accuracy_level": None,
                        "border_condition_level": bc_level,
                        "attitudes_level_label": lvl_label,
                        "attitudes_value": lvl_val,
                        "predicted_mean": y_hat,
                        "se_pred": se_pred,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "nobs_model": b.nobs,
                        "n_participants_model": b.n_participants,
                        "n_artworks_model": b.n_artworks,
                    })

    return pd.DataFrame(emm_rows) if emm_rows else pd.DataFrame()


# ============================================================
# 10) Random effects summaries
# ============================================================

def summarize_random_effects(
    bundles: List[LMMBundle],
    cfg: Study4Config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    part_rows: List[Dict[str, Any]] = []
    art_rows: List[Dict[str, Any]] = []

    for b in bundles:
        # Participant-level BLUPs
        try:
            re_dict = b.fitted.random_effects
            for pid, re_vec in re_dict.items():
                if isinstance(re_vec, (np.ndarray, list, tuple)):
                    val = float(re_vec[0]) if len(re_vec) > 0 else np.nan
                else:
                    try:
                        val = float(re_vec)
                    except Exception:
                        val = np.nan

                part_rows.append({
                    "dataset": cfg.dataset_label,
                    "outcome": b.outcome,
                    "model": b.model_name,
                    "participant_id": pid,
                    "ranef_intercept": val,
                })
        except Exception:
            err_text = traceback.format_exc()
            save_text(
                err_text,
                cfg.output_dir / "diagnostics" / f"{b.outcome}__{b.model_name}" / "ranef_participants_ERROR.txt",
            )

        # Artwork-level residual summaries
        try:
            dfi = b.data_used
            residuals = np.asarray(b.fitted.resid)
            art_df = pd.DataFrame({
                "artwork_id": dfi[cfg.artwork_id_col].values,
                "residual": residuals,
            })
            grp = art_df.groupby("artwork_id")
            agg = grp["residual"].agg(["count", "mean", "std"]).reset_index()
            agg.rename(columns={"count": "n_obs", "mean": "mean_resid", "std": "sd_resid"}, inplace=True)

            for _, row in agg.iterrows():
                art_rows.append({
                    "dataset": cfg.dataset_label,
                    "outcome": b.outcome,
                    "model": b.model_name,
                    "artwork_id": row["artwork_id"],
                    "n_obs": int(row["n_obs"]),
                    "mean_resid": float(row["mean_resid"]) if pd.notna(row["mean_resid"]) else np.nan,
                    "sd_resid": float(row["sd_resid"]) if pd.notna(row["sd_resid"]) else np.nan,
                })
        except Exception:
            err_text = traceback.format_exc()
            save_text(
                err_text,
                cfg.output_dir / "diagnostics" / f"{b.outcome}__{b.model_name}" / "ranef_artworks_ERROR.txt",
            )

    ranef_participants_df = pd.DataFrame(part_rows) if part_rows else pd.DataFrame()
    ranef_artworks_df = pd.DataFrame(art_rows) if art_rows else pd.DataFrame()

    return ranef_participants_df, ranef_artworks_df


# ============================================================
# 11) Study 4 pipelines
# ============================================================

def run_study4_combined_1_3(cfg: Study4Config) -> None:
    """
    Run cross-study analyses for Combined Studies 1+3 (H4.1–H4.3).
    """
    ensure_dirs(cfg)

    # Load and validate
    df_raw = load_data(cfg.input_csv)
    validate_columns(df_raw, cfg, require_border=False)

    # Clean & composites
    df = coerce_types(df_raw, cfg)
    centered_col = df.attrs.get("attitudes_centered_col", cfg.attitudes_col)
    cfg.attitudes_col_for_model = centered_col

    df = add_composites(df, cfg)

    # Save analysis-ready
    analysis_ready_name = cfg.input_csv.stem + "_analysis_ready.csv"
    analysis_ready_path = cfg.output_dir / analysis_ready_name
    save_df(df, analysis_ready_path)
    save_df(df, cfg.data_dir / analysis_ready_name)

    # Cell counts for PresentedAttribution × ActualOrigin
    cell_counts = (
        df.groupby(
            [cfg.presented_attribution_col, cfg.actual_origin_col],
            dropna=False,
            observed=False,
        )
        .size()
        .reset_index(name="n")
    )
    save_df(cell_counts, cfg.output_dir / "cell_counts_presented_x_origin.csv")

    # Composite coverage and correlations
    cov = composite_coverage_report(df_raw, df, cfg)
    save_df(cov, cfg.output_dir / "composite_coverage_report.csv")

    comp_corr = composite_component_correlations(df, cfg)
    save_df(comp_corr, cfg.output_dir / "composite_component_correlations.csv")

    # Metadata
    meta = pd.DataFrame([{
        "dataset_label": cfg.dataset_label,
        "input_csv": str(cfg.input_csv),
        "n_rows_raw": int(df_raw.shape[0]),
        "n_rows_after_processing": int(df.shape[0]),
        "attitudes_centered": bool(cfg.center_attitudes),
        "attitudes_center_mean": df.attrs.get("attitudes_center_mean", np.nan),
        "attitudes_col_for_model": cfg.attitudes_col_for_model,
        "use_sum_contrasts_for_factors": bool(cfg.use_sum_contrasts_for_factors),
        "timestamp": datetime.now().isoformat(),
    }])
    save_df(meta, cfg.output_dir / "run_metadata.csv")

    # Descriptives
    save_df(
        descriptives_overall(df, cfg.outcomes),
        cfg.output_dir / "descriptives_overall.csv",
    )

    save_df(
        descriptives_by_group(df, [cfg.presented_attribution_col], cfg.outcomes),
        cfg.output_dir / "descriptives_by_presentedattribution.csv",
    )

    save_df(
        descriptives_by_group(
            df,
            [cfg.presented_attribution_col, cfg.actual_origin_col],
            cfg.outcomes,
        ),
        cfg.output_dir / "descriptives_by_presentedattribution_x_origin.csv",
    )

    save_df(
        descriptives_by_group(df, [cfg.label_accuracy_col], cfg.outcomes),
        cfg.output_dir / "descriptives_by_labelaccuracy.csv",
    )

    out_diag = outcome_distribution_diagnostics(df, cfg.outcomes)
    save_df(out_diag, cfg.output_dir / "outcome_distribution_diagnostics.csv")

    # --------------------------------------------------------
    # Model fitting (H4.1–H4.3)
    # --------------------------------------------------------
    bundles: List[LMMBundle] = []
    model_errors: List[dict] = []

    # H4.1 + H4.3: pooled attribution model
    rhs_attr = rhs_cs13_attribution(cfg)
    rhs_attr_cols = rhs_cs13_attribution_required_cols(cfg)

    for outcome in cfg.outcomes:
        try:
            b = fit_lmm_crossed_intercepts(
                df=df,
                cfg=cfg,
                outcome=outcome,
                rhs=rhs_attr,
                rhs_required_cols=rhs_attr_cols,
                model_name="cs13_attribution",
                reml=True,
            )
            bundles.append(b)
            save_text(
                str(b.fitted.summary()),
                cfg.output_dir / "model_summaries" / f"{outcome}__cs13_attribution_summary.txt",
            )
        except Exception as e:
            model_errors.append({
                "dataset": cfg.dataset_label,
                "outcome": outcome,
                "model": "cs13_attribution",
                "formula": f"{outcome} ~ {rhs_attr}",
                "rhs_required_cols": rhs_attr_cols,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })
            save_text(
                traceback.format_exc(),
                cfg.output_dir / "model_summaries" / f"{outcome}__cs13_attribution_ERROR.txt",
            )

    # H4.2 + H4.3: labelled-only accuracy moderator
    rhs_acc = rhs_cs13_accuracy(cfg)
    rhs_acc_cols = rhs_cs13_accuracy_required_cols(cfg)

    # Restrict to labelled trials with non-null, non-"NoLabel" LabelAccuracy
    la_series = df[cfg.label_accuracy_col].astype("string")
    pa_series = df[cfg.presented_attribution_col].astype("string")

    mask_labelled = (
        pa_series.notna()
        & (pa_series != cfg.ref_presented_attr)
        & la_series.notna()
        & (la_series != cfg.ref_label_accuracy)
    )
    df_labelled = df[mask_labelled].copy()

    for outcome in cfg.outcomes:
        try:
            needed_check = [
                outcome,
                cfg.participant_id_col,
                cfg.artwork_id_col,
                cfg.presented_attribution_col,
                cfg.label_accuracy_col,
                cfg.attitudes_col_for_model,
            ]
            if drop_missing(df_labelled, needed_check).shape[0] < 10:
                continue

            b = fit_lmm_crossed_intercepts(
                df=df_labelled,
                cfg=cfg,
                outcome=outcome,
                rhs=rhs_acc,
                rhs_required_cols=rhs_acc_cols,
                model_name="cs13_accuracy_moderator",
                reml=True,
            )
            bundles.append(b)
            save_text(
                str(b.fitted.summary()),
                cfg.output_dir / "model_summaries" / f"{outcome}__cs13_accuracy_moderator_summary.txt",
            )
        except Exception as e:
            model_errors.append({
                "dataset": cfg.dataset_label,
                "outcome": outcome,
                "model": "cs13_accuracy_moderator",
                "formula": f"{outcome} ~ {rhs_acc}",
                "rhs_required_cols": rhs_acc_cols,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })
            save_text(
                traceback.format_exc(),
                cfg.output_dir / "model_summaries" / f"{outcome}__cs13_accuracy_moderator_ERROR.txt",
            )

    # Aggregate outputs
    fixed_all = pd.concat(
        [
            b.fixed_effects.assign(
                dataset=cfg.dataset_label,
                outcome=b.outcome,
                model=b.model_name,
                rhs=b.rhs,
                reml=b.reml,
                nobs=b.nobs,
            )
            for b in bundles
        ],
        ignore_index=True,
    ) if bundles else pd.DataFrame()

    vc_all = pd.concat(
        [
            b.variance_components.assign(
                dataset=cfg.dataset_label,
                outcome=b.outcome,
                model=b.model_name,
                rhs=b.rhs,
                reml=b.reml,
                nobs=b.nobs,
            )
            for b in bundles
        ],
        ignore_index=True,
    ) if bundles else pd.DataFrame()

    fit_all = pd.concat(
        [b.fit_stats.assign(dataset=cfg.dataset_label) for b in bundles],
        ignore_index=True,
    ) if bundles else pd.DataFrame()

    req_rows: List[Dict[str, Any]] = []
    for b in bundles:
        req_rows.append({
            "dataset": cfg.dataset_label,
            "outcome": b.outcome,
            "model": b.model_name,
            "rhs": b.rhs,
            "reml": b.reml,
            "required_cols": "|".join(b.required_cols),
            "nobs": b.nobs,
            "n_participants": b.n_participants,
            "n_artworks": b.n_artworks,
        })
    required_cols_report = pd.DataFrame(req_rows) if req_rows else pd.DataFrame()

    save_df(required_cols_report, cfg.output_dir / "lmm_required_columns_by_model.csv")
    save_df(fixed_all, cfg.output_dir / "lmm_fixed_effects_all.csv")
    save_df(vc_all, cfg.output_dir / "lmm_variance_components_all.csv")
    save_df(fit_all, cfg.output_dir / "lmm_fit_statistics_all.csv")

    # Key terms (H4.1–H4.3)
    save_df(
        extract_key_terms_study4(fixed_all, cfg),
        cfg.output_dir / "key_hypothesis_tests_study4_cs13.csv",
    )

    # Residuals & influence
    if bundles:
        resid_diag_df = collect_residual_diagnostics(bundles, cfg)
        save_df(resid_diag_df, cfg.output_dir / "lmm_residual_diagnostics.csv")
    else:
        save_df(pd.DataFrame(), cfg.output_dir / "lmm_residual_diagnostics.csv")

    # EMMs
    emm_df = compute_emm_for_study4_models(bundles, cfg)
    save_df(emm_df, cfg.output_dir / "emm_study4_cs13_models.csv")

    # Random effects
    ranef_participants_df, ranef_artworks_df = summarize_random_effects(bundles, cfg)
    save_df(ranef_participants_df, cfg.output_dir / "ranef_participants.csv")
    save_df(ranef_artworks_df, cfg.output_dir / "ranef_artworks.csv")

    # Model errors
    if model_errors:
        save_df(pd.DataFrame(model_errors), cfg.output_dir / "model_errors.csv")
    else:
        save_df(pd.DataFrame(), cfg.output_dir / "model_errors.csv")


def run_study4_combined_1_2(cfg: Study4Config) -> None:
    """
    Run cross-study analyses for Combined Studies 1+2 (H4.4; border robustness).
    """
    ensure_dirs(cfg)

    # Load and validate (border required)
    df_raw = load_data(cfg.input_csv)
    validate_columns(df_raw, cfg, require_border=True)

    # Clean & composites
    df = coerce_types(df_raw, cfg)
    centered_col = df.attrs.get("attitudes_centered_col", cfg.attitudes_col)
    cfg.attitudes_col_for_model = centered_col

    df = add_composites(df, cfg)

    # Save analysis-ready
    analysis_ready_name = cfg.input_csv.stem + "_analysis_ready.csv"
    analysis_ready_path = cfg.output_dir / analysis_ready_name
    save_df(df, analysis_ready_path)
    save_df(df, cfg.data_dir / analysis_ready_name)

    # Restrict to labelled trials (PresentedAttribution != NoLabel)
    pa_series = df[cfg.presented_attribution_col].astype("string")
    df_labelled = df[pa_series != cfg.ref_presented_attr].copy()

    # Cell counts (for labelled subset)
    cell_counts = (
        df_labelled.groupby(
            [cfg.presented_attribution_col, cfg.actual_origin_col],
            dropna=False,
            observed=False,
        )
        .size()
        .reset_index(name="n")
    )
    save_df(cell_counts, cfg.output_dir / "cell_counts_presented_x_origin_labelled.csv")

    # Composite coverage and correlations
    cov = composite_coverage_report(df_raw, df, cfg)
    save_df(cov, cfg.output_dir / "composite_coverage_report.csv")

    comp_corr = composite_component_correlations(df, cfg)
    save_df(comp_corr, cfg.output_dir / "composite_component_correlations.csv")

    # Metadata
    meta = pd.DataFrame([{
        "dataset_label": cfg.dataset_label,
        "input_csv": str(cfg.input_csv),
        "n_rows_raw": int(df_raw.shape[0]),
        "n_rows_after_processing": int(df.shape[0]),
        "n_rows_labelled": int(df_labelled.shape[0]),
        "attitudes_centered": bool(cfg.center_attitudes),
        "attitudes_center_mean": df.attrs.get("attitudes_center_mean", np.nan),
        "attitudes_col_for_model": cfg.attitudes_col_for_model,
        "use_sum_contrasts_for_factors": bool(cfg.use_sum_contrasts_for_factors),
        "timestamp": datetime.now().isoformat(),
    }])
    save_df(meta, cfg.output_dir / "run_metadata.csv")

    # Descriptives (using labelled subset, as this is where BorderCondition is defined)
    save_df(
        descriptives_overall(df_labelled, cfg.outcomes),
        cfg.output_dir / "descriptives_overall_labelled.csv",
    )

    save_df(
        descriptives_by_group(df_labelled, [cfg.border_condition_col], cfg.outcomes),
        cfg.output_dir / "descriptives_by_bordercondition_labelled.csv",
    )

    save_df(
        descriptives_by_group(df_labelled, [cfg.presented_attribution_col], cfg.outcomes),
        cfg.output_dir / "descriptives_by_presentedattribution_labelled.csv",
    )

    out_diag = outcome_distribution_diagnostics(df_labelled, cfg.outcomes)
    save_df(out_diag, cfg.output_dir / "outcome_distribution_diagnostics_labelled.csv")

    # --------------------------------------------------------
    # Model fitting (H4.4)
    # --------------------------------------------------------
    bundles: List[LMMBundle] = []
    model_errors: List[dict] = []

    rhs_border = rhs_cs12_border(cfg)
    rhs_border_cols = rhs_cs12_border_required_cols(cfg)

    for outcome in cfg.outcomes:
        try:
            b = fit_lmm_crossed_intercepts(
                df=df_labelled,
                cfg=cfg,
                outcome=outcome,
                rhs=rhs_border,
                rhs_required_cols=rhs_border_cols,
                model_name="cs12_border_effect",
                reml=True,
            )
            bundles.append(b)
            save_text(
                str(b.fitted.summary()),
                cfg.output_dir / "model_summaries" / f"{outcome}__cs12_border_effect_summary.txt",
            )
        except Exception as e:
            model_errors.append({
                "dataset": cfg.dataset_label,
                "outcome": outcome,
                "model": "cs12_border_effect",
                "formula": f"{outcome} ~ {rhs_border}",
                "rhs_required_cols": rhs_border_cols,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })
            save_text(
                traceback.format_exc(),
                cfg.output_dir / "model_summaries" / f"{outcome}__cs12_border_effect_ERROR.txt",
            )

    # Aggregate outputs
    fixed_all = pd.concat(
        [
            b.fixed_effects.assign(
                dataset=cfg.dataset_label,
                outcome=b.outcome,
                model=b.model_name,
                rhs=b.rhs,
                reml=b.reml,
                nobs=b.nobs,
            )
            for b in bundles
        ],
        ignore_index=True,
    ) if bundles else pd.DataFrame()

    vc_all = pd.concat(
        [
            b.variance_components.assign(
                dataset=cfg.dataset_label,
                outcome=b.outcome,
                model=b.model_name,
                rhs=b.rhs,
                reml=b.reml,
                nobs=b.nobs,
            )
            for b in bundles
        ],
        ignore_index=True,
    ) if bundles else pd.DataFrame()

    fit_all = pd.concat(
        [b.fit_stats.assign(dataset=cfg.dataset_label) for b in bundles],
        ignore_index=True,
    ) if bundles else pd.DataFrame()

    req_rows: List[Dict[str, Any]] = []
    for b in bundles:
        req_rows.append({
            "dataset": cfg.dataset_label,
            "outcome": b.outcome,
            "model": b.model_name,
            "rhs": b.rhs,
            "reml": b.reml,
            "required_cols": "|".join(b.required_cols),
            "nobs": b.nobs,
            "n_participants": b.n_participants,
            "n_artworks": b.n_artworks,
        })
    required_cols_report = pd.DataFrame(req_rows) if req_rows else pd.DataFrame()

    save_df(required_cols_report, cfg.output_dir / "lmm_required_columns_by_model.csv")
    save_df(fixed_all, cfg.output_dir / "lmm_fixed_effects_all.csv")
    save_df(vc_all, cfg.output_dir / "lmm_variance_components_all.csv")
    save_df(fit_all, cfg.output_dir / "lmm_fit_statistics_all.csv")

    # Key terms (H4.4)
    save_df(
        extract_key_terms_study4(fixed_all, cfg),
        cfg.output_dir / "key_hypothesis_tests_study4_cs12.csv",
    )

    # Residuals & influence
    if bundles:
        resid_diag_df = collect_residual_diagnostics(bundles, cfg)
        save_df(resid_diag_df, cfg.output_dir / "lmm_residual_diagnostics.csv")
    else:
        save_df(pd.DataFrame(), cfg.output_dir / "lmm_residual_diagnostics.csv")

    # EMMs (border-focused)
    emm_df = compute_emm_for_study4_models(bundles, cfg)
    save_df(emm_df, cfg.output_dir / "emm_study4_cs12_models.csv")

    # Random effects
    ranef_participants_df, ranef_artworks_df = summarize_random_effects(bundles, cfg)
    save_df(ranef_participants_df, cfg.output_dir / "ranef_participants.csv")
    save_df(ranef_artworks_df, cfg.output_dir / "ranef_artworks.csv")

    # Model errors
    if model_errors:
        save_df(pd.DataFrame(model_errors), cfg.output_dir / "model_errors.csv")
    else:
        save_df(pd.DataFrame(), cfg.output_dir / "model_errors.csv")


def main() -> None:
    """
    Entry point for Study 4 cross-study analyses.

    Runs:
      - Combined Studies 1+3 (H4.1–H4.3)
      - Combined Studies 1+2 (H4.4)
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg_13 = build_config_for_dataset(
        dataset_label="Study1_Study3",
        input_filename="Generaite_Study1_Study3_Combined_1-26-2026.csv",
        stamp=stamp,
    )

    cfg_12 = build_config_for_dataset(
        dataset_label="Study1_Study2",
        input_filename="Generaite_Study1_Study2_Combined_1-21-2026.csv",
        stamp=stamp,
    )

    # Ensure project/data directories exist
    cfg_13.project_root.mkdir(parents=True, exist_ok=True)
    cfg_13.data_dir.mkdir(parents=True, exist_ok=True)

    # Run analyses
    run_study4_combined_1_3(cfg_13)
    run_study4_combined_1_2(cfg_12)

    print("Study 4 cross-study analyses complete.")
    print(f"  Combined Studies 1+3 outputs:\n    {cfg_13.output_dir}")
    print(f"  Combined Studies 1+2 outputs:\n    {cfg_12.output_dir}")


if __name__ == "__main__":
    main()
