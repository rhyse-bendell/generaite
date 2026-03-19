"""
Gener-AI-te — Study 2 Analysis Script
(Border Color as Contextual Moderator of Labelling and AI Attitudes)

Author: Dr. Rhyse Bendell + GPT-5.1 Thinking
Date: 2026-01-26

What this script assumes
------------------------
Your Study 2 CSV is in long format (one row per ParticipantID × ArtworkID response).

This script uses the standardized column schema:

REQUIRED IDENTIFIERS
- ParticipantID
- ArtworkID

CORE ATTRIBUTION / LABELLING CONSTRUCTS
- PresentedAttribution    : {None/NoLabel, Human, AI}
    NOTE: Missing values are treated as missing data, not recoded.
- ActualOrigin            : {Human, AI}
- LabelAccuracy           : {None, Accurate, Deceptive}

VISUAL / PRESENTATION
- BorderCondition         : {Original, Swapped, Neutral, ...}
    - In Study 2, this factor is central and is modeled explicitly.

ARTWORK DESCRIPTORS
- ArtStyle                : {Abstract, Impressionist, Baroque, Realism} (optional, descriptives only)

PARTICIPANT MODERATOR
- AttitudesTowardAI       : numeric
    - This script creates a centered version, AttitudesTowardAI_c, for modeling.

STUDY IDENTIFIER
- StudyID                 : {1,2,3} (optional; not used in Study 2-only models but saved if present)

OUTCOMES (DVs) — analyzed endpoints (composites + standalone sliders)
Composites:
- AestheticJudgment   = mean(CreativityRating, AestheticRating,
                             FormalExecutionRating, CuriosityRating/CuriousityRating)
- NegativeEmotion     = mean(EmotionNegHighAvg, EmotionNegLowAvg)
- PositiveEmotion     = mean(EmotionPosLowAvg, EmotionPosHighAvg)

Standalone endpoints:
- IsArtSlider
- LikeThisArtSlider

Study 2 hypotheses supported by this script
-------------------------------------------
H2.1 — Replication of Attribution Effects
  Across border conditions, artwork labelled as Human will be evaluated more
  positively than artwork labelled as AI, and both will differ from the no-label
  condition.

  Tested via the "replication" mixed model:
      Outcome ~ PresentedAttribution * AttitudesTowardAI_c
  (identical structure to Study 1 primary) to check that core labelling effects
  hold in the Study 2 dataset.

H2.2 — Border Color × Attribution Interaction
  The magnitude of attribution-based evaluation differences will depend on border color.

  Tested via the border-interaction mixed model:
      Outcome ~ PresentedAttribution * BorderCondition * AttitudesTowardAI_c
  The two-way PresentedAttribution × BorderCondition terms directly test visual
  contextual modulation of labelling effects.

H2.3 — Moderation by Attitudes Toward AI
  Attitudes toward AI will moderate the interaction between border color and
  attribution content.

  Tested via three-way interaction terms from the same border-interaction model:
      PresentedAttribution × BorderCondition × AttitudesTowardAI_c

Equivalence testing (TOST) for border-related effects
-----------------------------------------------------
To support conclusions that border color does not introduce practically meaningful
differences, this script additionally performs TOST (Two One-Sided Tests)
equivalence testing on all fixed-effect coefficients:

- Equivalence bounds are set in standardized-DV units:
    ±tost_delta_std_y (default ±0.20 SD_y).
- For each fixed effect, TOST is conducted on the raw-scale coefficient using
  bounds: ±(tost_delta_std_y × SD_y).
- For border-related terms (those involving BorderCondition), we get:
    - TOST p-values for each one-sided test and the max p-value.
    - An indicator for whether the term is statistically equivalent to 0
      within the specified equivalence bounds at tost_alpha (default .05).
- These results are written to:
    - tost_equivalence_all_fixed_effects.csv
    - tost_equivalence_key_border_terms_study2.csv
  allowing direct inspection of whether border-related effects are both
  non-significant in the usual NHST sense and statistically equivalent to
  zero-sized effects within the chosen bounds.

What this script does (high-level)
----------------------------------
1) Load and validate data:
   - Reads the Study 2 CSV from:
       C:\\Post-doc Work\\Gener-ai-te\\Data\\Generaite_Study2_1-21-2026.csv
   - Checks that required columns are present (ID variables, outcome components,
     predictors, and BorderCondition).

2) Clean and coerce variables:
   - ParticipantID / ArtworkID:
       - Converted to string with whitespace removed.
       - Empty strings and "nan" (as text) treated as missing.
   - Label variables (PresentedAttribution, ActualOrigin, LabelAccuracy):
       - Converted to pandas Categorical.
       - Missing values remain missing (no imputation).
       - Reference categories (e.g., "NoLabel", "Human") placed first if present.
   - BorderCondition:
       - Cleaned, converted to Categorical with "Neutral" as reference if present.
   - Optional variables (ArtStyle, StudyID):
       - Cleaned to string if present.
   - AttitudesTowardAI:
       - Coerced to numeric.
       - Grand-mean centered to AttitudesTowardAI_c for modeling.

3) Construct composite outcomes:
   - AestheticJudgment, NegativeEmotion, PositiveEmotion:
       - Component items coerced to numeric.
       - Composite = row-wise mean with strict missing-data rule:
           * AestheticJudgment: 4 non-missing components required.
           * NegativeEmotion, PositiveEmotion: 2 non-missing components required.

4) Save an analysis-ready CSV:
   - Includes:
       * Original columns
       * Cleaned/categorical predictors
       * Centered AttitudesTowardAI_c
       * Composite outcomes
   - Saved into the run-specific results folder and into:
       C:\\Post-doc Work\\Gener-ai-te\\Data\\Generaite_Study2_analysis_ready.csv

5) Basic descriptives and coverage:
   - Cell counts for:
       * PresentedAttribution × ActualOrigin
       * PresentedAttribution × ActualOrigin × BorderCondition
   - Composite coverage report (valid rows, component missingness).
   - Component correlation matrices for each composite.
   - Overall descriptives for each DV.
   - Descriptives by:
       * PresentedAttribution
       * PresentedAttribution × ActualOrigin
       * LabelAccuracy
       * BorderCondition
       * PresentedAttribution × BorderCondition
   - Outcome distribution diagnostics (mean, SD, skewness, kurtosis, normality tests).

6) Mixed-effects models (LMMs) for Study 2:
   A) Replication model (per outcome; H2.1):
        Outcome ~ PresentedAttribution * AttitudesTowardAI_c
      Random intercepts:
        (1 | ParticipantID) + (1 | ArtworkID)

   B) Border interaction model (per outcome; H2.2 & H2.3):
        Outcome ~ PresentedAttribution * BorderCondition * AttitudesTowardAI_c
      Same crossed random intercept structure.

   - ParticipantID random intercept via the MixedLM "groups" argument.
   - ArtworkID random intercept via a variance component (vc_formula).
   - No imputation: rows are dropped only if they lack variables required for
     each specific model.

   For each fitted model, we record:
      - Fixed effects, SEs, Wald z-tests, p-values, 95% CIs.
      - DV-standardized effect sizes.
      - Standardized betas for AttitudesTowardAI terms.
      - Variance components (participant, artwork, residual).
      - Approximate Nakagawa pseudo-R² (marginal, conditional).
      - Fit diagnostics (AIC, BIC, log-likelihood, convergence flag).

7) Extended diagnostics:
   A) Outcome distribution diagnostics:
      - N, mean, SD, min, max, skewness, kurtosis, Shapiro–Wilk p-value (where feasible).
      - Saved as outcome_distribution_diagnostics.csv.

   B) Model-level residual diagnostics:
      - Residual vs fitted plots (PNG).
      - Q–Q plots (PNG).
      - Residuals CSV per model.
      - Residual summary statistics and normality tests.
      - Aggregated into lmm_residual_diagnostics.csv.

   C) OLS-based influence diagnostics (approximate):
      - Fit OLS with same fixed-effects RHS for each LMM.
      - Compute leverage, Cook’s distance, studentized residuals.
      - Saved per model as influence_ols_approx.csv.

8) Estimated marginal means (EMMs) for replication models:
   - For each outcome, from the replication model:
       * EMMs by PresentedAttribution, at AttitudesTowardAI_c = 0 (mean),
         +1 SD, and -1 SD (if SD > 0).
   - Saved as:
       emm_study2_replication.csv

9) Random effects summaries:
   - Participant-level random intercept BLUPs.
   - Artwork-level residual summaries as proxies for artwork random effects.
   - Saved as:
       ranef_participants.csv
       ranef_artworks.csv

10) Key hypothesis tests + TOST outputs
---------------------------------------
- key_hypothesis_tests_study2.csv
    * Subset of fixed effects involving PresentedAttribution, BorderCondition, and
      AttitudesTowardAI_c from both models.

- tost_equivalence_all_fixed_effects.csv
    * TOST results for every fixed effect in every model/outcome.

- tost_equivalence_key_border_terms_study2.csv
    * TOST subset for fixed effects involving BorderCondition (main and interaction
      terms), used to support claims that border effects are practically negligible
      within the chosen equivalence bounds.

Inference notes
---------------
- MixedLM fixed-effect p-values are Wald z-tests (normal approximation).
- No imputation is performed.
- Pseudo R² values are approximate and descriptive.
- OLS-based influence diagnostics are approximations for the mixed model.
- TOST uses normal-approximation z-tests (large-sample) with raw-scale bounds
  derived from the chosen standardized-DV equivalence bounds.
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
# 1) Patsy helpers (quoting column names in formulas)
# ============================================================

def Q_(colname: str) -> str:
    """
    Patsy quoting helper for column names.

    We use Q("colname") whenever a column name might contain spaces,
    punctuation, or other characters that Patsy would otherwise treat
    as special syntax.
    """
    return f'Q("{colname}")'


def C_(colname: str, use_sum: bool = False) -> str:
    """
    Safe categorical term wrapper.

    - If use_sum is False (default), we use Patsy's default coding
      for categorical variables (treatment/dummy coding).

    - If use_sum is True, we wrap with Sum contrasts.

    This affects parameterization but not underlying cell means.
    """
    if use_sum:
        return f"C({Q_(colname)}, Sum)"
    return f"C({Q_(colname)})"


# ============================================================
# 2) Configuration dataclass
# ============================================================

@dataclass
class Study2Config:
    """
    Configuration object for Study 2 analysis.

    Centralizes:
      - File paths
      - Column names
      - Composite outcome definitions
      - Reference category choices
      - Modeling options
      - Equivalence test (TOST) options
    """
    # Paths
    project_root: Path
    data_dir: Path
    input_csv: Path
    output_dir: Path

    # Column schema (standardized)
    participant_id_col: str = "ParticipantID"
    artwork_id_col: str = "ArtworkID"

    presented_attribution_col: str = "PresentedAttribution"
    actual_origin_col: str = "ActualOrigin"
    label_accuracy_col: str = "LabelAccuracy"

    border_condition_col: str = "BorderCondition"
    art_style_col: str = "ArtStyle"                 # optional
    study_id_col: str = "StudyID"                   # optional

    attitudes_col: str = "AttitudesTowardAI"
    attitudes_centered_suffix: str = "_c"

    # Outcomes
    outcomes: List[str] = None

    # Composite definitions and missing-data rules
    composites: Dict[str, List[str]] = None
    composite_min_nonmissing: Dict[str, int] = None

    # Reference labels (ordering only; no imputation)
    ref_presented_attr: str = "NoLabel"
    ref_actual_origin: str = "Human"
    ref_label_accuracy: str = "NoLabel"
    ref_border_condition: str = "Neutral"

    # Center attitudes?
    center_attitudes: bool = True

    # Sum contrasts option
    use_sum_contrasts_for_factors: bool = False

    # Whether to run replication models and border interaction models
    run_replication_models: bool = True
    run_border_interaction_models: bool = True

    # TOST equivalence test settings
    # Equivalence bounds are ±tost_delta_std_y in standardized-DV units.
    tost_delta_std_y: float = 0.20
    tost_alpha: float = 0.05


def build_default_config() -> Study2Config:
    """
    Build the default configuration for Study 2.

    Adjust input_csv if your Study 2 file has a different name.
    """
    project_root = Path(r"C:\Post-doc Work\Gener-ai-te")
    data_dir = project_root / "Data"
    input_csv = data_dir / "Generaite_Study2_1-21-2026.csv"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "Results" / "Study2" / f"run_{stamp}"

    cfg = Study2Config(
        project_root=project_root,
        data_dir=data_dir,
        input_csv=input_csv,
        output_dir=output_dir,
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
                "CuriosityRating",
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
        run_replication_models=True,
        run_border_interaction_models=True,
        tost_delta_std_y=0.20,
        tost_alpha=0.05,
    )
    return cfg


# ============================================================
# 3) IO, validation, and coercion
# ============================================================

def ensure_dirs(cfg: Study2Config) -> None:
    """
    Ensure that the output directory exists.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


def load_data(cfg: Study2Config) -> pd.DataFrame:
    """
    Load the raw Study 2 CSV.
    """
    if not cfg.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {cfg.input_csv}")
    return pd.read_csv(cfg.input_csv)


def validate_columns(df: pd.DataFrame, cfg: Study2Config) -> None:
    """
    Validates that all required columns are present in the raw CSV.

    We distinguish:
      - Core ID and predictor columns.
      - Standalone outcome columns.
      - Component columns needed to compute composites.
    """
    required_core = [
        cfg.participant_id_col,
        cfg.artwork_id_col,
        cfg.presented_attribution_col,
        cfg.actual_origin_col,
        cfg.label_accuracy_col,
        cfg.border_condition_col,
        cfg.attitudes_col,
    ]

    standalone_outcomes: List[str] = []
    if cfg.outcomes:
        for y in cfg.outcomes:
            if cfg.composites and y in cfg.composites:
                continue
            standalone_outcomes.append(y)

    required: List[str] = list(required_core) + list(standalone_outcomes)

    # Add composite components
    if cfg.composites:
        for _, cols in cfg.composites.items():
            for col in cols:
                if col == "CuriosityRating":
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
            "Missing required columns in CSV:\n"
            + "\n".join([f"  - {c}" for c in missing])
            + "\n\nYour file must use the standardized schema (or update Study2Config)."
        )


def _relevel(categories: List[str], reference: str) -> List[str]:
    """
    Helper to re-order category levels.

    If the reference label is present, it is moved
    to the front; otherwise categories are returned unchanged.
    """
    categories = [c for c in categories if c is not None]
    if reference in categories:
        return [reference] + [c for c in categories if c != reference]
    return categories


def _to_clean_string_series(s: pd.Series) -> pd.Series:
    """
    Convert a series to a pandas StringDtype with whitespace stripped
    and missing-like values normalized to <NA>.
    """
    out = s.astype("string")
    out = out.str.strip()
    out = out.mask(out.str.lower() == "nan", pd.NA)
    out = out.mask(out == "", pd.NA)
    return out


def _clean_id_series(s: pd.Series) -> pd.Series:
    """
    Clean ID-type columns (e.g., ParticipantID, ArtworkID).
    """
    out = s.astype("string").str.strip()
    out = out.mask(out.str.lower() == "nan", pd.NA)
    out = out.mask(out == "", pd.NA)
    return out


def coerce_types(df: pd.DataFrame, cfg: Study2Config) -> pd.DataFrame:
    """
    Coerce types and set up categorical predictors for Study 2.

    - Clean ParticipantID and ArtworkID.
    - Categorical PresentedAttribution, ActualOrigin, LabelAccuracy, BorderCondition.
    - Clean ArtStyle and StudyID as strings.
    - Numeric AttitudesTowardAI with optional centering.
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

    # BorderCondition (central factor in Study 2)
    bc = _to_clean_string_series(out[cfg.border_condition_col])
    bc_cats = _relevel([x for x in bc.dropna().unique().tolist()], cfg.ref_border_condition)
    out[cfg.border_condition_col] = pd.Categorical(bc, categories=bc_cats, ordered=False)

    # Optional descriptives-only columns
    if cfg.art_style_col in out.columns:
        out[cfg.art_style_col] = _to_clean_string_series(out[cfg.art_style_col])
    if cfg.study_id_col in out.columns:
        out[cfg.study_id_col] = _to_clean_string_series(out[cfg.study_id_col])

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
    """
    Drop rows with missing values in any of the specified columns.
    """
    return df.dropna(subset=cols).copy()


# ============================================================
# 3B) Composite construction
# ============================================================

def _resolve_column(df: pd.DataFrame, preferred: str, fallbacks: List[str]) -> str:
    """
    Resolve ambiguous column names (e.g., CuriosityRating vs CuriousityRating).
    """
    if preferred in df.columns:
        return preferred
    for fb in fallbacks:
        if fb in df.columns:
            return fb
    raise KeyError(f"None of these columns were found: {[preferred] + fallbacks}")


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Safely coerce a series to numeric, turning non-numeric entries into NaN.
    """
    return pd.to_numeric(s, errors="coerce")


def add_composites(df: pd.DataFrame, cfg: Study2Config) -> pd.DataFrame:
    """
    Add composite outcome columns to the dataframe according to cfg.composites.
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
# 3C) Composite coverage reporting
# ============================================================

def composite_coverage_report(
    df_raw: pd.DataFrame,
    df_with_composites: pd.DataFrame,
    cfg: Study2Config
) -> pd.DataFrame:
    """
    Summarize coverage and missingness for each composite and component.
    """
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

def composite_component_correlations(df: pd.DataFrame, cfg: Study2Config) -> pd.DataFrame:
    """
    Compute Pearson correlations among component items for each composite.
    """
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
# 4) Descriptive statistics and outcome diagnostics
# ============================================================

def descriptives_overall(df: pd.DataFrame, outcomes: List[str]) -> pd.DataFrame:
    """
    Overall descriptive statistics for each outcome.
    """
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
    """
    Descriptives by levels of one or more grouping variables.
    """
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
    """
    Distribution-level diagnostics for each outcome.
    """
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
    Container for all relevant pieces of a single fitted LMM.
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
    """
    Extract fixed-effect estimates and Wald z-tests from a MixedLM fit.
    """
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
    """
    Extract variance components from a MixedLM fit.
    """
    rows: List[Dict[str, Any]] = []

    try:
        rows.append({"component": "participant_intercept_var", "value": float(fitted.cov_re.iloc[0, 0])})
    except Exception:
        pass

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
    Approximate Nakagawa-style pseudo-R² for a MixedLM.
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


def add_standardized_effects(
    fe: pd.DataFrame,
    dfi: pd.DataFrame,
    outcome: str
) -> pd.DataFrame:
    """
    Add DV-standardized effect sizes to the fixed-effects table.
    """
    sd_y = float(np.std(dfi[outcome].dropna(), ddof=1))
    fe2 = fe.copy()
    if sd_y > 0 and not np.isnan(sd_y):
        fe2["estimate_std_y"] = fe2["estimate"] / sd_y
        fe2["ci95_low_std_y"] = fe2["ci95_low"] / sd_y
        fe2["ci95_high_std_y"] = fe2["ci95_high"] / sd_y
    else:
        fe2["estimate_std_y"] = np.nan
        fe2["ci95_low_std_y"] = np.nan
        fe2["ci95_high_std_y"] = np.nan
    return fe2


def compute_effect_sizes_for_fixed_effects(
    fe: pd.DataFrame,
    dfi: pd.DataFrame,
    outcome: str,
    att_col_for_model: Optional[str] = None
) -> pd.DataFrame:
    """
    Add effect sizes to the fixed-effects table, including:
      - DV-standardized coefficients.
      - Standardized betas for the attitudes term(s).
    """
    fe2 = fe.copy()

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
    """
    Build a stable, unique list of columns required for a model.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for c in base_cols + rhs_cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def fit_lmm_crossed_intercepts(
    df: pd.DataFrame,
    cfg: Study2Config,
    outcome: str,
    rhs: str,
    rhs_required_cols: List[str],
    model_name: str,
    reml: bool = True
) -> LMMBundle:
    """
    Fit a linear mixed model with crossed random intercepts:
      Outcome ~ rhs
      Random effects:
        - ParticipantID intercept (group-level)
        - ArtworkID intercept (via variance component)
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
    fe = add_standardized_effects(fe, dfi, outcome)

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
# 6) Study 2 model specifications
# ============================================================

def rhs_study2_replication(cfg: Study2Config) -> str:
    """
    Replication model for H2.1:
      Outcome ~ PresentedAttribution * AttitudesTowardAI_c
    """
    pa = C_(cfg.presented_attribution_col, use_sum=cfg.use_sum_contrasts_for_factors)
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    att = Q_(att_col)
    return f"{pa} * {att}"


def rhs_study2_replication_required_cols(cfg: Study2Config) -> List[str]:
    """
    Columns required on the RHS of the replication model.
    """
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    return [cfg.presented_attribution_col, att_col]


def rhs_study2_border_interaction(cfg: Study2Config) -> str:
    """
    Border interaction model for H2.2 and H2.3:
      Outcome ~ PresentedAttribution * BorderCondition * AttitudesTowardAI_c
    """
    pa = C_(cfg.presented_attribution_col, use_sum=cfg.use_sum_contrasts_for_factors)
    bc = C_(cfg.border_condition_col, use_sum=cfg.use_sum_contrasts_for_factors)
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    att = Q_(att_col)
    return f"{pa} * {bc} * {att}"


def rhs_study2_border_required_cols(cfg: Study2Config) -> List[str]:
    """
    Columns required on the RHS of the border interaction model.
    """
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    return [cfg.presented_attribution_col, cfg.border_condition_col, att_col]


# ============================================================
# 7) Saving helpers and key term extraction
# ============================================================

def save_df(df: pd.DataFrame, path: Path) -> None:
    """
    Save a DataFrame to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_text(text: str, path: Path) -> None:
    """
    Save a string to a UTF-8 text file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract_key_terms_study2(fixed_all: pd.DataFrame, cfg: Study2Config) -> pd.DataFrame:
    """
    Extract fixed effects central to Study 2 hypotheses:

      - Terms involving PresentedAttribution.
      - Terms involving BorderCondition.
      - Terms involving AttitudesTowardAI.
    """
    if fixed_all.empty:
        return fixed_all

    term = fixed_all["term"].astype(str)
    pa_pat = f'C(Q("{cfg.presented_attribution_col}"))'
    bc_pat = f'C(Q("{cfg.border_condition_col}"))'
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    att_pat = f'Q("{att_col}")'

    keep = (
        term.str.contains(pa_pat, regex=False)
        | term.str.contains(bc_pat, regex=False)
        | term.str.contains(att_pat, regex=False)
    )

    return fixed_all.loc[keep].copy()


# ============================================================
# 8) Diagnostics: residuals and influence
# ============================================================

def make_residual_plots(
    bundle: LMMBundle,
    cfg: Study2Config,
    diagnostics_dir: Path
) -> Dict[str, Any]:
    """
    Generate residual diagnostics for a fitted LMM.
    """
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

    fig, ax = plt.subplots()
    ax.scatter(res_df["fitted"], res_df["residual"], alpha=0.5)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals vs Fitted: {bundle.outcome} ({bundle.model_name})")
    fig.tight_layout()
    fig.savefig(diagnostics_dir / "residuals_vs_fitted.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    stats.probplot(res_df["residual"], dist="norm", plot=ax)
    ax.set_title(f"Normal Q–Q Plot: {bundle.outcome} ({bundle.model_name})")
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
    cfg: Study2Config,
    diagnostics_dir: Path
) -> None:
    """
    Approximate influence diagnostics using an OLS model with the same
    fixed-effects formula as the mixed model.
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
    cfg: Study2Config
) -> pd.DataFrame:
    """
    For each fitted LMM, generate residual plots and influence diagnostics,
    aggregating residual summaries into one DataFrame.
    """
    summaries: List[Dict[str, Any]] = []

    for b in bundles:
        diag_dir = cfg.output_dir / "diagnostics" / f"{b.outcome}__{b.model_name}"
        try:
            summary = make_residual_plots(b, cfg, diag_dir)
            summaries.append(summary)
        except Exception:
            err_text = traceback.format_exc()
            save_text(
                err_text,
                diag_dir / "residual_diagnostics_ERROR.txt",
            )

        try:
            compute_influence_ols_approx(b, cfg, diag_dir)
        except Exception:
            err_text = traceback.format_exc()
            save_text(
                err_text,
                diag_dir / "influence_ols_approx_ERROR.txt",
            )

    return pd.DataFrame(summaries) if summaries else pd.DataFrame()


# ============================================================
# 9) Estimated marginal means (EMMs) for replication models
# ============================================================

def compute_emm_for_replication_models(
    bundles: List[LMMBundle],
    cfg: Study2Config
) -> pd.DataFrame:
    """
    Compute EMMs for Study 2 replication models:
      Outcome ~ PresentedAttribution * AttitudesTowardAI_c

    For each bundle with model_name == "study2_replication":
      - Evaluate at attitudes = 0 (mean), +1 SD, -1 SD (if SD > 0).
      - For each PresentedAttribution level.
    """
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    emm_rows: List[Dict[str, Any]] = []

    for b in bundles:
        if b.model_name != "study2_replication":
            continue

        dfi = b.data_used

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

        pa_series = dfi[cfg.presented_attribution_col]
        pa_levels = [lvl for lvl in pa_series.dropna().unique().tolist()]

        if not pa_levels:
            continue

        fe_params = b.fitted.fe_params
        cov_fe = b.fitted.cov_params()

        for pa_level in pa_levels:
            for lvl_label, lvl_val in levels_info:
                grid = pd.DataFrame({
                    cfg.presented_attribution_col: [pa_level],
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
                    "outcome": b.outcome,
                    "model": b.model_name,
                    "presented_attribution_level": pa_level,
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
# 10) Random effects summaries (participants and artworks)
# ============================================================

def summarize_random_effects(
    bundles: List[LMMBundle],
    cfg: Study2Config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize random effects for participants and artworks.
    """
    part_rows: List[Dict[str, Any]] = []
    art_rows: List[Dict[str, Any]] = []

    for b in bundles:
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
# 11) TOST equivalence testing for fixed effects
# ============================================================

def run_tost_equivalence(
    bundles: List[LMMBundle],
    cfg: Study2Config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform TOST equivalence tests on all fixed-effect coefficients
    from the fitted mixed models.

    For each bundle (outcome × model):
      - Compute SD_y from the model's data_used.
      - Define raw-scale equivalence bounds:
          lower = -delta * SD_y
          upper = +delta * SD_y
        where delta = cfg.tost_delta_std_y.
      - For each fixed effect (estimate, se):
          * Compute TOST z-values and one-sided p-values:
              H01: beta <= lower vs H1: beta > lower
              H02: beta >= upper vs H2: beta < upper
          * Combine into TOST p-value = max(p_low, p_high).
          * Also compute standard two-sided null p-value for beta = 0
            (using Wald z and normal approximation) for comparison.
      - Summarize results, including standardized effect sizes if available.

    Returns
    -------
    tost_all : DataFrame
        One row per fixed effect × model × outcome.
    tost_key : DataFrame
        Subset of tost_all for terms involving BorderCondition
        (used for key Study 2 border hypotheses).
    """
    all_rows: List[Dict[str, Any]] = []

    # Pattern for identifying BorderCondition terms
    bc_pat = f'C(Q("{cfg.border_condition_col}"))'

    for b in bundles:
        y = b.data_used[b.outcome].dropna()
        if y.shape[0] < 2:
            sd_y = np.nan
        else:
            sd_y = float(np.std(y, ddof=1))

        if np.isnan(sd_y) or sd_y <= 0:
            lower_raw = np.nan
            upper_raw = np.nan
        else:
            lower_raw = -cfg.tost_delta_std_y * sd_y
            upper_raw = cfg.tost_delta_std_y * sd_y

        for _, fe_row in b.fixed_effects.iterrows():
            term = str(fe_row["term"])
            est = float(fe_row["estimate"])
            se = float(fe_row["se"])

            # Default values
            z_low = z_high = p_low = p_high = p_tost = np.nan
            z_null = p_null = np.nan
            equivalent = False
            null_significant = False
            note = ""

            if (se is not None) and (se > 0) and not np.isnan(se) and not np.isnan(lower_raw) and not np.isnan(upper_raw):
                # TOST lower bound: H01: beta <= lower_raw vs H1: beta > lower_raw
                z_low = (est - lower_raw) / se
                p_low = 1.0 - stats.norm.cdf(z_low)  # one-sided

                # TOST upper bound: H02: beta >= upper_raw vs H2: beta < upper_raw
                z_high = (est - upper_raw) / se
                p_high = stats.norm.cdf(z_high)  # one-sided

                p_tost = max(p_low, p_high)
                equivalent = bool((p_low < cfg.tost_alpha) and (p_high < cfg.tost_alpha))

                # Standard null test for reference
                z_null = est / se
                p_null = 2.0 * (1.0 - stats.norm.cdf(abs(z_null)))
                null_significant = bool(p_null < cfg.tost_alpha)
            else:
                note = "Insufficient information for TOST (SD_y or SE invalid)."

            est_std_y = float(fe_row["estimate_std_y"]) if "estimate_std_y" in fe_row and pd.notna(fe_row["estimate_std_y"]) else np.nan
            ci_low_std_y = float(fe_row["ci95_low_std_y"]) if "ci95_low_std_y" in fe_row and pd.notna(fe_row["ci95_low_std_y"]) else np.nan
            ci_high_std_y = float(fe_row["ci95_high_std_y"]) if "ci95_high_std_y" in fe_row and pd.notna(fe_row["ci95_high_std_y"]) else np.nan

            equiv_ci = False
            if not np.isnan(ci_low_std_y) and not np.isnan(ci_high_std_y):
                if (ci_low_std_y > -cfg.tost_delta_std_y) and (ci_high_std_y < cfg.tost_delta_std_y):
                    equiv_ci = True

            all_rows.append({
                "outcome": b.outcome,
                "model": b.model_name,
                "term": term,
                "estimate": est,
                "se": se,
                "estimate_std_y": est_std_y,
                "ci95_low_std_y": ci_low_std_y,
                "ci95_high_std_y": ci_high_std_y,
                "sd_y": sd_y,
                "equivalence_bound_std_y": cfg.tost_delta_std_y,
                "equivalence_bound_raw_lower": lower_raw,
                "equivalence_bound_raw_upper": upper_raw,
                "tost_alpha": cfg.tost_alpha,
                "tost_z_lower": z_low,
                "tost_p_lower": p_low,
                "tost_z_upper": z_high,
                "tost_p_upper": p_high,
                "tost_p_max": p_tost,
                "tost_equivalent": bool(equivalent),
                "null_z": z_null,
                "null_p_two_sided": p_null,
                "null_significant": bool(null_significant),
                "equivalent_by_ci_within_bounds": bool(equiv_ci),
                "note": note,
            })

    tost_all = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    if tost_all.empty:
        return tost_all, tost_all

    term_series = tost_all["term"].astype(str)
    is_border_term = term_series.str.contains(bc_pat, regex=False)
    tost_key = tost_all.loc[is_border_term].copy()

    return tost_all, tost_key


# ============================================================
# 12) Main pipeline for Study 2
# ============================================================

def run_study2(cfg: Study2Config) -> None:
    """
    Orchestrate the full Study 2 analysis.
    """
    ensure_dirs(cfg)

    # --------------------------------------------------------
    # Load and validate data
    # --------------------------------------------------------
    df_raw = load_data(cfg)
    validate_columns(df_raw, cfg)

    # --------------------------------------------------------
    # Clean, coerce, and construct composites
    # --------------------------------------------------------
    df = coerce_types(df_raw, cfg)

    centered_col = df.attrs.get("attitudes_centered_col", cfg.attitudes_col)
    cfg.attitudes_col_for_model = centered_col

    df = add_composites(df, cfg)

    # --------------------------------------------------------
    # Save analysis-ready dataset
    # --------------------------------------------------------
    analysis_ready_path = cfg.output_dir / "Generaite_Study2_analysis_ready.csv"
    save_df(df, analysis_ready_path)
    save_df(df, cfg.data_dir / "Generaite_Study2_analysis_ready.csv")

    # --------------------------------------------------------
    # Cell counts
    # --------------------------------------------------------
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

    cell_counts_pab = (
        df.groupby(
            [cfg.presented_attribution_col, cfg.actual_origin_col, cfg.border_condition_col],
            dropna=False,
            observed=False,
        )
        .size()
        .reset_index(name="n")
    )
    save_df(cell_counts_pab, cfg.output_dir / "cell_counts_presented_x_origin_x_border.csv")

    # --------------------------------------------------------
    # Composite coverage and component correlations
    # --------------------------------------------------------
    cov = composite_coverage_report(df_raw, df, cfg)
    save_df(cov, cfg.output_dir / "composite_coverage_report.csv")

    comp_corr = composite_component_correlations(df, cfg)
    save_df(comp_corr, cfg.output_dir / "composite_component_correlations.csv")

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------
    meta = pd.DataFrame([{
        "input_csv": str(cfg.input_csv),
        "n_rows_raw": int(df_raw.shape[0]),
        "n_rows_after_processing": int(df.shape[0]),
        "attitudes_centered": bool(cfg.center_attitudes),
        "attitudes_center_mean": df.attrs.get("attitudes_center_mean", np.nan),
        "attitudes_col_for_model": cfg.attitudes_col_for_model,
        "use_sum_contrasts_for_factors": bool(cfg.use_sum_contrasts_for_factors),
        "run_replication_models": bool(cfg.run_replication_models),
        "run_border_interaction_models": bool(cfg.run_border_interaction_models),
        "tost_delta_std_y": float(cfg.tost_delta_std_y),
        "tost_alpha": float(cfg.tost_alpha),
        "timestamp": datetime.now().isoformat(),
    }])
    save_df(meta, cfg.output_dir / "run_metadata.csv")

    # --------------------------------------------------------
    # Descriptives and outcome distribution diagnostics
    # --------------------------------------------------------
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

    save_df(
        descriptives_by_group(df, [cfg.border_condition_col], cfg.outcomes),
        cfg.output_dir / "descriptives_by_bordercondition.csv",
    )

    save_df(
        descriptives_by_group(
            df,
            [cfg.presented_attribution_col, cfg.border_condition_col],
            cfg.outcomes,
        ),
        cfg.output_dir / "descriptives_by_presentedattribution_x_bordercondition.csv",
    )

    out_diag = outcome_distribution_diagnostics(df, cfg.outcomes)
    save_df(out_diag, cfg.output_dir / "outcome_distribution_diagnostics.csv")

    # --------------------------------------------------------
    # Model fitting
    # --------------------------------------------------------
    bundles: List[LMMBundle] = []
    model_errors: List[dict] = []

    # Replication models (H2.1)
    if cfg.run_replication_models:
        rhs_rep = rhs_study2_replication(cfg)
        rhs_rep_cols = rhs_study2_replication_required_cols(cfg)

        for outcome in cfg.outcomes:
            try:
                b = fit_lmm_crossed_intercepts(
                    df=df,
                    cfg=cfg,
                    outcome=outcome,
                    rhs=rhs_rep,
                    rhs_required_cols=rhs_rep_cols,
                    model_name="study2_replication",
                    reml=True,
                )
                bundles.append(b)
                save_text(
                    str(b.fitted.summary()),
                    cfg.output_dir / "model_summaries" / f"{outcome}__study2_replication_summary.txt",
                )
            except Exception as e:
                model_errors.append({
                    "outcome": outcome,
                    "model": "study2_replication",
                    "formula": f"{outcome} ~ {rhs_rep}",
                    "rhs_required_cols": rhs_rep_cols,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                })
                save_text(
                    traceback.format_exc(),
                    cfg.output_dir / "model_summaries" / f"{outcome}__study2_replication_ERROR.txt",
                )

    # Border interaction models (H2.2, H2.3)
    if cfg.run_border_interaction_models:
        rhs_border = rhs_study2_border_interaction(cfg)
        rhs_border_cols = rhs_study2_border_required_cols(cfg)

        for outcome in cfg.outcomes:
            try:
                b = fit_lmm_crossed_intercepts(
                    df=df,
                    cfg=cfg,
                    outcome=outcome,
                    rhs=rhs_border,
                    rhs_required_cols=rhs_border_cols,
                    model_name="study2_border_interaction",
                    reml=True,
                )
                bundles.append(b)
                save_text(
                    str(b.fitted.summary()),
                    cfg.output_dir / "model_summaries" / f"{outcome}__study2_border_interaction_summary.txt",
                )
            except Exception as e:
                model_errors.append({
                    "outcome": outcome,
                    "model": "study2_border_interaction",
                    "formula": f"{outcome} ~ {rhs_border}",
                    "rhs_required_cols": rhs_border_cols,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                })
                save_text(
                    traceback.format_exc(),
                    cfg.output_dir / "model_summaries" / f"{outcome}__study2_border_interaction_ERROR.txt",
                )

    # --------------------------------------------------------
    # Aggregate model outputs
    # --------------------------------------------------------
    fixed_all = pd.concat(
        [
            b.fixed_effects.assign(
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
        [b.fit_stats for b in bundles],
        ignore_index=True,
    ) if bundles else pd.DataFrame()

    req_rows: List[Dict[str, Any]] = []
    for b in bundles:
        req_rows.append({
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

    # Key terms table for Study 2 hypotheses
    save_df(
        extract_key_terms_study2(fixed_all, cfg),
        cfg.output_dir / "key_hypothesis_tests_study2.csv",
    )

    # --------------------------------------------------------
    # TOST equivalence testing for fixed effects (including border terms)
    # --------------------------------------------------------
    if bundles:
        tost_all_df, tost_key_df = run_tost_equivalence(bundles, cfg)
    else:
        tost_all_df, tost_key_df = pd.DataFrame(), pd.DataFrame()

    save_df(tost_all_df, cfg.output_dir / "tost_equivalence_all_fixed_effects.csv")
    save_df(tost_key_df, cfg.output_dir / "tost_equivalence_key_border_terms_study2.csv")

    # --------------------------------------------------------
    # Residual diagnostics and OLS-based influence
    # --------------------------------------------------------
    if bundles:
        resid_diag_df = collect_residual_diagnostics(bundles, cfg)
        save_df(resid_diag_df, cfg.output_dir / "lmm_residual_diagnostics.csv")
    else:
        save_df(pd.DataFrame(), cfg.output_dir / "lmm_residual_diagnostics.csv")

    # --------------------------------------------------------
    # EMMs for replication models
    # --------------------------------------------------------
    emm_df = compute_emm_for_replication_models(bundles, cfg)
    save_df(emm_df, cfg.output_dir / "emm_study2_replication.csv")

    # --------------------------------------------------------
    # Random effects summaries
    # --------------------------------------------------------
    ranef_participants_df, ranef_artworks_df = summarize_random_effects(bundles, cfg)
    save_df(ranef_participants_df, cfg.output_dir / "ranef_participants.csv")
    save_df(ranef_artworks_df, cfg.output_dir / "ranef_artworks.csv")

    # --------------------------------------------------------
    # Model errors (if any)
    # --------------------------------------------------------
    if model_errors:
        save_df(pd.DataFrame(model_errors), cfg.output_dir / "model_errors.csv")
    else:
        save_df(pd.DataFrame(), cfg.output_dir / "model_errors.csv")


def main() -> None:
    """
    Entry point for the Study 2 analysis script.
    """
    cfg = build_default_config()

    cfg.project_root.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    run_study2(cfg)
    print(f"Study 2 complete. Outputs saved to:\n  {cfg.output_dir}")


if __name__ == "__main__":
    main()
