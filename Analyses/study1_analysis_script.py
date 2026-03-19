"""
Gener-AI-te — Study 1 Analysis Script (Effect of Labelling and AI Attitudes)
Extended Version with Diagnostics, EMMs, and Random Effects Summaries

Author: Dr. Rhyse Bendell + GPT-5.1 Thinking
Date: 2026-01-26

What this script assumes
------------------------
Your Study 1 CSV is in long format (one row per ParticipantID × ArtworkID response).

This script uses the standardized column schema:

REQUIRED IDENTIFIERS
- ParticipantID
- ArtworkID

CORE ATTRIBUTION / LABELLING CONSTRUCTS
- PresentedAttribution    : {None/NoLabel, Human, AI}
    NOTE: Missing values are treated as missing data, not recoded.
- ActualOrigin            : {Human, AI}
- LabelAccuracy           : {None, Accurate, Deceptive}  (Study 1 should be None or Accurate)

VISUAL / PRESENTATION
- BorderCondition         : {Original, Swapped, Neutral} (optional, used in descriptives only)

ARTWORK DESCRIPTORS
- ArtStyle                : {Abstract, Impressionist, Baroque, Realism} (optional, used in descriptives only)

PARTICIPANT MODERATOR
- AttitudesTowardAI       : numeric
    - This script creates a centered version, AttitudesTowardAI_c, for modeling.

STUDY IDENTIFIER
- StudyID                 : {1,2,3} (optional; not used in Study 1-only models but saved if present)

OUTCOMES (DVs) — analyzed endpoints (composites + standalone sliders)
Composites:
- AestheticJudgment   = mean(CreativityRating, AestheticRating,
                             FormalExecutionRating, CuriosityRating/CuriousityRating)
- NegativeEmotion     = mean(EmotionNegHighAvg, EmotionNegLowAvg)
- PositiveEmotion     = mean(EmotionPosLowAvg, EmotionPosHighAvg)

Standalone endpoints:
- IsArtSlider
- LikeThisArtSlider

What this script does (high-level)
----------------------------------
1) Load and validate data:
   - Reads the Study 1 CSV from:
       C:\\Post-doc Work\\Gener-ai-te\\Data\\Generaite_Study1_1-21-2026.csv
   - Checks that required columns are present (ID variables, outcome components, and predictors).

2) Clean and coerce variables:
   - ParticipantID / ArtworkID:
       - Converted to string with leading/trailing whitespace removed.
       - Empty strings and "nan" (as text) are treated as missing, not as literal IDs.
   - Label variables (PresentedAttribution, ActualOrigin, LabelAccuracy):
       - Converted to pandas Categorical.
       - Missing values remain missing (no imputation).
       - Reference categories (e.g., "NoLabel", "Human") are placed first if present.
   - Optional variables (BorderCondition, ArtStyle, StudyID):
       - Cleaned to string if present.
   - AttitudesTowardAI:
       - Coerced to numeric.
       - Grand-mean centered to a new column AttitudesTowardAI_c, stored for modeling.

3) Construct composite outcomes:
   - AestheticJudgment, NegativeEmotion, PositiveEmotion:
       - Component items are coerced to numeric.
       - Composite score is row-wise mean with a conservative missing-data rule:
           * AestheticJudgment requires 4 non-missing components.
           * NegativeEmotion and PositiveEmotion require 2 non-missing components.
       - Rows with insufficient component data are set to NaN on that composite.

4) Save an analysis-ready CSV:
   - Contains:
       * Original columns
       * Cleaned/categorical predictors
       * Centered AttitudesTowardAI_c
       * Composite outcomes
   - Saved both into the run-specific results folder and into:
       C:\\Post-doc Work\\Gener-ai-te\\Data\\Generaite_Study1_analysis_ready.csv

5) Basic descriptives and coverage:
   - Cell counts for PresentedAttribution × ActualOrigin.
   - Composite coverage report (how many rows contribute valid composites, component missingness).
   - Component correlation matrices for each composite.
   - Overall descriptives for each DV.
   - Descriptives by:
       * PresentedAttribution
       * PresentedAttribution × ActualOrigin
       * LabelAccuracy
   - Outcome distribution diagnostics (mean, SD, skewness, kurtosis, normality tests).

6) Mixed-effects models (LMMs) for Study 1:
   A) Primary Study 1 model (per outcome):
        Outcome ~ PresentedAttribution * AttitudesTowardAI_c
      Random intercepts:
        (1 | ParticipantID) + (1 | ArtworkID)
      - ParticipantID random intercept is implemented via the MixedLM "groups" argument.
      - ArtworkID random intercept is implemented via a variance component (vc_formula).
      - No imputation: rows are dropped only if they lack variables required for that model.

   B) Optional unlabelled-only origin test:
      - Runs only if:
          * run_unlabelled_origin_test is True in config, AND
          * PresentedAttribution has an explicit category matching ref_presented_attr
            (e.g., "NoLabel"), AND
          * at least 10 usable rows remain for the model.
      - Subset: PresentedAttribution == ref_presented_attr
      - Model:
          Outcome ~ ActualOrigin * AttitudesTowardAI_c
      - Same random effects structure as the primary model.

   For each fitted model, we record:
      - Fixed effects (coefficients, SEs, Wald z-tests, p-values, 95% CIs).
      - DV-standardized effect sizes (estimate / SD of the DV).
      - Standardized betas for AttitudesTowardAI terms.
      - Variance components (participant, artwork, residual).
      - Approximate Nakagawa pseudo-R² (marginal and conditional).
      - Fit diagnostics (AIC, BIC, log-likelihood, convergence flag).

7) Extended diagnostics (added per your request):
   A) Outcome distribution diagnostics:
      - For each outcome:
          * N, mean, SD, min, max, skewness, kurtosis, Shapiro–Wilk p-value (when feasible).
      - Saved as outcome_distribution_diagnostics.csv.

   B) Model-level residual diagnostics:
      For each (Outcome, Model) combination:
        - Compute model residuals and fitted values.
        - Save residual vs fitted data as CSV per model.
        - Produce:
            * Residual vs Fitted scatterplot (PNG).
            * Q–Q plot of residuals vs normal (PNG).
        - Summarize residual distribution:
            * Mean, SD, skewness, kurtosis.
            * Correlation between empirical and theoretical normal quantiles.
            * Shapiro–Wilk and Jarque–Bera tests (where feasible).
      - Combined summary saved as lmm_residual_diagnostics.csv.

   C) Influence / leverage diagnostics (OLS-based approximation):
      - MixedLM does not provide direct influence measures.
      - As an approximation, we fit an OLS model with the same fixed-effects formula
        on the data used by each LMM and compute:
            * Leverage (hat matrix diagonal).
            * Cook’s distance.
            * Studentized residuals.
      - Results saved per (Outcome, Model) as:
            diagnostics/<outcome>__<model>/influence_ols_approx.csv
      - Top influential observations (by Cook’s distance) are easy to identify from this file.

8) Estimated marginal means (EMMs) for the primary Study 1 model:
   - For each outcome, from the primary model:
       * We compute EMMs for:
            - Each PresentedAttribution level (e.g., Human, AI, NoLabel).
            - At selected AttitudesTowardAI_c values:
                - Mean (0, after centering).
                - +1 SD of AttitudesTowardAI_c (if SD > 0).
                - -1 SD of AttitudesTowardAI_c (if SD > 0).
       * EMMs are based on the fixed-effects parameter estimates and covariance matrix,
         using a Patsy design matrix to ensure the parameterization matches the LMM.
       * For each condition, we report:
            - Predicted mean.
            - Standard error of the prediction.
            - 95% confidence interval.
   - All EMMs aggregated and saved as:
       emm_study1_primary.csv

9) Random effects summaries:
   - Participant-level random effects (BLUPs):
       * For each model and outcome, we extract the participant-specific
         random intercepts (conditional modes).
       * These are the deviations from the population intercept for each ParticipantID.
       * Saved as:
           ranef_participants.csv
         with columns:
           [outcome, model, participant_id, ranef_intercept]

   - Artwork-level random-effect proxies:
       * Because the artwork-specific random intercepts are specified via a
         variance-component structure (vc_formula) rather than as the primary
         grouping factor, statsmodels does not expose simple per-artwork BLUPs.
       * As a practical and transparent approximation, we compute, for each
         (Outcome, Model):
             - The mean model residual per ArtworkID.
             - The number of observations per ArtworkID.
             - The SD of residuals per ArtworkID.
       * These "artwork residual means" are not exact BLUPs, but they provide a
         reasonable empirical summary of artwork-level deviations after accounting
         for fixed effects and participant random intercepts.
       * Saved as:
           ranef_artworks.csv
         with columns:
           [outcome, model, artwork_id, n_obs, mean_resid, sd_resid]

10) Output structure
--------------------
All outputs for a given run are written to:

  C:\\Post-doc Work\\Gener-ai-te\\Results\\Study1\\run_YYYYMMDD_HHMMSS\\

Key files:
- run_metadata.csv
- Generaite_Study1_analysis_ready.csv     (also replicated into Data\\)
- composite_coverage_report.csv
- composite_component_correlations.csv
- cell_counts_presented_x_origin.csv
- descriptives_overall.csv
- descriptives_by_presentedattribution.csv
- descriptives_by_presentedattribution_x_origin.csv
- descriptives_by_labelaccuracy.csv
- outcome_distribution_diagnostics.csv
- lmm_fixed_effects_all.csv
- lmm_variance_components_all.csv
- lmm_fit_statistics_all.csv
- lmm_required_columns_by_model.csv
- lmm_residual_diagnostics.csv
- emm_study1_primary.csv
- ranef_participants.csv
- ranef_artworks.csv
- model_summaries/*.txt
- diagnostics/<outcome>__<model>/*.png, *.csv
- model_errors.csv (if any)

Inference notes
---------------
- MixedLM fixed-effect p-values are Wald z-tests (normal approximation).
- No imputation is performed: models use listwise deletion on the variables
  required for that specific model.
- Pseudo R² values are approximate and should be interpreted as descriptive
  summaries rather than exact analogues of OLS R².
- Influence measures are based on a parallel OLS model and should be considered
  diagnostic tools rather than definitive for the mixed model.

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
      for categorical variables (typically treatment/dummy coding).

    - If use_sum is True, we wrap with Sum contrasts, which are closer
      to the "sum-to-zero" contrasts often used in JASP's Type III tests.

    This affects the parameterization of the model (which coefficients
    appear and how they are interpreted), but not the underlying fitted
    cell means.
    """
    if use_sum:
        return f"C({Q_(colname)}, Sum)"
    return f"C({Q_(colname)})"


# ============================================================
# 2) Configuration dataclass
# ============================================================

@dataclass
class Study1Config:
    """
    Configuration object for Study 1 analysis.

    Centralizes:
      - File paths
      - Column names
      - Composite outcome definitions
      - Reference category choices
      - Modeling options (e.g., contrast coding)
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

    border_condition_col: str = "BorderCondition"   # optional
    art_style_col: str = "ArtStyle"                 # optional
    study_id_col: str = "StudyID"                   # optional

    attitudes_col: str = "AttitudesTowardAI"        # raw numeric attitudes
    attitudes_centered_suffix: str = "_c"           # new suffix for centered column

    # Outcomes
    outcomes: List[str] = None

    # Composite definitions and missing-data rules
    composites: Dict[str, List[str]] = None
    composite_min_nonmissing: Dict[str, int] = None

    # Reference labels (used for ordering categories, not for imputation)
    ref_presented_attr: str = "NoLabel"   # unlabelled reference, if present
    ref_actual_origin: str = "Human"
    ref_label_accuracy: str = "NoLabel"

    # Whether to center AttitudesTowardAI
    center_attitudes: bool = True

    # Optional sum contrasts for factors (aligned with JASP if desired)
    use_sum_contrasts_for_factors: bool = False

    # Whether to run the optional unlabelled-only origin test
    run_unlabelled_origin_test: bool = True


def build_default_config() -> Study1Config:
    """
    Build the default configuration for Study 1.

    This function encodes all hard-coded paths and composite definitions
    so they are not scattered throughout the script.
    """
    project_root = Path(r"C:\Post-doc Work\Gener-ai-te")
    data_dir = project_root / "Data"
    input_csv = data_dir / "Generaite_Study1_1-21-2026.csv"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "Results" / "Study1" / f"run_{stamp}"

    cfg = Study1Config(
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
                # CuriosityRating is misspelled in some files as CuriousityRating;
                # we resolve that discrepancy later.
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
        # Conservative missingness rules: require all components for each composite.
        composite_min_nonmissing={
            "AestheticJudgment": 4,
            "NegativeEmotion": 2,
            "PositiveEmotion": 2,
        },
        use_sum_contrasts_for_factors=False,
        run_unlabelled_origin_test=True,
    )
    return cfg


# ============================================================
# 3) IO, validation, and coercion
# ============================================================

def ensure_dirs(cfg: Study1Config) -> None:
    """
    Ensure that the output directory exists.

    This function is separated so we can centralize any directory-related
    behavior (e.g., cleaning old runs or logging).
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


def load_data(cfg: Study1Config) -> pd.DataFrame:
    """
    Load the raw Study 1 CSV.

    This is intentionally simple: all schema checks and coercions are
    done in later functions.
    """
    if not cfg.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {cfg.input_csv}")
    return pd.read_csv(cfg.input_csv)


def validate_columns(df: pd.DataFrame, cfg: Study1Config) -> None:
    """
    Validates that all required columns are present in the raw CSV.

    We distinguish:
      - Core ID and predictor columns.
      - Standalone outcome columns.
      - Component columns needed to compute composites.

    If any are missing, we fail fast with a helpful error message.
    """
    required_core = [
        cfg.participant_id_col,
        cfg.artwork_id_col,
        cfg.presented_attribution_col,
        cfg.actual_origin_col,
        cfg.label_accuracy_col,
        cfg.attitudes_col,
    ]

    # Standalone outcomes that must exist in the raw CSV
    standalone_outcomes: List[str] = []
    if cfg.outcomes:
        for y in cfg.outcomes:
            if cfg.composites and y in cfg.composites:
                # This outcome will be derived as a composite.
                continue
            standalone_outcomes.append(y)

    required: List[str] = list(required_core) + list(standalone_outcomes)

    # Add required component columns for each composite
    if cfg.composites:
        for _, cols in cfg.composites.items():
            for col in cols:
                if col == "CuriosityRating":
                    # We accept either CuriosityRating or CuriousityRating;
                    # we check that at least one exists.
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
            + "\n\nYour file must use the standardized schema (or update Study1Config)."
        )


def _relevel(categories: List[str], reference: str) -> List[str]:
    """
    Helper to re-order category levels.

    If the reference label is present in the category list, it is moved
    to the front; otherwise the categories are returned unchanged.
    """
    categories = [c for c in categories if c is not None]
    if reference in categories:
        return [reference] + [c for c in categories if c != reference]
    return categories


def _to_clean_string_series(s: pd.Series) -> pd.Series:
    """
    Convert a series to a pandas StringDtype with whitespace stripped
    and missing-like values normalized to <NA>.

    Rules:
      - Leading/trailing whitespace is removed.
      - Literal "nan" (case-insensitive) is treated as missing.
      - Empty strings are treated as missing.
    """
    out = s.astype("string")
    out = out.str.strip()
    out = out.mask(out.str.lower() == "nan", pd.NA)
    out = out.mask(out == "", pd.NA)
    return out


def _clean_id_series(s: pd.Series) -> pd.Series:
    """
    Clean ID-type columns (e.g., ParticipantID, ArtworkID).

    We treat both "nan" and empty strings as missing, but we keep true
    missing values as <NA> rather than converting them into literal
    "nan" string IDs.
    """
    out = s.astype("string").str.strip()
    out = out.mask(out.str.lower() == "nan", pd.NA)
    out = out.mask(out == "", pd.NA)
    return out


def coerce_types(df: pd.DataFrame, cfg: Study1Config) -> pd.DataFrame:
    """
    Coerce types and set up categorical predictors.

    This function:
      - Cleans ParticipantID and ArtworkID.
      - Converts PresentedAttribution, ActualOrigin, LabelAccuracy into
        Categoricals (without imputing missing values).
      - Cleans optional columns (BorderCondition, ArtStyle, StudyID).
      - Converts AttitudesTowardAI to numeric and, optionally, creates
        a centered version (AttitudesTowardAI_c).
    """
    out = df.copy()

    # IDs (missing-aware; avoid literal "nan" IDs)
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

    # Optional categorical-like columns (kept as cleaned strings)
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
    """
    Drop rows with missing values in any of the specified columns.

    This is used in a model-specific way so that each model only loses
    rows that lack the variables it actually needs.
    """
    return df.dropna(subset=cols).copy()


# ============================================================
# 3B) Composite construction
# ============================================================

def _resolve_column(df: pd.DataFrame, preferred: str, fallbacks: List[str]) -> str:
    """
    Helper to resolve ambiguous column names.

    If the preferred column is present, we use it.
    Otherwise we try each fallback in order.
    If none exist, we raise an error.
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


def add_composites(df: pd.DataFrame, cfg: Study1Config) -> pd.DataFrame:
    """
    Add composite outcome columns to the dataframe according to cfg.composites.

    Missingness policy:
      - For each composite, we compute the row-wise mean across its components.
      - We then enforce a minimum number of non-missing components
        (cfg.composite_min_nonmissing[comp_name]).
      - Rows that do not meet this threshold are set to NaN on that composite.

    This is deliberately conservative and avoids "smearing" composites across
    different sets of contributing items.
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

        # Ensure component columns exist and are numeric
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
    cfg: Study1Config
) -> pd.DataFrame:
    """
    Summarize coverage and missingness for each composite.

    For each composite and each component:
      - How many rows have non-missing composite values?
      - How many rows are missing that component?
      - What is the composite's mean, SD, min, max?
      - What is the required minimum number of non-missing components?

    This makes it transparent how strict the composite rules are in practice.
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

def composite_component_correlations(df: pd.DataFrame, cfg: Study1Config) -> pd.DataFrame:
    """
    Compute Pearson correlations among component items for each composite.

    This is a quick psychometric check: do the items that make up a composite
    actually correlate in the expected direction and magnitude?

    Returns a tidy DataFrame with:
      - composite : composite name
      - var1, var2: pair of component variable names
      - n         : pairwise N
      - r         : Pearson correlation coefficient
      - p_value   : two-sided p-value (if n >= 3)
    """
    if not cfg.composites:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for comp_name, comp_cols in cfg.composites.items():
        # Resolve any ambiguous columns (e.g., CuriosityRating vs CuriousityRating)
        resolved_cols: List[str] = []
        for col in comp_cols:
            if col == "CuriosityRating":
                resolved_cols.append(_resolve_column(df, "CuriosityRating", ["CuriousityRating"]))
            else:
                resolved_cols.append(col)

        # Ensure numeric
        sub = df[resolved_cols].apply(pd.to_numeric, errors="coerce")

        # Pairwise correlations
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
    Compute simple descriptive statistics for each outcome:
      - N, mean, SD, SE, 95% CI.

    This is a straightforward summary of the observed distributions.
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
    Compute descriptive statistics for each outcome within levels of one or more grouping variables.

    For each outcome y and grouping combination g:
      - n, mean, SD, SE, 95% CI

    Missing group levels are preserved (dropna=False), and we set observed=False
    so that pandas does not silently drop unused categorical combinations.
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
    Compute distribution-level diagnostics for each outcome.

    For each outcome:
      - n, mean, sd, min, max
      - skewness (Fisher, bias-corrected)
      - kurtosis (Fisher, bias-corrected; 0 for normal)
      - Shapiro–Wilk p-value (for moderate n), else NaN
      - Jarque–Bera p-value (omnibus normality test)

    This helps identify whether outcomes are strongly skewed or heavy-tailed,
    which may motivate transformations or robust modeling in future work.
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

        # Shapiro–Wilk is accurate for n up to ~5000; we skip for very large samples.
        if 3 <= n <= 5000:
            try:
                shapiro_p = float(stats.shapiro(s)[1])
            except Exception:
                shapiro_p = np.nan
        else:
            shapiro_p = np.nan

        # Jarque–Bera omnibus normality test.
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

    We keep:
      - Basic identifiers (outcome, model name, formula string).
      - Required columns and data subset used to fit this model.
      - The fitted statsmodels MixedLMResults object.
      - Derived tables:
          * fixed_effects
          * variance_components
          * fit_stats
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

    For each term:
      - estimate
      - standard error
      - Wald z (estimate / SE)
      - two-sided p-value (normal approximation)
      - 95% CI
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

    We summarize:
      - Participant intercept variance
      - Artwork intercept variance (from vc_formula) if available
      - Residual variance
    """
    rows: List[Dict[str, Any]] = []

    # Participant intercept variance
    try:
        rows.append({"component": "participant_intercept_var", "value": float(fitted.cov_re.iloc[0, 0])})
    except Exception:
        pass

    # Artwork variance component (first entry in vcomp corresponds to vc_formula order)
    if getattr(fitted, "vcomp", None) is not None and len(fitted.vcomp) > 0:
        try:
            rows.append({"component": "artwork_intercept_var", "value": float(fitted.vcomp[0])})
        except Exception:
            pass

    # Residual variance
    rows.append({"component": "residual_var", "value": float(fitted.scale)})
    return pd.DataFrame(rows)


def nakagawa_pseudo_r2(
    fitted: sm.regression.mixed_linear_model.MixedLMResults
) -> Tuple[Optional[float], Optional[float]]:
    """
    Approximate Nakagawa-style pseudo-R² for a MixedLM:

      - marginal: variance explained by fixed effects.
      - conditional: variance explained by fixed + random effects.

    Implementation:
      - We approximate the variance of the fixed-effect linear predictor,
        then combine with the random-effect and residual components.
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

    For each coefficient:
      - estimate_std_y = estimate / SD(Y)
      - ci95_low_std_y = ci95_low / SD(Y)
      - ci95_high_std_y = ci95_high / SD(Y)

    This allows comparison of effect magnitudes across different outcomes.
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

      1) DV-standardized coefficients (estimate_std_y, etc.).
      2) Standardized betas for the AttitudesTowardAI term(s):
           beta_std_xy = estimate * SD(X) / SD(Y)

    Only coefficients whose term names mention Q("AttitudesTowardAI_c")
    (or the configured attitudes_col_for_model) receive beta_std_xy entries.
    """
    fe2 = fe.copy()

    # --- DV-standardization ---
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

    # --- Standardized betas for attitudes term(s) ---
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
    Helper to build a stable, unique list of columns required for a model.

    We combine:
      - base_cols: [outcome, ParticipantID, ArtworkID]
      - rhs_cols : predictors required for the right-hand side

    and ensure no duplicates, preserving first occurrence order.
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
    cfg: Study1Config,
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
        - ParticipantID intercept (group-level random intercept)
        - ArtworkID intercept (via variance component)

    Implementation details:
      - We first determine which columns are required for this model.
      - We drop rows with missing values on those columns.
      - We fit statsmodels MixedLM with:
          groups    = ParticipantID
          vc_formula= {"artwork": 0 + C(ArtworkID)}
      - We then extract fixed effects, variance components, and fit statistics.
    """
    base_cols = [outcome, cfg.participant_id_col, cfg.artwork_id_col]
    needed = _required_cols_from_rhs(rhs_required_cols, base_cols)
    dfi = drop_missing(df, needed)

    formula = f"{Q_(outcome)} ~ {rhs}"

    # Crossed random intercept for ArtworkID via variance components
    vc = {"artwork": f"0 + C({Q_(cfg.artwork_id_col)})"}

    model = smf.mixedlm(
        formula=formula,
        data=dfi,
        groups=dfi[cfg.participant_id_col],
        vc_formula=vc,
        re_formula="1",  # intercept-only for participant
    )

    fitted = model.fit(reml=reml, method="lbfgs")

    fe = extract_fixed_effects(fitted)

    # Add standardized effect sizes based on the data used in this model
    fe = add_standardized_effects(fe, dfi, outcome)

    # Also compute standardized betas for the attitudes term(s)
    att_col_for_model = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    fe = compute_effect_sizes_for_fixed_effects(
        fe=fe,
        dfi=dfi,
        outcome=outcome,
        att_col_for_model=att_col_for_model,
    )

    vc_tab = extract_variance_components(fitted)
    r2m, r2c = nakagawa_pseudo_r2(fitted)

    # Fit statistics (note: REML-based AIC/BIC should not be used to compare models with different fixed effects)
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
# 6) Study 1 model specifications
# ============================================================

def rhs_study1_primary(cfg: Study1Config) -> str:
    """
    Primary Study 1 fixed-effects specification:

      Outcome ~ PresentedAttribution * AttitudesTowardAI_c

    Rationale:
      - In Study 1 with accurate labels, PresentedAttribution is structurally
        confounded with ActualOrigin for labelled trials.
      - To avoid collinearity and over-interpretation, we do not include
        ActualOrigin in the primary model.
    """
    pa = C_(cfg.presented_attribution_col, use_sum=cfg.use_sum_contrasts_for_factors)
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    att = Q_(att_col)
    return f"{pa} * {att}"


def rhs_study1_primary_required_cols(cfg: Study1Config) -> List[str]:
    """
    Columns required on the RHS of the primary model.
    """
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    return [cfg.presented_attribution_col, att_col]


def rhs_study1_unlabelled_origin_test(cfg: Study1Config) -> str:
    """
    Fixed-effects specification for the optional unlabelled baseline model:

      Outcome ~ ActualOrigin * AttitudesTowardAI_c

    This model is estimated only on explicitly unlabelled trials
    (PresentedAttribution == ref_presented_attr), assuming the data
    include such rows.
    """
    ao = C_(cfg.actual_origin_col, use_sum=cfg.use_sum_contrasts_for_factors)
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    att = Q_(att_col)
    return f"{ao} * {att}"


def rhs_study1_unlabelled_required_cols(cfg: Study1Config) -> List[str]:
    """
    Columns required on the RHS of the unlabelled-origin model.
    """
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    return [cfg.actual_origin_col, att_col]


# ============================================================
# 7) Saving helpers and key term extraction
# ============================================================

def save_df(df: pd.DataFrame, path: Path) -> None:
    """
    Save a DataFrame to CSV, creating parent directories if necessary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_text(text: str, path: Path) -> None:
    """
    Save a string to a UTF-8 text file, creating parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract_key_terms_study1(fixed_all: pd.DataFrame, cfg: Study1Config) -> pd.DataFrame:
    """
    Extract fixed effects that are central to the Study 1 hypotheses:

      - Terms involving PresentedAttribution.
      - Terms involving ActualOrigin (for the unlabelled model).
      - Terms involving AttitudesTowardAI.

    This creates a compact table that is easier to reference in manuscripts.
    """
    if fixed_all.empty:
        return fixed_all

    term = fixed_all["term"].astype(str)
    pa_pat = f'C(Q("{cfg.presented_attribution_col}"))'
    ao_pat = f'C(Q("{cfg.actual_origin_col}"))'
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    att_pat = f'Q("{att_col}")'

    keep = (
        term.str.contains(pa_pat, regex=False)
        | term.str.contains(ao_pat, regex=False)
        | term.str.contains(att_pat, regex=False)
    )

    return fixed_all.loc[keep].copy()


# ============================================================
# 8) Diagnostics: residuals and influence
# ============================================================

def make_residual_plots(
    bundle: LMMBundle,
    cfg: Study1Config,
    diagnostics_dir: Path
) -> Dict[str, Any]:
    """
    Generate residual diagnostics for a fitted LMM:

      - Residual vs Fitted scatterplot (PNG).
      - Q–Q plot of residuals vs normal (PNG).
      - CSV of residuals and fitted values per observation.
      - Summary statistics for residuals (returned as a dictionary).

    The returned summary is later aggregated into lmm_residual_diagnostics.csv.
    """
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    fitted_vals = bundle.fitted.fittedvalues
    residuals = bundle.fitted.resid
    dfi = bundle.data_used

    # Combine residuals with IDs for easy debugging and plotting.
    res_df = pd.DataFrame({
        "row_index": dfi.index,
        "participant_id": dfi[cfg.participant_id_col].values,
        "artwork_id": dfi[cfg.artwork_id_col].values,
        "fitted": np.asarray(fitted_vals),
        "residual": np.asarray(residuals),
    })
    save_df(res_df, diagnostics_dir / "residuals.csv")

    # Residual vs fitted plot
    fig, ax = plt.subplots()
    ax.scatter(res_df["fitted"], res_df["residual"], alpha=0.5)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals vs Fitted: {bundle.outcome} ({bundle.model_name})")
    fig.tight_layout()
    fig.savefig(diagnostics_dir / "residuals_vs_fitted.png", dpi=150)
    plt.close(fig)

    # Q–Q plot of residuals
    fig, ax = plt.subplots()
    stats.probplot(res_df["residual"], dist="norm", plot=ax)
    ax.set_title(f"Normal Q–Q Plot: {bundle.outcome} ({bundle.model_name})")
    fig.tight_layout()
    fig.savefig(diagnostics_dir / "qqplot_residuals.png", dpi=150)
    plt.close(fig)

    # Summary diagnostics for residuals
    r = res_df["residual"].dropna()
    n = int(r.shape[0])
    if n > 0:
        r_mean = float(r.mean())
        r_sd = float(r.std(ddof=1)) if n > 1 else np.nan
        r_skew = float(stats.skew(r, bias=False)) if n > 2 else np.nan
        r_kurt = float(stats.kurtosis(r, fisher=True, bias=False)) if n > 3 else np.nan

        # Correlation between sorted residuals and theoretical normal quantiles
        r_sorted = np.sort(r.values)
        probs = (np.arange(1, n + 1) - 0.5) / n
        q_theor = stats.norm.ppf(probs)
        if n > 1:
            corr_qq = float(np.corrcoef(r_sorted, q_theor)[0, 1])
        else:
            corr_qq = np.nan

        # Shapiro–Wilk (avoid extremely large n for stability)
        if 3 <= n <= 5000:
            try:
                shapiro_p = float(stats.shapiro(r)[1])
            except Exception:
                shapiro_p = np.nan
        else:
            shapiro_p = np.nan

        # Jarque–Bera
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
    cfg: Study1Config,
    diagnostics_dir: Path
) -> None:
    """
    Approximate influence diagnostics using an OLS model with the same
    fixed-effects formula as the mixed model.

    Rationale:
      - statsmodels MixedLM does not currently provide direct influence
        measures (e.g., leverage, Cook's distance).
      - As a pragmatic diagnostic, we fit an OLS model with the same
        fixed-effects formula and compute influence metrics from it.

    Outputs:
      - influence_ols_approx.csv in diagnostics_dir with:
          * row_index, participant_id, artwork_id
          * leverage (hat matrix diagonal)
          * Cook's distance
          * studentized residuals
    """
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    dfi = bundle.data_used
    outcome = bundle.outcome

    # Construct an OLS formula using the same RHS as the mixed model
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
        # If anything goes wrong, we record an error file but do not crash the run.
        err_text = traceback.format_exc()
        save_text(err_text, diagnostics_dir / "influence_ols_approx_ERROR.txt")


def collect_residual_diagnostics(
    bundles: List[LMMBundle],
    cfg: Study1Config
) -> pd.DataFrame:
    """
    For each fitted LMM, generate residual plots and collect summary
    diagnostics into a single DataFrame.

    This function:
      - Loops over all bundles.
      - Calls make_residual_plots and compute_influence_ols_approx.
      - Aggregates the returned residual summaries.
    """
    summaries: List[Dict[str, Any]] = []

    for b in bundles:
        diag_dir = cfg.output_dir / "diagnostics" / f"{b.outcome}__{b.model_name}"
        try:
            summary = make_residual_plots(b, cfg, diag_dir)
            summaries.append(summary)
        except Exception:
            # If diagnostic plotting fails, we log the error but keep the run alive.
            err_text = traceback.format_exc()
            save_text(
                err_text,
                diag_dir / "residual_diagnostics_ERROR.txt",
            )

        # OLS-based influence diagnostics (best-effort)
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
# 9) Estimated marginal means (EMMs) for primary models
# ============================================================

def compute_emm_for_primary_models(
    bundles: List[LMMBundle],
    cfg: Study1Config
) -> pd.DataFrame:
    """
    Compute estimated marginal means (EMMs) for the primary Study 1 models.

    For each bundle with model_name == "study1_primary":
      - We identify the centered attitudes column used in the model.
      - We compute SD of that column in the model's data.
      - We define attitudes levels:
          * mean (0 after centering)
          * +1 SD (if SD > 0)
          * -1 SD (if SD > 0)
      - For each PresentedAttribution level and each attitudes level,
        we build a small "prediction grid" and construct a Patsy design
        matrix for the fixed-effects RHS:
          X_new = dmatrix("1 + " + rhs, grid)
      - We then compute:
          * Predicted mean = X_new @ fe_params
          * SE(pred)       = sqrt(diag(X_new @ cov_fe @ X_new^T))
          * 95% CI

    This yields a tidy EMM table with one row per
      (outcome, model, PresentedAttribution level, attitudes level).
    """
    att_col = getattr(cfg, "attitudes_col_for_model", cfg.attitudes_col)
    emm_rows: List[Dict[str, Any]] = []

    for b in bundles:
        if b.model_name != "study1_primary":
            continue

        dfi = b.data_used

        # Skip if the attitudes column is missing in this model's data
        if att_col not in dfi.columns:
            continue

        # Compute SD of attitudes for defining +/- 1 SD levels
        x = dfi[att_col].dropna()
        if x.shape[0] < 2:
            sd_att = np.nan
        else:
            sd_att = float(np.std(x, ddof=1))

        # Attitudes levels to evaluate
        levels_info: List[Tuple[str, float]] = [("mean", 0.0)]
        if sd_att and not np.isnan(sd_att) and sd_att > 0:
            levels_info.append(("plus1sd", sd_att))
            levels_info.append(("minus1sd", -sd_att))

        # Get PresentedAttribution levels actually present in this model's data
        pa_series = dfi[cfg.presented_attribution_col]
        pa_levels = [lvl for lvl in pa_series.dropna().unique().tolist()]

        if not pa_levels:
            continue

        fe_params = b.fitted.fe_params
        cov_fe = b.fitted.cov_params()

        for pa_level in pa_levels:
            for lvl_label, lvl_val in levels_info:
                # Prediction grid: one row for this combination
                grid = pd.DataFrame({
                    cfg.presented_attribution_col: [pa_level],
                    att_col: [lvl_val],
                })

                # Build design matrix for fixed effects: we prepend "1 + " to include intercept
                # We use the same RHS string as the mixed model to ensure compatibility.
                try:
                    X_new = dmatrix("1 + " + b.rhs, grid, return_type="dataframe")
                except Exception:
                    # If Patsy fails, skip this combination
                    continue

                # Align design matrix columns with available fixed-effect parameters
                common_cols = [c for c in X_new.columns if c in fe_params.index]
                if not common_cols:
                    continue

                X_use = X_new[common_cols]
                fe_use = fe_params[common_cols]

                # Align covariance matrix
                try:
                    cov_use = cov_fe.loc[common_cols, common_cols]
                except Exception:
                    # If covariance alignment fails, skip SE and CI, but still compute point estimate
                    cov_use = None

                # Predicted mean
                y_hat = float(np.dot(X_use.values, fe_use.values)[0])

                # Standard error and CI
                if cov_use is not None:
                    # This product yields a 1x1 matrix; extract the scalar safely.
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
    cfg: Study1Config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize random effects for participants and artworks.

    Participants:
      - We use MixedLMResults.random_effects, which returns a dictionary
        mapping group labels (ParticipantID) to their random-effect vector.
      - In this model, we have a random intercept per participant, so we
        extract the first element of each vector as the participant's
        BLUP intercept.

    Artworks:
      - Artwork random intercepts are specified via vc_formula, which
        does not expose simple per-artwork BLUPs in statsmodels.
      - As a practical approximation, we compute mean residual per
        ArtworkID for each (outcome, model), along with the number of
        observations and residual SD for that artwork.
      - These "artwork residual means" are not exact BLUPs, but they do
        provide a useful empirical summary of artwork-level deviations.

    Returns:
      - ranef_participants_df: tidy table of participant random intercepts.
      - ranef_artworks_df   : tidy table of artwork residual summaries.
    """
    part_rows: List[Dict[str, Any]] = []
    art_rows: List[Dict[str, Any]] = []

    for b in bundles:
        # Participant-level BLUPs
        try:
            re_dict = b.fitted.random_effects
            for pid, re_vec in re_dict.items():
                # Each re_vec is typically an array of length 1 for intercept-only
                if isinstance(re_vec, (np.ndarray, list, tuple)):
                    val = float(re_vec[0]) if len(re_vec) > 0 else np.nan
                else:
                    # If statsmodels changes representation, we attempt to coerce
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
            # Log error but do not fail the run
            err_text = traceback.format_exc()
            save_text(
                err_text,
                cfg.output_dir / "diagnostics" / f"{b.outcome}__{b.model_name}" / "ranef_participants_ERROR.txt",
            )

        # Artwork-level residual summaries (proxy for random effects)
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
# 11) Main pipeline for Study 1
# ============================================================

def run_study1(cfg: Study1Config) -> None:
    """
    Orchestrate the full Study 1 analysis:

      1) Load and validate raw data.
      2) Coerce types and build composites.
      3) Save analysis-ready dataset.
      4) Descriptives and outcome diagnostics.
      5) Fit primary and optional unlabelled-origin models.
      6) Generate residual and influence diagnostics.
      7) Compute EMMs for primary models.
      8) Summarize random effects.
      9) Save all outputs as CSVs and text summaries.
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

    # Determine which attitudes column the model will use
    centered_col = df.attrs.get("attitudes_centered_col", cfg.attitudes_col)
    cfg.attitudes_col_for_model = centered_col

    # Add composite outcomes
    df = add_composites(df, cfg)

    # --------------------------------------------------------
    # Save analysis-ready dataset
    # --------------------------------------------------------
    analysis_ready_path = cfg.output_dir / "Generaite_Study1_analysis_ready.csv"
    save_df(df, analysis_ready_path)
    # Also store a copy in the Data directory for downstream scripts
    save_df(df, cfg.data_dir / "Generaite_Study1_analysis_ready.csv")

    # --------------------------------------------------------
    # Cell counts for PresentedAttribution × ActualOrigin
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

    # --------------------------------------------------------
    # Composite coverage and component correlations
    # --------------------------------------------------------
    cov = composite_coverage_report(df_raw, df, cfg)
    save_df(cov, cfg.output_dir / "composite_coverage_report.csv")

    comp_corr = composite_component_correlations(df, cfg)
    save_df(comp_corr, cfg.output_dir / "composite_component_correlations.csv")

    # --------------------------------------------------------
    # Metadata for reproducibility
    # --------------------------------------------------------
    meta = pd.DataFrame([{
        "input_csv": str(cfg.input_csv),
        "n_rows_raw": int(df_raw.shape[0]),
        "n_rows_after_processing": int(df.shape[0]),
        "attitudes_centered": bool(cfg.center_attitudes),
        "attitudes_center_mean": df.attrs.get("attitudes_center_mean", np.nan),
        "attitudes_col_for_model": cfg.attitudes_col_for_model,
        "use_sum_contrasts_for_factors": bool(cfg.use_sum_contrasts_for_factors),
        "run_unlabelled_origin_test": bool(cfg.run_unlabelled_origin_test),
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

    # Outcome distribution diagnostics (skew, kurtosis, normality tests)
    out_diag = outcome_distribution_diagnostics(df, cfg.outcomes)
    save_df(out_diag, cfg.output_dir / "outcome_distribution_diagnostics.csv")

    # --------------------------------------------------------
    # Model fitting
    # --------------------------------------------------------
    bundles: List[LMMBundle] = []
    model_errors: List[dict] = []

    # Primary Study 1 models
    rhs_primary = rhs_study1_primary(cfg)
    rhs_primary_cols = rhs_study1_primary_required_cols(cfg)

    for outcome in cfg.outcomes:
        try:
            b = fit_lmm_crossed_intercepts(
                df=df,
                cfg=cfg,
                outcome=outcome,
                rhs=rhs_primary,
                rhs_required_cols=rhs_primary_cols,
                model_name="study1_primary",
                reml=True,
            )
            bundles.append(b)
            save_text(
                str(b.fitted.summary()),
                cfg.output_dir / "model_summaries" / f"{outcome}__study1_primary_summary.txt",
            )
        except Exception as e:
            model_errors.append({
                "outcome": outcome,
                "model": "study1_primary",
                "formula": f"{outcome} ~ {rhs_primary}",
                "rhs_required_cols": rhs_primary_cols,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })
            save_text(
                traceback.format_exc(),
                cfg.output_dir / "model_summaries" / f"{outcome}__study1_primary_ERROR.txt",
            )

    # Optional unlabelled-only origin test
    if cfg.run_unlabelled_origin_test:
        rhs_unlab = rhs_study1_unlabelled_origin_test(cfg)
        rhs_unlab_cols = rhs_study1_unlabelled_required_cols(cfg)

        pa_series = df[cfg.presented_attribution_col]
        has_ref_category = (
            cfg.ref_presented_attr in list(pa_series.cat.categories)
        ) if hasattr(pa_series, "cat") else False

        # Use only rows explicitly coded as the reference unlabelled category
        df_unlabelled = df[df[cfg.presented_attribution_col].astype("string") == cfg.ref_presented_attr].copy()

        if has_ref_category and df_unlabelled.shape[0] > 0:
            for outcome in cfg.outcomes:
                try:
                    att_col = cfg.attitudes_col_for_model
                    needed_check = [
                        outcome,
                        cfg.participant_id_col,
                        cfg.artwork_id_col,
                        cfg.actual_origin_col,
                        att_col,
                    ]
                    if drop_missing(df_unlabelled, needed_check).shape[0] < 10:
                        # Guardrail: skip models with too few usable rows
                        continue

                    b = fit_lmm_crossed_intercepts(
                        df=df_unlabelled,
                        cfg=cfg,
                        outcome=outcome,
                        rhs=rhs_unlab,
                        rhs_required_cols=rhs_unlab_cols,
                        model_name="study1_unlabelled_origin_test",
                        reml=True,
                    )
                    bundles.append(b)
                    save_text(
                        str(b.fitted.summary()),
                        cfg.output_dir / "model_summaries" / f"{outcome}__study1_unlabelled_origin_test_summary.txt",
                    )
                except Exception as e:
                    model_errors.append({
                        "outcome": outcome,
                        "model": "study1_unlabelled_origin_test",
                        "formula": f"{outcome} ~ {rhs_unlab}",
                        "rhs_required_cols": rhs_unlab_cols,
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                    })
                    save_text(
                        traceback.format_exc(),
                        cfg.output_dir / "model_summaries" / f"{outcome}__study1_unlabelled_origin_test_ERROR.txt",
                    )
        else:
            # Explicitly record why the unlabelled-origin models did not run
            save_text(
                "Unlabelled-only origin test was skipped.\n"
                f"Reason: PresentedAttribution does not contain explicit '{cfg.ref_presented_attr}' rows, "
                "or there were zero rows matching it after processing.\n"
                "If you intend to model an unlabelled condition, encode it explicitly in PresentedAttribution "
                f"(e.g., '{cfg.ref_presented_attr}') rather than leaving values missing.\n",
                cfg.output_dir / "model_summaries" / "UNLABELLED_ORIGIN_TEST_SKIPPED.txt",
            )
    else:
        save_text(
            "Unlabelled-only origin test disabled via config (run_unlabelled_origin_test = False).\n",
            cfg.output_dir / "model_summaries" / "UNLABELLED_ORIGIN_TEST_SKIPPED.txt",
        )

    # --------------------------------------------------------
    # Aggregate model outputs across bundles
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

    # Report which columns were required and how many rows were used for each model
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

    # Key terms table (focused on attribution, origin, and attitudes effects)
    save_df(
        extract_key_terms_study1(fixed_all, cfg),
        cfg.output_dir / "key_hypothesis_tests_study1.csv",
    )

    # --------------------------------------------------------
    # Residual diagnostics and OLS-based influence
    # --------------------------------------------------------
    if bundles:
        resid_diag_df = collect_residual_diagnostics(bundles, cfg)
        save_df(resid_diag_df, cfg.output_dir / "lmm_residual_diagnostics.csv")
    else:
        save_df(pd.DataFrame(), cfg.output_dir / "lmm_residual_diagnostics.csv")

    # --------------------------------------------------------
    # Estimated marginal means (EMMs) for primary models
    # --------------------------------------------------------
    emm_df = compute_emm_for_primary_models(bundles, cfg)
    save_df(emm_df, cfg.output_dir / "emm_study1_primary.csv")

    # --------------------------------------------------------
    # Random effects summaries (participants and artworks)
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
        # Create an empty file so that downstream scripts can rely on its existence.
        save_df(pd.DataFrame(), cfg.output_dir / "model_errors.csv")


def main() -> None:
    """
    Entry point for the Study 1 analysis script.

    This simply:
      - Builds the default configuration.
      - Ensures project/data directories exist.
      - Runs the Study 1 pipeline.
      - Prints the output directory for convenience.
    """
    cfg = build_default_config()

    cfg.project_root.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    run_study1(cfg)
    print(f"Study 1 complete. Outputs saved to:\n  {cfg.output_dir}")


if __name__ == "__main__":
    main()
