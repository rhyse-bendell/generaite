from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from Analyses import study1_analysis_script as s1
from Analyses import study2_analysis_script as s2
from Analyses import study3_analysis_script as s3
from Analyses import study4_analysis_script as s4


@dataclass
class StudyMetadata:
    study_key: str
    display_name: str
    script_path: Path
    default_input: Path
    default_output_pattern: str
    required_columns: List[str]
    optional_columns: List[str]
    outcomes: List[str]
    composites: Dict[str, List[str]]
    composite_aliases: Dict[str, List[str]]
    model_descriptions: List[str]
    notable_logic: List[str]
    run_callable: Callable[[Path, Optional[str]], Path]
    dataset_modes: Optional[List[str]] = None


def _shared_required_from_cfg(cfg) -> List[str]:
    return [
        cfg.participant_id_col,
        cfg.artwork_id_col,
        cfg.presented_attribution_col,
        cfg.actual_origin_col,
        cfg.attitudes_col,
        cfg.label_accuracy_col,
    ]


def _shared_optional_from_cfg(cfg) -> List[str]:
    return [cfg.border_condition_col, cfg.art_style_col, cfg.study_id_col]


def _composite_aliases() -> Dict[str, List[str]]:
    return {"CuriosityRating": ["CuriousityRating"]}


def _error_metadata(study_key: str, display_name: str, script_path: str, error: Exception) -> StudyMetadata:
    error_text = f"{type(error).__name__}: {error}"

    def _run_unavailable(_: Path, __: Optional[str] = None) -> Path:
        raise RuntimeError(f"{display_name} metadata failed to load. {error_text}")

    return StudyMetadata(
        study_key=study_key,
        display_name=f"{display_name} ⚠",
        script_path=Path(script_path),
        default_input=Path("<metadata-load-failed>"),
        default_output_pattern="<metadata-load-failed>",
        required_columns=[],
        optional_columns=[],
        outcomes=[],
        composites={},
        composite_aliases=_composite_aliases(),
        model_descriptions=["Metadata unavailable due to a loading error."],
        notable_logic=[error_text],
        run_callable=_run_unavailable,
    )


def run_study1(selected_input: Path, _: Optional[str] = None) -> Path:
    cfg = s1.build_default_config()
    cfg.input_csv = selected_input
    s1.run_study1(cfg)
    return cfg.output_dir


def run_study2(selected_input: Path, _: Optional[str] = None) -> Path:
    cfg = s2.build_default_config()
    cfg.input_csv = selected_input
    s2.run_study2(cfg)
    return cfg.output_dir


def run_study3(selected_input: Path, _: Optional[str] = None) -> Path:
    cfg = s3.build_default_config_study3()
    cfg.input_csv = selected_input
    s3.run_study3(cfg)
    return cfg.output_dir


def run_study4(selected_input: Path, dataset_mode: Optional[str] = None) -> Path:
    if dataset_mode not in {"Study1_Study3", "Study1_Study2"}:
        raise ValueError("Study 4 requires dataset mode: Study1_Study3 or Study1_Study2")
    cfg = s4.build_config_for_dataset(
        dataset_label=dataset_mode,
        input_filename=selected_input.name,
        stamp=s4.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    cfg.input_csv = selected_input
    if dataset_mode == "Study1_Study3":
        s4.run_study4_combined_1_3(cfg)
    else:
        s4.run_study4_combined_1_2(cfg)
    return cfg.output_dir


def _build_study1_metadata() -> StudyMetadata:
    cfg1 = s1.build_default_config()
    return StudyMetadata(
        study_key="study1",
        display_name="Study 1",
        script_path=Path("Analyses/study1_analysis_script.py"),
        default_input=cfg1.input_csv,
        default_output_pattern="Results/Study1/run_<timestamp>",
        required_columns=sorted(set(_shared_required_from_cfg(cfg1) + s1.rhs_study1_primary_required_cols(cfg1))),
        optional_columns=_shared_optional_from_cfg(cfg1),
        outcomes=cfg1.outcomes,
        composites=cfg1.composites,
        composite_aliases=_composite_aliases(),
        model_descriptions=[
            f"Primary mixed model: Outcome ~ {s1.rhs_study1_primary(cfg1)}",
            f"Optional unlabelled-origin test: Outcome ~ {s1.rhs_study1_unlabelled_origin_test(cfg1)}",
        ],
        notable_logic=[
            "Writes analysis-ready CSV to run folder and Data/Generaite_Study1_analysis_ready.csv",
            "Supports CuriosityRating fallback to CuriousityRating",
        ],
        run_callable=run_study1,
    )


def _build_study2_metadata() -> StudyMetadata:
    cfg2 = s2.build_default_config()
    return StudyMetadata(
        study_key="study2",
        display_name="Study 2",
        script_path=Path("Analyses/study2_analysis_script.py"),
        default_input=cfg2.input_csv,
        default_output_pattern="Results/Study2/run_<timestamp>",
        required_columns=sorted(set(_shared_required_from_cfg(cfg2) + [cfg2.border_condition_col] + s2.rhs_study2_border_required_cols(cfg2))),
        optional_columns=[cfg2.art_style_col, cfg2.study_id_col],
        outcomes=cfg2.outcomes,
        composites=cfg2.composites,
        composite_aliases=_composite_aliases(),
        model_descriptions=[
            f"Replication model: Outcome ~ {s2.rhs_study2_replication(cfg2)}",
            f"Border interaction model: Outcome ~ {s2.rhs_study2_border_interaction(cfg2)}",
            "TOST equivalence output for border-related fixed effects",
        ],
        notable_logic=[
            "Writes analysis-ready CSV to run folder and Data/Generaite_Study2_analysis_ready.csv",
            "BorderCondition is modeled explicitly and included in key hypothesis outputs",
        ],
        run_callable=run_study2,
    )


def _build_study3_metadata() -> StudyMetadata:
    cfg3 = s3.build_default_config_study3()
    return StudyMetadata(
        study_key="study3",
        display_name="Study 3",
        script_path=Path("Analyses/study3_analysis_script.py"),
        default_input=cfg3.input_csv,
        default_output_pattern="Results/Study3/run_<timestamp>",
        required_columns=sorted(set([
            cfg3.participant_id_col,
            cfg3.artwork_id_col,
            cfg3.presented_attribution_col,
            cfg3.actual_origin_col,
            cfg3.attitudes_col,
        ] + s3.rhs_study3_primary_required_cols(cfg3))),
        optional_columns=[cfg3.label_accuracy_col, cfg3.border_condition_col, cfg3.art_style_col, cfg3.study_id_col],
        outcomes=cfg3.outcomes,
        composites=cfg3.composites,
        composite_aliases=_composite_aliases(),
        model_descriptions=[f"Primary mixed model: Outcome ~ {s3.rhs_study3_primary(cfg3)}"],
        notable_logic=[
            "Writes analysis-ready CSV to run folder and Data/Generaite_Study3_analysis_ready.csv",
            "Unlabelled-origin test intentionally disabled in Study 3 config",
        ],
        run_callable=run_study3,
    )


def _build_study4_metadata() -> StudyMetadata:
    stamp = "YYYYMMDD_HHMMSS"
    cfg4_13 = s4.build_config_for_dataset("Study1_Study3", "Generaite_Study1_Study3_Combined_1-26-2026.csv", stamp)
    return StudyMetadata(
        study_key="study4",
        display_name="Study 4",
        script_path=Path("Analyses/study4_analysis_script.py"),
        default_input=cfg4_13.input_csv,
        default_output_pattern="Results/Study4/Study1_Study3|Study1_Study2/run_<timestamp>",
        required_columns=sorted(set([
            cfg4_13.participant_id_col,
            cfg4_13.artwork_id_col,
            cfg4_13.study_id_col,
            cfg4_13.presented_attribution_col,
            cfg4_13.actual_origin_col,
            cfg4_13.label_accuracy_col,
            cfg4_13.attitudes_col,
        ])),
        optional_columns=[cfg4_13.border_condition_col, cfg4_13.art_style_col],
        outcomes=cfg4_13.outcomes,
        composites=cfg4_13.composites,
        composite_aliases=_composite_aliases(),
        model_descriptions=[
            f"Combined 1+3 attribution model: Outcome ~ {s4.rhs_cs13_attribution(cfg4_13)}",
            f"Combined 1+3 accuracy moderator model: Outcome ~ {s4.rhs_cs13_accuracy(cfg4_13)}",
            f"Combined 1+2 border robustness model: Outcome ~ {s4.rhs_cs12_border(cfg4_13)}",
        ],
        notable_logic=[
            "Study 4 runs dataset-specific workflows; choose mode Study1_Study3 or Study1_Study2",
            "Writes analysis-ready CSV to run folder and Data/<input_stem>_analysis_ready.csv for the selected combined dataset",
        ],
        run_callable=run_study4,
        dataset_modes=["Study1_Study3", "Study1_Study2"],
    )


def load_study_metadata() -> Dict[str, StudyMetadata]:
    metadata: Dict[str, StudyMetadata] = {}
    builders = [
        ("study1", "Study 1", "Analyses/study1_analysis_script.py", _build_study1_metadata),
        ("study2", "Study 2", "Analyses/study2_analysis_script.py", _build_study2_metadata),
        ("study3", "Study 3", "Analyses/study3_analysis_script.py", _build_study3_metadata),
        ("study4", "Study 4", "Analyses/study4_analysis_script.py", _build_study4_metadata),
    ]

    for study_key, display_name, script_path, builder in builders:
        try:
            metadata[study_key] = builder()
        except Exception as exc:
            metadata[study_key] = _error_metadata(study_key, display_name, script_path, exc)

    return metadata
