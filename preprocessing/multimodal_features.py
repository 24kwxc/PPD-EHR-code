"""Multimodal loading, sequence construction, and text-feature utilities.

This module consolidates configuration, raw-table readers, sequence data classes,
and optional note embedding functions used by the event-bundle builder.
"""
from __future__ import annotations


# ==============================================================================
# Configuration
# ==============================================================================

import os
from pathlib import Path
from typing import Dict, List, Tuple

# Base paths ---------------------------------------------------------------
# Raw clinical data are controlled-access and are deliberately not bundled.
# Callers set these paths before importing the preprocessing modules, e.g.:
#   NEUROMETABOLIC_MIMIC_ROOT=/path/to/mimic-iv-3.1
#   NEUROMETABOLIC_MIMIC_NOTE_ROOT=/path/to/extracted/mimic-iv-note
#   NEUROMETABOLIC_BUNDLE_WORKDIR=/path/to/derived/bundle-workdir
PACKAGE_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_ROOT: Path = Path(os.environ.get("NEUROMETABOLIC_MIMIC_ROOT", PACKAGE_ROOT))
HOSP_DIR: Path = Path(os.environ.get("NEUROMETABOLIC_MIMIC_HOSP", DATA_ROOT / "hosp"))
ICU_DIR: Path = Path(os.environ.get("NEUROMETABOLIC_MIMIC_ICU", DATA_ROOT / "icu"))
NOTE_DIR: Path = Path(os.environ.get("NEUROMETABOLIC_MIMIC_NOTE_ROOT", DATA_ROOT / "note"))
OUTPUT_DIR: Path = Path(os.environ.get("NEUROMETABOLIC_BUNDLE_WORKDIR", PACKAGE_ROOT / "artifacts"))
DEFAULT_SHARDED_BUNDLE_DIR: Path = OUTPUT_DIR / "preprocessed_clinicalbert" / "sharded_bundle_event"

# Ensure downstream scripts know which files to touch
DIAGNOSES_PATH: Path = HOSP_DIR / "diagnoses_icd.csv.gz"
PRESCRIPTIONS_PATH: Path = HOSP_DIR / "prescriptions.csv.gz"
LABEVENTS_PATH: Path = HOSP_DIR / "labevents.csv.gz"
D_LABITEMS_PATH: Path = HOSP_DIR / "d_labitems.csv.gz"
ADMISSIONS_PATH: Path = HOSP_DIR / "admissions.csv.gz"
PATIENTS_PATH: Path = HOSP_DIR / "patients.csv.gz"
CHARTEVENTS_PATH: Path = ICU_DIR / "chartevents.csv.gz"
ICUSTAYS_PATH: Path = ICU_DIR / "icustays.csv.gz"
D_ITEMS_PATH: Path = ICU_DIR / "d_items.csv.gz"
DISCHARGE_NOTES_PATH: Path = NOTE_DIR / "discharge.csv.gz"

# Filtering options --------------------------------------------------------
DIABETES_ICD9_PREFIXES: Tuple[str, ...] = ("250",)
DIABETES_ICD10_PREFIXES: Tuple[str, ...] = ("E10", "E11")

# Known complication code groups. Codes are stored as prefixes so we can
# match both ICD-9 and ICD-10 variants without enumerating every suffix.
COMPLICATION_CODE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "heart_failure": ("428", "I50"),
    "renal_failure": ("584", "585", "586", "N17", "N18", "N19"),
    "infection": ("995.9", "A41", "R65", "038"),
    "pneumonia": ("486", "481", "482", "483", "J12", "J13", "J14", "J15", "J18"),
    "cerebrovascular": (
        "430",
        "431",
        "432",
        "433",
        "434",
        "435",
        "436",
        "437",
        "438",
        "I60",
        "I61",
        "I62",
        "I63",
        "I64",
        "I65",
        "I66",
        "I67",
        "I68",
        "I69",
    ),
    "diabetic_foot": (
        "250.7",
        "250.8",
        "707.1",
        "E10.5",
        "E10.6",
        "E11.5",
        "E11.6",
        "L97",
    ),
}

# Patterns used to query dictionary tables for time-series features.
# Case-insensitive containment checks are performed on the label fields.
VITAL_LABEL_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "heart_rate": ("HEART RATE",),
    "resp_rate": ("RESPIRATORY RATE",),
    "spo2": ("SPO2", "OXYGEN SATURATION"),
    "sbp": ("SYSTOLIC", "NBP SYSTOLIC"),
    "dbp": ("DIASTOLIC", "NBP DIASTOLIC"),
    "mbp": ("MEAN BP", "NBP MEAN"),
    "temperature": ("TEMPERATURE",),
    "glucose_bedside": ("GLUCOSE (FINGERSTICK)", "GLUCOSE STICK"),
}

LAB_LABEL_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "glucose": ("GLUCOSE",),
    "creatinine": ("CREATININE",),
    "bun": ("UREA NITROGEN", "BLOOD UREA"),
    "wbc": ("WBC", "WHITE BLOOD"),
    "lactate": ("LACTATE",),
}

# Medication keyword groupings. Keywords are matched against the drug name.
MEDICATION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "insulin": ("INSULIN",),
    "metformin": ("METFORMIN",),
    "sulfonylurea": ("GLIPIZIDE", "GLYBURIDE", "GLIMEPIRIDE"),
    "sglt2": ("DAPAGLIFLOZIN", "CANAGLIFLOZIN", "EMPAGLIFLOZIN"),
    "dpp4": ("SITAGLIPTIN", "LINAGLIPTIN", "SAXAGLIPTIN"),
    "glp1": ("LIRAGLUTIDE", "SEMAGLUTIDE", "EXENATIDE"),
    "antibiotic": (
        "CEF", "PIPERACILLIN", "MEROPENEM", "VANCOMYCIN", "PIP/TAZ", "LEVOFLOXACIN"
    ),
    "anticoagulant": ("HEPARIN", "ENOXAPARIN", "WARFARIN", "APIXABAN"),
}

# If True, capture every available item instead of filtered subsets for each modality.
INCLUDE_ALL_VITALS: bool = True
INCLUDE_ALL_LABS: bool = True
INCLUDE_ALL_MEDICATIONS: bool = True

# General pipeline hyper-parameters ---------------------------------------
TIME_FREQUENCY: str = "1H"
MAX_STAYS: int = 75000  # adjust if memory allows more records
MIN_SEQUENCE_LENGTH: int = 6  # require at least 6 hourly bins per admission

# I/O and chunking parameters --------------------------------------------
CHUNK_SIZE_DIAGNOSES: int = 100_000
CHUNK_SIZE_VITALS: int = 100_000
CHUNK_SIZE_LABS: int = 100_000
CHUNK_SIZE_MEDS: int = 100_000

# Training defaults --------------------------------------------------------
BATCH_SIZE: int = 16
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 3e-5
NUM_EPOCHS: int = 40
NUM_WORKERS: int = 4
RANDOM_STATE: int = 42
NOTE_MAX_FEATURES: int = 768
GRAD_CLIP: float = 3.0


def ensure_output_dirs() -> None:
    """Create output directories when they do not yet exist."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


__all__: List[str] = [
    "DATA_ROOT",
    "HOSP_DIR",
    "ICU_DIR",
    "NOTE_DIR",
    "OUTPUT_DIR",
    "DEFAULT_SHARDED_BUNDLE_DIR",
    "DIAGNOSES_PATH",
    "PRESCRIPTIONS_PATH",
    "LABEVENTS_PATH",
    "D_LABITEMS_PATH",
    "ADMISSIONS_PATH",
    "PATIENTS_PATH",
    "CHARTEVENTS_PATH",
    "ICUSTAYS_PATH",
    "D_ITEMS_PATH",
    "DISCHARGE_NOTES_PATH",
    "DIABETES_ICD9_PREFIXES",
    "DIABETES_ICD10_PREFIXES",
    "COMPLICATION_CODE_GROUPS",
    "VITAL_LABEL_PATTERNS",
    "LAB_LABEL_PATTERNS",
    "MEDICATION_KEYWORDS",
    "TIME_FREQUENCY",
    "MAX_STAYS",
    "MIN_SEQUENCE_LENGTH",
    "CHUNK_SIZE_DIAGNOSES",
    "CHUNK_SIZE_VITALS",
    "CHUNK_SIZE_LABS",
    "CHUNK_SIZE_MEDS",
    "INCLUDE_ALL_VITALS",
    "INCLUDE_ALL_LABS",
    "INCLUDE_ALL_MEDICATIONS",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "NUM_EPOCHS",
    "NUM_WORKERS",
    "RANDOM_STATE",
    "NOTE_MAX_FEATURES",
    "GRAD_CLIP",
    "ensure_output_dirs",
]


# ==============================================================================
# Raw MIMIC-IV table loading
# ==============================================================================

"""Utilities to pull and filter raw MIMIC-IV tables for the multimodal pipeline."""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple

import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm 是可选依赖
    tqdm = None

logger = logging.getLogger(__name__)


def _standardise_code(value: str) -> str:
    """Uppercase and strip punctuation so comparisons are stable."""

    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _normalise_icd(value: str) -> str:
    """Return an ICD code string stripped of punctuation for prefix comparison."""

    if not value:
        return ""
    return value.upper().replace(".", "").replace("-", "").strip()


def _code_matches_prefix(code: str, prefixes: Iterable[str]) -> bool:
    normalised = _normalise_icd(code)
    return any(normalised.startswith(_normalise_icd(prefix)) for prefix in prefixes)


def _iterate_with_progress(iterator: Iterable[pd.DataFrame], desc: str) -> Iterable[pd.DataFrame]:
    if tqdm is None:
        yield from iterator
    else:  # pragma: no branch - 简单地把迭代器交给 tqdm
        yield from tqdm(iterator, desc=desc, unit="chunk", leave=False)


def _normalise_frequency(freq: str | None) -> str | None:
    if freq is None:
        return None
    return freq.lower()


def load_icu_stays(
    max_records: int | None = None,
    diabetes_only: bool = True,
) -> pd.DataFrame:
    """Return ICU stays with optional diabetes filtering."""

    icu_cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
    icu_df = pd.read_csv(ICUSTAYS_PATH, compression="gzip", usecols=icu_cols)
    icu_df = icu_df.dropna(subset=["stay_id"]).astype({"stay_id": int})
    icu_df["intime"] = pd.to_datetime(icu_df["intime"], errors="coerce")
    icu_df["outtime"] = pd.to_datetime(icu_df["outtime"], errors="coerce")
    icu_df = icu_df.dropna(subset=["intime", "outtime"])
    icu_df = icu_df.drop_duplicates(subset=["stay_id"]).sort_values(["subject_id", "stay_id"])

    if not diabetes_only:
        if max_records is not None:
            icu_df = icu_df.head(max_records)
        logger.info("Selected %d ICU stays (diabetes_only=%s)", len(icu_df), diabetes_only)
        return icu_df[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]]

    diag_path = DIAGNOSES_PATH
    logger.info("Scanning diagnoses for diabetes admissions: %s", diag_path)

    col_names = ["subject_id", "hadm_id", "icd_code", "icd_version"]
    diabetic_rows: List[pd.DataFrame] = []

    chunksize = CHUNK_SIZE_DIAGNOSES
    reader = pd.read_csv(diag_path, compression="gzip", usecols=col_names, chunksize=chunksize)
    for chunk in _iterate_with_progress(reader, "Scanning diagnoses"):
        chunk["icd_code"] = chunk["icd_code"].map(_standardise_code)
        chunk = chunk.dropna(subset=["hadm_id", "subject_id"])

        is_diabetes_icd9 = (chunk["icd_version"] == 9) & chunk["icd_code"].apply(
            _code_matches_prefix, prefixes=DIABETES_ICD9_PREFIXES
        )
        is_diabetes_icd10 = (chunk["icd_version"] == 10) & chunk["icd_code"].apply(
            _code_matches_prefix, prefixes=DIABETES_ICD10_PREFIXES
        )

        diabetic = chunk[is_diabetes_icd9 | is_diabetes_icd10][["subject_id", "hadm_id"]]
        if diabetic.empty:
            continue
        diabetic_rows.append(diabetic.drop_duplicates())
        logger.debug("Matched %d diabetic admissions in chunk", len(diabetic))

    if not diabetic_rows:
        logger.warning("No diabetic admissions were found.")
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])

    diabetic_df = pd.concat(diabetic_rows, ignore_index=True).drop_duplicates()
    merged = diabetic_df.merge(icu_df, on=["subject_id", "hadm_id"], how="inner")
    merged = merged.drop_duplicates(subset=["stay_id"]).sort_values(["subject_id", "stay_id"])

    if merged.empty:
        logger.warning("No ICU stays matched the diabetic admissions filter.")
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])

    if max_records is not None:
        merged = merged.head(max_records)

    logger.info("Identified %d diabetic ICU stays", len(merged))
    return merged[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]]


def load_diabetic_stays(max_records: int | None = None) -> pd.DataFrame:
    """Backward-compatible wrapper for diabetes-filtered ICU stays."""

    return load_icu_stays(max_records=max_records, diabetes_only=True)


def load_complication_labels(
    stay_frame: pd.DataFrame, complications: Mapping[str, Tuple[str, ...]]
) -> pd.DataFrame:
    """Build a multi-label DataFrame keyed by stay."""

    if stay_frame.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", *complications.keys()])

    diag_path = DIAGNOSES_PATH
    logger.info("Deriving complication labels from diagnoses: %s", diag_path)

    target_hadm: Set[int] = set(stay_frame["hadm_id"].tolist())
    col_names = ["subject_id", "hadm_id", "icd_code"]
    label_records: MutableMapping[Tuple[int, int], Dict[str, int]] = defaultdict(
        lambda: {name: 0 for name in complications}
    )

    chunksize = CHUNK_SIZE_DIAGNOSES
    reader = pd.read_csv(diag_path, compression="gzip", usecols=col_names, chunksize=chunksize)
    for chunk in _iterate_with_progress(reader, "Deriving labels"):
        chunk = chunk.dropna(subset=["hadm_id", "subject_id", "icd_code"])
        chunk = chunk[chunk["hadm_id"].isin(target_hadm)]
        if chunk.empty:
            continue
        chunk["icd_code"] = chunk["icd_code"].map(_standardise_code)

        for _, row in chunk.iterrows():
            key = (int(row["subject_id"]), int(row["hadm_id"]))
            code = row["icd_code"]
            for label, prefixes in complications.items():
                if _code_matches_prefix(code, prefixes):
                    label_records[key][label] = 1

    rows = []
    for _, row in stay_frame.iterrows():
        subject_id = int(row["subject_id"])
        hadm_id = int(row["hadm_id"])
        stay_id = int(row["stay_id"])
        key = (subject_id, hadm_id)
        labels = label_records.get(key, {name: 0 for name in complications})
        rows.append({
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "stay_id": stay_id,
            **labels,
        })

    label_df = pd.DataFrame(rows)
    logger.info("Constructed labels for %d ICU stays", len(label_df))
    return label_df


def load_static_features(stay_frame: pd.DataFrame) -> pd.DataFrame:
    if stay_frame.empty:
        return pd.DataFrame(columns=["stay_id"])

    admissions_cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "admission_type",
        "insurance",
        "ethnicity",
    ]
    patients_cols = ["subject_id", "gender", "anchor_age", "anchor_year"]

    admissions_header = pd.read_csv(ADMISSIONS_PATH, compression="gzip", nrows=0)
    available_admissions_cols = [col for col in admissions_cols if col in admissions_header.columns]
    admissions = pd.read_csv(
        ADMISSIONS_PATH,
        compression="gzip",
        usecols=available_admissions_cols,
    )
    for col in admissions_cols:
        if col not in admissions:
            admissions[col] = np.nan

    patients_header = pd.read_csv(PATIENTS_PATH, compression="gzip", nrows=0)
    available_patient_cols = [col for col in patients_cols if col in patients_header.columns]
    patients = pd.read_csv(
        PATIENTS_PATH,
        compression="gzip",
        usecols=available_patient_cols,
    )
    for col in patients_cols:
        if col not in patients:
            patients[col] = np.nan

    merged = (
        stay_frame.merge(admissions, on=["subject_id", "hadm_id"], how="left")
        .merge(patients, on="subject_id", how="left")
    )

    merged["admittime"] = pd.to_datetime(merged["admittime"], errors="coerce")
    merged["dischtime"] = pd.to_datetime(merged["dischtime"], errors="coerce")
    merged["intime"] = pd.to_datetime(merged["intime"], errors="coerce")
    merged["outtime"] = pd.to_datetime(merged["outtime"], errors="coerce")

    merged["icu_los_hours"] = (
        (merged["outtime"] - merged["intime"]).dt.total_seconds() / 3600.0
    )
    merged.loc[merged["icu_los_hours"].isna(), "icu_los_hours"] = 0.0

    merged["age"] = merged["anchor_age"]
    valid_age = merged["admittime"].notna() & merged["anchor_year"].notna()
    merged.loc[valid_age, "age"] = (
        merged.loc[valid_age, "anchor_age"]
        + merged.loc[valid_age, "admittime"].dt.year
        - merged.loc[valid_age, "anchor_year"].astype(float)
    )
    merged["age"] = merged["age"].clip(lower=18, upper=100)

    gender_series = merged["gender"].astype(str).str.upper()
    merged["gender_male"] = (gender_series == "M").astype(float)
    merged["gender_female"] = (gender_series == "F").astype(float)

    cat_features = [
        col
        for col in ["admission_type", "insurance", "ethnicity"]
        if col in merged and merged[col].notna().any()
    ]
    if cat_features:
        prefix_map = {col: col for col in cat_features}
        cat_dummies = pd.get_dummies(merged[cat_features], prefix=prefix_map)
    else:
        cat_dummies = pd.DataFrame(index=merged.index)

    static = pd.concat(
        [
            merged[["subject_id", "hadm_id", "stay_id", "age", "icu_los_hours"]],
            cat_dummies,
            merged[[col for col in ["gender_male", "gender_female"] if col in merged]],
        ],
        axis=1,
    )

    static = static.drop_duplicates(subset=["stay_id"])
    return static


def load_discharge_notes(stay_frame: pd.DataFrame) -> pd.DataFrame:
    if stay_frame.empty:
        return pd.DataFrame(columns=["stay_id", "text"])

    note_cols = ["subject_id", "hadm_id", "text"]
    notes = pd.read_csv(DISCHARGE_NOTES_PATH, compression="gzip", usecols=note_cols)
    notes = notes.dropna(subset=["text"])
    notes = notes.groupby(["subject_id", "hadm_id"], as_index=False)["text"].agg(
        lambda series: "\n".join(series.astype(str))
    )

    merged = stay_frame.merge(notes, on=["subject_id", "hadm_id"], how="left")
    merged["text"] = merged["text"].fillna("")
    merged = merged[["stay_id", "text"]].drop_duplicates(subset=["stay_id"])
    return merged


def _resolve_itemids(
    dictionary_path: Path,
    patterns: Mapping[str, Tuple[str, ...]] | None,
    *,
    include_all: bool = False,
    linksto: str | None = None,
    prefix: str = "feature",
) -> Dict[str, Set[int]]:
    usecols = ["itemid", "label"]
    if include_all and linksto is not None:
        usecols.append("linksto")
    dictionary = pd.read_csv(dictionary_path, compression="gzip", usecols=usecols)
    dictionary = dictionary.dropna(subset=["itemid"]).copy()
    dictionary["itemid"] = dictionary["itemid"].astype(int)

    if include_all and linksto is not None and "linksto" in dictionary:
        dictionary["linksto"] = dictionary["linksto"].astype(str).str.lower()
        dictionary = dictionary[dictionary["linksto"] == linksto.lower()]

    if include_all:
        mapping: Dict[str, Set[int]] = {}
        for _, row in dictionary.iterrows():
            itemid = int(row["itemid"])
            raw_label = str(row.get("label") or "")
            label = re.sub(r"[^0-9A-Za-z]+", "_", raw_label.strip().lower()).strip("_")
            if not label:
                label = prefix
            if len(label) > 60:
                label = label[:60]
            if label[0].isdigit():
                label = f"{prefix}_{label}"
            feature_name = label
            if feature_name in mapping:
                feature_name = f"{label}_{itemid}"
            mapping.setdefault(feature_name, set()).add(itemid)
        return mapping

    if not patterns:
        return {}

    dictionary["label_upper"] = dictionary["label"].astype(str).str.upper()

    mapping: Dict[str, Set[int]] = {}
    for feature_name, keywords in patterns.items():
        mask = False
        for keyword in keywords:
            keyword = keyword.upper()
            mask = mask | dictionary["label_upper"].str.contains(keyword, na=False, regex=False)
        feature_itemids = set(dictionary.loc[mask, "itemid"].astype(int).tolist())
        mapping[feature_name] = feature_itemids
        logger.debug("Resolved %d itemids for %s", len(feature_itemids), feature_name)

    return mapping


def resolve_vital_itemids() -> Dict[str, Set[int]]:
    include_all = getattr(config, "INCLUDE_ALL_VITALS", False)
    patterns = None if include_all else VITAL_LABEL_PATTERNS
    return _resolve_itemids(
        D_ITEMS_PATH,
        patterns,
        include_all=include_all,
        linksto="chartevents",
        prefix="vital",
    )


def resolve_lab_itemids() -> Dict[str, Set[int]]:
    include_all = getattr(config, "INCLUDE_ALL_LABS", False)
    patterns = None if include_all else LAB_LABEL_PATTERNS
    return _resolve_itemids(
        D_LABITEMS_PATH,
        patterns,
        include_all=include_all,
        prefix="lab",
    )


def _build_itemid_lookup(feature_map: Mapping[str, Set[int]]) -> Dict[int, str]:
    lookup: Dict[int, str] = {}
    for feature_name, itemids in feature_map.items():
        for itemid in itemids:
            lookup[itemid] = feature_name
    return lookup


def extract_vitals(
    stay_frame: pd.DataFrame,
    freq: str | None = None,
    itemids: Mapping[str, Set[int]] | None = None,
) -> pd.DataFrame:
    """Return a long-form vital sign table with timestamps relative to ICU admission."""

    if itemids is None:
        itemids = resolve_vital_itemids()
    lookup = _build_itemid_lookup(itemids)
    target_itemids = set(lookup.keys())
    if not target_itemids:
        logger.warning("No vital sign itemids were resolved; returning empty frame.")
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "time_bin", "feature", "value"])

    target_stays: Set[int] = set(stay_frame["stay_id"].tolist())
    stay_lookup = stay_frame[["subject_id", "hadm_id", "stay_id"]]
    stay_intime = stay_frame[["stay_id", "intime"]].drop_duplicates(subset=["stay_id"]).copy()
    stay_intime["intime"] = pd.to_datetime(stay_intime["intime"], errors="coerce")
    intime_map = stay_intime.set_index("stay_id")["intime"]
    col_names = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"]
    header = pd.read_csv(CHARTEVENTS_PATH, compression="gzip", nrows=0)
    available_cols = [col for col in col_names if col in header.columns]
    agg_frames: List[pd.DataFrame] = []

    logger.info("Extracting vital signs for %d stays", len(target_stays))
    freq_norm = _normalise_frequency(freq)
    chunksize = CHUNK_SIZE_VITALS
    reader = pd.read_csv(
        CHARTEVENTS_PATH,
        compression="gzip",
        usecols=available_cols,
        chunksize=chunksize,
    )
    for chunk in _iterate_with_progress(reader, "Extracting vitals"):
        chunk = chunk.dropna(subset=["charttime", "itemid", "valuenum"])
        if "stay_id" not in chunk or chunk["stay_id"].isna().all():
            chunk = chunk.merge(stay_lookup, on=["subject_id", "hadm_id"], how="inner")
        chunk = chunk.dropna(subset=["stay_id"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(target_stays) & chunk["itemid"].isin(target_itemids)]
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["intime"] = chunk["stay_id"].map(intime_map)
        chunk = chunk.dropna(subset=["charttime", "intime"])
        chunk["time_bin"] = chunk["charttime"] - chunk["intime"]
        if freq_norm is not None:
            chunk["time_bin"] = chunk["time_bin"].dt.floor(freq_norm)
        chunk = chunk[chunk["time_bin"] >= pd.Timedelta(0)]
        chunk["feature"] = chunk["itemid"].map(lookup)
        chunk = chunk.dropna(subset=["feature"])

        grouped = (
            chunk.groupby(["subject_id", "hadm_id", "stay_id", "time_bin", "feature"], as_index=False)[
                "valuenum"
            ]
            .mean()
            .rename(columns={"valuenum": "value"})
        )
        agg_frames.append(grouped)

    if not agg_frames:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "time_bin", "feature", "value"])

    vitals = pd.concat(agg_frames, ignore_index=True)
    vitals = vitals.sort_values(["subject_id", "stay_id", "time_bin", "feature"]).reset_index(drop=True)
    return vitals


def extract_labs(
    stay_frame: pd.DataFrame,
    freq: str | None = None,
    itemids: Mapping[str, Set[int]] | None = None,
) -> pd.DataFrame:
    """Return a long-form laboratory table with timestamps relative to ICU admission."""

    if itemids is None:
        itemids = resolve_lab_itemids()
    lookup = _build_itemid_lookup(itemids)
    target_itemids = set(lookup.keys())
    if not target_itemids:
        logger.warning("No lab itemids were resolved; returning empty frame.")
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "time_bin", "feature", "value"])

    target_stays: Set[int] = set(stay_frame["stay_id"].tolist())
    stay_lookup = stay_frame[["subject_id", "hadm_id", "stay_id"]]
    stay_intime = stay_frame[["stay_id", "intime"]].drop_duplicates(subset=["stay_id"]).copy()
    stay_intime["intime"] = pd.to_datetime(stay_intime["intime"], errors="coerce")
    intime_map = stay_intime.set_index("stay_id")["intime"]
    col_names = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"]
    header = pd.read_csv(LABEVENTS_PATH, compression="gzip", nrows=0)
    available_cols = [col for col in col_names if col in header.columns]
    agg_frames: List[pd.DataFrame] = []

    logger.info("Extracting lab results for %d stays", len(target_stays))
    freq_norm = _normalise_frequency(freq)
    chunksize = CHUNK_SIZE_LABS
    reader = pd.read_csv(
        LABEVENTS_PATH,
        compression="gzip",
        usecols=available_cols,
        chunksize=chunksize,
    )
    for chunk in _iterate_with_progress(reader, "Extracting labs"):
        chunk = chunk.dropna(subset=["charttime", "itemid", "valuenum"])
        if "stay_id" not in chunk or chunk["stay_id"].isna().all():
            chunk = chunk.merge(stay_lookup, on=["subject_id", "hadm_id"], how="inner")
        chunk = chunk.dropna(subset=["stay_id"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(target_stays) & chunk["itemid"].isin(target_itemids)]
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["intime"] = chunk["stay_id"].map(intime_map)
        chunk = chunk.dropna(subset=["charttime", "intime"])
        chunk["time_bin"] = chunk["charttime"] - chunk["intime"]
        if freq_norm is not None:
            chunk["time_bin"] = chunk["time_bin"].dt.floor(freq_norm)
        chunk = chunk[chunk["time_bin"] >= pd.Timedelta(0)]
        chunk["feature"] = chunk["itemid"].map(lookup)
        chunk = chunk.dropna(subset=["feature"])

        grouped = (
            chunk.groupby(["subject_id", "hadm_id", "stay_id", "time_bin", "feature"], as_index=False)[
                "valuenum"
            ]
            .mean()
            .rename(columns={"valuenum": "value"})
        )
        agg_frames.append(grouped)

    if not agg_frames:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "time_bin", "feature", "value"])

    labs = pd.concat(agg_frames, ignore_index=True)
    labs = labs.sort_values(["subject_id", "stay_id", "time_bin", "feature"]).reset_index(drop=True)
    return labs


def _normalise_drug_feature(drug_name: str) -> str | None:
    if not isinstance(drug_name, str):
        return None
    slug = re.sub(r"[^0-9A-Za-z]+", "_", drug_name.strip().lower()).strip("_")
    if not slug:
        return None
    if len(slug) > 60:
        slug = slug[:60]
    if slug[0].isdigit():
        slug = f"med_{slug}"
    return f"med_{slug}"


def _match_drug_to_group(drug_name: str) -> str | None:
    if getattr(config, "INCLUDE_ALL_MEDICATIONS", False):
        return _normalise_drug_feature(drug_name)
    if not isinstance(drug_name, str):
        return None
    normalized = drug_name.upper()
    for group, keywords in MEDICATION_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return group
    return None


def extract_medications(
    stay_frame: pd.DataFrame,
    freq: str = TIME_FREQUENCY,
) -> pd.DataFrame:
    """Return hourly medication utilisation counts relative to ICU admission."""

    target_stays: Set[int] = set(stay_frame["stay_id"].tolist())
    stay_lookup = stay_frame[["subject_id", "hadm_id", "stay_id"]]
    stay_intime = stay_frame[["stay_id", "intime"]].drop_duplicates(subset=["stay_id"]).copy()
    stay_intime["intime"] = pd.to_datetime(stay_intime["intime"], errors="coerce")
    intime_map = stay_intime.set_index("stay_id")["intime"]
    col_names = ["subject_id", "hadm_id", "stay_id", "drug", "starttime", "stoptime", "dose_val_rx"]
    header = pd.read_csv(PRESCRIPTIONS_PATH, compression="gzip", nrows=0)
    available_cols = [col for col in col_names if col in header.columns]
    agg_frames: List[pd.DataFrame] = []

    logger.info("Extracting medications for %d stays", len(target_stays))
    freq_norm = _normalise_frequency(freq)
    chunksize = CHUNK_SIZE_MEDS
    reader = pd.read_csv(
        PRESCRIPTIONS_PATH,
        compression="gzip",
        usecols=available_cols,
        chunksize=chunksize,
    )
    for chunk in _iterate_with_progress(reader, "Extracting medications"):
        chunk = chunk.dropna(subset=["starttime", "drug"])
        if "stay_id" not in chunk or chunk["stay_id"].isna().all():
            chunk = chunk.merge(stay_lookup, on=["subject_id", "hadm_id"], how="inner")
        chunk = chunk.dropna(subset=["stay_id"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(target_stays)]
        if chunk.empty:
            continue
        chunk["drug_group"] = chunk["drug"].apply(_match_drug_to_group)
        chunk = chunk.dropna(subset=["drug_group"])
        if chunk.empty:
            continue

        chunk["event_time"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk["intime"] = chunk["stay_id"].map(intime_map)
        chunk = chunk.dropna(subset=["event_time", "intime"])
        if freq_norm is not None:
            chunk["event_time"] = chunk["event_time"].dt.floor(freq_norm)
        chunk["time_bin"] = chunk["event_time"] - chunk["intime"]
        chunk = chunk[chunk["time_bin"] >= pd.Timedelta(0)]
        chunk["value"] = pd.to_numeric(chunk["dose_val_rx"], errors="coerce")
        chunk.loc[chunk["value"].isna(), "value"] = 1.0  # fall back to a simple usage count

        grouped = (
            chunk.groupby(["subject_id", "hadm_id", "stay_id", "time_bin", "drug_group"], as_index=False)[
                "value"
            ]
            .sum()
            .rename(columns={"drug_group": "feature"})
        )
        agg_frames.append(grouped)

    if not agg_frames:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "time_bin", "feature", "value"])

    meds = pd.concat(agg_frames, ignore_index=True)
    meds = meds.sort_values(["subject_id", "stay_id", "time_bin", "feature"]).reset_index(drop=True)
    return meds


__all__ = [
    "load_diabetic_stays",
    "load_icu_stays",
    "load_complication_labels",
    "load_static_features",
    "load_discharge_notes",
    "resolve_vital_itemids",
    "resolve_lab_itemids",
    "extract_vitals",
    "extract_labs",
    "extract_medications",
]


# ==============================================================================
# Sequence construction
# ==============================================================================

"""Feature engineering helpers to align modalities into transformer-ready tensors."""

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SequenceExample:
    subject_id: int
    hadm_id: int
    stay_id: int
    timestamps: np.ndarray  # shape (T,), timedeltas from ICU admission
    vitals: np.ndarray  # shape (T, V)
    vitals_mask: np.ndarray  # shape (T, V)
    labs: np.ndarray  # shape (T, L)
    labs_mask: np.ndarray  # shape (T, L)
    meds: np.ndarray  # shape (T, M)
    meds_mask: np.ndarray  # shape (T, M)
    static: np.ndarray  # shape (S,)
    note: np.ndarray  # shape (N,)
    label: np.ndarray  # shape (C,)


@dataclass
class ModalityMetadata:
    feature_names: Tuple[str, ...]


@dataclass
class SequenceBundle:
    sequences: List[SequenceExample]
    vitals_meta: ModalityMetadata
    labs_meta: ModalityMetadata
    meds_meta: ModalityMetadata
    static_meta: ModalityMetadata
    note_meta: ModalityMetadata
    label_names: Tuple[str, ...]


def _long_to_wide(df: pd.DataFrame, features: Iterable[str] | None = None) -> pd.DataFrame:
    if df.empty:
        if features is None:
            return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "time_bin"])
        empty_cols = list(features)
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "time_bin", *empty_cols])

    pivot = (
        df.pivot_table(
            index=["subject_id", "hadm_id", "stay_id", "time_bin"],
            columns="feature",
            values="value",
            aggfunc="mean",
        )
        .sort_index()
        .reset_index()
    )
    pivot.columns.name = None

    if features is not None:
        for feature in features:
            if feature not in pivot:
                pivot[feature] = np.nan
        pivot = pivot[["subject_id", "hadm_id", "stay_id", "time_bin", *features]]
    return pivot


def _group_by_stay(wide_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    grouped: Dict[int, pd.DataFrame] = {}
    if wide_df.empty or "stay_id" not in wide_df:
        return grouped
    for stay_id, frame in wide_df.groupby("stay_id", sort=True):
        grouped[int(stay_id)] = frame.sort_values("time_bin").reset_index(drop=True)
    return grouped


def _ensure_feature_frame(
    frame: pd.DataFrame | None,
    columns: Sequence[str],
    timeline: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    if frame is None or frame.empty:
        empty = pd.DataFrame(index=pd.Index(timeline, name="time_bin"), columns=list(columns), dtype=float)
        return empty
    indexed = frame.set_index("time_bin")
    indexed = indexed.reindex(timeline)
    return indexed[list(columns)]


def _frame_to_tensor(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    values = frame.to_numpy(dtype=np.float32)
    mask = (~np.isnan(values)).astype(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values, mask


def build_sequence_bundle(
    vitals_long: pd.DataFrame,
    labs_long: pd.DataFrame,
    meds_long: pd.DataFrame,
    label_df: pd.DataFrame,
    static_df: Optional[pd.DataFrame],
    note_embeddings: Optional[pd.DataFrame],
    min_sequence_length: int,
) -> SequenceBundle:
    label_names = tuple(name for name in label_df.columns if name not in {"subject_id", "hadm_id", "stay_id"})
    vitals_features = tuple(sorted(vitals_long["feature"].unique())) if not vitals_long.empty else tuple()
    labs_features = tuple(sorted(labs_long["feature"].unique())) if not labs_long.empty else tuple()
    meds_features = tuple(sorted(meds_long["feature"].unique())) if not meds_long.empty else tuple()

    static_features: Tuple[str, ...] = tuple()
    static_lookup: Optional[pd.DataFrame] = None
    if static_df is not None and not static_df.empty:
        static_features = tuple(
            column
            for column in static_df.columns
            if column not in {"subject_id", "hadm_id", "stay_id"}
        )
        if static_features:
            static_lookup = (
                static_df.set_index("stay_id")[list(static_features)]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )
            static_lookup.index = static_lookup.index.astype(int)

    note_features: Tuple[str, ...] = tuple()
    note_lookup: Optional[pd.DataFrame] = None
    if note_embeddings is not None and not note_embeddings.empty:
        note_features = tuple(
            column for column in note_embeddings.columns if column != "stay_id"
        )
        if note_features:
            note_lookup = (
                note_embeddings.set_index("stay_id")[list(note_features)]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )
            note_lookup.index = note_lookup.index.astype(int)

    vitals_wide = _long_to_wide(vitals_long, vitals_features)
    labs_wide = _long_to_wide(labs_long, labs_features)
    meds_wide = _long_to_wide(meds_long, meds_features)

    vitals_by_stay = _group_by_stay(vitals_wide)
    labs_by_stay = _group_by_stay(labs_wide)
    meds_by_stay = _group_by_stay(meds_wide)

    sequences: List[SequenceExample] = []
    for _, row in label_df.iterrows():
        subject_id = int(row["subject_id"])
        hadm_id = int(row["hadm_id"])
        stay_id = int(row["stay_id"])

        vitals_frame = vitals_by_stay.get(stay_id)
        labs_frame = labs_by_stay.get(stay_id)
        meds_frame = meds_by_stay.get(stay_id)

        if vitals_frame is None and labs_frame is None and meds_frame is None:
            continue

        timeline_set = set()
        for frame in (vitals_frame, labs_frame, meds_frame):
            if frame is not None:
                timeline_set.update(frame["time_bin"].tolist())
        if not timeline_set:
            continue
        timeline = sorted(timeline_set)
        if len(timeline) < min_sequence_length:
            continue

        vitals_tensor = np.zeros((len(timeline), len(vitals_features)), dtype=np.float32)
        vitals_mask = np.zeros_like(vitals_tensor)
        if vitals_features:
            vitals_prepared = _ensure_feature_frame(vitals_frame, vitals_features, timeline)
            vitals_tensor, vitals_mask = _frame_to_tensor(vitals_prepared)

        labs_tensor = np.zeros((len(timeline), len(labs_features)), dtype=np.float32)
        labs_mask = np.zeros_like(labs_tensor)
        if labs_features:
            labs_prepared = _ensure_feature_frame(labs_frame, labs_features, timeline)
            labs_tensor, labs_mask = _frame_to_tensor(labs_prepared)

        meds_tensor = np.zeros((len(timeline), len(meds_features)), dtype=np.float32)
        meds_mask = np.zeros_like(meds_tensor)
        if meds_features:
            meds_prepared = _ensure_feature_frame(meds_frame, meds_features, timeline)
            meds_tensor, meds_mask = _frame_to_tensor(meds_prepared)

        timestamps = np.array(timeline, dtype="timedelta64[s]")
        label_values = row[list(label_names)].to_numpy(dtype=np.float32)

        if static_features and static_lookup is not None and stay_id in static_lookup.index:
            static_vec = static_lookup.loc[stay_id].to_numpy(dtype=np.float32)
        else:
            static_vec = np.zeros(len(static_features), dtype=np.float32)

        if note_features and note_lookup is not None and stay_id in note_lookup.index:
            note_vec = note_lookup.loc[stay_id].to_numpy(dtype=np.float32)
        else:
            note_vec = np.zeros(len(note_features), dtype=np.float32)

        sequences.append(
            SequenceExample(
                subject_id=int(subject_id),
                hadm_id=int(hadm_id),
                stay_id=stay_id,
                timestamps=timestamps,
                vitals=vitals_tensor,
                vitals_mask=vitals_mask,
                labs=labs_tensor,
                labs_mask=labs_mask,
                meds=meds_tensor,
                meds_mask=meds_mask,
                static=static_vec,
                note=note_vec,
                label=label_values,
            )
        )

    logger.info("Built %d usable sequences", len(sequences))
    return SequenceBundle(
        sequences=sequences,
        vitals_meta=ModalityMetadata(feature_names=vitals_features),
        labs_meta=ModalityMetadata(feature_names=labs_features),
        meds_meta=ModalityMetadata(feature_names=meds_features),
        static_meta=ModalityMetadata(feature_names=static_features),
        note_meta=ModalityMetadata(feature_names=note_features),
        label_names=label_names,
    )


__all__ = [
    "SequenceExample",
    "SequenceBundle",
    "ModalityMetadata",
    "build_sequence_bundle",
]


# ==============================================================================
# Optional text features
# ==============================================================================

"""Utilities for turning clinical notes into numeric embeddings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class NoteEmbeddings:
    matrix: pd.DataFrame
    feature_names: Tuple[str, ...]
    vectorizer: Optional[TfidfVectorizer] = None
    model_name: Optional[str] = None
    pooling: Optional[str] = None
    mean: Optional[pd.Series] = None
    std: Optional[pd.Series] = None
    standardization_applied: bool = False


def fit_tfidf_embeddings(
    notes: pd.DataFrame,
    text_column: str = "text",
    stay_id_column: str = "stay_id",
    max_features: int = 512,
    min_df: int = 5,
) -> NoteEmbeddings:
    if notes.empty:
        empty = pd.DataFrame(columns=[stay_id_column])
        vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
        return NoteEmbeddings(matrix=empty, feature_names=tuple(), vectorizer=vectorizer)

    corpus = notes[text_column].fillna("").astype(str).tolist()
    fitted_vectorizer: Optional[TfidfVectorizer] = None
    matrix = None
    for df_threshold in (min_df, 1):
        try:
            candidate = TfidfVectorizer(
                max_features=max_features,
                min_df=df_threshold,
                ngram_range=(1, 2),
                strip_accents="unicode",
            )
            matrix = candidate.fit_transform(corpus)
            fitted_vectorizer = candidate
            break
        except ValueError as err:
            if "After pruning, no terms remain" not in str(err):
                raise
            if df_threshold == 1:
                # No terms found even with the loosest threshold; fall back to empty features.
                empty = pd.DataFrame({stay_id_column: notes[stay_id_column].values})
                vectorizer = candidate
                return NoteEmbeddings(matrix=empty, feature_names=tuple(), vectorizer=vectorizer)
            continue

    if matrix is None or fitted_vectorizer is None:
        empty = pd.DataFrame({stay_id_column: notes[stay_id_column].values})
        vectorizer = TfidfVectorizer(max_features=max_features, min_df=1)
        return NoteEmbeddings(matrix=empty, feature_names=tuple(), vectorizer=vectorizer)

    vectorizer = fitted_vectorizer
    feature_names = tuple(vectorizer.get_feature_names_out())
    dense = matrix.astype(np.float32).toarray()

    embedding_df = pd.DataFrame(dense, columns=feature_names)
    embedding_df.insert(0, stay_id_column, notes[stay_id_column].values)
    return NoteEmbeddings(matrix=embedding_df, feature_names=feature_names, vectorizer=vectorizer)


def _build_feature_names(prefix: str, width: int) -> Tuple[str, ...]:
    return tuple(f"{prefix}_{idx:03d}" for idx in range(width))


def _batch_indices(total: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield start, end


def build_clinicalbert_embeddings(
    notes: pd.DataFrame,
    text_column: str = "text",
    stay_id_column: str = "stay_id",
    model_dir: str | Path = "data/models/ClinicalBERT",
    batch_size: int = 8,
    max_length: int = 512,
    pooling: str = "cls",
    device: Optional[str] = None,
    standardize: bool = False,
) -> NoteEmbeddings:
    if notes.empty:
        empty = pd.DataFrame(columns=[stay_id_column])
        return NoteEmbeddings(matrix=empty, feature_names=tuple(), model_name=str(model_dir), pooling=pooling)

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError("transformers and torch are required for ClinicalBERT embeddings") from err

    resolved_model_path = Path(model_dir).expanduser().resolve()
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"ClinicalBERT model not found at {resolved_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, local_files_only=True)
    model = AutoModel.from_pretrained(resolved_model_path, local_files_only=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    texts = notes[text_column].fillna("").astype(str).reset_index(drop=True)
    stay_ids = notes[stay_id_column].reset_index(drop=True)

    embedding_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start, end in _batch_indices(len(texts), batch_size):
            batch_texts = texts.iloc[start:end].tolist()
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden_states = outputs.last_hidden_state

            if pooling == "cls":
                pooled = hidden_states[:, 0, :]
            elif pooling == "mean":
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                masked = hidden_states * attention_mask
                lengths = attention_mask.sum(dim=1).clamp(min=1)
                pooled = masked.sum(dim=1) / lengths
            else:
                raise ValueError(f"Unsupported pooling strategy: {pooling}")

            embedding_chunks.append(pooled.cpu().numpy())

    matrix = np.concatenate(embedding_chunks, axis=0).astype(np.float32, copy=False)
    feature_names = _build_feature_names("clinicalbert", matrix.shape[1])

    embedding_df = pd.DataFrame(matrix, columns=feature_names)
    embedding_df.insert(0, stay_id_column, stay_ids.values)

    if device == "cuda":  # pragma: no cover - depends on hardware
        torch.cuda.empty_cache()

    mean: Optional[pd.Series] = None
    std: Optional[pd.Series] = None
    if standardize:
        feature_df = embedding_df.loc[:, feature_names]
        mean = feature_df.mean(axis=0)
        std = feature_df.std(axis=0).replace(0.0, 1.0)
        embedding_df.loc[:, feature_names] = (feature_df - mean) / std

    return NoteEmbeddings(
        matrix=embedding_df,
        feature_names=feature_names,
        model_name=str(resolved_model_path),
        pooling=pooling,
        mean=mean,
        std=std,
        standardization_applied=standardize,
    )


__all__ = [
    "NoteEmbeddings",
    "build_clinicalbert_embeddings",
    "fit_tfidf_embeddings",
]
