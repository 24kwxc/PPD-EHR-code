"""Build the analysis-ready eICU external-validation clinical table."""
from __future__ import annotations

"""Build the eICU master cohort, column-aligned with MIMIC for transportability.

eICU's table structure differs from MIMIC but every concept is available:
severity (apachePatientResult), vasopressors (infusionDrug), steroids
(medication), ventilation (respiratoryCare + actualventdays), RRT (treatment),
glucose (lab + nurseCharting bedside glucose), enteral nutrition (intakeOutput),
mortality (actualicumortality/hospitalmortality), psychiatric + diabetes
history (pastHistory), delirium (nurseCharting).

Expanded (v2): Added vitals, labs, IV medications, oral medication flags,
parenteral nutrition / dextrose, and baseline comorbidities to align with MIMIC.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_definitions import load_config, project_path
from clinical_definitions import (
    BIN_HOURS, EICU_GLUCOSE_LAB, EICU_GLUCOSE_POC_LABEL,
    EICU_VASOPRESSOR_DRUGS, EICU_STEROID_DRUG_PATTERNS, EICU_DELIRIUM_LABELS,
    EICU_GCS_LABEL, EICU_RESP_RATE_LABEL, EICU_RRT_TREATMENTS,
    EICU_NUTRITION_TERMS, EICU_PASTHISTORY_DIABETES, EICU_PASTHISTORY_PSYCH,
    # New eICU concepts
    EICU_VITALS_LABELS, EICU_LAB_LABELS, EICU_IV_DRUG_PATTERNS,
    EICU_ORAL_DRUG_PATTERNS, EICU_PN_DEXTROSE_TERMS,
    EICU_PASTHISTORY_HYPERTENSION, EICU_PASTHISTORY_CKD,
    EICU_PASTHISTORY_LIVER, EICU_PASTHISTORY_OBESITY,
)
from clinical_definitions import base_time_bins, within_horizon, assign_bin

BIN_MINUTES = BIN_HOURS * 60
MAX_MINUTES = 72 * 60
MAX_BINS_EICU = MAX_MINUTES // BIN_MINUTES


def _contains_any(series: pd.Series, terms) -> pd.Series:
    text = series.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for t in terms:
        mask = mask | text.str.contains(t.lower(), regex=False)
    return mask


def _contains_any_lower(text: pd.Series, terms) -> pd.Series:
    mask = pd.Series(False, index=text.index)
    for t in terms:
        mask = mask | text.str.contains(str(t).lower(), regex=False, na=False)
    return mask


def _first_number(value) -> float:
    m = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(m.group(0)) if m else np.nan


def process_nurse_charting(
    eicu: Path,
    nrows,
    chunksize: int = 2_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Single-pass processing of nurseCharting to extract glucose POC, delirium, and vitals."""
    print("Reading nurseCharting.csv.gz...")

    all_terms = [EICU_GLUCOSE_POC_LABEL] + EICU_DELIRIUM_LABELS + [EICU_RESP_RATE_LABEL, EICU_GCS_LABEL]
    for labels in EICU_VITALS_LABELS.values():
        all_terms.extend(labels)

    cols = [
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypevallabel",
        "nursingchartvalue",
    ]
    reader = pd.read_csv(
        eicu / "nurseCharting.csv.gz",
        usecols=cols,
        nrows=nrows,
        chunksize=chunksize,
        low_memory=False,
    )
    vital_terms = [("gcs", [EICU_GCS_LABEL]), ("resp_rate", [EICU_RESP_RATE_LABEL])]
    vital_terms.extend(EICU_VITALS_LABELS.items())
    glucose_parts = []
    delirium_parts = []
    vital_parts: dict[str, list[pd.DataFrame]] = {name: [] for name, _ in vital_terms}

    for i, nc in enumerate(reader, start=1):
        label_text = nc["nursingchartcelltypevallabel"].fillna("").astype(str).str.lower()
        nc = nc.loc[_contains_any_lower(label_text, all_terms)].copy()
        if nc.empty:
            continue
        label_text = label_text.loc[nc.index]
        offset = pd.to_numeric(nc["nursingchartoffset"], errors="coerce")
        nc = nc.loc[(offset >= 0) & (offset < MAX_MINUTES)].copy()
        if nc.empty:
            continue
        label_text = label_text.loc[nc.index]
        nc["nursingchartoffset"] = offset.loc[nc.index]
        nc["bin"] = (nc["nursingchartoffset"] // BIN_MINUTES).astype(int)

        g_mask = _contains_any_lower(label_text, [EICU_GLUCOSE_POC_LABEL])
        g_df = nc.loc[g_mask, ["patientunitstayid", "nursingchartoffset", "nursingchartvalue", "bin"]].copy()
        if not g_df.empty:
            g_df["glucose_mg_dl"] = pd.to_numeric(g_df["nursingchartvalue"], errors="coerce")
            g_df = g_df.dropna(subset=["glucose_mg_dl"])
            g_df = g_df[(g_df["glucose_mg_dl"] > 10) & (g_df["glucose_mg_dl"] < 2000)]
            if not g_df.empty:
                glucose_parts.append(g_df[["patientunitstayid", "nursingchartoffset", "glucose_mg_dl", "bin"]])

        d_mask = _contains_any_lower(label_text, EICU_DELIRIUM_LABELS)
        d_df = nc.loc[d_mask, ["patientunitstayid", "bin", "nursingchartvalue"]].copy()
        if not d_df.empty:
            d_label = label_text.loc[d_df.index]
            d_value = d_df["nursingchartvalue"].astype(str).str.lower()
            d_pos = d_df.loc[
                d_value.isin(["yes", "1", "true", "present"])
                | (d_label.str.contains("delirium", regex=False, na=False)
                   & d_value.str.contains("yes|present|1", regex=True, na=False))
            ]
            if not d_pos.empty:
                delirium_parts.append(
                    d_pos.drop_duplicates(["patientunitstayid", "bin"])
                    .assign(delirium_flag=1)[["patientunitstayid", "bin", "delirium_flag"]]
                )

        for vital_name, labels in vital_terms:
            v_mask = _contains_any_lower(label_text, labels)
            v_sub = nc.loc[v_mask, ["patientunitstayid", "bin", "nursingchartoffset", "nursingchartvalue"]].copy()
            if v_sub.empty:
                continue
            v_label = label_text.loc[v_sub.index]
            v_sub["val"] = pd.to_numeric(v_sub["nursingchartvalue"], errors="coerce")
            v_sub = v_sub.dropna(subset=["val"])
            if v_sub.empty:
                continue
            if vital_name == "temperature":
                f_mask = v_label.loc[v_sub.index].str.contains("temperature (f)", regex=False, na=False)
                f_mask = f_mask | v_label.loc[v_sub.index].str.contains("temperature f", regex=False, na=False)
                v_sub.loc[f_mask, "val"] = (v_sub.loc[f_mask, "val"] - 32) * 5 / 9
            v_sub = (
                v_sub.sort_values("nursingchartoffset")
                .groupby(["patientunitstayid", "bin"])
                .tail(1)[["patientunitstayid", "bin", "nursingchartoffset", "val"]]
            )
            vital_parts[vital_name].append(v_sub)

        if i % 10 == 0:
            print(f"  processed nurseCharting chunks: {i}", flush=True)

    g_df = (
        pd.concat(glucose_parts, ignore_index=True)
        if glucose_parts
        else pd.DataFrame(columns=["patientunitstayid", "nursingchartoffset", "glucose_mg_dl", "bin"])
    )
    delirium_flag = (
        pd.concat(delirium_parts, ignore_index=True).drop_duplicates(["patientunitstayid", "bin"])
        if delirium_parts
        else pd.DataFrame(columns=["patientunitstayid", "bin", "delirium_flag"])
    )

    vitals_dfs = []
    for vital_name, parts in vital_parts.items():
        if not parts:
            continue
        v = pd.concat(parts, ignore_index=True)
        v = (
            v.sort_values("nursingchartoffset")
            .groupby(["patientunitstayid", "bin"])["val"]
            .last()
            .rename(vital_name)
            .reset_index()
        )
        vitals_dfs.append(v)
    if vitals_dfs:
        vitals_merged = vitals_dfs[0]
        for df in vitals_dfs[1:]:
            vitals_merged = vitals_merged.merge(df, on=["patientunitstayid", "bin"], how="outer")
    else:
        vitals_merged = pd.DataFrame(columns=["patientunitstayid", "bin"])

    return g_df, delirium_flag, vitals_merged


def glucose(eicu: Path, nc_g_df: pd.DataFrame, nrows) -> pd.DataFrame:
    """lab glucose + nurseCharting bedside glucose (POC equivalent)."""
    parts = []
    
    # 1. Lab glucose
    lab = pd.read_csv(eicu / "lab.csv.gz",
                      usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
                      nrows=nrows, low_memory=False)
    g = lab[_contains_any(lab["labname"], [EICU_GLUCOSE_LAB])].copy()
    g["glucose_mg_dl"] = pd.to_numeric(g["labresult"], errors="coerce")
    g = g.rename(columns={"labresultoffset": "offset"})[["patientunitstayid", "offset", "glucose_mg_dl"]]
    g["source"] = "lab"
    parts.append(g)

    # 2. Bedside glucose (passed from process_nurse_charting)
    nc_g = nc_g_df.rename(columns={"nursingchartoffset": "offset"})[["patientunitstayid", "offset", "glucose_mg_dl"]]
    nc_g["source"] = "poc"
    parts.append(nc_g)
    
    glu = pd.concat(parts, ignore_index=True).dropna(subset=["glucose_mg_dl"])
    glu = glu[(glu["glucose_mg_dl"] > 10) & (glu["glucose_mg_dl"] < 2000)]
    glu = glu[(glu["offset"] >= 0) & (glu["offset"] < MAX_MINUTES)].copy()
    glu["bin"] = (glu["offset"] // BIN_MINUTES).astype(int)
    
    agg = glu.groupby(["patientunitstayid", "bin"])["glucose_mg_dl"].agg(
        glucose_mean="mean", glucose_min="min", glucose_max="max",
        glucose_std="std", glucose_count="count").reset_index()
        
    # TIR / hypo / hyper
    flags = glu.assign(
        in_range=((glu["glucose_mg_dl"] >= 70) & (glu["glucose_mg_dl"] <= 180)).astype(int),
        hypo=(glu["glucose_mg_dl"] < 70).astype(int),
        hyper_severe=(glu["glucose_mg_dl"] > 250).astype(int),
    )
    tir = (flags.groupby(["patientunitstayid", "bin"])
           .agg(tir_fraction=("in_range", "mean"),
                hypo_fraction=("hypo", "mean"),
                hyper_severe_fraction=("hyper_severe", "mean")).reset_index())
    agg = agg.merge(tir, on=["patientunitstayid", "bin"], how="left")
    agg["glucose_cv"] = agg["glucose_std"] / agg["glucose_mean"]
    return agg


def process_labs(eicu: Path, nrows) -> pd.DataFrame:
    """Extract last value of lab tests per bin."""
    print("Reading lab.csv.gz for clinical laboratory values...")
    lab = pd.read_csv(eicu / "lab.csv.gz",
                      usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
                      nrows=nrows, low_memory=False)
    
    lab = lab[(lab["labresultoffset"] >= 0) & (lab["labresultoffset"] < MAX_MINUTES)].copy()
    lab["bin"] = (lab["labresultoffset"] // BIN_MINUTES).astype(int)
    
    lab_dfs = []
    for lab_name, patterns in EICU_LAB_LABELS.items():
        sub = lab[_contains_any(lab["labname"], patterns)].copy()
        sub["val"] = pd.to_numeric(sub["labresult"], errors="coerce")
        sub = sub.dropna(subset=["val"])
        if not sub.empty:
            sub_bin = sub.sort_values("labresultoffset").groupby(["patientunitstayid", "bin"])["val"].last().rename(lab_name).reset_index()
            lab_dfs.append(sub_bin)
            
    if not lab_dfs:
        return pd.DataFrame(columns=["patientunitstayid", "bin"])
        
    merged = lab_dfs[0]
    for df in lab_dfs[1:]:
        merged = merged.merge(df, on=["patientunitstayid", "bin"], how="outer")
    return merged


def enteral_exposure(eicu: Path, nrows) -> pd.DataFrame:
    io = pd.read_csv(eicu / "intakeOutput.csv.gz",
                     usecols=["patientunitstayid", "intakeoutputoffset", "cellpath",
                              "celllabel", "cellvaluenumeric"],
                     nrows=nrows, low_memory=False)
    text = io[["cellpath", "celllabel"]].astype(str).agg(" | ".join, axis=1)
    ent = io[_contains_any(text, EICU_NUTRITION_TERMS)].copy()
    ent["amount"] = pd.to_numeric(ent["cellvaluenumeric"], errors="coerce")
    ent = ent.dropna(subset=["amount"])
    ent = ent[(ent["intakeoutputoffset"] >= 0) & (ent["intakeoutputoffset"] < MAX_MINUTES)].copy()
    ent["bin"] = (ent["intakeoutputoffset"] // BIN_MINUTES).astype(int)
    agg = (ent.groupby(["patientunitstayid", "bin"])["amount"]
           .agg(enteral_kcal_sum="sum", enteral_event_count="count").reset_index())
    return agg


def process_fluid_and_tpn(eicu: Path, nrows) -> pd.DataFrame:
    """Extract total fluid intake, parenteral nutrition and dextrose."""
    print("Reading intakeOutput.csv.gz for fluids and parenteral nutrition...")
    io = pd.read_csv(eicu / "intakeOutput.csv.gz",
                     usecols=["patientunitstayid", "intakeoutputoffset", "cellpath",
                              "celllabel", "cellvaluenumeric"],
                     nrows=nrows, low_memory=False)
    io = io[(io["intakeoutputoffset"] >= 0) & (io["intakeoutputoffset"] < MAX_MINUTES)].copy()
    io["bin"] = (io["intakeoutputoffset"] // BIN_MINUTES).astype(int)
    
    # 1. Total fluid intake
    io["amount"] = pd.to_numeric(io["cellvaluenumeric"], errors="coerce")
    io = io.dropna(subset=["amount"])
    
    fluid_agg = io.groupby(["patientunitstayid", "bin"])["amount"].sum().rename("total_fluid_ml").reset_index()
    
    # 2. Parenteral nutrition & dextrose kcal
    text = io[["cellpath", "celllabel"]].astype(str).agg(" | ".join, axis=1)
    
    pn_mask = _contains_any(text, ["tpn", "parenteral nutrition", "pn ", "lipid"])
    pn_df = io[pn_mask].copy()
    pn_agg = pn_df.groupby(["patientunitstayid", "bin"])["amount"].sum().rename("parenteral_nutrition_kcal").reset_index()
    
    dex_mask = _contains_any(text, ["d5", "d10", "d50", "dextrose"])
    dex_df = io[dex_mask].copy()
    dex_agg = dex_df.groupby(["patientunitstayid", "bin"])["amount"].sum().rename("dextrose_kcal").reset_index()
    
    merged = fluid_agg.merge(pn_agg, on=["patientunitstayid", "bin"], how="outer")
    merged = merged.merge(dex_agg, on=["patientunitstayid", "bin"], how="outer")
    return merged


def vasopressor(eicu: Path, nrows) -> pd.DataFrame:
    inf = pd.read_csv(eicu / "infusionDrug.csv.gz",
                      usecols=["patientunitstayid", "infusionoffset", "drugname", "drugrate"],
                      nrows=nrows, low_memory=False)
    v = inf[_contains_any(inf["drugname"], EICU_VASOPRESSOR_DRUGS)].copy()
    v["rate"] = pd.to_numeric(v["drugrate"], errors="coerce")
    v = v.dropna(subset=["rate"])
    v = v[(v["infusionoffset"] >= 0) & (v["infusionoffset"] < MAX_MINUTES)].copy()
    v["bin"] = (v["infusionoffset"] // BIN_MINUTES).astype(int)
    return v.groupby(["patientunitstayid", "bin"])["rate"].agg(
        vasopressor_sum="sum", vasopressor_count="count").reset_index()


def process_iv_medications(eicu: Path, nrows) -> pd.DataFrame:
    """Extract infusion rates for sedatives, opioids, antibiotics, etc."""
    print("Reading infusionDrug.csv.gz for IV medications...")
    inf = pd.read_csv(eicu / "infusionDrug.csv.gz",
                      usecols=["patientunitstayid", "infusionoffset", "drugname", "drugrate"],
                      nrows=nrows, low_memory=False)
    inf = inf[(inf["infusionoffset"] >= 0) & (inf["infusionoffset"] < MAX_MINUTES)].copy()
    inf["bin"] = (inf["infusionoffset"] // BIN_MINUTES).astype(int)
    
    meds_dfs = []
    for drug_name, patterns in EICU_IV_DRUG_PATTERNS.items():
        v = inf[_contains_any(inf["drugname"], patterns)].copy()
        v["rate"] = pd.to_numeric(v["drugrate"], errors="coerce")
        v = v.dropna(subset=["rate"])
        if not v.empty:
            v_bin = v.groupby(["patientunitstayid", "bin"])["rate"].sum().rename(f"{drug_name}_rate_sum").reset_index()
            meds_dfs.append(v_bin)
            
    if not meds_dfs:
        return pd.DataFrame(columns=["patientunitstayid", "bin"])
        
    merged = meds_dfs[0]
    for df in meds_dfs[1:]:
        merged = merged.merge(df, on=["patientunitstayid", "bin"], how="outer")
        
    # Create opioid/antibiotic composite flags
    opioid_cols = [f"{o}_rate_sum" for o in ["fentanyl", "morphine", "hydromorphone"] if f"{o}_rate_sum" in merged.columns]
    if opioid_cols:
        merged["opioid_any"] = (merged[opioid_cols].fillna(0).sum(axis=1) > 0).astype(int)
    else:
        merged["opioid_any"] = 0
        
    abx_cols = [f"{a}_rate_sum" for a in ["vancomycin", "piperacillin_tazo", "meropenem", "cefepime", "metronidazole"] if f"{a}_rate_sum" in merged.columns]
    if abx_cols:
        merged["antibiotic_any"] = (merged[abx_cols].fillna(0).sum(axis=1) > 0).astype(int)
    else:
        merged["antibiotic_any"] = 0
        
    return merged


def insulin_exposure(eicu: Path, nrows) -> pd.DataFrame:
    """infusion insulin (rate) + medication insulin (parsed dose)."""
    inf = pd.read_csv(eicu / "infusionDrug.csv.gz",
                      usecols=["patientunitstayid", "infusionoffset", "drugname", "drugrate"],
                      nrows=nrows, low_memory=False)
    ii = inf[_contains_any(inf["drugname"], ["insulin"])].copy()
    ii["rate"] = pd.to_numeric(ii["drugrate"], errors="coerce")
    ii = ii.dropna(subset=["rate"])
    ii = ii[(ii["infusionoffset"] >= 0) & (ii["infusionoffset"] < MAX_MINUTES)].copy()
    ii["bin"] = (ii["infusionoffset"] // BIN_MINUTES).astype(int)
    inf_agg = ii.groupby(["patientunitstayid", "bin"])["rate"].agg(
        insulin_rate_sum="sum", insulin_count="count").reset_index()

    med = pd.read_csv(eicu / "medication.csv.gz",
                      usecols=["patientunitstayid", "drugstartoffset", "drugname", "dosage"],
                      nrows=nrows, low_memory=False)
    mi = med[_contains_any(med["drugname"], ["insulin"])].copy()
    mi["dose"] = mi["dosage"].apply(_first_number)
    mi = mi.dropna(subset=["dose"])
    mi = mi[(mi["drugstartoffset"] >= 0) & (mi["drugstartoffset"] < MAX_MINUTES)].copy()
    mi["bin"] = (mi["drugstartoffset"] // BIN_MINUTES).astype(int)
    med_agg = mi.groupby(["patientunitstayid", "bin"])["dose"].agg(
        insulin_med_dose_sum="sum").reset_index()
    out = inf_agg.merge(med_agg, on=["patientunitstayid", "bin"], how="outer")
    return out


def steroid(eicu: Path, nrows) -> pd.DataFrame:
    med = pd.read_csv(eicu / "medication.csv.gz",
                      usecols=["patientunitstayid", "drugstartoffset", "drugname"],
                      nrows=nrows, low_memory=False)
    st = med[_contains_any(med["drugname"], EICU_STEROID_DRUG_PATTERNS)].copy()
    st = st[(st["drugstartoffset"] >= 0) & (st["drugstartoffset"] < MAX_MINUTES)].copy()
    st["bin"] = (st["drugstartoffset"] // BIN_MINUTES).astype(int)
    flag = st.drop_duplicates(["patientunitstayid", "bin"]).assign(steroid_flag=1)[
        ["patientunitstayid", "bin", "steroid_flag"]]
    return flag


def process_oral_medications(eicu: Path, nrows) -> pd.DataFrame:
    """Extract flags for non-IV (oral/enteral) medications."""
    print("Reading medication.csv.gz for oral/enteral drug flags...")
    med = pd.read_csv(eicu / "medication.csv.gz",
                      usecols=["patientunitstayid", "drugstartoffset", "drugname"],
                      nrows=nrows, low_memory=False)
    med = med[(med["drugstartoffset"] >= 0) & (med["drugstartoffset"] < MAX_MINUTES)].copy()
    med["bin"] = (med["drugstartoffset"] // BIN_MINUTES).astype(int)
    
    flags_dfs = []
    for flag_name, patterns in EICU_ORAL_DRUG_PATTERNS.items():
        st = med[_contains_any(med["drugname"], patterns)].copy()
        if not st.empty:
            flag = st.drop_duplicates(["patientunitstayid", "bin"]).assign(**{f"{flag_name}_flag": 1})[
                ["patientunitstayid", "bin", f"{flag_name}_flag"]
            ]
            flags_dfs.append(flag)
            
    if not flags_dfs:
        return pd.DataFrame(columns=["patientunitstayid", "bin"])
        
    merged = flags_dfs[0]
    for df in flags_dfs[1:]:
        merged = merged.merge(df, on=["patientunitstayid", "bin"], how="outer")
    return merged


def vent_rrt(eicu: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """vent flag (respiratoryCare spans), rrt flag (treatment), actualventdays."""
    rc = pd.read_csv(eicu / "respiratoryCare.csv.gz",
                     usecols=["patientunitstayid", "ventstartoffset", "ventendoffset"],
                     low_memory=False)
    vent_rows = []
    for _, r in rc.dropna(subset=["ventstartoffset"]).iterrows():
        start = float(r["ventstartoffset"])
        end = r["ventendoffset"] if pd.notna(r["ventendoffset"]) else r["ventstartoffset"]
        end = float(end)
        if end < 0 or start >= MAX_MINUTES:
            continue
        start = max(0.0, start)
        end = min(float(MAX_MINUTES), end)
        if end <= start:
            b0 = b1 = max(0, min(MAX_BINS_EICU - 1, int(start // BIN_MINUTES)))
        else:
            b0 = max(0, int(start // BIN_MINUTES))
            b1 = min(MAX_BINS_EICU - 1, int((end - 1e-9) // BIN_MINUTES))
        for b in range(b0, b1 + 1):
            vent_rows.append({"patientunitstayid": r["patientunitstayid"], "bin": b, "mv_flag": 1})
    vent = (pd.DataFrame(vent_rows).drop_duplicates(["patientunitstayid", "bin"])
            if vent_rows else pd.DataFrame(columns=["patientunitstayid", "bin", "mv_flag"]))

    tr = pd.read_csv(eicu / "treatment.csv.gz",
                     usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"],
                     low_memory=False)
    rrt = tr[_contains_any(tr["treatmentstring"], EICU_RRT_TREATMENTS)].copy()
    rrt = rrt[(rrt["treatmentoffset"] >= 0) & (rrt["treatmentoffset"] < MAX_MINUTES)].copy()
    rrt["bin"] = (rrt["treatmentoffset"] // BIN_MINUTES).astype(int)
    rrt_flag = rrt.drop_duplicates(["patientunitstayid", "bin"]).assign(rrt_flag=1)[
        ["patientunitstayid", "bin", "rrt_flag"]]

    return vent, rrt_flag, rc


def baseline_and_outcomes(eicu: Path) -> pd.DataFrame:
    """stay-level: APACHE, mortality, LOS, vent days, age/sex, baseline comorbidities."""
    pat = pd.read_csv(eicu / "patient.csv.gz", low_memory=False)
    base = pat[["patientunitstayid", "age", "gender", "admissionweight",
                "admissionheight", "unitdischargeoffset", "hospitaldischargestatus",
                "unitdischargestatus", "ethnicity", "unittype", "unitadmitsource", 
                "hospitaladmitoffset"]].copy()
    
    # Age parsing
    age_txt = base["age"].astype(str).str.strip().str.replace(">", "", regex=False)
    base["age"] = pd.to_numeric(age_txt, errors="coerce")
    base.loc[base["age"] > 89, "age"] = 90
    base["weight_kg"] = pd.to_numeric(base["admissionweight"], errors="coerce")
    base["height_cm"] = pd.to_numeric(base["admissionheight"], errors="coerce")

    # BMI
    h_m = base["height_cm"] / 100.0
    base["bmi"] = base["weight_kg"] / (h_m ** 2)
    base.loc[(base["bmi"] < 10) | (base["bmi"] > 100), "bmi"] = np.nan

    # admission type / ethnicity / icu type
    base["admission_type"] = base["unitadmitsource"].fillna("Unknown")
    base["ethnicity"] = base["ethnicity"].fillna("Unknown")
    base["icu_type"] = base["unittype"].fillna("Unknown")
    base["los_before_icu"] = pd.to_numeric(base["hospitaladmitoffset"].abs(), errors="coerce") / 60.0

    # APACHE score + predicted mortality
    ap = pd.read_csv(eicu / "apachePatientResult.csv.gz",
                     usecols=["patientunitstayid", "apachescore",
                              "predictedicumortality", "predictediculos",
                              "actualventdays"], low_memory=False)
    ap = (
        ap.groupby("patientunitstayid", as_index=False)
        .agg(
            apachescore=("apachescore", "max"),
            predictedicumortality=("predictedicumortality", "mean"),
            predictediculos=("predictediculos", "mean"),
            actualventdays=("actualventdays", "max"),
        )
    )
    base = base.merge(ap, on="patientunitstayid", how="left")

    # mortality flags
    base["hospital_mortality"] = (base["hospitaldischargestatus"].astype(str).str.lower() == "expired").astype(int)
    base["icu_mortality"] = (base["unitdischargestatus"].astype(str).str.lower() == "expired").astype(int)
    base["icu_los_hours"] = pd.to_numeric(base["unitdischargeoffset"], errors="coerce") / 60.0
    base["vent_hours"] = pd.to_numeric(base["actualventdays"], errors="coerce") * 24
    base["vent_free_days_28"] = (28 - base["vent_hours"] / 24.0).clip(lower=0, upper=28)

    # history: diabetes, psychiatric, and other new baseline comorbidities from pastHistory
    ph = pd.read_csv(eicu / "pastHistory.csv.gz",
                     usecols=["patientunitstayid", "pasthistorypath", "pasthistoryvaluetext"],
                     low_memory=False)
    text = ph[["pasthistorypath", "pasthistoryvaluetext"]].astype(str).agg(" | ".join, axis=1).str.lower()
    
    ph_dm = ph[_contains_any(text, EICU_PASTHISTORY_DIABETES)].drop_duplicates("patientunitstayid")
    base["diabetes"] = base["patientunitstayid"].isin(ph_dm["patientunitstayid"]).astype(int)
    
    # Psychiatric history
    for label, terms in EICU_PASTHISTORY_PSYCH.items():
        if not terms:
            continue
        ph_x = ph[_contains_any(text, terms)].drop_duplicates("patientunitstayid")
        base[f"hx_{label}"] = base["patientunitstayid"].isin(ph_x["patientunitstayid"]).astype(int)
    hx_cols = [f"hx_{l}" for l in EICU_PASTHISTORY_PSYCH if l and f"hx_{l}" in base.columns]
    base["hx_any_psych"] = base[hx_cols].sum(axis=1).gt(0).astype(int) if hx_cols else 0

    # New baseline comorbidities
    for name, terms in [("hypertension", EICU_PASTHISTORY_HYPERTENSION),
                        ("ckd", EICU_PASTHISTORY_CKD),
                        ("liver_disease", EICU_PASTHISTORY_LIVER),
                        ("obesity", EICU_PASTHISTORY_OBESITY)]:
        ph_x = ph[_contains_any(text, terms)].drop_duplicates("patientunitstayid")
        base[f"hx_{name}"] = base["patientunitstayid"].isin(ph_x["patientunitstayid"]).astype(int)

    # Proxy Charlson index from past history strings
    charlson_score = pd.Series(0, index=base.index)
    # Define pastHistory keywords mappings to weights
    charlson_weights = {
        "myocardial_infarction": (1, ["myocardial infarction", "heart attack", "coronary artery disease"]),
        "congestive_heart_failure": (1, ["congestive heart failure", "chf", "heart failure"]),
        "peripheral_vascular": (1, ["peripheral vascular", "pvd", "claudication"]),
        "cerebrovascular": (1, ["stroke", "tia", "cerebrovascular"]),
        "dementia": (1, ["dementia", "alzheimer"]),
        "chronic_pulmonary": (1, ["copd", "asthma", "emphysema", "chronic bronchitis"]),
        "rheumatic": (1, ["rheumatoid", "lupus", "connective tissue"]),
        "peptic_ulcer": (1, ["peptic ulcer", "gastric ulcer"]),
        "mild_liver": (1, ["hepatitis", "mild liver", "cirrhosis"]),
        "diabetes_uncomplicated": (1, ["diabetes", "diabetic"]),
        "diabetes_complicated": (2, ["diabetic nephropathy", "diabetic neuropathy", "diabetic retinopathy"]),
        "hemiplegia": (2, ["hemiplegia", "paraplegia"]),
        "renal": (2, ["chronic kidney", "ckd", "renal failure", "esrd"]),
        "malignancy": (2, ["cancer", "malignancy", "tumor", "lymphoma", "leukemia"]),
        "moderate_severe_liver": (3, ["liver failure", "ascites", "varices"]),
        "metastatic_solid_tumor": (6, ["metastatic", "stage iv"]),
        "aids": (6, ["aids", "hiv"]),
    }
    for comp, (weight, terms) in charlson_weights.items():
        ph_comp = ph[_contains_any(text, terms)].drop_duplicates("patientunitstayid")
        base[f"cci_{comp}"] = base["patientunitstayid"].isin(ph_comp["patientunitstayid"]).astype(int)
        charlson_score += base[f"cci_{comp}"] * weight
        
    base["charlson_index"] = charlson_score
    return base


def apply_exclusion_criteria(master: pd.DataFrame, cfg: dict, smoke: bool = False) -> pd.DataFrame:
    """Mirror the MIMIC target-trial exclusions for eICU validation."""
    counts = {"total_stays": int(master["stay_id"].nunique())}

    if "anchor_age" in master.columns:
        age = master.groupby("stay_id")["anchor_age"].first()
        keep = age[age >= 18].index
        master = master[master["stay_id"].isin(keep)].copy()
    counts["after_adult_filter"] = int(master["stay_id"].nunique())

    if "icu_los_hours" in master.columns:
        los = master.groupby("stay_id")["icu_los_hours"].first()
        keep = los[(los > 0) & (los <= 720)].index
        master = master[master["stay_id"].isin(keep)].copy()
    counts["after_los_filter"] = int(master["stay_id"].nunique())

    if "icu_los_hours" in master.columns and "icu_mortality" in master.columns:
        info = master.groupby("stay_id").agg(
            mort=("icu_mortality", "first"),
            los=("icu_los_hours", "first"),
        )
        early_death = info[(info["mort"] == 1) & (info["los"] < 24)].index
        master = master[~master["stay_id"].isin(early_death)].copy()
    counts["after_early_death_exclusion"] = int(master["stay_id"].nunique())

    if "glucose_mean" in master.columns:
        has_glucose = master.groupby("stay_id")["glucose_mean"].apply(lambda s: s.notna().any())
        master = master[master["stay_id"].isin(has_glucose[has_glucose].index)].copy()
    counts["after_glucose_filter"] = int(master["stay_id"].nunique())

    fname = "eicu_exclusion_counts_smoke.csv" if smoke else "eicu_exclusion_counts.csv"
    path = project_path(cfg, "results", "tables", fname)
    pd.DataFrame([counts]).to_csv(path, index=False)
    print(f"eICU exclusion counts saved → {path}")
    for k, v in counts.items():
        print(f"  {k}: {v:,}")
    return master


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    eicu = Path(cfg["eicu_root"])
    nrows = 1_500_000 if args.smoke else None

    pat = pd.read_csv(eicu / "patient.csv.gz", low_memory=False)
    if args.smoke:
        pat = pat.head(5000)
    ids = pat["patientunitstayid"].drop_duplicates()
    out = base_time_bins(ids, "patientunitstayid")
    out = out.rename(columns={"patientunitstayid": "stay_id"})

    # 1. Process nurseCharting (one pass)
    nc_g_df, delirium_flag, vitals_merged = process_nurse_charting(eicu, nrows)
    delirium_flag = delirium_flag.rename(columns={"patientunitstayid": "stay_id"})
    vitals_merged = vitals_merged.rename(columns={"patientunitstayid": "stay_id"})

    print("glucose..."); out = out.merge(glucose(eicu, nc_g_df, nrows).rename(columns={"patientunitstayid": "stay_id"}), on=["stay_id", "bin"], how="left")
    print("enteral..."); out = out.merge(enteral_exposure(eicu, nrows).rename(columns={"patientunitstayid": "stay_id"}), on=["stay_id", "bin"], how="left")
    print("vasopressor..."); out = out.merge(vasopressor(eicu, nrows).rename(columns={"patientunitstayid": "stay_id"}), on=["stay_id", "bin"], how="left")
    print("insulin..."); out = out.merge(insulin_exposure(eicu, nrows).rename(columns={"patientunitstayid": "stay_id"}), on=["stay_id", "bin"], how="left")
    print("steroid..."); out = out.merge(steroid(eicu, nrows).rename(columns={"patientunitstayid": "stay_id"}), on=["stay_id", "bin"], how="left")
    
    # 2. Add new vital signs and delirium
    out = out.merge(delirium_flag, on=["stay_id", "bin"], how="left")
    out = out.merge(vitals_merged, on=["stay_id", "bin"], how="left")

    # 3. Add new labs
    print("labs..."); labs_df = process_labs(eicu, nrows)
    if not labs_df.empty:
        out = out.merge(labs_df.rename(columns={"patientunitstayid": "stay_id"}), on=["stay_id", "bin"], how="left")

    # 4. Add new IV drugs
    print("IV drugs..."); iv_df = process_iv_medications(eicu, nrows)
    if not iv_df.empty:
        out = out.merge(iv_df.rename(columns={"patientunitstayid": "stay_id"}), on=["stay_id", "bin"], how="left")

    # 5. Add new fluid, TPN and dextrose
    print("fluid + TPN + dextrose..."); tpn_df = process_fluid_and_tpn(eicu, nrows)
    if not tpn_df.empty:
        out = out.merge(tpn_df.rename(columns={"patientunitstayid": "stay_id"}), on=["stay_id", "bin"], how="left")

    # 6. Add new oral drug flags
    print("oral drug flags..."); oral_df = process_oral_medications(eicu, nrows)
    if not oral_df.empty:
        out = out.merge(oral_df.rename(columns={"patientunitstayid": "stay_id"}), on=["stay_id", "bin"], how="left")

    # 7. Vent / RRT
    print("vent/rrt..."); vent, rrt, _ = vent_rrt(eicu)
    out = out.merge(vent.rename(columns={"patientunitstayid": "stay_id"}), on=["stay_id", "bin"], how="left")
    out = out.merge(rrt.rename(columns={"patientunitstayid": "stay_id"}), on=["stay_id", "bin"], how="left")

    # 8. Baseline & Outcomes
    print("baseline+outcomes..."); bl = baseline_and_outcomes(eicu)
    bl = bl.rename(columns={"patientunitstayid": "stay_id", "apachescore": "apache_score",
                            "age": "anchor_age", "gender": "gender"})
    out = out.merge(bl, on="stay_id", how="left")
    if "apache_score" in out.columns:
        out["severity_score"] = out["apache_score"]
    if "gender" in out.columns:
        out["sex_male"] = out["gender"].astype(str).str.upper().str.startswith("M").astype(int)

    # caloric target (HB*1.3 stress factor)
    sex = out["gender"].astype(str).str.upper()
    w = pd.to_numeric(out["weight_kg"], errors="coerce")
    h = pd.to_numeric(out["height_cm"], errors="coerce")
    a = pd.to_numeric(out["anchor_age"], errors="coerce")
    hb = pd.Series(np.nan, index=out.index)
    hb[sex == "M"] = 88.362 + 13.397 * w + 4.799 * h - 5.677 * a
    hb[sex == "F"] = 447.593 + 9.247 * w + 3.098 * h - 4.330 * a
    out["caloric_target_kcal_day"] = (hb * 1.3).clip(1000, 4000)
    out["pct_caloric_target"] = out["enteral_kcal_sum"].fillna(0) / (out["caloric_target_kcal_day"] * (BIN_HOURS / 24.0))

    # Tidy flags
    binary_flags = ["mv_flag", "rrt_flag", "steroid_flag", "delirium_flag", "diabetes",
                    "prokinetic_flag", "oral_hypoglycemic_flag", "betablocker_flag",
                    "antipsychotic_flag", "ppi_h2_flag", "antiemetic_flag",
                    "immunosuppressant_flag", "opioid_any", "antibiotic_any"]
    for c in binary_flags:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)
            
    zero_fill = ["vasopressor_sum", "vasopressor_count", "enteral_kcal_sum",
                 "insulin_rate_sum", "insulin_med_dose_sum", "pct_caloric_target"]
    # Also zero-fill all IV drug rate columns
    zero_fill += [c for c in out.columns if c.endswith("_rate_sum") or c.endswith("_amount_sum")]
    for c in zero_fill:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    # Ensure essential columns exist
    for c in ["vasopressor_sum", "vasopressor_count", "mv_flag", "rrt_flag",
              "steroid_flag", "sofa", "rass", "resp_rate",
              "heart_rate", "sbp", "dbp", "map", "spo2", "temperature"]:
        if c not in out.columns:
            out[c] = np.nan

    # future glucose windows
    out = out.sort_values(["stay_id", "bin"])
    out["future_24h_glucose_mean"] = (
        out.groupby("stay_id")["glucose_mean"]
        .transform(lambda s: s.shift(-1).rolling(4, min_periods=1).mean().shift(-3)))
    out["future_24h_tir"] = (
        out.groupby("stay_id")["tir_fraction"]
        .transform(lambda s: s.shift(-1).rolling(4, min_periods=1).mean().shift(-3)))

    out["database"] = "eICU"
    out = apply_exclusion_criteria(out, cfg, smoke=args.smoke)
    duplicate_keys = out.duplicated(["stay_id", "bin"]).sum()
    if duplicate_keys:
        raise SystemExit(f"eICU master cohort has duplicated stay-bin rows: {duplicate_keys:,}")
    path = project_path(cfg, "data", "interim",
                        "eicu_master_cohort_smoke.csv" if args.smoke else "eicu_master_cohort.csv")
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out):,} rows, {out.shape[1]} cols)")
    nz = out["enteral_kcal_sum"].gt(0)
    print(f"fed bins: {nz.sum():,} ({100*nz.mean():.1f}%), "
          f"hospital_mortality: {out.groupby('stay_id')['hospital_mortality'].first().mean():.3f}")
    print(f"insulin_rate_sum non-zero bins: {(out.get('insulin_rate_sum', pd.Series(dtype=float)).fillna(0) > 0).sum()}")


if __name__ == "__main__":
    main()
