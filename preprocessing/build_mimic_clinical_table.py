"""Build the analysis-ready MIMIC-IV clinical table.

Stages are kept in one file while retaining isolated namespaces so identically
named helpers cannot overwrite one another. Run ``python build_mimic_clinical_table.py
--help`` to see the six auditable stages.
"""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace


# ==============================================================================
# Confounders and clinical scores
# ==============================================================================

def _load_confounders():
    """Build time-varying and baseline confounders for the MIMIC dynamic cohort.

    Reduces high-frequency chartevents/inputevents/procedureevents/labevents into
    6-hour bins for the first 72 hours, plus baseline (admission-level) covariates.

    Expanded (v2): Adds vital signs (HR, BP, SpO2, Temperature), comprehensive
    labs (lactate, albumin, electrolytes, liver/renal panels, inflammatory markers),
    IV medications (sedatives, opioids, antibiotics, paralytics, diuretics),
    oral drug flags, parenteral nutrition, and expanded baseline comorbidities
    (Charlson, BMI, admission type, ethnicity, ICU type).

    Outputs data/interim/mimic_confounders.csv keyed by (stay_id, bin).
    """

    import argparse
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from clinical_definitions import load_config, project_path
    from clinical_definitions import (
        BIN_HOURS, MAX_HOURS, MIMIC_ITEMIDS, MIMIC_STEROID_DRUG_PATTERNS,
        ICD_DIABETES, ICD_PSYCHIATRIC,
        # New: drug patterns for prescriptions-based extraction
        MIMIC_PROKINETIC_DRUG_PATTERNS, MIMIC_ORAL_HYPOGLYCEMIC_PATTERNS,
        MIMIC_BETABLOCKER_PATTERNS, MIMIC_ANTIPSYCHOTIC_PATTERNS,
        MIMIC_PPI_H2_PATTERNS, MIMIC_ANTIEMETIC_PATTERNS,
        MIMIC_IMMUNOSUPPRESSANT_PATTERNS,
        # New: baseline comorbidity ICD codes
        ICD_HYPERTENSION, ICD_CKD, ICD_LIVER_DISEASE, ICD_OBESITY,
    )
    from clinical_definitions import (
        base_time_bins, hours_from_icu, within_horizon, assign_bin,
        aggregate_numeric_by_bin, last_value_by_bin, coverage_flag_by_bin,
    )
    from clinical_definitions import compute_sofa, charlson_from_icd


    def _load_chartevents(icu: Path, itemids: list[int], nrows, usecols_extra=None) -> pd.DataFrame:
        cols = ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"]
        if nrows is not None:
            df = pd.read_csv(
                icu / "chartevents.csv.gz",
                usecols=cols, nrows=nrows, low_memory=False,
                parse_dates=["charttime"],
            )
            return df[df["itemid"].isin(itemids)].copy()
        else:
            # Read in chunks of 10 million rows to stay light on RAM
            chunks = []
            for chunk in pd.read_csv(
                icu / "chartevents.csv.gz",
                usecols=cols, chunksize=10_000_000, low_memory=False,
                parse_dates=["charttime"],
            ):
                filtered = chunk[chunk["itemid"].isin(itemids)].copy()
                if not filtered.empty:
                    chunks.append(filtered)
            return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)


    def _load_inputevents(icu: Path, itemids: list[int], nrows) -> pd.DataFrame:
        cols = ["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime", "amount", "rate"]
        if nrows is not None:
            df = pd.read_csv(
                icu / "inputevents.csv.gz",
                usecols=cols, nrows=nrows, low_memory=False,
                parse_dates=["starttime", "endtime"],
            )
            return df[df["itemid"].isin(itemids)].copy()
        else:
            chunks = []
            for chunk in pd.read_csv(
                icu / "inputevents.csv.gz",
                usecols=cols, chunksize=2_000_000, low_memory=False,
                parse_dates=["starttime", "endtime"],
            ):
                filtered = chunk[chunk["itemid"].isin(itemids)].copy()
                if not filtered.empty:
                    chunks.append(filtered)
            return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)


    def _load_labevents(hosp: Path, itemids: list[int], stay_map: pd.DataFrame,
                        nrows=None) -> pd.DataFrame:
        """Load lab values, map hadm_id → stay_id, compute hours & bin."""
        cols = ["hadm_id", "itemid", "charttime", "valuenum"]
        if nrows is not None:
            lab = pd.read_csv(
                hosp / "labevents.csv.gz",
                usecols=cols, nrows=nrows, low_memory=False,
                parse_dates=["charttime"],
            )
            lab = lab[lab["itemid"].isin(itemids)].copy()
            lab = lab.merge(stay_map, on="hadm_id", how="inner")
            lab["valuenum"] = pd.to_numeric(lab["valuenum"], errors="coerce")
            lab["hours_from_icu"] = hours_from_icu(lab["charttime"], lab["intime"])
            lab = within_horizon(lab)
            lab["bin"] = assign_bin(lab)
            return lab
        else:
            chunks = []
            for chunk in pd.read_csv(
                hosp / "labevents.csv.gz",
                usecols=cols, chunksize=5_000_000, low_memory=False,
                parse_dates=["charttime"],
            ):
                filtered = chunk[chunk["itemid"].isin(itemids)].copy()
                if not filtered.empty:
                    merged = filtered.merge(stay_map, on="hadm_id", how="inner")
                    if not merged.empty:
                        chunks.append(merged)
            if not chunks:
                return pd.DataFrame(columns=["hadm_id", "itemid", "charttime", "valuenum", "stay_id", "intime", "hours_from_icu", "bin"])
            lab = pd.concat(chunks, ignore_index=True)
            lab["valuenum"] = pd.to_numeric(lab["valuenum"], errors="coerce")
            lab["hours_from_icu"] = hours_from_icu(lab["charttime"], lab["intime"])
            lab = within_horizon(lab)
            lab["bin"] = assign_bin(lab)
            return lab


    def _pull_last_lab(lab_df: pd.DataFrame, itemids: list[int],
                       name: str) -> pd.DataFrame:
        """Extract last value per (stay_id, bin) for a set of lab itemids."""
        sub = lab_df[lab_df["itemid"].isin(itemids)].dropna(subset=["valuenum"])
        if sub.empty:
            return pd.DataFrame(columns=["stay_id", "bin", name])
        return last_value_by_bin(sub, "stay_id", "valuenum", name)


    def _pull_last_chartevent(ce_df: pd.DataFrame, itemids: list[int],
                              name: str) -> pd.DataFrame:
        """Extract last value per (stay_id, bin) for chartevents."""
        sub = ce_df[ce_df["itemid"].isin(itemids)].copy()
        sub["valuenum"] = pd.to_numeric(sub["valuenum"], errors="coerce")
        sub = sub.dropna(subset=["valuenum"])
        if sub.empty:
            return pd.DataFrame(columns=["stay_id", "bin", name])
        return last_value_by_bin(sub, "stay_id", "valuenum", name)


    def _pull_iv_drug_rate(icu: Path, itemids: list[int], stay_times: pd.DataFrame,
                           name: str, nrows=None) -> pd.DataFrame:
        """Sum IV drug rate/amount per (stay_id, bin) from inputevents."""
        inp = _load_inputevents(icu, itemids, nrows)
        if inp.empty:
            return pd.DataFrame(columns=["stay_id", "bin", f"{name}_rate_sum"])
        inp = inp.merge(stay_times[["stay_id", "intime"]], on="stay_id", how="inner")
        inp["hours_from_icu"] = hours_from_icu(inp["starttime"], inp["intime"])
        inp["rate"] = pd.to_numeric(inp["rate"], errors="coerce").fillna(0)
        inp["amount"] = pd.to_numeric(inp["amount"], errors="coerce").fillna(0)
        inp = within_horizon(inp)
        inp["bin"] = assign_bin(inp)
        agg = inp.groupby(["stay_id", "bin"]).agg(
            **{f"{name}_rate_sum": ("rate", "sum"),
               f"{name}_amount_sum": ("amount", "sum")}
        ).reset_index()
        return agg


    def _prescriptions_flag(hosp: Path, stay_times: pd.DataFrame,
                            patterns: list[str], flag_name: str,
                            nrows=None) -> pd.DataFrame:
        """Binary per-bin flag for drug presence from prescriptions table."""
        rx_path = hosp / "prescriptions.csv.gz"
        if not rx_path.exists():
            return pd.DataFrame(columns=["stay_id", "bin", flag_name])
    
        cols = ["hadm_id", "drug", "starttime"]
        pattern_regex = "|".join(patterns)
    
        if nrows is not None:
            rx = pd.read_csv(rx_path, usecols=cols, nrows=nrows, low_memory=False)
            rx["drug_l"] = rx["drug"].fillna("").astype(str).str.lower()
            rx = rx[rx["drug_l"].str.contains(pattern_regex, regex=True)].copy()
        else:
            chunks = []
            for chunk in pd.read_csv(rx_path, usecols=cols, chunksize=2_000_000, low_memory=False):
                chunk["drug_l"] = chunk["drug"].fillna("").astype(str).str.lower()
                filtered = chunk[chunk["drug_l"].str.contains(pattern_regex, regex=True)].copy()
                if not filtered.empty:
                    chunks.append(filtered)
            rx = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
        
        if rx.empty:
            return pd.DataFrame(columns=["stay_id", "bin", flag_name])
        rx = rx.merge(stay_times[["stay_id", "hadm_id", "intime"]],
                      on="hadm_id", how="inner")
        rx = rx.dropna(subset=["starttime"])
        rx["starttime"] = pd.to_datetime(rx["starttime"], errors="coerce")
        rx["hours_from_icu"] = hours_from_icu(rx["starttime"], rx["intime"])
        rx = within_horizon(rx)
        rx["bin"] = assign_bin(rx)
        flag = (rx.drop_duplicates(["stay_id", "bin"])
                .assign(**{flag_name: 1})[["stay_id", "bin", flag_name]])
        return flag


    def time_varying(icu: Path, hosp: Path, stay_times: pd.DataFrame, nrows) -> pd.DataFrame:
        """Build all time-varying confounders (6h bins)."""
        ids = stay_times["stay_id"].drop_duplicates()
        out = base_time_bins(ids, "stay_id")
        stay_map = stay_times[["stay_id", "hadm_id", "intime"]].drop_duplicates(["stay_id", "hadm_id"])

        # ==================================================================
        # 1. VASOPRESSOR DOSE (sum of amount per bin)
        # ==================================================================
        vaso = _load_inputevents(icu, MIMIC_ITEMIDS["vasopressor"], nrows)
        if not vaso.empty:
            vaso = vaso.merge(stay_times[["stay_id", "intime"]], on="stay_id", how="inner")
            vaso["hours_from_icu"] = hours_from_icu(vaso["starttime"], vaso["intime"])
            vaso["amount"] = pd.to_numeric(vaso["amount"], errors="coerce")
            vaso = within_horizon(vaso)
            vaso["bin"] = assign_bin(vaso)
            out = out.merge(aggregate_numeric_by_bin(vaso, "stay_id", "amount", "vasopressor"),
                            on=["stay_id", "bin"], how="left")

        # ==================================================================
        # 2. MECHANICAL VENTILATION coverage
        # ==================================================================
        vent = pd.read_csv(
            icu / "procedureevents.csv.gz",
            usecols=["stay_id", "itemid", "starttime", "endtime"], nrows=nrows,
            low_memory=False, parse_dates=["starttime", "endtime"],
        )
        vent = vent[vent["itemid"].isin(MIMIC_ITEMIDS["vent_proc"] + MIMIC_ITEMIDS["vent_mode"])].copy()
        if not vent.empty:
            vent = vent.merge(stay_times[["stay_id", "intime"]], on="stay_id", how="inner")
            vent["sh"] = hours_from_icu(vent["starttime"], vent["intime"])
            vent["eh"] = hours_from_icu(vent["endtime"], vent["intime"])
            flag = coverage_flag_by_bin(vent.rename(columns={"sh": "start", "eh": "end"}),
                                        "stay_id", "start", "end", "mv_flag")
            out = out.merge(flag, on=["stay_id", "bin"], how="left")

        # ==================================================================
        # 3. RRT coverage
        # ==================================================================
        rrt2 = pd.read_csv(
            icu / "procedureevents.csv.gz",
            usecols=["stay_id", "itemid", "starttime", "endtime"], nrows=nrows,
            low_memory=False, parse_dates=["starttime", "endtime"],
        )
        rrt2 = rrt2[rrt2["itemid"].isin(MIMIC_ITEMIDS["rrt_proc"])].copy()
        if not rrt2.empty:
            rrt2 = rrt2.merge(stay_times[["stay_id", "intime"]], on="stay_id", how="inner")
            rrt2["sh"] = hours_from_icu(rrt2["starttime"], rrt2["intime"])
            rrt2["eh"] = hours_from_icu(rrt2["endtime"], rrt2["intime"])
            flag = coverage_flag_by_bin(rrt2.rename(columns={"sh": "start", "eh": "end"}),
                                        "stay_id", "start", "end", "rrt_flag")
            out = out.merge(flag, on=["stay_id", "bin"], how="left")

        # ==================================================================
        # 4. VITAL SIGNS from chartevents (NEW)
        # ==================================================================
        vital_ids = []
        for key in ["heart_rate", "sbp", "dbp", "map", "spo2", "temperature", "fio2"]:
            vital_ids.extend(MIMIC_ITEMIDS[key])
        # Also include existing scores
        score_ids = (MIMIC_ITEMIDS["sofa_total"] + MIMIC_ITEMIDS["gcs_eye"] +
                     MIMIC_ITEMIDS["gcs_verbal"] + MIMIC_ITEMIDS["gcs_motor"] +
                     MIMIC_ITEMIDS["rass"] + MIMIC_ITEMIDS["resp_rate"])
        all_ce_ids = list(set(vital_ids + score_ids))

        ce = _load_chartevents(icu, all_ce_ids, nrows)
        if not ce.empty:
            ce = ce.merge(stay_times[["stay_id", "intime"]], on="stay_id", how="inner")
            ce["hours_from_icu"] = hours_from_icu(ce["charttime"], ce["intime"])
            ce = within_horizon(ce)
            ce["bin"] = assign_bin(ce)

            # Temperature normalization: convert F → C
            temp_f_mask = ce["itemid"].isin([223762])  # 223762 = Temp(F)
            ce.loc[temp_f_mask, "valuenum"] = (
                pd.to_numeric(ce.loc[temp_f_mask, "valuenum"], errors="coerce") - 32
            ) * 5 / 9

            # Existing scores
            out = out.merge(_pull_last_chartevent(ce, MIMIC_ITEMIDS["sofa_total"], "sofa_stored"),
                            on=["stay_id", "bin"], how="left")
            out = out.merge(_pull_last_chartevent(ce, MIMIC_ITEMIDS["rass"], "rass"),
                            on=["stay_id", "bin"], how="left")
            out = out.merge(_pull_last_chartevent(ce, MIMIC_ITEMIDS["resp_rate"], "resp_rate"),
                            on=["stay_id", "bin"], how="left")

            # NEW vital signs
            for vital_key in ["heart_rate", "sbp", "dbp", "map", "spo2",
                              "temperature", "fio2"]:
                out = out.merge(
                    _pull_last_chartevent(ce, MIMIC_ITEMIDS[vital_key], vital_key),
                    on=["stay_id", "bin"], how="left"
                )

        # ==================================================================
        # 5. CAM-ICU delirium
        # ==================================================================
        cam = _load_chartevents(icu, MIMIC_ITEMIDS["cam_icu"], nrows)
        if not cam.empty:
            cam = cam.merge(stay_times[["stay_id", "intime"]], on="stay_id", how="inner")
            cam["hours_from_icu"] = hours_from_icu(cam["charttime"], cam["intime"])
            cam = within_horizon(cam)
            cam["bin"] = assign_bin(cam)
            cam["v"] = pd.to_numeric(cam["valuenum"], errors="coerce")
            cam_pos = (cam.groupby(["stay_id", "bin"])["v"].max()
                       .rename("cam_icu_max").reset_index())
            out = out.merge(cam_pos, on=["stay_id", "bin"], how="left")

        # ==================================================================
        # 6. STEROIDS from prescriptions
        # ==================================================================
        ster = _prescriptions_flag(hosp, stay_times, MIMIC_STEROID_DRUG_PATTERNS,
                                   "steroid_flag", nrows)
        if not ster.empty:
            out = out.merge(ster, on=["stay_id", "bin"], how="left")

        # ==================================================================
        # 7. LABORATORY VALUES (NEW — comprehensive panel)
        # ==================================================================
        lab_keys = [
            "lactate", "albumin", "prealbumin", "bun", "creatinine",
            "wbc", "hemoglobin", "platelet", "sodium", "potassium",
            "chloride", "bicarbonate", "pao2", "pco2", "ph",
            "magnesium", "phosphate", "ast", "alt", "bilirubin",
            "crp", "triglycerides",
        ]
        all_lab_ids = []
        for key in lab_keys:
            all_lab_ids.extend(MIMIC_ITEMIDS[key])

        lab = _load_labevents(hosp, all_lab_ids, stay_map, nrows)
        if not lab.empty:
            for key in lab_keys:
                out = out.merge(
                    _pull_last_lab(lab, MIMIC_ITEMIDS[key], key),
                    on=["stay_id", "bin"], how="left"
                )

        # ==================================================================
        # 8. IV MEDICATIONS (NEW — sedatives, opioids, antibiotics, etc.)
        # ==================================================================
        iv_drugs = [
            ("propofol", MIMIC_ITEMIDS["propofol"]),
            ("midazolam", MIMIC_ITEMIDS["midazolam"]),
            ("dexmedetomidine", MIMIC_ITEMIDS["dexmedetomidine"]),
            ("fentanyl", MIMIC_ITEMIDS["fentanyl"]),
            ("morphine", MIMIC_ITEMIDS["morphine"]),
            ("hydromorphone", MIMIC_ITEMIDS["hydromorphone"]),
            ("vancomycin", MIMIC_ITEMIDS["vancomycin"]),
            ("piperacillin_tazo", MIMIC_ITEMIDS["piperacillin_tazo"]),
            ("meropenem", MIMIC_ITEMIDS["meropenem"]),
            ("cefepime", MIMIC_ITEMIDS["cefepime"]),
            ("metronidazole", MIMIC_ITEMIDS["metronidazole"]),
            ("cisatracurium", MIMIC_ITEMIDS["cisatracurium"]),
            ("furosemide", MIMIC_ITEMIDS["furosemide"]),
            ("heparin", MIMIC_ITEMIDS["heparin"]),
        ]
        for drug_name, drug_ids in iv_drugs:
            drug_df = _pull_iv_drug_rate(icu, drug_ids, stay_times, drug_name, nrows)
            if not drug_df.empty:
                out = out.merge(drug_df, on=["stay_id", "bin"], how="left")

        # ==================================================================
        # 9. PARENTERAL NUTRITION & DEXTROSE (NEW — critical confounders!)
        # ==================================================================
        pn_df = _pull_iv_drug_rate(icu, MIMIC_ITEMIDS["parenteral_nutrition"],
                                    stay_times, "parenteral_nutrition", nrows)
        if not pn_df.empty:
            out = out.merge(pn_df, on=["stay_id", "bin"], how="left")
        dex_df = _pull_iv_drug_rate(icu, MIMIC_ITEMIDS["dextrose"],
                                     stay_times, "dextrose", nrows)
        if not dex_df.empty:
            out = out.merge(dex_df, on=["stay_id", "bin"], how="left")

        # ==================================================================
        # 10. ORAL/ENTERAL DRUG FLAGS from prescriptions (NEW)
        # ==================================================================
        rx_flags = [
            (MIMIC_PROKINETIC_DRUG_PATTERNS, "prokinetic_flag"),
            (MIMIC_ORAL_HYPOGLYCEMIC_PATTERNS, "oral_hypoglycemic_flag"),
            (MIMIC_BETABLOCKER_PATTERNS, "betablocker_flag"),
            (MIMIC_ANTIPSYCHOTIC_PATTERNS, "antipsychotic_flag"),
            (MIMIC_PPI_H2_PATTERNS, "ppi_h2_flag"),
            (MIMIC_ANTIEMETIC_PATTERNS, "antiemetic_flag"),
            (MIMIC_IMMUNOSUPPRESSANT_PATTERNS, "immunosuppressant_flag"),
        ]
        for patterns, flag_name in rx_flags:
            flag_df = _prescriptions_flag(hosp, stay_times, patterns, flag_name, nrows)
            if not flag_df.empty:
                out = out.merge(flag_df, on=["stay_id", "bin"], how="left")

        # Composite flags for convenience.
        opioid_cols = [c for c in out.columns
                       if any(c.startswith(f"{d}_") for d in ["fentanyl", "morphine", "hydromorphone"])]
        if opioid_cols:
            out["opioid_any"] = (out[opioid_cols].fillna(0).sum(axis=1) > 0).astype(int)

        abx_cols = [c for c in out.columns
                    if any(c.startswith(f"{d}_") for d in ["vancomycin", "piperacillin_tazo",
                                                            "meropenem", "cefepime", "metronidazole"])]
        if abx_cols:
            out["antibiotic_any"] = (out[abx_cols].fillna(0).sum(axis=1) > 0).astype(int)

        return out


    def baseline(icu: Path, hosp: Path, stays: pd.DataFrame, nrows=None) -> pd.DataFrame:
        """Admission-level covariates, collapsed to one row per stay_id."""
        base = stays[["stay_id", "hadm_id", "subject_id"]].drop_duplicates("stay_id").copy()

        # ---- age / sex ----
        pat = pd.read_csv(hosp / "patients.csv.gz",
                          usecols=["subject_id", "anchor_age", "gender"],
                          low_memory=False)
        base = base.merge(pat, on="subject_id", how="left")

        # ---- admission info ----
        adm_path = hosp / "admissions.csv.gz"
        if adm_path.exists():
            adm = pd.read_csv(adm_path,
                              usecols=["hadm_id", "admission_type", "race",
                                       "admittime", "dischtime"],
                              low_memory=False, parse_dates=["admittime", "dischtime"])
            adm = adm.rename(columns={"race": "ethnicity"})
            base = base.merge(adm[["hadm_id", "admission_type", "ethnicity"]],
                              on="hadm_id", how="left")
            # Time from hospital admission to ICU admission
            icu_info = stays[["stay_id", "hadm_id", "intime", "first_careunit"]].drop_duplicates("stay_id")
            merged_adm = icu_info.merge(adm[["hadm_id", "admittime"]], on="hadm_id", how="left")
            merged_adm["los_before_icu_hours"] = (
                hours_from_icu(merged_adm["intime"], merged_adm["admittime"])
            )
            base = base.merge(merged_adm[["stay_id", "los_before_icu_hours"]],
                              on="stay_id", how="left")

        # ---- ICU type ----
        if "first_careunit" in stays.columns:
            icu_type = stays[["stay_id", "first_careunit"]].drop_duplicates("stay_id")
            base = base.merge(icu_type, on="stay_id", how="left")

        # ---- ICD-based diagnoses ----
        icd = pd.read_csv(hosp / "diagnoses_icd.csv.gz", low_memory=False)
        icd["icd_code_norm"] = (
            icd["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False).str.strip()
        )

        # Diabetes history
        dm_rows = []
        for ver, prefixes in ICD_DIABETES.items():
            norm_prefixes = tuple(p.replace(".", "").upper() for p in prefixes)
            m = (icd["icd_version"] == ver) & icd["icd_code_norm"].str.startswith(norm_prefixes)
            dm_rows.append(icd.loc[m, ["hadm_id"]])
        dm = (pd.concat(dm_rows).drop_duplicates() if dm_rows else pd.DataFrame(columns=["hadm_id"]))
        dm["diabetes"] = 1
        base = base.merge(dm, on="hadm_id", how="left")

        # Psychiatric history
        for label, mapping in ICD_PSYCHIATRIC.items():
            rows = []
            for ver, prefixes in mapping.items():
                norm_prefixes = tuple(p.replace(".", "").upper() for p in prefixes)
                m = (icd["icd_version"] == ver) & icd["icd_code_norm"].str.startswith(norm_prefixes)
                rows.append(icd.loc[m, ["hadm_id"]])
            sub = pd.concat(rows).drop_duplicates() if rows else pd.DataFrame(columns=["hadm_id"])
            sub[f"hx_{label}"] = 1
            base = base.merge(sub.rename(columns={"hadm_id": "hadm_id"}), on="hadm_id", how="left")

        # NEW: Additional comorbidities
        for label, icd_map in [("hypertension", ICD_HYPERTENSION),
                               ("ckd", ICD_CKD),
                               ("liver_disease", ICD_LIVER_DISEASE),
                               ("obesity", ICD_OBESITY)]:
            rows = []
            for ver, prefixes in icd_map.items():
                norm_prefixes = tuple(p.replace(".", "").upper() for p in prefixes)
                m = (icd["icd_version"] == ver) & icd["icd_code_norm"].str.startswith(norm_prefixes)
                rows.append(icd.loc[m, ["hadm_id"]])
            sub = pd.concat(rows).drop_duplicates() if rows else pd.DataFrame(columns=["hadm_id"])
            sub[f"hx_{label}"] = 1
            base = base.merge(sub, on="hadm_id", how="left")

        # NEW: Charlson Comorbidity Index
        try:
            cci = charlson_from_icd(hosp, stays[["stay_id", "hadm_id"]].drop_duplicates("stay_id"))
            base = base.merge(cci[["stay_id", "charlson_index"]], on="stay_id", how="left")
        except Exception as e:
            print(f"Charlson computation skipped: {e}")

        # ---- weight / height / BMI ----
        w = _load_chartevents(icu, MIMIC_ITEMIDS["daily_weight"], nrows)
        if not w.empty:
            w["valuenum"] = pd.to_numeric(w["valuenum"], errors="coerce")
            adm_w = (w.dropna(subset=["valuenum"]).sort_values("charttime")
                     .groupby("stay_id")["valuenum"].first().rename("weight_kg").reset_index())
            base = base.merge(adm_w, on="stay_id", how="left")
        h = _load_chartevents(icu, MIMIC_ITEMIDS["height_cm"], nrows)
        if not h.empty:
            h["valuenum"] = pd.to_numeric(h["valuenum"], errors="coerce")
            adm_h = (h.dropna(subset=["valuenum"]).sort_values("charttime")
                     .groupby("stay_id")["valuenum"].first().rename("height_cm").reset_index())
            base = base.merge(adm_h, on="stay_id", how="left")

        # BMI
        if "weight_kg" in base.columns and "height_cm" in base.columns:
            h_m = base["height_cm"] / 100.0
            base["bmi"] = base["weight_kg"] / (h_m ** 2)
            # Clip implausible BMI values
            base.loc[(base["bmi"] < 10) | (base["bmi"] > 100), "bmi"] = np.nan

        # ---- admission SOFA ----
        try:
            from clinical_definitions import compute_sofa as _cs
            sofa_df = _cs(icu, hosp, stays[["stay_id", "hadm_id", "intime"]].copy(), nrows)
            if "sofa_total" in sofa_df.columns:
                adm_s = (sofa_df.dropna(subset=["sofa_total"]).sort_values(["stay_id", "bin"])
                         .groupby("stay_id")["sofa_total"].first().rename("admission_sofa").reset_index())
                base = base.merge(adm_s, on="stay_id", how="left")
        except Exception as e:
            print(f"admission SOFA skipped: {e}")

        return base


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", default="configs/paths.json")
        ap.add_argument("--smoke", action="store_true")
        args = ap.parse_args()
        cfg = load_config(args.config)
        icu = Path(cfg["mimic_icu"])
        hosp = Path(cfg["mimic_hosp"])
        nrows = 1_500_000 if args.smoke else None

        stays = pd.read_csv(icu / "icustays.csv.gz", parse_dates=["intime", "outtime"], low_memory=False)
        if args.smoke:
            stays = stays.head(5000)
        stay_times = stays[["stay_id", "hadm_id", "subject_id", "intime", "outtime"]].copy()

        tv = time_varying(icu, hosp, stay_times, nrows)

        # Compute SOFA from components (stored aggregate is empty in MIMIC-IV).
        print("Computing SOFA from organ components...")
        sofa_df = compute_sofa(icu, hosp, stay_times, nrows)
        sofa_df = sofa_df.rename(columns={"sofa_total": "sofa"})[["stay_id", "bin", "sofa"]]
        tv = tv.drop(columns=[c for c in ["sofa", "sofa_stored"] if c in tv.columns], errors="ignore")
        tv = tv.merge(sofa_df, on=["stay_id", "bin"], how="left")

        bl = baseline(icu, hosp, stays, nrows)

        out = tv.merge(bl, on="stay_id", how="left")

        # ---- Tidy flags ----
        binary_flags = ["mv_flag", "rrt_flag", "steroid_flag", "diabetes",
                        "prokinetic_flag", "oral_hypoglycemic_flag", "betablocker_flag",
                        "antipsychotic_flag", "ppi_h2_flag", "antiemetic_flag",
                        "immunosuppressant_flag", "opioid_any", "antibiotic_any"]
        for c in binary_flags:
            if c in out.columns:
                out[c] = out[c].fillna(0).astype(int)

        zero_fill = ["vasopressor_sum", "vasopressor_count"]
        # Also zero-fill all IV drug rate columns
        zero_fill += [c for c in out.columns
                      if c.endswith("_rate_sum") or c.endswith("_amount_sum")]
        for c in zero_fill:
            if c in out.columns:
                out[c] = out[c].fillna(0)

        # Ensure essential columns exist even if data was empty
        for c in ["vasopressor_sum", "vasopressor_count", "mv_flag", "rrt_flag",
                  "steroid_flag", "sofa", "rass", "resp_rate", "cam_icu_max",
                  "heart_rate", "sbp", "dbp", "map", "spo2", "temperature"]:
            if c not in out.columns:
                out[c] = np.nan

        out = out.sort_values(["stay_id", "bin"])
        path = project_path(cfg, "data", "interim",
                            "mimic_confounders_smoke.csv" if args.smoke else "mimic_confounders.csv")
        out.to_csv(path, index=False)
        print(f"Wrote {path} ({len(out):,} rows, {out.shape[1]} columns)")

        # Coverage report
        print("\nNon-null coverage (rows with value):")
        report_cols = [
            # Original
            "vasopressor_sum", "mv_flag", "rrt_flag", "steroid_flag",
            "sofa", "rass", "diabetes", "weight_kg", "height_cm",
            # New vitals
            "heart_rate", "sbp", "spo2", "temperature", "fio2",
            # New labs
            "lactate", "albumin", "potassium", "creatinine", "wbc", "crp",
            # New drugs
            "propofol_rate_sum", "fentanyl_rate_sum", "vancomycin_rate_sum",
            "parenteral_nutrition_rate_sum", "dextrose_rate_sum",
            # New flags
            "opioid_any", "antibiotic_any", "oral_hypoglycemic_flag",
            "betablocker_flag", "prokinetic_flag",
            # New baseline
            "bmi", "charlson_index", "admission_type",
        ]
        for c in report_cols:
            if c in out.columns:
                n = out[c].notna().sum()
                pct = 100 * out[c].notna().mean()
                print(f"  {c:40s}: {n:>10,}  ({pct:5.1f}%)")

    return SimpleNamespace(**locals())

_confounders = _load_confounders()


# ==============================================================================
# Glucose time series
# ==============================================================================

def _load_glucose():
    """Build a high-frequency glucose table merging labs + POC finger sticks.

    The original cohort only used laboratory glucose (labevents 50931), yielding
    ~2% bin coverage. Point-of-care finger sticks (chartevents 225664 etc.) are
    far more frequent in ICU and are needed for time-in-range, coefficient of
    variation, and hypo/hyperglycemia outcomes. This module merges all glucose
    sources, bins them, and derives the glycemic outcomes that the analysis plan
    promises but the original code never computed.
    """

    import argparse
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from clinical_definitions import load_config, project_path
    from clinical_definitions import BIN_HOURS, MAX_HOURS, MIMIC_ITEMIDS
    from clinical_definitions import base_time_bins, hours_from_icu, within_horizon, assign_bin

    # Time-in-range / thresholds (mg/dL), per analysis plan.
    TIR_LO, TIR_HI = 70, 180
    HYPER_SEVERE = 250
    HYPO = 70


    def _load_labevents_itemids(
        hosp: Path,
        itemids: list[int],
        nrows: int | None = None,
        chunksize: int = 10_000_000,
    ) -> pd.DataFrame:
        cols = ["hadm_id", "itemid", "charttime", "valuenum"]
        if nrows is not None:
            df = pd.read_csv(
                hosp / "labevents.csv.gz",
                usecols=cols,
                nrows=nrows,
                low_memory=False,
                parse_dates=["charttime"],
            )
            return df[df["itemid"].isin(itemids)].copy()

        chunks = []
        for chunk in pd.read_csv(
            hosp / "labevents.csv.gz",
            usecols=cols,
            chunksize=chunksize,
            low_memory=False,
            parse_dates=["charttime"],
        ):
            filtered = chunk[chunk["itemid"].isin(itemids)].copy()
            if not filtered.empty:
                chunks.append(filtered)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)


    def _load_chartevents_itemids(
        icu: Path,
        itemids: list[int],
        nrows: int | None = None,
        chunksize: int = 10_000_000,
    ) -> pd.DataFrame:
        cols = ["stay_id", "itemid", "charttime", "valuenum"]
        if nrows is not None:
            df = pd.read_csv(
                icu / "chartevents.csv.gz",
                usecols=cols,
                nrows=nrows,
                low_memory=False,
                parse_dates=["charttime"],
            )
            return df[df["itemid"].isin(itemids)].copy()

        chunks = []
        for chunk in pd.read_csv(
            icu / "chartevents.csv.gz",
            usecols=cols,
            chunksize=chunksize,
            low_memory=False,
            parse_dates=["charttime"],
        ):
            filtered = chunk[chunk["itemid"].isin(itemids)].copy()
            if not filtered.empty:
                chunks.append(filtered)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", default="configs/paths.json")
        ap.add_argument("--smoke", action="store_true")
        args = ap.parse_args()
        cfg = load_config(args.config)
        icu = Path(cfg["mimic_icu"])
        hosp = Path(cfg["mimic_hosp"])
        nrows = 1_500_000 if args.smoke else None

        stays = pd.read_csv(icu / "icustays.csv.gz", parse_dates=["intime", "outtime"], low_memory=False)
        if args.smoke:
            stays = stays.head(5000)
        stay_times = stays[["stay_id", "hadm_id", "intime"]].drop_duplicates("hadm_id")

        # map tables: stay_id -> (hadm_id, intime)
        stay_map = stays[["stay_id", "hadm_id", "intime"]].drop_duplicates("stay_id")

        # lab glucose (50931): map hadm_id -> stay_id
        lab = _load_labevents_itemids(hosp, MIMIC_ITEMIDS["glucose_lab"], nrows)
        lab = lab.merge(stay_map, on="hadm_id", how="inner")
        lab["source"] = "lab"

        # POC finger-stick glucose (chartevents): already has stay_id
        poc = _load_chartevents_itemids(icu, MIMIC_ITEMIDS["glucose_poc"], nrows)
        poc = poc.merge(stay_map[["stay_id", "intime"]], on="stay_id", how="inner")
        poc["source"] = "poc"

        glu = pd.concat([lab[["stay_id", "intime", "charttime", "valuenum", "source"]],
                         poc[["stay_id", "intime", "charttime", "valuenum", "source"]]],
                        ignore_index=True)
        glu["glucose_mg_dl"] = pd.to_numeric(glu["valuenum"], errors="coerce")
        glu = glu.dropna(subset=["glucose_mg_dl"])
        glu = glu[(glu["glucose_mg_dl"] > 10) & (glu["glucose_mg_dl"] < 2000)].copy()
        glu["hours_from_icu"] = hours_from_icu(glu["charttime"], glu["intime"])
        glu = within_horizon(glu)
        glu["bin"] = assign_bin(glu)

        out = base_time_bins(stays["stay_id"], "stay_id")
        agg = (
            glu.groupby(["stay_id", "bin"])["glucose_mg_dl"]
            .agg(glucose_mean="mean", glucose_min="min", glucose_max="max",
                 glucose_std="std", glucose_count="count")
            .reset_index()
        )
        out = out.merge(agg, on=["stay_id", "bin"], how="left")

        # coefficient of variation within bin
        out["glucose_cv"] = out["glucose_std"] / out["glucose_mean"]

        # time-in-range / hypo / hyper from raw measurements re-aggregated
        flags = glu.assign(
            in_range=((glu["glucose_mg_dl"] >= TIR_LO) & (glu["glucose_mg_dl"] <= TIR_HI)).astype(int),
            hypo=(glu["glucose_mg_dl"] < HYPO).astype(int),
            hyper_severe=(glu["glucose_mg_dl"] > HYPER_SEVERE).astype(int),
        )
        tir = (flags.groupby(["stay_id", "bin"])
               .agg(tir_fraction=("in_range", "mean"),
                    hypo_fraction=("hypo", "mean"),
                    hyper_severe_fraction=("hyper_severe", "mean"),
                    n_glucose=("glucose_mg_dl", "count")).reset_index())
        out = out.merge(tir, on=["stay_id", "bin"], how="left")

        # future 24h mean glucose (next 4 bins)
        out = out.sort_values(["stay_id", "bin"])
        out["future_24h_glucose_mean"] = (
            out.groupby("stay_id")["glucose_mean"]
            .transform(lambda s: s.shift(-1).rolling(4, min_periods=1).mean().shift(-3))
        )
        out["future_24h_tir"] = (
            out.groupby("stay_id")["tir_fraction"]
            .transform(lambda s: s.shift(-1).rolling(4, min_periods=1).mean().shift(-3))
        )

        out = out.sort_values(["stay_id", "bin"])
        path = project_path(cfg, "data", "interim",
                            "mimic_glucose_full_smoke.csv" if args.smoke else "mimic_glucose_full.csv")
        out.to_csv(path, index=False)
        print(f"Wrote {path} ({len(out):,} rows)")
        nz = out["glucose_count"].fillna(0).gt(0)
        print(f"bins with glucose: {nz.sum():,} ({100*nz.mean():.1f}%)")
        print(f"tir_fraction among measured: median={out.loc[nz,'tir_fraction'].median():.2f}")

    return SimpleNamespace(**locals())

_glucose = _load_glucose()


# ==============================================================================
# Optional note-derived features
# ==============================================================================

def _load_notes():
    import argparse
    import re
    from pathlib import Path

    import pandas as pd

    from clinical_definitions import load_config, project_path


    BIN_HOURS = 6
    MAX_HOURS = 72

    CONCEPT_PATTERNS = {
        "note_enteral_mentions": r"\b(enteral|tube feeds?|tube feeding|feeding tube|jevity|nepro|glucerna|osmolite|nutren|peptamen|replete|promote|ensure)\b",
        "note_npo_interruption_mentions": r"\b(npo|nothing by mouth|hold tube feeds?|held tube feeds?|feeding held|tube feeds held|residual|emesis|vomit|aspiration|dysphagia|failed swallow)\b",
        "note_glucose_mentions": r"\b(glucose|hyperglyc|hypoglyc|blood sugar|fingerstick|fsbg|a1c|hba1c)\b",
        "note_insulin_mentions": r"\b(insulin|glargine|lispro|aspart|nph|regular insulin)\b",
        "note_delirium_mentions": r"\b(delirium|confusion|confused|altered mental status|encephalopathy|agitation|cam-icu|rass)\b",
        "note_mental_health_mentions": r"\b(depression|depressed|anxiety|ptsd|bipolar|psychiatric|suicidal)\b",
        "note_infection_mentions": r"\b(sepsis|septic|infection|pneumonia|bacteremia|uti|antibiotic)\b",
        "note_ventilation_mentions": r"\b(intubat|ventilat|mechanical ventilation|extubat|trach)\b",
    }


    def count_pattern(text: object, pattern: str) -> int:
        return len(re.findall(pattern, str(text).lower()))


    def load_notes(note_root: Path, smoke: bool) -> pd.DataFrame:
        frames = []
        nrows = 100_000 if smoke else None
        for table in ["radiology.csv.gz", "discharge.csv.gz"]:
            path = note_root / table
            if not path.exists():
                continue
            df = pd.read_csv(
                path,
                usecols=["note_id", "subject_id", "hadm_id", "note_type", "charttime", "text"],
                parse_dates=["charttime"],
                nrows=nrows,
                low_memory=False,
            )
            df["note_source_table"] = table
            df["is_discharge_note"] = int(table.startswith("discharge"))
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="configs/paths.json")
        parser.add_argument("--smoke", action="store_true")
        args = parser.parse_args()
        config = load_config(args.config)

        note_root = Path(config["mimic_note"])
        icu_root = Path(config["mimic_icu"])
        stays = pd.read_csv(
            icu_root / "icustays.csv.gz",
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
            parse_dates=["intime", "outtime"],
            low_memory=False,
        )
        if args.smoke:
            stays = stays.head(5000)

        notes = load_notes(note_root, args.smoke)
        if notes.empty:
            out = project_path(config, "data", "interim", "mimic_note_features_smoke.csv" if args.smoke else "mimic_note_features.csv")
            pd.DataFrame().to_csv(out, index=False)
            print(f"Wrote empty {out}")
            return

        notes = notes.dropna(subset=["hadm_id", "charttime"]).copy()
        notes["hadm_id"] = notes["hadm_id"].astype(stays["hadm_id"].dtype, copy=False)
        merged = notes.merge(stays, on=["subject_id", "hadm_id"], how="inner")
        merged["hours_from_icu"] = (merged["charttime"] - merged["intime"]).dt.total_seconds() / 3600
        merged = merged[(merged["hours_from_icu"] >= 0) & (merged["hours_from_icu"] < MAX_HOURS)].copy()
        merged["bin"] = (merged["hours_from_icu"] // BIN_HOURS).astype(int)
        merged["note_char_count"] = merged["text"].fillna("").astype(str).str.len()
        merged["note_word_count"] = merged["text"].fillna("").astype(str).str.split().str.len()
        for col, pattern in CONCEPT_PATTERNS.items():
            merged[col] = merged["text"].apply(lambda x: count_pattern(x, pattern))

        group_cols = ["subject_id", "hadm_id", "stay_id", "bin"]
        agg_map = {
            "note_id": "count",
            "note_char_count": "sum",
            "note_word_count": "sum",
            "is_discharge_note": "sum",
        }
        for col in CONCEPT_PATTERNS:
            agg_map[col] = "sum"

        features = merged.groupby(group_cols).agg(agg_map).reset_index()
        features = features.rename(columns={"note_id": "note_count", "is_discharge_note": "discharge_note_count"})
        features["has_potential_note_leakage"] = features["discharge_note_count"].gt(0).astype(int)

        out = project_path(config, "data", "interim", "mimic_note_features_smoke.csv" if args.smoke else "mimic_note_features.csv")
        features.to_csv(out, index=False)
        print(f"Wrote {out} ({len(features)} note-feature rows)")

    return SimpleNamespace(**locals())

_notes = _load_notes()


# ==============================================================================
# Clinical outcomes
# ==============================================================================

def _load_outcomes():
    """Build clinical hard endpoints and the delirium (brain-function) outcome.

    Hard endpoints (mortality/LOS) anchor the clinical "so what" question and
    support the mediation analysis (glucose as mediator of nutrition -> death).
    Delirium is the third leg of the nutrition-glucose-brain function triangle:
    nutrition -> glycemic stability -> delirium has a plausible neuro-metabolic
    pathway and is the honest replacement for the dropped NHANES-prior-GNN story.
    """

    import argparse
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from clinical_definitions import load_config, project_path
    from clinical_definitions import MIMIC_ITEMIDS
    from clinical_definitions import base_time_bins, hours_from_icu, within_horizon, assign_bin


    def stay_level_outcomes(icu: Path, hosp: Path, stays: pd.DataFrame) -> pd.DataFrame:
        base = stays[["stay_id", "hadm_id", "subject_id", "intime", "outtime"]].copy()

        # hospital mortality
        adm = pd.read_csv(hosp / "admissions.csv.gz",
                          usecols=["hadm_id", "deathtime", "hospital_expire_flag", "dischtime"],
                          parse_dates=["deathtime", "dischtime"], low_memory=False)
        base = base.merge(adm[["hadm_id", "hospital_expire_flag", "deathtime", "dischtime"]],
                          on="hadm_id", how="left")
        base["hospital_mortality"] = base["hospital_expire_flag"].fillna(0).astype(int)

        # ICU mortality: death time within [intime, outtime]
        base["icu_mortality"] = (
            base["deathtime"].notna() &
            (base["deathtime"] >= base["intime"]) &
            (base["deathtime"] <= base["outtime"])
        ).astype(int)

        # 28-day mortality from patients.dod
        pat = pd.read_csv(hosp / "patients.csv.gz", usecols=["subject_id", "dod"],
                          parse_dates=["dod"], low_memory=False)
        base = base.merge(pat, on="subject_id", how="left")
        base["mortality_28d"] = (
            base["dod"].notna() &
            ((base["dod"] - base["intime"]).dt.days <= 28)
        ).astype(int)

        # ICU length of stay (hours)
        base["icu_los_hours"] = (base["outtime"] - base["intime"]).dt.total_seconds() / 3600.0
        base["icu_los_hours"] = base["icu_los_hours"].clip(upper=720)  # 30-day cap

        # ventilator-free days (first 28d), from procedureevents intubation spans
        pe = pd.read_csv(icu / "procedureevents.csv.gz",
                         usecols=["stay_id", "itemid", "starttime", "endtime"],
                         low_memory=False, parse_dates=["starttime", "endtime"])
        vent = pe[pe["itemid"].isin(MIMIC_ITEMIDS["vent_proc"])].copy()
        if not vent.empty:
            vent = vent.merge(stays[["stay_id", "intime"]], on="stay_id", how="inner")
            vent["dur_h"] = (vent["endtime"] - vent["starttime"]).dt.total_seconds() / 3600.0
            vent["dur_h"] = vent["dur_h"].clip(lower=0)
            ventdays = vent.groupby("stay_id")["dur_h"].sum().rename("vent_hours").reset_index()
            base = base.merge(ventdays, on="stay_id", how="left")
            base["vent_hours"] = base["vent_hours"].fillna(0)
            # ventilator-free days in first 28 days
            base["vent_free_days_28"] = (28 - base["vent_hours"] / 24.0).clip(lower=0, upper=28)
        else:
            base["vent_hours"] = 0.0
            base["vent_free_days_28"] = 28.0

        return base[["stay_id", "hospital_mortality", "icu_mortality", "mortality_28d",
                     "icu_los_hours", "vent_hours", "vent_free_days_28"]]


    def _load_chartevents_itemids(
        icu: Path,
        itemids: list[int],
        nrows: int | None = None,
        chunksize: int = 10_000_000,
    ) -> pd.DataFrame:
        cols = ["stay_id", "itemid", "charttime", "valuenum"]
        if nrows is not None:
            df = pd.read_csv(
                icu / "chartevents.csv.gz",
                usecols=cols,
                nrows=nrows,
                low_memory=False,
                parse_dates=["charttime"],
            )
            return df[df["itemid"].isin(itemids)].copy()

        chunks = []
        for chunk in pd.read_csv(
            icu / "chartevents.csv.gz",
            usecols=cols,
            chunksize=chunksize,
            low_memory=False,
            parse_dates=["charttime"],
        ):
            filtered = chunk[chunk["itemid"].isin(itemids)].copy()
            if not filtered.empty:
                chunks.append(filtered)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)


    def delirium_outcomes(icu: Path, stay_times: pd.DataFrame, nrows) -> pd.DataFrame:
        """Bin-level delirium flags and stay-level delirium-free days.

        Uses CAM-ICU components (positive feature = 1) plus RASS <-3 as a severe
        sedation proxy. Stay-level delirium-free days within the 72h horizon.
        """
        out = base_time_bins(stay_times["stay_id"], "stay_id")
        ce = _load_chartevents_itemids(icu, MIMIC_ITEMIDS["cam_icu"] + MIMIC_ITEMIDS["rass"], nrows)
        if ce.empty:
            out["delirium_flag"] = 0
            out["deep_sedation_flag"] = 0
            return out
        ce = ce.merge(stay_times[["stay_id", "intime"]], on="stay_id", how="inner")
        ce["hours_from_icu"] = hours_from_icu(ce["charttime"], ce["intime"])
        ce = within_horizon(ce)
        ce["bin"] = assign_bin(ce)
        ce["v"] = pd.to_numeric(ce["valuenum"], errors="coerce")

        # CAM-ICU: any positive component present in bin -> possible delirium
        cam = ce[ce["itemid"].isin(MIMIC_ITEMIDS["cam_icu"])]
        if not cam.empty:
            cam_flag = (cam.groupby(["stay_id", "bin"])["v"].max()
                        .rename("delirium_flag").reset_index())
            cam_flag["delirium_flag"] = (cam_flag["delirium_flag"] >= 1).astype(int)
            out = out.merge(cam_flag, on=["stay_id", "bin"], how="left")

        # RASS <= -3 deep sedation
        rass = ce[ce["itemid"].isin(MIMIC_ITEMIDS["rass"])]
        if not rass.empty:
            rass_flag = (rass.groupby(["stay_id", "bin"])["v"].min()
                         .rename("min_rass").reset_index())
            rass_flag["deep_sedation_flag"] = (rass_flag["min_rass"] <= -3).astype(int)
            out = out.merge(rass_flag[["stay_id", "bin", "deep_sedation_flag"]],
                            on=["stay_id", "bin"], how="left")

        out["delirium_flag"] = out.get("delirium_flag", 0).fillna(0).astype(int)
        out["deep_sedation_flag"] = out.get("deep_sedation_flag", 0).fillna(0).astype(int)

        # delirium-free days over the 72h horizon (12 bins = 0.5 days each)
        dfd = (out.groupby("stay_id")["delirium_flag"].apply(
            lambda s: (12 - s.sum()) * 0.5).rename("delirium_free_days_72h").reset_index())
        out = out.merge(dfd, on="stay_id", how="left")
        return out


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", default="configs/paths.json")
        ap.add_argument("--smoke", action="store_true")
        args = ap.parse_args()
        cfg = load_config(args.config)
        icu = Path(cfg["mimic_icu"])
        hosp = Path(cfg["mimic_hosp"])
        nrows = 1_500_000 if args.smoke else None

        stays = pd.read_csv(icu / "icustays.csv.gz", parse_dates=["intime", "outtime"], low_memory=False)
        if args.smoke:
            stays = stays.head(5000)
        stay_times = stays[["stay_id", "intime"]].copy()

        hard = stay_level_outcomes(icu, hosp, stays)
        delir = delirium_outcomes(icu, stay_times, nrows)

        out = delir.merge(hard, on="stay_id", how="left")
        out = out.sort_values(["stay_id", "bin"])
        path = project_path(cfg, "data", "interim",
                            "mimic_outcomes_smoke.csv" if args.smoke else "mimic_outcomes.csv")
        out.to_csv(path, index=False)
        print(f"Wrote {path} ({len(out):,} rows)")
        print(f"hospital_mortality rate: {hard['hospital_mortality'].mean():.3f}")
        print(f"delirium bins flagged: {delir['delirium_flag'].sum():,}")
        print(f"delirium_free_days_72h median: {hard.merge(delir[['stay_id','delirium_free_days_72h']].drop_duplicates(),on='stay_id',how='left')['delirium_free_days_72h'].median():.1f}")

    return SimpleNamespace(**locals())

_outcomes = _load_outcomes()


# ==============================================================================
# Standardized exposure variables
# ==============================================================================

def _load_exposure():
    """Standardize enteral nutrition into % of caloric target per bin.

    The raw `enteral_amount_sum` mixes formulas of different caloric density (e.g.
    Jevity 1.06 kcal/mL vs TwoCal 2.0 kcal/mL). The clinically meaningful and
    journal-acceptable exposure is the fraction of the daily caloric target
    achieved. We read real kcal directly from ingredientevents itemid 226060
    (Calories), then divide by an estimated caloric target using the Penn State
    2003b equation for mechanically ventilated ICU patients, falling back to
    Harris-Benedict x stress factor when inputs are missing.

    Outputs data/interim/mimic_exposure_standardized.csv keyed by (stay_id, bin).
    """

    import argparse
    import math
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from clinical_definitions import load_config, project_path
    from clinical_definitions import BIN_HOURS, MIMIC_ITEMIDS
    from clinical_definitions import (
        base_time_bins, hours_from_icu, within_horizon, assign_bin,
        aggregate_numeric_by_bin, last_value_by_bin,
    )


    def harris_benedict(sex, weight_kg, height_cm, age) -> pd.Series:
        """Harris-Benedict BMR (kcal/day). Vectorized over pandas Series."""
        male = (sex.astype(str).str.upper() == "M")
        female = (sex.astype(str).str.upper() == "F")
        bmr = pd.Series(np.nan, index=sex.index, dtype=float)
        w = pd.to_numeric(pd.Series(weight_kg), errors="coerce")
        h = pd.to_numeric(pd.Series(height_cm), errors="coerce")
        a = pd.to_numeric(pd.Series(age), errors="coerce")
        bmr[male] = 88.362 + 13.397 * w[male] + 4.799 * h[male] - 5.677 * a[male]
        bmr[female] = 447.593 + 9.247 * w[female] + 3.098 * h[female] - 4.330 * a[female]
        return bmr


    def penn_state_2003b(bmr, ve_approx, tmax_c=37.0) -> pd.Series:
        """Penn State 2003b resting metabolic rate (kcal/day).

        Published formula (Frankenfield et al., JPEN 2004):
            RMR = HBE × 0.85 + Ve × 33 + Tmax × 175 − 6433

        Where:
            HBE  = Harris-Benedict estimated BMR (kcal/day)
            Ve   = minute ventilation (L/min)
            Tmax = maximum body temperature in prior 24 h (°C)

        Since true minute ventilation (Ve) is rarely charted directly in MIMIC-IV,
        we approximate Ve ≈ respiratory_rate × 0.5 L (assumed tidal volume of
        500 mL).  This is a standard approximation when tidal volume is
        unavailable (normal Ve ≈ 6-8 L/min for adults).  When tmax is missing we
        assume normothermia (37 °C).

        Parameters
        ----------
        bmr : pd.Series – Harris-Benedict BMR estimates (kcal/day).
        ve_approx : pd.Series – Approximate minute ventilation (L/min),
            typically ``resp_rate * 0.5``.
        tmax_c : float or pd.Series – Maximum temperature in °C (default 37.0).

        Returns
        -------
        pd.Series – Estimated RMR clipped to [1000, 4000] kcal/day.
        """
        ve = pd.to_numeric(ve_approx, errors="coerce")
        target = bmr * 0.85 + ve * 33 + tmax_c * 175 - 6433
        # Clinically bounded sanity range: very few patients need < 1000 or > 4000
        target = target.clip(1000, 4000)
        return target


    def compute_target(df: pd.DataFrame) -> pd.Series:
        """Per-row caloric target; falls back to HB × 1.3 when resp rate missing.

        Hierarchy:
          1. Penn State 2003b (requires BMR + resp_rate for Ve approximation)
          2. Harris-Benedict × 1.3 stress factor (fallback)
        """
        hb = harris_benedict(df["gender"], df["weight_kg"], df["height_cm"], df["anchor_age"])
        # Approximate Ve = resp_rate × 0.5 L (assumed tidal volume 500 mL)
        ve_approx = pd.to_numeric(df["resp_rate"], errors="coerce") * 0.5
        ps = penn_state_2003b(hb, ve_approx)
        fallback = hb * 1.3
        return ps.fillna(fallback)


    def _load_chartevents_itemids(
        icu: Path,
        itemids: list[int],
        nrows: int | None = None,
        chunksize: int = 10_000_000,
    ) -> pd.DataFrame:
        """Read selected chartevents itemids without truncating full runs."""
        cols = ["stay_id", "itemid", "charttime", "valuenum"]
        if nrows is not None:
            df = pd.read_csv(
                icu / "chartevents.csv.gz",
                usecols=cols,
                nrows=nrows,
                low_memory=False,
                parse_dates=["charttime"],
            )
            return df[df["itemid"].isin(itemids)].copy()

        chunks = []
        for chunk in pd.read_csv(
            icu / "chartevents.csv.gz",
            usecols=cols,
            chunksize=chunksize,
            low_memory=False,
            parse_dates=["charttime"],
        ):
            filtered = chunk[chunk["itemid"].isin(itemids)].copy()
            if not filtered.empty:
                chunks.append(filtered)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", default="configs/paths.json")
        ap.add_argument("--smoke", action="store_true")
        args = ap.parse_args()
        cfg = load_config(args.config)
        icu = Path(cfg["mimic_icu"])
        hosp = Path(cfg["mimic_hosp"])
        nrows = 1_500_000 if args.smoke else None

        stays = pd.read_csv(icu / "icustays.csv.gz", parse_dates=["intime", "outtime"], low_memory=False)
        if args.smoke:
            stays = stays.head(5000)
        stay_times = stays[["stay_id", "subject_id", "hadm_id", "intime"]].copy()

        out = base_time_bins(stay_times["stay_id"], "stay_id")

        # ---- real kcal from ingredientevents (Calories itemid 226060) ----
        ing = pd.read_csv(
            icu / "ingredientevents.csv.gz",
            usecols=["stay_id", "itemid", "starttime", "amount", "amountuom"], nrows=nrows,
            low_memory=False, parse_dates=["starttime"],
        )
        ing = ing[ing["itemid"].isin(MIMIC_ITEMIDS["enteral_calories"])].copy()
        ing["amount"] = pd.to_numeric(ing["amount"], errors="coerce")
        ing = ing.dropna(subset=["amount"])
        if not ing.empty:
            ing = ing.merge(stay_times[["stay_id", "intime"]], on="stay_id", how="inner")
            ing["hours_from_icu"] = hours_from_icu(ing["starttime"], ing["intime"])
            ing = within_horizon(ing)
            ing["bin"] = assign_bin(ing)
            kcal_bin = aggregate_numeric_by_bin(ing, "stay_id", "amount", "enteral_kcal")
            out = out.merge(kcal_bin, on=["stay_id", "bin"], how="left")

        if "enteral_kcal_sum" not in out.columns:
            out["enteral_kcal_sum"] = 0.0
        out["enteral_kcal_sum"] = out["enteral_kcal_sum"].fillna(0.0)

        # ---- per-patient caloric target (needs baseline covariates) ----
        pat = pd.read_csv(hosp / "patients.csv.gz",
                          usecols=["subject_id", "anchor_age", "gender"], low_memory=False)
        base = stay_times.merge(pat, on="subject_id", how="left")

        # weight (admission)
        w = _load_chartevents_itemids(icu, MIMIC_ITEMIDS["daily_weight"], nrows)
        w = w[w["itemid"].isin(MIMIC_ITEMIDS["daily_weight"])].copy()
        w["valuenum"] = pd.to_numeric(w["valuenum"], errors="coerce")
        if not w.empty:
            adm_w = (w.dropna(subset=["valuenum"]).sort_values("charttime")
                     .groupby("stay_id")["valuenum"].first().rename("weight_kg").reset_index())
            base = base.merge(adm_w, on="stay_id", how="left")

        # height
        h = _load_chartevents_itemids(icu, MIMIC_ITEMIDS["height_cm"], nrows)
        h = h[h["itemid"].isin(MIMIC_ITEMIDS["height_cm"])].copy()
        h["valuenum"] = pd.to_numeric(h["valuenum"], errors="coerce")
        if not h.empty:
            adm_h = (h.dropna(subset=["valuenum"]).sort_values("charttime")
                     .groupby("stay_id")["valuenum"].first().rename("height_cm").reset_index())
            base = base.merge(adm_h, on="stay_id", how="left")

        # representative respiratory rate (median over first 72h)
        rr = _load_chartevents_itemids(icu, MIMIC_ITEMIDS["resp_rate"], nrows)
        rr = rr[rr["itemid"].isin(MIMIC_ITEMIDS["resp_rate"])].copy()
        rr["valuenum"] = pd.to_numeric(rr["valuenum"], errors="coerce")
        rr = rr.dropna(subset=["valuenum"])
        if not rr.empty:
            rr = rr.merge(stay_times[["stay_id", "intime"]], on="stay_id", how="inner")
            rr["hours_from_icu"] = hours_from_icu(rr["charttime"], rr["intime"])
            rr = within_horizon(rr)
            med_rr = rr.groupby("stay_id")["valuenum"].median().rename("resp_rate").reset_index()
            base = base.merge(med_rr, on="stay_id", how="left")

        for c in ["weight_kg", "height_cm", "resp_rate"]:
            if c not in base.columns:
                base[c] = np.nan

        base["caloric_target_kcal_day"] = compute_target(base)
        target = base[["stay_id", "caloric_target_kcal_day"]]

        out = out.merge(target, on="stay_id", how="left")

        # kcal per bin -> extrapolate to a day-equivalent fraction of target.
        # 6h bin holds 6/24 = 0.25 of a day, so a bin meeting target would provide
        # target*0.25 kcal. pct = enteral_kcal / (target*0.25)
        out["pct_caloric_target"] = out["enteral_kcal_sum"] / (out["caloric_target_kcal_day"] * (BIN_HOURS / 24.0))
        out["pct_caloric_target"] = out["pct_caloric_target"].replace([np.inf, -np.inf], np.nan)

        out = out.sort_values(["stay_id", "bin"])
        path = project_path(cfg, "data", "interim",
                            "mimic_exposure_standardized_smoke.csv" if args.smoke else "mimic_exposure_standardized.csv")
        out.to_csv(path, index=False)
        print(f"Wrote {path} ({len(out):,} rows)")
        nz = out["enteral_kcal_sum"].gt(0)
        print(f"enteral_kcal non-zero bins: {nz.sum():,} ({100*nz.mean():.1f}%)")
        print(f"pct_caloric_target: median={out.loc[nz,'pct_caloric_target'].median():.2f} "
              f"mean={out.loc[nz,'pct_caloric_target'].mean():.2f}")

    return SimpleNamespace(**locals())

_exposure = _load_exposure()


# ==============================================================================
# Final MIMIC master cohort
# ==============================================================================

def _load_master():
    """Assemble the final MIMIC master analytic table.

    Merges every interim component produced by the batch-1/batch-2 scripts into one
    stay-bin panel ready for modeling and causal estimation:

      - standardized exposure (pct_caloric_target, enteral_kcal)
      - full glucose outcomes (mean/cv/tir/hypo/hyper, future windows)
      - time-varying + baseline confounders (vasopressor/MV/RRT/steroid/SOFA/RASS/
        diabetes/psychiatric history/age/sex/weight/height)
      - note-derived concept counts (enteral/NPO/delirium/mental-health mentions)
      - hard endpoints (hospital/ICU/28d mortality, ICU LOS, vent-free days) and
        delirium outcomes (stay + bin level)

    This is the single input the modeling layer (deep dynamic treatment-effect
    model, IPW/TMLE/g-formula, causal forest) consumes.
    """

    import argparse
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from clinical_definitions import load_config, project_path
    from clinical_definitions import MAX_BINS


    def apply_exclusion_criteria(master: pd.DataFrame, cfg: dict, smoke: bool = False) -> pd.DataFrame:
        """Apply sequential exclusion criteria and save counts for CONSORT diagram.

        Criteria (applied in order):
          1. Adults only (anchor_age ≥ 18)
          2. Plausible ICU LOS (0 < LOS ≤ 720 h = 30 days)
          3. Exclude early death with no treatment opportunity (ICU LOS < 24 h
             AND ICU mortality = 1) — these confound the treatment effect
          4. At least one glucose measurement in 72 h (required to compute
             the primary glycaemic outcome)

        Parameters
        ----------
        master : pd.DataFrame
            The assembled master cohort (stay_id × bin panel).
        cfg : dict
            Project config (for ``project_path``).
        smoke : bool
            If True, append ``_smoke`` to the output filename.

        Returns
        -------
        pd.DataFrame
            Filtered master cohort.
        """
        counts: dict[str, int] = {"total_stays": int(master["stay_id"].nunique())}

        # 1. Adults only (≥ 18 years)
        if "anchor_age" in master.columns:
            adult_stays = master.groupby("stay_id")["anchor_age"].first()
            adult_stays = adult_stays[adult_stays >= 18].index
            master = master[master["stay_id"].isin(adult_stays)].copy()
        counts["after_adult_filter"] = int(master["stay_id"].nunique())

        # 2. Plausible ICU length-of-stay (> 0 h and ≤ 30 days = 720 h)
        if "icu_los_hours" in master.columns:
            los_per_stay = master.groupby("stay_id")["icu_los_hours"].first()
            valid_los = los_per_stay[(los_per_stay > 0) & (los_per_stay <= 720)].index
            master = master[master["stay_id"].isin(valid_los)].copy()
        counts["after_los_filter"] = int(master["stay_id"].nunique())

        # 3. Exclude early death (< 24 h LOS with ICU mortality)
        #    Patients who die very shortly after admission never received a
        #    meaningful treatment window, biasing the exposure–outcome link.
        if "icu_los_hours" in master.columns and "icu_mortality" in master.columns:
            stay_info = master.groupby("stay_id").agg(
                mort=("icu_mortality", "first"),
                los=("icu_los_hours", "first"),
            )
            early_death_ids = stay_info[
                (stay_info["mort"] == 1) & (stay_info["los"] < 24)
            ].index
            master = master[~master["stay_id"].isin(early_death_ids)].copy()
        counts["after_early_death_exclusion"] = int(master["stay_id"].nunique())

        # 4. At least one glucose measurement in 72 h
        if "glucose_mean" in master.columns:
            has_glucose = (
                master.groupby("stay_id")["glucose_mean"]
                .apply(lambda s: s.notna().any())
            )
            master = master[master["stay_id"].isin(has_glucose[has_glucose].index)].copy()
        counts["after_glucose_filter"] = int(master["stay_id"].nunique())

        # ---- persist exclusion counts for CONSORT flow diagram ----
        counts_df = pd.DataFrame([counts])
        fname = "exclusion_counts_smoke.csv" if smoke else "exclusion_counts.csv"
        path = project_path(cfg, "results", "tables", fname)
        counts_df.to_csv(path, index=False)
        print(f"Exclusion counts saved → {path}")
        for k, v in counts.items():
            print(f"  {k}: {v:,}")

        return master


    def _safe_read(cfg, name, smoke_name=None, smoke=False):
        root = Path(cfg["output_root"]) / "data" / "interim"
        target = smoke_name if (smoke and smoke_name) else name
        path = root / target
        if not path.exists():
            # fall back to the non-smoke name if only that exists
            alt = root / name
            if alt.exists():
                path = alt
            else:
                return None
        return pd.read_csv(path, low_memory=False)


    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", default="configs/paths.json")
        ap.add_argument("--smoke", action="store_true")
        args = ap.parse_args()
        cfg = load_config(args.config)
        s = args.smoke

        # base panel: stay x bin
        conf = _safe_read(cfg, "mimic_confounders.csv",
                          "mimic_confounders_smoke.csv", s)
        if conf is None:
            raise SystemExit("Run build_mimic_confounders.py first")
        master = conf.copy()

        # standardized exposure
        exp = _safe_read(cfg, "mimic_exposure_standardized.csv",
                         "mimic_exposure_standardized_smoke.csv", s)
        if exp is None:
            raise SystemExit("Run standardize_exposure.py first")
        keep = [c for c in ["stay_id", "bin", "enteral_kcal_sum",
                            "caloric_target_kcal_day", "pct_caloric_target"]
                if c in exp.columns]
        missing_exp = {"stay_id", "bin", "enteral_kcal_sum", "pct_caloric_target"} - set(keep)
        if missing_exp:
            raise SystemExit(f"Exposure table is missing required columns: {sorted(missing_exp)}")
        master = master.merge(exp[keep], on=["stay_id", "bin"], how="left")

        # full glucose
        glu = _safe_read(cfg, "mimic_glucose_full.csv",
                         "mimic_glucose_full_smoke.csv", s)
        if glu is None:
            raise SystemExit("Run build_mimic_glucose_full.py first")
        drop = {"stay_id", "bin", "bin_start_hour", "bin_end_hour"}
        gcols = [c for c in glu.columns if c not in drop]
        required_glu = {"stay_id", "bin", "glucose_mean", "future_24h_glucose_mean"}
        missing_glu = required_glu - set(glu.columns)
        if missing_glu:
            raise SystemExit(f"Glucose table is missing required columns: {sorted(missing_glu)}")
        master = master.merge(glu[["stay_id", "bin"] + gcols],
                              on=["stay_id", "bin"], how="left")

        # outcomes (hard endpoints + delirium)
        outc = _safe_read(cfg, "mimic_outcomes.csv",
                          "mimic_outcomes_smoke.csv", s)
        if outc is None:
            raise SystemExit("Run build_mimic_outcomes.py first")
        stay_cols = ["stay_id", "hospital_mortality", "icu_mortality",
                     "mortality_28d", "icu_los_hours", "vent_hours",
                     "vent_free_days_28", "delirium_free_days_72h"]
        stay_cols = [c for c in stay_cols if c in outc.columns]
        stay_outc = outc[stay_cols].drop_duplicates("stay_id")
        master = master.merge(stay_outc, on="stay_id", how="left")
        bin_cols = [c for c in ["stay_id", "bin", "delirium_flag", "deep_sedation_flag"]
                    if c in outc.columns]
        master = master.merge(outc[bin_cols], on=["stay_id", "bin"], how="left")

        # note features (built by the original build_mimic_note_features.py)
        notes = _safe_read(cfg, "mimic_note_features.csv",
                           "mimic_note_features_smoke.csv", s)
        if notes is not None and not notes.empty:
            master = master.merge(notes, on=["subject_id", "hadm_id", "stay_id", "bin"],
                                  how="left", suffixes=("", "_note"))

        # ---- derive a few modeling helpers ----
        # cumulative enteral kcal up to and including current bin (running exposure)
        master = master.sort_values(["stay_id", "bin"])
        master["enteral_kcal_cum"] = master.groupby("stay_id")["enteral_kcal_sum"].cumsum()
        # early feeding by 24h (bins 0-3 any nonzero kcal)
        early = (master[master["bin"] <= 3]
                 .groupby("stay_id")["enteral_kcal_sum"].sum().gt(0)
                 .rename("early_feeding_24h").reset_index())
        master = master.merge(early, on="stay_id", how="left")
        master["early_feeding_24h"] = master["early_feeding_24h"].fillna(False).astype(int)

        # fill note columns where missing
        note_fill = ["note_count", "note_enteral_mentions", "note_npo_interruption_mentions",
                     "note_glucose_mentions", "note_insulin_mentions",
                     "note_delirium_mentions", "note_mental_health_mentions",
                     "note_infection_mentions", "note_ventilation_mentions",
                     "has_potential_note_leakage"]
        for c in note_fill:
            if c in master.columns:
                master[c] = master[c].fillna(0)

        # exposure missing -> 0 (no kcal recorded)
        for c in ["enteral_kcal_sum", "pct_caloric_target", "enteral_kcal_cum"]:
            if c in master.columns:
                master[c] = master[c].fillna(0)

        # any psychiatric history composite
        hx_cols = [c for c in master.columns if c.startswith("hx_")]
        if hx_cols:
            master["hx_any_psych"] = master[hx_cols].fillna(0).sum(axis=1).gt(0).astype(int)
        else:
            master["hx_any_psych"] = 0

        if "sofa" in master.columns:
            master["severity_score"] = master["sofa"]
        if "gender" in master.columns:
            master["sex_male"] = master["gender"].astype(str).str.upper().eq("M").astype(int)

        master = master.sort_values(["stay_id", "bin"])

        # ---- Apply exclusion criteria before saving ----
        master = apply_exclusion_criteria(master, cfg, smoke=s)

        path = project_path(cfg, "data", "interim",
                            "mimic_master_cohort_smoke.csv" if s else "mimic_master_cohort.csv")
        master.to_csv(path, index=False)
        print(f"Wrote {path} ({len(master):,} rows, {master.shape[1]} cols)")
        # quick summary for sanity
        uniq = master["stay_id"].nunique()
        print(f"unique stays: {uniq:,}")
        if "hospital_mortality" in master.columns:
            print(f"hospital_mortality rate: {master.groupby('stay_id')['hospital_mortality'].first().mean():.3f}")
        if "pct_caloric_target" in master.columns:
            nz = master["pct_caloric_target"].gt(0)
            print(f"fed bins: {nz.sum():,} ({100*nz.mean():.1f}%), "
                  f"early_feeding_24h rate: {master.groupby('stay_id')['early_feeding_24h'].first().mean():.3f}")
        print(f"columns: {master.columns.tolist()}")

    return SimpleNamespace(**locals())

_master = _load_master()


STAGES = {
    "confounders": _confounders.main,
    "glucose": _glucose.main,
    "notes": _notes.main,
    "outcomes": _outcomes.main,
    "exposure": _exposure.main,
    "master": _master.main,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build analysis-ready MIMIC-IV clinical tables")
    parser.add_argument("stage", choices=STAGES, help="Pipeline stage to run")
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    stage = sys.argv[1]
    if stage not in STAGES:
        parser.error(f"invalid stage: {stage}")
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    STAGES[stage]()


if __name__ == "__main__":
    main()
