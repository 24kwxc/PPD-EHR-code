"""Clinical-state and external-transport analyses.

Former one-off scripts are retained as clearly labelled stages in isolated
namespaces. This prevents helper-name collisions while giving each manuscript
section one analysis entry point.
"""
from __future__ import annotations

import argparse
import sys
import types
from types import SimpleNamespace


def _register_legacy(name: str, namespace: SimpleNamespace) -> None:
    module = types.ModuleType(name)
    module.__dict__.update(vars(namespace))
    sys.modules[name] = module


# ==============================================================================
# State-transition dataset
# ==============================================================================

def _load_state_dataset():
    """Build the frozen longitudinal medication--nutrition pathway dataset.

    The script reconstructs the archived six-component outcome-free clinical GMM,
    verifies it against archived hard assignments, exports the full posterior
    probability vector, and builds a patient-level landmark table with:

        baseline 0--6 h -> exposure 6--24 h -> state trajectory 24--48 h
        -> strict CAM 48--72 h.

    No outcome is used to fit or name the state model.  The archived state model is
    reconstructed because the original release saved hard assignments and maximum
    posterior probabilities, but not the target renal-state posterior probability.
    """

    import argparse
    import hashlib
    import json
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler


    SEED = 20260712
    PHYS = [
        "heart_rate", "map", "temperature", "spo2", "glucose_mean",
        "glucose_cv", "lactate", "creatinine", "bun", "wbc",
    ]
    STATE_NAMES = {
        0: "Mixed vulnerability 1",
        1: "Inflammatory-haemodynamic stress",
        2: "Mixed vulnerability 2",
        3: "Hyperglycaemic instability",
        4: "Renal-metabolic vulnerability",
        5: "Stable metabolic",
    }
    RENAL_STATE_ID = 4
    BOUNDS = {
        "heart_rate": (20, 250), "map": (20, 200),
        "temperature": (30, 43), "spo2": (60, 100),
        "glucose_mean": (20, 1000), "glucose_cv": (0, 3),
        "lactate": (0, 30), "creatinine": (0, 20),
        "bun": (0, 200), "wbc": (0, 200),
    }


    def parse_args() -> argparse.Namespace:
        package = Path(__file__).resolve().parents[2]
        data_root = package / "data"
        historical = data_root / "derived" / "historical_release"
        source_root = data_root / "derived" / "latent_trajectory"
        p = argparse.ArgumentParser()
        p.add_argument(
            "--clinical-windows",
            type=Path,
            default=data_root / "derived" / "clinical_order_silent_deployment_windows.csv",
        )
        p.add_argument(
            "--intervention-windows",
            type=Path,
            default=source_root / "mimic_intervention_features_6h.csv.gz",
        )
        p.add_argument("--matched-stays", type=Path, default=source_root / "matched_stays_1to2.csv")
        p.add_argument(
            "--archived-state-assignments",
            type=Path,
            default=source_root / "latent_state_assignments.csv.gz",
        )
        p.add_argument(
            "--archived-state-profiles",
            type=Path,
            default=source_root / "latent_state_profiles.csv",
        )
        p.add_argument(
            "--strict-outcome-table",
            type=Path,
            default=historical / "diabetic_overlap_validation" / "ppd_ehr_overlap_28226_analysis_rows.csv",
        )
        p.add_argument(
            "--icustays-table", type=Path,
            default=data_root / "restricted" / "mimic_iv" / "icu" / "icustays.csv.gz",
        )
        p.add_argument(
            "--admissions-table", type=Path,
            default=data_root / "restricted" / "mimic_iv" / "hosp" / "admissions.csv.gz",
        )
        p.add_argument(
            "--output-dir",
            type=Path,
            default=package / "outputs" / "downstream" / "longitudinal_pathway",
        )
        p.add_argument(
            "--model-output",
            type=Path,
            default=package / "model" / "clinical_state_gmm_reconstructed_20260713.joblib",
        )
        return p.parse_args()


    def stable_subject_arm(subject_id: int) -> str:
        h = hashlib.sha256(f"seed42:{int(subject_id)}".encode("utf-8")).digest()
        value = int.from_bytes(h[:8], "little") / 2**64
        return "development" if value < 0.70 else "confirmation"


    def clip_thresholds(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for c, (lo, hi) in BOUNDS.items():
            x = pd.to_numeric(frame[c], errors="coerce")
            if c == "temperature":
                f = x.between(70, 120)
                x.loc[f] = (x.loc[f] - 32) * 5 / 9
            x = x.where(x.between(lo, hi))
            if x.notna().sum() > 100:
                qlo, qhi = x.quantile([0.0025, 0.9975])
                out[c] = (float(qlo), float(qhi))
            else:
                out[c] = (float(lo), float(hi))
        return out


    def cleaned_phys(frame: pd.DataFrame, thresholds: dict[str, tuple[float, float]]) -> pd.DataFrame:
        x = frame[PHYS].copy()
        for c, (lo, hi) in BOUNDS.items():
            x[c] = pd.to_numeric(x[c], errors="coerce")
            if c == "temperature":
                f = x[c].between(70, 120)
                x.loc[f, c] = (x.loc[f, c] - 32) * 5 / 9
            x[c] = x[c].where(x[c].between(lo, hi))
            qlo, qhi = thresholds[c]
            x[c] = x[c].clip(qlo, qhi)
        return x


    def fit_archived_gmm(
        matched_windows: pd.DataFrame,
    ) -> tuple[SimpleImputer, StandardScaler, GaussianMixture, dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
        sample = matched_windows.sample(
            min(100_000, len(matched_windows)),
            random_state=SEED,
            weights=matched_windows["match_weight"].clip(lower=1e-6),
        )
        train_thresholds = clip_thresholds(sample)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_x = scaler.fit_transform(imputer.fit_transform(cleaned_phys(sample, train_thresholds)))
        gmm = GaussianMixture(
            n_components=6,
            covariance_type="diag",
            n_init=3,
            max_iter=300,
            random_state=SEED,
        ).fit(train_x)
        prediction_thresholds = clip_thresholds(matched_windows)
        return imputer, scaler, gmm, train_thresholds, prediction_thresholds


    def predict_state_probabilities(
        frame: pd.DataFrame,
        imputer: SimpleImputer,
        scaler: StandardScaler,
        gmm: GaussianMixture,
        thresholds: dict[str, tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        x = scaler.transform(imputer.transform(cleaned_phys(frame, thresholds)))
        return gmm.predict(x), gmm.predict_proba(x)


    def zscore_contract(values: pd.Series, arm: pd.Series) -> tuple[pd.Series, dict[str, float]]:
        ref = pd.to_numeric(values[arm.eq("development")], errors="coerce")
        mean = float(ref.mean())
        sd = float(ref.std(ddof=0))
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        return (pd.to_numeric(values, errors="coerce") - mean) / sd, {"mean": mean, "sd": sd}


    def main() -> None:
        args = parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.model_output.parent.mkdir(parents=True, exist_ok=True)

        required = [
            args.clinical_windows, args.intervention_windows, args.matched_stays,
            args.archived_state_assignments, args.archived_state_profiles,
            args.strict_outcome_table, args.icustays_table, args.admissions_table,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing required inputs: " + "; ".join(missing))

        raw_cols = [
            "stay_id", "subject_id", "hadm_id", "bin", "bin_start_hour", "bin_end_hour",
            "diabetes", "gender", "anchor_age", "admission_type", "first_careunit",
            "charlson_index", "admission_sofa", "weight_kg", "caloric_target_kcal_day",
            "sofa", "mv_flag", "rrt_flag", "vasopressor_sum", "rass",
            "midazolam_amount_sum", "dexmedetomidine_amount_sum", "ppi_h2_flag",
            "antiemetic_flag", "heparin_amount_sum", "cam_icu_max", "delirium_flag",
            "deep_sedation_flag", "delirium_free_days_72h", "icu_los_hours",
            "icu_mortality", "hospital_mortality",
        ] + PHYS
        raw = pd.read_csv(
            args.clinical_windows,
            usecols=lambda c: c in set(raw_cols),
            low_memory=False,
        ).sort_values(["stay_id", "bin"]).reset_index(drop=True)

        matched = pd.read_csv(args.matched_stays)
        matched_windows = raw.merge(matched, on=["stay_id", "diabetes"], how="inner")
        imputer, scaler, gmm, train_clip, predict_clip = fit_archived_gmm(matched_windows)
        hard, probs = predict_state_probabilities(
            matched_windows, imputer, scaler, gmm, predict_clip
        )
        matched_windows = matched_windows.copy()
        matched_windows["latent_state_id"] = hard
        matched_windows["latent_state"] = matched_windows["latent_state_id"].map(STATE_NAMES)
        for k in range(6):
            matched_windows[f"p_state_{k}"] = probs[:, k]

        archived = pd.read_csv(
            args.archived_state_assignments,
            usecols=["stay_id", "bin", "latent_state_id", "latent_state", "latent_state_probability"],
        )
        check = matched_windows[
            ["stay_id", "bin", "latent_state_id", "latent_state"]
        ].merge(archived, on=["stay_id", "bin"], suffixes=("_new", "_archived"))
        ari = float(adjusted_rand_score(check["latent_state_id_new"], check["latent_state_id_archived"]))
        agreement = float((check["latent_state_id_new"] == check["latent_state_id_archived"]).mean())
        name_agreement = float((check["latent_state_new"] == check["latent_state_archived"]).mean())
        max_prob = probs.max(axis=1)
        archived_prob = matched_windows[["stay_id", "bin"]].merge(
            archived[["stay_id", "bin", "latent_state_probability"]],
            on=["stay_id", "bin"], how="left",
        )["latent_state_probability"].to_numpy(float)
        max_prob_abs_error = float(np.nanmax(np.abs(max_prob - archived_prob)))
        if ari < 0.999999 or agreement < 0.999999 or name_agreement < 0.999999:
            raise RuntimeError(
                f"Archived state reconstruction failed: ARI={ari}, agreement={agreement}, "
                f"name_agreement={name_agreement}"
            )

        model_payload = {
            "imputer": imputer,
            "scaler": scaler,
            "gmm": gmm,
            "train_clip_thresholds": train_clip,
            "prediction_clip_thresholds": predict_clip,
            "physiology_features": PHYS,
            "state_names": STATE_NAMES,
            "renal_state_id": RENAL_STATE_ID,
            "seed": SEED,
            "boundary": "Deterministic reconstruction of the archived outcome-free clinical GMM.",
        }
        joblib.dump(model_payload, args.model_output)

        strict = pd.read_csv(args.strict_outcome_table, low_memory=False)
        causal_ids = strict[["stay_id", "subject_id", "hadm_id"]].drop_duplicates("stay_id")
        causal_windows = raw.merge(causal_ids[["stay_id"]], on="stay_id", how="inner")
        _, causal_probs = predict_state_probabilities(
            causal_windows, imputer, scaler, gmm, predict_clip
        )
        causal_windows = causal_windows.copy()
        causal_windows["latent_state_id"] = causal_probs.argmax(axis=1)
        causal_windows["latent_state"] = causal_windows["latent_state_id"].map(STATE_NAMES)
        for k in range(6):
            causal_windows[f"p_state_{k}"] = causal_probs[:, k]
        state_export_cols = [
            "stay_id", "subject_id", "hadm_id", "bin", "bin_start_hour", "bin_end_hour",
            "latent_state_id", "latent_state",
        ] + [f"p_state_{k}" for k in range(6)]
        causal_windows[state_export_cols].to_csv(
            args.output_dir / "clinical_state_full_posteriors_0_72h.csv.gz",
            index=False, compression="gzip",
        )

        intervention_cols = [
            "stay_id", "bin", "enteral_kcal_sum", "enteral_kcal_tv_24h",
            "feeding_restart_6h", "feeding_interruption_6h", "propofol_kcal",
            "insulin_units", "insulin_response_residual",
            "insulin_response_residual_primary_eligible", "steroid_flag",
            "steroid_rx_hc_equiv_mg_6h", "dextrose_amount_sum",
            "midazolam_amount_sum", "dexmedetomidine_amount_sum",
        ]
        intervention = pd.read_csv(
            args.intervention_windows, usecols=intervention_cols, low_memory=False
        )
        intervention = intervention.merge(causal_ids[["stay_id"]], on="stay_id", how="inner")
        exposure_windows = intervention[intervention["bin"].between(1, 3)].copy()
        exposure_windows = exposure_windows.sort_values(["stay_id", "bin"])
        exposure_windows["enteral_internal_abs_change_6_24"] = (
            exposure_windows.groupby("stay_id")["enteral_kcal_sum"].diff().abs().fillna(0)
        )
        exposure = exposure_windows.groupby(
            "stay_id", as_index=False
        ).agg(
            enteral_kcal_6_24=("enteral_kcal_sum", "sum"),
            enteral_tv_6_24=("enteral_internal_abs_change_6_24", "sum"),
            feeding_restart_count_6_24=("feeding_restart_6h", "sum"),
            feeding_interruption_count_6_24=("feeding_interruption_6h", "sum"),
            zero_feed_bins_6_24=("enteral_kcal_sum", lambda x: int(pd.to_numeric(x, errors="coerce").fillna(0).le(0).sum())),
            propofol_kcal_6_24=("propofol_kcal", "sum"),
            insulin_units_6_24=("insulin_units", "sum"),
            prescribed_steroid_any_6_24=("steroid_flag", "max"),
            prescribed_steroid_fraction_6_24=("steroid_flag", "mean"),
            prescribed_steroid_hc_equiv_6_24=("steroid_rx_hc_equiv_mg_6h", "sum"),
            dextrose_g_6_24=("dextrose_amount_sum", "sum"),
            midazolam_6_24=("midazolam_amount_sum", "sum"),
            dexmedetomidine_6_24=("dexmedetomidine_amount_sum", "sum"),
        )
        early_grdp = intervention[
            intervention["bin"].between(0, 1)
            & intervention["insulin_response_residual_primary_eligible"].eq(1)
        ].groupby("stay_id", as_index=False).agg(
            early_grdp=("insulin_response_residual", "mean"),
            early_grdp_bins=("insulin_response_residual", "count"),
        )
        exposure = exposure.merge(early_grdp, on="stay_id", how="left")

        raw_exposure = causal_windows[causal_windows["bin"].between(1, 3)].groupby(
            "stay_id", as_index=False
        ).agg(
            caloric_target_kcal_day=("caloric_target_kcal_day", "median"),
            weight_kg_exposure=("weight_kg", "median"),
            procedure_mv_proxy_any_6_24=("mv_flag", "max"),
            rrt_any_6_24=("rrt_flag", "max"),
            vasopressor_max_6_24=("vasopressor_sum", "max"),
            sofa_max_6_24=("sofa", "max"),
            rass_min_6_24=("rass", "min"),
            ppi_h2_any_6_24=("ppi_h2_flag", "max"),
            antiemetic_any_6_24=("antiemetic_flag", "max"),
            heparin_any_6_24=("heparin_amount_sum", lambda x: int(pd.to_numeric(x, errors="coerce").fillna(0).gt(0).any())),
        )
        exposure = exposure.merge(raw_exposure, on="stay_id", how="left")

        baseline = causal_windows[causal_windows["bin"].eq(0)].copy()
        baseline_cols = [
            "stay_id", "subject_id", "hadm_id", "gender", "anchor_age",
            "admission_type", "first_careunit", "charlson_index", "admission_sofa",
            "weight_kg", "sofa", "mv_flag", "rrt_flag", "vasopressor_sum", "rass",
            "lactate", "creatinine", "glucose_mean", "glucose_cv",
            "delirium_free_days_72h", "icu_los_hours", "icu_mortality",
            "hospital_mortality",
        ] + [f"p_state_{k}" for k in range(6)]
        baseline = baseline[baseline_cols].rename(columns={
            "sofa": "baseline_sofa_0_6", "mv_flag": "baseline_mv_0_6",
            "rrt_flag": "baseline_rrt_0_6", "vasopressor_sum": "baseline_vasopressor_0_6",
            "rass": "baseline_rass_0_6", "lactate": "baseline_lactate_0_6",
            "creatinine": "baseline_creatinine_0_6", "glucose_mean": "baseline_glucose_mean_0_6",
            "glucose_cv": "baseline_glucose_cv_0_6",
            **{f"p_state_{k}": f"baseline_p_state_{k}" for k in range(6)},
        })

        post = causal_windows[causal_windows["bin"].between(4, 7)].groupby(
            "stay_id", as_index=False
        ).agg(
            post_state_bins=("bin", "nunique"),
            renal_transition_burden=("p_state_4", "mean"),
            renal_transition_peak=("p_state_4", "max"),
            **{f"post_mean_p_state_{k}": (f"p_state_{k}", "mean") for k in range(6)},
        )
        current = causal_windows[causal_windows["bin"].eq(3)][
            ["stay_id", "latent_state_id", "latent_state"] + [f"p_state_{k}" for k in range(6)]
        ].rename(columns={
            "latent_state_id": "state_id_18_24", "latent_state": "state_18_24",
            **{f"p_state_{k}": f"p_state_{k}_18_24" for k in range(6)},
        })
        immediate = causal_windows[causal_windows["bin"].eq(4)][
            ["stay_id", "latent_state_id", "latent_state"] + [f"p_state_{k}" for k in range(6)]
        ].rename(columns={
            "latent_state_id": "state_id_24_30", "latent_state": "state_24_30",
            **{f"p_state_{k}": f"p_state_{k}_24_30" for k in range(6)},
        })

        rows = strict.merge(baseline, on=["stay_id", "subject_id", "hadm_id"], how="inner")
        for c in [
            "gender", "anchor_age", "admission_type", "first_careunit",
            "charlson_index", "admission_sofa", "weight_kg",
        ]:
            left, right = f"{c}_x", f"{c}_y"
            if left in rows and right in rows:
                rows[c] = rows[left].combine_first(rows[right])
                rows = rows.drop(columns=[left, right])
        rows = rows.merge(exposure, on="stay_id", how="left")
        rows = rows.merge(post, on="stay_id", how="left")
        rows = rows.merge(current, on="stay_id", how="left")
        rows = rows.merge(immediate, on="stay_id", how="left")
        rows["analysis_arm"] = rows["subject_id"].map(stable_subject_arm)

        icu = pd.read_csv(
            args.icustays_table,
            usecols=["stay_id", "hadm_id", "intime", "outtime"],
            parse_dates=["intime", "outtime"],
        )
        admissions = pd.read_csv(
            args.admissions_table,
            usecols=["hadm_id", "dischtime", "deathtime"],
            parse_dates=["dischtime", "deathtime"],
        )
        rows = rows.merge(icu, on=["stay_id", "hadm_id"], how="left", validate="one_to_one")
        rows = rows.merge(admissions, on="hadm_id", how="left", validate="many_to_one")
        h48 = rows["intime"] + pd.to_timedelta(48, unit="h")
        h72 = rows["intime"] + pd.to_timedelta(72, unit="h")
        rows["death_48_72"] = rows["deathtime"].between(h48, h72, inclusive="both")
        rows["icu_discharge_48_72"] = rows["outtime"].between(h48, h72, inclusive="both")
        rows["hospital_discharge_48_72"] = rows["dischtime"].between(h48, h72, inclusive="both")

        development_weight = pd.to_numeric(
            rows.loc[rows.analysis_arm.eq("development"), "weight_kg_exposure"], errors="coerce"
        ).median()
        rows["analysis_weight_kg"] = pd.to_numeric(rows["weight_kg_exposure"], errors="coerce")
        rows["analysis_weight_kg"] = rows["analysis_weight_kg"].fillna(float(development_weight))
        fallback_target = 25.0 * rows["analysis_weight_kg"]
        rows["caloric_target_kcal_day"] = pd.to_numeric(rows["caloric_target_kcal_day"], errors="coerce").fillna(fallback_target)
        rows["caloric_target_kcal_6_24"] = rows["caloric_target_kcal_day"] * 18.0 / 24.0
        target = rows["caloric_target_kcal_6_24"].clip(lower=1.0)
        rows["energy_adequacy_6_24"] = rows["enteral_kcal_6_24"] / target
        rows["energy_deficit_6_24"] = (1.0 - rows["energy_adequacy_6_24"]).clip(lower=0, upper=2)
        rows["energy_tv_scaled_6_24"] = rows["enteral_tv_6_24"] / target
        rows["zero_feed_fraction_6_24"] = rows["zero_feed_bins_6_24"] / 3.0
        rows["feeding_event_fraction_6_24"] = (
            rows["feeding_restart_count_6_24"] + rows["feeding_interruption_count_6_24"]
        ) / 6.0
        rows["dextrose_g_per_kg_6_24"] = rows["dextrose_g_6_24"] / rows["analysis_weight_kg"].clip(lower=20)
        rows["insulin_units_per_kg_6_24"] = rows["insulin_units_6_24"] / rows["analysis_weight_kg"].clip(lower=20)
        rows["propofol_kcal_per_kg_6_24"] = rows["propofol_kcal_6_24"] / rows["analysis_weight_kg"].clip(lower=20)

        score_contract: dict[str, dict[str, float]] = {}
        score_parts = []
        for c in [
            "energy_deficit_6_24", "energy_tv_scaled_6_24",
            "zero_feed_fraction_6_24", "feeding_event_fraction_6_24",
        ]:
            z, contract = zscore_contract(rows[c], rows["analysis_arm"])
            rows[f"z_{c}"] = z
            score_parts.append(z)
            score_contract[c] = contract
        rows["energy_instability_score"] = pd.concat(score_parts, axis=1).mean(axis=1)
        dev_score = rows.loc[rows.analysis_arm.eq("development"), "energy_instability_score"]
        regimes = {
            "stable_energy_q25": float(dev_score.quantile(0.25)),
            "unstable_energy_q75": float(dev_score.quantile(0.75)),
        }
        rows["strict_cam_assessed_48_72"] = rows["strict_cam_positive_48_72"].notna()
        rows["complete_state_24_48"] = rows["post_state_bins"].eq(4)
        rows["pathway_landmark_eligible"] = (
            rows["strict_landmark_48h"].eq(True) & rows["complete_state_24_48"]
        )

        rows.to_csv(
            args.output_dir / "longitudinal_state_transition_analysis_rows.csv.gz",
            index=False, compression="gzip",
        )

        profile_source = cleaned_phys(matched_windows, predict_clip)
        profile_source["latent_state_id"] = matched_windows["latent_state_id"].to_numpy()
        state_profiles = profile_source.groupby("latent_state_id", as_index=False).agg(
            **{c: (c, "mean") for c in PHYS},
        )
        state_profiles["n_windows"] = (
            matched_windows.groupby("latent_state_id").size().reindex(state_profiles.latent_state_id).to_numpy()
        )
        state_profiles["latent_state"] = state_profiles["latent_state_id"].map(STATE_NAMES)
        state_profiles.to_csv(args.output_dir / "reconstructed_clinical_state_profiles.csv", index=False)

        archived_profiles = pd.read_csv(args.archived_state_profiles)
        profile_check = state_profiles.merge(
            archived_profiles, on=["latent_state_id", "latent_state"], suffixes=("_new", "_archived")
        )
        profile_max_abs_error = 0.0
        for c in PHYS:
            profile_max_abs_error = max(
                profile_max_abs_error,
                float(np.nanmax(np.abs(profile_check[f"{c}_new"] - profile_check[f"{c}_archived"]))),
            )

        support = rows.groupby("analysis_arm", as_index=False).agg(
            stays=("stay_id", "nunique"),
            subjects=("subject_id", "nunique"),
            pathway_landmark=("pathway_landmark_eligible", "sum"),
            strict_cam_assessed=("strict_cam_assessed_48_72", "sum"),
            strict_cam_events=("strict_cam_positive_48_72", "sum"),
            prescribed_steroid_exposed=("prescribed_steroid_any_6_24", "sum"),
            ppi_h2_exposed=("ppi_h2_any_6_24", "sum"),
            antiemetic_exposed=("antiemetic_any_6_24", "sum"),
            early_grdp_supported=("early_grdp", lambda x: int(x.notna().sum())),
        )
        support.to_csv(args.output_dir / "longitudinal_pathway_support_audit.csv", index=False)

        audit = {
            "state_reconstruction": {
                "matched_windows": int(len(matched_windows)),
                "matched_stays": int(matched_windows.stay_id.nunique()),
                "selected_k_archived": 6,
                "hard_label_ari": ari,
                "hard_label_agreement": agreement,
                "state_name_agreement": name_agreement,
                "max_posterior_max_abs_error": max_prob_abs_error,
                "profile_max_abs_error": profile_max_abs_error,
                "renal_state_id": RENAL_STATE_ID,
                "renal_state_name": STATE_NAMES[RENAL_STATE_ID],
            },
            "causal_dataset": {
                "stays": int(rows.stay_id.nunique()),
                "subjects": int(rows.subject_id.nunique()),
                "complete_24_48_state_stays": int(rows.complete_state_24_48.sum()),
                "pathway_landmark_stays": int(rows.pathway_landmark_eligible.sum()),
                "strict_cam_assessed_in_pathway_landmark": int(
                    rows.loc[rows.pathway_landmark_eligible, "strict_cam_assessed_48_72"].sum()
                ),
                "strict_cam_events_in_pathway_landmark": int(
                    rows.loc[rows.pathway_landmark_eligible, "strict_cam_positive_48_72"].sum()
                ),
                "death_48_72_in_pathway_landmark": int(
                    rows.loc[rows.pathway_landmark_eligible, "death_48_72"].sum()
                ),
                "icu_discharge_48_72_in_pathway_landmark": int(
                    rows.loc[rows.pathway_landmark_eligible, "icu_discharge_48_72"].sum()
                ),
            },
            "energy_score_contract": score_contract,
            "regime_values": regimes,
            "delirium_free_days_boundary": (
                "The archived 72-h delirium-free-days variable counts unrecorded CAM bins as free; "
                "it is not a confirmatory delirium-coma-free-days endpoint."
            ),
            "claim_boundary": (
                "Retrospective observational landmark dataset. The state model excludes treatment "
                "and future CAM, but standardized pathway effects depend on measured-confounding, "
                "positivity and observation-process assumptions. Steroid exposure is prescription-"
                "based; the archived mechanical-ventilation variable is a procedure-derived proxy."
            ),
        }
        (args.output_dir / "longitudinal_dataset_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        print(json.dumps(audit, indent=2))

    return SimpleNamespace(**locals())

_state_dataset = _load_state_dataset()

_register_legacy("build_longitudinal_state_transition_dataset_20260713", _state_dataset)


# ==============================================================================
# Primary 0–48 h state model
# ==============================================================================

def _load_primary():
    """Fit the primary patient-disjoint, outcome-free 0--48 h clinical state model.

    The archived clinical GMM was fitted on windows extending through 72 h.  This
    script creates the prespecified causal-analysis state model using only 0--48 h
    windows from the subject-level development arm.  It freezes the model before
    applying it to the confirmation arm and to the 48--72 h CAM analysis.
    """

    import argparse
    import json
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from build_longitudinal_state_transition_dataset_20260713 import (
        BOUNDS,
        PHYS,
        SEED,
        cleaned_phys,
        clip_thresholds,
        stable_subject_arm,
    )


    def parse_args() -> argparse.Namespace:
        package = Path(__file__).resolve().parents[2]
        data_root = package / "data"
        source_root = data_root / "derived" / "latent_trajectory"
        p = argparse.ArgumentParser()
        p.add_argument(
            "--clinical-windows", type=Path,
            default=data_root / "derived" / "clinical_order_silent_deployment_windows.csv",
        )
        p.add_argument("--matched-stays", type=Path, default=source_root / "matched_stays_1to2.csv")
        p.add_argument(
            "--archived-assignments", type=Path,
            default=source_root / "latent_state_assignments.csv.gz",
        )
        p.add_argument(
            "--archived-analysis-rows", type=Path,
            default=package / "outputs" / "downstream" / "longitudinal_pathway"
            / "longitudinal_state_transition_analysis_rows.csv.gz",
        )
        p.add_argument(
            "--output-dir", type=Path,
            default=package / "outputs" / "downstream" / "longitudinal_pathway",
        )
        p.add_argument(
            "--model-output", type=Path,
            default=package / "model" / "clinical_state_gmm_primary_0_48h_20260713.joblib",
        )
        return p.parse_args()


    def name_components(profiles: pd.DataFrame) -> dict[int, str]:
        indexed = profiles.set_index("state_id")
        z = indexed[PHYS].copy()
        for c in PHYS:
            sd = float(z[c].std(ddof=0))
            z[c] = (z[c] - z[c].mean()) / sd if sd > 0 else 0.0
        targets = {
            "heart_rate": (80, 20), "map": (80, 15), "temperature": (36.8, 0.7),
            "spo2": (97, 3), "glucose_mean": (130, 50), "glucose_cv": (0.05, 0.10),
            "lactate": (1.5, 2), "creatinine": (1.0, 1.5), "bun": (20, 30),
            "wbc": (10, 10),
        }
        stable_distance = pd.Series(0.0, index=indexed.index)
        for c, (target, scale) in targets.items():
            stable_distance += ((indexed[c] - target) / scale).abs()
        names: dict[int, str] = {}
        stable = int(stable_distance.idxmin())
        names[stable] = "Stable metabolic"
        remaining = [int(x) for x in indexed.index if int(x) != stable]
        hyper = int(z.loc[remaining, ["glucose_mean", "glucose_cv"]].mean(axis=1).idxmax())
        names[hyper] = "Hyperglycaemic instability"
        remaining.remove(hyper)
        renal = int(z.loc[remaining, ["creatinine", "bun"]].mean(axis=1).idxmax())
        names[renal] = "Renal-metabolic vulnerability"
        remaining.remove(renal)
        stress_score = (
            z.loc[remaining, ["lactate", "heart_rate", "wbc"]].mean(axis=1)
            - z.loc[remaining, ["map", "spo2"]].mean(axis=1)
        )
        stress = int(stress_score.idxmax())
        names[stress] = "Inflammatory-haemodynamic stress"
        remaining.remove(stress)
        for i, state in enumerate(remaining, start=1):
            names[state] = f"Mixed vulnerability {i}"
        return names


    def main() -> None:
        args = parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        for p in [args.clinical_windows, args.matched_stays, args.archived_assignments, args.archived_analysis_rows]:
            if not p.exists():
                raise FileNotFoundError(p)

        cols = ["stay_id", "subject_id", "bin", "diabetes"] + PHYS
        raw = pd.read_csv(args.clinical_windows, usecols=cols, low_memory=False)
        matched = pd.read_csv(args.matched_stays)
        work = raw.merge(matched, on=["stay_id", "diabetes"], how="inner")
        work = work[work["bin"].between(0, 7)].sort_values(["stay_id", "bin"]).reset_index(drop=True)
        work["analysis_arm"] = work["subject_id"].map(stable_subject_arm)
        development = work[work["analysis_arm"].eq("development")].copy()

        sample = development.sample(
            min(100_000, len(development)), random_state=SEED,
            weights=development["match_weight"].clip(lower=1e-6),
        )
        train_clip = clip_thresholds(sample)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_x = scaler.fit_transform(imputer.fit_transform(cleaned_phys(sample, train_clip)))
        gmm = GaussianMixture(
            n_components=6, covariance_type="diag", n_init=5, max_iter=300,
            random_state=SEED,
        ).fit(train_x)
        prediction_clip = clip_thresholds(development)
        all_x = scaler.transform(imputer.transform(cleaned_phys(work, prediction_clip)))
        hard = gmm.predict(all_x)
        probs = gmm.predict_proba(all_x)
        work["primary_state_id"] = hard
        clean_for_profiles = cleaned_phys(work, prediction_clip)
        clean_for_profiles["state_id"] = hard
        profiles = clean_for_profiles.groupby("state_id", as_index=False)[PHYS].mean()
        profiles["n_windows"] = (
            pd.Series(hard).value_counts().reindex(profiles.state_id).to_numpy()
        )
        names = name_components(profiles)
        profiles["primary_state"] = profiles["state_id"].map(names)
        renal_id = next(k for k, v in names.items() if v == "Renal-metabolic vulnerability")
        for k in range(6):
            work[f"primary_p_state_{k}"] = probs[:, k]
        work["primary_state"] = work["primary_state_id"].map(names)

        archived = pd.read_csv(
            args.archived_assignments,
            usecols=["stay_id", "bin", "latent_state_id", "latent_state"],
        )
        comparison = work[["stay_id", "subject_id", "bin", "analysis_arm", "primary_state_id"]].merge(
            archived, on=["stay_id", "bin"], how="inner"
        )
        contingency = pd.crosstab(comparison.primary_state_id, comparison.latent_state_id)
        row, col = linear_sum_assignment(-contingency.to_numpy())
        alignment = {
            int(contingency.index[r]): int(contingency.columns[c]) for r, c in zip(row, col)
        }
        comparison["primary_aligned_archived_id"] = comparison.primary_state_id.map(alignment)
        concordance_rows = []
        for arm in ["development", "confirmation", "all"]:
            q = comparison if arm == "all" else comparison[comparison.analysis_arm.eq(arm)]
            concordance_rows.append({
                "analysis_arm": arm,
                "n_windows": len(q),
                "hard_state_ari": adjusted_rand_score(q.latent_state_id, q.primary_aligned_archived_id),
                "aligned_label_agreement": float((q.latent_state_id == q.primary_aligned_archived_id).mean()),
            })
        concordance = pd.DataFrame(concordance_rows)

        causal = pd.read_csv(args.archived_analysis_rows, low_memory=False)
        causal_ids = causal[["stay_id"]].drop_duplicates()
        state = work.merge(causal_ids, on="stay_id", how="inner")
        baseline = state[state.bin.eq(0)][
            ["stay_id", "primary_state_id", "primary_state"]
            + [f"primary_p_state_{k}" for k in range(6)]
        ].rename(columns={
            "primary_state_id": "primary_state_id_0_6",
            "primary_state": "primary_state_0_6",
            **{f"primary_p_state_{k}": f"primary_baseline_p_state_{k}" for k in range(6)},
        })
        current = state[state.bin.eq(3)][
            ["stay_id", "primary_state_id", "primary_state"]
            + [f"primary_p_state_{k}" for k in range(6)]
        ].rename(columns={
            "primary_state_id": "primary_state_id_18_24",
            "primary_state": "primary_state_18_24",
            **{f"primary_p_state_{k}": f"primary_p_state_{k}_18_24" for k in range(6)},
        })
        immediate = state[state.bin.eq(4)][
            ["stay_id", "primary_state_id", "primary_state"]
            + [f"primary_p_state_{k}" for k in range(6)]
        ].rename(columns={
            "primary_state_id": "primary_state_id_24_30",
            "primary_state": "primary_state_24_30",
            **{f"primary_p_state_{k}": f"primary_p_state_{k}_24_30" for k in range(6)},
        })
        post_aggs = {
            "primary_post_state_bins": ("bin", "nunique"),
            "primary_renal_transition_burden": (f"primary_p_state_{renal_id}", "mean"),
            "primary_renal_transition_peak": (f"primary_p_state_{renal_id}", "max"),
        }
        for k in range(6):
            post_aggs[f"primary_post_mean_p_state_{k}"] = (f"primary_p_state_{k}", "mean")
        post = state[state.bin.between(4, 7)].groupby("stay_id", as_index=False).agg(**post_aggs)

        drop_existing = [
            c for c in causal.columns
            if c.startswith("primary_")
        ]
        causal = causal.drop(columns=drop_existing, errors="ignore")
        causal = causal.merge(baseline, on="stay_id", how="left")
        causal = causal.merge(current, on="stay_id", how="left")
        causal = causal.merge(immediate, on="stay_id", how="left")
        causal = causal.merge(post, on="stay_id", how="left")
        causal["primary_complete_state_24_48"] = causal.primary_post_state_bins.eq(4)
        causal["primary_pathway_landmark_eligible"] = (
            causal.strict_landmark_48h.eq(True) & causal.primary_complete_state_24_48
        )
        causal.to_csv(
            args.output_dir / "longitudinal_state_transition_analysis_rows_primary_0_48.csv.gz",
            index=False, compression="gzip",
        )
        profiles.to_csv(args.output_dir / "primary_0_48_clinical_state_profiles.csv", index=False)
        concordance.to_csv(args.output_dir / "primary_vs_archived_state_concordance.csv", index=False)

        model_payload = {
            "imputer": imputer,
            "scaler": scaler,
            "gmm": gmm,
            "train_clip_thresholds": train_clip,
            "prediction_clip_thresholds": prediction_clip,
            "state_names": names,
            "renal_state_id": renal_id,
            "features": PHYS,
            "fit_arm": "subject-level development",
            "fit_bins": [0, 1, 2, 3, 4, 5, 6, 7],
            "outcome_free": True,
            "seed": SEED,
        }
        joblib.dump(model_payload, args.model_output)
        audit = {
            "development_subjects": int(development.subject_id.nunique()),
            "development_windows_0_48": int(len(development)),
            "confirmation_subjects": int(work.loc[work.analysis_arm.eq("confirmation"), "subject_id"].nunique()),
            "selected_k_frozen": 6,
            "renal_state_id": int(renal_id),
            "state_names": {str(k): v for k, v in names.items()},
            "archived_alignment": {str(k): int(v) for k, v in alignment.items()},
            "concordance": concordance.to_dict("records"),
            "pathway_landmark_stays": int(causal.primary_pathway_landmark_eligible.sum()),
            "strict_cam_assessed": int(
                causal.loc[causal.primary_pathway_landmark_eligible, "strict_cam_assessed_48_72"].sum()
            ),
            "strict_cam_events": int(
                causal.loc[causal.primary_pathway_landmark_eligible, "strict_cam_positive_48_72"].sum()
            ),
            "boundary": (
                "The primary state model is fitted only in the subject-level development arm "
                "using 0--48 h outcome-free physiology. Component names are physiological "
                "interpretations, not disease diagnoses."
            ),
        }
        (args.output_dir / "primary_0_48_state_model_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        print(json.dumps(audit, indent=2))

    return SimpleNamespace(**locals())

_primary = _load_primary()


# ==============================================================================
# PPD latent-trajectory linkage
# ==============================================================================

def _load_latent():
    import argparse
    import json
    from dataclasses import dataclass
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy.stats import chi2
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler


    PACKAGE = Path(__file__).resolve().parents[2]
    DATA_ROOT = PACKAGE / "data"
    DEFAULT_EXPORT_ROOT = PACKAGE / "outputs" / "latent_export"
    DEFAULT_CLIN_ROOT = DATA_ROOT / "derived" / "latent_trajectory"
    DEFAULT_CLIN_WINDOWS = DATA_ROOT / "derived" / "clinical_order_silent_deployment_windows.csv"
    DEFAULT_STRICT = DATA_ROOT / "derived" / "historical_release" / "ppd_ehr_overlap_analysis_rows.csv"
    HORIZONS = [6, 12, 24, 48]
    H2BIN = {6: 0, 12: 1, 24: 3, 48: 7}
    PROBS = [
        "heart_failure", "renal_failure", "infection", "pneumonia",
        "cerebrovascular", "diabetic_foot",
    ]
    RNG = np.random.default_rng(20260713)


    def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(
            description=(
                "Fit the PPD latent-state model on the seed-42 7:2:1 validation arm "
                "and apply it without refitting to the held-out test arm."
            )
        )
        p.add_argument("--discovery-dir", type=Path, default=DEFAULT_EXPORT_ROOT / "validation")
        p.add_argument("--confirmation-dir", type=Path, default=DEFAULT_EXPORT_ROOT / "test")
        p.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT_ROOT / "downstream")
        p.add_argument("--clinical-root", type=Path, default=DEFAULT_CLIN_ROOT)
        p.add_argument("--clinical-windows", type=Path, default=DEFAULT_CLIN_WINDOWS)
        p.add_argument("--strict-outcome", type=Path, default=DEFAULT_STRICT)
        p.add_argument("--n-cam-bootstrap", type=int, default=300)
        return p.parse_args()


    def load_embeddings(input_dir: Path, arm: str) -> pd.DataFrame:
        rows: list[dict] = []
        for h in HORIZONS:
            p = input_dir / f"h{h:02d}_physiology_only_temporal.npz"
            if not p.exists():
                raise FileNotFoundError(f"Missing {arm} fixed-time export: {p}")
            a = np.load(p, allow_pickle=True)
            valid_icu_anchor = np.isfinite(a["icu_offset_seconds"].astype(float)) & (a["icu_offset_seconds"].astype(float) >= 0)
            eligible = a["landmark_eligible"].astype(bool) & valid_icu_anchor
            emb = a["temporal_mean_repr"].astype(np.float32)
            probs = a["probs"].astype(np.float32)
            for idx in np.where(eligible)[0]:
                r = {
                    "stay_id": int(a["stay_id"][idx]),
                    "subject_id": int(a["subject_id"][idx]),
                    "hadm_id": int(a["hadm_id"][idx]),
                    "subject_disjoint_sensitivity_eligible": bool(a["subject_disjoint_sensitivity_eligible"][idx]),
                    "hadm_disjoint_sensitivity_eligible": bool(a["hadm_disjoint_sensitivity_eligible"][idx]),
                    "horizon_hours": h,
                    "analysis_arm": arm,
                }
                r.update({f"e{j:03d}": float(emb[idx, j]) for j in range(emb.shape[1])})
                r.update({f"prob_{name}": float(probs[idx, j]) for j, name in enumerate(PROBS)})
                rows.append(r)
        out = pd.DataFrame(rows)
        if out.empty:
            raise RuntimeError(f"No landmark-eligible embeddings found in {input_dir}")
        return out


    @dataclass
    class FrozenStateModel:
        horizon_means: pd.DataFrame
        imputer: SimpleImputer
        scaler: StandardScaler
        pca: PCA
        gmm: GaussianMixture
        embedding_columns: list[str]
        selected_k: int

        def transform_predict(self, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            x = d[self.embedding_columns].copy()
            centred = x.copy()
            for h in HORIZONS:
                idx = d.horizon_hours.eq(h)
                if idx.any():
                    centred.loc[idx] = x.loc[idx] - self.horizon_means.loc[h].to_numpy()
            z = self.scaler.transform(self.imputer.transform(centred))
            xp = self.pca.transform(z)
            return self.gmm.predict(xp), self.gmm.predict_proba(xp).max(axis=1)


    def fit_state_model(discovery: pd.DataFrame) -> tuple[FrozenStateModel, pd.DataFrame]:
        ecols = [c for c in discovery if c.startswith("e")]
        x = discovery[ecols].copy()
        horizon_means = x.groupby(discovery.horizon_hours).mean().reindex(HORIZONS)
        if horizon_means.isna().all(axis=1).any():
            raise RuntimeError("Discovery arm lacks at least one required landmark")
        centred = x.copy()
        for h in HORIZONS:
            idx = discovery.horizon_hours.eq(h)
            centred.loc[idx] = x.loc[idx] - horizon_means.loc[h].to_numpy()
        imputer = SimpleImputer(strategy="median").fit(centred)
        scaler = StandardScaler().fit(imputer.transform(centred))
        z = scaler.transform(imputer.transform(centred))
        pca = PCA(n_components=min(20, z.shape[1]), random_state=42).fit(z)
        xp = pca.transform(z)

        rows: list[dict] = []
        models: dict[int, GaussianMixture] = {}
        anchor_idx = RNG.choice(np.arange(len(xp)), size=min(8000, len(xp)), replace=False)
        for k in range(3, 7):
            model = GaussianMixture(
                n_components=k, covariance_type="diag", n_init=5,
                max_iter=400, random_state=42,
            ).fit(xp)
            ref = model.predict(xp[anchor_idx])
            aris = []
            for seed in [101, 102, 103]:
                idx = RNG.choice(np.arange(len(xp)), size=min(15000, len(xp)), replace=True)
                boot = GaussianMixture(
                    n_components=k, covariance_type="diag", n_init=3,
                    max_iter=300, random_state=seed,
                ).fit(xp[idx])
                aris.append(adjusted_rand_score(ref, boot.predict(xp[anchor_idx])))
            rows.append({
                "k": k,
                "bic": model.bic(xp),
                "aic": model.aic(xp),
                "anchor_ARI": float(np.mean(aris)),
                "anchor_ARI_min": float(np.min(aris)),
                "converged": bool(model.converged_),
                "discovery_windows": len(discovery),
            })
            models[k] = model
        selection = pd.DataFrame(rows)
        stable = selection[selection.anchor_ARI.ge(0.60)]
        chosen = int(
            (stable.sort_values("bic").iloc[0] if not stable.empty else
             selection.sort_values(["anchor_ARI", "bic"], ascending=[False, True]).iloc[0]).k
        )
        selection["selected"] = selection.k.eq(chosen)
        frozen = FrozenStateModel(
            horizon_means=horizon_means,
            imputer=imputer,
            scaler=scaler,
            pca=pca,
            gmm=models[chosen],
            embedding_columns=ecols,
            selected_k=chosen,
        )
        return frozen, selection


    def assign_states(d: pd.DataFrame, model: FrozenStateModel) -> pd.DataFrame:
        out = d.copy()
        state, probability = model.transform_predict(out)
        out["ppd_state_id"] = state
        out["ppd_state_probability"] = probability
        return out


    def add_clinical_windows(d: pd.DataFrame, clinical_windows: Path) -> pd.DataFrame:
        cols = [
            "stay_id", "bin", "heart_rate", "map", "temperature", "spo2",
            "glucose_mean", "glucose_cv", "lactate", "creatinine", "bun",
            "wbc", "sofa", "mv_flag", "rass",
        ]
        w = pd.read_csv(clinical_windows, usecols=cols, low_memory=False)
        w["horizon_hours"] = w.bin.map({v: k for k, v in H2BIN.items()})
        w = w[w.horizon_hours.notna()].drop(columns="bin")
        return d.merge(w, on=["stay_id", "horizon_hours"], how="left")


    def name_states(discovery: pd.DataFrame) -> tuple[dict[int, str], pd.DataFrame]:
        profile_cols = [
            "heart_rate", "map", "temperature", "spo2", "glucose_mean",
            "glucose_cv", "lactate", "creatinine", "bun", "wbc", "sofa",
            "mv_flag",
        ] + [f"prob_{p}" for p in PROBS]
        prof = discovery.groupby("ppd_state_id")[profile_cols].mean().reset_index()
        prof["n_windows"] = discovery.groupby("ppd_state_id").size().values
        q = prof.set_index("ppd_state_id")
        names: dict[int, str] = {}
        stable_score = (
            (q.glucose_mean - 130).abs() / 50
            + (q.creatinine - 1).abs() / 1.5
            + (q.lactate - 1.5).abs() / 2
            + (q.sofa - q.sofa.min()) / 10
        )
        stable_id = int(stable_score.idxmin())
        names[stable_id] = "Stable metabolic"
        remaining = [int(i) for i in q.index if int(i) != stable_id]
        if remaining:
            score = q.loc[remaining, "creatinine"] + 3 * q.loc[remaining, "prob_renal_failure"] + q.loc[remaining, "prob_cerebrovascular"]
            state_id = int(score.idxmax())
            names[state_id] = "Cardiorenal-neuro vulnerability"
            remaining.remove(state_id)
        if remaining:
            score = q.loc[remaining, "glucose_mean"] + 150 * q.loc[remaining, "glucose_cv"]
            state_id = int(score.idxmax())
            names[state_id] = "Hyperglycaemic instability"
            remaining.remove(state_id)
        if remaining:
            score = q.loc[remaining, "lactate"] + q.loc[remaining, "wbc"] / 10 + 2 * q.loc[remaining, "prob_infection"] + q.loc[remaining, "prob_pneumonia"]
            state_id = int(score.idxmax())
            names[state_id] = "Inflammatory-respiratory stress"
            remaining.remove(state_id)
        for j, state_id in enumerate(remaining, 1):
            renal_metabolic = (
                q.loc[state_id, "creatinine"] > q.creatinine.median()
                and q.loc[state_id, "bun"] > q.bun.median()
            )
            names[state_id] = (
                "Mixed renal-metabolic stress" if renal_metabolic
                else f"Mixed multisystem state {j}"
            )
        prof["ppd_state"] = prof.ppd_state_id.map(names)
        return names, prof


    def transition_outputs(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        q = d.sort_values(["stay_id", "horizon_hours"]).copy()
        q["next_state"] = q.groupby("stay_id").ppd_state.shift(-1)
        q["next_horizon"] = q.groupby("stay_id").horizon_hours.shift(-1)
        trans = q[q.next_state.notna()].groupby(
            ["horizon_hours", "next_horizon", "ppd_state", "next_state"], as_index=False
        ).agg(n=("stay_id", "size"), n_stays=("stay_id", "nunique"))
        trans["probability"] = trans.n / trans.groupby(
            ["horizon_hours", "next_horizon", "ppd_state"]
        ).n.transform("sum")
        occ = q.groupby(["horizon_hours", "ppd_state"], as_index=False).agg(
            n=("stay_id", "size"), mean_state_probability=("ppd_state_probability", "mean")
        )
        occ["proportion"] = occ.n / occ.groupby("horizon_hours").n.transform("sum")
        return occ, trans


    def intervention_link(
        confirmation: pd.DataFrame, target_state: str, intervention_file: Path,
        analysis_arm: str = "test_confirmation",
    ) -> tuple[pd.DataFrame, dict]:
        state = confirmation.pivot(index="stay_id", columns="horizon_hours", values="ppd_state").reset_index().rename(
            columns={12: "state_12h", 24: "state_24h", 48: "state_48h"}
        )
        ids = confirmation.groupby("stay_id", as_index=False).agg(subject_id=("subject_id", "first"))
        intervention = pd.read_csv(intervention_file, low_memory=False)
        supported_rows = intervention[intervention.bin.between(2, 3)].copy()
        exposure = supported_rows.groupby("stay_id", as_index=False).agg(
            exposure_bins_present=("bin", "nunique"),
            enteral_kcal=("enteral_kcal_sum", lambda x: x.sum(min_count=1)),
            nutrition_instability=("enteral_kcal_abs_change_6h", lambda x: x.sum(min_count=1)),
            treatment_aware_instability=("treatment_aware_kcal_abs_change_6h", lambda x: x.sum(min_count=1)),
            steroid_any=("steroid_flag", "max"),
            prescribed_steroid_hc_equiv=("steroid_rx_hc_equiv_mg_6h", lambda x: x.sum(min_count=1)),
            insulin_units=("insulin_units", lambda x: x.sum(min_count=1)),
            insulin_response_residual=("insulin_response_residual", "mean"),
            propofol_kcal=("propofol_kcal", lambda x: x.sum(min_count=1)),
            sofa=("sofa", "max"), mv_flag=("mv_flag", "max"),
            vasopressor=("vasopressor_sum", "max"), lactate=("lactate", "max"),
            creatinine=("creatinine", "max"), rass=("rass", "min"),
        )
        exposure = exposure[exposure.exposure_bins_present.eq(2)].copy()
        state_eligible = state.dropna(subset=["state_12h", "state_24h"])
        z = state_eligible.merge(ids, on="stay_id").merge(exposure, on="stay_id", how="inner")
        z["transition_to_target"] = z.state_24h.eq(target_state).astype(int)
        numeric = [
            "nutrition_instability", "treatment_aware_instability", "steroid_any",
            "insulin_units", "insulin_response_residual", "propofol_kcal", "sofa",
            "mv_flag", "vasopressor", "lactate", "creatinine", "rass",
        ]
        for c in numeric:
            z[c] = pd.to_numeric(z[c], errors="coerce")
        rows: list[dict] = []
        common = ["sofa", "mv_flag", "vasopressor", "lactate", "creatinine", "rass"]

        def fit(name: str, terms: list[str], work: pd.DataFrame) -> None:
            q = work.dropna(subset=terms).copy()
            for c in common:
                q[c] = q[c].fillna(q[c].median() if q[c].notna().any() else 0)
            if len(q) < 100 or q.transition_to_target.nunique() < 2:
                return
            state_dummies = pd.get_dummies(q.state_12h, prefix="state", drop_first=True, dtype=float)
            x = sm.add_constant(pd.concat([q[terms + common], state_dummies], axis=1), has_constant="add").astype(float)
            model = sm.GLM(q.transition_to_target, x, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": q.subject_id}
            )
            for term in terms:
                rows.append({
                    "analysis_arm": analysis_arm,
                    "model": name, "term": term,
                    "odds_ratio": np.exp(model.params[term]),
                    "ci_lower": np.exp(model.conf_int().loc[term, 0]),
                    "ci_upper": np.exp(model.conf_int().loc[term, 1]),
                    "p_value": model.pvalues[term], "n": len(q),
                    "events": int(q.transition_to_target.sum()),
                })

        nutrition = z.dropna(subset=["nutrition_instability", "steroid_any"]).copy()
        nutrition["nutrition_z"] = (nutrition.nutrition_instability - nutrition.nutrition_instability.mean()) / max(nutrition.nutrition_instability.std(), 1e-8)
        nutrition["nutrition_x_steroid"] = nutrition.nutrition_z * nutrition.steroid_any
        fit("nutrition_x_steroid", ["nutrition_z", "steroid_any", "nutrition_x_steroid"], nutrition)
        insulin = z[z.insulin_units.gt(0) & z.insulin_response_residual.notna()].copy()
        insulin["insulin_residual_z"] = (
            insulin.insulin_response_residual - insulin.insulin_response_residual.mean()
        ) / max(insulin.insulin_response_residual.std(), 1e-8)
        if len(insulin) > 200:
            fit("insulin_response_residual", ["insulin_residual_z", "insulin_units"], insulin)
        propofol = z.dropna(subset=["propofol_kcal"]).copy()
        propofol["propofol_z"] = (propofol.propofol_kcal - propofol.propofol_kcal.mean()) / max(propofol.propofol_kcal.std(), 1e-8)
        fit("propofol_context", ["propofol_z"], propofol)
        support = {
            "analysis_arm": analysis_arm,
            "state_transition_eligible_stays": int(state_eligible.stay_id.nunique()),
            "stays_with_both_12_18h_and_18_24h_exposure_bins": int(exposure.stay_id.nunique()),
            "linked_supported_stays": int(z.stay_id.nunique()),
            "nutrition_complete_stays": int(len(nutrition)),
            "insulin_residual_complete_stays": int(len(insulin)),
            "propofol_complete_stays": int(len(propofol)),
            "missing_exposure_policy": "Exposure terms are complete-case within stays having both 12-18 h and 18-24 h source bins; absent exposure support is not imputed as zero or median.",
        }
        return pd.DataFrame(rows), support


    def strict_cam_anchor(
        confirmation: pd.DataFrame, strict_file: Path, analysis_arm: str = "test_confirmation"
    ) -> pd.DataFrame:
        strict = pd.read_csv(
            strict_file,
            usecols=["stay_id", "strict_followup_assessed_48_72", "strict_cam_positive_48_72"],
        )
        z = confirmation[confirmation.horizon_hours.eq(48)][["stay_id", "subject_id", "ppd_state"]].merge(
            strict, on="stay_id", how="inner"
        )
        z = z[z.strict_followup_assessed_48_72.eq(True) & z.strict_cam_positive_48_72.notna()].copy()
        raw = z.groupby("ppd_state", as_index=False).agg(
            n=("stay_id", "size"), events=("strict_cam_positive_48_72", "sum"),
            cam_risk=("strict_cam_positive_48_72", "mean"),
        )
        raw.insert(0, "analysis_arm", analysis_arm)
        return raw.sort_values("cam_risk", ascending=False)


    def strict_cam_adjusted(
        confirmation: pd.DataFrame, strict_file: Path, n_boot: int,
        analysis_arm: str = "test_confirmation",
        reference_state: str | None = None,
    ) -> pd.DataFrame:
        strict = pd.read_csv(
            strict_file,
            usecols=[
                "stay_id", "strict_followup_assessed_48_72", "strict_cam_positive_48_72",
                "pre_sofa", "pre_mv_flag", "pre_lactate", "pre_creatinine",
                "pre_glucose_mean", "pre_glucose_cv", "pre_tir_fraction",
            ],
        )
        z = confirmation[confirmation.horizon_hours.eq(48)][["stay_id", "subject_id", "ppd_state"]].merge(
            strict, on="stay_id", how="inner"
        )
        z = z[z.strict_followup_assessed_48_72.eq(True) & z.strict_cam_positive_48_72.notna()].copy()
        if z.empty:
            return pd.DataFrame()
        states = sorted(z.ppd_state.dropna().unique())
        if reference_state is None or reference_state not in states:
            raise RuntimeError(f"Prespecified CAM reference state is unavailable: {reference_state}")
        numeric = [
            "pre_sofa", "pre_mv_flag", "pre_lactate", "pre_creatinine",
            "pre_glucose_mean", "pre_glucose_cv", "pre_tir_fraction",
        ]
        for c in numeric:
            z[c] = pd.to_numeric(z[c], errors="coerce")
            z[c] = z[c].fillna(z[c].median())
        state_categories = pd.CategoricalDtype(categories=states, ordered=True)
        state_cols = pd.get_dummies(
            pd.Series(states, dtype=state_categories), prefix="state", drop_first=True, dtype=float
        ).columns
        try:
            global_sd = pd.get_dummies(
                z.ppd_state.astype(state_categories), prefix="state", drop_first=True, dtype=float
            ).reindex(columns=state_cols, fill_value=0)
            base_x = sm.add_constant(z[numeric].astype(float), has_constant="add")
            state_x = sm.add_constant(pd.concat([z[numeric].astype(float), global_sd], axis=1), has_constant="add")
            base_fit = sm.GLM(z.strict_cam_positive_48_72.astype(int), base_x, family=sm.families.Binomial()).fit()
            state_fit = sm.GLM(z.strict_cam_positive_48_72.astype(int), state_x, family=sm.families.Binomial()).fit()
            global_lrt = max(0.0, 2 * (state_fit.llf - base_fit.llf))
            global_df = max(1, len(state_cols))
            global_p = float(chi2.sf(global_lrt, global_df))
        except Exception:
            global_lrt, global_df, global_p = np.nan, len(state_cols), np.nan

        def estimate(q: pd.DataFrame) -> dict[str, float]:
            encoded_state = q.ppd_state.astype(state_categories)
            sd = pd.get_dummies(encoded_state, prefix="state", drop_first=True, dtype=float).reindex(columns=state_cols, fill_value=0)
            x = np.c_[q[numeric].to_numpy(float), sd.to_numpy(float)]
            y = q.strict_cam_positive_48_72.astype(int).to_numpy()
            model = LogisticRegression(max_iter=3000, C=0.5).fit(x, y)
            estimates = {}
            for state in states:
                ss = pd.DataFrame(0.0, index=np.arange(len(q)), columns=state_cols)
                col = f"state_{state}"
                if col in ss:
                    ss[col] = 1.0
                estimates[state] = float(
                    model.predict_proba(np.c_[q[numeric].to_numpy(float), ss.to_numpy(float)])[:, 1].mean()
                )
            return estimates

        point = estimate(z)
        subjects = z.subject_id.drop_duplicates().to_numpy()
        by_subject = {sid: z[z.subject_id.eq(sid)] for sid in subjects}
        boots = {state: [] for state in states}
        for _ in range(n_boot):
            q = pd.concat(
                [by_subject[sid] for sid in RNG.choice(subjects, size=len(subjects), replace=True)],
                ignore_index=True,
            )
            try:
                est = estimate(q)
                for state in states:
                    boots[state].append(est[state])
            except Exception:
                continue
        reference = reference_state
        rows = []
        for state in states:
            arr = np.asarray(boots[state])
            ref_arr = np.asarray(boots[reference])
            rr = arr / np.maximum(ref_arr, 1e-8)
            rows.append({
                "analysis_arm": analysis_arm,
                "ppd_state": state,
                "standardized_cam_risk": point[state],
                "ci_lower": np.quantile(arr, 0.025), "ci_upper": np.quantile(arr, 0.975),
                "reference_state": reference, "risk_ratio": point[state] / point[reference],
                "rr_ci_lower": np.quantile(rr, 0.025), "rr_ci_upper": np.quantile(rr, 0.975),
                "bootstrap_replicates": len(arr),
                "n": int(z.ppd_state.eq(state).sum()),
                "events": int(z.loc[z.ppd_state.eq(state), "strict_cam_positive_48_72"].sum()),
                "analysis_n": len(z),
                "analysis_events": int(z.strict_cam_positive_48_72.sum()),
                "global_state_lrt_chi2": global_lrt,
                "global_state_lrt_df": global_df,
                "global_state_lrt_p": global_p,
                "reference_selection_rule": "Largest 48-h validation occupancy; selected without test CAM outcomes.",
            })
        return pd.DataFrame(rows).sort_values("standardized_cam_risk", ascending=False)


    def clinical_state_concordance(
        confirmation: pd.DataFrame, clinical_states: Path,
        analysis_arm: str = "test_confirmation",
    ) -> pd.DataFrame:
        c = pd.read_csv(clinical_states, usecols=["stay_id", "bin", "latent_state"])
        c = c[c.bin.isin(H2BIN.values())].copy()
        c["horizon_hours"] = c.bin.map({v: k for k, v in H2BIN.items()})
        merged = confirmation.merge(c, on=["stay_id", "horizon_hours"], how="inner")
        return pd.DataFrame([{
            "analysis_arm": analysis_arm, "n_windows": len(merged),
            "n_stays": merged.stay_id.nunique(),
            "ARI": adjusted_rand_score(merged.ppd_state, merged.latent_state),
            "NMI": normalized_mutual_info_score(merged.ppd_state, merged.latent_state),
            "boundary": "Frozen validation-derived PPD states versus independently learned clinical-variable states in the test arm.",
        }])


    def main() -> None:
        args = parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        discovery = load_embeddings(args.discovery_dir, "validation_discovery")
        confirmation = load_embeddings(args.confirmation_dir, "test_confirmation")
        overlap = set(discovery.stay_id.astype(int)) & set(confirmation.stay_id.astype(int))
        if overlap:
            raise RuntimeError(f"Discovery and confirmation arms overlap in {len(overlap)} stay_id values")
        discovery_subjects = set(discovery.subject_id.astype(int))
        discovery_hadm = set(discovery.hadm_id.astype(int))
        confirmation_subjects = set(confirmation.subject_id.astype(int))
        confirmation_hadm = set(confirmation.hadm_id.astype(int))
        subject_overlap = discovery_subjects & confirmation_subjects
        hadm_overlap = discovery_hadm & confirmation_hadm

        frozen, selection = fit_state_model(discovery)
        discovery = add_clinical_windows(assign_states(discovery, frozen), args.clinical_windows)
        confirmation = add_clinical_windows(assign_states(confirmation, frozen), args.clinical_windows)
        names, profiles = name_states(discovery)
        discovery["ppd_state"] = discovery.ppd_state_id.map(names)
        confirmation["ppd_state"] = confirmation.ppd_state_id.map(names)
        confirmation_subject_disjoint = confirmation[
            confirmation.subject_disjoint_sensitivity_eligible.eq(True)
        ].copy()
        confirmation_subject_disjoint["analysis_arm"] = "test_subject_disjoint_sensitivity"

        val_occ, val_trans = transition_outputs(discovery)
        test_occ, test_trans = transition_outputs(confirmation)
        sensitivity_occ, sensitivity_trans = transition_outputs(confirmation_subject_disjoint)
        target = (
            "Hyperglycaemic instability" if "Hyperglycaemic instability" in set(profiles.ppd_state)
            else profiles.assign(score=profiles.glucose_mean + 100 * profiles.glucose_cv).sort_values("score").iloc[-1].ppd_state
        )
        intervention_file = args.clinical_root / "mimic_intervention_features_6h.csv.gz"
        clinical_states = args.clinical_root / "latent_state_assignments.csv.gz"
        cam_reference = str(
            val_occ[val_occ.horizon_hours.eq(48)].sort_values("n", ascending=False).iloc[0].ppd_state
        )
        effects, exposure_support = intervention_link(confirmation, str(target), intervention_file)
        sensitivity_effects, sensitivity_exposure_support = intervention_link(
            confirmation_subject_disjoint, str(target), intervention_file,
            analysis_arm="test_subject_disjoint_sensitivity",
        )
        cam = strict_cam_anchor(confirmation, args.strict_outcome)
        cam_adjusted = strict_cam_adjusted(
            confirmation, args.strict_outcome, args.n_cam_bootstrap,
            reference_state=cam_reference,
        )
        concordance = clinical_state_concordance(confirmation, clinical_states)
        sensitivity_cam = strict_cam_anchor(
            confirmation_subject_disjoint, args.strict_outcome,
            analysis_arm="test_subject_disjoint_sensitivity",
        )
        sensitivity_cam_adjusted = strict_cam_adjusted(
            confirmation_subject_disjoint, args.strict_outcome, args.n_cam_bootstrap,
            analysis_arm="test_subject_disjoint_sensitivity",
            reference_state=cam_reference,
        )
        sensitivity_concordance = clinical_state_concordance(
            confirmation_subject_disjoint, clinical_states,
            analysis_arm="test_subject_disjoint_sensitivity",
        )

        selection.to_csv(args.output_dir / "ppd_latent_model_selection_validation.csv", index=False)
        profiles.to_csv(args.output_dir / "ppd_latent_state_profiles_validation.csv", index=False)
        keep = [
            "stay_id", "subject_id", "hadm_id", "subject_disjoint_sensitivity_eligible",
            "hadm_disjoint_sensitivity_eligible", "horizon_hours", "analysis_arm", "ppd_state_id",
            "ppd_state", "ppd_state_probability",
        ] + [f"prob_{p}" for p in PROBS]
        discovery[keep].to_csv(args.output_dir / "ppd_fixed_time_latent_states_validation.csv.gz", index=False, compression="gzip")
        confirmation[keep].to_csv(args.output_dir / "ppd_fixed_time_latent_states_test.csv.gz", index=False, compression="gzip")
        confirmation_subject_disjoint[keep].to_csv(
            args.output_dir / "ppd_fixed_time_latent_states_test_subject_disjoint.csv.gz",
            index=False, compression="gzip",
        )
        val_occ.to_csv(args.output_dir / "ppd_state_occupancy_validation.csv", index=False)
        val_trans.to_csv(args.output_dir / "ppd_state_transitions_validation.csv", index=False)
        test_occ.to_csv(args.output_dir / "ppd_state_occupancy_test.csv", index=False)
        test_trans.to_csv(args.output_dir / "ppd_state_transitions_test.csv", index=False)
        sensitivity_occ.to_csv(args.output_dir / "ppd_state_occupancy_test_subject_disjoint.csv", index=False)
        sensitivity_trans.to_csv(args.output_dir / "ppd_state_transitions_test_subject_disjoint.csv", index=False)
        effects.to_csv(args.output_dir / "ppd_intervention_conditioned_transition_effects_test.csv", index=False)
        sensitivity_effects.to_csv(
            args.output_dir / "ppd_intervention_conditioned_transition_effects_test_subject_disjoint.csv",
            index=False,
        )
        pd.DataFrame([exposure_support, sensitivity_exposure_support]).to_csv(
            args.output_dir / "ppd_intervention_exposure_support_audit.csv", index=False
        )
        cam.to_csv(args.output_dir / "ppd_state_strict_cam_anchor_test.csv", index=False)
        cam_adjusted.to_csv(args.output_dir / "ppd_state_strict_cam_adjusted_bootstrap_test.csv", index=False)
        concordance.to_csv(args.output_dir / "ppd_vs_clinical_state_concordance_test.csv", index=False)
        sensitivity_cam.to_csv(args.output_dir / "ppd_state_strict_cam_anchor_test_subject_disjoint.csv", index=False)
        sensitivity_cam_adjusted.to_csv(
            args.output_dir / "ppd_state_strict_cam_adjusted_bootstrap_test_subject_disjoint.csv",
            index=False,
        )
        sensitivity_concordance.to_csv(
            args.output_dir / "ppd_vs_clinical_state_concordance_test_subject_disjoint.csv",
            index=False,
        )
        joblib.dump({
            "horizon_means": frozen.horizon_means,
            "imputer": frozen.imputer,
            "scaler": frozen.scaler,
            "pca": frozen.pca,
            "gmm": frozen.gmm,
            "embedding_columns": frozen.embedding_columns,
            "selected_k": frozen.selected_k,
            "state_names": names,
            "fit_arm": "seed42 validation",
        }, args.output_dir / "ppd_validation_fitted_state_model.joblib")

        summary = {
            "checkpoint_protocol": "seed42_70_20_10",
            "state_discovery_arm": "20% validation",
            "state_confirmation_arm": "10% test held out from gradient updates and checkpoint selection",
            "validation_stays": int(discovery.stay_id.nunique()),
            "test_stays": int(confirmation.stay_id.nunique()),
            "validation_embedding_windows": len(discovery),
            "test_embedding_windows": len(confirmation),
            "stay_id_overlap": 0,
            "subject_id_overlap": len(subject_overlap),
            "hadm_id_overlap": len(hadm_overlap),
            "test_stays_with_subject_seen_in_any_other_split_arm": int(
                confirmation.groupby("stay_id").subject_disjoint_sensitivity_eligible.first().eq(False).sum()
            ),
            "test_stays_with_hadm_seen_in_any_other_split_arm": int(
                confirmation.groupby("stay_id").hadm_disjoint_sensitivity_eligible.first().eq(False).sum()
            ),
            "test_subject_disjoint_stays": int(confirmation_subject_disjoint.stay_id.nunique()),
            "test_subject_disjoint_embedding_windows": len(confirmation_subject_disjoint),
            "selected_k_on_validation": frozen.selected_k,
            "target_state_named_on_validation": str(target),
            "cam_reference_state_selected_on_validation": cam_reference,
            "test_strict_cam_assessed": int(cam.n.sum()) if len(cam) else 0,
            "test_strict_cam_events": int(cam.events.sum()) if len(cam) else 0,
            "test_subject_disjoint_strict_cam_assessed": int(sensitivity_cam.n.sum()) if len(sensitivity_cam) else 0,
            "test_subject_disjoint_strict_cam_events": int(sensitivity_cam.events.sum()) if len(sensitivity_cam) else 0,
            "claim_boundary": (
                "PCA, preprocessing, GMM selection and state naming were fitted only in the seed-42 validation arm. "
                "The test arm was held out from gradient updates and checkpoint selection, received frozen state "
                "assignments, and was used for CAM anchoring and intervention-conditioned candidate associations. "
                "Historical training records include test metrics, which limits a strict prospective-independence claim."
            ),
        }
        (args.output_dir / "ppd_latent_linkage_seed42_721_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    return SimpleNamespace(**locals())

_latent = _load_latent()


# ==============================================================================
# eICU clinical-state transport
# ==============================================================================

def _load_transport():
    import argparse
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import gaussian_kde, pearsonr, spearmanr
    from sklearn.compose import ColumnTransformer
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, normalized_mutual_info_score
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler


    PACKAGE = Path(__file__).resolve().parents[2]
    OUT = PACKAGE / "outputs" / "downstream" / "eicu_transport"
    DATA_ROOT = PACKAGE / "data"
    MIMIC_WINDOWS = DATA_ROOT / "derived" / "clinical_order_silent_deployment_windows.csv"
    MIMIC_ASSIGN = DATA_ROOT / "derived" / "latent_trajectory" / "latent_state_assignments.csv.gz"
    MIMIC_OCC = MIMIC_ASSIGN.parent / "state_occupancy_dm_vs_nondm.csv"
    MIMIC_DIFF = MIMIC_ASSIGN.parent / "state_transition_differences_dm_minus_nondm.csv"
    EICU_WINDOWS = DATA_ROOT / "derived" / "eicu_master_cohort_en_repaired.csv"

    PHYS = [
        "heart_rate", "map", "temperature", "spo2", "glucose_mean", "glucose_cv",
        "lactate", "creatinine", "bun", "wbc",
    ]
    BOUNDS = {
        "heart_rate": (20, 250), "map": (20, 200), "temperature": (30, 43),
        "spo2": (60, 100), "glucose_mean": (20, 1000), "glucose_cv": (0, 3),
        "lactate": (0, 30), "creatinine": (0, 20), "bun": (0, 200), "wbc": (0, 200),
    }
    RNG = np.random.default_rng(20260713)


    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, default=OUT)
        parser.add_argument("--mimic-windows", type=Path, default=MIMIC_WINDOWS)
        parser.add_argument("--mimic-assignments", type=Path, default=MIMIC_ASSIGN)
        parser.add_argument("--mimic-occupancy", type=Path, default=None)
        parser.add_argument("--mimic-transition-differences", type=Path, default=None)
        parser.add_argument("--eicu-windows", type=Path, default=EICU_WINDOWS)
        parser.add_argument("--bootstrap", type=int, default=200, help="Stay-level bootstrap replicates; use 0 to skip.")
        return parser.parse_args()


    TRANSPORT_STATES = [
        "Stable metabolic", "Hyperglycaemic instability", "Renal-metabolic vulnerability",
        "Inflammatory-haemodynamic stress", "Mixed vulnerability 1", "Mixed vulnerability 2",
    ]


    def _compact_stay_edges(assignments: pd.DataFrame) -> pd.DataFrame:
        edges = assignments.sort_values(["stay_id", "bin"]).copy()
        edges["next_state"] = edges.groupby("stay_id").transported_state.shift(-1)
        edges["next_bin"] = edges.groupby("stay_id").bin.shift(-1)
        edges = edges[edges.next_state.notna() & edges.next_bin.eq(edges.bin + 1)].copy()
        edges["edge"] = edges.transported_state + "|||" + edges.next_state
        return edges.groupby(["diabetes", "stay_id", "edge"], as_index=False).agg(
            weighted_n=("transport_weight", "sum")
        )


    def _transition_difference_vector(edges: pd.DataFrame) -> np.ndarray:
        index = pd.MultiIndex.from_product([TRANSPORT_STATES, TRANSPORT_STATES], names=["state", "next_state"])
        vectors = []
        for diabetes in (0, 1):
            group = edges[edges.diabetes.eq(diabetes)].copy()
            split = group.edge.str.split(r"\|\|\|", expand=True)
            group["state"], group["next_state"] = split[0], split[1]
            counts = group.groupby(["state", "next_state"]).weighted_n.sum().reindex(index, fill_value=0.0)
            denominator = counts.groupby(level=0).transform("sum").replace(0, np.nan)
            vectors.append((counts / denominator).fillna(0).to_numpy())
        return vectors[1] - vectors[0]


    def bootstrap_transition_uncertainty(assignments: pd.DataFrame, mimic_differences: Path,
                                         output_dir: Path, replicates: int) -> None:
        """Bootstrap cross-cohort transition replication at the eICU stay level."""
        rng = np.random.default_rng(20260713)
        compact = _compact_stay_edges(assignments)
        mimic = pd.read_csv(mimic_differences)
        index = pd.MultiIndex.from_product(
            [TRANSPORT_STATES, TRANSPORT_STATES], names=["latent_state", "next_state"]
        )
        mimic_vector = mimic.set_index(["latent_state", "next_state"])[
            "difference_dm_minus_non_dm"
        ].reindex(index).to_numpy(float)
        stay_ids = {
            diabetes: compact.loc[compact.diabetes.eq(diabetes), "stay_id"].unique()
            for diabetes in (0, 1)
        }
        rows, edge_rows = [], []
        for replicate in range(replicates):
            sampled_groups = []
            for diabetes in (0, 1):
                sampled = rng.choice(stay_ids[diabetes], size=len(stay_ids[diabetes]), replace=True)
                multiplicity = pd.Series(sampled).value_counts().rename("mult").rename_axis("stay_id").reset_index()
                group = compact[compact.diabetes.eq(diabetes)].merge(multiplicity, on="stay_id", how="inner")
                group["weighted_n"] *= group["mult"]
                sampled_groups.append(group.drop(columns="mult"))
            external_vector = _transition_difference_vector(pd.concat(sampled_groups, ignore_index=True))
            rows.append({
                "replicate": replicate,
                "pearson_r": pearsonr(mimic_vector, external_vector).statistic,
                "spearman_r": spearmanr(mimic_vector, external_vector).statistic,
                "sign_concordance": np.mean(np.sign(mimic_vector) == np.sign(external_vector)),
            })
            for origin, target in [
                ("Stable metabolic", "Hyperglycaemic instability"),
                ("Renal-metabolic vulnerability", "Hyperglycaemic instability"),
            ]:
                position = TRANSPORT_STATES.index(origin) * len(TRANSPORT_STATES) + TRANSPORT_STATES.index(target)
                edge_rows.append({
                    "replicate": replicate, "state": origin, "next_state": target,
                    "difference_dm_minus_non_dm": external_vector[position],
                })

        replicate_table = pd.DataFrame(rows)
        edge_table = pd.DataFrame(edge_rows)
        replicate_table.to_csv(output_dir / "eicu_transition_replication_bootstrap_replicates.csv", index=False)
        edge_table.to_csv(output_dir / "eicu_key_edge_bootstrap_replicates.csv", index=False)
        summary = []
        for column in ["pearson_r", "spearman_r", "sign_concordance"]:
            values = replicate_table[column]
            summary.append({
                "metric": column, "estimate": values.mean(), "ci_lower": values.quantile(.025),
                "ci_upper": values.quantile(.975), "replicates": replicates,
            })
        pd.DataFrame(summary).to_csv(output_dir / "eicu_transition_replication_bootstrap_summary.csv", index=False)
        edge_summary = edge_table.groupby(["state", "next_state"]).difference_dm_minus_non_dm.agg(
            estimate="mean", ci_lower=lambda x: x.quantile(.025),
            ci_upper=lambda x: x.quantile(.975), replicates="count",
        ).reset_index()
        edge_summary.to_csv(output_dir / "eicu_key_edge_bootstrap_summary.csv", index=False)
        contract = {
            "replicates": replicates,
            "sampling_unit": "eICU stay_id stratified by diabetes",
            "state_model": "fixed transported MIMIC clinical-state classifier",
        }
        (output_dir / "eicu_transition_replication_bootstrap_contract.json").write_text(
            json.dumps(contract, indent=2), encoding="utf-8"
        )


    def clean_raw(frame: pd.DataFrame, quantiles: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
        x = frame[PHYS].copy()
        for c, (lo, hi) in BOUNDS.items():
            x[c] = pd.to_numeric(x[c], errors="coerce")
            if c == "temperature":
                f = x[c].between(70, 120)
                x.loc[f, c] = (x.loc[f, c] - 32) * 5 / 9
            x[c] = x[c].where(x[c].between(lo, hi))
            if quantiles and c in quantiles:
                qlo, qhi = quantiles[c]
                x[c] = x[c].clip(qlo, qhi)
        return x


    def derive_quantiles(x: pd.DataFrame) -> dict[str, tuple[float, float]]:
        out = {}
        for c in PHYS:
            s = pd.to_numeric(x[c], errors="coerce")
            if s.notna().sum() > 100:
                lo, hi = s.quantile([0.0025, 0.9975])
                out[c] = (float(lo), float(hi))
        return out


    class DiagonalGaussianStateClassifier:
        def fit(self, x: np.ndarray, y: np.ndarray):
            self.labels_ = np.asarray(sorted(pd.unique(y)), dtype=object)
            self.means_, self.vars_, self.logpriors_ = [], [], []
            for label in self.labels_:
                z = x[y == label]
                self.means_.append(z.mean(axis=0))
                self.vars_.append(np.maximum(z.var(axis=0), 1e-3))
                self.logpriors_.append(np.log(len(z) / len(x)))
            self.means_ = np.asarray(self.means_)
            self.vars_ = np.asarray(self.vars_)
            self.logpriors_ = np.asarray(self.logpriors_)
            return self

        def decision_function(self, x: np.ndarray) -> np.ndarray:
            d = x[:, None, :] - self.means_[None, :, :]
            return -0.5 * ((d * d) / self.vars_[None, :, :]).sum(axis=2) - 0.5 * np.log(
                self.vars_
            ).sum(axis=1)[None, :] + self.logpriors_[None, :]

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            s = self.decision_function(x)
            s = s - s.max(axis=1, keepdims=True)
            p = np.exp(s)
            return p / p.sum(axis=1, keepdims=True)

        def predict(self, x: np.ndarray) -> np.ndarray:
            return self.labels_[self.decision_function(x).argmax(axis=1)]


    def load_mimic_labelled() -> pd.DataFrame:
        a = pd.read_csv(MIMIC_ASSIGN, usecols=["stay_id", "bin", "latent_state"])
        ids = set(a.stay_id.astype(int))
        parts = []
        use = ["stay_id", "bin"] + PHYS
        for chunk in pd.read_csv(MIMIC_WINDOWS, usecols=use, chunksize=250_000, low_memory=False):
            q = chunk[chunk.stay_id.isin(ids)]
            if not q.empty:
                parts.append(q)
        w = pd.concat(parts, ignore_index=True)
        return a.merge(w, on=["stay_id", "bin"], how="inner", validate="one_to_one")


    def load_eicu() -> pd.DataFrame:
        use = [
            "stay_id", "bin", "bin_start_hour", "bin_end_hour", "diabetes", "anchor_age",
            "sex_male", "severity_score", "unittype", "icu_los_hours", "delirium_flag", "gcs",
        ] + PHYS
        parts = []
        for chunk in pd.read_csv(EICU_WINDOWS, usecols=use, chunksize=250_000, low_memory=False):
            q = chunk[chunk.bin.between(0, 7)].copy()
            q = q[pd.to_numeric(q.icu_los_hours, errors="coerce") >= pd.to_numeric(q.bin_end_hour, errors="coerce")]
            if not q.empty:
                parts.append(q)
        d = pd.concat(parts, ignore_index=True)
        return d.sort_values(["stay_id", "bin"]).drop_duplicates(["stay_id", "bin"])


    def fit_transport_classifier(mimic: pd.DataFrame):
        initial = clean_raw(mimic)
        quantiles = derive_quantiles(initial)
        clean = clean_raw(mimic, quantiles)
        stays = mimic.stay_id.astype(np.int64)
        train = (pd.util.hash_pandas_object(stays, index=False).to_numpy() % 5) != 0
        imp = SimpleImputer(strategy="median").fit(clean.loc[train])
        sc = StandardScaler().fit(imp.transform(clean.loc[train]))
        xtr = sc.transform(imp.transform(clean.loc[train]))
        xte = sc.transform(imp.transform(clean.loc[~train]))
        ytr = mimic.loc[train, "latent_state"].astype(str).to_numpy()
        yte = mimic.loc[~train, "latent_state"].astype(str).to_numpy()
        clf = DiagonalGaussianStateClassifier().fit(xtr, ytr)
        pred = clf.predict(xte)
        fidelity = {
            "n_train_windows": int(train.sum()),
            "n_test_windows": int((~train).sum()),
            "accuracy": float((pred == yte).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "ARI": float(adjusted_rand_score(yte, pred)),
            "NMI": float(normalized_mutual_info_score(yte, pred)),
        }
        # Refit preprocessing and classifier on all labelled MIMIC windows.
        imp = SimpleImputer(strategy="median").fit(clean)
        sc = StandardScaler().fit(imp.transform(clean))
        x = sc.transform(imp.transform(clean))
        clf = DiagonalGaussianStateClassifier().fit(x, mimic.latent_state.astype(str).to_numpy())
        return quantiles, imp, sc, clf, x, fidelity


    def kde_overlap(a: np.ndarray, b: np.ndarray) -> float:
        lo = min(np.quantile(a, 0.005), np.quantile(b, 0.005))
        hi = max(np.quantile(a, 0.995), np.quantile(b, 0.995))
        g = np.linspace(lo, hi, 512)
        return float(np.trapezoid(np.minimum(gaussian_kde(a)(g), gaussian_kde(b)(g)), g))


    def support_geometry(xm: np.ndarray, xe: np.ndarray) -> dict:
        mi = RNG.choice(len(xm), min(100_000, len(xm)), replace=False)
        ei = RNG.choice(len(xe), min(100_000, len(xe)), replace=False)
        both = np.vstack([xm[mi], xe[ei]])
        pca = PCA(n_components=min(8, both.shape[1]), random_state=42).fit(xm[mi])
        pm, pe = pca.transform(xm[mi]), pca.transform(xe[ei])
        lw = LedoitWolf().fit(pm)
        mdm = np.sqrt(lw.mahalanobis(pm)); mde = np.sqrt(lw.mahalanobis(pe)); q95 = np.quantile(mdm, .95)
        overlaps = [kde_overlap(pm[:, j], pe[:, j]) for j in range(min(5, pm.shape[1]))]
        xx = np.vstack([pm, pe]); yy = np.r_[np.zeros(len(pm)), np.ones(len(pe))]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=.2)
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        auc = cross_val_score(clf, xx, yy, cv=cv, scoring="roc_auc").mean()
        return {
            "mimic_windows_sampled": len(pm), "eicu_windows_sampled": len(pe),
            "mahalanobis_mimic_median": float(np.median(mdm)),
            "mahalanobis_eicu_median": float(np.median(mde)),
            "mahalanobis_eicu_to_mimic_ratio": float(np.median(mde)/np.median(mdm)),
            "eicu_above_mimic_95pct": float((mde > q95).mean()),
            "kde_overlap_pc1_to_pc5_mean": float(np.mean(overlaps)),
            "kde_overlap_pc1_to_pc5_min": float(np.min(overlaps)),
            "dataset_classifier_auroc": float(auc),
            "boundary": "Clinical-variable state transport geometry; not PPD-EHR embedding transport.",
        }


    def refit_eicu_states(eicu: pd.DataFrame, xe: np.ndarray, transported: np.ndarray, mimic_clf) -> tuple[np.ndarray, dict]:
        idx = RNG.choice(len(xe), min(100_000, len(xe)), replace=False)
        gmm = GaussianMixture(n_components=len(mimic_clf.labels_), covariance_type="diag", n_init=3, max_iter=300, random_state=42).fit(xe[idx])
        cluster = gmm.predict(xe)
        cmeans = np.vstack([xe[cluster == k].mean(axis=0) for k in range(gmm.n_components)])
        cost = ((cmeans[:, None, :] - mimic_clf.means_[None, :, :]) ** 2).sum(axis=2)
        row, col = linear_sum_assignment(cost)
        mapping = {int(r): str(mimic_clf.labels_[c]) for r, c in zip(row, col)}
        aligned = np.asarray([mapping[int(k)] for k in cluster], dtype=object)
        return aligned, {
            "n_components": gmm.n_components,
            "converged": bool(gmm.converged_),
            "transport_vs_eicu_refit_ARI": float(adjusted_rand_score(transported, aligned)),
            "transport_vs_eicu_refit_NMI": float(normalized_mutual_info_score(transported, aligned)),
            "mapping": mapping,
        }


    def propensity_weights(e: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        base = e.sort_values(["stay_id", "bin"]).groupby("stay_id", as_index=False).first()
        num = ["anchor_age", "sex_male", "severity_score"]
        cat = ["unittype"]
        prep = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ])
        model = Pipeline([("prep", prep), ("lr", LogisticRegression(max_iter=2000, C=.5))])
        y = base.diabetes.astype(int)
        model.fit(base[num + cat], y)
        ps = np.clip(model.predict_proba(base[num + cat])[:, 1], .01, .99)
        prev = y.mean()
        w = np.where(y.eq(1), prev/ps, (1-prev)/(1-ps))
        lo, hi = np.quantile(w, [.01, .99]); base["transport_weight"] = np.clip(w, lo, hi)
        balance=[]
        for c in num:
            x=pd.to_numeric(base[c],errors="coerce")
            def smd(weight=None):
                vals=[]
                for g in [1,0]:
                    m=y.eq(g)&x.notna(); ww=np.ones(m.sum()) if weight is None else weight[m]
                    xx=x[m].to_numpy(); mu=np.average(xx,weights=ww); var=np.average((xx-mu)**2,weights=ww); vals.append((mu,var))
                return (vals[0][0]-vals[1][0])/np.sqrt((vals[0][1]+vals[1][1])/2)
            balance.append({"variable":c,"smd_unweighted":smd(),"smd_weighted":smd(base.transport_weight.to_numpy())})
        return e.merge(base[["stay_id","transport_weight"]],on="stay_id",how="left"), pd.DataFrame(balance)


    def occupancy_transitions(e: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        occ=e.groupby(["diabetes","bin","transported_state"],as_index=False).agg(weighted_n=("transport_weight","sum"),n=("stay_id","size"),n_stays=("stay_id","nunique"))
        occ["proportion"]=occ.weighted_n/occ.groupby(["diabetes","bin"]).weighted_n.transform("sum")
        q=e.sort_values(["stay_id","bin"]).copy(); q["next_state"]=q.groupby("stay_id").transported_state.shift(-1); q["next_bin"]=q.groupby("stay_id").bin.shift(-1)
        q=q[q.next_state.notna()&q.next_bin.eq(q.bin+1)]
        tp=q.groupby(["diabetes","transported_state","next_state"],as_index=False).agg(weighted_n=("transport_weight","sum"),n=("stay_id","size"),n_stays=("stay_id","nunique"))
        tp["probability"]=tp.weighted_n/tp.groupby(["diabetes","transported_state"]).weighted_n.transform("sum")
        p1=tp[tp.diabetes.eq(1)].rename(columns={"probability":"p_dm"}); p0=tp[tp.diabetes.eq(0)].rename(columns={"probability":"p_non_dm"})
        diff=p1[["transported_state","next_state","p_dm"]].merge(p0[["transported_state","next_state","p_non_dm"]],on=["transported_state","next_state"],how="outer").fillna(0)
        diff["difference_dm_minus_non_dm"]=diff.p_dm-diff.p_non_dm
        return occ,tp,diff


    def replication_metrics(eocc: pd.DataFrame, ediff: pd.DataFrame) -> pd.DataFrame:
        mocc=pd.read_csv(MIMIC_OCC); mocc=mocc[mocc.bin.between(0,7)].rename(columns={"latent_state":"state","proportion":"mimic"})
        eo=eocc.rename(columns={"transported_state":"state","proportion":"eicu"})
        om=mocc[["diabetes","bin","state","mimic"]].merge(eo[["diabetes","bin","state","eicu"]],on=["diabetes","bin","state"],how="inner")
        md=pd.read_csv(MIMIC_DIFF).rename(columns={"latent_state":"state","difference_dm_minus_non_dm":"mimic"})
        ed=ediff.rename(columns={"transported_state":"state","difference_dm_minus_non_dm":"eicu"})
        dm=md[["state","next_state","mimic"]].merge(ed[["state","next_state","eicu"]],on=["state","next_state"],how="inner")
        top=dm.loc[dm.mimic.abs().nlargest(min(10,len(dm))).index]
        rows=[]
        for name,z in [("state_occupancy",om),("transition_dm_minus_nondm",dm),("top10_transition_dm_minus_nondm",top)]:
            rows.append({"comparison":name,"n_cells":len(z),"pearson_r":pearsonr(z.mimic,z.eicu).statistic,"spearman_r":spearmanr(z.mimic,z.eicu).statistic,"sign_concordance":float((np.sign(z.mimic)==np.sign(z.eicu)).mean()),"mean_absolute_difference":float(np.mean(np.abs(z.mimic-z.eicu)))})
        return pd.DataFrame(rows)


    def main():
        global OUT, MIMIC_WINDOWS, MIMIC_ASSIGN, MIMIC_OCC, MIMIC_DIFF, EICU_WINDOWS
        args = parse_args()
        OUT = args.output_dir
        MIMIC_WINDOWS = args.mimic_windows
        MIMIC_ASSIGN = args.mimic_assignments
        MIMIC_OCC = args.mimic_occupancy or (MIMIC_ASSIGN.parent / "state_occupancy_dm_vs_nondm.csv")
        MIMIC_DIFF = args.mimic_transition_differences or (
            MIMIC_ASSIGN.parent / "state_transition_differences_dm_minus_nondm.csv"
        )
        EICU_WINDOWS = args.eicu_windows
        for path in [MIMIC_WINDOWS, MIMIC_ASSIGN, MIMIC_OCC, MIMIC_DIFF, EICU_WINDOWS]:
            if not path.is_file():
                raise FileNotFoundError(path)
        OUT.mkdir(parents=True,exist_ok=True)
        mimic=load_mimic_labelled(); eicu=load_eicu()
        quant,imp,sc,clf,xm,fidelity=fit_transport_classifier(mimic)
        raw_e=clean_raw(eicu,quant); observed=raw_e.notna().sum(axis=1); xe=sc.transform(imp.transform(raw_e))
        prob=clf.predict_proba(xe); transported=clf.labels_[prob.argmax(axis=1)]
        eicu=eicu.copy(); eicu["transported_state"]=transported; eicu["transported_state_probability"]=prob.max(axis=1); eicu["n_phys_observed"]=observed
        geometry=support_geometry(xm,xe); refit,refit_audit=refit_eicu_states(eicu,xe,transported,clf); eicu["eicu_refit_aligned_state"]=refit
        eicu,balance=propensity_weights(eicu); occ,tp,diff=occupancy_transitions(eicu); replication=replication_metrics(occ,diff)

        # MAP is completely unavailable in the current eICU panel. Refit a frozen
        # nine-variable MIMIC transport classifier excluding MAP and repeat the
        # external dynamic comparison as a prespecified missing-variable sensitivity.
        phys9 = [c for c in PHYS if c != "map"]
        m9 = clean_raw(mimic, quant)[phys9]
        e9 = raw_e[phys9]
        imp9 = SimpleImputer(strategy="median").fit(m9)
        sc9 = StandardScaler().fit(imp9.transform(m9))
        xm9 = sc9.transform(imp9.transform(m9)); xe9 = sc9.transform(imp9.transform(e9))
        clf9 = DiagonalGaussianStateClassifier().fit(xm9, mimic.latent_state.astype(str).to_numpy())
        state9 = clf9.predict(xe9)
        e9frame = eicu.copy(); e9frame["transported_state"] = state9
        occ9, tp9, diff9 = occupancy_transitions(e9frame)
        replication9 = replication_metrics(occ9, diff9)
        replication["analysis_variant"] = "ten_variable_map_median_imputed"
        replication9["analysis_variant"] = "nine_variable_map_excluded"
        replication = pd.concat([replication, replication9], ignore_index=True)
        map_sensitivity = {
            "map_eicu_observed_fraction": float(pd.to_numeric(eicu["map"], errors="coerce").notna().mean()),
            "primary_vs_map_excluded_ARI": float(adjusted_rand_score(eicu.transported_state, state9)),
            "primary_vs_map_excluded_NMI": float(normalized_mutual_info_score(eicu.transported_state, state9)),
            "boundary": "The current eICU panel contains no usable MAP values; the nine-variable analysis is required for external interpretation.",
        }

        # Profile replication on the same plausibility-cleaned scale used for state
        # assignment; raw chart outliers must not enter clinical profile summaries.
        mimic_clean = clean_raw(mimic, quant).copy()
        mimic_clean["transported_state"] = mimic.latent_state.astype(str).to_numpy()
        eicu_clean = raw_e.copy()
        eicu_clean["transported_state"] = eicu.transported_state.astype(str).to_numpy()
        mp=mimic_clean.groupby("transported_state")[PHYS].mean()
        ep=eicu_clean.groupby("transported_state")[PHYS].mean().reindex(mp.index)
        centre = mimic_clean[PHYS].mean()
        scale = mimic_clean[PHYS].std(ddof=0).replace(0, np.nan)
        mpz = (mp - centre) / scale
        epz = (ep - centre) / scale
        profiles=[]
        for state in mp.index:
            ok=mpz.loc[state].notna()&epz.loc[state].notna()
            profiles.append({"state":state,"profile_z_pearson_r":pearsonr(mpz.loc[state,ok],epz.loc[state,ok]).statistic,"profile_z_rmse":float(np.sqrt(np.mean((mpz.loc[state,ok]-epz.loc[state,ok])**2))),"n_features":int(ok.sum()),**{f"mimic_{c}":mp.loc[state,c] for c in PHYS},**{f"eicu_{c}":ep.loc[state,c] for c in PHYS}})
        profiles=pd.DataFrame(profiles)

        neuro=pd.DataFrame([{
            "eicu_windows":len(eicu),"eicu_stays":eicu.stay_id.nunique(),"delirium_flag_positive_windows":int(pd.to_numeric(eicu.delirium_flag,errors="coerce").fillna(0).gt(0).sum()),"delirium_flag_positive_stays":int(eicu.assign(flag=pd.to_numeric(eicu.delirium_flag,errors="coerce").fillna(0).gt(0)).groupby("stay_id").flag.max().sum()),"gcs_observed_windows":int(pd.to_numeric(eicu.gcs,errors="coerce").notna().sum()),"boundary":"eICU delirium_flag is not CAM-equivalent and was not used to validate the MIMIC strict CAM endpoint."}])
        support=pd.DataFrame([{"feature":c,"mimic_observed_fraction":float(pd.to_numeric(mimic[c],errors="coerce").notna().mean()),"eicu_observed_fraction":float(pd.to_numeric(eicu[c],errors="coerce").notna().mean())} for c in PHYS])
        support_by_bin=eicu.groupby("bin",as_index=False).agg(
            n_windows=("stay_id","size"), n_stays=("stay_id","nunique"),
            dm_windows=("diabetes","sum"), median_state_probability=("transported_state_probability","median"),
            median_phys_observed=("n_phys_observed","median"),
        )
        support_by_bin["low_confidence_below_0_60"]=[float((eicu.loc[eicu.bin.eq(b),"transported_state_probability"]<.60).mean()) for b in support_by_bin.bin]

        eicu[["stay_id","bin","bin_start_hour","bin_end_hour","diabetes","transported_state","transported_state_probability","eicu_refit_aligned_state","n_phys_observed","transport_weight"]].to_csv(OUT/"eicu_transported_state_assignments.csv.gz",index=False,compression="gzip")
        occ.to_csv(OUT/"eicu_state_occupancy_by_dm.csv",index=False); tp.to_csv(OUT/"eicu_state_transition_probabilities_by_dm.csv",index=False); diff.to_csv(OUT/"eicu_state_transition_differences_dm_minus_nondm.csv",index=False)
        replication.to_csv(OUT/"mimic_eicu_dynamic_replication_metrics.csv",index=False); profiles.to_csv(OUT/"mimic_eicu_state_profile_replication.csv",index=False); balance.to_csv(OUT/"eicu_diabetes_weighting_balance.csv",index=False); support.to_csv(OUT/"mimic_eicu_variable_support.csv",index=False); support_by_bin.to_csv(OUT/"eicu_state_transport_support_by_bin.csv",index=False); neuro.to_csv(OUT/"eicu_neurologic_outcome_support_audit.csv",index=False)
        (OUT/"mimic_state_surrogate_fidelity.json").write_text(json.dumps(fidelity,indent=2),encoding="utf8"); (OUT/"clinical_state_transport_geometry.json").write_text(json.dumps(geometry,indent=2),encoding="utf8"); (OUT/"eicu_refit_state_audit.json").write_text(json.dumps(refit_audit,indent=2),encoding="utf8")
        (OUT/"eicu_map_missing_sensitivity.json").write_text(json.dumps(map_sensitivity,indent=2),encoding="utf8")
        summary={"mimic_labelled_windows":len(mimic),"mimic_labelled_stays":int(mimic.stay_id.nunique()),"eicu_landmark_windows":len(eicu),"eicu_stays":int(eicu.stay_id.nunique()),"eicu_dm_stays":int(eicu.groupby('stay_id').diabetes.first().sum()),"surrogate_fidelity":fidelity,"geometry":geometry,"refit":refit_audit,"map_sensitivity":map_sensitivity,"claim_boundary":"External transport of a clinical-variable state taxonomy; not PPD-EHR embedding validation and not strict CAM replication."}
        (OUT/"eicu_clinical_state_transport_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8"); print(json.dumps(summary,indent=2))
        if args.bootstrap > 0:
            bootstrap_transition_uncertainty(eicu, MIMIC_DIFF, OUT, args.bootstrap)

    return SimpleNamespace(**locals())

_transport = _load_transport()


# ==============================================================================
# Domain-applicability audit
# ==============================================================================

def _load_domain():
    import argparse
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from scipy.stats import gaussian_kde, ks_2samp
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler


    PACKAGE = Path(__file__).resolve().parents[2]
    DATA_ROOT = PACKAGE / "data"
    DEFAULT_EXPORT_ROOT = PACKAGE / "outputs" / "latent_export"
    STRICT = DATA_ROOT / "derived" / "historical_release" / "ppd_ehr_overlap_analysis_rows.csv"
    INTERVENTION = DATA_ROOT / "derived" / "latent_trajectory" / "mimic_intervention_features_6h.csv.gz"
    ADMIN_SOURCE = DATA_ROOT / "derived" / "clinical_mechanism" / "administered_exposures_0_24h.csv"
    HORIZONS = [6, 12, 24, 48]
    RNG = np.random.default_rng(20260713)


    def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(
            description="Validation-referenced applicability audit for the held-out seed-42 test arm."
        )
        p.add_argument("--discovery-dir", type=Path, default=DEFAULT_EXPORT_ROOT / "validation")
        p.add_argument("--confirmation-dir", type=Path, default=DEFAULT_EXPORT_ROOT / "test")
        p.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT_ROOT / "downstream")
        p.add_argument("--n-bootstrap", type=int, default=300)
        return p.parse_args()


    def macro_metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
        aucs, aprs = [], []
        for j in range(y.shape[1]):
            yy = (y[:, j] > 0.5).astype(int)
            if 0 < yy.sum() < len(yy):
                aucs.append(roc_auc_score(yy, p[:, j]))
                aprs.append(average_precision_score(yy, p[:, j]))
        return float(np.mean(aucs)), float(np.mean(aprs))


    def patient_bootstrap_indices(subjects: np.ndarray) -> np.ndarray:
        unique = np.unique(subjects)
        by = {sid: np.flatnonzero(subjects == sid) for sid in unique}
        sampled = RNG.choice(unique, size=len(unique), replace=True)
        return np.concatenate([by[sid] for sid in sampled])


    def kde_overlap_1d(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 10 or len(b) < 10 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return float("nan")
        lo = min(np.quantile(a, 0.005), np.quantile(b, 0.005))
        hi = max(np.quantile(a, 0.995), np.quantile(b, 0.995))
        grid = np.linspace(lo, hi, 384)
        return float(np.trapezoid(np.minimum(gaussian_kde(a)(grid), gaussian_kde(b)(grid)), grid))


    def fit_reference_geometry(x: np.ndarray) -> dict:
        scaler = StandardScaler().fit(x)
        z = scaler.transform(x)
        full = PCA(random_state=42).fit(z)
        n95 = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), 0.95) + 1)
        ncomp = max(2, min(20, n95, z.shape[1]))
        pca = PCA(n_components=ncomp, random_state=42).fit(z)
        xp = pca.transform(z)
        covariance = LedoitWolf().fit(xp)
        md = np.sqrt(covariance.mahalanobis(xp))
        return {
            "scaler": scaler, "pca": pca, "covariance": covariance,
            "reference_pc": xp, "reference_md": md,
            "reference_95": float(np.quantile(md, 0.95)),
            "pca_components": ncomp,
            "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        }


    def evaluate_target_geometry(reference: dict, target_x: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        target_pc = reference["pca"].transform(reference["scaler"].transform(target_x))
        target_md = np.sqrt(reference["covariance"].mahalanobis(target_pc))
        overlaps = [
            kde_overlap_1d(reference["reference_pc"][:, j], target_pc[:, j])
            for j in range(min(5, reference["pca_components"]))
        ]
        return {
            "mahalanobis_validation_median": float(np.median(reference["reference_md"])),
            "mahalanobis_target_median": float(np.median(target_md)),
            "mahalanobis_target_to_validation_ratio": float(np.median(target_md) / np.median(reference["reference_md"])),
            "target_above_validation_95pct": float((target_md > reference["reference_95"]).mean()),
            "mahalanobis_ks_target_vs_validation": float(ks_2samp(target_md, reference["reference_md"]).statistic),
            "mahalanobis_ks_p": float(ks_2samp(target_md, reference["reference_md"]).pvalue),
            "kde_overlap_pc1_to_pc5_mean": float(np.nanmean(overlaps)),
            "kde_overlap_pc1_to_pc5_min": float(np.nanmin(overlaps)),
        }, target_pc, target_md


    def grouped_domain_auc(
        reference_pc: np.ndarray, target_pc: np.ndarray,
        reference_subject: np.ndarray, target_subject: np.ndarray,
    ) -> float:
        x = np.vstack([reference_pc, target_pc])
        y = np.r_[np.zeros(len(reference_pc), dtype=int), np.ones(len(target_pc), dtype=int)]
        groups = np.r_[reference_subject, target_subject]
        if y.sum() < 25 or (1 - y).sum() < 25:
            return float("nan")
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        model = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.2)
        return float(cross_val_score(model, x, y, groups=groups, cv=cv, scoring="roc_auc").mean())


    def bootstrap_target(
        reference: dict, target_x: np.ndarray, target_y: np.ndarray,
        target_p: np.ndarray, target_subject: np.ndarray,
        reference_metrics: tuple[float, float], n_boot: int,
    ) -> dict:
        values = []
        for _ in range(n_boot):
            idx = patient_bootstrap_indices(target_subject)
            try:
                auc, apr = macro_metrics(target_y[idx], target_p[idx])
                geometry, _, _ = evaluate_target_geometry(reference, target_x[idx])
                values.append([
                    auc, apr, auc - reference_metrics[0], apr - reference_metrics[1],
                    geometry["mahalanobis_target_median"],
                    geometry["kde_overlap_pc1_to_pc5_mean"],
                ])
            except (ValueError, np.linalg.LinAlgError):
                continue
        a = np.asarray(values)
        names = [
            "auroc", "auprc", "delta_auroc", "delta_auprc",
            "mahalanobis_target_median", "kde_overlap_pc1_to_pc5_mean",
        ]
        out = {"patient_bootstrap_replicates": len(a)}
        if len(a):
            for j, name in enumerate(names):
                out[f"{name}_ci_lower"] = float(np.quantile(a[:, j], 0.025))
                out[f"{name}_ci_upper"] = float(np.quantile(a[:, j], 0.975))
        return out


    def load_arm(input_dir: Path, horizon: int) -> dict:
        a = np.load(input_dir / f"h{horizon:02d}_physiology_only_temporal.npz", allow_pickle=True)
        valid_icu_anchor = np.isfinite(a["icu_offset_seconds"].astype(float)) & (a["icu_offset_seconds"].astype(float) >= 0)
        eligible = a["landmark_eligible"].astype(bool) & valid_icu_anchor
        return {
            "stay": a["stay_id"].astype(int)[eligible],
            "subject": a["subject_id"].astype(int)[eligible],
            "x": a["temporal_mean_repr"].astype(float)[eligible],
            "y": a["labels"].astype(float)[eligible],
            "p": a["probs"].astype(float)[eligible],
        }


    def load_target_sets() -> dict[str, set[int]]:
        strict = pd.read_csv(
            STRICT,
            usecols=["stay_id", "strict_landmark_48h", "strict_followup_assessed_48_72", "strict_cam_positive_48_72"],
        )
        cam = strict[
            strict.strict_landmark_48h.eq(True)
            & strict.strict_followup_assessed_48_72.eq(True)
            & strict.strict_cam_positive_48_72.notna()
        ]
        intervention = pd.read_csv(INTERVENTION, usecols=["stay_id", "bin"])
        supported = intervention[intervention.bin.between(2, 3)].groupby("stay_id").bin.nunique()
        intervention_ids = set(supported[supported.eq(2)].index.astype(int))
        admin_ids: set[int] = set()
        if ADMIN_SOURCE.exists():
            admin_ids = set(pd.read_csv(ADMIN_SOURCE, usecols=["stay_id"]).stay_id.astype(int))
        return {
            "clinical_timewindow_linked": set(strict.stay_id.astype(int)),
            "intervention_12_24h_supported": intervention_ids,
            "strict_cam_analysis": set(cam.stay_id.astype(int)),
            "separate_admin_source_linked": admin_ids,
        }


    def main() -> None:
        args = parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        targets = load_target_sets()
        geometry_rows, performance_rows = [], []
        validation_stays, test_stays = set(), set()
        for h in HORIZONS:
            validation = load_arm(args.discovery_dir, h)
            test = load_arm(args.confirmation_dir, h)
            validation_stays.update(validation["stay"].tolist())
            test_stays.update(test["stay"].tolist())
            reference = fit_reference_geometry(validation["x"])
            reference_metrics = macro_metrics(validation["y"], validation["p"])

            for name, idset in targets.items():
                if name == "strict_cam_analysis" and h != 48:
                    continue
                target_mask = np.asarray([int(stay) in idset for stay in test["stay"]], dtype=bool)
                if target_mask.sum() < 20:
                    continue
                target_x = test["x"][target_mask]
                target_y = test["y"][target_mask]
                target_p = test["p"][target_mask]
                target_subject = test["subject"][target_mask]
                geometry, target_pc, _ = evaluate_target_geometry(reference, target_x)
                geometry["domain_classifier_auroc_grouped_by_subject"] = grouped_domain_auc(
                    reference["reference_pc"], target_pc,
                    validation["subject"], target_subject,
                )
                auc, apr = macro_metrics(target_y, target_p)
                boot = bootstrap_target(
                    reference, target_x, target_y, target_p, target_subject,
                    reference_metrics, args.n_bootstrap,
                )
                geometry_rows.append({
                    "horizon_hours": h, "target_subset": name,
                    "reference": "20% validation arm only",
                    "n_reference": len(validation["stay"]), "n_target": int(target_mask.sum()),
                    "reference_unique_subjects": int(np.unique(validation["subject"]).size),
                    "target_unique_subjects": int(np.unique(target_subject).size),
                    "pca_components": reference["pca_components"],
                    "pca_variance_explained": reference["pca_variance_explained"],
                    **geometry,
                    "mahalanobis_target_median_ci_lower": boot.get("mahalanobis_target_median_ci_lower", np.nan),
                    "mahalanobis_target_median_ci_upper": boot.get("mahalanobis_target_median_ci_upper", np.nan),
                    "kde_overlap_pc1_to_pc5_mean_ci_lower": boot.get("kde_overlap_pc1_to_pc5_mean_ci_lower", np.nan),
                    "kde_overlap_pc1_to_pc5_mean_ci_upper": boot.get("kde_overlap_pc1_to_pc5_mean_ci_upper", np.nan),
                    "patient_bootstrap_replicates": boot["patient_bootstrap_replicates"],
                    "estimand_boundary": "Preprocessing, PCA and covariance were fitted only on validation. Test target rows never enter the reference fit.",
                })
                performance_rows.append({
                    "horizon_hours": h, "target_subset": name,
                    "reference": "20% validation arm only", "n": int(target_mask.sum()),
                    "unique_subjects": int(np.unique(target_subject).size),
                    "macro_auroc": auc, "macro_auprc": apr,
                    "delta_auroc_vs_validation": auc - reference_metrics[0],
                    "delta_auprc_vs_validation": apr - reference_metrics[1],
                    "validation_macro_auroc": reference_metrics[0],
                    "validation_macro_auprc": reference_metrics[1],
                    "auroc_ci_lower": boot.get("auroc_ci_lower", np.nan),
                    "auroc_ci_upper": boot.get("auroc_ci_upper", np.nan),
                    "auprc_ci_lower": boot.get("auprc_ci_lower", np.nan),
                    "auprc_ci_upper": boot.get("auprc_ci_upper", np.nan),
                    "delta_auroc_ci_lower": boot.get("delta_auroc_ci_lower", np.nan),
                    "delta_auroc_ci_upper": boot.get("delta_auroc_ci_upper", np.nan),
                    "delta_auprc_ci_lower": boot.get("delta_auprc_ci_lower", np.nan),
                    "delta_auprc_ci_upper": boot.get("delta_auprc_ci_upper", np.nan),
                    "patient_bootstrap_replicates": boot["patient_bootstrap_replicates"],
                    "inference_boundary": "Patient-cluster bootstrap in the test target subset; validation reference metric is fixed.",
                })

        geometry = pd.DataFrame(geometry_rows)
        performance = pd.DataFrame(performance_rows)
        geometry.to_csv(args.output_dir / "domain_applicability_internal_geometry_test.csv", index=False)
        performance.to_csv(args.output_dir / "domain_applicability_same_task_performance_test.csv", index=False)
        non_dm = {
            "dm_vs_non_dm_embedding_transport": "not_estimable",
            "reason": "No non-DM cohort exists in the exact model-ready sparse-event schema.",
        }
        (args.output_dir / "domain_applicability_non_dm_status.json").write_text(
            json.dumps(non_dm, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = {
            "checkpoint_protocol": "seed42_70_20_10",
            "reference_arm": "20% validation only",
            "target_arm": "10% test held out from gradient updates and checkpoint selection; historical workflow recorded test metrics",
            "validation_stays_with_any_eligible_landmark": len(validation_stays),
            "test_stays_with_any_eligible_landmark": len(test_stays),
            "stay_id_overlap": len(validation_stays & test_stays),
            "target_rows_entered_reference_fit": False,
            "bootstrap_unit": "subject_id",
            "geometry_rows": len(geometry), "performance_rows": len(performance),
        }
        (args.output_dir / "domain_applicability_summary_test.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    return SimpleNamespace(**locals())

_domain = _load_domain()


STAGES = {
    "dataset": _state_dataset.main,
    "primary-model": _primary.main,
    "latent-linkage": _latent.main,
    "eicu-transport": _transport.main,
    "domain-audit": _domain.main,
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Clinical-state and external-transport analyses')
    parser.add_argument("stage", choices=STAGES)
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
