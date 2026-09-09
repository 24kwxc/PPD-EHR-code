"""Longitudinal pathway and sensitivity analyses.

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

import analysis_clinical_states as _clinical_states
_register_legacy(
    "build_longitudinal_state_transition_dataset_20260713",
    _clinical_states._state_dataset,
)


# ==============================================================================
# Full-ICU longitudinal dataset
# ==============================================================================

def _load_full_icu():
    """Build the full-ICU secondary longitudinal pathway cohort.

    The primary paper-aligned analysis remains restricted to the PPD-EHR diabetes
    cohort.  This secondary cohort applies the frozen patient-disjoint 0--48 h
    clinical state model to all 92,083 MIMIC-IV stays to improve precision and test
    whether the longitudinal pathway differs by diabetes status.
    """

    import argparse
    import json
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd

    from build_longitudinal_state_transition_dataset_20260713 import (
        cleaned_phys, stable_subject_arm, zscore_contract,
    )


    def parse_args() -> argparse.Namespace:
        package = Path(__file__).resolve().parents[2]
        data_root = package / "data"
        source_root = data_root / "derived" / "clinical_mechanism"
        p = argparse.ArgumentParser()
        p.add_argument(
            "--clinical-table", type=Path,
            default=source_root / "clinical_mechanism_final_20260712"
            / "clinical_mechanism_analysis_table.csv",
        )
        p.add_argument(
            "--clinical-windows", type=Path,
            default=data_root / "derived" / "clinical_order_silent_deployment_windows.csv",
        )
        p.add_argument(
            "--intervention-windows", type=Path,
            default=data_root / "derived" / "latent_trajectory" / "mimic_intervention_features_6h.csv.gz",
        )
        p.add_argument(
            "--state-model", type=Path,
            default=package / "model" / "clinical_state_gmm_primary_0_48h_20260713.joblib",
        )
        p.add_argument("--icustays-table", type=Path, default=data_root / "restricted" / "mimic_iv" / "icu" / "icustays.csv.gz")
        p.add_argument("--admissions-table", type=Path, default=data_root / "restricted" / "mimic_iv" / "hosp" / "admissions.csv.gz")
        p.add_argument(
            "--output-dir", type=Path,
            default=package / "outputs" / "downstream" / "longitudinal_pathway_full_icu",
        )
        return p.parse_args()


    def main() -> None:
        args = parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for p in [args.clinical_table, args.clinical_windows, args.intervention_windows,
                  args.state_model, args.icustays_table, args.admissions_table]:
            if not p.exists():
                raise FileNotFoundError(p)
        clinical = pd.read_csv(args.clinical_table, low_memory=False)
        model = joblib.load(args.state_model)
        state_names = {int(k): v for k, v in model["state_names"].items()}
        renal_id = int(model["renal_state_id"])
        phys = list(model["features"])

        raw_cols = [
            "stay_id", "subject_id", "hadm_id", "bin", "bin_start_hour", "bin_end_hour",
            "caloric_target_kcal_day", "weight_kg", "sofa", "mv_flag", "rrt_flag",
            "vasopressor_sum", "rass", "ppi_h2_flag", "antiemetic_flag",
            "heparin_amount_sum", "midazolam_amount_sum", "dexmedetomidine_amount_sum",
        ] + phys
        raw = pd.read_csv(args.clinical_windows, usecols=lambda c: c in set(raw_cols), low_memory=False)
        raw = raw[raw.stay_id.isin(clinical.stay_id)].copy().sort_values(["stay_id", "bin"])
        x = model["scaler"].transform(
            model["imputer"].transform(cleaned_phys(raw, model["prediction_clip_thresholds"]))
        )
        probs = model["gmm"].predict_proba(x)
        raw["primary_state_id"] = probs.argmax(axis=1)
        raw["primary_state"] = raw.primary_state_id.map(state_names)
        for k in range(6):
            raw[f"primary_p_state_{k}"] = probs[:, k]

        baseline = raw[raw.bin.eq(0)][
            ["stay_id", "primary_state_id", "primary_state"] + [f"primary_p_state_{k}" for k in range(6)]
        ].rename(columns={
            "primary_state_id": "primary_state_id_0_6", "primary_state": "primary_state_0_6",
            **{f"primary_p_state_{k}": f"primary_baseline_p_state_{k}" for k in range(6)},
        })
        current = raw[raw.bin.eq(3)][
            ["stay_id", "primary_state_id", "primary_state"] + [f"primary_p_state_{k}" for k in range(6)]
        ].rename(columns={
            "primary_state_id": "primary_state_id_18_24", "primary_state": "primary_state_18_24",
            **{f"primary_p_state_{k}": f"primary_p_state_{k}_18_24" for k in range(6)},
        })
        immediate = raw[raw.bin.eq(4)][
            ["stay_id", "primary_state_id", "primary_state"] + [f"primary_p_state_{k}" for k in range(6)]
        ].rename(columns={
            "primary_state_id": "primary_state_id_24_30", "primary_state": "primary_state_24_30",
            **{f"primary_p_state_{k}": f"primary_p_state_{k}_24_30" for k in range(6)},
        })
        post_aggs = {
            "primary_post_state_bins": ("bin", "nunique"),
            "primary_renal_transition_burden": (f"primary_p_state_{renal_id}", "mean"),
            "primary_renal_transition_peak": (f"primary_p_state_{renal_id}", "max"),
        }
        for k in range(6):
            post_aggs[f"primary_post_mean_p_state_{k}"] = (f"primary_p_state_{k}", "mean")
        post = raw[raw.bin.between(4, 7)].groupby("stay_id", as_index=False).agg(**post_aggs)

        interventions = pd.read_csv(args.intervention_windows, low_memory=False)
        interventions = interventions[interventions.stay_id.isin(clinical.stay_id)]
        ew = interventions[interventions.bin.between(1, 3)].copy().sort_values(["stay_id", "bin"])
        ew["enteral_internal_abs_change_6_24"] = ew.groupby("stay_id")["enteral_kcal_sum"].diff().abs().fillna(0)
        exposure = ew.groupby("stay_id", as_index=False).agg(
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
        raw_exp = raw[raw.bin.between(1, 3)].groupby("stay_id", as_index=False).agg(
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
        exposure = exposure.merge(raw_exp, on="stay_id", how="left")

        rows = clinical.merge(baseline, on="stay_id", how="left")
        rows = rows.merge(current, on="stay_id", how="left").merge(immediate, on="stay_id", how="left")
        rows = rows.merge(post, on="stay_id", how="left").merge(exposure, on="stay_id", how="left")
        rows["analysis_arm"] = rows.subject_id.map(stable_subject_arm)
        wmed = pd.to_numeric(rows.loc[rows.analysis_arm.eq("development"), "weight_kg_exposure"], errors="coerce").median()
        rows["analysis_weight_kg"] = pd.to_numeric(rows.weight_kg_exposure, errors="coerce").fillna(float(wmed))
        rows["caloric_target_kcal_day"] = pd.to_numeric(rows.caloric_target_kcal_day, errors="coerce").fillna(25 * rows.analysis_weight_kg)
        target = (rows.caloric_target_kcal_day * 18 / 24).clip(lower=1)
        rows["energy_adequacy_6_24"] = rows.enteral_kcal_6_24 / target
        rows["energy_deficit_6_24"] = (1 - rows.energy_adequacy_6_24).clip(lower=0, upper=2)
        rows["energy_tv_scaled_6_24"] = rows.enteral_tv_6_24 / target
        rows["zero_feed_fraction_6_24"] = rows.zero_feed_bins_6_24 / 3
        rows["feeding_event_fraction_6_24"] = (
            rows.feeding_restart_count_6_24 + rows.feeding_interruption_count_6_24
        ) / 6
        rows["dextrose_g_per_kg_6_24"] = rows.dextrose_g_6_24 / rows.analysis_weight_kg.clip(lower=20)
        rows["insulin_units_per_kg_6_24"] = rows.insulin_units_6_24 / rows.analysis_weight_kg.clip(lower=20)
        rows["propofol_kcal_per_kg_6_24"] = rows.propofol_kcal_6_24 / rows.analysis_weight_kg.clip(lower=20)
        score_contract = {}; pieces = []
        for c in ["energy_deficit_6_24", "energy_tv_scaled_6_24", "zero_feed_fraction_6_24", "feeding_event_fraction_6_24"]:
            z, contract = zscore_contract(rows[c], rows.analysis_arm)
            rows[f"z_{c}"] = z; pieces.append(z); score_contract[c] = contract
        rows["energy_instability_score"] = pd.concat(pieces, axis=1).mean(axis=1)
        dev = rows.loc[rows.analysis_arm.eq("development"), "energy_instability_score"]
        regimes = {"stable_energy_q25": float(dev.quantile(.25)), "unstable_energy_q75": float(dev.quantile(.75))}
        rows["strict_cam_assessed_48_72"] = rows.strict_cam_positive_48_72.notna()
        rows["primary_complete_state_24_48"] = rows.primary_post_state_bins.eq(4)
        rows["primary_pathway_landmark_eligible"] = rows.strict_landmark_48h.eq(True) & rows.primary_complete_state_24_48

        icu = pd.read_csv(args.icustays_table, usecols=["stay_id", "hadm_id", "intime", "outtime"], parse_dates=["intime", "outtime"])
        adm = pd.read_csv(args.admissions_table, usecols=["hadm_id", "dischtime", "deathtime"], parse_dates=["dischtime", "deathtime"])
        rows = rows.merge(icu, on=["stay_id", "hadm_id"], how="left", validate="one_to_one").merge(adm, on="hadm_id", how="left", validate="many_to_one")
        h48 = rows.intime + pd.to_timedelta(48, unit="h"); h72 = rows.intime + pd.to_timedelta(72, unit="h")
        rows["death_48_72"] = rows.deathtime.between(h48, h72, inclusive="both")
        rows["icu_discharge_time_48_72"] = rows.outtime.between(h48, h72, inclusive="both")
        rows.to_csv(args.output_dir / "longitudinal_state_transition_analysis_rows_full_icu.csv.gz", index=False, compression="gzip")
        audit = {
            "stays": int(rows.stay_id.nunique()), "subjects": int(rows.subject_id.nunique()),
            "pathway_landmark": int(rows.primary_pathway_landmark_eligible.sum()),
            "strict_cam_assessed": int(rows.loc[rows.primary_pathway_landmark_eligible, "strict_cam_assessed_48_72"].sum()),
            "strict_cam_events": int(rows.loc[rows.primary_pathway_landmark_eligible, "strict_cam_positive_48_72"].sum()),
            "diabetes_landmark": int(rows.loc[rows.primary_pathway_landmark_eligible, "diabetes"].sum()),
            "death_48_72": int(rows.loc[rows.primary_pathway_landmark_eligible, "death_48_72"].sum()),
            "regime_values": regimes, "energy_score_contract": score_contract,
            "boundary": "Full-ICU secondary cohort; diabetes interaction/subgroup must be reported and PPD alignment is secondary.",
        }
        (args.output_dir / "full_icu_longitudinal_dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(json.dumps(audit, indent=2))

    return SimpleNamespace(**locals())

_full_icu = _load_full_icu()


# ==============================================================================
# Longitudinal state-transition pathway
# ==============================================================================

def _load_pathway():
    """Run the longitudinal state-transition pathway analysis.

    Primary chronology:
        L0 0--6 h -> treatment/nutrition 6--24 h -> soft clinical state
        trajectory 24--48 h -> strict CAM 48--72 h.

    The primary estimator is patient-grouped cross-fitted conditional-mean
    g-computation with explicit modelling of CAM observation.  Confidence intervals
    are subject-cluster bootstraps of cross-fitted individual standardized
    contributions, conditional on the fitted nuisance models.  The script does not
    claim a complete sequential TMLE implementation.
    """

    import argparse
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from scipy.special import expit, logit
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, mean_absolute_error, r2_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler


    SEED = 20260713
    RNG = np.random.default_rng(SEED)
    STATE_NAMES = {
        0: "Inflammatory-haemodynamic stress",
        1: "Mixed vulnerability 1",
        2: "Stable metabolic",
        3: "Renal-metabolic vulnerability",
        4: "Mixed vulnerability 2",
        5: "Hyperglycaemic instability",
    }


    def parse_args() -> argparse.Namespace:
        package = Path(__file__).resolve().parents[2]
        p = argparse.ArgumentParser()
        p.add_argument(
            "--analysis-rows", type=Path,
            default=package / "outputs" / "downstream" / "longitudinal_pathway"
            / "longitudinal_state_transition_analysis_rows_primary_0_48.csv.gz",
        )
        p.add_argument(
            "--dataset-audit", type=Path,
            default=package / "outputs" / "downstream" / "longitudinal_pathway"
            / "longitudinal_dataset_audit.json",
        )
        p.add_argument(
            "--validation-h24", type=Path,
            default=package / "outputs" / "latent_export" / "validation"
            / "h24_physiology_only_temporal.npz",
        )
        p.add_argument(
            "--test-h24", type=Path,
            default=package / "outputs" / "latent_export" / "test"
            / "h24_physiology_only_temporal.npz",
        )
        p.add_argument(
            "--output-dir", type=Path,
            default=package / "outputs" / "downstream" / "longitudinal_pathway",
        )
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--bootstrap", type=int, default=1000)
        p.add_argument("--atlas-bootstrap", type=int, default=500)
        p.add_argument("--max-iter", type=int, default=180)
        p.add_argument(
            "--target-population",
            default="PPD-EHR-overlap diabetes ICU stays surviving to the 48-h landmark with complete 24-48 h state trajectory",
        )
        return p.parse_args()


    def numeric(s: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(s):
            return s.astype(float)
        return pd.to_numeric(s, errors="coerce")


    def feature_frame(data: pd.DataFrame, include_mediator: bool = False) -> pd.DataFrame:
        numeric_cols = [
            "anchor_age", "charlson_index", "admission_sofa", "pre_sofa",
            "pre_vasopressor_sum", "pre_mv_flag", "pre_rrt_flag", "pre_rass",
            "pre_lactate", "pre_creatinine", "pre_glucose_mean", "pre_glucose_cv",
            "pre_tir_fraction", "pre_opioid_any", "pre_steroid_flag",
            "pre_enteral_kcal_sum", "pre_0_6_insulin_units", "pre_0_6_dextrose_g",
            "pre_0_6_propofol_mg", "primary_baseline_p_state_0",
            "primary_baseline_p_state_1", "primary_baseline_p_state_2",
            "primary_baseline_p_state_3", "primary_baseline_p_state_4",
            "primary_baseline_p_state_5", "energy_adequacy_6_24",
            "energy_instability_score", "dextrose_g_per_kg_6_24",
            "insulin_units_per_kg_6_24", "prescribed_steroid_any_6_24",
            "propofol_kcal_per_kg_6_24", "midazolam_6_24", "dexmedetomidine_6_24",
            "procedure_mv_proxy_any_6_24", "rass_min_6_24",
            "vasopressor_max_6_24", "rrt_any_6_24",
        ]
        if include_mediator:
            numeric_cols.append("primary_renal_transition_burden")
        cats = ["gender", "admission_type", "first_careunit"]
        out = pd.DataFrame(index=data.index)
        for c in numeric_cols:
            if c in data:
                x = numeric(data[c])
                if c in {
                    "pre_vasopressor_sum", "pre_enteral_kcal_sum", "pre_0_6_insulin_units",
                    "pre_0_6_dextrose_g", "pre_0_6_propofol_mg", "dextrose_g_per_kg_6_24",
                    "insulin_units_per_kg_6_24", "propofol_kcal_per_kg_6_24",
                    "midazolam_6_24", "dexmedetomidine_6_24", "vasopressor_max_6_24",
                }:
                    x = np.log1p(x.clip(lower=0))
                out[c] = x
        cat = pd.get_dummies(
            data[[c for c in cats if c in data]].fillna("Missing").astype(str),
            prefix=[c for c in cats if c in data], dtype=float,
        )
        return pd.concat([out, cat], axis=1)


    def hgb_reg(max_iter: int, seed: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.045, max_iter=max_iter,
            max_leaf_nodes=15, min_samples_leaf=35, l2_regularization=1.0,
            random_state=seed,
        )


    def hgb_clf(max_iter: int, seed: int) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.045, max_iter=max_iter,
            max_leaf_nodes=15, min_samples_leaf=35, l2_regularization=1.0,
            random_state=seed,
        )


    def cluster_bootstrap_indices(subjects: np.ndarray, n_boot: int, rng: np.random.Generator):
        unique = np.unique(subjects)
        positions = {sid: np.where(subjects == sid)[0] for sid in unique}
        for _ in range(n_boot):
            draw = rng.choice(unique, size=len(unique), replace=True)
            yield np.concatenate([positions[sid] for sid in draw])


    def interval(values: np.ndarray) -> tuple[float, float]:
        return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


    def bootstrap_summary(
        individual: pd.DataFrame,
        regime_names: list[str],
        n_boot: int,
        arm: str,
        contrast_label: str = "unstable_energy_plus_prescribed_steroid_vs_stable_energy_no_steroid",
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
        q = individual if arm == "all" else individual[individual.analysis_arm.eq(arm)].copy()
        rng = np.random.default_rng(SEED + {"all": 0, "development": 11, "confirmation": 23}[arm])
        boot_values: dict[str, list[float]] = {}
        for r in regime_names:
            boot_values[f"m_{r}"] = []
            boot_values[f"y_{r}"] = []
        for key in ["total_effect", "direct_effect", "indirect_effect"]:
            boot_values[key] = []
        for idx in cluster_bootstrap_indices(q.subject_id.to_numpy(), n_boot, rng):
            b = q.iloc[idx]
            for r in regime_names:
                boot_values[f"m_{r}"].append(float(b[f"m_{r}"].mean()))
                boot_values[f"y_{r}"].append(float(b[f"y_{r}"].mean()))
            y00 = b["y_reference_m_reference"].mean()
            y10 = b["y_adverse_m_reference"].mean()
            y11 = b["y_adverse_m_adverse"].mean()
            boot_values["total_effect"].append(float(y11 - y00))
            boot_values["direct_effect"].append(float(y10 - y00))
            boot_values["indirect_effect"].append(float(y11 - y10))
        arrays = {k: np.asarray(v, float) for k, v in boot_values.items()}

        regimes = []
        for r in regime_names:
            m = float(q[f"m_{r}"].mean())
            y = float(q[f"y_{r}"].mean())
            ml, mu = interval(arrays[f"m_{r}"])
            yl, yu = interval(arrays[f"y_{r}"])
            regimes.extend([
                {"analysis_arm": arm, "regime": r, "estimand": "renal_transition_burden",
                 "estimate": m, "ci_lower": ml, "ci_upper": mu, "scale": "probability",
                 "n": len(q), "subjects": q.subject_id.nunique()},
                {"analysis_arm": arm, "regime": r, "estimand": "strict_cam_risk",
                 "estimate": y, "ci_lower": yl, "ci_upper": yu, "scale": "risk",
                 "n": len(q), "subjects": q.subject_id.nunique()},
            ])
        path = []
        point = {
            "total_effect": float(q.y_adverse_m_adverse.mean() - q.y_reference_m_reference.mean()),
            "direct_effect": float(q.y_adverse_m_reference.mean() - q.y_reference_m_reference.mean()),
            "indirect_effect": float(q.y_adverse_m_adverse.mean() - q.y_adverse_m_reference.mean()),
        }
        for e in ["total_effect", "direct_effect", "indirect_effect"]:
            lo, hi = interval(arrays[e])
            path.append({
                "analysis_arm": arm, "contrast": contrast_label,
                "estimand": e, "risk_difference": point[e], "ci_lower": lo, "ci_upper": hi,
                "risk_difference_per_1000": 1000 * point[e], "ci_lower_per_1000": 1000 * lo,
                "ci_upper_per_1000": 1000 * hi, "bootstrap_replicates": n_boot,
                "n_target": len(q), "subjects": q.subject_id.nunique(),
                "method": "patient-grouped cross-fitted conditional-mean longitudinal g-computation",
                "boundary": "Observational interventional analogue; nuisance-model uncertainty is conditional in this bootstrap.",
            })
        return pd.DataFrame(regimes), pd.DataFrame(path), arrays


    def fit_assessment_weights(data: pd.DataFrame, folds: int, max_iter: int) -> tuple[np.ndarray, pd.DataFrame]:
        x = feature_frame(data, include_mediator=True)
        y = data.strict_cam_assessed_48_72.astype(int).to_numpy()
        groups = data.subject_id.to_numpy()
        pred = np.full(len(data), np.nan)
        splitter = GroupKFold(n_splits=folds)
        for fold, (tr, te) in enumerate(splitter.split(x, y, groups)):
            model = hgb_clf(max_iter, SEED + fold)
            model.fit(x.iloc[tr], y[tr])
            pred[te] = model.predict_proba(x.iloc[te])[:, 1]
        pred = np.clip(pred, 0.02, 0.98)
        marginal = float(y.mean())
        raw_weight = marginal / pred
        assessed_weights = raw_weight[y == 1]
        lo, hi = np.quantile(assessed_weights, [0.01, 0.99])
        weight = np.clip(raw_weight, lo, hi)
        audit = pd.DataFrame([{
            "n_landmark": len(data), "assessed": int(y.sum()),
            "events": int(pd.to_numeric(data.strict_cam_positive_48_72, errors="coerce").sum()),
            "assessment_rate": marginal,
            "assessment_AUROC": roc_auc_score(y, pred),
            "assessment_AUPRC": average_precision_score(y, pred),
            "assessment_Brier": brier_score_loss(y, pred),
            "weight_p01": float(lo), "weight_p99": float(hi),
            "weight_mean_assessed": float(weight[y == 1].mean()),
            "weight_max_assessed": float(weight[y == 1].max()),
            "effective_sample_size": float(weight[y == 1].sum() ** 2 / np.square(weight[y == 1]).sum()),
        }])
        return weight, audit


    def pathway_gcomp(
        data: pd.DataFrame,
        low_energy: float,
        high_energy: float,
        folds: int,
        max_iter: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        base_x = feature_frame(data, include_mediator=False)
        outcome_x = feature_frame(data, include_mediator=True)
        mediator = np.clip(numeric(data.primary_renal_transition_burden).to_numpy(), 1e-5, 1 - 1e-5)
        mediator_logit = logit(mediator)
        assessed = data.strict_cam_assessed_48_72.astype(bool).to_numpy()
        y = numeric(data.strict_cam_positive_48_72).fillna(0).astype(int).to_numpy()
        groups = data.subject_id.to_numpy()
        weights = data.assessment_ipcw.to_numpy(float)
        regimes = {
            "stable_energy_no_steroid": (low_energy, 0.0),
            "unstable_energy_no_steroid": (high_energy, 0.0),
            "stable_energy_prescribed_steroid": (low_energy, 1.0),
            "unstable_energy_prescribed_steroid": (high_energy, 1.0),
        }
        keep = ["stay_id", "subject_id", "analysis_arm"]
        if "diabetes" in data:
            keep.append("diabetes")
        individual = data[keep].copy()
        for r in regimes:
            individual[f"m_{r}"] = np.nan
            individual[f"y_{r}"] = np.nan
        individual["y_reference_m_reference"] = np.nan
        individual["y_adverse_m_reference"] = np.nan
        individual["y_adverse_m_adverse"] = np.nan
        observed_pred = np.full(len(data), np.nan)

        splitter = GroupKFold(n_splits=folds)
        for fold, (tr, te) in enumerate(splitter.split(base_x, y, groups)):
            m_model = hgb_reg(max_iter, SEED + 100 + fold)
            m_model.fit(base_x.iloc[tr], mediator_logit[tr])
            m_by_regime: dict[str, np.ndarray] = {}
            for name, (energy, steroid) in regimes.items():
                xx = base_x.iloc[te].copy()
                xx["energy_instability_score"] = energy
                xx["prescribed_steroid_any_6_24"] = steroid
                m_by_regime[name] = expit(m_model.predict(xx))

            tr_assessed = tr[assessed[tr]]
            y_model = hgb_clf(max_iter, SEED + 200 + fold)
            y_model.fit(outcome_x.iloc[tr_assessed], y[tr_assessed], sample_weight=weights[tr_assessed])
            if assessed[te].any():
                observed_pred[te[assessed[te]]] = y_model.predict_proba(outcome_x.iloc[te[assessed[te]]])[:, 1]
            for name, (energy, steroid) in regimes.items():
                xx = outcome_x.iloc[te].copy()
                xx["energy_instability_score"] = energy
                xx["prescribed_steroid_any_6_24"] = steroid
                xx["primary_renal_transition_burden"] = m_by_regime[name]
                individual.loc[individual.index[te], f"m_{name}"] = m_by_regime[name]
                individual.loc[individual.index[te], f"y_{name}"] = y_model.predict_proba(xx)[:, 1]

            ref = "stable_energy_no_steroid"
            adv = "unstable_energy_prescribed_steroid"
            x00 = outcome_x.iloc[te].copy()
            x00["energy_instability_score"] = regimes[ref][0]
            x00["prescribed_steroid_any_6_24"] = regimes[ref][1]
            x00["primary_renal_transition_burden"] = m_by_regime[ref]
            x10 = outcome_x.iloc[te].copy()
            x10["energy_instability_score"] = regimes[adv][0]
            x10["prescribed_steroid_any_6_24"] = regimes[adv][1]
            x10["primary_renal_transition_burden"] = m_by_regime[ref]
            x11 = outcome_x.iloc[te].copy()
            x11["energy_instability_score"] = regimes[adv][0]
            x11["prescribed_steroid_any_6_24"] = regimes[adv][1]
            x11["primary_renal_transition_burden"] = m_by_regime[adv]
            individual.loc[individual.index[te], "y_reference_m_reference"] = y_model.predict_proba(x00)[:, 1]
            individual.loc[individual.index[te], "y_adverse_m_reference"] = y_model.predict_proba(x10)[:, 1]
            individual.loc[individual.index[te], "y_adverse_m_adverse"] = y_model.predict_proba(x11)[:, 1]

        ok = assessed & np.isfinite(observed_pred)
        performance = pd.DataFrame([{
            "model": "strict_cam_outcome_nuisance",
            "n_assessed": int(ok.sum()), "events": int(y[ok].sum()),
            "AUROC": roc_auc_score(y[ok], observed_pred[ok]),
            "AUPRC": average_precision_score(y[ok], observed_pred[ok]),
            "Brier": brier_score_loss(y[ok], observed_pred[ok]),
        }])
        return individual, performance


    def glycaemic_domain_gcomp(
        data: pd.DataFrame,
        folds: int,
        max_iter: int,
    ) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
        """Secondary supported contrast for dextrose--insulin management patterns."""
        base_x = feature_frame(data, include_mediator=False)
        outcome_x = feature_frame(data, include_mediator=True)
        mediator = np.clip(numeric(data.primary_renal_transition_burden).to_numpy(), 1e-5, 1 - 1e-5)
        mediator_logit = logit(mediator)
        assessed = data.strict_cam_assessed_48_72.astype(bool).to_numpy()
        y = numeric(data.strict_cam_positive_48_72).fillna(0).astype(int).to_numpy()
        groups = data.subject_id.to_numpy()
        weights = data.assessment_ipcw.to_numpy(float)
        dev = data.analysis_arm.eq("development")
        dextrose = numeric(data.dextrose_g_per_kg_6_24).clip(lower=0)
        insulin = numeric(data.insulin_units_per_kg_6_24).clip(lower=0)
        d_low, d_high = [float(np.log1p(v)) for v in dextrose[dev].quantile([0.25, 0.75])]
        i_low, i_high = [float(np.log1p(v)) for v in insulin[dev].quantile([0.25, 0.75])]
        regimes = {
            "low_dextrose_low_insulin": (d_low, i_low),
            "low_dextrose_high_insulin": (d_low, i_high),
            "high_dextrose_low_insulin": (d_high, i_low),
            "high_dextrose_high_insulin": (d_high, i_high),
        }
        keep = ["stay_id", "subject_id", "analysis_arm"]
        if "diabetes" in data:
            keep.append("diabetes")
        individual = data[keep].copy()
        for r in regimes:
            individual[f"m_{r}"] = np.nan
            individual[f"y_{r}"] = np.nan
        individual["y_reference_m_reference"] = np.nan
        individual["y_adverse_m_reference"] = np.nan
        individual["y_adverse_m_adverse"] = np.nan
        splitter = GroupKFold(n_splits=folds)
        for fold, (tr, te) in enumerate(splitter.split(base_x, y, groups)):
            mm = hgb_reg(max_iter, SEED + 1000 + fold)
            mm.fit(base_x.iloc[tr], mediator_logit[tr])
            m_by: dict[str, np.ndarray] = {}
            for r, (dval, ival) in regimes.items():
                xx = base_x.iloc[te].copy()
                xx["dextrose_g_per_kg_6_24"] = dval
                xx["insulin_units_per_kg_6_24"] = ival
                m_by[r] = expit(mm.predict(xx))
            yy = hgb_clf(max_iter, SEED + 1100 + fold)
            train = tr[assessed[tr]]
            yy.fit(outcome_x.iloc[train], y[train], sample_weight=weights[train])
            for r, (dval, ival) in regimes.items():
                xx = outcome_x.iloc[te].copy()
                xx["dextrose_g_per_kg_6_24"] = dval
                xx["insulin_units_per_kg_6_24"] = ival
                xx["primary_renal_transition_burden"] = m_by[r]
                individual.loc[individual.index[te], f"m_{r}"] = m_by[r]
                individual.loc[individual.index[te], f"y_{r}"] = yy.predict_proba(xx)[:, 1]
            ref = "low_dextrose_high_insulin"
            adv = "high_dextrose_low_insulin"
            x00 = outcome_x.iloc[te].copy()
            x00["dextrose_g_per_kg_6_24"] = regimes[ref][0]
            x00["insulin_units_per_kg_6_24"] = regimes[ref][1]
            x00["primary_renal_transition_burden"] = m_by[ref]
            x10 = outcome_x.iloc[te].copy()
            x10["dextrose_g_per_kg_6_24"] = regimes[adv][0]
            x10["insulin_units_per_kg_6_24"] = regimes[adv][1]
            x10["primary_renal_transition_burden"] = m_by[ref]
            x11 = outcome_x.iloc[te].copy()
            x11["dextrose_g_per_kg_6_24"] = regimes[adv][0]
            x11["insulin_units_per_kg_6_24"] = regimes[adv][1]
            x11["primary_renal_transition_burden"] = m_by[adv]
            individual.loc[individual.index[te], "y_reference_m_reference"] = yy.predict_proba(x00)[:, 1]
            individual.loc[individual.index[te], "y_adverse_m_reference"] = yy.predict_proba(x10)[:, 1]
            individual.loc[individual.index[te], "y_adverse_m_adverse"] = yy.predict_proba(x11)[:, 1]
        contract = {
            "dextrose_q25_log1p_g_per_kg": d_low,
            "dextrose_q75_log1p_g_per_kg": d_high,
            "insulin_q25_log1p_units_per_kg": i_low,
            "insulin_q75_log1p_units_per_kg": i_high,
        }
        return individual, list(regimes), contract


    def state_outcome_curve(
        data: pd.DataFrame,
        folds: int,
        max_iter: int,
        n_boot: int,
    ) -> pd.DataFrame:
        x = feature_frame(data, include_mediator=True)
        y = numeric(data.strict_cam_positive_48_72).fillna(0).astype(int).to_numpy()
        assessed = data.strict_cam_assessed_48_72.astype(bool).to_numpy()
        groups = data.subject_id.to_numpy()
        weights = data.assessment_ipcw.to_numpy(float)
        grid = np.linspace(0.0, 1.0, 21)
        pred = {g: np.full(len(data), np.nan) for g in grid}
        splitter = GroupKFold(n_splits=folds)
        for fold, (tr, te) in enumerate(splitter.split(x, y, groups)):
            train = tr[assessed[tr]]
            model = hgb_clf(max_iter, SEED + 300 + fold)
            model.fit(x.iloc[train], y[train], sample_weight=weights[train])
            for g in grid:
                xx = x.iloc[te].copy()
                xx["primary_renal_transition_burden"] = g
                pred[g][te] = model.predict_proba(xx)[:, 1]
        rows = []
        rng = np.random.default_rng(SEED + 301)
        boot_idx = list(cluster_bootstrap_indices(data.subject_id.to_numpy(), n_boot, rng))
        for g in grid:
            point = float(np.mean(pred[g]))
            b = np.asarray([float(np.mean(pred[g][idx])) for idx in boot_idx])
            lo, hi = interval(b)
            rows.append({"renal_transition_burden": g, "standardized_cam_risk": point,
                         "ci_lower": lo, "ci_upper": hi, "n_target": len(data),
                         "bootstrap_replicates": n_boot})
        # Marginal 0.10 contrast over support where both values exist.
        diffs = []
        grid_keys = [float(g) for g in grid]
        for i in range(len(grid_keys) - 2):
            diffs.append(pred[grid_keys[i + 2]] - pred[grid_keys[i]])
        mean_individual_diff = np.mean(np.vstack(diffs), axis=0)
        b = np.asarray([float(mean_individual_diff[idx].mean()) for idx in boot_idx])
        lo, hi = interval(b)
        rows.append({
            "renal_transition_burden": np.nan,
            "standardized_cam_risk": float(mean_individual_diff.mean()),
            "ci_lower": lo, "ci_upper": hi, "n_target": len(data),
            "bootstrap_replicates": n_boot,
            "estimand": "average risk difference per 0.10 higher renal transition burden",
        })
        return pd.DataFrame(rows)


    def falsification_contrast(
        data: pd.DataFrame,
        exposure: str,
        folds: int,
        max_iter: int,
        n_boot: int,
    ) -> pd.DataFrame:
        base = feature_frame(data, include_mediator=False).drop(columns=[exposure], errors="ignore")
        base[exposure] = numeric(data[exposure]).fillna(0).to_numpy()
        outcome = feature_frame(data, include_mediator=True).drop(columns=[exposure], errors="ignore")
        outcome[exposure] = numeric(data[exposure]).fillna(0).to_numpy()
        m = numeric(data.primary_renal_transition_burden).to_numpy(float)
        y = numeric(data.strict_cam_positive_48_72).fillna(0).astype(int).to_numpy()
        assessed = data.strict_cam_assessed_48_72.astype(bool).to_numpy()
        groups = data.subject_id.to_numpy()
        weights = data.assessment_ipcw.to_numpy(float)
        m0 = np.full(len(data), np.nan); m1 = np.full(len(data), np.nan)
        y0 = np.full(len(data), np.nan); y1 = np.full(len(data), np.nan)
        splitter = GroupKFold(n_splits=folds)
        for fold, (tr, te) in enumerate(splitter.split(base, y, groups)):
            mm = hgb_reg(max_iter, SEED + 400 + fold)
            mm.fit(base.iloc[tr], m[tr])
            bx0 = base.iloc[te].copy(); bx0[exposure] = 0
            bx1 = base.iloc[te].copy(); bx1[exposure] = 1
            m0[te] = np.clip(mm.predict(bx0), 0, 1)
            m1[te] = np.clip(mm.predict(bx1), 0, 1)
            yy = hgb_clf(max_iter, SEED + 500 + fold)
            train = tr[assessed[tr]]
            yy.fit(outcome.iloc[train], y[train], sample_weight=weights[train])
            ox0 = outcome.iloc[te].copy(); ox0[exposure] = 0; ox0["primary_renal_transition_burden"] = m0[te]
            ox1 = outcome.iloc[te].copy(); ox1[exposure] = 1; ox1["primary_renal_transition_burden"] = m1[te]
            y0[te] = yy.predict_proba(ox0)[:, 1]
            y1[te] = yy.predict_proba(ox1)[:, 1]
        dm = m1 - m0; dy = y1 - y0
        rng = np.random.default_rng(SEED + 600 + len(exposure))
        bm, by = [], []
        for idx in cluster_bootstrap_indices(data.subject_id.to_numpy(), n_boot, rng):
            bm.append(float(dm[idx].mean())); by.append(float(dy[idx].mean()))
        ml, mu = interval(np.asarray(bm)); yl, yu = interval(np.asarray(by))
        return pd.DataFrame([
            {"exposure": exposure, "estimand": "renal_transition_burden_difference_1_vs_0",
             "estimate": float(dm.mean()), "ci_lower": ml, "ci_upper": mu,
             "n": len(data), "exposed": int(numeric(data[exposure]).fillna(0).gt(0).sum())},
            {"exposure": exposure, "estimand": "strict_cam_risk_difference_1_vs_0",
             "estimate": float(dy.mean()), "ci_lower": yl, "ci_upper": yu,
             "n": len(data), "exposed": int(numeric(data[exposure]).fillna(0).gt(0).sum())},
        ])


    def transition_atlas(
        data: pd.DataFrame,
        low_energy: float,
        high_energy: float,
        folds: int,
        max_iter: int,
        n_boot: int,
    ) -> pd.DataFrame:
        x = feature_frame(data, include_mediator=False)
        current = data.primary_state_id_18_24.astype(int).to_numpy()
        for k in range(6):
            x[f"current_state_{k}"] = (current == k).astype(float)
        groups = data.subject_id.to_numpy()
        regimes = {
            "stable_energy_no_steroid": (low_energy, 0.0),
            "unstable_energy_no_steroid": (high_energy, 0.0),
            "stable_energy_prescribed_steroid": (low_energy, 1.0),
            "unstable_energy_prescribed_steroid": (high_energy, 1.0),
        }
        targets = [numeric(data[f"primary_p_state_{k}_24_30"]).to_numpy(float) for k in range(6)]
        predictions = {(r, k): np.full(len(data), np.nan) for r in regimes for k in range(6)}
        splitter = GroupKFold(n_splits=folds)
        dummy_y = current
        for fold, (tr, te) in enumerate(splitter.split(x, dummy_y, groups)):
            models = []
            for k in range(6):
                model = hgb_reg(max_iter, SEED + 700 + 10 * fold + k)
                model.fit(x.iloc[tr], targets[k][tr])
                models.append(model)
            for r, (energy, steroid) in regimes.items():
                xx = x.iloc[te].copy()
                xx["energy_instability_score"] = energy
                xx["prescribed_steroid_any_6_24"] = steroid
                raw = np.column_stack([np.clip(model.predict(xx), 0, 1) for model in models])
                raw = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-8)
                for k in range(6):
                    predictions[(r, k)][te] = raw[:, k]
        rows = []
        rng = np.random.default_rng(SEED + 701)
        for r in regimes:
            for origin in range(6):
                mask = current == origin
                if mask.sum() < 20:
                    continue
                subjects = data.loc[mask, "subject_id"].to_numpy()
                boot_idx = list(cluster_bootstrap_indices(subjects, n_boot, rng))
                for dest in range(6):
                    vals = predictions[(r, dest)][mask]
                    b = np.asarray([float(vals[idx].mean()) for idx in boot_idx])
                    lo, hi = interval(b)
                    rows.append({
                        "regime": r, "current_state_id": origin,
                        "current_state": STATE_NAMES.get(origin, f"State {origin}"),
                        "next_state_id": dest, "next_state": STATE_NAMES.get(dest, f"State {dest}"),
                        "standardized_probability": float(vals.mean()),
                        "ci_lower": lo, "ci_upper": hi,
                        "n_stays": int(mask.sum()), "subjects": int(np.unique(subjects).size),
                        "bootstrap_replicates": n_boot,
                        "transition_window": "18-24 h to 24-30 h",
                    })
        return pd.DataFrame(rows)


    def ppd_bridge(
        rows: pd.DataFrame,
        validation_npz: Path,
        test_npz: Path,
        n_boot: int,
    ) -> pd.DataFrame:
        baseline_cols = [
            "anchor_age", "admission_sofa", "charlson_index",
            "primary_baseline_p_state_0", "primary_baseline_p_state_1",
            "primary_baseline_p_state_2", "primary_baseline_p_state_3",
            "primary_baseline_p_state_4", "primary_baseline_p_state_5",
        ]
        target = "primary_renal_transition_burden"

        def load(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
            z = np.load(path, allow_pickle=True)
            d = pd.DataFrame({
                "stay_id": z["stay_id"].astype(int),
                "subject_id_npz": z["subject_id"].astype(int),
                "landmark_eligible": z["landmark_eligible"].astype(bool),
                "subject_disjoint": z["subject_disjoint_sensitivity_eligible"].astype(bool),
            })
            return d, z["fused_repr"].astype(float)

        vd, vx = load(validation_npz); td, tx = load(test_npz)
        v = vd.merge(rows[["stay_id", "subject_id", target] + baseline_cols], on="stay_id", how="inner")
        t = td.merge(rows[["stay_id", "subject_id", target] + baseline_cols], on="stay_id", how="inner")
        vi = vd[vd.stay_id.isin(v.stay_id)].index.to_numpy()
        ti = td[td.stay_id.isin(t.stay_id)].index.to_numpy()
        vx = vx[vi]; tx = tx[ti]
        v = v.reset_index(drop=True); t = t.reset_index(drop=True)
        vmask = v.landmark_eligible & v[target].notna(); tmask = t.landmark_eligible & t[target].notna()
        v, vx = v[vmask].reset_index(drop=True), vx[vmask.to_numpy()]
        t, tx = t[tmask].reset_index(drop=True), tx[tmask.to_numpy()]

        def baseline_matrix(d: pd.DataFrame) -> np.ndarray:
            return d[baseline_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

        alphas = np.logspace(-3, 4, 30)
        base_model = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas)),
        ])
        ext_model = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas)),
        ])
        yv = v[target].to_numpy(float); yt = t[target].to_numpy(float)
        bv, bt = baseline_matrix(v), baseline_matrix(t)
        base_model.fit(bv, yv); ext_model.fit(np.c_[bv, vx], yv)
        pb = base_model.predict(bt); pe = ext_model.predict(np.c_[bt, tx])

        def metrics(mask: np.ndarray, arm: str) -> dict:
            yy, bb, ee = yt[mask], pb[mask], pe[mask]
            return {
                "analysis_arm": arm, "n": int(mask.sum()),
                "baseline_R2": r2_score(yy, bb), "extended_R2": r2_score(yy, ee),
                "delta_R2": r2_score(yy, ee) - r2_score(yy, bb),
                "baseline_MAE": mean_absolute_error(yy, bb),
                "extended_MAE": mean_absolute_error(yy, ee),
                "delta_MAE": mean_absolute_error(yy, ee) - mean_absolute_error(yy, bb),
                "target": target, "representation": "h24 physiology-only fused PPD-EHR representation",
            }

        out = [metrics(np.ones(len(t), dtype=bool), "test")]
        dis = t.subject_disjoint.to_numpy(bool)
        if dis.sum() >= 50:
            out.append(metrics(dis, "test_subject_disjoint"))
        result = pd.DataFrame(out)
        # Bootstrap delta R2 in each test arm.
        rng = np.random.default_rng(SEED + 900)
        for i, row in result.iterrows():
            mask = np.ones(len(t), dtype=bool) if row.analysis_arm == "test" else dis
            q = t[mask].reset_index(drop=True); yy, bb, ee = yt[mask], pb[mask], pe[mask]
            deltas = []
            for idx in cluster_bootstrap_indices(q.subject_id.to_numpy(), n_boot, rng):
                if len(np.unique(yy[idx])) < 2:
                    continue
                deltas.append(r2_score(yy[idx], ee[idx]) - r2_score(yy[idx], bb[idx]))
            lo, hi = interval(np.asarray(deltas))
            result.loc[i, "delta_R2_ci_lower"] = lo
            result.loc[i, "delta_R2_ci_upper"] = hi
            result.loc[i, "bootstrap_replicates"] = len(deltas)
        return result


    def main() -> None:
        args = parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for p in [args.analysis_rows, args.dataset_audit, args.validation_h24, args.test_h24]:
            if not p.exists():
                raise FileNotFoundError(p)
        rows = pd.read_csv(args.analysis_rows, low_memory=False)
        audit = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
        low_energy = float(audit["regime_values"]["stable_energy_q25"])
        high_energy = float(audit["regime_values"]["unstable_energy_q75"])
        landmark = rows[rows.primary_pathway_landmark_eligible.eq(True)].copy().reset_index(drop=True)
        landmark["strict_cam_assessed_48_72"] = landmark.strict_cam_positive_48_72.notna()

        weights, weight_audit = fit_assessment_weights(landmark, args.folds, args.max_iter)
        landmark["assessment_ipcw"] = weights
        individual, outcome_performance = pathway_gcomp(
            landmark, low_energy, high_energy, args.folds, args.max_iter
        )
        regime_names = [
            "stable_energy_no_steroid", "unstable_energy_no_steroid",
            "stable_energy_prescribed_steroid", "unstable_energy_prescribed_steroid",
        ]
        regime_tables, path_tables = [], []
        for arm in ["all", "development", "confirmation"]:
            regimes, path, _ = bootstrap_summary(
                individual, regime_names, args.bootstrap, arm
            )
            regime_tables.append(regimes); path_tables.append(path)
        if "diabetes" in individual:
            for diabetes_value, label in [(0, "non_diabetes"), (1, "diabetes")]:
                subgroup = individual[individual.diabetes.eq(diabetes_value)].copy()
                if len(subgroup) < 200:
                    continue
                subgroup["analysis_arm"] = "all"
                regimes, path, _ = bootstrap_summary(
                    subgroup, regime_names, min(args.bootstrap, 1000), "all"
                )
                regimes["analysis_arm"] = label
                path["analysis_arm"] = label
                regime_tables.append(regimes); path_tables.append(path)
        regime_results = pd.concat(regime_tables, ignore_index=True)
        pathway_results = pd.concat(path_tables, ignore_index=True)

        gly_individual, gly_regime_names, gly_contract = glycaemic_domain_gcomp(
            landmark, args.folds, args.max_iter
        )
        gly_regimes, gly_path, _ = bootstrap_summary(
            gly_individual, gly_regime_names, min(args.bootstrap, 1000), "all",
            contrast_label="high_dextrose_low_insulin_vs_low_dextrose_high_insulin",
        )
        if "diabetes" in gly_individual:
            gly_subgroup_tables, gly_subgroup_paths = [gly_regimes], [gly_path]
            for diabetes_value, label in [(0, "non_diabetes"), (1, "diabetes")]:
                subgroup = gly_individual[gly_individual.diabetes.eq(diabetes_value)].copy()
                if len(subgroup) < 200:
                    continue
                subgroup["analysis_arm"] = "all"
                gr, gp, _ = bootstrap_summary(
                    subgroup, gly_regime_names, min(args.bootstrap, 1000), "all",
                    contrast_label="high_dextrose_low_insulin_vs_low_dextrose_high_insulin",
                )
                gr["analysis_arm"] = label; gp["analysis_arm"] = label
                gly_subgroup_tables.append(gr); gly_subgroup_paths.append(gp)
            gly_regimes = pd.concat(gly_subgroup_tables, ignore_index=True)
            gly_path = pd.concat(gly_subgroup_paths, ignore_index=True)

        curve = state_outcome_curve(
            landmark, args.folds, args.max_iter, min(args.bootstrap, 1000)
        )
        falsification = pd.concat([
            falsification_contrast(
                landmark, "ppi_h2_any_6_24", args.folds, args.max_iter,
                min(args.bootstrap, 1000),
            ),
            falsification_contrast(
                landmark, "antiemetic_any_6_24", args.folds, args.max_iter,
                min(args.bootstrap, 1000),
            ),
        ], ignore_index=True)

        atlas_source = rows[
            rows.primary_complete_state_24_48.eq(True)
            & rows.primary_state_id_18_24.notna()
            & rows.primary_state_id_24_30.notna()
        ].copy().reset_index(drop=True)
        atlas = transition_atlas(
            atlas_source, low_energy, high_energy, args.folds,
            args.max_iter, args.atlas_bootstrap,
        )
        bridge = ppd_bridge(
            rows, args.validation_h24, args.test_h24,
            min(args.bootstrap, 1000),
        )

        regime_results.to_csv(args.output_dir / "joint_regime_standardized_effects.csv", index=False)
        individual.to_csv(
            args.output_dir / "standardized_individual_contributions.csv.gz",
            index=False, compression="gzip",
        )
        pathway_results.to_csv(args.output_dir / "longitudinal_interventional_pathway_effects.csv", index=False)
        curve.to_csv(args.output_dir / "continuous_state_outcome_curve.csv", index=False)
        falsification.to_csv(args.output_dir / "falsification_exposure_effects.csv", index=False)
        atlas.to_csv(args.output_dir / "standardized_treatment_transition_atlas.csv", index=False)
        bridge.to_csv(args.output_dir / "ppd_clinical_continuous_bridge.csv", index=False)
        gly_regimes.to_csv(args.output_dir / "glycaemic_domain_standardized_effects.csv", index=False)
        gly_path.to_csv(args.output_dir / "glycaemic_domain_pathway_effects.csv", index=False)
        (args.output_dir / "glycaemic_domain_regime_contract.json").write_text(
            json.dumps(gly_contract, indent=2), encoding="utf-8"
        )
        weight_audit.to_csv(args.output_dir / "cam_observation_weight_audit.csv", index=False)
        outcome_performance.to_csv(args.output_dir / "outcome_nuisance_model_performance.csv", index=False)

        all_path = pathway_results[pathway_results.analysis_arm.eq("all")]
        indirect = all_path[all_path.estimand.eq("indirect_effect")].iloc[0]
        te = all_path[all_path.estimand.eq("total_effect")].iloc[0]
        falsification_similar = bool(
            (falsification.estimand.eq("renal_transition_burden_difference_1_vs_0")
             & falsification.estimate.abs().ge(
                 abs(
                     regime_results[
                         (regime_results.analysis_arm.eq("all"))
                         & (regime_results.estimand.eq("renal_transition_burden"))
                         & (regime_results.regime.eq("unstable_energy_prescribed_steroid"))
                     ].estimate.iloc[0]
                     - regime_results[
                         (regime_results.analysis_arm.eq("all"))
                         & (regime_results.estimand.eq("renal_transition_burden"))
                         & (regime_results.regime.eq("stable_energy_no_steroid"))
                     ].estimate.iloc[0]
                 )
             )).any()
        )
        summary = {
            "target_population": args.target_population,
            "n_landmark": int(len(landmark)),
            "subjects_landmark": int(landmark.subject_id.nunique()),
            "strict_cam_assessed": int(landmark.strict_cam_assessed_48_72.sum()),
            "strict_cam_events": int(landmark.strict_cam_positive_48_72.sum()),
            "regime_reference": "stable energy delivery (development Q25), no prescribed steroid",
            "regime_adverse": "unstable energy delivery (development Q75), prescribed steroid",
            "total_effect_risk_difference": float(te.risk_difference),
            "total_effect_ci": [float(te.ci_lower), float(te.ci_upper)],
            "indirect_effect_risk_difference": float(indirect.risk_difference),
            "indirect_effect_ci": [float(indirect.ci_lower), float(indirect.ci_upper)],
            "indirect_effect_interval_excludes_zero": bool(indirect.ci_lower > 0 or indirect.ci_upper < 0),
            "pathway_confirmed": bool(
                (indirect.ci_lower > 0 or indirect.ci_upper < 0) and not falsification_similar
            ),
            "falsification_effect_as_large_as_primary_transition_contrast": falsification_similar,
            "tmle_status": "not claimed; no validated sequential longitudinal mediation TMLE implementation was used",
            "bootstrap": {
                "replicates": args.bootstrap,
                "type": "subject-cluster bootstrap of cross-fitted standardized individual contributions",
                "boundary": "conditional on fitted nuisance models",
            },
            "claim_boundary": (
                "Observational interventional analogue. Prescribed steroid is not confirmed administration; "
                "the procedure-derived ventilation flag is a limited proxy; fixed-window missing CAM is handled "
                "through an observation model and is not coded as CAM-negative."
            ),
        }
        (args.output_dir / "longitudinal_pathway_analysis_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))

    return SimpleNamespace(**locals())

_pathway = _load_pathway()

_register_legacy("run_longitudinal_state_transition_pathway_20260713", _pathway)


# ==============================================================================
# Landmark and competing-risk sensitivity
# ==============================================================================

def _load_sensitivity():
    """Landmark-selection and 48--72 h competing-process sensitivity analysis."""

    import argparse
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    from run_longitudinal_state_transition_pathway_20260713 import (
        SEED, cluster_bootstrap_indices, feature_frame, interval,
    )


    def parse_args() -> argparse.Namespace:
        package = Path(__file__).resolve().parents[2]
        p = argparse.ArgumentParser()
        p.add_argument(
            "--analysis-rows", type=Path,
            default=package / "outputs" / "downstream" / "longitudinal_pathway"
            / "longitudinal_state_transition_analysis_rows_primary_0_48.csv.gz",
        )
        p.add_argument(
            "--individual-contributions", type=Path,
            default=package / "outputs" / "downstream" / "longitudinal_pathway"
            / "standardized_individual_contributions.csv.gz",
        )
        p.add_argument(
            "--output-dir", type=Path,
            default=package / "outputs" / "downstream" / "longitudinal_pathway",
        )
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--bootstrap", type=int, default=1000)
        return p.parse_args()


    def weighted_mean(x: pd.Series, w: pd.Series) -> float:
        ok = x.notna() & w.notna() & np.isfinite(w) & w.gt(0)
        return float(np.average(x[ok].to_numpy(float), weights=w[ok].to_numpy(float)))


    def main() -> None:
        args = parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for p in [args.analysis_rows, args.individual_contributions]:
            if not p.exists():
                raise FileNotFoundError(p)
        rows = pd.read_csv(args.analysis_rows, low_memory=False)
        selected = rows.primary_pathway_landmark_eligible.eq(True).to_numpy()
        x = feature_frame(rows, include_mediator=False)
        groups = rows.subject_id.to_numpy()
        pred = np.full(len(rows), np.nan)
        splitter = GroupKFold(n_splits=args.folds)
        for fold, (tr, te) in enumerate(splitter.split(x, selected.astype(int), groups)):
            model = HistGradientBoostingClassifier(
                loss="log_loss", learning_rate=0.045, max_iter=160,
                max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1,
                random_state=SEED + 1200 + fold,
            )
            model.fit(x.iloc[tr], selected[tr].astype(int))
            pred[te] = model.predict_proba(x.iloc[te])[:, 1]
        pred = np.clip(pred, 0.01, 0.99)
        marginal = float(selected.mean())
        raw_weight = marginal / pred
        lo, hi = np.quantile(raw_weight[selected], [0.01, 0.99])
        selection_weight = np.clip(raw_weight, lo, hi)
        weight_table = rows.loc[selected, ["stay_id", "subject_id"]].copy()
        weight_table["landmark_selection_weight"] = selection_weight[selected]
        contributions = pd.read_csv(args.individual_contributions)
        q = contributions.merge(weight_table, on=["stay_id", "subject_id"], how="inner", validate="one_to_one")
        point = {
            "total_effect": weighted_mean(q.y_adverse_m_adverse - q.y_reference_m_reference, q.landmark_selection_weight),
            "direct_effect": weighted_mean(q.y_adverse_m_reference - q.y_reference_m_reference, q.landmark_selection_weight),
            "indirect_effect": weighted_mean(q.y_adverse_m_adverse - q.y_adverse_m_reference, q.landmark_selection_weight),
        }
        rng = np.random.default_rng(SEED + 1250)
        boots = {k: [] for k in point}
        for idx in cluster_bootstrap_indices(q.subject_id.to_numpy(), args.bootstrap, rng):
            b = q.iloc[idx]
            boots["total_effect"].append(weighted_mean(
                b.y_adverse_m_adverse - b.y_reference_m_reference, b.landmark_selection_weight
            ))
            boots["direct_effect"].append(weighted_mean(
                b.y_adverse_m_reference - b.y_reference_m_reference, b.landmark_selection_weight
            ))
            boots["indirect_effect"].append(weighted_mean(
                b.y_adverse_m_adverse - b.y_adverse_m_reference, b.landmark_selection_weight
            ))
        effects = []
        for key in ["total_effect", "direct_effect", "indirect_effect"]:
            l, u = interval(np.asarray(boots[key]))
            effects.append({
                "estimand": key, "risk_difference": point[key],
                "ci_lower": l, "ci_upper": u,
                "risk_difference_per_1000": 1000 * point[key],
                "ci_lower_per_1000": 1000 * l, "ci_upper_per_1000": 1000 * u,
                "bootstrap_replicates": args.bootstrap,
                "selection_weight_policy": "48-h landmark stabilized IPCW truncated at selected-arm p01/p99",
                "boundary": "Sensitivity analysis conditional on fitted nuisance models.",
            })
        pd.DataFrame(effects).to_csv(
            args.output_dir / "landmark_selection_weighted_pathway_effects.csv", index=False
        )
        weight_table.to_csv(
            args.output_dir / "landmark_selection_weights.csv.gz", index=False, compression="gzip"
        )

        landmark = rows[selected].copy()
        assessed = landmark.strict_cam_positive_48_72.notna()
        positive = landmark.strict_cam_positive_48_72.eq(1)
        death = landmark.death_48_72.fillna(False).astype(bool)
        discharge_col = (
            "icu_discharge_time_48_72" if "icu_discharge_time_48_72" in landmark
            else "icu_discharge_48_72"
        )
        discharge = landmark[discharge_col].fillna(False).astype(bool)
        category = np.full(len(landmark), "unresolved_no_explicit_assessment", dtype=object)
        category[discharge.to_numpy()] = "icu_discharge_without_explicit_CAM"
        category[death.to_numpy()] = "death_48_72_without_observed_CAM"
        category[(assessed & landmark.strict_cam_positive_48_72.eq(0)).to_numpy()] = "assessed_no_CAM"
        category[positive.to_numpy()] = "strict_CAM_positive"
        landmark["fixed_window_category"] = category
        competing = landmark.groupby("fixed_window_category", as_index=False).agg(
            stays=("stay_id", "size"), subjects=("subject_id", "nunique"),
            diabetes=("diabetes", "sum"),
        )
        competing["proportion"] = competing.stays / len(landmark)
        competing.to_csv(args.output_dir / "fixed_window_competing_process_counts.csv", index=False)

        audit = {
            "source_stays": int(len(rows)), "source_subjects": int(rows.subject_id.nunique()),
            "landmark_stays": int(selected.sum()), "landmark_subjects": int(rows.loc[selected, "subject_id"].nunique()),
            "landmark_rate": marginal,
            "selection_AUROC": roc_auc_score(selected.astype(int), pred),
            "selection_AUPRC": average_precision_score(selected.astype(int), pred),
            "selection_Brier": brier_score_loss(selected.astype(int), pred),
            "weight_p01": float(lo), "weight_p99": float(hi),
            "selection_weight_ESS": float(selection_weight[selected].sum() ** 2 / np.square(selection_weight[selected]).sum()),
            "competing_categories": competing.set_index("fixed_window_category").stays.to_dict(),
            "boundary": (
                "The fixed 48-72 h analysis distinguishes assessed CAM, death, discharge and unresolved "
                "observation. It is not a Fine-Gray time-to-event analysis because exact CAM timing is not "
                "available for every unassessed stay."
            ),
        }
        (args.output_dir / "landmark_selection_competing_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        print(json.dumps(audit, indent=2))

    return SimpleNamespace(**locals())

_sensitivity = _load_sensitivity()


STAGES = {
    "full-icu-dataset": _full_icu.main,
    "pathway": _pathway.main,
    "sensitivity": _sensitivity.main,
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Longitudinal pathway and sensitivity analyses')
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
