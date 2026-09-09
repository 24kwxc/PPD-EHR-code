"""Metabolic-buffering-failure and robustness analyses.

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
# Core metabolic-buffering-failure analysis
# ==============================================================================

def _load_core():
    """Run metabolic buffering failure (MBF) analyses.

    This script implements the frozen hypothesis:

        diabetes susceptibility
            -> impaired metabolic buffering capacity (MBF index)
            -> renal-metabolic vulnerability trajectory
            -> acute brain dysfunction.

    The MBF index is a cross-fitted residual from an early input-response model.
    It is not interpreted as a treatment error.  Medication and nutrition variables
    are treated as metabolic challenge/context, not as causal treatment mechanisms.
    """

    import argparse
    import json
    import math
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from scipy.stats import chi2
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler


    SEED = 20260713
    RNG = np.random.default_rng(SEED)


    def parse_args() -> argparse.Namespace:
        package = Path(__file__).resolve().parents[2]
        data_root = package / "data"
        p = argparse.ArgumentParser()
        p.add_argument(
            "--full-icu-rows",
            type=Path,
            default=package / "outputs" / "downstream" / "longitudinal_pathway_full_icu"
            / "longitudinal_state_transition_analysis_rows_full_icu.csv.gz",
        )
        p.add_argument(
            "--diabetes-rows",
            type=Path,
            default=package / "outputs" / "downstream" / "longitudinal_pathway"
            / "longitudinal_state_transition_analysis_rows_primary_0_48.csv.gz",
        )
        p.add_argument(
            "--archived-diabetes-rows",
            type=Path,
            default=package / "outputs" / "downstream" / "longitudinal_pathway"
            / "longitudinal_state_transition_analysis_rows_archived_state_sensitivity.csv.gz",
        )
        p.add_argument(
            "--clinical-windows",
            type=Path,
            default=data_root / "derived" / "clinical_order_silent_deployment_windows.csv",
        )
        p.add_argument(
            "--intervention-windows",
            type=Path,
            default=data_root / "derived" / "latent_trajectory" / "mimic_intervention_features_6h.csv.gz",
        )
        p.add_argument(
            "--output-dir",
            type=Path,
            default=package / "outputs" / "downstream" / "metabolic_buffering_failure",
        )
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--bootstrap", type=int, default=500)
        p.add_argument("--max-iter", type=int, default=180)
        p.add_argument("--chunksize", type=int, default=350_000)
        return p.parse_args()


    def numeric(s: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(s):
            return s.astype(float)
        return pd.to_numeric(s, errors="coerce")


    def canonicalize_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize primary/non-primary column names used by previous scripts."""
        out = df.copy()
        rename = {}
        for base in [
            "pathway_landmark_eligible", "complete_state_24_48",
            "renal_transition_burden", "renal_transition_peak",
        ]:
            primary = "primary_" + base
            if primary in out.columns and base not in out.columns:
                rename[primary] = base
        for k in range(6):
            for base in [
                f"baseline_p_state_{k}", f"post_mean_p_state_{k}",
                f"p_state_{k}_18_24", f"p_state_{k}_24_30",
            ]:
                primary = "primary_" + base
                if primary in out.columns and base not in out.columns:
                    rename[primary] = base
        if "primary_state_id_18_24" in out.columns and "state_id_18_24" not in out.columns:
            rename["primary_state_id_18_24"] = "state_id_18_24"
        if "primary_state_id_24_30" in out.columns and "state_id_24_30" not in out.columns:
            rename["primary_state_id_24_30"] = "state_id_24_30"
        out = out.rename(columns=rename)
        return out


    def landmark_subset(df: pd.DataFrame) -> pd.DataFrame:
        q = canonicalize_rows(df)
        if "pathway_landmark_eligible" in q:
            q = q[q["pathway_landmark_eligible"].astype(bool)]
        if "complete_state_24_48" in q:
            q = q[q["complete_state_24_48"].astype(bool)]
        return q.copy()


    def read_landmark_rows(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        return landmark_subset(df)


    def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        v = numeric(values)
        w = numeric(weights).fillna(0)
        ok = v.notna() & w.gt(0)
        if ok.any():
            return float(np.average(v[ok], weights=w[ok]))
        return float(v.mean(skipna=True))


    def aggregate_clinical_windows(path: Path, stay_ids: set[int], chunksize: int) -> pd.DataFrame:
        usecols = [
            "stay_id", "bin", "glucose_mean", "glucose_cv", "tir_fraction",
            "hyper_severe_fraction", "glucose_count", "n_glucose", "sofa",
            "vasopressor_sum", "mv_flag", "rrt_flag", "lactate", "creatinine",
            "bun", "map", "heart_rate", "temperature", "spo2",
        ]
        frames = []
        for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize, low_memory=False):
            q = chunk[chunk["stay_id"].isin(stay_ids) & chunk["bin"].between(0, 3)].copy()
            if not q.empty:
                frames.append(q)
        if not frames:
            raise RuntimeError("No matching clinical windows found for MBF cohort.")
        w = pd.concat(frames, ignore_index=True)
        if "glucose_count" not in w:
            w["glucose_count"] = np.nan
        if "n_glucose" in w:
            w["glucose_count"] = numeric(w["glucose_count"]).fillna(numeric(w["n_glucose"]))
        else:
            w["glucose_count"] = numeric(w["glucose_count"])

        baseline = w[w["bin"].eq(0)].drop_duplicates("stay_id").copy()
        baseline_cols = {
            "glucose_mean": "mbf_baseline_glucose_mean_0_6",
            "glucose_cv": "mbf_baseline_glucose_cv_0_6",
            "tir_fraction": "mbf_baseline_tir_fraction_0_6",
            "hyper_severe_fraction": "mbf_baseline_hyper_severe_fraction_0_6",
            "sofa": "mbf_baseline_sofa_0_6",
            "vasopressor_sum": "mbf_baseline_vasopressor_0_6",
            "mv_flag": "mbf_baseline_mv_0_6",
            "rrt_flag": "mbf_baseline_rrt_0_6",
            "lactate": "mbf_baseline_lactate_0_6",
            "creatinine": "mbf_baseline_creatinine_0_6",
            "bun": "mbf_baseline_bun_0_6",
            "map": "mbf_baseline_map_0_6",
            "heart_rate": "mbf_baseline_heart_rate_0_6",
            "temperature": "mbf_baseline_temperature_0_6",
            "spo2": "mbf_baseline_spo2_0_6",
        }
        baseline = baseline[["stay_id"] + [c for c in baseline_cols if c in baseline]].rename(columns=baseline_cols)

        early = w[w["bin"].between(0, 1)].groupby("stay_id", as_index=False).agg(
            mbf_sofa_max_0_12=("sofa", "max"),
            mbf_vasopressor_max_0_12=("vasopressor_sum", "max"),
            mbf_mv_any_0_12=("mv_flag", "max"),
            mbf_rrt_any_0_12=("rrt_flag", "max"),
            mbf_lactate_max_0_12=("lactate", "max"),
            mbf_creatinine_max_0_12=("creatinine", "max"),
            mbf_bun_max_0_12=("bun", "max"),
        )

        resp_src = w[w["bin"].between(2, 3)].copy()
        resp_rows = []
        for stay_id, g in resp_src.groupby("stay_id", sort=False):
            resp_rows.append({
                "stay_id": stay_id,
                "mbf_response_glucose_mean_12_24": weighted_mean(g["glucose_mean"], g["glucose_count"]),
                "mbf_response_glucose_cv_12_24": float(numeric(g["glucose_cv"]).mean(skipna=True)),
                "mbf_response_tir_fraction_12_24": float(numeric(g["tir_fraction"]).mean(skipna=True)),
                "mbf_response_hyper_severe_fraction_12_24": float(numeric(g["hyper_severe_fraction"]).mean(skipna=True)),
                "mbf_response_glucose_count_12_24": float(numeric(g["glucose_count"]).fillna(0).sum()),
            })
        response = pd.DataFrame(resp_rows)
        return baseline.merge(early, on="stay_id", how="outer").merge(response, on="stay_id", how="outer")


    def aggregate_intervention_windows(path: Path, stay_ids: set[int], chunksize: int) -> pd.DataFrame:
        usecols = [
            "stay_id", "bin", "enteral_kcal_sum", "propofol_kcal",
            "treatment_aware_kcal", "insulin_units", "steroid_flag",
            "dextrose_amount_sum", "parenteral_nutrition_amount_sum",
            "sofa", "vasopressor_sum", "rrt_flag", "lactate", "creatinine",
            "weight_kg",
        ]
        frames = []
        for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize, low_memory=False):
            q = chunk[chunk["stay_id"].isin(stay_ids) & chunk["bin"].between(0, 1)].copy()
            if not q.empty:
                frames.append(q)
        if not frames:
            raise RuntimeError("No matching intervention windows found for MBF cohort.")
        w = pd.concat(frames, ignore_index=True)
        agg = w.groupby("stay_id", as_index=False).agg(
            mbf_enteral_kcal_0_12=("enteral_kcal_sum", "sum"),
            mbf_propofol_kcal_0_12=("propofol_kcal", "sum"),
            mbf_treatment_aware_kcal_0_12=("treatment_aware_kcal", "sum"),
            mbf_insulin_units_0_12=("insulin_units", "sum"),
            mbf_steroid_any_0_12=("steroid_flag", "max"),
            mbf_dextrose_g_0_12=("dextrose_amount_sum", "sum"),
            mbf_parenteral_nutrition_amount_0_12=("parenteral_nutrition_amount_sum", "sum"),
            mbf_intervention_sofa_max_0_12=("sofa", "max"),
            mbf_intervention_vasopressor_max_0_12=("vasopressor_sum", "max"),
            mbf_intervention_rrt_any_0_12=("rrt_flag", "max"),
            mbf_intervention_lactate_max_0_12=("lactate", "max"),
            mbf_intervention_creatinine_max_0_12=("creatinine", "max"),
            mbf_weight_kg_window=("weight_kg", "median"),
        )
        return agg


    def add_mbf_windows(rows: pd.DataFrame, clinical_windows: Path, intervention_windows: Path, chunksize: int) -> pd.DataFrame:
        stay_ids = set(pd.to_numeric(rows["stay_id"], errors="coerce").dropna().astype(int))
        clinical = aggregate_clinical_windows(clinical_windows, stay_ids, chunksize)
        intervention = aggregate_intervention_windows(intervention_windows, stay_ids, chunksize)
        out = rows.merge(clinical, on="stay_id", how="left").merge(intervention, on="stay_id", how="left")
        weight = numeric(out.get("analysis_weight_kg", out.get("weight_kg", out.get("mbf_weight_kg_window"))))
        weight = weight.fillna(numeric(out.get("weight_kg", out.get("mbf_weight_kg_window"))))
        weight = weight.fillna(numeric(out.get("mbf_weight_kg_window", pd.Series(np.nan, index=out.index))))
        weight = weight.clip(lower=30, upper=250)
        out["mbf_weight_kg"] = weight
        for c in ["mbf_dextrose_g_0_12", "mbf_enteral_kcal_0_12", "mbf_insulin_units_0_12",
                  "mbf_propofol_kcal_0_12", "mbf_treatment_aware_kcal_0_12"]:
            out[c] = numeric(out[c]).fillna(0)
            out[c.replace("_0_12", "_per_kg_0_12")] = out[c] / out["mbf_weight_kg"]
        out["mbf_response_tir_loss_12_24"] = 1 - numeric(out["mbf_response_tir_fraction_12_24"])
        out["mbf_observed_glucose_delta_12_24"] = (
            numeric(out["mbf_response_glucose_mean_12_24"])
            - numeric(out["mbf_baseline_glucose_mean_0_6"])
        )
        return out


    def zscore(s: pd.Series) -> pd.Series:
        x = numeric(s)
        sd = float(x.std(skipna=True))
        if not np.isfinite(sd) or sd <= 1e-12:
            return x * np.nan
        return (x - float(x.mean(skipna=True))) / sd


    def add_challenge_index(data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        components = {
            "dextrose_load": np.log1p(numeric(out["mbf_dextrose_g_per_kg_0_12"]).clip(lower=0)),
            "enteral_energy": numeric(out["mbf_enteral_kcal_per_kg_0_12"]).clip(lower=0),
            "propofol_energy": numeric(out["mbf_propofol_kcal_per_kg_0_12"]).clip(lower=0),
        }
        z = []
        for name, values in components.items():
            col = f"mbf_challenge_component_{name}_z"
            out[col] = zscore(values)
            z.append(out[col])
        out["mbf_metabolic_challenge_index"] = pd.concat(z, axis=1).mean(axis=1)
        return out


    def hgb_reg(max_iter: int, seed: int) -> Pipeline:
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("reg", HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.045, max_iter=max_iter,
                max_leaf_nodes=15, min_samples_leaf=35, l2_regularization=1.0,
                random_state=seed,
            )),
        ])


    def hgb_clf(max_iter: int, seed: int) -> Pipeline:
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                loss="log_loss", learning_rate=0.045, max_iter=max_iter,
                max_leaf_nodes=15, min_samples_leaf=35, l2_regularization=1.0,
                random_state=seed,
            )),
        ])


    def mbf_definition_features(data: pd.DataFrame) -> list[str]:
        candidates = [
            "anchor_age", "charlson_index", "admission_sofa", "weight_kg",
            "mbf_baseline_glucose_mean_0_6", "mbf_baseline_glucose_cv_0_6",
            "mbf_baseline_hyper_severe_fraction_0_6", "mbf_baseline_tir_fraction_0_6",
            "mbf_baseline_sofa_0_6", "mbf_baseline_vasopressor_0_6",
            "mbf_baseline_mv_0_6", "mbf_baseline_rrt_0_6", "mbf_baseline_lactate_0_6",
            "mbf_baseline_creatinine_0_6", "mbf_baseline_bun_0_6",
            "mbf_sofa_max_0_12", "mbf_vasopressor_max_0_12", "mbf_mv_any_0_12",
            "mbf_rrt_any_0_12", "mbf_lactate_max_0_12", "mbf_creatinine_max_0_12",
            "mbf_bun_max_0_12", "mbf_dextrose_g_per_kg_0_12",
            "mbf_enteral_kcal_per_kg_0_12", "mbf_propofol_kcal_per_kg_0_12",
            "mbf_treatment_aware_kcal_per_kg_0_12", "mbf_insulin_units_per_kg_0_12",
            "mbf_steroid_any_0_12", "mbf_parenteral_nutrition_amount_0_12",
        ] + [f"baseline_p_state_{i}" for i in range(6)]
        return [c for c in candidates if c in data.columns]


    def base_renal_features(data: pd.DataFrame) -> list[str]:
        candidates = [
            "anchor_age", "charlson_index", "admission_sofa", "weight_kg",
            "mbf_baseline_glucose_mean_0_6", "mbf_baseline_glucose_cv_0_6",
            "mbf_baseline_sofa_0_6", "mbf_baseline_vasopressor_0_6",
            "mbf_baseline_mv_0_6", "mbf_baseline_rrt_0_6", "mbf_baseline_lactate_0_6",
            "mbf_baseline_creatinine_0_6", "mbf_baseline_bun_0_6",
            "mbf_sofa_max_0_12", "mbf_vasopressor_max_0_12",
            "mbf_mv_any_0_12", "mbf_rrt_any_0_12", "mbf_lactate_max_0_12",
            "mbf_creatinine_max_0_12", "mbf_bun_max_0_12",
        ] + [f"baseline_p_state_{i}" for i in range(6)]
        return [c for c in candidates if c in data.columns]


    def traditional_glucose_features(data: pd.DataFrame) -> list[str]:
        candidates = [
            "mbf_response_glucose_mean_12_24",
            "mbf_response_glucose_cv_12_24",
            "mbf_response_hyper_severe_fraction_12_24",
            "mbf_response_tir_loss_12_24",
        ]
        return [c for c in candidates if c in data.columns]


    def crossfit_regression(
        data: pd.DataFrame,
        features: list[str],
        target: str,
        groups: pd.Series,
        folds: int,
        max_iter: int,
        seed_offset: int,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        ok = data[target].notna()
        pred = np.full(len(data), np.nan)
        x = data[features].apply(pd.to_numeric, errors="coerce")
        y = numeric(data[target]).to_numpy(float)
        splitter = GroupKFold(n_splits=folds)
        for fold, (tr0, te0) in enumerate(splitter.split(x.loc[ok], y[ok], groups.loc[ok])):
            idx_ok = np.where(ok.to_numpy())[0]
            tr = idx_ok[tr0]; te = idx_ok[te0]
            model = hgb_reg(max_iter, SEED + seed_offset + fold)
            model.fit(x.iloc[tr], y[tr])
            pred[te] = model.predict(x.iloc[te])
        q = ok & np.isfinite(pred)
        metrics = regression_metrics(y[q], pred[q], "crossfit", len(features))
        return pred, metrics


    def regression_metrics(y: np.ndarray, pred: np.ndarray, model: str, n_features: int) -> pd.DataFrame:
        rmse = math.sqrt(mean_squared_error(y, pred))
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        slope, intercept = calibration_slope_intercept(y, pred)
        return pd.DataFrame([{
            "model": model, "n": int(len(y)), "n_features": int(n_features),
            "R2": float(r2), "RMSE": float(rmse), "MAE": float(mae),
            "calibration_slope": float(slope), "calibration_intercept": float(intercept),
        }])


    def calibration_slope_intercept(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
        if len(y) < 3 or np.nanstd(pred) <= 1e-12:
            return np.nan, np.nan
        lr = LinearRegression().fit(pred.reshape(-1, 1), y)
        return float(lr.coef_[0]), float(lr.intercept_)


    def cluster_bootstrap_indices(subjects: np.ndarray, n_boot: int, rng: np.random.Generator):
        unique = np.unique(subjects)
        positions = {sid: np.where(subjects == sid)[0] for sid in unique}
        for _ in range(n_boot):
            draw = rng.choice(unique, size=len(unique), replace=True)
            yield np.concatenate([positions[sid] for sid in draw])


    def interval(values: np.ndarray) -> tuple[float, float]:
        return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


    def fit_mbf_index(data: pd.DataFrame, folds: int, max_iter: int, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        out = data.copy()
        features = mbf_definition_features(out)
        target = "mbf_observed_glucose_delta_12_24"
        pred, metrics = crossfit_regression(out, features, target, out["subject_id"], folds, max_iter, 10)
        out["mbf_expected_glucose_delta_12_24"] = pred
        out["mbf_raw_residual"] = numeric(out[target]) - out["mbf_expected_glucose_delta_12_24"]
        mu = float(out["mbf_raw_residual"].mean(skipna=True))
        sd = float(out["mbf_raw_residual"].std(skipna=True))
        out["mbf_index"] = (out["mbf_raw_residual"] - mu) / sd
        out["mbf_index_available"] = out["mbf_index"].notna()
        metrics["target"] = target
        metrics["definition"] = (
            "Cross-fitted residual: observed 12-24 h glucose change from 0-6 h baseline "
            "minus expected glucose change from 0-6 h reserve and 0-12 h metabolic challenge."
        )
        metrics["features"] = ", ".join(features)
        importance = permutation_importance_table(out, features, target, max_iter)
        audit = pd.DataFrame([{
            "n_landmark": int(len(out)),
            "subjects": int(out.subject_id.nunique()),
            "mbf_available": int(out["mbf_index_available"].sum()),
            "response_nonmissing": int(out[target].notna().sum()),
            "response_delta_mean": float(numeric(out[target]).mean(skipna=True)),
            "response_glucose_mean": float(numeric(out["mbf_response_glucose_mean_12_24"]).mean(skipna=True)),
            "residual_mean": float(out["mbf_raw_residual"].mean(skipna=True)),
            "residual_sd": float(out["mbf_raw_residual"].std(skipna=True)),
            "bootstrap_replicates_for_downstream": int(n_boot),
        }])
        return out, pd.concat([audit, metrics], axis=1), importance


    def permutation_importance_table(data: pd.DataFrame, features: list[str], target: str, max_iter: int) -> pd.DataFrame:
        q = data[data[target].notna()].copy()
        if len(q) > 6000:
            q = q.sample(n=6000, random_state=SEED)
        x = q[features].apply(pd.to_numeric, errors="coerce")
        y = numeric(q[target]).to_numpy(float)
        model = hgb_reg(max_iter, SEED + 77)
        model.fit(x, y)
        pi = permutation_importance(
            model, x, y, n_repeats=5, random_state=SEED + 78,
            scoring="r2", n_jobs=1,
        )
        result = pd.DataFrame({
            "feature": features,
            "permutation_delta_R2_mean": pi.importances_mean,
            "permutation_delta_R2_sd": pi.importances_std,
            "n_importance_sample": len(q),
        }).sort_values("permutation_delta_R2_mean", ascending=False)
        result["feature_group"] = result["feature"].map(feature_group)
        return result


    def feature_group(feature: str) -> str:
        if "dextrose" in feature:
            return "glucose load"
        if "enteral" in feature or "treatment_aware" in feature or "propofol" in feature:
            return "energy input/context"
        if "insulin" in feature:
            return "insulin exposure"
        if "creatinine" in feature or "bun" in feature or "rrt" in feature:
            return "renal reserve"
        if "lactate" in feature or "sofa" in feature or "vasopressor" in feature or "mv_" in feature:
            return "illness severity"
        if "glucose" in feature or "tir" in feature or "hyper" in feature:
            return "baseline glycaemia"
        if "state" in feature:
            return "baseline state"
        return "demographic/comorbidity"


    def crossfit_model_performance(
        data: pd.DataFrame,
        specs: list[tuple[str, list[str]]],
        target: str,
        folds: int,
        max_iter: int,
        seed_offset: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        preds = pd.DataFrame({"stay_id": data["stay_id"].values, "subject_id": data["subject_id"].values})
        tables = []
        for i, (name, features) in enumerate(specs):
            pred, metrics = crossfit_regression(data, features, target, data["subject_id"], folds, max_iter, seed_offset + i * 20)
            preds[name] = pred
            m = metrics.copy()
            m["model"] = name
            m["target"] = target
            m["features"] = ", ".join(features)
            tables.append(m)
        perf = pd.concat(tables, ignore_index=True)
        if len(perf) >= 2:
            base_r2 = dict(zip(perf.model, perf.R2))
            perf["delta_R2_vs_model_A"] = perf.R2 - base_r2.get("Model A: severity/reserve", np.nan)
            perf["delta_R2_vs_model_B"] = perf.R2 - base_r2.get("Model B: + traditional glycaemia", np.nan)
        return perf, preds


    def linear_lr_table(data: pd.DataFrame, specs: list[tuple[str, list[str]]], target: str) -> pd.DataFrame:
        rows = []
        prev = None
        prev_name = None
        for name, cols in specs:
            q = data[[target] + cols].copy()
            y = numeric(q[target]).to_numpy(float)
            x = q[cols].apply(pd.to_numeric, errors="coerce")
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("lr", LinearRegression())])
            ok = np.isfinite(y)
            pipe.fit(x.loc[ok], y[ok])
            pred = pipe.predict(x.loc[ok])
            rss = float(np.square(y[ok] - pred).sum())
            n = int(ok.sum())
            p = len(cols) + 1
            ll = -0.5 * n * (math.log(2 * math.pi) + 1 + math.log(max(rss / n, 1e-12)))
            row = {"model": name, "target": target, "n": n, "df": p, "RSS": rss, "gaussian_loglik": ll}
            if prev is not None:
                lrt = 2 * (ll - prev["gaussian_loglik"])
                df = p - prev["df"]
                row["nested_vs"] = prev_name
                row["LR_statistic"] = lrt
                row["LR_df"] = df
                row["LR_p"] = float(chi2.sf(lrt, max(df, 1)))
            rows.append(row)
            prev = row
            prev_name = name
        return pd.DataFrame(rows)


    def mbf_specificity(data: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        q = data[data["mbf_index"].notna()].copy()
        covars = [
            "diabetes", "mbf_baseline_sofa_0_6", "mbf_baseline_lactate_0_6",
            "mbf_baseline_creatinine_0_6", "mbf_baseline_vasopressor_0_6",
            "mbf_baseline_rrt_0_6", "mbf_baseline_glucose_mean_0_6",
            "anchor_age", "charlson_index", "admission_sofa",
        ]
        covars = [c for c in covars if c in q.columns]
        coef = linear_coef_bootstrap(q, "mbf_index", covars, "diabetes", n_boot, SEED + 3000)
        dist = q.groupby("diabetes", dropna=False).agg(
            n=("stay_id", "size"),
            subjects=("subject_id", "nunique"),
            mbf_mean=("mbf_index", "mean"),
            mbf_sd=("mbf_index", "std"),
            mbf_median=("mbf_index", "median"),
            renal_transition_burden=("renal_transition_burden", "mean"),
        ).reset_index()
        coef["model"] = "MBF index ~ diabetes + severity/reserve covariates"
        coef["interpretation"] = "Positive diabetes coefficient indicates higher MBF at comparable measured severity/reserve."
        return dist, coef


    def linear_coef_bootstrap(
        data: pd.DataFrame,
        target: str,
        covars: list[str],
        coef_name: str,
        n_boot: int,
        seed: int,
    ) -> pd.DataFrame:
        q = data[[target, "subject_id"] + covars].copy()
        q = q[q[target].notna()].reset_index(drop=True)
        point = fit_linear_coef(q, target, covars, coef_name)
        rng = np.random.default_rng(seed)
        vals = []
        for idx in cluster_bootstrap_indices(q["subject_id"].to_numpy(), n_boot, rng):
            b = q.iloc[idx].reset_index(drop=True)
            try:
                vals.append(fit_linear_coef(b, target, covars, coef_name))
            except np.linalg.LinAlgError:
                continue
        vals = np.asarray(vals, float)
        lo, hi = interval(vals)
        return pd.DataFrame([{
            "target": target,
            "coefficient": coef_name,
            "estimate": float(point),
            "ci_lower": lo,
            "ci_upper": hi,
            "bootstrap_replicates": int(len(vals)),
            "n": int(len(q)),
            "subjects": int(q.subject_id.nunique()),
            "covariates": ", ".join(covars),
        }])


    def fit_linear_coef(data: pd.DataFrame, target: str, covars: list[str], coef_name: str) -> float:
        y = numeric(data[target]).to_numpy(float)
        x = data[covars].apply(pd.to_numeric, errors="coerce")
        x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=covars)
        for c in covars:
            if c == coef_name or set(pd.Series(x[c]).dropna().unique()).issubset({0, 1}):
                continue
            sd = float(x[c].std())
            if np.isfinite(sd) and sd > 1e-12:
                x[c] = (x[c] - float(x[c].mean())) / sd
        design = np.c_[np.ones(len(x)), x.to_numpy(float)]
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        pos = covars.index(coef_name) + 1
        return float(beta[pos])


    def mbf_quartile_table(data: pd.DataFrame, label: str, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        q = data[data["mbf_index"].notna()].copy()
        q["mbf_quartile"] = pd.qcut(q["mbf_index"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        summary = q.groupby("mbf_quartile", observed=False).agg(
            n=("stay_id", "size"),
            subjects=("subject_id", "nunique"),
            mbf_mean=("mbf_index", "mean"),
            renal_transition_burden=("renal_transition_burden", "mean"),
            hyperglycaemic_instability_burden=("post_mean_p_state_5", "mean"),
            assessed=("strict_cam_assessed_48_72", "sum"),
            cam_events=("strict_cam_positive_48_72", "sum"),
            cam_rate=("strict_cam_positive_48_72", "mean"),
            response_glucose_mean=("mbf_response_glucose_mean_12_24", "mean"),
            response_glucose_cv=("mbf_response_glucose_cv_12_24", "mean"),
            metabolic_challenge=("mbf_metabolic_challenge_index", "mean"),
            insulin_units_per_kg=("mbf_insulin_units_per_kg_0_12", "mean"),
        ).reset_index()
        summary.insert(0, "cohort", label)
        q["mbf_q4"] = q["mbf_quartile"].eq("Q4").astype(float)
        q["mbf_q1"] = q["mbf_quartile"].eq("Q1").astype(float)
        q14 = q[q["mbf_quartile"].isin(["Q1", "Q4"])].copy()
        covars = ["mbf_q4"] + base_renal_features(q14)
        eff = linear_coef_bootstrap(q14, "renal_transition_burden", covars, "mbf_q4", n_boot, SEED + 3100)
        eff.insert(0, "cohort", label)
        eff["contrast"] = "Q4 vs Q1 MBF index"
        return summary, eff


    def cam_bridge_models(data: pd.DataFrame, label: str, folds: int, max_iter: int) -> pd.DataFrame:
        q = data[data["strict_cam_assessed_48_72"].astype(float).eq(1) & data["mbf_index"].notna()].copy()
        y = numeric(q["strict_cam_positive_48_72"]).fillna(0).astype(int).to_numpy()
        groups = q["subject_id"]
        specs = [
            ("Model A: severity/reserve", base_renal_features(q)),
            ("Model B: + MBF index", base_renal_features(q) + ["mbf_index"]),
            ("Model C: + MBF index + renal trajectory", base_renal_features(q) + ["mbf_index", "renal_transition_burden"]),
        ]
        rows = []
        for si, (name, features) in enumerate(specs):
            x = q[features].apply(pd.to_numeric, errors="coerce")
            pred = np.full(len(q), np.nan)
            splitter = GroupKFold(n_splits=folds)
            for fold, (tr, te) in enumerate(splitter.split(x, y, groups)):
                model = hgb_clf(max_iter, SEED + 4100 + 10 * si + fold)
                model.fit(x.iloc[tr], y[tr])
                pred[te] = model.predict_proba(x.iloc[te])[:, 1]
            rows.append({
                "cohort": label, "model": name, "n": int(len(q)), "events": int(y.sum()),
                "AUROC": float(roc_auc_score(y, pred)),
                "AUPRC": float(average_precision_score(y, pred)),
                "Brier": float(brier_score_loss(y, pred)),
                "features": ", ".join(features),
            })
        result = pd.DataFrame(rows)
        result["delta_AUROC_vs_model_A"] = result.AUROC - result.AUROC.iloc[0]
        result["delta_AUPRC_vs_model_A"] = result.AUPRC - result.AUPRC.iloc[0]
        return result


    def diabetes_attenuation(data: pd.DataFrame, n_boot: int) -> pd.DataFrame:
        q = data[data["mbf_index"].notna()].copy()
        outcomes = [
            ("renal_transition_burden", "renal-metabolic vulnerability trajectory"),
            ("post_mean_p_state_5", "hyperglycaemic-instability burden"),
        ]
        rows = []
        base_covars = [
            "diabetes", "anchor_age", "charlson_index", "admission_sofa",
            "mbf_baseline_glucose_mean_0_6", "mbf_baseline_glucose_cv_0_6",
            "mbf_baseline_creatinine_0_6", "mbf_baseline_lactate_0_6",
            "mbf_baseline_sofa_0_6", "mbf_baseline_vasopressor_0_6",
            "mbf_baseline_rrt_0_6",
        ] + [f"baseline_p_state_{i}" for i in range(6) if f"baseline_p_state_{i}" in q]
        base_covars = [c for c in base_covars if c in q.columns]
        for outcome, label in outcomes:
            if outcome not in q:
                continue
            model1 = linear_coef_bootstrap(q, outcome, base_covars, "diabetes", n_boot, SEED + 5000 + len(rows))
            model2_covars = base_covars + ["mbf_index"]
            model2 = linear_coef_bootstrap(q, outcome, model2_covars, "diabetes", n_boot, SEED + 5100 + len(rows))
            mbf_coef = linear_coef_bootstrap(q, outcome, model2_covars, "mbf_index", n_boot, SEED + 5200 + len(rows))
            b1 = model1.estimate.iloc[0]
            b2 = model2.estimate.iloc[0]
            attenuation = (b1 - b2) / b1 if abs(b1) > 1e-12 else np.nan
            # Bootstrap attenuation by paired approximation from independent draws is avoided;
            # report point attenuation and coefficient intervals separately.
            rows.extend([
                {"outcome": outcome, "outcome_label": label, "model": "Model 1: diabetes + covariates",
                 "diabetes_beta": b1, "ci_lower": model1.ci_lower.iloc[0],
                 "ci_upper": model1.ci_upper.iloc[0], "mbf_beta": np.nan,
                 "attenuation_fraction": np.nan, "n": int(model1.n.iloc[0]),
                 "subjects": int(model1.subjects.iloc[0])},
                {"outcome": outcome, "outcome_label": label, "model": "Model 2: diabetes + MBF index + covariates",
                 "diabetes_beta": b2, "ci_lower": model2.ci_lower.iloc[0],
                 "ci_upper": model2.ci_upper.iloc[0], "mbf_beta": mbf_coef.estimate.iloc[0],
                 "mbf_beta_ci_lower": mbf_coef.ci_lower.iloc[0],
                 "mbf_beta_ci_upper": mbf_coef.ci_upper.iloc[0],
                 "attenuation_fraction": attenuation, "n": int(model2.n.iloc[0]),
                 "subjects": int(model2.subjects.iloc[0])},
            ])
        return pd.DataFrame(rows)


    def falsification_sensitivity(data: pd.DataFrame, n_boot: int) -> pd.DataFrame:
        q = data[data["mbf_index"].notna()].copy()
        exposures = [c for c in ["ppi_h2_any_6_24", "antiemetic_any_6_24", "heparin_any_6_24"] if c in q]
        rows = []
        for exposure in exposures:
            covars = [exposure] + base_renal_features(q) + ["mbf_index"]
            coef = linear_coef_bootstrap(q, "renal_transition_burden", covars, exposure, n_boot, SEED + 6100 + len(rows))
            coef["exposure"] = exposure
            coef["exposed"] = int(numeric(q[exposure]).fillna(0).gt(0).sum())
            rows.append(coef)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


    def archived_state_sensitivity(archived_path: Path, mbf_data: pd.DataFrame, n_boot: int, folds: int, max_iter: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not archived_path.exists():
            return pd.DataFrame(), pd.DataFrame()
        arch = read_landmark_rows(archived_path)
        keep = ["stay_id"] + [c for c in mbf_data.columns if c.startswith("mbf_")]
        arch = arch.merge(mbf_data[[c for c in keep if c in mbf_data]], on="stay_id", how="inner")
        arch = canonicalize_rows(arch)
        arch = add_challenge_index(arch)
        specs = [
            ("Model A: severity/reserve", base_renal_features(arch)),
            ("Model B: + traditional glycaemia", base_renal_features(arch) + traditional_glucose_features(arch)),
            ("Model C: + MBF index", base_renal_features(arch) + traditional_glucose_features(arch) + ["mbf_index"]),
        ]
        perf, _ = crossfit_model_performance(
            arch[arch.mbf_index.notna()].copy(), specs, "renal_transition_burden", folds, max_iter, 7000
        )
        perf.insert(0, "cohort", "Archived-state diabetes sensitivity")
        qt, eff = mbf_quartile_table(arch, "Archived-state diabetes sensitivity", n_boot)
        return perf, pd.concat([qt, eff], ignore_index=True, sort=False)


    def mbf_landscape(data: pd.DataFrame, label: str, max_iter: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        q = data[data["mbf_index"].notna()].copy()
        features = base_renal_features(q) + [
            "mbf_metabolic_challenge_index", "mbf_index",
            "mbf_insulin_units_per_kg_0_12",
        ]
        features = [c for c in features if c in q]
        model = hgb_reg(max_iter, SEED + 8000)
        x = q[features].apply(pd.to_numeric, errors="coerce")
        y = numeric(q["renal_transition_burden"]).to_numpy(float)
        model.fit(x, y)
        c_lo, c_hi = np.nanquantile(q["mbf_metabolic_challenge_index"], [0.02, 0.98])
        m_lo, m_hi = np.nanquantile(q["mbf_index"], [0.02, 0.98])
        challenge_grid = np.linspace(c_lo, c_hi, 31)
        mbf_grid = np.linspace(m_lo, m_hi, 31)
        base = x.median(numeric_only=True).to_frame().T
        rows = []
        for c in challenge_grid:
            for m in mbf_grid:
                xx = pd.concat([base] * 1, ignore_index=True)
                xx["mbf_metabolic_challenge_index"] = c
                xx["mbf_index"] = m
                if "mbf_insulin_units_per_kg_0_12" in xx:
                    xx["mbf_insulin_units_per_kg_0_12"] = float(q["mbf_insulin_units_per_kg_0_12"].median())
                rows.append({
                    "cohort": label,
                    "metabolic_challenge_index": float(c),
                    "mbf_index": float(m),
                    "predicted_renal_transition_burden": float(model.predict(xx[features])[0]),
                })
        surface = pd.DataFrame(rows)
        # Support map for reviewer-visible empirical support.
        q["challenge_bin"] = pd.qcut(q["mbf_metabolic_challenge_index"], 8, labels=False, duplicates="drop")
        q["mbf_bin"] = pd.qcut(q["mbf_index"], 8, labels=False, duplicates="drop")
        support = q.groupby(["challenge_bin", "mbf_bin"], dropna=True).agg(
            n=("stay_id", "size"),
            renal_transition_burden=("renal_transition_burden", "mean"),
            metabolic_challenge_index=("mbf_metabolic_challenge_index", "mean"),
            mbf_index=("mbf_index", "mean"),
        ).reset_index()
        support.insert(0, "cohort", label)
        return surface, support


    def selection_bias_audit(data: pd.DataFrame, label: str) -> pd.DataFrame:
        q = data.copy()
        return pd.DataFrame([{
            "cohort": label,
            "n_landmark": int(len(q)),
            "mbf_available": int(q.mbf_index.notna().sum()),
            "strict_cam_assessed": int(q.strict_cam_assessed_48_72.astype(float).fillna(0).sum()),
            "strict_cam_events": int(numeric(q.strict_cam_positive_48_72).fillna(0).sum()),
            "death_48_72": int(numeric(q.get("death_48_72", pd.Series(0, index=q.index))).fillna(0).sum()),
            "icu_discharge_48_72": int(numeric(q.get("icu_discharge_48_72", pd.Series(0, index=q.index))).fillna(0).sum()),
            "hospital_discharge_48_72": int(numeric(q.get("hospital_discharge_48_72", pd.Series(0, index=q.index))).fillna(0).sum()),
            "missing_or_unassessed_cam": int((~q.strict_cam_assessed_48_72.astype(bool)).sum()),
            "boundary": "Missing CAM was not coded as negative; death/discharge/unassessed observation are reported as competing observation processes.",
        }])


    def main() -> None:
        args = parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)

        full = read_landmark_rows(args.full_icu_rows)
        full = add_mbf_windows(full, args.clinical_windows, args.intervention_windows, args.chunksize)
        full = add_challenge_index(full)
        full, mbf_audit, importance = fit_mbf_index(full, args.folds, args.max_iter, args.bootstrap)
        diabetes = full[full["diabetes"].astype(int).eq(1)].copy()
        non_diabetes = full[full["diabetes"].astype(int).eq(0)].copy()

        full.to_csv(args.output_dir / "mbf_analysis_rows_full_icu.csv.gz", index=False, compression="gzip")
        diabetes.to_csv(args.output_dir / "mbf_analysis_rows_diabetes.csv.gz", index=False, compression="gzip")
        mbf_audit.to_csv(args.output_dir / "mbf_index_generation_audit.csv", index=False)
        importance.to_csv(args.output_dir / "mbf_index_variable_contributions.csv", index=False)

        model_specs = [
            ("Model A: severity/reserve", base_renal_features(full)),
            ("Model B: + traditional glycaemia", base_renal_features(full) + traditional_glucose_features(full)),
            ("Model C: + MBF index", base_renal_features(full) + traditional_glucose_features(full) + ["mbf_index"]),
        ]
        perf_full, preds_full = crossfit_model_performance(
            full[full.mbf_index.notna()].copy(), model_specs, "renal_transition_burden",
            args.folds, args.max_iter, 1000,
        )
        perf_full.insert(0, "cohort", "Full ICU")
        lr_full = linear_lr_table(full[full.mbf_index.notna()].copy(), model_specs, "renal_transition_burden")
        lr_full.insert(0, "cohort", "Full ICU")
        model_specs_dm = [
            ("Model A: severity/reserve", base_renal_features(diabetes)),
            ("Model B: + traditional glycaemia", base_renal_features(diabetes) + traditional_glucose_features(diabetes)),
            ("Model C: + MBF index", base_renal_features(diabetes) + traditional_glucose_features(diabetes) + ["mbf_index"]),
        ]
        perf_dm, preds_dm = crossfit_model_performance(
            diabetes[diabetes.mbf_index.notna()].copy(), model_specs_dm, "renal_transition_burden",
            args.folds, args.max_iter, 1200,
        )
        perf_dm.insert(0, "cohort", "Diabetes")
        lr_dm = linear_lr_table(diabetes[diabetes.mbf_index.notna()].copy(), model_specs_dm, "renal_transition_burden")
        lr_dm.insert(0, "cohort", "Diabetes")
        pd.concat([perf_full, perf_dm], ignore_index=True).to_csv(
            args.output_dir / "mbf_incremental_value_renal_transition.csv", index=False
        )
        pd.concat([lr_full, lr_dm], ignore_index=True).to_csv(
            args.output_dir / "mbf_incremental_value_linear_lr_sensitivity.csv", index=False
        )
        preds_full.to_csv(args.output_dir / "mbf_incremental_predictions_full_icu.csv", index=False)
        preds_dm.to_csv(args.output_dir / "mbf_incremental_predictions_diabetes.csv", index=False)

        dist, severity_coef = mbf_specificity(full, args.bootstrap)
        dist.to_csv(args.output_dir / "mbf_distribution_by_diabetes.csv", index=False)
        severity_coef.to_csv(args.output_dir / "mbf_severity_independence_model.csv", index=False)

        qt_full, eff_full = mbf_quartile_table(full, "Full ICU", args.bootstrap)
        qt_dm, eff_dm = mbf_quartile_table(diabetes, "Diabetes", args.bootstrap)
        pd.concat([qt_full, qt_dm], ignore_index=True).to_csv(args.output_dir / "mbf_quartile_outcomes.csv", index=False)
        pd.concat([eff_full, eff_dm], ignore_index=True).to_csv(args.output_dir / "mbf_quartile_adjusted_effects.csv", index=False)

        cam_full = cam_bridge_models(full, "Full ICU", args.folds, args.max_iter)
        cam_dm = cam_bridge_models(diabetes, "Diabetes", args.folds, args.max_iter)
        pd.concat([cam_full, cam_dm], ignore_index=True).to_csv(args.output_dir / "mbf_cam_bridge_models.csv", index=False)

        attenuation = diabetes_attenuation(full, args.bootstrap)
        attenuation.to_csv(args.output_dir / "diabetes_effect_attenuation_by_mbf.csv", index=False)

        surface_full, support_full = mbf_landscape(full, "Full ICU", args.max_iter)
        surface_dm, support_dm = mbf_landscape(diabetes, "Diabetes", args.max_iter)
        pd.concat([surface_full, surface_dm], ignore_index=True).to_csv(args.output_dir / "mbf_landscape_surface.csv", index=False)
        pd.concat([support_full, support_dm], ignore_index=True).to_csv(args.output_dir / "mbf_landscape_empirical_support.csv", index=False)

        fals = falsification_sensitivity(full, args.bootstrap)
        fals.to_csv(args.output_dir / "mbf_falsification_exposure_sensitivity.csv", index=False)

        arch_perf, arch_qt = archived_state_sensitivity(
            args.archived_diabetes_rows, diabetes, args.bootstrap, args.folds, args.max_iter
        )
        arch_perf.to_csv(args.output_dir / "mbf_archived_state_incremental_sensitivity.csv", index=False)
        arch_qt.to_csv(args.output_dir / "mbf_archived_state_quartile_sensitivity.csv", index=False)

        pd.concat([
            selection_bias_audit(full, "Full ICU"),
            selection_bias_audit(diabetes, "Diabetes"),
            selection_bias_audit(non_diabetes, "Non-diabetes"),
        ], ignore_index=True).to_csv(args.output_dir / "mbf_cam_observation_audit.csv", index=False)

        # Correlation with conventional glycaemic features helps show whether MBF is
        # reducible to glucose variability or hyperglycaemia.
        corr_cols = ["mbf_index"] + traditional_glucose_features(full)
        if "mbf_observed_glucose_delta_12_24" in full:
            corr_cols.append("mbf_observed_glucose_delta_12_24")
        corr = full[corr_cols].corr(numeric_only=True).reset_index().rename(columns={"index": "feature"})
        corr.to_csv(args.output_dir / "mbf_vs_traditional_glycaemia_correlations.csv", index=False)

        summary = {
            "hypothesis": (
                "Diabetes susceptibility -> impaired metabolic buffering capacity "
                "measured by MBF index -> renal-metabolic vulnerability trajectory "
                "-> acute brain dysfunction."
            ),
            "mbf_index_definition": (
                "Cross-fitted residual from observed 12-24 h glucose change minus expected "
                "12-24 h glucose change predicted from 0-6 h reserve and 0-12 h metabolic challenge."
            ),
            "n_full_icu_landmark": int(len(full)),
            "n_diabetes_landmark": int(len(diabetes)),
            "n_non_diabetes_landmark": int(len(non_diabetes)),
            "mbf_available_full_icu": int(full.mbf_index.notna().sum()),
            "mbf_available_diabetes": int(diabetes.mbf_index.notna().sum()),
            "pathway_language": "attenuation/explanatory pathway; no formal causal mediation claim",
            "positive_gates": {
                "mbf_not_ordinary_glucose_metric": bool(
                    perf_full.loc[perf_full.model.eq("Model C: + MBF index"), "R2"].iloc[0]
                    > perf_full.loc[perf_full.model.eq("Model B: + traditional glycaemia"), "R2"].iloc[0]
                ),
                "mbf_higher_in_diabetes_adjusted": bool(severity_coef.estimate.iloc[0] > 0),
                "diabetes_beta_attenuated_for_renal_transition": bool(
                    attenuation[
                        attenuation.outcome.eq("renal_transition_burden")
                        & attenuation.model.str.startswith("Model 2")
                    ].attenuation_fraction.iloc[0] > 0
                ),
            },
        }
        (args.output_dir / "mbf_analysis_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return SimpleNamespace(**locals())

_core = _load_core()

_register_legacy("run_metabolic_buffering_failure_analysis_20260713", _core)


# ==============================================================================
# Reliability, external validation, and specificity
# ==============================================================================

def _load_reliability():
    """Reliability, external validation and biological specificity for MBF.

    This script addresses three reviewer-facing additions:

    A. MBF reliability across early windows:
       - 0-12 h reserve/challenge -> 12-24 h response
       - 6-12 h reserve/challenge -> 12-24 h response

    B. External eICU validation:
       - construct the same residual phenotype from eICU 6-h windows
       - test whether diabetes is associated with higher MBF

    C. Biological specificity:
       - test whether MBF is most consistently associated with
         hyperglycaemic-instability burden rather than unrelated medication or
         inflammatory/context targets.

    The script does not treat MBF as a formal causal mediator.
    """

    import argparse
    import json
    import math
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline


    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    from run_metabolic_buffering_failure_analysis_20260713 import (  # noqa: E402
        SEED,
        base_renal_features,
        linear_coef_bootstrap,
        numeric,
    )


    def parse_args() -> argparse.Namespace:
        package = Path(__file__).resolve().parents[2]
        data_root = package / "data"
        p = argparse.ArgumentParser()
        p.add_argument(
            "--mbf-output-dir",
            type=Path,
            default=package / "outputs" / "downstream" / "metabolic_buffering_failure",
        )
        p.add_argument(
            "--output-dir",
            type=Path,
            default=package / "outputs" / "downstream" / "metabolic_buffering_failure_validation",
        )
        p.add_argument(
            "--clinical-windows",
            type=Path,
            default=data_root / "derived" / "clinical_order_silent_deployment_windows.csv",
        )
        p.add_argument(
            "--intervention-windows",
            type=Path,
            default=data_root / "derived" / "latent_trajectory" / "mimic_intervention_features_6h.csv.gz",
        )
        p.add_argument(
            "--eicu-windows",
            type=Path,
            default=data_root / "derived" / "eicu_master_cohort_en_repaired.csv",
        )
        p.add_argument("--chunksize", type=int, default=350_000)
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--bootstrap", type=int, default=50)
        p.add_argument("--max-iter", type=int, default=160)
        p.add_argument("--skip-mimic", action="store_true")
        p.add_argument("--skip-eicu", action="store_true")
        return p.parse_args()


    def hgb_reg(max_iter: int, seed: int) -> Pipeline:
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("reg", HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.045,
                max_iter=max_iter,
                max_leaf_nodes=15,
                min_samples_leaf=35,
                l2_regularization=1.0,
                random_state=seed,
            )),
        ])


    def calibration_slope_intercept(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
        if len(y) < 3 or np.nanstd(pred) <= 1e-12:
            return np.nan, np.nan
        lr = LinearRegression().fit(pred.reshape(-1, 1), y)
        return float(lr.coef_[0]), float(lr.intercept_)


    def regression_metrics(y: np.ndarray, pred: np.ndarray, model: str, n_features: int) -> dict:
        rmse = math.sqrt(mean_squared_error(y, pred))
        slope, intercept = calibration_slope_intercept(y, pred)
        return {
            "model": model,
            "n": int(len(y)),
            "n_features": int(n_features),
            "R2": float(r2_score(y, pred)),
            "RMSE": float(rmse),
            "MAE": float(mean_absolute_error(y, pred)),
            "calibration_slope": float(slope),
            "calibration_intercept": float(intercept),
        }


    def crossfit_expected(
        data: pd.DataFrame,
        features: list[str],
        target: str,
        groups: pd.Series,
        folds: int,
        max_iter: int,
        seed_offset: int,
    ) -> tuple[np.ndarray, dict]:
        ok = data[target].notna()
        pred = np.full(len(data), np.nan)
        x = data[features].apply(pd.to_numeric, errors="coerce")
        y = numeric(data[target]).to_numpy(float)
        n_groups = groups.loc[ok].nunique()
        n_splits = min(folds, int(n_groups))
        if n_splits < 2:
            raise RuntimeError("Not enough groups for cross-fitting.")
        splitter = GroupKFold(n_splits=n_splits)
        idx_ok = np.where(ok.to_numpy())[0]
        for fold, (tr0, te0) in enumerate(splitter.split(x.loc[ok], y[ok], groups.loc[ok])):
            tr = idx_ok[tr0]
            te = idx_ok[te0]
            model = hgb_reg(max_iter, SEED + seed_offset + fold)
            model.fit(x.iloc[tr], y[tr])
            pred[te] = model.predict(x.iloc[te])
        q = ok & np.isfinite(pred)
        return pred, regression_metrics(y[q], pred[q], "grouped_crossfit", len(features))


    def weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
        v = numeric(values)
        if weights is None:
            return float(v.mean(skipna=True))
        w = numeric(weights).fillna(0)
        ok = v.notna() & w.gt(0)
        if ok.any():
            return float(np.average(v[ok], weights=w[ok]))
        return float(v.mean(skipna=True))


    def read_mimic_clinical_for_windows(path: Path, stay_ids: set[int], chunksize: int) -> pd.DataFrame:
        usecols = [
            "stay_id", "bin", "glucose_mean", "glucose_cv", "tir_fraction",
            "hyper_severe_fraction", "glucose_count", "n_glucose", "sofa",
            "vasopressor_sum", "mv_flag", "rrt_flag", "lactate", "creatinine",
            "bun",
        ]
        frames = []
        for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize, low_memory=False):
            q = chunk[chunk["stay_id"].isin(stay_ids) & chunk["bin"].between(0, 3)].copy()
            if not q.empty:
                frames.append(q)
        if not frames:
            raise RuntimeError("No MIMIC clinical windows found.")
        out = pd.concat(frames, ignore_index=True)
        if "glucose_count" not in out:
            out["glucose_count"] = np.nan
        if "n_glucose" in out:
            out["glucose_count"] = numeric(out["glucose_count"]).fillna(numeric(out["n_glucose"]))
        out["glucose_count"] = numeric(out["glucose_count"])
        return out


    def read_mimic_interventions_for_windows(path: Path, stay_ids: set[int], chunksize: int) -> pd.DataFrame:
        usecols = [
            "stay_id", "bin", "enteral_kcal_sum", "propofol_kcal",
            "treatment_aware_kcal", "insulin_units", "steroid_flag",
            "dextrose_amount_sum", "parenteral_nutrition_amount_sum", "weight_kg",
        ]
        frames = []
        for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize, low_memory=False):
            q = chunk[chunk["stay_id"].isin(stay_ids) & chunk["bin"].between(0, 1)].copy()
            if not q.empty:
                frames.append(q)
        if not frames:
            raise RuntimeError("No MIMIC intervention windows found.")
        return pd.concat(frames, ignore_index=True)


    def aggregate_variant_windows(
        clinical: pd.DataFrame,
        interventions: pd.DataFrame,
        rows: pd.DataFrame,
        name: str,
        input_bins: list[int],
        response_bins: list[int],
    ) -> pd.DataFrame:
        """Return stay-level features for a MBF reliability variant."""
        clin_in = clinical[clinical["bin"].isin(input_bins)].copy()
        clin_resp = clinical[clinical["bin"].isin(response_bins)].copy()
        input_rows = []
        for sid, g in clin_in.groupby("stay_id", sort=False):
            input_rows.append({
                "stay_id": sid,
                f"{name}_input_glucose_mean": weighted_mean(g["glucose_mean"], g["glucose_count"]),
                f"{name}_input_glucose_cv": float(numeric(g["glucose_cv"]).mean(skipna=True)),
                f"{name}_input_tir_fraction": float(numeric(g["tir_fraction"]).mean(skipna=True)),
                f"{name}_input_hyper_severe_fraction": float(numeric(g["hyper_severe_fraction"]).mean(skipna=True)),
                f"{name}_input_sofa_max": float(numeric(g["sofa"]).max(skipna=True)),
                f"{name}_input_vasopressor_max": float(numeric(g["vasopressor_sum"]).max(skipna=True)),
                f"{name}_input_mv_any": float(numeric(g["mv_flag"]).max(skipna=True)),
                f"{name}_input_rrt_any": float(numeric(g["rrt_flag"]).max(skipna=True)),
                f"{name}_input_lactate_max": float(numeric(g["lactate"]).max(skipna=True)),
                f"{name}_input_creatinine_max": float(numeric(g["creatinine"]).max(skipna=True)),
                f"{name}_input_bun_max": float(numeric(g["bun"]).max(skipna=True)),
            })
        resp_rows = []
        for sid, g in clin_resp.groupby("stay_id", sort=False):
            resp_rows.append({
                "stay_id": sid,
                f"{name}_response_glucose_mean": weighted_mean(g["glucose_mean"], g["glucose_count"]),
                f"{name}_response_glucose_cv": float(numeric(g["glucose_cv"]).mean(skipna=True)),
                f"{name}_response_hyper_severe_fraction": float(numeric(g["hyper_severe_fraction"]).mean(skipna=True)),
            })
        inp = pd.DataFrame(input_rows)
        resp = pd.DataFrame(resp_rows)

        iv = interventions[interventions["bin"].isin(input_bins)].copy()
        iv_agg = iv.groupby("stay_id", as_index=False).agg(
            _iv_enteral_kcal=("enteral_kcal_sum", "sum"),
            _iv_propofol_kcal=("propofol_kcal", "sum"),
            _iv_treatment_aware_kcal=("treatment_aware_kcal", "sum"),
            _iv_insulin_units=("insulin_units", "sum"),
            _iv_steroid_any=("steroid_flag", "max"),
            _iv_dextrose_g=("dextrose_amount_sum", "sum"),
            _iv_parenteral_nutrition=("parenteral_nutrition_amount_sum", "sum"),
            _iv_window_weight_kg=("weight_kg", "median"),
        )
        out = rows.merge(inp, on="stay_id", how="left").merge(resp, on="stay_id", how="left").merge(iv_agg, on="stay_id", how="left")
        weight = numeric(out.get("analysis_weight_kg", out.get("weight_kg", out.get("_iv_window_weight_kg"))))
        weight = weight.fillna(numeric(out.get("_iv_window_weight_kg", pd.Series(np.nan, index=out.index))))
        weight = weight.clip(lower=30, upper=250)
        out[f"{name}_weight_kg"] = weight
        for c in ["enteral_kcal", "propofol_kcal", "treatment_aware_kcal", "insulin_units", "dextrose_g"]:
            out[f"{name}_{c}_per_kg"] = numeric(out[f"_iv_{c}"]).fillna(0) / out[f"{name}_weight_kg"]
        out[f"{name}_steroid_any"] = numeric(out["_iv_steroid_any"]).fillna(0)
        out[f"{name}_parenteral_nutrition"] = numeric(out["_iv_parenteral_nutrition"]).fillna(0)
        out[f"{name}_observed_glucose_delta"] = numeric(out[f"{name}_response_glucose_mean"]) - numeric(out[f"{name}_input_glucose_mean"])
        return out


    def variant_features(data: pd.DataFrame, name: str) -> list[str]:
        candidates = [
            "anchor_age", "charlson_index", "admission_sofa", f"{name}_weight_kg",
            f"{name}_input_glucose_mean", f"{name}_input_glucose_cv",
            f"{name}_input_hyper_severe_fraction", f"{name}_input_tir_fraction",
            f"{name}_input_sofa_max", f"{name}_input_vasopressor_max",
            f"{name}_input_mv_any", f"{name}_input_rrt_any",
            f"{name}_input_lactate_max", f"{name}_input_creatinine_max",
            f"{name}_input_bun_max", f"{name}_dextrose_g_per_kg",
            f"{name}_enteral_kcal_per_kg", f"{name}_propofol_kcal_per_kg",
            f"{name}_treatment_aware_kcal_per_kg", f"{name}_insulin_units_per_kg",
            f"{name}_steroid_any", f"{name}_parenteral_nutrition",
        ]
        return [c for c in candidates if c in data.columns]


    def fit_variant_mbf(data: pd.DataFrame, name: str, folds: int, max_iter: int, seed_offset: int) -> tuple[pd.DataFrame, dict]:
        out = data.copy()
        target = f"{name}_observed_glucose_delta"
        features = variant_features(out, name)
        pred, metrics = crossfit_expected(out, features, target, out["subject_id"], folds, max_iter, seed_offset)
        out[f"{name}_expected_glucose_delta"] = pred
        out[f"{name}_raw_residual"] = numeric(out[target]) - out[f"{name}_expected_glucose_delta"]
        mu = float(out[f"{name}_raw_residual"].mean(skipna=True))
        sd = float(out[f"{name}_raw_residual"].std(skipna=True))
        out[f"{name}_index"] = (out[f"{name}_raw_residual"] - mu) / sd
        metrics.update({
            "variant": name,
            "target": target,
            "features": ", ".join(features),
            "available": int(out[f"{name}_index"].notna().sum()),
        })
        return out, metrics


    def simple_covars_for_variant(data: pd.DataFrame, mbf_col: str, name: str) -> list[str]:
        candidates = [
            mbf_col, "diabetes", "anchor_age", "charlson_index", "admission_sofa",
            f"{name}_input_glucose_mean", f"{name}_input_glucose_cv",
            f"{name}_input_creatinine_max", f"{name}_input_lactate_max",
            f"{name}_input_sofa_max", f"{name}_input_vasopressor_max",
            f"{name}_input_rrt_any",
        ]
        return [c for c in candidates if c in data.columns]


    def reliability_readouts(data: pd.DataFrame, variants: list[str], n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows = []
        coeffs = []
        for name in variants:
            mbf_col = f"{name}_index"
            q = data[data[mbf_col].notna()].copy()
            # diabetes -> MBF
            dm_covars = [c for c in [
                "diabetes", "anchor_age", "charlson_index", "admission_sofa",
                f"{name}_input_glucose_mean", f"{name}_input_glucose_cv",
                f"{name}_input_creatinine_max", f"{name}_input_lactate_max",
                f"{name}_input_sofa_max", f"{name}_input_vasopressor_max",
                f"{name}_input_rrt_any",
            ] if c in q.columns]
            dm = linear_coef_bootstrap(q, mbf_col, dm_covars, "diabetes", n_boot, SEED + 21000)
            dm.insert(0, "variant", name)
            dm["association"] = "diabetes -> MBF index"
            coeffs.append(dm)
            for target, label in [
                ("post_mean_p_state_5", "hyperglycaemic-instability burden"),
                ("renal_transition_burden", "renal-metabolic vulnerability trajectory"),
            ]:
                covars = simple_covars_for_variant(q, mbf_col, name)
                if target in q.columns:
                    coef = linear_coef_bootstrap(q, target, covars, mbf_col, n_boot, SEED + 21100)
                    coef.insert(0, "variant", name)
                    coef["association"] = f"MBF index -> {label}"
                    coeffs.append(coef)
            rows.append({
                "variant": name,
                "n": int(q.shape[0]),
                "subjects": int(q["subject_id"].nunique()),
                "mbf_mean": float(numeric(q[mbf_col]).mean(skipna=True)),
                "diabetes_mbf_mean": float(numeric(q.loc[q.diabetes.astype(int).eq(1), mbf_col]).mean(skipna=True)),
                "non_diabetes_mbf_mean": float(numeric(q.loc[q.diabetes.astype(int).eq(0), mbf_col]).mean(skipna=True)),
                "hyperglycaemic_burden_mean": float(numeric(q["post_mean_p_state_5"]).mean(skipna=True)),
                "renal_transition_burden_mean": float(numeric(q["renal_transition_burden"]).mean(skipna=True)),
            })
        # pairwise correlations among available variants plus primary MBF.
        corr_cols = [c for c in ["mbf_index"] + [f"{v}_index" for v in variants] if c in data.columns]
        corr = data[corr_cols].corr(numeric_only=True).reset_index().rename(columns={"index": "metric"})
        return pd.DataFrame(rows), pd.concat(coeffs, ignore_index=True), corr


    def read_eicu_windows(path: Path, chunksize: int) -> pd.DataFrame:
        usecols = [
            "stay_id", "bin", "glucose_mean", "glucose_cv", "tir_fraction",
            "hyper_severe_fraction", "glucose_count", "sofa", "vasopressor_sum",
            "mv_flag", "rrt_flag", "lactate", "creatinine", "bun",
            "enteral_kcal_sum_repaired", "enteral_kcal_sum", "propofol_rate_sum",
            "insulin_rate_sum", "insulin_med_dose_sum", "steroid_flag",
            "dextrose_kcal", "parenteral_nutrition_kcal", "weight_kg",
            "anchor_age", "charlson_index", "apache_score", "severity_score",
            "diabetes", "hospital_mortality", "icu_mortality",
        ]
        frames = []
        for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize, low_memory=False):
            q = chunk[chunk["bin"].between(0, 3)].copy()
            if not q.empty:
                frames.append(q)
        if not frames:
            raise RuntimeError("No eICU windows found.")
        return pd.concat(frames, ignore_index=True)


    def aggregate_eicu_variant(eicu: pd.DataFrame, name: str, input_bins: list[int], response_bins: list[int]) -> pd.DataFrame:
        in_df = eicu[eicu["bin"].isin(input_bins)].copy()
        resp_df = eicu[eicu["bin"].isin(response_bins)].copy()
        base_static = eicu.sort_values("bin").groupby("stay_id", as_index=False).first()
        input_rows = []
        for sid, g in in_df.groupby("stay_id", sort=False):
            input_rows.append({
                "stay_id": sid,
                "subject_id": sid,
                f"{name}_input_glucose_mean": weighted_mean(g["glucose_mean"], g["glucose_count"]),
                f"{name}_input_glucose_cv": float(numeric(g["glucose_cv"]).mean(skipna=True)),
                f"{name}_input_tir_fraction": float(numeric(g["tir_fraction"]).mean(skipna=True)),
                f"{name}_input_hyper_severe_fraction": float(numeric(g["hyper_severe_fraction"]).mean(skipna=True)),
                f"{name}_input_sofa_max": float(numeric(g.get("sofa", pd.Series(np.nan))).max(skipna=True)),
                f"{name}_input_vasopressor_max": float(numeric(g["vasopressor_sum"]).max(skipna=True)),
                f"{name}_input_mv_any": float(numeric(g["mv_flag"]).max(skipna=True)),
                f"{name}_input_rrt_any": float(numeric(g["rrt_flag"]).max(skipna=True)),
                f"{name}_input_lactate_max": float(numeric(g["lactate"]).max(skipna=True)),
                f"{name}_input_creatinine_max": float(numeric(g["creatinine"]).max(skipna=True)),
                f"{name}_input_bun_max": float(numeric(g["bun"]).max(skipna=True)),
                f"{name}_enteral_kcal": float(numeric(g.get("enteral_kcal_sum_repaired", g.get("enteral_kcal_sum"))).fillna(0).sum()),
                f"{name}_propofol_rate_sum": float(numeric(g["propofol_rate_sum"]).fillna(0).sum()),
                f"{name}_insulin_units": float((numeric(g["insulin_rate_sum"]).fillna(0) + numeric(g["insulin_med_dose_sum"]).fillna(0)).sum()),
                f"{name}_steroid_any": float(numeric(g["steroid_flag"]).fillna(0).max()),
                f"{name}_dextrose_kcal": float(numeric(g["dextrose_kcal"]).fillna(0).sum()),
                f"{name}_parenteral_nutrition": float(numeric(g["parenteral_nutrition_kcal"]).fillna(0).sum()),
                f"{name}_weight_kg": float(numeric(g["weight_kg"]).median(skipna=True)),
            })
        resp_rows = []
        for sid, g in resp_df.groupby("stay_id", sort=False):
            resp_rows.append({
                "stay_id": sid,
                f"{name}_response_glucose_mean": weighted_mean(g["glucose_mean"], g["glucose_count"]),
            })
        out = pd.DataFrame(input_rows).merge(pd.DataFrame(resp_rows), on="stay_id", how="left")
        static_cols = [
            "stay_id", "anchor_age", "charlson_index", "apache_score", "severity_score",
            "diabetes", "hospital_mortality", "icu_mortality",
        ]
        out = out.merge(base_static[[c for c in static_cols if c in base_static.columns]], on="stay_id", how="left")
        weight = numeric(out[f"{name}_weight_kg"]).clip(lower=30, upper=250)
        out[f"{name}_weight_kg"] = weight
        for c in ["enteral_kcal", "propofol_rate_sum", "insulin_units", "dextrose_kcal"]:
            out[f"{name}_{c}_per_kg"] = numeric(out[f"{name}_{c}"]).fillna(0) / weight
        out[f"{name}_observed_glucose_delta"] = numeric(out[f"{name}_response_glucose_mean"]) - numeric(out[f"{name}_input_glucose_mean"])
        return out


    def eicu_features(data: pd.DataFrame, name: str) -> list[str]:
        candidates = [
            "anchor_age", "charlson_index", "apache_score", "severity_score",
            f"{name}_weight_kg", f"{name}_input_glucose_mean", f"{name}_input_glucose_cv",
            f"{name}_input_hyper_severe_fraction", f"{name}_input_tir_fraction",
            f"{name}_input_sofa_max", f"{name}_input_vasopressor_max",
            f"{name}_input_mv_any", f"{name}_input_rrt_any", f"{name}_input_lactate_max",
            f"{name}_input_creatinine_max", f"{name}_input_bun_max",
            f"{name}_dextrose_kcal_per_kg", f"{name}_enteral_kcal_per_kg",
            f"{name}_propofol_rate_sum_per_kg", f"{name}_insulin_units_per_kg",
            f"{name}_steroid_any", f"{name}_parenteral_nutrition",
        ]
        return [c for c in candidates if c in data.columns]


    def fit_eicu_mbf(data: pd.DataFrame, name: str, folds: int, max_iter: int, seed_offset: int) -> tuple[pd.DataFrame, dict]:
        out = data.copy()
        target = f"{name}_observed_glucose_delta"
        features = eicu_features(out, name)
        pred, metrics = crossfit_expected(out, features, target, out["stay_id"], folds, max_iter, seed_offset)
        out[f"{name}_expected_glucose_delta"] = pred
        out[f"{name}_raw_residual"] = numeric(out[target]) - out[f"{name}_expected_glucose_delta"]
        mu = float(out[f"{name}_raw_residual"].mean(skipna=True))
        sd = float(out[f"{name}_raw_residual"].std(skipna=True))
        out[f"{name}_index"] = (out[f"{name}_raw_residual"] - mu) / sd
        metrics.update({
            "cohort": "eICU",
            "variant": name,
            "target": target,
            "features": ", ".join(features),
            "available": int(out[f"{name}_index"].notna().sum()),
        })
        return out, metrics


    def eicu_dm_readout(data: pd.DataFrame, variants: list[str], n_boot: int) -> pd.DataFrame:
        rows = []
        for name in variants:
            mbf_col = f"{name}_index"
            q = data[data[mbf_col].notna()].copy()
            covars = [c for c in [
                "diabetes", "anchor_age", "charlson_index", "apache_score", "severity_score",
                f"{name}_input_glucose_mean", f"{name}_input_glucose_cv",
                f"{name}_input_creatinine_max", f"{name}_input_lactate_max",
                f"{name}_input_vasopressor_max", f"{name}_input_rrt_any",
            ] if c in q.columns]
            coef = linear_coef_bootstrap(q, mbf_col, covars, "diabetes", n_boot, SEED + 31000)
            coef.insert(0, "cohort", "eICU")
            coef.insert(1, "variant", name)
            coef["association"] = "diabetes -> MBF index"
            coef["diabetes_mbf_mean"] = float(numeric(q.loc[q.diabetes.astype(int).eq(1), mbf_col]).mean(skipna=True))
            coef["non_diabetes_mbf_mean"] = float(numeric(q.loc[q.diabetes.astype(int).eq(0), mbf_col]).mean(skipna=True))
            rows.append(coef)
        return pd.concat(rows, ignore_index=True)


    def specificity_readouts(data: pd.DataFrame, mbf_cols: list[str], n_boot: int) -> pd.DataFrame:
        target_labels = [
            ("post_mean_p_state_5", "positive control: hyperglycaemic-instability burden", "state_probability"),
            ("renal_transition_burden", "comparison: renal-metabolic vulnerability trajectory", "state_probability"),
            ("post_mean_p_state_0", "negative/context: inflammatory-haemodynamic state burden", "state_probability"),
            ("ppi_h2_any_6_24", "negative/context: PPI/H2 exposure", "binary_context"),
            ("antiemetic_any_6_24", "negative/context: antiemetic exposure", "binary_context"),
            ("heparin_any_6_24", "negative/context: heparin exposure", "binary_context"),
            ("pre_antibiotic_any", "negative/context: baseline antibiotic exposure", "binary_context"),
        ]
        rows = []
        for mbf_col in mbf_cols:
            q = data[data[mbf_col].notna()].copy()
            for target, label, family in target_labels:
                if target not in q.columns:
                    continue
                covars = base_renal_features(q)
                # base_renal_features includes baseline glucose/severity/state context.
                covars = [c for c in covars if c in q.columns] + [mbf_col]
                coef = linear_coef_bootstrap(q, target, covars, mbf_col, n_boot, SEED + 41000 + len(rows))
                coef.insert(0, "mbf_metric", mbf_col)
                coef.insert(1, "target_label", label)
                coef["target_family"] = family
                rows.append(coef)
        return pd.concat(rows, ignore_index=True)


    def main() -> None:
        args = parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)

        variants = {
            "mbf_0_12_to_12_24": ([0, 1], [2, 3]),
            "mbf_6_12_to_12_24": ([1], [2, 3]),
        }
        variant_names = list(variants.keys())
        if not args.skip_mimic:
            full = pd.read_csv(args.mbf_output_dir / "mbf_analysis_rows_full_icu.csv.gz", low_memory=False)
            stay_ids = set(pd.to_numeric(full["stay_id"], errors="coerce").dropna().astype(int))
            clinical = read_mimic_clinical_for_windows(args.clinical_windows, stay_ids, args.chunksize)
            interventions = read_mimic_interventions_for_windows(args.intervention_windows, stay_ids, args.chunksize)

            reliability = full.copy()
            metrics_rows = []
            for i, (name, (input_bins, response_bins)) in enumerate(variants.items()):
                var_data = aggregate_variant_windows(clinical, interventions, full, name, input_bins, response_bins)
                reliability, metrics = fit_variant_mbf(var_data, name, args.folds, args.max_iter, 5000 + i * 100)
                # Carry forward previously fitted variant columns to the combined table.
                cols_to_add = ["stay_id"] + [c for c in reliability.columns if c.startswith(name)]
                full = full.merge(reliability[cols_to_add], on="stay_id", how="left")
                reliability = full.copy()
                metrics["cohort"] = "MIMIC full ICU"
                metrics["input_bins"] = ",".join(map(str, input_bins))
                metrics["response_bins"] = ",".join(map(str, response_bins))
                metrics_rows.append(metrics)

            rel_summary, rel_coeffs, rel_corr = reliability_readouts(reliability, variant_names, args.bootstrap)
            pd.DataFrame(metrics_rows).to_csv(args.output_dir / "mbf_reliability_model_metrics_mimic.csv", index=False)
            rel_summary.to_csv(args.output_dir / "mbf_reliability_window_summary_mimic.csv", index=False)
            rel_coeffs.to_csv(args.output_dir / "mbf_reliability_window_coefficients_mimic.csv", index=False)
            rel_corr.to_csv(args.output_dir / "mbf_reliability_window_correlations_mimic.csv", index=False)
            keep_cols = [
                "stay_id", "subject_id", "diabetes", "mbf_index", "post_mean_p_state_5",
                "renal_transition_burden", "post_mean_p_state_0",
            ] + [f"{v}_index" for v in variant_names]
            reliability[[c for c in keep_cols if c in reliability.columns]].to_csv(
                args.output_dir / "mbf_reliability_analysis_rows_mimic.csv.gz", index=False, compression="gzip"
            )

            # Biological specificity in MIMIC.
            mbf_cols = [c for c in ["mbf_index"] + [f"{v}_index" for v in variant_names] if c in reliability.columns]
            spec = specificity_readouts(reliability, mbf_cols, args.bootstrap)
            spec.to_csv(args.output_dir / "mbf_biological_specificity_mimic.csv", index=False)
        else:
            rel_coeffs = pd.read_csv(args.output_dir / "mbf_reliability_window_coefficients_mimic.csv")
            spec = pd.read_csv(args.output_dir / "mbf_biological_specificity_mimic.csv")

        # eICU external validation. If the file is absent, write a clear audit instead.
        eicu_audit = {"eicu_windows": str(args.eicu_windows), "available": bool(args.eicu_windows.exists())}
        if (not args.skip_eicu) and args.eicu_windows.exists():
            eicu = read_eicu_windows(args.eicu_windows, args.chunksize)
            eicu_combined = None
            eicu_metrics = []
            for i, (name, (input_bins, response_bins)) in enumerate(variants.items()):
                ev = aggregate_eicu_variant(eicu, name, input_bins, response_bins)
                ev, metrics = fit_eicu_mbf(ev, name, args.folds, args.max_iter, 7000 + i * 100)
                eicu_metrics.append(metrics)
                cols = ["stay_id", "subject_id", "diabetes"] + [c for c in ev.columns if c.startswith(name)]
                if eicu_combined is None:
                    eicu_combined = ev[cols].copy()
                else:
                    eicu_combined = eicu_combined.merge(ev[["stay_id"] + [c for c in ev.columns if c.startswith(name)]], on="stay_id", how="outer")
            if eicu_combined is not None:
                eicu_readout = eicu_dm_readout(eicu_combined, variant_names, args.bootstrap)
                pd.DataFrame(eicu_metrics).to_csv(args.output_dir / "mbf_external_eicu_model_metrics.csv", index=False)
                eicu_readout.to_csv(args.output_dir / "mbf_external_eicu_diabetes_validation.csv", index=False)
                eicu_combined.to_csv(args.output_dir / "mbf_external_eicu_analysis_rows.csv.gz", index=False, compression="gzip")
                eicu_audit.update({
                    "n_eicu_rows": int(eicu_combined.shape[0]),
                    "mbf_available_0_12": int(eicu_combined["mbf_0_12_to_12_24_index"].notna().sum()),
                    "mbf_available_6_12": int(eicu_combined["mbf_6_12_to_12_24_index"].notna().sum()),
                })
        (args.output_dir / "mbf_external_eicu_audit.json").write_text(json.dumps(eicu_audit, indent=2), encoding="utf-8")

        # High-level evidence gates.
        gates = {
            "mimic_reliability_diabetes_direction_consistent": bool(
                (rel_coeffs[rel_coeffs["association"].eq("diabetes -> MBF index")]["estimate"] > 0).all()
            ),
            "mimic_reliability_hyperglycaemic_direction_consistent": bool(
                (rel_coeffs[rel_coeffs["association"].str.contains("hyperglycaemic", regex=False)]["estimate"] > 0).all()
            ),
            "mimic_reliability_renal_direction_consistent": bool(
                (rel_coeffs[rel_coeffs["association"].str.contains("renal-metabolic", regex=False)]["estimate"] > 0).all()
            ),
            "specificity_primary_mbf_hyper_positive": bool(
                spec[
                    spec["mbf_metric"].eq("mbf_index")
                    & spec["target_label"].str.contains("hyperglycaemic", regex=False)
                ]["estimate"].iloc[0] > 0
            ),
        }
        if (args.output_dir / "mbf_external_eicu_diabetes_validation.csv").exists():
            eicu_readout = pd.read_csv(args.output_dir / "mbf_external_eicu_diabetes_validation.csv")
            gates["eicu_diabetes_direction_consistent"] = bool((eicu_readout["estimate"] > 0).all())
        summary = {
            "purpose": "MBF reliability, eICU external validation and biological specificity checks.",
            "windows": {
                "mbf_0_12_to_12_24": "0-12 h reserve/challenge -> 12-24 h glucose response",
                "mbf_6_12_to_12_24": "6-12 h reserve/challenge -> 12-24 h glucose response",
            },
            "interpretation_boundary": "Association/attenuation and specificity checks only; no formal mediation claim.",
            "gates": gates,
        }
        (args.output_dir / "mbf_reliability_external_specificity_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return SimpleNamespace(**locals())

_reliability = _load_reliability()


# ==============================================================================
# Extension sensitivity
# ==============================================================================

def _load_extension():
    """Additional sensitivity analyses for the metabolic buffering failure axis.

    This script intentionally reads the already generated MBF analysis rows and
    adds reviewer-facing checks that are not part of the primary MBF construction:

    1. whether MBF adds information for the hyperglycaemic-instability trajectory,
       not only the renal-metabolic trajectory;
    2. whether signed MBF and absolute input-response mismatch behave differently;
    3. empirical landscape support, including an effective-sample-size field.

    The analyses preserve the manuscript language boundary: MBF is treated as an
    early metabolic resilience phenotype, not as a formal causal mediator and not as
    evidence of treatment-management failure.
    """

    import argparse
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd


    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    from run_metabolic_buffering_failure_analysis_20260713 import (  # noqa: E402
        SEED,
        base_renal_features,
        crossfit_model_performance,
        linear_coef_bootstrap,
        linear_lr_table,
        numeric,
        traditional_glucose_features,
    )


    def parse_args() -> argparse.Namespace:
        package = Path(__file__).resolve().parents[2]
        p = argparse.ArgumentParser()
        p.add_argument(
            "--mbf-output-dir",
            type=Path,
            default=package / "outputs" / "downstream" / "metabolic_buffering_failure",
        )
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--bootstrap", type=int, default=500)
        p.add_argument("--max-iter", type=int, default=180)
        return p.parse_args()


    def read_rows(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        full = pd.read_csv(output_dir / "mbf_analysis_rows_full_icu.csv.gz", low_memory=False)
        diabetes = pd.read_csv(output_dir / "mbf_analysis_rows_diabetes.csv.gz", low_memory=False)
        return add_mismatch_features(full), add_mismatch_features(diabetes)


    def add_mismatch_features(data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        out["mbf_abs_index"] = numeric(out["mbf_index"]).abs()
        out["mbf_positive_index"] = numeric(out["mbf_index"]).clip(lower=0)
        out["mbf_negative_index"] = (-numeric(out["mbf_index"])).clip(lower=0)
        return out


    def target_specs() -> list[tuple[str, str]]:
        return [
            ("renal_transition_burden", "renal-metabolic vulnerability trajectory"),
            ("post_mean_p_state_5", "hyperglycaemic-instability burden"),
        ]


    def incremental_value_for_target(
        data: pd.DataFrame,
        cohort: str,
        target: str,
        target_label: str,
        folds: int,
        max_iter: int,
        seed_offset: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        q = data[data["mbf_index"].notna()].copy()
        specs = [
            ("Model A: severity/reserve", base_renal_features(q)),
            ("Model B: + traditional glycaemia", base_renal_features(q) + traditional_glucose_features(q)),
            ("Model C: + MBF index", base_renal_features(q) + traditional_glucose_features(q) + ["mbf_index"]),
        ]
        perf, _ = crossfit_model_performance(q, specs, target, folds, max_iter, seed_offset)
        perf.insert(0, "cohort", cohort)
        perf.insert(1, "target_label", target_label)
        lr = linear_lr_table(q, specs, target)
        lr.insert(0, "cohort", cohort)
        lr.insert(1, "target_label", target_label)
        return perf, lr


    def all_incremental_values(
        full: pd.DataFrame,
        diabetes: pd.DataFrame,
        folds: int,
        max_iter: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        perf_rows = []
        lr_rows = []
        for i, (target, label) in enumerate(target_specs()):
            for cohort, data, offset in [
                ("Full ICU", full, 11000 + i * 100),
                ("Diabetes", diabetes, 12000 + i * 100),
            ]:
                perf, lr = incremental_value_for_target(data, cohort, target, label, folds, max_iter, offset)
                perf_rows.append(perf)
                lr_rows.append(lr)
        return pd.concat(perf_rows, ignore_index=True), pd.concat(lr_rows, ignore_index=True)


    def abs_mismatch_quartile_table(
        data: pd.DataFrame,
        cohort: str,
        n_boot: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        q = data[data["mbf_abs_index"].notna()].copy()
        q["mbf_abs_quartile"] = pd.qcut(
            q["mbf_abs_index"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
        )
        summary = q.groupby("mbf_abs_quartile", observed=False).agg(
            n=("stay_id", "size"),
            subjects=("subject_id", "nunique"),
            mbf_abs_mean=("mbf_abs_index", "mean"),
            mbf_signed_mean=("mbf_index", "mean"),
            renal_transition_burden=("renal_transition_burden", "mean"),
            hyperglycaemic_instability_burden=("post_mean_p_state_5", "mean"),
            assessed=("strict_cam_assessed_48_72", "sum"),
            cam_events=("strict_cam_positive_48_72", "sum"),
            cam_rate=("strict_cam_positive_48_72", "mean"),
            response_glucose_mean=("mbf_response_glucose_mean_12_24", "mean"),
            metabolic_challenge=("mbf_metabolic_challenge_index", "mean"),
            insulin_units_per_kg=("mbf_insulin_units_per_kg_0_12", "mean"),
        ).reset_index()
        summary.insert(0, "cohort", cohort)

        q14 = q[q["mbf_abs_quartile"].isin(["Q1", "Q4"])].copy()
        q14["mbf_abs_q4"] = q14["mbf_abs_quartile"].eq("Q4").astype(float)
        effects = []
        for j, (target, label) in enumerate(target_specs()):
            covars = ["mbf_abs_q4"] + base_renal_features(q14)
            eff = linear_coef_bootstrap(q14, target, covars, "mbf_abs_q4", n_boot, SEED + 13000 + j)
            eff.insert(0, "cohort", cohort)
            eff.insert(1, "target_label", label)
            eff["contrast"] = "Q4 vs Q1 absolute MBF mismatch"
            effects.append(eff)
        return summary, pd.concat(effects, ignore_index=True)


    def directional_coefficients(
        data: pd.DataFrame,
        cohort: str,
        n_boot: int,
    ) -> pd.DataFrame:
        q = data[data["mbf_index"].notna()].copy()
        rows = []
        for j, (target, label) in enumerate(target_specs()):
            covars = base_renal_features(q) + ["mbf_positive_index", "mbf_negative_index"]
            for coef_name in ["mbf_positive_index", "mbf_negative_index"]:
                coef = linear_coef_bootstrap(q, target, covars, coef_name, n_boot, SEED + 14000 + j * 10)
                coef.insert(0, "cohort", cohort)
                coef.insert(1, "target_label", label)
                coef["interpretation"] = (
                    "Positive MBF captures higher-than-expected glucose rise; "
                    "negative MBF captures lower-than-expected glucose change. "
                    "Both are adjusted for severity/reserve and baseline states."
                )
                rows.append(coef)
        return pd.concat(rows, ignore_index=True)


    def landscape_support_with_ess(data: pd.DataFrame, cohort: str) -> pd.DataFrame:
        q = data[data["mbf_index"].notna() & data["mbf_metabolic_challenge_index"].notna()].copy()
        q["challenge_bin"] = pd.qcut(q["mbf_metabolic_challenge_index"], 6, labels=False, duplicates="drop")
        q["mbf_bin"] = pd.qcut(q["mbf_index"], 8, labels=False, duplicates="drop")
        grouped = q.groupby(["challenge_bin", "mbf_bin"], dropna=True)
        rows = []
        for (challenge_bin, mbf_bin), g in grouped:
            # No stabilized analysis weight is carried in the MBF landscape rows.
            # Effective sample size is therefore equal to the unweighted bin count.
            assessed = numeric(g["strict_cam_assessed_48_72"]).fillna(0).eq(1)
            rows.append({
                "cohort": cohort,
                "challenge_bin": int(challenge_bin),
                "mbf_bin": int(mbf_bin),
                "n": int(len(g)),
                "effective_sample_size": float(len(g)),
                "weight_policy": "unweighted landscape support; ESS equals bin n",
                "renal_transition_burden": float(numeric(g["renal_transition_burden"]).mean(skipna=True)),
                "hyperglycaemic_instability_burden": float(numeric(g["post_mean_p_state_5"]).mean(skipna=True)),
                "metabolic_challenge_index": float(numeric(g["mbf_metabolic_challenge_index"]).mean(skipna=True)),
                "mbf_index": float(numeric(g["mbf_index"]).mean(skipna=True)),
                "cam_assessed": int(assessed.sum()),
                "cam_events": int(numeric(g.loc[assessed, "strict_cam_positive_48_72"]).fillna(0).sum()),
            })
        return pd.DataFrame(rows)


    def write_extension_summary(output_dir: Path, perf: pd.DataFrame, abs_eff: pd.DataFrame, directional: pd.DataFrame) -> None:
        def model_r2(cohort: str, target_label: str, model: str) -> float:
            q = perf[
                perf["cohort"].eq(cohort)
                & perf["target_label"].eq(target_label)
                & perf["model"].eq(model)
            ]
            return float(q["R2"].iloc[0]) if len(q) else float("nan")

        hyper_label = "hyperglycaemic-instability burden"
        renal_label = "renal-metabolic vulnerability trajectory"
        summary = {
            "extension_scope": (
                "Sensitivity analyses separating signed MBF from absolute input-response mismatch "
                "and comparing renal-metabolic versus hyperglycaemic-instability trajectories."
            ),
            "primary_boundary": (
                "The added analyses support MBF as a diabetes-associated metabolic resilience phenotype, "
                "but do not establish a robust MBF-mediated renal pathway."
            ),
            "incremental_R2": {
                "full_icu_renal_model_B": model_r2("Full ICU", renal_label, "Model B: + traditional glycaemia"),
                "full_icu_renal_model_C": model_r2("Full ICU", renal_label, "Model C: + MBF index"),
                "diabetes_renal_model_B": model_r2("Diabetes", renal_label, "Model B: + traditional glycaemia"),
                "diabetes_renal_model_C": model_r2("Diabetes", renal_label, "Model C: + MBF index"),
                "full_icu_hyperglycaemic_model_B": model_r2("Full ICU", hyper_label, "Model B: + traditional glycaemia"),
                "full_icu_hyperglycaemic_model_C": model_r2("Full ICU", hyper_label, "Model C: + MBF index"),
                "diabetes_hyperglycaemic_model_B": model_r2("Diabetes", hyper_label, "Model B: + traditional glycaemia"),
                "diabetes_hyperglycaemic_model_C": model_r2("Diabetes", hyper_label, "Model C: + MBF index"),
            },
            "abs_mismatch_outputs": str(output_dir / "mbf_absolute_mismatch_quartile_effects.csv"),
            "directional_outputs": str(output_dir / "mbf_signed_directional_coefficients.csv"),
        }
        (output_dir / "mbf_extension_sensitivity_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


    def main() -> None:
        args = parse_args()
        full, diabetes = read_rows(args.mbf_output_dir)

        perf, lr = all_incremental_values(full, diabetes, args.folds, args.max_iter)
        perf.to_csv(args.mbf_output_dir / "mbf_incremental_value_multi_target.csv", index=False)
        lr.to_csv(args.mbf_output_dir / "mbf_incremental_value_multi_target_lr_sensitivity.csv", index=False)

        abs_tables = []
        abs_effects = []
        directional_tables = []
        support_tables = []
        for cohort, data in [("Full ICU", full), ("Diabetes", diabetes)]:
            summary, effects = abs_mismatch_quartile_table(data, cohort, args.bootstrap)
            abs_tables.append(summary)
            abs_effects.append(effects)
            directional_tables.append(directional_coefficients(data, cohort, args.bootstrap))
            support_tables.append(landscape_support_with_ess(data, cohort))

        abs_summary = pd.concat(abs_tables, ignore_index=True)
        abs_eff = pd.concat(abs_effects, ignore_index=True)
        directional = pd.concat(directional_tables, ignore_index=True)
        support = pd.concat(support_tables, ignore_index=True)

        abs_summary.to_csv(args.mbf_output_dir / "mbf_absolute_mismatch_quartile_outcomes.csv", index=False)
        abs_eff.to_csv(args.mbf_output_dir / "mbf_absolute_mismatch_quartile_effects.csv", index=False)
        directional.to_csv(args.mbf_output_dir / "mbf_signed_directional_coefficients.csv", index=False)
        support.to_csv(args.mbf_output_dir / "mbf_landscape_empirical_support_with_ess.csv", index=False)

        write_extension_summary(args.mbf_output_dir, perf, abs_eff, directional)

    return SimpleNamespace(**locals())

_extension = _load_extension()


# ==============================================================================
# PPD temporal robustness
# ==============================================================================

def _load_temporal():
    """Closure analyses for the MBF extension.

    This script adds the final three requested checks:

    1. PPD-EHR latent representation/state probability versus MBF.
    2. A temporal-precedence timeline for the MBF/state/outcome windows.
    3. Robustness of MBF to alternative response definitions:
       glucose CV and severe-hyperglycaemia burden.

    The analyses are observational and are used for linkage/robustness language,
    not for formal mediation claims.
    """

    import json
    import math
    import sys
    from pathlib import Path

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler


    ROOT = Path(__file__).resolve().parents[2]
    OUT = ROOT / "outputs" / "downstream" / "metabolic_buffering_failure_closure"
    MBF_OUT = ROOT / "outputs" / "downstream" / "metabolic_buffering_failure"
    FIG = ROOT / "figures"
    SRC = ROOT / "source_data"
    QA = ROOT / "qa"
    MAN = ROOT / "manuscript"

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    MAN.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT / "code" / "downstream"))
    from run_metabolic_buffering_failure_analysis_20260713 import (  # noqa: E402
        SEED,
        base_renal_features,
        crossfit_regression,
        linear_coef_bootstrap,
        mbf_definition_features,
        numeric,
    )


    PPD_PROB_COLS = [
        "prob_heart_failure",
        "prob_renal_failure",
        "prob_infection",
        "prob_pneumonia",
        "prob_cerebrovascular",
        "prob_diabetic_foot",
    ]

    MBF_USECOLS = [
        "stay_id", "subject_id", "hadm_id", "diabetes", "anchor_age", "charlson_index",
        "admission_sofa", "weight_kg", "mbf_index", "post_mean_p_state_5",
        "renal_transition_burden", "post_mean_p_state_0",
        "mbf_response_glucose_mean_12_24", "mbf_response_glucose_cv_12_24",
        "mbf_response_hyper_severe_fraction_12_24", "mbf_response_tir_loss_12_24",
        "mbf_observed_glucose_delta_12_24", "strict_cam_assessed_48_72",
        "strict_cam_positive_48_72", "analysis_arm",
    ]


    def mm_to_in(mm: float) -> float:
        return mm / 25.4


    def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
        q = pd.DataFrame({"x": numeric(x), "y": numeric(y)}).dropna()
        if len(q) < 10 or q["x"].nunique() < 3 or q["y"].nunique() < 3:
            return np.nan, np.nan, int(len(q))
        r, p = spearmanr(q["x"], q["y"])
        return float(r), float(p), int(len(q))


    def load_mbf_rows() -> pd.DataFrame:
        full = pd.read_csv(MBF_OUT / "mbf_analysis_rows_full_icu.csv.gz")
        needed = list(dict.fromkeys(MBF_USECOLS + mbf_definition_features(full) + base_renal_features(full)))
        needed = [c for c in needed if c in full.columns]
        return full[needed].copy()


    def load_ppd_state(split: str) -> pd.DataFrame:
        p = ROOT / "outputs" / "downstream" / "ppd" / f"ppd_fixed_time_latent_states_{split}.csv.gz"
        cols = [
            "stay_id", "subject_id", "hadm_id", "horizon_hours", "analysis_arm",
            "subject_disjoint_sensitivity_eligible", "hadm_disjoint_sensitivity_eligible",
            "ppd_state_id", "ppd_state", "ppd_state_probability",
        ] + PPD_PROB_COLS
        return pd.read_csv(p, usecols=[c for c in cols if c in pd.read_csv(p, nrows=0).columns])


    def load_ppd_repr(split: str, horizon: int) -> pd.DataFrame:
        p = ROOT / "outputs" / "latent_export" / split / f"h{horizon:02d}_physiology_only_temporal.npz"
        z = np.load(p, allow_pickle=True)
        data = {
            "stay_id": z["stay_id"].astype(int),
            "subject_id_ppd_npz": z["subject_id"].astype(int),
            "hadm_id_ppd_npz": z["hadm_id"].astype(int),
            "cohort_arm": split,
            "horizon_hours": float(horizon),
            "landmark_eligible_ppd_npz": z["landmark_eligible"].astype(bool),
            "subject_disjoint_sensitivity_eligible_npz": z["subject_disjoint_sensitivity_eligible"].astype(bool),
        }
        probs = z["probs"].astype(float)
        label_names = [str(x) for x in z["label_names"]]
        for i, name in enumerate(label_names):
            data[f"npz_prob_{name}"] = probs[:, i]
        repr_arr = z["fused_repr"].astype(float)
        for j in range(repr_arr.shape[1]):
            data[f"ppd_repr_{j:03d}"] = repr_arr[:, j]
        return pd.DataFrame(data)


    def ppd_mbf_bridge(mbf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        merged_all = []
        for split in ["validation", "test"]:
            state = load_ppd_state(split)
            for horizon in [12, 24]:
                repr_df = load_ppd_repr(split, horizon)
                s = state[state["horizon_hours"].astype(int).eq(horizon)].copy()
                m = s.merge(repr_df, on=["stay_id", "horizon_hours"], how="left")
                m = m.merge(mbf, on="stay_id", how="inner", suffixes=("_ppd", ""))
                m["split"] = split
                merged_all.append(m)
        merged = pd.concat(merged_all, ignore_index=True)
        merged.to_csv(OUT / "mbf_ppd_overlap_analysis_rows.csv.gz", index=False, compression="gzip")

        state_summary = merged.groupby(["horizon_hours", "split", "ppd_state_id", "ppd_state"], dropna=False).agg(
            n=("stay_id", "size"),
            subjects=("subject_id", "nunique"),
            mbf_mean=("mbf_index", "mean"),
            mbf_sd=("mbf_index", "std"),
            hyperglycaemic_instability_burden=("post_mean_p_state_5", "mean"),
            renal_transition_burden=("renal_transition_burden", "mean"),
            ppd_state_probability_mean=("ppd_state_probability", "mean"),
        ).reset_index()

        corr_rows = []
        for horizon in [12, 24]:
            for split in ["validation", "test", "pooled"]:
                if split == "pooled":
                    q = merged[merged["horizon_hours"].astype(int).eq(horizon)].copy()
                else:
                    q = merged[merged["horizon_hours"].astype(int).eq(horizon) & merged["split"].eq(split)].copy()
                for feature in ["ppd_state_probability"] + PPD_PROB_COLS:
                    r, p, n = safe_corr(q["mbf_index"], q[feature])
                    corr_rows.append({
                        "horizon_hours": horizon,
                        "split": split,
                        "feature": feature,
                        "spearman_r_with_mbf": r,
                        "p_value": p,
                        "n": n,
                    })
        correlations = pd.DataFrame(corr_rows)

        perf_rows = []
        for horizon in [12, 24]:
            q = merged[merged["horizon_hours"].astype(int).eq(horizon) & merged["mbf_index"].notna()].copy()
            if len(q) < 100:
                continue
            repr_cols = [c for c in q.columns if c.startswith("ppd_repr_")]
            clinical_cols = [c for c in ["diabetes", "anchor_age", "charlson_index", "admission_sofa"] if c in q.columns]
            prob_cols = [c for c in PPD_PROB_COLS if c in q.columns]
            specs = [
                ("clinical_minimal", clinical_cols, []),
                ("ppd_task_probabilities_only", prob_cols + ["ppd_state_probability"], []),
                ("clinical_plus_ppd_state_probabilities", clinical_cols + prob_cols + ["ppd_state_probability"], ["ppd_state_id"]),
                ("ppd_h24_fused_embedding_only" if horizon == 24 else "ppd_h12_fused_embedding_only", repr_cols, []),
                ("clinical_plus_ppd_fused_embedding", clinical_cols + repr_cols, []),
            ]
            for model_name, num_cols, cat_cols in specs:
                if not num_cols and not cat_cols:
                    continue
                perf = crossfit_ridge(q, num_cols, cat_cols, "mbf_index", "subject_id", model_name)
                perf["horizon_hours"] = horizon
                perf_rows.append(perf)
        performance = pd.concat(perf_rows, ignore_index=True)

        # Compact source-data table for the bridge figure/text.
        h24_state = state_summary[state_summary["horizon_hours"].astype(int).eq(24)].copy()
        h24_corr = correlations[
            correlations["horizon_hours"].astype(int).eq(24)
            & correlations["split"].eq("pooled")
        ].copy()
        h24_perf = performance[performance["horizon_hours"].astype(int).eq(24)].copy()
        bridge_summary = pd.concat([
            h24_state.assign(result_family="state_summary"),
            h24_corr.assign(result_family="correlation"),
            h24_perf.assign(result_family="predictive_bridge"),
        ], ignore_index=True, sort=False)

        state_summary.to_csv(OUT / "mbf_ppd_state_summary.csv", index=False)
        correlations.to_csv(OUT / "mbf_ppd_probability_correlations.csv", index=False)
        performance.to_csv(OUT / "mbf_ppd_embedding_prediction_performance.csv", index=False)
        bridge_summary.to_csv(OUT / "mbf_ppd_bridge_summary.csv", index=False)

        return merged, state_summary, correlations, performance


    def crossfit_ridge(
        data: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
        target: str,
        group_col: str,
        model_name: str,
    ) -> pd.DataFrame:
        q = data[data[target].notna()].copy()
        y = numeric(q[target]).to_numpy(float)
        groups = q[group_col].to_numpy()
        pred = np.full(len(q), np.nan)
        transformer_parts = []
        if numeric_cols:
            transformer_parts.append((
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_cols,
            ))
        if categorical_cols:
            transformer_parts.append((
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_cols,
            ))
        pipe = Pipeline([
            ("pre", ColumnTransformer(transformer_parts, remainder="drop")),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ])
        n_splits = min(5, max(2, len(np.unique(groups))))
        splitter = GroupKFold(n_splits=n_splits)
        for tr, te in splitter.split(q, y, groups):
            pipe.fit(q.iloc[tr], y[tr])
            pred[te] = pipe.predict(q.iloc[te])
        valid = np.isfinite(pred)
        r2 = r2_score(y[valid], pred[valid])
        rmse = math.sqrt(mean_squared_error(y[valid], pred[valid]))
        mae = mean_absolute_error(y[valid], pred[valid])
        return pd.DataFrame([{
            "model": model_name,
            "target": target,
            "n": int(valid.sum()),
            "subjects": int(pd.Series(groups[valid]).nunique()),
            "R2": float(r2),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "numeric_features": ", ".join(numeric_cols),
            "categorical_features": ", ".join(categorical_cols),
        }])


    def zscore_residual(actual: pd.Series, expected: np.ndarray) -> pd.Series:
        residual = numeric(actual) - pd.Series(expected, index=actual.index)
        mu = residual.mean(skipna=True)
        sd = residual.std(skipna=True)
        return (residual - mu) / sd


    def alternative_mbf_definitions(mbf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        out = mbf.copy()
        definitions = [
            (
                "mbf_glucose_cv_index",
                "mbf_response_glucose_cv_12_24",
                "Residual 12-24 h glucose CV after early reserve/challenge adjustment",
            ),
            (
                "mbf_hyperglycaemia_burden_index",
                "mbf_response_hyper_severe_fraction_12_24",
                "Residual 12-24 h severe-hyperglycaemia burden after early reserve/challenge adjustment",
            ),
        ]
        metrics_all = []
        features = mbf_definition_features(out)
        for index_col, target, definition in definitions:
            pred, metrics = crossfit_regression(out, features, target, out["subject_id"], folds=5, max_iter=120, seed_offset=8300 + len(metrics_all))
            out[f"{index_col}_expected"] = pred
            out[index_col] = zscore_residual(out[target], pred)
            metrics["target"] = target
            metrics["alternative_mbf_index"] = index_col
            metrics["definition"] = definition
            metrics_all.append(metrics)
        metrics_df = pd.concat(metrics_all, ignore_index=True)

        effect_rows = []
        cov_base = [c for c in ["diabetes"] + base_renal_features(out) if c in out.columns]
        for index_col, target, definition in definitions:
            q = out[out[index_col].notna()].copy()
            r_primary, p_primary, n_corr = safe_corr(q["mbf_index"], q[index_col])
            effect_rows.append({
                "alternative_mbf_index": index_col,
                "analysis": "correlation_with_primary_mbf",
                "target": "mbf_index",
                "estimate": r_primary,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value": p_primary,
                "n": n_corr,
                "interpretation": "Spearman correlation with primary glucose-delta MBF index",
            })
            dm = linear_coef_bootstrap(
                q,
                index_col,
                [c for c in ["diabetes", "anchor_age", "charlson_index", "admission_sofa", "weight_kg"] if c in q.columns],
                "diabetes",
                n_boot=300,
                seed=SEED + 8400 + len(effect_rows),
            )
            effect_rows.append({
                "alternative_mbf_index": index_col,
                "analysis": "diabetes_adjusted_association",
                "target": index_col,
                "estimate": float(dm.estimate.iloc[0]),
                "ci_lower": float(dm.ci_lower.iloc[0]),
                "ci_upper": float(dm.ci_upper.iloc[0]),
                "p_value": np.nan,
                "n": int(dm.n.iloc[0]),
                "interpretation": "Positive coefficient indicates higher alternative MBF in diabetes after measured covariate adjustment",
            })
            for outcome, label in [
                ("post_mean_p_state_5", "hyperglycaemic_instability_burden"),
                ("renal_transition_burden", "renal_transition_burden"),
            ]:
                covars = [index_col] + [c for c in cov_base if c != index_col]
                coef = linear_coef_bootstrap(
                    q,
                    outcome,
                    covars,
                    index_col,
                    n_boot=300,
                    seed=SEED + 8500 + len(effect_rows),
                )
                effect_rows.append({
                    "alternative_mbf_index": index_col,
                    "analysis": f"association_with_{label}",
                    "target": outcome,
                    "estimate": float(coef.estimate.iloc[0]),
                    "ci_lower": float(coef.ci_lower.iloc[0]),
                    "ci_upper": float(coef.ci_upper.iloc[0]),
                    "p_value": np.nan,
                    "n": int(coef.n.iloc[0]),
                    "interpretation": f"Adjusted association between {index_col} and {label}",
                })
        effects_df = pd.DataFrame(effect_rows)

        keep_cols = [
            "stay_id", "subject_id", "diabetes", "mbf_index",
            "mbf_glucose_cv_index", "mbf_hyperglycaemia_burden_index",
            "post_mean_p_state_5", "renal_transition_burden",
        ]
        out[[c for c in keep_cols if c in out.columns]].to_csv(
            OUT / "mbf_alternative_definition_analysis_rows.csv.gz",
            index=False,
            compression="gzip",
        )
        metrics_df.to_csv(OUT / "mbf_alternative_definition_model_metrics.csv", index=False)
        effects_df.to_csv(OUT / "mbf_alternative_definition_effects.csv", index=False)
        return out, metrics_df, effects_df


    def timeline_source_data() -> pd.DataFrame:
        rows = [
            {
                "stage": "Baseline reserve",
                "start_hour": 0,
                "end_hour": 6,
                "role": "Pre-exposure reserve",
                "variables": "admission glucose, early glucose variability, creatinine/BUN, lactate, SOFA, vasopressor, ventilation, RRT",
                "leakage_boundary": "Uses only 0-6 h information",
            },
            {
                "stage": "Metabolic challenge",
                "start_hour": 0,
                "end_hour": 12,
                "role": "Early metabolic inputs and stress context",
                "variables": "dextrose, enteral/propofol energy, insulin, steroid, early severity context",
                "leakage_boundary": "Uses only 0-12 h information",
            },
            {
                "stage": "MBF index",
                "start_hour": 12,
                "end_hour": 24,
                "role": "Observed minus expected metabolic response",
                "variables": "12-24 h glucose response residual conditioned on early reserve/challenge",
                "leakage_boundary": "Computed before 24-48 h state window",
            },
            {
                "stage": "PPD-EHR linkage",
                "start_hour": 24,
                "end_hour": 24,
                "role": "AI-derived latent representation/state probability",
                "variables": "h24 physiology-only fused PPD-EHR representation and PPD state probabilities",
                "leakage_boundary": "Used for linkage to MBF; not a mediator claim",
            },
            {
                "stage": "Renal-metabolic trajectory",
                "start_hour": 24,
                "end_hour": 48,
                "role": "Dynamic organ vulnerability state",
                "variables": "clinical state probabilities and renal-transition burden",
                "leakage_boundary": "Occurs after MBF response window",
            },
            {
                "stage": "Strict CAM",
                "start_hour": 48,
                "end_hour": 72,
                "role": "Acute brain dysfunction outcome",
                "variables": "strict CAM assessment among eligible landmark rows",
                "leakage_boundary": "Outcome window begins after state-transition window",
            },
        ]
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "mbf_temporal_precedence_timeline_source.csv", index=False)
        df.to_csv(SRC / "Figure_6a_mbf_temporal_precedence_timeline.csv", index=False)
        return df


    def draw_timeline(df: pd.DataFrame) -> None:
        mpl.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        })
        colors = {
            "Baseline reserve": "#d9e8f5",
            "Metabolic challenge": "#fee8c8",
            "MBF index": "#fdbb84",
            "PPD-EHR linkage": "#d7b5d8",
            "Renal-metabolic trajectory": "#b3cde3",
            "Strict CAM": "#c7e9c0",
        }
        fig, ax = plt.subplots(figsize=(mm_to_in(183), mm_to_in(78)))
        y_positions = np.arange(len(df))[::-1]
        for i, (_, row) in enumerate(df.iterrows()):
            y = y_positions[i]
            start = float(row["start_hour"])
            end = float(row["end_hour"])
            width = max(end - start, 1.2)
            if end == start:
                ax.scatter([start], [y], s=80, color=colors[row["stage"]], edgecolor="#333333", zorder=3)
                ax.text(start + 1.0, y, row["stage"], va="center", ha="left", fontweight="bold")
            else:
                ax.barh(y, width, left=start, height=0.55, color=colors[row["stage"]], edgecolor="#333333", lw=0.7)
                ax.text(start + width / 2, y + 0.02, row["stage"], va="center", ha="center", fontweight="bold")
            ax.text(73.5, y, row["role"], va="center", ha="left", color="#4d4d4d", fontsize=6.2)

        for x in [6, 12, 24, 48, 72]:
            ax.axvline(x, color="#bdbdbd", lw=0.6, ls="--", zorder=0)

        ax.annotate("MBF precedes state-transition and CAM windows",
                    xy=(24, 2.8), xytext=(33, 4.1),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#202124"),
                    fontsize=6.5, ha="left")
        ax.set_xlim(-1, 104)
        ax.set_ylim(-1.1, len(df))
        ax.set_yticks([])
        ax.set_xticks([0, 6, 12, 24, 48, 72])
        ax.set_xticklabels(["0", "6", "12", "24", "48", "72"])
        ax.set_xlabel("Hours after ICU admission", labelpad=8)
        ax.set_title("Temporal ordering of MBF, latent trajectory and brain-outcome windows", loc="left", fontsize=8.5, pad=6)
        fig.text(
            0.055, 0.035,
            "Design boundary: reserve/challenge/response windows end before 24-48 h renal-metabolic state estimation; CAM is assessed at 48-72 h.",
            fontsize=6.0, color="#4d4d4d", ha="left", va="bottom",
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        base = FIG / "Supplementary_Figure_MBF_temporal_precedence_timeline"
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
        plt.close(fig)


    def write_manuscript_addendum(
        state_summary: pd.DataFrame,
        correlations: pd.DataFrame,
        performance: pd.DataFrame,
        alt_metrics: pd.DataFrame,
        alt_effects: pd.DataFrame,
    ) -> None:
        h24_perf = performance[performance["horizon_hours"].astype(int).eq(24)].copy()
        h24_corr = correlations[
            correlations["horizon_hours"].astype(int).eq(24)
            & correlations["split"].eq("pooled")
        ].copy()
        best_corr = h24_corr.iloc[h24_corr["spearman_r_with_mbf"].abs().fillna(-1).argmax()]
        emb = h24_perf[h24_perf["model"].eq("ppd_h24_fused_embedding_only")].iloc[0]
        clin = h24_perf[h24_perf["model"].eq("clinical_minimal")].iloc[0]
        clin_emb = h24_perf[h24_perf["model"].eq("clinical_plus_ppd_fused_embedding")].iloc[0]
        cv_eff = alt_effects[
            alt_effects["alternative_mbf_index"].eq("mbf_glucose_cv_index")
            & alt_effects["analysis"].eq("association_with_hyperglycaemic_instability_burden")
        ].iloc[0]
        hyp_eff = alt_effects[
            alt_effects["alternative_mbf_index"].eq("mbf_hyperglycaemia_burden_index")
            & alt_effects["analysis"].eq("association_with_hyperglycaemic_instability_burden")
        ].iloc[0]
        cv_dm = alt_effects[
            alt_effects["alternative_mbf_index"].eq("mbf_glucose_cv_index")
            & alt_effects["analysis"].eq("diabetes_adjusted_association")
        ].iloc[0]
        hyp_dm = alt_effects[
            alt_effects["alternative_mbf_index"].eq("mbf_hyperglycaemia_burden_index")
            & alt_effects["analysis"].eq("diabetes_adjusted_association")
        ].iloc[0]
        text = f"""# MBF closure addendum: PPD linkage, temporal precedence and alternative definitions

    Date: 2026-07-13

    ## One-sentence argument

    AI-derived PPD-EHR trajectories showed only weak direct overlap with the clinically reconstructed MBF phenotype, while the time-window design placed MBF before the 24–48 h renal–metabolic trajectory and 48–72 h CAM outcome; alternative response definitions supported the glycaemic-instability component but did not convert MBF into a confirmed renal–brain mediator.

    ## Results text

    ### PPD-EHR latent discovery was not reducible to MBF

    We next asked whether the original PPD-EHR latent discovery contained the clinically measurable metabolic buffering failure (MBF) phenotype. In the overlap between fixed-time PPD-EHR representations and the MBF landmark cohort, the h24 physiology-only fused representation did not materially predict the MBF index by cross-validation (R2 = {emb.R2:.3f}, n = {int(emb.n):,}). A minimal clinical model containing diabetes, age, Charlson index and admission SOFA achieved R2 = {clin.R2:.3f}, and adding the PPD-EHR fused representation to these clinical variables yielded only a small cross-validated R2 = {clin_emb.R2:.3f}. At the probability level, the strongest h24 PPD-EHR probability correlation with MBF was {best_corr.feature} (Spearman r = {best_corr.spearman_r_with_mbf:.3f}, n = {int(best_corr.n):,}). These findings argue against treating MBF as a direct surrogate of the PPD-EHR state labels or embedding. Instead, MBF should be positioned as a clinically reconstructed metabolic-resilience phenotype that complements the broader latent dynamics captured by PPD-EHR.

    ### MBF was temporally positioned before the vulnerability-state and outcome windows

    The final timeline analysis formalized the temporal ordering of the extension. Baseline reserve was measured over 0–6 h, metabolic challenge over 0–12 h, and the MBF index was derived from the 12–24 h glucose response conditional on earlier reserve and challenge variables. The renal–metabolic vulnerability trajectory was then reconstructed over 24–48 h, followed by strict CAM assessment over 48–72 h. This design supports a temporal-precedence statement—MBF was measured before the state-transition and brain-outcome windows—but does not by itself establish causal mediation.

    ### Alternative MBF definitions supported the glycaemic-instability axis

    To test whether the MBF result depended on the original glucose-delta response definition, we rebuilt the residual phenotype using two alternative 12–24 h response targets. The glucose-CV residual MBF definition was higher in diabetes after adjustment (beta = {cv_dm.estimate:.3f}, 95% CI {cv_dm.ci_lower:.3f} to {cv_dm.ci_upper:.3f}) and showed an adjusted association with hyperglycaemic-instability burden (beta = {cv_eff.estimate:.5f}, 95% CI {cv_eff.ci_lower:.5f} to {cv_eff.ci_upper:.5f}). The severe-hyperglycaemia-burden residual definition was also higher in diabetes (beta = {hyp_dm.estimate:.3f}, 95% CI {hyp_dm.ci_lower:.3f} to {hyp_dm.ci_upper:.3f}) and was strongly associated with subsequent hyperglycaemic-instability burden (beta = {hyp_eff.estimate:.5f}, 95% CI {hyp_eff.ci_lower:.5f} to {hyp_eff.ci_upper:.5f}). Thus, the glycaemic-instability component of the MBF story was robust to response redefinition, whereas renal-transition effects remained weak or directionally negative for these alternative definitions.

    ## Methods text

    PPD-EHR linkage analysis used h12 and h24 physiology-only fixed-time representations exported from the seed42 model. PPD-EHR state assignments, assigned-state probabilities and task probabilities were merged to MBF landmark rows by `stay_id`. We summarized MBF by PPD-EHR state and calculated Spearman correlations between MBF and PPD-EHR probability outputs. To test whether the PPD-EHR representation contained MBF-related information, we fit subject-grouped cross-validated ridge models predicting MBF from clinical variables, PPD-EHR task/state probabilities, the 128-dimensional fused representation, or clinical variables plus the fused representation.

    Alternative MBF definitions used the same early reserve/challenge covariates as the primary MBF model but changed the 12–24 h response target to glucose coefficient of variation or severe-hyperglycaemia burden. Each alternative MBF index was defined as a standardized cross-fitted residual. We then tested adjusted diabetes associations and adjusted associations with 24–48 h hyperglycaemic-instability and renal–metabolic trajectory burdens using patient-cluster bootstrap intervals.

    ## Discussion boundary

    The closure analyses strengthen the timing and robustness of the MBF construct, but they also set an important boundary for the AI-to-clinical bridge. PPD-EHR latent discovery was not reducible to MBF and should be presented as a broader discovery engine rather than as a direct MBF estimator. The defensible final language is that impaired metabolic buffering capacity characterized diabetes-associated hyperglycaemic-instability trajectories, whereas renal–metabolic vulnerability remained the downstream brain-risk state. MBF should not be described as a confirmed mediator from diabetes to renal–metabolic vulnerability to CAM.

    ## 中文要点

    - PPD-EHR 和 MBF 的直接连接很弱：h24 PPD embedding 基本不能预测 MBF，概率相关也很小；这说明两者不是同一个东西。
    - 时间顺序已经干净：0–6 h reserve，0–12 h challenge，12–24 h MBF response，24–48 h renal–metabolic state，48–72 h CAM。
    - 替代定义支持“glycaemic-instability axis”：glucose-CV residual 和 severe-hyperglycaemia residual 都显示糖尿病患者更高，并且都预测后续高血糖不稳定。
    - 不能写成正式 mediation；应写成 PPD-EHR 发现更宽的 latent dynamics，MBF 是独立重建的 metabolic-resilience phenotype。
    """
        (MAN / "MBF_CLOSURE_ADDENDUM_20260713.md").write_text(text, encoding="utf-8")


    def write_qa(
        performance: pd.DataFrame,
        correlations: pd.DataFrame,
        alt_effects: pd.DataFrame,
    ) -> None:
        rows = []
        h24_perf = performance[performance["horizon_hours"].astype(int).eq(24)]
        emb_r2 = float(h24_perf[h24_perf["model"].eq("ppd_h24_fused_embedding_only")]["R2"].iloc[0])
        clin_emb_r2 = float(h24_perf[h24_perf["model"].eq("clinical_plus_ppd_fused_embedding")]["R2"].iloc[0])
        rows.append({
            "gate": "ppd_embedding_contains_mbf_information",
            "status": "pass" if emb_r2 > 0.02 else "partial",
            "evidence": f"h24 PPD fused embedding predicts MBF with cross-fitted R2={emb_r2:.3f}",
            "interpretation": (
                "Direct PPD embedding-to-MBF prediction is weak; MBF should not be treated as a direct surrogate of the PPD latent state."
                if emb_r2 <= 0.02 else
                "PPD latent representation contains measurable MBF-related information."
            ),
        })
        rows.append({
            "gate": "ppd_clinical_bridge_increment",
            "status": "pass" if clin_emb_r2 > 0.02 else "partial",
            "evidence": f"clinical + PPD fused embedding R2={clin_emb_r2:.3f}",
            "interpretation": (
                "The AI-to-clinical bridge is weak for MBF specifically; PPD likely captures broader latent dynamics beyond MBF."
                if clin_emb_r2 <= 0.02 else
                "PPD representation can be used as an AI-to-clinical bridge, not as a direct duplicate of MBF."
            ),
        })
        h24_corr = correlations[
            correlations["horizon_hours"].astype(int).eq(24)
            & correlations["split"].eq("pooled")
        ]
        max_abs_corr = float(h24_corr["spearman_r_with_mbf"].abs().max())
        rows.append({
            "gate": "ppd_probability_mbf_correlation",
            "status": "pass" if max_abs_corr > 0.10 else "partial",
            "evidence": f"maximum absolute h24 probability Spearman correlation with MBF={max_abs_corr:.3f}",
            "interpretation": (
                "PPD probability outputs show only weak MBF overlap."
                if max_abs_corr <= 0.10 else
                "PPD probability outputs carry measurable MBF information."
            ),
        })
        rows.append({
            "gate": "temporal_precedence",
            "status": "pass",
            "evidence": "0-6h reserve; 0-12h challenge; 12-24h MBF response; 24-48h renal-metabolic state; 48-72h CAM.",
            "interpretation": "MBF temporally precedes state and outcome windows.",
        })
        for index_col in ["mbf_glucose_cv_index", "mbf_hyperglycaemia_burden_index"]:
            dm = alt_effects[
                alt_effects["alternative_mbf_index"].eq(index_col)
                & alt_effects["analysis"].eq("diabetes_adjusted_association")
            ].iloc[0]
            hyper = alt_effects[
                alt_effects["alternative_mbf_index"].eq(index_col)
                & alt_effects["analysis"].eq("association_with_hyperglycaemic_instability_burden")
            ].iloc[0]
            rows.append({
                "gate": f"alternative_definition_diabetes_{index_col}",
                "status": "pass" if dm.ci_lower > 0 else "partial",
                "evidence": f"diabetes coefficient={dm.estimate:.3f}, 95% CI {dm.ci_lower:.3f} to {dm.ci_upper:.3f}",
                "interpretation": "Alternative definition is diabetes-enriched." if dm.ci_lower > 0 else "Alternative definition does not show robust diabetes enrichment.",
            })
            rows.append({
                "gate": f"alternative_definition_hyperglycaemic_axis_{index_col}",
                "status": "pass" if hyper.ci_lower > 0 else "partial",
                "evidence": f"hyperglycaemic-instability coefficient={hyper.estimate:.5f}, 95% CI {hyper.ci_lower:.5f} to {hyper.ci_upper:.5f}",
                "interpretation": "Alternative definition supports the glycaemic-instability axis." if hyper.ci_lower > 0 else "Alternative definition gives weak hyperglycaemic-instability support.",
            })
        qa = pd.DataFrame(rows)
        qa.to_csv(QA / "MBF_CLOSURE_QA_20260713.csv", index=False)
        summary = {
            "date": "2026-07-13",
            "scope": "Final MBF closure checks: PPD linkage, temporal precedence, alternative definitions",
            "status_counts": {str(k): int(v) for k, v in qa["status"].value_counts().items()},
            "overall_interpretation": (
                "PPD-EHR is not reducible to MBF, MBF temporally precedes 24-48 h state and 48-72 h CAM windows, "
                "and alternative response definitions support the glycaemic-instability axis with boundaries."
            ),
            "not_supported": [
                "Formal MBF mediation from diabetes through renal-metabolic trajectory to CAM",
                "Claim that PPD-EHR embedding directly estimates MBF",
            ],
        }
        (QA / "MBF_CLOSURE_QA_20260713.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        md = "# MBF closure QA — 2026-07-13\n\n" + qa.to_markdown(index=False) + "\n"
        (QA / "MBF_CLOSURE_QA_20260713.md").write_text(md, encoding="utf-8")


    def copy_source_data() -> None:
        copies = {
            OUT / "mbf_ppd_state_summary.csv": SRC / "Supplement_mbf_ppd_state_summary.csv",
            OUT / "mbf_ppd_probability_correlations.csv": SRC / "Supplement_mbf_ppd_probability_correlations.csv",
            OUT / "mbf_ppd_embedding_prediction_performance.csv": SRC / "Supplement_mbf_ppd_embedding_prediction_performance.csv",
            OUT / "mbf_ppd_bridge_summary.csv": SRC / "Supplement_mbf_ppd_bridge_summary.csv",
            OUT / "mbf_alternative_definition_model_metrics.csv": SRC / "Supplement_mbf_alternative_definition_model_metrics.csv",
            OUT / "mbf_alternative_definition_effects.csv": SRC / "Supplement_mbf_alternative_definition_effects.csv",
        }
        for src, dst in copies.items():
            pd.read_csv(src).to_csv(dst, index=False)


    def json_safe(obj):
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_safe(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj


    def write_summary(performance: pd.DataFrame, correlations: pd.DataFrame, alt_effects: pd.DataFrame) -> None:
        h24_perf = performance[performance["horizon_hours"].astype(int).eq(24)]
        h24_corr = correlations[
            correlations["horizon_hours"].astype(int).eq(24)
            & correlations["split"].eq("pooled")
        ]
        summary = {
            "date": "2026-07-13",
            "purpose": "Close final MBF reviewer-risk gaps: PPD linkage, temporal precedence, and alternative response definitions.",
            "ppd_mbf_bridge": h24_perf[["model", "n", "R2", "RMSE", "MAE"]].to_dict(orient="records"),
            "ppd_probability_correlations_h24_pooled": h24_corr[["feature", "spearman_r_with_mbf", "p_value", "n"]].to_dict(orient="records"),
            "alternative_definition_effects": alt_effects.to_dict(orient="records"),
            "timeline": "Baseline reserve 0-6h; challenge 0-12h; MBF response 12-24h; renal-metabolic state 24-48h; CAM 48-72h.",
            "recommended_claim": (
                "AI-derived PPD-EHR latent discovery captures broader dynamics that are not reducible to MBF; "
                "MBF temporally precedes downstream state/outcome windows and robustly supports the glycaemic-instability axis."
            ),
            "boundary": "No formal mediation claim from MBF to renal-metabolic trajectory to CAM.",
        }
        (OUT / "mbf_closure_summary.json").write_text(
            json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )


    def main() -> None:
        mbf = load_mbf_rows()
        merged, state_summary, correlations, performance = ppd_mbf_bridge(mbf)
        alt_rows, alt_metrics, alt_effects = alternative_mbf_definitions(mbf)
        timeline = timeline_source_data()
        draw_timeline(timeline)
        write_manuscript_addendum(state_summary, correlations, performance, alt_metrics, alt_effects)
        write_qa(performance, correlations, alt_effects)
        copy_source_data()
        write_summary(performance, correlations, alt_effects)
        print("MBF closure complete")
        print("PPD overlap rows:", len(merged))
        print("outputs:", OUT)

    return SimpleNamespace(**locals())

_temporal = _load_temporal()


STAGES = {
    "core": _core.main,
    "reliability": _reliability.main,
    "extension": _extension.main,
    "temporal-robustness": _temporal.main,
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Metabolic-buffering-failure and robustness analyses')
    parser.add_argument("stage", choices=STAGES)
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    stage = sys.argv[1]
    if stage not in STAGES:
        parser.error(f"invalid stage: {stage}")
    if stage == "temporal-robustness" and any(arg in {"-h", "--help"} for arg in sys.argv[2:]):
        print("temporal-robustness: runs the prespecified PPD temporal-robustness analysis (no extra options).")
        return
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    STAGES[stage]()


if __name__ == "__main__":
    main()
