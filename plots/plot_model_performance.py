"""Model-data and performance figures."""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace


# ==============================================================================
# Figure 1: observation density
# ==============================================================================

def _load_observation():
    """Prepare all Figure 1 observation-density and availability source tables.

    This consolidates the three development scripts previously used for hourly
    event density, measurement missingness, and the 61-feature recorded-rate map.
    """
    import argparse
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import pandas as pd


    ROOT = Path(__file__).resolve().parents[2]
    DEFAULT_INPUT = ROOT / "data" / "restricted" / "interim"
    DEFAULT_OUTPUT = ROOT / "data" / "derived" / "figure1_observation_density"
    CHUNK_SIZE = 250_000

    MEASUREMENTS = [
        "sofa", "rass", "resp_rate", "heart_rate", "sbp", "dbp", "map", "spo2", "temperature",
        "lactate", "albumin", "prealbumin", "bun", "creatinine", "wbc", "hemoglobin", "platelet",
        "sodium", "potassium", "chloride", "bicarbonate", "pao2", "pco2", "ph", "magnesium",
        "phosphate", "ast", "alt", "bilirubin", "crp", "triglycerides", "glucose_mean", "glucose_min",
        "glucose_max", "glucose_std", "glucose_cv", "tir_fraction", "hypo_fraction", "hyper_severe_fraction",
    ]
    EVENTS = [
        "vasopressor_sum", "mv_flag", "rrt_flag", "fio2", "steroid_flag", "propofol_rate_sum",
        "midazolam_rate_sum", "dexmedetomidine_rate_sum", "fentanyl_rate_sum", "morphine_rate_sum",
        "hydromorphone_rate_sum", "vancomycin_rate_sum", "piperacillin_tazo_rate_sum", "meropenem_rate_sum",
        "cefepime_rate_sum", "metronidazole_rate_sum", "cisatracurium_rate_sum", "furosemide_rate_sum",
        "heparin_rate_sum", "parenteral_nutrition_rate_sum", "dextrose_rate_sum", "antibiotic_any",
    ]
    CONTINUOUS_EVENTS = [
        "heart_rate", "resp_rate", "spo2", "map", "temperature", "rass", "fio2", "lactate", "albumin",
        "potassium", "sodium", "creatinine", "bun", "bicarbonate", "glucose_lab", "hemoglobin", "platelet",
        "wbc", "insulin_input", "dextrose", "vasopressor", "propofol", "fentanyl", "dexmedetomidine",
        "enteral_calories", "parenteral_nutrition",
    ]


    def summarise_event_stream(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        counts: dict[tuple[str, int], int] = defaultdict(int)
        stay_ids: set[int] = set()
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
            chunk = chunk.loc[
                chunk["hours_from_icu"].between(0, 72, inclusive="left")
                & chunk["variable_name"].isin(CONTINUOUS_EVENTS)
            ].copy()
            chunk["hour"] = np.floor(chunk["hours_from_icu"]).astype(int)
            stay_ids.update(chunk["stay_id"].unique())
            for (feature, hour), value in chunk.groupby(["variable_name", "hour"], sort=False).size().items():
                counts[(feature, int(hour))] += int(value)

        frame = pd.DataFrame(0.0, index=CONTINUOUS_EVENTS, columns=range(72))
        denominator = max(len(stay_ids), 1)
        for (feature, hour), value in counts.items():
            frame.loc[feature, hour] = value / denominator
        frame.index.name = "feature"
        frame.columns = [f"{hour}-{hour + 1} h" for hour in frame.columns]
        summary = pd.DataFrame({
            "feature": frame.index,
            "mean_events_per_stay_hour": frame.mean(axis=1).to_numpy(),
            "event_density": frame.mean(axis=1).to_numpy(),
        })
        return frame, summary


    def summarise_master_cohort(path: Path, cohort: str, sample_size: int, seed: int):
        header = set(pd.read_csv(path, nrows=0).columns)
        aliases = {feature: feature for feature in [*MEASUREMENTS, *EVENTS]}
        if cohort == "eicu":
            aliases.update({
                "parenteral_nutrition_rate_sum": "parenteral_nutrition_kcal",
                "dextrose_rate_sum": "dextrose_kcal",
            })
        sources = sorted({source for source in aliases.values() if source in header})
        totals = pd.Series(0, index=range(12), dtype="int64")
        recorded = pd.DataFrame(0, index=range(12), columns=[*MEASUREMENTS, *EVENTS], dtype="int64")
        rng = np.random.default_rng(seed)
        sample = pd.DataFrame(columns=[*MEASUREMENTS, "_sample_key"])

        for chunk in pd.read_csv(path, usecols=["bin", *sources], chunksize=CHUNK_SIZE, low_memory=False):
            chunk = chunk[chunk["bin"].between(0, 11)]
            totals += chunk.groupby("bin").size().reindex(range(12), fill_value=0).astype("int64")
            for feature in MEASUREMENTS:
                source = aliases[feature]
                if source in chunk:
                    counts = chunk.groupby("bin")[source].count()
                    recorded.loc[counts.index, feature] += counts.astype("int64")
            for feature in EVENTS:
                source = aliases[feature]
                if source in chunk:
                    present = pd.to_numeric(chunk[source], errors="coerce").fillna(0).gt(0)
                    counts = chunk.assign(_recorded=present).groupby("bin")["_recorded"].sum()
                    recorded.loc[counts.index, feature] += counts.astype("int64")

            observed = pd.DataFrame(index=chunk.index)
            for feature in MEASUREMENTS:
                source = aliases[feature]
                observed[feature] = chunk[source].notna().astype("uint8") if source in chunk else 0
            observed["_sample_key"] = rng.random(len(observed))
            sample = pd.concat([sample, observed], ignore_index=True).nsmallest(sample_size, "_sample_key")

        rates = recorded.div(totals.replace(0, np.nan), axis=0).T
        rates.columns = [f"{start}-{start + 6} h" for start in range(0, 72, 6)]
        rates.index.name = "feature"
        missingness = 1 - rates.loc[MEASUREMENTS]
        denominators = pd.DataFrame({"bin": totals.index, "n_stay_bin_rows": totals.values})
        return missingness, rates, denominators, sample.drop(columns="_sample_key")


    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
        parser.add_argument("--sample-size", type=int, default=160_000)
        return parser.parse_args()


    def main() -> None:
        args = parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for offset, cohort in enumerate(("mimic", "eicu")):
            event_path = args.input_dir / f"{cohort}_continuous_time.csv"
            master_path = args.input_dir / f"{cohort}_master_cohort.csv"
            if not event_path.is_file() or not master_path.is_file():
                raise FileNotFoundError(f"Missing Figure 1 input for {cohort}: {event_path} or {master_path}")

            hourly, event_summary = summarise_event_stream(event_path)
            missingness, rates, denominators, sample = summarise_master_cohort(
                master_path, cohort, args.sample_size, 2026 + offset
            )
            hourly.to_csv(args.output_dir / f"ref02_{cohort}_hourly_event_density.csv")
            event_summary.to_csv(args.output_dir / f"ref02_{cohort}_event_density_summary.csv", index=False)
            missingness.to_csv(args.output_dir / f"ref02_{cohort}_actual_missingness.csv")
            denominators.to_csv(args.output_dir / f"ref02_{cohort}_actual_bin_denominators.csv", index=False)
            rates.to_csv(args.output_dir / f"ref02_{cohort}_61_feature_recorded_rate.csv")
            sample.to_csv(args.output_dir / f"ref03_{cohort}_actual_observed_sample.csv", index=False)
            print(f"{cohort}: Figure 1 source tables written to {args.output_dir}")

    return SimpleNamespace(**locals())

_observation = _load_observation()


# ==============================================================================
# Figure 2: performance and calibration
# ==============================================================================

def _load_performance():
    import argparse
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.linear_model import LogisticRegression
    from torch.utils.data import DataLoader


    RELEASE_ROOT = Path(__file__).resolve().parents[2]
    RUNTIME_ROOT = RELEASE_ROOT / "code" / "training" / "runtime"
    DEFAULT_DATA_ROOT = RELEASE_ROOT / "data" / "restricted" / "mimic_v3_event_bundle"
    DEFAULT_CHECKPOINT = RELEASE_ROOT / "model" / "model_best.pt"
    DEFAULT_RAW_TEST = RELEASE_ROOT / "data" / "derived" / "figure2_performance_calibration" / "Figure3_locked_test_raw_probabilities.npy"
    DEFAULT_OUTPUT = RELEASE_ROOT / "outputs" / "locked_calibration_rerun"

    ENDPOINTS = ["Heart failure", "Renal failure", "Infection", "Pneumonia", "Cerebrovascular", "Diabetic foot"]


    def load_components():
        model_root = RUNTIME_ROOT / "paper_experiments_v2"
        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
        from models.data_utils import collate_events, load_event_data
        from models.enhanced_cdt import EnhancedCausalDigitalTwin
        return EnhancedCausalDigitalTwin, collate_events, load_event_data


    def split_data(data, seed: int = 42, train_fraction: float = 0.70, validation_fraction: float = 0.20):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(data))
        n_train = int(len(data) * train_fraction)
        n_validation = int(len(data) * validation_fraction)
        return (
            [data[i] for i in indices[:n_train]],
            [data[i] for i in indices[n_train:n_train + n_validation]],
            [data[i] for i in indices[n_train + n_validation:]],
        )


    def build_model(model_class):
        return model_class(
            n_outputs=6,
            d_model=128,
            n_buckets=512,
            n_channels=3,
            n_transformer_layers=3,
            dropout=0.25,
            use_diffusion=True,
            use_gnn=True,
            static_dim=18,
            note_dim=768,
        )


    def predict(model, data, args, device, collate_events):
        loader = DataLoader(
            data,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=lambda x: collate_events(x, max_events=512),
        )
        all_probs, all_labels = [], []
        model.eval()
        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader, start=1):
                batch = {
                    "events": torch.nan_to_num(batch["events"]),
                    "event_mask": torch.nan_to_num(batch["event_mask"]),
                    "static": torch.nan_to_num(batch["static"], posinf=500.0, neginf=-100.0),
                    "notes": torch.nan_to_num(batch["notes"], posinf=50.0, neginf=-50.0),
                    "labels": torch.nan_to_num(batch["labels"], posinf=1.0, neginf=0.0),
                }
                events = batch["events"].to(device, non_blocking=True)
                event_mask = batch["event_mask"].to(device, non_blocking=True)
                static = batch["static"].to(device, non_blocking=True)
                notes = batch["notes"].to(device, non_blocking=True)
                try:
                    output = model(events, event_mask, static=static, notes=notes)
                except TypeError:
                    output = model(events, event_mask, static, notes)
                if isinstance(output, dict):
                    logits = output.get("factual_logits", output.get("logits", output.get("pred")))
                elif isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"Non-finite logits in batch {batch_idx}")
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(batch["labels"].numpy())
        return np.vstack(all_labels).astype(int), np.vstack(all_probs).astype(float)


    def binned_ece(y, p, bins=15):
        edges = np.linspace(0.0, 1.0, bins + 1)
        which = np.digitize(p, edges[1:-1], right=True)
        ece = 0.0
        n = len(y)
        for bin_id in range(bins):
            mask = which == bin_id
            if np.any(mask):
                ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
        return ece


    def bootstrap_ece(y, p, n_bootstrap=1000, bins=15, seed=20260801):
        rng = np.random.default_rng(seed)
        n = len(y)
        values = np.empty((n_bootstrap, p.shape[1]), dtype=float)
        for i in range(n_bootstrap):
            sample = rng.integers(0, n, size=n)
            for j in range(p.shape[1]):
                values[i, j] = binned_ece(y[sample, j], p[sample, j], bins=bins)
        return values


    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Recreate locked-test calibration source data.")
        parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
        parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
        parser.add_argument("--stored-raw-test", type=Path, default=DEFAULT_RAW_TEST)
        parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        return parser.parse_args()


    def main():
        args = parse_args()
        output = args.output
        output.mkdir(parents=True, exist_ok=True)
        model_class, collate_events, load_event_data = load_components()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading MIMIC bundle from {args.data_root}")
        data = load_event_data(args.data_root, limit=None, verbose=True)
        if not data:
            raise RuntimeError(f"No event shards found in {args.data_root}")
        train, val, test = split_data(data)
        print(f"Split sizes: train={len(train)}, validation={len(val)}, test={len(test)}")

        model = build_model(model_class).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        args = type("Args", (), {"max_events": 512, "val_batch_size": 128, "num_workers": 0})()
        y_val, raw_val = predict(model, val, args, device, collate_events)
        y_test, raw_test = predict(model, test, args, device, collate_events)

        stored_raw_test = np.load(args.stored_raw_test).astype(float)
        if stored_raw_test.shape != raw_test.shape:
            raise RuntimeError(f"Stored raw test shape {stored_raw_test.shape} != re-inferred {raw_test.shape}")
        raw_max_abs_diff = float(np.max(np.abs(stored_raw_test - raw_test)))
        print(f"Raw test comparison: max_abs_diff={raw_max_abs_diff:.8g}")
        if raw_max_abs_diff > 1e-4:
            raise RuntimeError("Re-inferred raw test probabilities do not match the locked raw source")

        calibrators, platt_test = [], np.zeros_like(raw_test)
        coefficient_rows = []
        for j, endpoint in enumerate(ENDPOINTS):
            calibrator = LogisticRegression(C=1e6, max_iter=400, solver="lbfgs")
            calibrator.fit(raw_val[:, j].reshape(-1, 1), y_val[:, j])
            platt_test[:, j] = calibrator.predict_proba(raw_test[:, j].reshape(-1, 1))[:, 1]
            calibrators.append(calibrator)
            coefficient_rows.append(
                {
                    "endpoint": endpoint,
                    "intercept": float(calibrator.intercept_[0]),
                    "slope": float(calibrator.coef_[0, 0]),
                    "n_validation": int(len(y_val)),
                    "fit_protocol": "Platt logistic calibration on full seed42 validation split",
                }
            )

        raw_ece = np.array([binned_ece(y_test[:, j], raw_test[:, j]) for j in range(6)])
        platt_ece = np.array([binned_ece(y_test[:, j], platt_test[:, j]) for j in range(6)])
        raw_bootstrap = bootstrap_ece(y_test, raw_test, seed=20260801)
        platt_bootstrap = bootstrap_ece(y_test, platt_test, seed=20260802)
        summary = pd.DataFrame(
            {
                "endpoint": ENDPOINTS,
                "raw_ece": raw_ece,
                "platt_ece": platt_ece,
                "n_validation_fit": len(y_val),
                "n_locked_test": len(y_test),
                "n_bins": 15,
                "n_bootstrap": 1000,
            }
        )
        bootstrap_rows = []
        for method, matrix in (("Raw", raw_bootstrap), ("Platt", platt_bootstrap)):
            for j, endpoint in enumerate(ENDPOINTS):
                for i, value in enumerate(matrix[:, j], start=1):
                    bootstrap_rows.append(
                        {
                            "endpoint": endpoint,
                            "method": method,
                            "bootstrap": i,
                            "ece": float(value),
                            "n_patients": len(y_test),
                            "n_bins": 15,
                        }
                    )

        np.save(output / "Figure3_locked_test_raw_probabilities.npy", raw_test)
        np.save(output / "Figure3_locked_test_labels.npy", y_test)
        np.save(output / "Figure3_locked_test_platt_probabilities.npy", platt_test)
        pd.DataFrame(raw_test, columns=ENDPOINTS).to_csv(output / "Figure3_locked_test_raw_probabilities.csv", index=False)
        pd.DataFrame(platt_test, columns=ENDPOINTS).to_csv(output / "Figure3_locked_test_platt_probabilities.csv", index=False)
        pd.DataFrame(coefficient_rows).to_csv(output / "Figure3_platt_calibration_coefficients.csv", index=False)
        summary.to_csv(output / "Figure3_locked_test_calibration_summary.csv", index=False)
        pd.DataFrame(bootstrap_rows).to_csv(output / "Figure3_locked_test_calibration_ece_bootstrap.csv", index=False)
        metadata = {
            "checkpoint": str(args.checkpoint),
            "data_bundle": str(args.data_root),
            "seed": 42,
            "split": {"train": len(train), "validation": len(val), "test": len(test), "fractions": "70/20/10"},
            "device": str(device),
            "raw_source_comparison": {"source": str(args.stored_raw_test), "max_abs_diff": raw_max_abs_diff},
            "macro_ece": {"raw": float(raw_ece.mean()), "platt": float(platt_ece.mean())},
            "protocol": "Fit one logistic Platt calibrator per endpoint on the full seed42 validation split; apply to the locked seed42 test split.",
        }
        (output / "Figure3_locked_test_calibration_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
        print(summary.to_string(index=False))

    return SimpleNamespace(**locals())

_performance = _load_performance()


STAGES = {
    "observation-density": _observation.main,
    "performance-calibration": _performance.main,
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Model-data and performance figures')
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
