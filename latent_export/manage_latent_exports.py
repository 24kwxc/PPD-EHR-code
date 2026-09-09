from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


HORIZONS = [6, 12, 24, 48]
ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate or finalize seed-42 fixed-time latent exports.")
    p.add_argument("action", nargs="?", choices=("validate", "finalize"), default="validate")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs_seed42_721")
    p.add_argument("--discovery-dir", type=Path, default=ROOT / "outputs_seed42_721" / "validation")
    p.add_argument("--confirmation-dir", type=Path, default=ROOT / "outputs_seed42_721" / "test")
    p.add_argument("--output", type=Path, default=ROOT / "outputs_seed42_721" / "latent_input_preflight.json")
    return p.parse_args()


def inspect_arm(path: Path, arm: str) -> tuple[dict, pd.DataFrame]:
    reference_ids: pd.DataFrame | None = None
    rows = []
    for h in HORIZONS:
        file = path / f"h{h:02d}_physiology_only_temporal.npz"
        if not file.exists():
            raise FileNotFoundError(f"Missing {arm} export: {file}")
        a = np.load(file, allow_pickle=True)
        required = {
            "stay_id", "subject_id", "hadm_id", "landmark_eligible",
            "icu_offset_seconds",
            "subject_disjoint_sensitivity_eligible", "hadm_disjoint_sensitivity_eligible",
            "temporal_mean_repr", "labels", "probs",
        }
        missing = sorted(required - set(a.files))
        if missing:
            raise RuntimeError(f"{file} lacks arrays: {missing}")
        stay = a["stay_id"].astype(int)
        subject = a["subject_id"].astype(int)
        hadm = a["hadm_id"].astype(int)
        if len(np.unique(stay)) != len(stay):
            raise RuntimeError(f"Duplicate stay_id values in {file}")
        current_ids = pd.DataFrame({
            "stay_id": stay, "subject_id": subject, "hadm_id": hadm,
            "subject_disjoint_sensitivity_eligible": a["subject_disjoint_sensitivity_eligible"].astype(bool),
            "hadm_disjoint_sensitivity_eligible": a["hadm_disjoint_sensitivity_eligible"].astype(bool),
        })
        if reference_ids is None:
            reference_ids = current_ids
        elif not reference_ids.equals(current_ids):
            raise RuntimeError(f"Stay/subject/admission order differs across horizons in {arm}: {file}")
        reported_eligible = a["landmark_eligible"].astype(bool)
        valid_icu_anchor = np.isfinite(a["icu_offset_seconds"].astype(float)) & (a["icu_offset_seconds"].astype(float) >= 0)
        eligible = reported_eligible & valid_icu_anchor
        rows.append({
            "horizon_hours": h,
            "exported_stays": len(stay),
            "reported_landmark_eligible_stays": int(reported_eligible.sum()),
            "invalid_icu_anchor_stays": int((~valid_icu_anchor).sum()),
            "effective_landmark_eligible_stays": int(eligible.sum()),
            "embedding_dimensions": int(a["temporal_mean_repr"].shape[1]),
        })
    assert reference_ids is not None
    audit_path = path / "run_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    time_origin = audit.get("landmark_time_origin", audit.get("time_origin", "not_reported"))
    return {
        "arm": arm, "directory": str(path), "horizons": rows,
        "landmark_time_origin": time_origin,
        "time_origin_verification": audit.get(
            "time_origin_verification", audit.get("time_origin_caveat", "not_reported")
        ),
    }, reference_ids


def validate_exports(args: argparse.Namespace) -> None:
    discovery, discovery_ids = inspect_arm(args.discovery_dir, "validation_discovery")
    confirmation, confirmation_ids = inspect_arm(args.confirmation_dir, "test_confirmation")
    stay_overlap = set(discovery_ids.stay_id) & set(confirmation_ids.stay_id)
    subject_overlap = set(discovery_ids.subject_id) & set(confirmation_ids.subject_id)
    hadm_overlap = set(discovery_ids.hadm_id) & set(confirmation_ids.hadm_id)
    if stay_overlap:
        raise RuntimeError(f"Invalid split reconstruction: {len(stay_overlap)} overlapping stay_id values")

    subject_disjoint = confirmation_ids[confirmation_ids.subject_disjoint_sensitivity_eligible].copy()
    hadm_disjoint = confirmation_ids[confirmation_ids.hadm_disjoint_sensitivity_eligible].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subject_disjoint.to_csv(args.output.parent / "subject_disjoint_test_stay_ids.csv", index=False)
    hadm_disjoint.to_csv(args.output.parent / "hadm_disjoint_test_stay_ids.csv", index=False)
    report = {
        "protocol": "seed42_70_20_10",
        "validation_unique_stays": discovery_ids.stay_id.nunique(),
        "test_unique_stays": confirmation_ids.stay_id.nunique(),
        "stay_id_overlap": 0,
        "validation_unique_subjects": discovery_ids.subject_id.nunique(),
        "test_unique_subjects": confirmation_ids.subject_id.nunique(),
        "subject_id_overlap": len(subject_overlap),
        "test_stays_with_subject_seen_in_validation": int(confirmation_ids.subject_id.isin(subject_overlap).sum()),
        "test_stays_with_subject_seen_in_any_other_split_arm": int((~confirmation_ids.subject_disjoint_sensitivity_eligible).sum()),
        "subject_disjoint_test_stays": len(subject_disjoint),
        "subject_disjoint_test_fraction": len(subject_disjoint) / len(confirmation_ids),
        "validation_unique_hadm_ids": discovery_ids.hadm_id.nunique(),
        "test_unique_hadm_ids": confirmation_ids.hadm_id.nunique(),
        "hadm_id_overlap": len(hadm_overlap),
        "test_stays_with_hadm_seen_in_validation": int(confirmation_ids.hadm_id.isin(hadm_overlap).sum()),
        "test_stays_with_hadm_seen_in_any_other_split_arm": int((~confirmation_ids.hadm_disjoint_sensitivity_eligible).sum()),
        "hadm_disjoint_test_stays": len(hadm_disjoint),
        "hadm_disjoint_test_fraction": len(hadm_disjoint) / len(confirmation_ids),
        "interpretation": "Stay-level arms are disjoint. Validation-test subject/admission overlap is reported separately; sensitivity eligibility uses exporter flags against all other split arms, including training.",
        "validation": discovery,
        "test": confirmation,
        "status": "pass",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    aurocs, auprcs = [], []
    for endpoint in range(labels.shape[1]):
        target = (labels[:, endpoint] > 0.5).astype(int)
        if 0 < target.sum() < len(target):
            aurocs.append(float(roc_auc_score(target, probabilities[:, endpoint])))
            auprcs.append(float(average_precision_score(target, probabilities[:, endpoint])))
    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
    }


def finalize_valid_anchors(output_root: Path) -> None:
    """Apply the non-negative ICU-anchor rule without recomputing latents."""
    script_hash = _sha256(Path(__file__).resolve())
    for arm in ("validation", "test"):
        directory = output_root / arm
        ids_path = directory / f"{arm}_stay_ids_seed42.csv"
        ids = pd.read_csv(ids_path)
        ids["valid_icu_anchor"] = ids["icu_offset_seconds"].ge(0)
        ids.to_csv(ids_path, index=False)
        valid = ids["valid_icu_anchor"].to_numpy(bool)
        old_coverage = pd.read_csv(directory / "fixed_time_coverage_audit.csv").set_index("horizon_hours")
        performance_rows, coverage_rows = [], []

        for npz_path in sorted(directory.glob("h*_*.npz")):
            with np.load(npz_path, allow_pickle=False) as archive:
                payload = {key: archive[key] for key in archive.files}
            payload["valid_icu_anchor"] = valid
            payload["landmark_eligible"] = payload["landmark_eligible"].astype(bool) & valid
            np.savez_compressed(npz_path, **payload)

        for horizon in HORIZONS:
            horizon_files = sorted(directory.glob(f"h{horizon:02d}_*.npz"))
            with np.load(horizon_files[0], allow_pickle=False) as reference:
                los = reference["los_only_eligible"].astype(bool)
                observed = reference["observed_event_support"].astype(bool)
                primary = reference["landmark_eligible"].astype(bool)
            coverage = old_coverage.loc[float(horizon)].to_dict()
            coverage.update({
                "horizon_hours": float(horizon), "n_sequences": len(ids),
                "n_valid_icu_anchor": int(valid.sum()),
                "n_invalid_negative_icu_offset": int((~valid).sum()),
                "n_los_only_eligible_raw_icu_los_ge_horizon": int(los.sum()),
                "n_observed_event_supported": int(observed.sum()),
                "n_primary_landmark_eligible_valid_anchor_los_and_observed_event": int(primary.sum()),
            })
            coverage_rows.append(coverage)
            for npz_path in horizon_files:
                variant = npz_path.stem.split("_", 1)[1]
                with np.load(npz_path, allow_pickle=False) as archive:
                    labels, probabilities = archive["labels"], archive["probs"]
                    eligible = archive["landmark_eligible"].astype(bool)
                    subject_disjoint = eligible & archive["subject_disjoint_sensitivity_eligible"].astype(bool)
                for scope, mask in [
                    (f"all_{arm}", np.ones(len(ids), dtype=bool)),
                    ("landmark_eligible_valid_icu_anchor", eligible),
                    ("landmark_eligible_valid_anchor_subject_disjoint_sensitivity", subject_disjoint),
                ]:
                    performance_rows.append({
                        "horizon_hours": horizon, "variant": variant, "analysis_scope": scope,
                        "n": int(mask.sum()), **_metrics(labels[mask], probabilities[mask]),
                    })

        predictions_path = directory / "fixed_time_predictions.csv.gz"
        predictions = pd.read_csv(predictions_path)
        valid_map = ids.set_index("stay_id")["valid_icu_anchor"]
        predictions["valid_icu_anchor"] = predictions["stay_id"].map(valid_map).astype(bool)
        predictions["landmark_eligible"] = predictions["landmark_eligible"].astype(bool) & predictions["valid_icu_anchor"]
        predictions.to_csv(predictions_path, index=False)
        pd.DataFrame(performance_rows).to_csv(directory / "fixed_time_performance.csv", index=False)
        pd.DataFrame(coverage_rows).to_csv(directory / "fixed_time_coverage_audit.csv", index=False)

        audit_path = directory / "run_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["valid_icu_anchor_finalization"] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "finalizer": str(Path(__file__).resolve()), "finalizer_sha256": script_hash,
            "rule": "valid_icu_anchor = icu_offset_seconds >= 0",
            "invalid_negative_offset_rows": int((~valid).sum()), "valid_anchor_rows": int(valid.sum()),
            "latent_values_recomputed": False,
        }
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.action == "validate":
        validate_exports(args)
    else:
        finalize_valid_anchors(args.output_root)


if __name__ == "__main__":
    main()
