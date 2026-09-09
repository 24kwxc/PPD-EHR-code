#!/usr/bin/env python
"""Frozen-checkpoint, fixed-time latent export for formal seed-42 split arms.

This script is deliberately external to the original PPD-EHR/ECDT codebase. It
does not train or modify the checkpoint. Fixed horizons are defined from the
numeric bundle timestamps (empirically seconds) and all note content is removed.

Primary exports
---------------
1. temporal_only_no_text:
   all temporal events (vitals, labs, medications), static and notes masked.
2. static_temporal_no_text_los_masked:
   all temporal events plus baseline static variables, with full ICU LOS masked.
3. physiology_only_temporal:
   vitals and labs only; medications, static and notes masked.
4. physiology_plus_static_los_masked:
   vitals and labs plus baseline static variables, with medications, notes and
   full ICU LOS masked.

For every variant/horizon, the script exports predictions, the model's 128-D
fused representation, and two 128-D summaries of the pre-fusion temporal tokens.
The pre-fusion physiology-only representation is the preferred candidate latent
for intervention-to-state analyses because medication exposure is not encoded in
its input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


LABEL_NAMES = [
    "heart_failure",
    "renal_failure",
    "infection",
    "pneumonia",
    "cerebrovascular",
    "diabetic_foot",
]

VARIANTS = (
    "temporal_only_no_text",
    "static_temporal_no_text_los_masked",
    "physiology_only_temporal",
    "physiology_plus_static_los_masked",
)


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parents[2]
    data_root = package / "data"
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runtime-root",
        type=Path,
        default=package / "code" / "training" / "runtime",
        help="Directory containing the versioned ECDT training runtime.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=data_root / "restricted" / "mimic_v3_event_bundle",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=package / "model" / "model_best.pt",
    )
    p.add_argument(
        "--icustays",
        type=Path,
        default=data_root / "restricted" / "mimic_iv" / "icu" / "icustays.csv.gz",
    )
    p.add_argument(
        "--admissions",
        type=Path,
        default=data_root / "restricted" / "mimic_iv" / "hosp" / "admissions.csv.gz",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--cohort-arm",
        choices=("validation", "test"),
        default="test",
        help="Validation is for state discovery; test is for frozen confirmation.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--horizons", type=str, default="6,12,24,48")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-events", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--limit", type=int, default=0, help="Smoke-test only; 0 exports the complete stay-level cohort arm.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_formal_checkpoint_strict(model: torch.nn.Module, checkpoint: Path, device: str) -> Dict[str, int | str]:
    """Load the formal suite checkpoint without prefix guessing or partial loading."""
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError("Formal checkpoint must contain a model_state_dict payload")
    state = payload["model_state_dict"]
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: {incompatible}")
    return {
        "load_mode": "strict_model_state_dict",
        "loaded_tensors": len(state),
        "missing_tensors": len(incompatible.missing_keys),
        "unexpected_tensors": len(incompatible.unexpected_keys),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_best_val_auroc": float(payload.get("best_val_auroc", np.nan)),
        "checkpoint_config": json.dumps(payload.get("config", {}), sort_keys=True),
    }


def json_dump(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def load_icu_offsets(icustays_path: Path, admissions_path: Path) -> Tuple[pd.DataFrame, Dict]:
    icu = pd.read_csv(
        icustays_path,
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    )
    admissions = pd.read_csv(
        admissions_path,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
    )
    for col in ("intime", "outtime"):
        icu[col] = pd.to_datetime(icu[col], errors="raise")
    for col in ("admittime", "dischtime"):
        admissions[col] = pd.to_datetime(admissions[col], errors="raise")
    merged = icu.merge(
        admissions,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="many_to_one",
    )
    merged["icu_offset_seconds"] = (merged["intime"] - merged["admittime"]).dt.total_seconds()
    audit = {
        "icustays_rows": len(icu),
        "admissions_rows": len(admissions),
        "merged_rows": len(merged),
        "missing_offset_rows": int(merged["icu_offset_seconds"].isna().sum()),
        "negative_offset_rows": int((merged["icu_offset_seconds"] < 0).sum()),
    }
    return merged, audit


def sha256_stay_ids(samples: Sequence[Mapping]) -> str:
    ids = np.asarray([int(s.get("stay_id")) for s in samples], dtype="<i8")
    return hashlib.sha256(ids.tobytes()).hexdigest()


def clone_sample(sample: Mapping) -> Dict:
    out = dict(sample)
    for key in ("timestamps", "label", "static", "note", "note_stats"):
        if key in out:
            out[key] = np.asarray(out[key]).copy()
    for modality in ("vitals", "labs", "meds"):
        mod = sample.get(modality, {})
        out[modality] = {
            "t": list(mod.get("t", [])),
            "f": list(mod.get("f", [])),
            "v": list(mod.get("v", [])),
        }
    return out


def fixed_time_crop(
    sample: Mapping,
    horizon_hours: float,
    icu_offset_seconds: float,
) -> Tuple[Dict, Dict[str, int | float]]:
    """Crop hospital-relative timestamps to an ICU-relative landmark window."""
    out = clone_sample(sample)
    ts = np.asarray(sample.get("timestamps", []), dtype=np.float64)
    window_start = float(icu_offset_seconds)
    window_end = window_start + float(horizon_hours) * 3600.0
    start_idx = int(np.searchsorted(ts, window_start, side="left")) if ts.size else 0
    stop_idx = int(np.searchsorted(ts, window_end, side="right")) if ts.size else 0
    n_keep = max(stop_idx - start_idx, 0)
    out["timestamps"] = (ts[start_idx:stop_idx] - window_start).copy()

    audit: Dict[str, int | float] = {
        "n_timestamp_total": int(ts.size),
        "n_timestamp_kept": n_keep,
        "icu_offset_seconds": window_start,
        "window_end_seconds_from_hospital_admission": window_end,
        "n_events_total": 0,
        "n_events_kept": 0,
    }
    for modality in ("vitals", "labs", "meds"):
        mod = sample.get(modality, {})
        t = np.asarray(mod.get("t", []), dtype=np.int64)
        f = np.asarray(mod.get("f", []))
        v = np.asarray(mod.get("v", []))
        valid = (t >= start_idx) & (t < stop_idx)
        out[modality] = {
            "t": (t[valid] - start_idx).astype(np.int32).tolist(),
            "f": f[valid].tolist(),
            "v": v[valid].tolist(),
        }
        audit["n_events_total"] += int(t.size)
        audit["n_events_kept"] += int(valid.sum())
    return out, audit


def prepare_variant(samples: Sequence[Mapping], variant: str) -> List[Dict]:
    result: List[Dict] = []
    physiology_only = variant.startswith("physiology")
    use_static = "plus_static" in variant or variant.startswith("static_temporal")
    for sample in samples:
        out = clone_sample(sample)
        # Strict no-text: the formal model receives a zero note tensor.
        out["note"] = np.zeros(768, dtype=np.float32)
        out["note_stats"] = np.zeros(4, dtype=np.float32)
        static = np.asarray(out.get("static", np.zeros(18)), dtype=np.float32).copy()
        if not use_static:
            static[:] = 0.0
        elif static.size > 1:
            # Frozen index.json names element 1 as icu_los_hours. This is a full-
            # stay future variable and is invalid at an early fixed-time horizon.
            static[1] = 0.0
        out["static"] = static
        if physiology_only:
            out["meds"] = {"t": [], "f": [], "v": []}
        result.append(out)
    return result


def pooled_temporal_summaries(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return tokens.mean(dim=1), tokens[:, 0, :]


@torch.inference_mode()
def encode_temporal(base: torch.nn.Module, events: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
    """Exact deterministic eval-time temporal path from EnhancedCDT-V5.forward."""
    x = base.sparse_processor(events)
    if base.feature_gnn is not None:
        feature_ids = events[:, :, 2].long()
        x = x + base.feature_gnn(x, feature_ids)
    elif hasattr(base, "feature_mlp"):
        x = x + base.feature_mlp(x)
    if base.use_diffusion:
        x = base.diffusion(x, training=False, stochastic=False)
    return base.temporal_transformer(x, event_mask)


@torch.inference_mode()
def predict_from_temporal(
    base: torch.nn.Module,
    temporal_tokens: torch.Tensor,
    static: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    zero_notes = torch.zeros(static.shape[0], 768, dtype=static.dtype, device=static.device)
    static_tokens = base.static_tokenizer(static)
    static_pooled = static_tokens.mean(dim=1)
    static_interaction = base.static_tokenizer.get_interaction_features(static)
    fusion_tokens = temporal_tokens + 0.15 * static_pooled.unsqueeze(1)
    fusion_tokens = fusion_tokens + 0.1 * static_interaction.unsqueeze(1)
    fused = base.multimodal_fusion(fusion_tokens, static, zero_notes)
    deep_logits = base.classifier(fused)
    label_corr_logits = base.label_correlation(fused)
    baseline_logits = base.static_baseline(static)
    w_base = torch.sigmoid(base.baseline_weight) * 0.5
    w_corr = torch.sigmoid(base.label_corr_weight) * 0.3
    w_deep = 1.0 - w_base - w_corr
    logits = w_base * baseline_logits + w_corr * label_corr_logits + w_deep * deep_logits
    return torch.sigmoid(logits), fused


def safe_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    aucs: List[float] = []
    aprs: List[float] = []
    for j in range(labels.shape[1]):
        y = (labels[:, j] > 0.5).astype(int)
        if y.sum() > 0 and y.sum() < len(y):
            aucs.append(float(roc_auc_score(y, probs[:, j])))
            aprs.append(float(average_precision_score(y, probs[:, j])))
    return {
        "macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "macro_auprc": float(np.mean(aprs)) if aprs else float("nan"),
    }


def inspect_time_units(
    data: Sequence[Mapping],
    icu_offsets: Mapping[int, float],
    raw_icu_los_hours: Mapping[int, float],
) -> Tuple[pd.DataFrame, Dict]:
    rows = []
    all_diffs: List[float] = []
    for sample in data:
        ts = np.asarray(sample.get("timestamps", []), dtype=np.float64)
        stay_id = int(sample.get("stay_id"))
        icu_offset = float(icu_offsets[stay_id])
        ts_relative = ts - icu_offset if ts.size else ts
        if ts.size > 1:
            d = np.diff(ts)
            all_diffs.extend(d[np.isfinite(d) & (d >= 0)].tolist())
        rows.append(
            {
                "stay_id": stay_id,
                "n_timestamps": int(ts.size),
                "timestamp_start_raw": float(ts.min()) if ts.size else np.nan,
                "timestamp_end_raw": float(ts.max()) if ts.size else np.nan,
                "timestamp_span_seconds": float(ts_relative.max()) if ts.size else np.nan,
                "timestamp_span_hours": float(ts_relative.max() / 3600.0) if ts.size else np.nan,
                "icu_offset_seconds_from_hospital_admission": icu_offset,
                "pre_icu_timestamp_count": int((ts < icu_offset).sum()),
                "raw_icu_los_hours": float(raw_icu_los_hours[stay_id]),
                "static_index1_full_los": float(np.asarray(sample.get("static", np.zeros(18)))[1]),
            }
        )
    frame = pd.DataFrame(rows)
    diffs = np.asarray(all_diffs, dtype=np.float64)
    audit = {
        "inferred_timestamp_unit": "seconds",
        "basis": [
            "Adjacent differences of 60, 120, 900 and 3,060 correspond to plausible 1-, 2-, 15- and 51-minute event spacing when interpreted as seconds.",
            "Interpreting the same values as minutes would imply implausible median event spacing and multi-year stay spans.",
            "Raw tables establish that bundle timestamps are hospital-admission-relative; ICU windows are anchored with intime minus admittime.",
        ],
        "timestamp_diff_percentiles": np.percentile(diffs, [0, 1, 25, 50, 75, 95, 99, 100]).tolist()
        if diffs.size
        else [],
        "timestamp_span_hours_percentiles": frame["timestamp_span_hours"]
        .quantile([0, 0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
        .to_dict(),
        "warning": "Fixed horizons are ICU-relative after joining stay_id/hadm_id/subject_id to raw icustays and admissions timestamps.",
        "bundle_observed_end_after_raw_icu_outtime_fraction": float(
            (frame["timestamp_span_hours"] > frame["raw_icu_los_hours"]).mean()
        ),
    }
    return frame, audit


def main() -> None:
    args = parse_args()
    args.runtime_root = args.runtime_root.resolve()
    args.data_dir = args.data_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.icustays = args.icustays.resolve()
    args.admissions = args.admissions.resolve()
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "outputs_seed42_721" / args.cohort_arm
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {args.output_dir}. Pass --overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if str(args.runtime_root) not in sys.path:
        sys.path.insert(0, str(args.runtime_root))
    os.environ["MIMIC_DATA_DIR"] = str(args.data_dir)

    import benchmark_mimic_to_eicu_baselines as bm
    from models.data_utils import collate_events
    from models.enhanced_cdt import EnhancedCausalDigitalTwin
    from npj_external_validation_v3 import stratified_split_3way

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    horizons = sorted({float(x.strip()) for x in args.horizons.split(",") if x.strip()})
    checkpoint_hash_before = sha256_file(args.checkpoint)
    script_hash_at_start = sha256_file(Path(__file__).resolve())
    icustays_hash = sha256_file(args.icustays)
    admissions_hash = sha256_file(args.admissions)

    if args.seed != 42 or not np.isclose(args.val_ratio, 0.20) or not np.isclose(args.test_ratio, 0.10):
        raise ValueError(
            "This exporter is locked to the formal seed-42 70/20/10 protocol "
            "(validation=0.20, test=0.10)."
        )

    print("Loading frozen MIMIC bundle...", flush=True)
    data = bm.load_mimic_data(limit=0)
    if len(data) != 28788:
        raise RuntimeError(f"Expected 28,788 sequences, found {len(data)}")
    raw_timing, raw_timing_audit = load_icu_offsets(args.icustays, args.admissions)
    bundle_ids = pd.DataFrame(
        {
            "stay_id": [int(s.get("stay_id")) for s in data],
            "subject_id": [int(s.get("subject_id")) for s in data],
            "hadm_id": [int(s.get("hadm_id")) for s in data],
        }
    )
    timing_link = bundle_ids.merge(
        raw_timing[["stay_id", "subject_id", "hadm_id", "intime", "outtime", "admittime", "icu_offset_seconds"]],
        on=["stay_id", "subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )
    timing_link["raw_icu_los_hours"] = (
        timing_link["outtime"] - timing_link["intime"]
    ).dt.total_seconds() / 3600.0
    if timing_link["icu_offset_seconds"].isna().any() or timing_link["raw_icu_los_hours"].isna().any():
        raise RuntimeError("Raw ICU timing linkage is incomplete")
    if len(timing_link) != len(data) or timing_link["stay_id"].duplicated().any():
        raise RuntimeError("Raw ICU timing linkage failed one-to-one coverage")
    raw_timing_audit.update(
        {
            "bundle_rows": len(bundle_ids),
            "bundle_rows_linked": int(timing_link["icu_offset_seconds"].notna().sum()),
            "bundle_linkage_fraction": float(timing_link["icu_offset_seconds"].notna().mean()),
        }
    )
    timing_by_stay = timing_link.set_index("stay_id")
    icu_offsets = timing_by_stay["icu_offset_seconds"].to_dict()
    raw_icu_los_by_stay = timing_by_stay["raw_icu_los_hours"].to_dict()
    # Reconstruct the exact formal training-domain split with the original
    # split implementation. The primary latent analysis is restricted to the
    # third arm: the 10% confirmation set held out from gradient updates and
    # checkpoint selection. Historical training logs monitored its metrics.
    train, validation, test = stratified_split_3way(
        data,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    expected_sizes = {"train": 20151, "validation": 5758, "test": 2879}
    observed_sizes = {"train": len(train), "validation": len(validation), "test": len(test)}
    if observed_sizes != expected_sizes:
        raise RuntimeError(
            f"Formal seed-42 split size mismatch: expected {expected_sizes}, found {observed_sizes}"
        )
    analysis_cohort = validation if args.cohort_arm == "validation" else test
    full_analysis_rows = len(analysis_cohort)
    full_arm_stay_id_sha256 = sha256_stay_ids(analysis_cohort)
    if args.limit > 0:
        analysis_cohort = analysis_cohort[: args.limit]
    exported_stay_id_sha256 = sha256_stay_ids(analysis_cohort)

    other_arms = (train, test) if args.cohort_arm == "validation" else (train, validation)
    other_arm_subjects = {
        int(s.get("subject_id"))
        for arm in other_arms
        for s in arm
    }
    other_arm_hadm_ids = {
        int(s.get("hadm_id"))
        for arm in other_arms
        for s in arm
    }
    ids = pd.DataFrame(
        {
            "cohort_row": np.arange(len(analysis_cohort), dtype=int),
            "cohort_arm": args.cohort_arm,
            "stay_id": [s.get("stay_id") for s in analysis_cohort],
            "subject_id": [s.get("subject_id") for s in analysis_cohort],
            "hadm_id": [s.get("hadm_id") for s in analysis_cohort],
            "static_index1_icu_los_hours": [float(np.asarray(s.get("static", np.zeros(18)))[1]) for s in analysis_cohort],
        }
    )
    ids = ids.join(
        timing_by_stay[["icu_offset_seconds", "raw_icu_los_hours"]],
        on="stay_id",
        validate="one_to_one",
    )
    ids["full_icu_los_hours"] = ids["raw_icu_los_hours"]
    ids["static_minus_raw_icu_los_hours"] = (
        ids["static_index1_icu_los_hours"] - ids["raw_icu_los_hours"]
    )
    ids["subject_overlap_with_other_split_arms"] = ids["subject_id"].isin(other_arm_subjects)
    ids["subject_disjoint_sensitivity_eligible"] = ~ids["subject_overlap_with_other_split_arms"]
    ids["hadm_overlap_with_other_split_arms"] = ids["hadm_id"].isin(other_arm_hadm_ids)
    ids["hadm_disjoint_sensitivity_eligible"] = ~ids["hadm_overlap_with_other_split_arms"]
    ids.to_csv(args.output_dir / f"{args.cohort_arm}_stay_ids_seed42.csv", index=False)
    if ids["stay_id"].isna().any() or ids["stay_id"].duplicated().any():
        raise RuntimeError("Cohort-arm stay_id list contains missing or duplicate IDs")

    time_frame, time_audit = inspect_time_units(
        analysis_cohort,
        icu_offsets,
        raw_icu_los_by_stay,
    )
    time_frame.to_csv(args.output_dir / "time_unit_patient_audit.csv", index=False)
    json_dump(args.output_dir / "time_unit_audit.json", time_audit)

    print(f"Loading frozen checkpoint on {device}...", flush=True)
    model = EnhancedCausalDigitalTwin(
        n_outputs=6,
        d_model=128,
        use_gnn=True,
        use_diffusion=True,
    ).to(device)
    load_meta = load_formal_checkpoint_strict(model, args.checkpoint, device)
    model.eval()
    model.requires_grad_(False)
    if load_meta.get("loaded_tensors", 0) != len(model.state_dict()):
        raise RuntimeError(f"Checkpoint compatibility failure: {load_meta}")
    base = model

    prediction_rows: List[Dict] = []
    metric_rows: List[Dict] = []
    coverage_rows: List[Dict] = []

    for horizon in horizons:
        print(f"Cropping {horizon:g} h...", flush=True)
        cropped: List[Dict] = []
        crop_audit: List[Dict] = []
        for sample in analysis_cohort:
            item, audit = fixed_time_crop(
                sample,
                horizon,
                icu_offset_seconds=icu_offsets[int(sample.get("stay_id"))],
            )
            cropped.append(item)
            crop_audit.append(audit)
        audit_df = pd.DataFrame(crop_audit)
        los_only_eligible = ids.full_icu_los_hours.to_numpy(np.float32) >= horizon
        observed_event_support = audit_df.n_events_kept.to_numpy(np.int64) > 0
        landmark_eligible = los_only_eligible & observed_event_support
        coverage_rows.append(
            {
                "horizon_hours": horizon,
                "n_sequences": len(cropped),
                "n_los_only_eligible_raw_icu_los_ge_horizon": int(los_only_eligible.sum()),
                "n_observed_event_supported": int(observed_event_support.sum()),
                "n_primary_landmark_eligible_los_and_observed_event": int(landmark_eligible.sum()),
                "zero_timestamp_sequences": int((audit_df.n_timestamp_kept == 0).sum()),
                "zero_event_sequences": int((audit_df.n_events_kept == 0).sum()),
                "median_timestamps_kept": float(audit_df.n_timestamp_kept.median()),
                "median_events_kept_before_512_sampling": float(audit_df.n_events_kept.median()),
                "p95_events_kept_before_512_sampling": float(audit_df.n_events_kept.quantile(0.95)),
            }
        )

        prepared = {v: prepare_variant(cropped, v) for v in VARIANTS}
        accum: Dict[str, Dict[str, List[np.ndarray]]] = {
            v: {"probs": [], "fused": [], "temporal_mean": [], "temporal_token0": [], "labels": []}
            for v in VARIANTS
        }

        for start in range(0, len(cropped), args.batch_size):
            stop = min(start + args.batch_size, len(cropped))
            # Run the formal model's complete forward path for every masked
            # input variant. This preserves its learned static skip and label-
            # correlation weights and avoids reconstructing logits externally.
            for variant in VARIANTS:
                batch = collate_events(prepared[variant][start:stop], max_events=args.max_events)
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                zero_notes = torch.zeros(
                    batch["static"].shape[0], 768, dtype=batch["static"].dtype, device=device
                )
                out = base(
                    batch["events"],
                    batch["event_mask"],
                    batch["static"],
                    zero_notes,
                    stochastic=False,
                )
                temporal = out["temporal_repr"]
                tmean, t0 = pooled_temporal_summaries(temporal)
                accum[variant]["probs"].append(torch.sigmoid(out["factual_logits"]).cpu().numpy())
                accum[variant]["fused"].append(out["fused_repr"].cpu().numpy())
                accum[variant]["temporal_mean"].append(tmean.cpu().numpy())
                accum[variant]["temporal_token0"].append(t0.cpu().numpy())
                accum[variant]["labels"].append(batch["labels"].cpu().numpy())

            if start % (args.batch_size * 20) == 0:
                print(f"  {stop}/{len(cropped)}", flush=True)

        for variant in VARIANTS:
            arrays = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in accum[variant].items()}
            npz_path = args.output_dir / f"h{int(horizon):02d}_{variant}.npz"
            np.savez_compressed(
                npz_path,
                stay_id=ids["stay_id"].to_numpy(np.int64),
                subject_id=ids["subject_id"].to_numpy(np.int64),
                hadm_id=ids["hadm_id"].to_numpy(np.int64),
                labels=arrays["labels"],
                probs=arrays["probs"],
                fused_repr=arrays["fused"],
                temporal_mean_repr=arrays["temporal_mean"],
                temporal_token0_repr=arrays["temporal_token0"],
                full_icu_los_hours=ids["full_icu_los_hours"].to_numpy(np.float32),
                static_index1_icu_los_hours=ids["static_index1_icu_los_hours"].to_numpy(np.float32),
                icu_offset_seconds=ids["icu_offset_seconds"].to_numpy(np.float64),
                los_only_eligible=los_only_eligible,
                observed_event_support=observed_event_support,
                landmark_eligible=landmark_eligible,
                subject_disjoint_sensitivity_eligible=ids["subject_disjoint_sensitivity_eligible"].to_numpy(bool),
                hadm_disjoint_sensitivity_eligible=ids["hadm_disjoint_sensitivity_eligible"].to_numpy(bool),
                label_names=np.asarray(LABEL_NAMES),
            )
            metrics = safe_metrics(arrays["labels"], arrays["probs"])
            metric_rows.append({"horizon_hours": horizon, "variant": variant, "analysis_scope": f"all_{args.cohort_arm}", "n": len(ids), **metrics})
            eligible = landmark_eligible
            eligible_metrics = safe_metrics(arrays["labels"][eligible], arrays["probs"][eligible])
            metric_rows.append({"horizon_hours": horizon, "variant": variant, "analysis_scope": "landmark_eligible", "n": int(eligible.sum()), **eligible_metrics})
            subject_disjoint = eligible & ids["subject_disjoint_sensitivity_eligible"].to_numpy(bool)
            subject_disjoint_metrics = safe_metrics(
                arrays["labels"][subject_disjoint], arrays["probs"][subject_disjoint]
            )
            metric_rows.append(
                {
                    "horizon_hours": horizon,
                    "variant": variant,
                    "analysis_scope": "landmark_eligible_subject_disjoint_sensitivity",
                    "n": int(subject_disjoint.sum()),
                    **subject_disjoint_metrics,
                }
            )
            for i in range(len(ids)):
                row = {
                    "cohort_row": i,
                    "cohort_arm": args.cohort_arm,
                    "stay_id": int(ids.iloc[i].stay_id),
                    "horizon_hours": horizon,
                    "variant": variant,
                    "npz_file": npz_path.name,
                    "npz_row": i,
                    "full_icu_los_hours": float(ids.iloc[i].full_icu_los_hours),
                    "los_only_eligible": bool(los_only_eligible[i]),
                    "observed_event_support": bool(observed_event_support[i]),
                    "landmark_eligible": bool(landmark_eligible[i]),
                }
                row.update({f"prob_{name}": float(arrays["probs"][i, j]) for j, name in enumerate(LABEL_NAMES)})
                prediction_rows.append(row)

    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "fixed_time_predictions.csv.gz", index=False)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "fixed_time_performance.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(args.output_dir / "fixed_time_coverage_audit.csv", index=False)

    checkpoint_hash_after = sha256_file(args.checkpoint)
    script_hash_at_end = sha256_file(Path(__file__).resolve())
    run_audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "runtime_root": args.runtime_root,
        "data_dir": args.data_dir,
        "export_script": Path(__file__).resolve(),
        "export_script_sha256_start": script_hash_at_start,
        "export_script_sha256_end": script_hash_at_end,
        "export_script_unchanged": script_hash_at_start == script_hash_at_end,
        "icustays": args.icustays,
        "icustays_sha256": icustays_hash,
        "admissions": args.admissions,
        "admissions_sha256": admissions_hash,
        "raw_timing_linkage": raw_timing_audit,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        "checkpoint_load": load_meta,
        "model_training_flag": model.training,
        "any_parameter_requires_grad": any(p.requires_grad for p in model.parameters()),
        "total_sequences": len(data),
        "train_rows_reconstructed": len(train),
        "validation_rows_reconstructed": len(validation),
        "heldout_test_rows_reconstructed": len(test),
        "exported_cohort_arm": args.cohort_arm,
        "full_cohort_rows_before_limit": full_analysis_rows,
        "exported_rows": len(analysis_cohort),
        "train_stay_id_sha256": sha256_stay_ids(train),
        "validation_stay_id_sha256": sha256_stay_ids(validation),
        "test_stay_id_sha256": sha256_stay_ids(test),
        "full_arm_stay_id_sha256": full_arm_stay_id_sha256,
        "exported_stay_id_sha256": exported_stay_id_sha256,
        "subject_overlap_with_other_split_arms_rows": int(ids["subject_overlap_with_other_split_arms"].sum()),
        "subject_disjoint_sensitivity_eligible_rows": int(ids["subject_disjoint_sensitivity_eligible"].sum()),
        "hadm_overlap_with_other_split_arms_rows": int(ids["hadm_overlap_with_other_split_arms"].sum()),
        "hadm_disjoint_sensitivity_eligible_rows": int(ids["hadm_disjoint_sensitivity_eligible"].sum()),
        "split_independence_caveat": "The formal split is stay-level; subject- and admission-disjoint flags are exported for sensitivity analyses.",
        "split_seed": args.seed,
        "split_design": "formal three-way internal split; the test arm was held out from gradient updates and checkpoint selection and is used for confirmation",
        "split_ratios": {"training": 0.70, "validation": args.val_ratio, "heldout_test": args.test_ratio},
        "split_implementation": "npj_external_validation_v3.stratified_split_3way",
        "analysis_role": (
            "state discovery, component/state-count selection, and state naming only"
            if args.cohort_arm == "validation"
            else "frozen state assignment and primary confirmation; no state selection is permitted; historical training logs monitored test metrics"
        ),
        "horizons_hours": horizons,
        "max_events": args.max_events,
        "variants": list(VARIANTS),
        "strict_no_text_implementation": "Called the frozen formal EnhancedCausalDigitalTwin with a 768-D zero note tensor.",
        "physiology_only_implementation": "Medication event lists were cleared before collate; vitals and labs were retained. The architecture accepts a single sparse event tensor, so this input-level mask is the exact feasible modality exclusion.",
        "temporal_only_caveat": "Predictions still pass zero static/note vectors through learned fusion layers. The exported temporal_mean_repr and temporal_token0_repr are pre-fusion and therefore the stricter temporal latent representations.",
        "static_leakage_control": "Static index 1 (named icu_los_hours in index.json) was set to zero at every early horizon.",
        "time_origin_verification": time_audit["warning"],
        "bundle_observed_end_after_raw_icu_outtime_fraction": time_audit[
            "bundle_observed_end_after_raw_icu_outtime_fraction"
        ],
        "raw_vs_static_icu_los_difference_hours": {
            "mean": float(ids["static_minus_raw_icu_los_hours"].mean()),
            "median": float(ids["static_minus_raw_icu_los_hours"].median()),
            "p01": float(ids["static_minus_raw_icu_los_hours"].quantile(0.01)),
            "p99": float(ids["static_minus_raw_icu_los_hours"].quantile(0.99)),
        },
        "landmark_support_rule": "Primary horizon analyses require raw ICU LOS >= horizon and at least one observed event inside the ICU-anchored window. LOS-only eligibility and zero-event attrition are exported separately.",
    }
    json_dump(args.output_dir / "run_audit.json", run_audit)
    if not run_audit["checkpoint_unchanged"] or not run_audit["export_script_unchanged"]:
        raise RuntimeError("Checkpoint or export script hash changed during a read-only export")

    print(f"Completed: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
