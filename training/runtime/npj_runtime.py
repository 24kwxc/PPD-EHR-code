import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


log = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent
RELEASE_ROOT = THIS_DIR.parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path and path.exists():
            return path
    return None


def resolve_models_root() -> Path:
    env = os.environ.get("MODELS_CODE_DIR", "").strip()
    if env:
        path = Path(env).expanduser()
        if path.exists():
            return path
    candidates = [
        WORKSPACE_ROOT / "export_no_renal_v2_bundle_with_data" / "paper_experiments_v2",
        PROJECT_ROOT / "paper_experiments_v2",
    ]
    found = _first_existing(candidates)
    if found is None:
        raise FileNotFoundError("Unable to resolve MODELS_CODE_DIR / paper_experiments_v2")
    return found


def resolve_mimic_dir() -> Path:
    env = os.environ.get("MIMIC_DATA_DIR", "").strip()
    if env:
        path = Path(env).expanduser()
        if path.exists():
            return path
    candidates = [
        WORKSPACE_ROOT / "export_no_renal_v2_bundle_with_data" / "artifacts" / "preprocessed_clinicalbert" / "sharded_bundle_event",
        PROJECT_ROOT / "data" / "mimic",
    ]
    found = _first_existing(candidates)
    if found is None:
        raise FileNotFoundError("Unable to resolve MIMIC sharded event bundle")
    return found


def resolve_eicu_bundle() -> Path:
    env = os.environ.get("EICU_DATA_PATH", "").strip()
    if env:
        path = Path(env).expanduser()
        if path.exists():
            return path
    candidates = [
        WORKSPACE_ROOT / "eicu-collaborative-research-database-2.0" / "processed" / "sequences" / "eicu_bundle.joblib",
        PROJECT_ROOT / "data" / "eicu" / "eicu_bundle.joblib",
    ]
    found = _first_existing(candidates)
    if found is None:
        raise FileNotFoundError("Unable to resolve eICU bundle")
    return found


def resolve_v5_checkpoint() -> Path:
    env = os.environ.get("ECDT_V5_CHECKPOINT", "").strip()
    if env:
        path = Path(env).expanduser()
        if path.exists():
            return path
    candidates = [
        RELEASE_ROOT / "model" / "model_best.pt",
        PROJECT_ROOT / "models" / "ecdt_v5_best.pt",
        WORKSPACE_ROOT / "ecdt_v5_best.pt",
        WORKSPACE_ROOT / "v5_download_bundle_20260316_134012" / "v5_download_bundle_20260316_134012" / "models" / "main_with_aug" / "v5_best.pt",
    ]
    found = _first_existing(candidates)
    if found is None:
        raise FileNotFoundError("Unable to resolve ECDT V5 checkpoint")
    return found


def resolve_reference_checkpoint(kind: str) -> Optional[Path]:
    env_map = {
        "full": "ECDT_V5_CHECKPOINT",
        "no_diff": "ECDT_NO_DIFF_CHECKPOINT",
        "no_gnn": "ECDT_NO_GNN_CHECKPOINT",
        "no_aug": "ECDT_NO_AUG_CHECKPOINT",
    }
    env_name = env_map.get(kind)
    if env_name:
        env = os.environ.get(env_name, "").strip()
        if env:
            path = Path(env).expanduser()
            if path.exists():
                return path

    candidates = {
        "full": [
            PROJECT_ROOT / "models" / "ecdt_v5_best.pt",
            WORKSPACE_ROOT / "ecdt_v5_best.pt",
            WORKSPACE_ROOT / "v5_download_bundle_20260316_134012" / "v5_download_bundle_20260316_134012" / "models" / "main_with_aug" / "v5_best.pt",
        ],
        "no_diff": [
            WORKSPACE_ROOT / "root" / "results" / "manual_runs" / "no_diff_re" / "run_no_diff" / "ecdt_no_diff.pt",
            WORKSPACE_ROOT / "root" / "results" / "ablation_v4_E100" / "run_no_diff" / "ecdt_no_diff.pt",
            WORKSPACE_ROOT / "NC_Experiments_Final_package_20260113" / "root" / "results" / "ablation_v4_E100" / "run_no_diff" / "ecdt_no_diff.pt",
            WORKSPACE_ROOT / "root" / "results" / "nc_experiments" / "ECDT_NoDiffusion" / "model_best.pt",
        ],
        "no_gnn": [
            WORKSPACE_ROOT / "root" / "results" / "ablation_v4_E100" / "run_no_gnn" / "ecdt_no_gnn.pt",
            WORKSPACE_ROOT / "NC_Experiments_Final_package_20260113" / "root" / "results" / "ablation_v4_E100" / "run_no_gnn" / "ecdt_no_gnn.pt",
            WORKSPACE_ROOT / "root" / "results" / "nc_experiments" / "ECDT_NoGNN" / "model_best.pt",
        ],
        "no_aug": [
            WORKSPACE_ROOT / "root" / "results" / "manual_runs" / "no_aug_re" / "run_no_aug" / "ecdt_no_aug.pt",
            WORKSPACE_ROOT / "root" / "results" / "ablation_v4_E100" / "run_no_aug" / "ecdt_no_aug.pt",
            WORKSPACE_ROOT / "root" / "results" / "nc_experiments" / "ECDT_NoAugmentation" / "model_best.pt",
        ],
    }.get(kind, [])

    return _first_existing(candidates)


def resolve_synthetic_records(kind: str = "tse_tuned") -> Path:
    env_name = "SYNTHETIC_RECORDS_PATH"
    env = os.environ.get(env_name, "").strip()
    if env:
        path = Path(env).expanduser()
        if path.exists():
            return path
    name = {
        "base": PROJECT_ROOT / "data" / "synthetic" / "synthetic_records_v6_comorbidity.joblib",
        "tse_tuned": PROJECT_ROOT / "data" / "synthetic_tse_tuned" / "synthetic_records_v6_comorbidity.joblib",
        "tse_tuned_20ep": PROJECT_ROOT / "data" / "synthetic_tse_tuned_20ep" / "synthetic_records_v6_comorbidity.joblib",
    }.get(kind, PROJECT_ROOT / "data" / "synthetic_tse_tuned" / "synthetic_records_v6_comorbidity.joblib")
    if not name.exists():
        raise FileNotFoundError(f"Unable to resolve synthetic records for kind={kind}")
    return name


def configure_environment(require_data: bool = True) -> Dict[str, str]:
    """Configure local module and data paths.

    ``require_data=False`` keeps imports and ``--help`` usable in a public
    code-only checkout. Actual training/evaluation still fails clearly when a
    required controlled-data file is opened.
    """
    if require_data:
        mimic_dir = resolve_mimic_dir()
        eicu_bundle = resolve_eicu_bundle()
    else:
        mimic_dir = Path(os.environ.get(
            "MIMIC_DATA_DIR",
            RELEASE_ROOT / "data" / "restricted" / "mimic_v3_event_bundle",
        ))
        eicu_bundle = Path(os.environ.get(
            "EICU_DATA_PATH",
            RELEASE_ROOT / "data" / "restricted" / "eicu_event_bundle.joblib",
        ))
    env_updates = {
        "MODELS_CODE_DIR": str(resolve_models_root()),
        "MIMIC_DATA_DIR": str(mimic_dir),
        "EICU_DATA_PATH": str(eicu_bundle),
    }
    for key, value in env_updates.items():
        os.environ[key] = value
    models_root = env_updates["MODELS_CODE_DIR"]
    if models_root not in sys.path:
        sys.path.insert(0, models_root)
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    return env_updates


def ensure_output_dir(name: str) -> Path:
    out_dir = PROJECT_ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def stable_label_signature(label: np.ndarray) -> int:
    label_bin = (np.asarray(label) > 0.5).astype(np.int32)
    return int(np.dot(label_bin, (1 << np.arange(label_bin.shape[0], dtype=np.int64))))


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def dataset_fingerprint(data: List[Dict], n_samples: int = 32) -> str:
    if not data:
        return "empty"
    rows = []
    for sample in data[: min(n_samples, len(data))]:
        label = np.asarray(sample.get("label", np.zeros(6)), dtype=np.float32).round(4).tolist()
        static = np.asarray(sample.get("static", np.zeros(18)), dtype=np.float32)
        rows.append(
            {
                "label": label,
                "static_sum": round(float(np.nan_to_num(static).sum()), 6),
                "note_nonzero": int(np.count_nonzero(np.nan_to_num(sample.get("note", np.zeros(768))))),
                "vitals_n": len(sample.get("vitals", {}).get("v", [])),
                "labs_n": len(sample.get("labs", {}).get("v", [])),
                "meds_n": len(sample.get("meds", {}).get("v", [])),
            }
        )
    return stable_hash(json.dumps(rows, sort_keys=True))


class ExperimentRegistry:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.jsonl_path = out_dir / "experiment_registry.jsonl"
        self.csv_path = out_dir / "provenance_manifest.csv"
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "experiment",
                        "run_id",
                        "script",
                        "seed",
                        "checkpoint",
                        "mimic_fingerprint",
                        "eicu_fingerprint",
                        "summary_json",
                        "table_csv",
                        "notes",
                    ],
                )
                writer.writeheader()

    def register(
        self,
        *,
        experiment: str,
        script_path: Path,
        seed: Optional[int],
        checkpoint: Optional[Path],
        mimic_fingerprint: str,
        eicu_fingerprint: str,
        summary_json: Optional[Path],
        table_csv: Optional[Path],
        extra: Optional[Dict] = None,
    ) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "timestamp": ts,
            "experiment": experiment,
            "script": str(script_path),
            "seed": seed,
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "mimic_fingerprint": mimic_fingerprint,
            "eicu_fingerprint": eicu_fingerprint,
            "summary_json": None if summary_json is None else str(summary_json),
            "table_csv": None if table_csv is None else str(table_csv),
        }
        if extra:
            payload.update(extra)
        run_id = stable_hash(json.dumps(payload, sort_keys=True, default=str))
        payload["run_id"] = run_id
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "experiment",
                    "run_id",
                    "script",
                    "seed",
                    "checkpoint",
                    "mimic_fingerprint",
                    "eicu_fingerprint",
                    "summary_json",
                    "table_csv",
                    "notes",
                ],
            )
            writer.writerow(
                {
                    "timestamp": ts,
                    "experiment": experiment,
                    "run_id": run_id,
                    "script": str(script_path),
                    "seed": "" if seed is None else seed,
                    "checkpoint": "" if checkpoint is None else str(checkpoint),
                    "mimic_fingerprint": mimic_fingerprint,
                    "eicu_fingerprint": eicu_fingerprint,
                    "summary_json": "" if summary_json is None else str(summary_json),
                    "table_csv": "" if table_csv is None else str(table_csv),
                    "notes": json.dumps(extra or {}, ensure_ascii=False),
                }
            )
        return run_id
