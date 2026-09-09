import argparse
import copy
import csv
import faulthandler
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from npj_metrics import aggregate_seed_metrics, flatten_per_label_rows, multilabel_metrics
from npj_runtime import (
    ExperimentRegistry,
    THIS_DIR,
    configure_environment,
    dataset_fingerprint,
    default_device,
    ensure_output_dir,
    resolve_v5_checkpoint,
    set_seed,
    stable_label_signature,
)


configure_environment(require_data=False)

import benchmark_mimic_to_eicu_baselines as bm  # noqa: E402
from models.data_utils import collate_events  # noqa: E402
from models.enhanced_cdt_v5 import EnhancedCausalDigitalTwinV5  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
faulthandler.enable()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="npj-grade real external validation suite for ECDT")
    parser.add_argument("--output-dir", type=str, default=str(ensure_output_dir("npj_real_external_validation")))
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--fractions", type=str, default="0.01,0.05,0.10,0.20")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--deep-epochs", type=int, default=4)
    parser.add_argument("--deep-hidden-dim", type=int, default=192)
    parser.add_argument("--replay-ratio", type=float, default=0.30)
    parser.add_argument("--finetune-lr", type=float, default=2e-4)
    parser.add_argument("--finetune-patience", type=int, default=2)
    parser.add_argument("--static-dropout-p", type=float, default=0.20)
    parser.add_argument("--notes-dropout-p", type=float, default=0.30)
    parser.add_argument("--order-aux-weight", type=float, default=0.0)
    parser.add_argument("--order-aux-margin", type=float, default=0.01)
    parser.add_argument("--max-events", type=int, default=512)
    parser.add_argument("--target-val-ratio", type=float, default=0.20)
    parser.add_argument("--target-val-mode", type=str, default="within_target", choices=["within_target", "pool_remainder"])
    parser.add_argument("--align-baseline-target-budget", action="store_true")
    parser.add_argument("--model-impl", type=str, default="auto", choices=["auto", "v5", "stable_branch"])
    parser.add_argument("--static-shortcut-indices", type=str, default="")
    parser.add_argument("--mimic-limit", type=int, default=0)
    parser.add_argument("--eicu-limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--quick-smoke", action="store_true")
    return parser.parse_args()


def clone_samples(data: Sequence[Dict]) -> List[Dict]:
    cloned: List[Dict] = []
    for sample in data:
        item = dict(sample)
        for key in ("label", "static", "note"):
            if key in item:
                item[key] = np.asarray(item[key], dtype=np.float32).copy()
        for key in ("vitals", "labs", "meds"):
            mod = item.get(key, {})
            if isinstance(mod, dict):
                item[key] = {
                    "t": list(mod.get("t", [])),
                    "f": list(mod.get("f", [])),
                    "v": list(mod.get("v", [])),
                }
        cloned.append(item)
    return cloned

def align_eicu_to_mimic(eicu_data: List[Dict]) -> List[Dict]:
    EICU_TO_MIMIC_VITAL = {
        'cvp': 156, 'dbp': 81, 'heart_rate': 330, 'mbp': 82,
        'resp_rate': 742, 'sbp': 83, 'spo2': 291, 'temperature': 850,
    }
    EICU_TO_MIMIC_LAB = {
        'albumin': 12, 'alt': 11, 'anion_gap': 29, 'ast': 36, 'base_excess': 50,
        'bicarbonate': 60, 'bilirubin': 67, 'bnp': 354, 'bun': 515, 'calcium': 85,
        'chloride': 118, 'creatinine': 145, 'crp': 81, 'glucose': 213, 'hematocrit': 228,
        'hemoglobin': 235, 'inr': 272, 'lactate': 282, 'magnesium': 313, 'pco2': 376,
        'ph': 378, 'phosphorus': 386, 'platelets': 395, 'po2': 396, 'potassium': 403,
        'pt': 424, 'ptt': 425, 'sodium': 460, 'troponin': 513, 'wbc': 544,
    }
    EICU_VITAL_NAMES = ['cvp', 'dbp', 'heart_rate', 'mbp', 'resp_rate', 'sbp', 'spo2', 'temperature']
    EICU_LAB_NAMES = ['albumin', 'alt', 'anion_gap', 'ast', 'base_excess', 'bicarbonate', 'bilirubin', 'bnp', 'bun', 'calcium', 'chloride', 'creatinine', 'crp', 'glucose', 'hematocrit', 'hemoglobin', 'inr', 'lactate', 'magnesium', 'pco2', 'ph', 'phosphorus', 'platelets', 'po2', 'potassium', 'pt', 'ptt', 'sodium', 'troponin', 'wbc']
    
    aligned = clone_samples(eicu_data)
    for sample in aligned:
        if 'vitals' in sample and isinstance(sample['vitals'], dict):
            new_f = []
            new_t = []
            new_v = []
            for t, f, v in zip(sample['vitals'].get('t', []), sample['vitals'].get('f', []), sample['vitals'].get('v', [])):
                if f < len(EICU_VITAL_NAMES):
                    name = EICU_VITAL_NAMES[f]
                    if name in EICU_TO_MIMIC_VITAL:
                        new_f.append(EICU_TO_MIMIC_VITAL[name])
                        new_t.append(t)
                        new_v.append(v)
            sample['vitals']['f'] = new_f
            sample['vitals']['t'] = new_t
            sample['vitals']['v'] = new_v
            
        if 'labs' in sample and isinstance(sample['labs'], dict):
            new_f = []
            new_t = []
            new_v = []
            for t, f, v in zip(sample['labs'].get('t', []), sample['labs'].get('f', []), sample['labs'].get('v', [])):
                if f < len(EICU_LAB_NAMES):
                    name = EICU_LAB_NAMES[f]
                    if name in EICU_TO_MIMIC_LAB:
                        new_f.append(EICU_TO_MIMIC_LAB[name])
                        new_t.append(t)
                        new_v.append(v)
            sample['labs']['f'] = new_f
            sample['labs']['t'] = new_t
            sample['labs']['v'] = new_v
    return aligned


def filter_mimic_to_aligned(mimic_data: List[Dict]) -> List[Dict]:
    """
    Systematically filter MIMIC-IV features to match the eICU core variable set (38 features).
    This resolves the distribution imbalance between the high-dimensional raw MIMIC data
    and the medically-merged eICU dataset.
    """
    EICU_TO_MIMIC_VITAL = {
        'cvp': 156, 'dbp': 81, 'heart_rate': 330, 'mbp': 82,
        'resp_rate': 742, 'sbp': 83, 'spo2': 291, 'temperature': 850,
    }
    EICU_TO_MIMIC_LAB = {
        'albumin': 12, 'alt': 11, 'anion_gap': 29, 'ast': 36, 'base_excess': 50,
        'bicarbonate': 60, 'bilirubin': 67, 'bnp': 354, 'bun': 515, 'calcium': 85,
        'chloride': 118, 'creatinine': 145, 'crp': 81, 'glucose': 213, 'hematocrit': 228,
        'hemoglobin': 235, 'inr': 272, 'lactate': 282, 'magnesium': 313, 'pco2': 376,
        'ph': 378, 'phosphorus': 386, 'platelets': 395, 'po2': 396, 'potassium': 403,
        'pt': 424, 'ptt': 425, 'sodium': 460, 'troponin': 513, 'wbc': 544,
    }
    allowed_vitals = set(EICU_TO_MIMIC_VITAL.values())
    allowed_labs = set(EICU_TO_MIMIC_LAB.values())
    
    filtered = clone_samples(mimic_data)
    for sample in filtered:
        # Vitals filtering
        if 'vitals' in sample and isinstance(sample['vitals'], dict):
            new_f, new_t, new_v = [], [], []
            for t, f, v in zip(sample['vitals'].get('t', []), sample['vitals'].get('f', []), sample['vitals'].get('v', [])):
                if f in allowed_vitals:
                    new_f.append(f)
                    new_t.append(t)
                    new_v.append(v)
            sample['vitals']['f'] = new_f
            sample['vitals']['t'] = new_t
            sample['vitals']['v'] = new_v
            
        # Labs filtering
        if 'labs' in sample and isinstance(sample['labs'], dict):
            new_f, new_t, new_v = [], [], []
            for t, f, v in zip(sample['labs'].get('t', []), sample['labs'].get('f', []), sample['labs'].get('v', [])):
                if f in allowed_labs:
                    new_f.append(f)
                    new_t.append(t)
                    new_v.append(v)
            sample['labs']['f'] = new_f
            sample['labs']['t'] = new_t
            sample['labs']['v'] = new_v
            
        # Drop medications (unaligned)
        if 'meds' in sample:
            sample['meds']['f'] = []
            sample['meds']['t'] = []
            sample['meds']['v'] = []
            
    return filtered

def add_note_signal(data: Sequence[Dict]) -> List[Dict]:
    enriched = clone_samples(data)
    for sample in enriched:
        note = np.asarray(sample.get("note", np.zeros(768, dtype=np.float32)), dtype=np.float32)
        note = np.nan_to_num(note, nan=0.0, posinf=0.0, neginf=0.0)
        nonzero = np.abs(note) > 1e-6
        stats = np.array(
            [
                float(nonzero.any()),
                float(nonzero.mean()),
                float(min(np.linalg.norm(note), 1_000.0) / 1_000.0),
                float(min(np.mean(np.abs(note)), 10.0) / 10.0),
            ],
            dtype=np.float32,
        )
        note = note.copy()
        note[-4:] = stats
        sample["note"] = note
        sample["note_stats"] = stats
        sample["note_is_missing"] = float(stats[0] < 0.5)
    return enriched


def mask_notes(data: Sequence[Dict]) -> List[Dict]:
    masked = clone_samples(data)
    for sample in masked:
        sample["note"] = np.zeros(768, dtype=np.float32)
        sample["note_stats"] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        sample["note_is_missing"] = 1.0
    return masked


def collate_with_note_stats(batch_data: Sequence[Dict], max_events: int = 512) -> Dict[str, torch.Tensor]:
    batch = collate_events(list(batch_data), max_events=max_events)
    stats = [
        np.asarray(sample.get("note_stats", np.zeros(4, dtype=np.float32)), dtype=np.float32)
        for sample in batch_data
    ]
    batch["note_stats"] = torch.tensor(np.asarray(stats), dtype=torch.float32)
    return batch


def apply_training_modality_dropout(
    batch: Dict[str, torch.Tensor],
    *,
    static_dropout_p: float,
    notes_dropout_p: float,
    static_zero_indices: Sequence[int] = (),
) -> Dict[str, torch.Tensor]:
    if static_zero_indices:
        valid = [int(i) for i in static_zero_indices if 0 <= int(i) < batch["static"].shape[1]]
        if valid:
            batch["static"][:, valid] = 0.0
    if static_dropout_p > 0:
        keep = (torch.rand(batch["static"].size(0), 1, device=batch["static"].device) >= static_dropout_p).float()
        batch["static"] = batch["static"] * keep
    if notes_dropout_p > 0:
        keep = (torch.rand(batch["notes"].size(0), 1, device=batch["notes"].device) >= notes_dropout_p).float()
        batch["notes"] = batch["notes"] * keep
        batch["note_stats"] = batch["note_stats"] * keep
    return batch


def shuffle_time_within_patient(data: Sequence[Dict], seed: int) -> List[Dict]:
    shuffled = clone_samples(data)
    rng = np.random.default_rng(seed)
    for sample in shuffled:
        for modality in ("vitals", "labs", "meds"):
            mod = sample.get(modality, {})
            if not isinstance(mod, dict):
                continue
            t = list(mod.get("t", []))
            if len(t) <= 1:
                continue
            order = np.asarray(t, dtype=np.int32).copy()
            rng.shuffle(order)
            mod["t"] = order.tolist()
            sample[modality] = mod
    return shuffled


def compute_order_margin_loss(
    model: nn.Module,
    batch_samples: Sequence[Dict],
    criterion: nn.Module,
    *,
    device: str,
    order_margin: float,
    static_zero_indices: Sequence[int] = (),
) -> torch.Tensor:
    shuffled = shuffle_time_within_patient(batch_samples, seed=random.randint(0, 10_000_000))
    shuf_batch = collate_with_note_stats(shuffled)
    shuf_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in shuf_batch.items()}
    shuf_batch = apply_training_modality_dropout(
        shuf_batch,
        static_dropout_p=0.0,
        notes_dropout_p=0.0,
        static_zero_indices=static_zero_indices,
    )
    shuf_out = model(
        shuf_batch["events"],
        shuf_batch["event_mask"],
        shuf_batch["static"],
        shuf_batch["notes"],
        shuf_batch["note_stats"],
    )
    shuf_loss = criterion(shuf_out["factual_logits"], shuf_batch["labels"])
    return shuf_loss


def make_signature_array(data: Sequence[Dict]) -> np.ndarray:
    return np.asarray([stable_label_signature(np.asarray(s["label"], dtype=np.float32)) for s in data], dtype=np.int64)


def parse_index_list(text: str) -> List[int]:
    values: List[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(int(chunk))
        except ValueError:
            log.warning("Ignoring non-integer static shortcut index: %s", chunk)
    return sorted(set(values))


def stratified_split(data: Sequence[Dict], test_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict], np.ndarray, np.ndarray]:
    indices = np.arange(len(data))
    sig = make_signature_array(data)
    bincount = np.bincount(sig) if len(sig) > 0 else np.zeros(1, dtype=np.int64)
    stratify = sig if np.min(bincount) >= 2 else None
    tr_idx, te_idx = train_test_split(indices, test_size=test_ratio, random_state=seed, stratify=stratify)
    train = [data[i] for i in tr_idx]
    test = [data[i] for i in te_idx]
    return train, test, tr_idx, te_idx


def stratified_split_3way(data: Sequence[Dict], val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Splits data into Train, Val, Test. For 7/2/1, use val_ratio=0.20, test_ratio=0.10"""
    indices = np.arange(len(data))
    sig = make_signature_array(data)
    bincount = np.bincount(sig) if len(sig) > 0 else np.zeros(1, dtype=np.int64)
    stratify = sig if np.min(bincount) >= 2 else None
    
    # First split out Test
    tr_va_idx, te_idx = train_test_split(indices, test_size=test_ratio, random_state=seed, stratify=stratify)
    
    # Then split Train and Val from the remainder
    # The proportion of val in the remaining data is val_ratio / (1 - test_ratio)
    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
    
    sig_rem = sig[tr_va_idx]
    bincount_rem = np.bincount(sig_rem) if len(sig_rem) > 0 else np.zeros(1, dtype=np.int64)
    stratify_rem = sig_rem if np.min(bincount_rem) >= 2 else None
    
    tr_idx, va_idx = train_test_split(tr_va_idx, test_size=adjusted_val_ratio, random_state=seed, stratify=stratify_rem)
    
    train = [data[i] for i in tr_idx]
    val = [data[i] for i in va_idx]
    test = [data[i] for i in te_idx]
    
    return train, val, test



class NoteAwareECDT(nn.Module):
    def __init__(self, base_model: nn.Module, note_dim: int = 768):
        super().__init__()
        self.base_model = base_model
        self.note_branch = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, note_dim),
        )
        nn.init.zeros_(self.note_branch[-1].weight)
        nn.init.zeros_(self.note_branch[-1].bias)

    def forward(self, events, event_mask, static, notes, note_stats=None):
        if note_stats is None:
            present = (notes.abs().sum(dim=1) > 1e-6).float()
            nonzero_frac = (notes.abs() > 1e-6).float().mean(dim=1)
            l2_norm = notes.norm(dim=1).clamp(max=1_000.0) / 1_000.0
            mean_abs = notes.abs().mean(dim=1).clamp(max=10.0) / 10.0
            note_stats = torch.stack([present, nonzero_frac, l2_norm, mean_abs], dim=1)
        adapted_notes = notes + self.note_branch(note_stats)
        return self.base_model(events, event_mask, static, adapted_notes)


class FocalBCE(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, smoothing: float = 0.02):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_s = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets_s, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class TransformerTabularBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 192, n_heads: int = 4, n_layers: int = 2, n_outputs: int = 6):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.embedding(x)
        h = self.encoder(h)
        return self.classifier(h.mean(dim=1))


def build_ecdt(
    use_diffusion: bool = True,
    use_gnn: bool = True,
    device: str = "cpu",
    model_impl: str = "v5",
) -> NoteAwareECDT:
    if model_impl == "stable_branch":
        from models_stable_branch import EnhancedCausalDigitalTwinV5Stable  # noqa: E402

        base = EnhancedCausalDigitalTwinV5Stable(
            n_outputs=6,
            d_model=128,
            use_gnn=use_gnn,
            use_diffusion=use_diffusion,
            use_domain_adaptation=False,
        )
    else:
        base = EnhancedCausalDigitalTwinV5(
            n_outputs=6,
            d_model=128,
            use_gnn=use_gnn,
            use_diffusion=use_diffusion,
            use_domain_adaptation=False,
        )
    model = NoteAwareECDT(base).to(device)
    return model


def load_checkpoint_flex(model: nn.Module, ckpt_path: Path, device: str) -> Dict[str, int]:
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_state = model.state_dict()
    filtered = {}
    for raw_key, value in state.items():
        normalized = raw_key.replace("module.", "")
        candidates = [
            normalized,
            f"base_model.{normalized}",
            normalized.replace("base_model.", ""),
        ]
        for candidate in candidates:
            if candidate in model_state:
                if model_state[candidate].shape == value.shape:
                    filtered[candidate] = value
                    break
                else:
                    log.warning("Shape mismatch for %s: model %s vs ckpt %s", candidate, model_state[candidate].shape, value.shape)
    missing = [k for k in model_state.keys() if k not in filtered]
    if not filtered:
        log.error("load_checkpoint_flex: 0 tensors matched! Check prefix mapping.")
    model.load_state_dict(filtered, strict=False)
    return {"loaded_tensors": len(filtered), "missing_tensors": len(missing)}


@torch.no_grad()
def predict_ecdt(
    model: nn.Module,
    data: Sequence[Dict],
    device: str,
    batch_size: int,
    threshold: float,
    label_names: Sequence[str],
    max_events: int = 512,
) -> Dict:
    model.eval()
    probs, labels = [], []
    for start in range(0, len(data), batch_size):
        batch = collate_with_note_stats(data[start:start + batch_size], max_events=max_events)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch["events"], batch["event_mask"], batch["static"], batch["notes"], batch["note_stats"])
        probs.append(torch.sigmoid(out["factual_logits"]).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(labels, axis=0)
    return {"probs": y_prob, "labels": y_true, "metrics": multilabel_metrics(y_true, y_prob, label_names=label_names, threshold=threshold)}


def finetune_ecdt(
    model: nn.Module,
    train_data: Sequence[Dict],
    val_data: Sequence[Dict],
    *,
    device: str,
    epochs: int,
    batch_size: int,
    threshold: float,
    label_names: Sequence[str],
    static_dropout_p: float = 0.0,
    notes_dropout_p: float = 0.0,
    static_zero_indices: Sequence[int] = (),
    order_aux_weight: float = 0.0,
    order_aux_margin: float = 0.01,
    lr: float = 2e-4,
    patience: int = 2,
    max_events: int = 512,
) -> Tuple[nn.Module, List[Dict]]:
    criterion = FocalBCE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history: List[Dict] = []
    best_auc = -1.0
    best_state = None
    stale = 0

    for epoch in tqdm(range(1, epochs + 1), desc="Finetuning ECDT", leave=False):
        model.train()
        train_list = list(train_data)
        random.shuffle(train_list)
        losses = []
        for start in range(0, len(train_list), batch_size):
            batch_samples = train_list[start:start + batch_size]
            batch = collate_with_note_stats(batch_samples, max_events=max_events)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch = apply_training_modality_dropout(
                batch,
                static_dropout_p=static_dropout_p,
                notes_dropout_p=notes_dropout_p,
                static_zero_indices=static_zero_indices,
            )
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["events"], batch["event_mask"], batch["static"], batch["notes"], batch["note_stats"])
            loss = criterion(out["factual_logits"], batch["labels"])
            total_loss = loss
            if order_aux_weight > 0:
                shuffled_loss = compute_order_margin_loss(
                    model,
                    batch_samples,
                    criterion,
                    device=device,
                    order_margin=order_aux_margin,
                    static_zero_indices=static_zero_indices,
                )
                order_aux = torch.relu(order_aux_margin + loss - shuffled_loss)
                total_loss = total_loss + order_aux_weight * order_aux
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(total_loss.item()))

        val_result = predict_ecdt(
            model,
            val_data,
            device=device,
            batch_size=batch_size,
            threshold=threshold,
            label_names=label_names,
            max_events=max_events,
        )
        val_auc = val_result["metrics"]["macro_auroc"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0, "val_macro_auroc": val_auc})
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return model, history


def train_predict_ovr(model_ctor, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    yb = (y_train > 0.5).astype(np.int32)
    preds = []
    for idx in range(yb.shape[1]):
        yj = yb[:, idx]
        if np.unique(yj).size < 2:
            preds.append(np.full(x_test.shape[0], float(yj[0]), dtype=np.float32))
            continue
        model = model_ctor()
        model.fit(x_train, yj)
        preds.append(model.predict_proba(x_test)[:, 1].astype(np.float32))
    return np.stack(preds, axis=1)


def train_deep_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: str,
    epochs: int,
    hidden_dim: int,
) -> TransformerTabularBaseline:
    model = TransformerTabularBaseline(x_train.shape[1], hidden_dim=hidden_dim, n_outputs=y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    ds_x = torch.tensor(np.nan_to_num(x_train), dtype=torch.float32)
    ds_y = torch.tensor((y_train > 0.5).astype(np.float32), dtype=torch.float32)
    val_x = torch.tensor(np.nan_to_num(x_val), dtype=torch.float32, device=device)
    val_y = (y_val > 0.5).astype(np.int32)
    best_auc = -1.0
    best_state = None
    patience = 2
    stale = 0

    for _ in tqdm(range(epochs), desc="Training Tabular Baseline", leave=False):
        model.train()
        indices = torch.randperm(ds_x.shape[0])
        for start in range(0, ds_x.shape[0], 64):
            batch_idx = indices[start:start + 64]
            xb = ds_x[batch_idx].to(device)
            yb = ds_y[batch_idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(val_x)).cpu().numpy()
        metrics = multilabel_metrics(val_y, probs)
        if metrics["macro_auroc"] > best_auc + 1e-4:
            best_auc = metrics["macro_auroc"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_deep_baseline(model: nn.Module, x: np.ndarray, y: np.ndarray, *, device: str, threshold: float, label_names: Sequence[str]) -> Dict:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(np.nan_to_num(x), dtype=torch.float32, device=device))).cpu().numpy()
    return {"probs": probs, "labels": y, "metrics": multilabel_metrics(y, probs, label_names=label_names, threshold=threshold)}


def baseline_ctors() -> Dict[str, callable]:
    return {
        "LogisticRegression": lambda: LogisticRegression(
            max_iter=300,
            solver="liblinear",
            class_weight="balanced",
        ),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=180, random_state=42, n_jobs=-1, class_weight="balanced"),
        "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=180, random_state=42, n_jobs=-1, class_weight="balanced"),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def make_joint_train(source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray, target_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.vstack([source_x, target_x]), np.vstack([source_y, target_y])


def resolve_model_impl(model_impl_arg: str, ckpt_path: Path) -> str:
    if model_impl_arg != "auto":
        return model_impl_arg
    env_impl = os.environ.get("TRAIN_MODEL_IMPL", "").strip().lower()
    if env_impl == "stable_branch":
        return "stable_branch"
    ckpt_text = str(ckpt_path).lower()
    if "stable_baseline_branch" in ckpt_text or "stable_baseline_internal" in ckpt_text:
        return "stable_branch"
    return "v5"


def split_target_train_val(
    pool: Sequence[Dict],
    n_target: int,
    *,
    seed: int,
    mode: str,
    val_ratio: float,
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    target_train = list(pool[:n_target])
    if mode == "pool_remainder":
        target_val = list(pool[n_target:])
        if len(target_val) == 0:
            mode = "within_target"
        else:
            return target_train, target_val, {
                "n_target": int(n_target),
                "n_target_train": int(len(target_train)),
                "n_target_val": int(len(target_val)),
                "val_mode": "pool_remainder",
            }

    target = list(pool[:n_target])
    target_sig = make_signature_array(target)
    target_idx = np.arange(len(target))
    bincount = np.bincount(target_sig) if len(target_sig) > 0 else np.zeros(1, dtype=np.int64)
    stratify = target_sig if np.min(bincount) >= 2 else None
    tr_idx, va_idx = train_test_split(target_idx, test_size=val_ratio, random_state=seed, stratify=stratify)
    target_train = [target[i] for i in tr_idx]
    target_val = [target[i] for i in va_idx]
    return target_train, target_val, {
        "n_target": int(n_target),
        "n_target_train": int(len(target_train)),
        "n_target_val": int(len(target_val)),
        "val_mode": "within_target",
    }


def rows_from_metrics(*, phase: str, model: str, seed: int, fraction: float, metrics: Dict) -> Dict:
    return {
        "phase": phase,
        "model": model,
        "seed": seed,
        "fewshot_fraction": fraction,
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "macro_brier": metrics["macro_brier"],
        "macro_ece": metrics["macro_ece"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "rare_label_recall": metrics["rare_label_recall"],
        "subset_accuracy": metrics["subset_accuracy"],
        "n_samples": metrics["n_samples"],
    }


def aggregate_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    numeric_cols = [
        "macro_auroc",
        "macro_auprc",
        "macro_brier",
        "macro_ece",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "rare_label_recall",
        "subset_accuracy",
    ]
    rows = []
    for keys, grp in df.groupby(group_cols):
        row = {}
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, value in zip(group_cols, keys):
            row[col] = value
        for col in numeric_cols:
            row[f"{col}_mean"] = float(grp[col].mean())
            row[f"{col}_std"] = float(grp[col].std(ddof=0))
        row["n_seeds"] = int(grp["seed"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = ExperimentRegistry(out_dir)
    device = default_device(args.device)
    static_shortcut_indices = parse_index_list(args.static_shortcut_indices)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
    if args.quick_smoke:
        args.mimic_limit = args.mimic_limit or 512
        args.eicu_limit = args.eicu_limit or 1024
        args.epochs = min(args.epochs, 1)
        args.deep_epochs = min(args.deep_epochs, 2)
        seeds = seeds[:1]
        fractions = fractions[:1]

    log.info(
        "Starting external validation | out_dir=%s device=%s model_impl_arg=%s fractions=%s quick_smoke=%s",
        out_dir,
        device,
        args.model_impl,
        fractions,
        bool(args.quick_smoke),
    )
    log.info("Loading MIMIC data | limit=%s", args.mimic_limit)
    mimic_data_raw = bm.load_mimic_data(limit=args.mimic_limit)
    log.info("Cross-Cohort Alignment: Reducing MIMIC features to 38 core variables to match eICU.")
    mimic_data_filtered = filter_mimic_to_aligned(mimic_data_raw)
    mimic_data = add_note_signal(mimic_data_filtered)
    log.info("Loaded and aligned MIMIC data | n=%s", len(mimic_data))
    log.info("Loading eICU data | limit=%s", args.eicu_limit)
    eicu_data_raw, _, label_names = bm.load_eicu_data(limit=args.eicu_limit)
    log.info("Aligning eICU features to MIMIC vocabulary")
    eicu_data_raw = align_eicu_to_mimic(eicu_data_raw)
    eicu_data = add_note_signal(eicu_data_raw)
    log.info("Loaded eICU data | n=%s", len(eicu_data))
    mimic_fp = dataset_fingerprint(mimic_data)
    eicu_fp = dataset_fingerprint(eicu_data)
    ckpt_path = resolve_v5_checkpoint()
    model_impl = resolve_model_impl(args.model_impl, ckpt_path)
    log.info("Resolved checkpoint=%s | effective_model_impl=%s", ckpt_path, model_impl)

    all_rows: List[Dict] = []
    per_label_rows: List[Dict] = []
    seed_payloads: List[Dict] = []

    for seed in seeds:
        log.info("Running seed=%s", seed)
        set_seed(seed)
        mimic_train, mimic_test, mimic_train_idx, mimic_test_idx = stratified_split(mimic_data, 0.30, seed)
        eicu_pool, eicu_test, eicu_pool_idx, eicu_test_idx = stratified_split(eicu_data, 0.30, seed)
        log.info(
            "Seed=%s split sizes | mimic_train=%s mimic_test=%s eicu_pool=%s eicu_test=%s",
            seed,
            len(mimic_train),
            len(mimic_test),
            len(eicu_pool),
            len(eicu_test),
        )

        x_mi_tr, y_mi_tr = bm.make_tabular_features(mimic_train)
        x_mi_te, y_mi_te = bm.make_tabular_features(mimic_test)
        x_ei_te, y_ei_te = bm.make_tabular_features(eicu_test)

        seed_result = {
            "seed": seed,
            "splits": {
                "mimic_train_idx": mimic_train_idx.tolist(),
                "mimic_test_idx": mimic_test_idx.tolist(),
                "eicu_pool_idx": eicu_pool_idx.tolist(),
                "eicu_test_idx": eicu_test_idx.tolist(),
            },
            "models": {},
        }

        # Tabular baselines: internal + zero-shot.
        log.info("Seed=%s starting internal + zero-shot tabular baselines", seed)
        for model_name, ctor in baseline_ctors().items():
            yhat_mi = train_predict_ovr(ctor, x_mi_tr, y_mi_tr, x_mi_te)
            yhat_ei = train_predict_ovr(ctor, x_mi_tr, y_mi_tr, x_ei_te)
            mi_metrics = multilabel_metrics(y_mi_te, yhat_mi, label_names=label_names, threshold=args.threshold)
            ei_metrics = multilabel_metrics(y_ei_te, yhat_ei, label_names=label_names, threshold=args.threshold)
            all_rows.append(rows_from_metrics(phase="internal", model=model_name, seed=seed, fraction=0.0, metrics=mi_metrics))
            all_rows.append(rows_from_metrics(phase="external_zero_shot", model=model_name, seed=seed, fraction=0.0, metrics=ei_metrics))
            per_label_rows.extend(flatten_per_label_rows(experiment="external_real", split_name="internal", model_name=model_name, seed=seed, metrics=mi_metrics))
            per_label_rows.extend(flatten_per_label_rows(experiment="external_real", split_name="external_zero_shot", model_name=model_name, seed=seed, metrics=ei_metrics))
            seed_result["models"].setdefault(model_name, {})
            seed_result["models"][model_name]["internal"] = mi_metrics
            seed_result["models"][model_name]["external_zero_shot"] = ei_metrics

        # Deep baseline.
        log.info("Seed=%s starting internal + zero-shot deep baseline", seed)
        deep_model = train_deep_baseline(
            x_mi_tr,
            y_mi_tr,
            x_mi_te,
            y_mi_te,
            device=device,
            epochs=args.deep_epochs,
            hidden_dim=args.deep_hidden_dim,
        )
        deep_internal = predict_deep_baseline(deep_model, x_mi_te, y_mi_te, device=device, threshold=args.threshold, label_names=label_names)
        deep_zero = predict_deep_baseline(deep_model, x_ei_te, y_ei_te, device=device, threshold=args.threshold, label_names=label_names)
        all_rows.append(rows_from_metrics(phase="internal", model="TransformerTabular", seed=seed, fraction=0.0, metrics=deep_internal["metrics"]))
        all_rows.append(rows_from_metrics(phase="external_zero_shot", model="TransformerTabular", seed=seed, fraction=0.0, metrics=deep_zero["metrics"]))
        per_label_rows.extend(flatten_per_label_rows(experiment="external_real", split_name="internal", model_name="TransformerTabular", seed=seed, metrics=deep_internal["metrics"]))
        per_label_rows.extend(flatten_per_label_rows(experiment="external_real", split_name="external_zero_shot", model_name="TransformerTabular", seed=seed, metrics=deep_zero["metrics"]))
        seed_result["models"]["TransformerTabular"] = {
            "internal": deep_internal["metrics"],
            "external_zero_shot": deep_zero["metrics"],
        }

        # ECDT full, raw notes and notes-masked ablation.
        log.info("Seed=%s starting ECDT internal + zero-shot evaluation", seed)
        ecdt = build_ecdt(device=device, model_impl=model_impl)
        ckpt_load = load_checkpoint_flex(ecdt, ckpt_path, device)
        ecdt_internal = predict_ecdt(
            ecdt,
            mimic_test,
            device=device,
            batch_size=args.batch_size,
            threshold=args.threshold,
            label_names=label_names,
            max_events=args.max_events,
        )
        ecdt_zero = predict_ecdt(
            ecdt,
            eicu_test,
            device=device,
            batch_size=args.batch_size,
            threshold=args.threshold,
            label_names=label_names,
            max_events=args.max_events,
        )
        masked_zero = predict_ecdt(
            ecdt,
            mask_notes(eicu_test),
            device=device,
            batch_size=args.batch_size,
            threshold=args.threshold,
            label_names=label_names,
            max_events=args.max_events,
        )
        for phase_name, metrics in (
            ("internal", ecdt_internal["metrics"]),
            ("external_zero_shot", ecdt_zero["metrics"]),
            ("external_zero_shot_notes_masked", masked_zero["metrics"]),
        ):
            all_rows.append(rows_from_metrics(phase=phase_name, model="ECDT_Full", seed=seed, fraction=0.0, metrics=metrics))
            per_label_rows.extend(flatten_per_label_rows(experiment="external_real", split_name=phase_name, model_name="ECDT_Full", seed=seed, metrics=metrics))
        seed_result["models"]["ECDT_Full"] = {
            "checkpoint": str(ckpt_path),
            "checkpoint_load": ckpt_load,
            "internal": ecdt_internal["metrics"],
            "external_zero_shot": ecdt_zero["metrics"],
            "external_zero_shot_notes_masked": masked_zero["metrics"],
        }

        # Few-shot adaptation.
        order = np.random.default_rng(seed).permutation(len(eicu_pool))
        pool = [eicu_pool[i] for i in order]
        x_pool, y_pool = bm.make_tabular_features(pool)
        for fraction in fractions:
            n_target = max(128, int(round(fraction * len(eicu_pool))))
            n_target = min(n_target, len(pool))
            log.info(
                "Seed=%s fraction=%.3f starting few-shot | n_target=%s val_mode=%s align_budget=%s max_events=%s",
                seed,
                fraction,
                n_target,
                args.target_val_mode,
                bool(args.align_baseline_target_budget),
                args.max_events,
            )
            target_train, target_val, split_meta = split_target_train_val(
                pool,
                n_target,
                seed=seed,
                mode=args.target_val_mode,
                val_ratio=args.target_val_ratio,
            )

            # ECDT replay training.
            log.info(
                "Seed=%s fraction=%.3f ECDT finetune | target_train=%s target_val=%s replay=%s epochs=%s patience=%s lr=%s",
                seed,
                fraction,
                len(target_train),
                len(target_val),
                int(len(target_train) * args.replay_ratio),
                args.epochs,
                args.finetune_patience,
                args.finetune_lr,
            )
            n_replay = int(len(target_train) * args.replay_ratio)
            replay = mimic_train if n_replay >= len(mimic_train) else random.sample(mimic_train, n_replay)
            ecdt_ft = build_ecdt(device=device, model_impl=model_impl)
            load_checkpoint_flex(ecdt_ft, ckpt_path, device)
            ecdt_ft, history = finetune_ecdt(
                ecdt_ft,
                train_data=list(target_train) + replay,
                val_data=target_val,
                device=device,
                epochs=args.epochs,
                batch_size=min(args.batch_size, 32),
                threshold=args.threshold,
                label_names=label_names,
                static_dropout_p=args.static_dropout_p,
                notes_dropout_p=args.notes_dropout_p,
                static_zero_indices=static_shortcut_indices,
                order_aux_weight=args.order_aux_weight,
                order_aux_margin=args.order_aux_margin,
                lr=args.finetune_lr,
                patience=args.finetune_patience,
                max_events=args.max_events,
            )
            ecdt_ft_eval = predict_ecdt(
                ecdt_ft,
                eicu_test,
                device=device,
                batch_size=args.batch_size,
                threshold=args.threshold,
                label_names=label_names,
                max_events=args.max_events,
            )
            all_rows.append(rows_from_metrics(phase="external_few_shot", model="ECDT_Full", seed=seed, fraction=fraction, metrics=ecdt_ft_eval["metrics"]))
            per_label_rows.extend(flatten_per_label_rows(experiment="external_real", split_name=f"external_few_shot_{int(fraction*100)}pct", model_name="ECDT_Full", seed=seed, metrics=ecdt_ft_eval["metrics"]))
            seed_result["models"]["ECDT_Full"].setdefault("fewshot", {})[f"{int(fraction*100)}pct"] = {
                **split_meta,
                "history": history,
                "metrics": ecdt_ft_eval["metrics"],
            }
            log.info(
                "Seed=%s fraction=%.3f ECDT done | macro_auroc=%.4f macro_auprc=%.4f",
                seed,
                fraction,
                ecdt_ft_eval["metrics"]["macro_auroc"],
                ecdt_ft_eval["metrics"]["macro_auprc"],
            )

            # Baselines with joint refit.
            if args.align_baseline_target_budget:
                x_ft, y_ft = bm.make_tabular_features(target_train)
            else:
                x_ft, y_ft = x_pool[:n_target], y_pool[:n_target]
            x_joint, y_joint = make_joint_train(x_mi_tr, y_mi_tr, x_ft, y_ft)
            log.info(
                "Seed=%s fraction=%.3f starting tabular/deep refit | baseline_target_n=%s",
                seed,
                fraction,
                x_ft.shape[0],
            )
            for model_name, ctor in baseline_ctors().items():
                yhat = train_predict_ovr(ctor, x_joint, y_joint, x_ei_te)
                metrics = multilabel_metrics(y_ei_te, yhat, label_names=label_names, threshold=args.threshold)
                all_rows.append(rows_from_metrics(phase="external_few_shot", model=model_name, seed=seed, fraction=fraction, metrics=metrics))
                per_label_rows.extend(flatten_per_label_rows(experiment="external_real", split_name=f"external_few_shot_{int(fraction*100)}pct", model_name=model_name, seed=seed, metrics=metrics))
                seed_result["models"][model_name].setdefault("fewshot", {})[f"{int(fraction*100)}pct"] = {
                    **split_meta,
                    "baseline_target_budget_aligned": bool(args.align_baseline_target_budget),
                    "metrics": metrics,
                }
                log.info(
                    "Seed=%s fraction=%.3f %s done | macro_auroc=%.4f macro_auprc=%.4f",
                    seed,
                    fraction,
                    model_name,
                    metrics["macro_auroc"],
                    metrics["macro_auprc"],
                )

            if len(target_val):
                x_val, y_val = bm.make_tabular_features(target_val)
            else:
                x_val, y_val = x_ft, y_ft
            deep_ft = train_deep_baseline(
                x_joint,
                y_joint,
                x_val,
                y_val,
                device=device,
                epochs=max(2, args.deep_epochs),
                hidden_dim=args.deep_hidden_dim,
            )
            deep_ft_eval = predict_deep_baseline(deep_ft, x_ei_te, y_ei_te, device=device, threshold=args.threshold, label_names=label_names)
            all_rows.append(rows_from_metrics(phase="external_few_shot", model="TransformerTabular", seed=seed, fraction=fraction, metrics=deep_ft_eval["metrics"]))
            per_label_rows.extend(flatten_per_label_rows(experiment="external_real", split_name=f"external_few_shot_{int(fraction*100)}pct", model_name="TransformerTabular", seed=seed, metrics=deep_ft_eval["metrics"]))
            seed_result["models"]["TransformerTabular"].setdefault("fewshot", {})[f"{int(fraction*100)}pct"] = {
                **split_meta,
                "baseline_target_budget_aligned": bool(args.align_baseline_target_budget),
                "metrics": deep_ft_eval["metrics"],
            }
            log.info(
                "Seed=%s fraction=%.3f TransformerTabular done | macro_auroc=%.4f macro_auprc=%.4f",
                seed,
                fraction,
                deep_ft_eval["metrics"]["macro_auroc"],
                deep_ft_eval["metrics"]["macro_auprc"],
            )

        seed_payloads.append(seed_result)
        log.info("Seed=%s complete", seed)

    raw_rows_path = out_dir / "external_validation_seed_rows.csv"
    pd.DataFrame(all_rows).to_csv(raw_rows_path, index=False)
    per_label_path = out_dir / "external_validation_per_label.csv"
    pd.DataFrame(per_label_rows).to_csv(per_label_path, index=False)

    df = pd.DataFrame(all_rows)
    internal_table = aggregate_table(df[df["phase"] == "internal"], ["model"])
    zero_table = aggregate_table(df[df["phase"] == "external_zero_shot"], ["model"])
    masked_table = aggregate_table(df[df["phase"] == "external_zero_shot_notes_masked"], ["model"])
    fewshot_table = aggregate_table(df[df["phase"] == "external_few_shot"], ["model", "fewshot_fraction"])
    internal_path = out_dir / "table_internal_validation.csv"
    zero_path = out_dir / "table_external_zero_shot.csv"
    masked_path = out_dir / "table_external_zero_shot_notes_masked.csv"
    fewshot_path = out_dir / "table_external_fewshot_curve.csv"
    internal_table.to_csv(internal_path, index=False)
    zero_table.to_csv(zero_path, index=False)
    masked_table.to_csv(masked_path, index=False)
    fewshot_table.to_csv(fewshot_path, index=False)

    summary = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "config": {
            "seeds": seeds,
            "fractions": fractions,
            "device": device,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "deep_epochs": args.deep_epochs,
            "replay_ratio": args.replay_ratio,
            "finetune_lr": args.finetune_lr,
            "finetune_patience": args.finetune_patience,
            "static_dropout_p": args.static_dropout_p,
            "notes_dropout_p": args.notes_dropout_p,
            "order_aux_weight": args.order_aux_weight,
            "order_aux_margin": args.order_aux_margin,
            "max_events": args.max_events,
            "target_val_ratio": args.target_val_ratio,
            "target_val_mode": args.target_val_mode,
            "align_baseline_target_budget": bool(args.align_baseline_target_budget),
            "model_impl": model_impl,
            "static_shortcut_indices": static_shortcut_indices,
            "threshold": args.threshold,
            "mimic_limit": args.mimic_limit,
            "eicu_limit": args.eicu_limit,
        },
        "paths": {
            "checkpoint": str(ckpt_path),
        },
        "data": {
            "mimic_n": len(mimic_data),
            "eicu_n": len(eicu_data),
            "mimic_fingerprint": mimic_fp,
            "eicu_fingerprint": eicu_fp,
            "labels": label_names,
        },
        "artifacts": {
            "seed_rows_csv": str(raw_rows_path),
            "per_label_csv": str(per_label_path),
            "internal_table_csv": str(internal_path),
            "external_zero_shot_csv": str(zero_path),
            "external_zero_shot_notes_masked_csv": str(masked_path),
            "external_fewshot_curve_csv": str(fewshot_path),
        },
        "seed_payloads_json": str(out_dir / "seed_payloads.json"),
    }
    (out_dir / "seed_payloads.json").write_text(json.dumps(seed_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = out_dir / "external_validation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    registry.register(
        experiment="npj_real_external_validation",
        script_path=THIS_DIR / "npj_external_validation_real.py",
        seed=None,
        checkpoint=ckpt_path,
        mimic_fingerprint=mimic_fp,
        eicu_fingerprint=eicu_fp,
        summary_json=summary_path,
        table_csv=fewshot_path,
        extra={
            "tables": {
                "internal": str(internal_path),
                "zero_shot": str(zero_path),
                "zero_shot_notes_masked": str(masked_path),
                "fewshot": str(fewshot_path),
            },
            "seed_count": len(seeds),
        },
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
