import argparse
import logging
import os
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
PACKAGE_DIR = ROOT_DIR.parent
RUNTIME_DIR = BASE_DIR / "runtime"
MODEL_CODE_DIR = RUNTIME_DIR / "paper_experiments_v2"

# Load the versioned local runtime shipped with this release before importing
# the training and architecture modules.
if not RUNTIME_DIR.is_dir() or not MODEL_CODE_DIR.is_dir():
    raise FileNotFoundError(
        "Missing code/training/runtime. Restore the vendored training runtime "
        "before invoking this training entry point."
    )
sys.path.insert(0, str(RUNTIME_DIR))
sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("MODELS_CODE_DIR", str(MODEL_CODE_DIR))
DEFAULT_MIMIC_DATA_DIR = PACKAGE_DIR / "data" / "restricted" / "mimic_v3_event_bundle"
os.environ.setdefault("MIMIC_DATA_DIR", str(DEFAULT_MIMIC_DATA_DIR))
os.environ.setdefault("EICU_DATA_PATH", str(PACKAGE_DIR / "data" / "restricted" / "eicu_event_bundle.joblib"))

from typing import Dict, Sequence, List
import numpy as np
import torch
from tqdm import tqdm

from npj_external_validation_v3 import (
    build_ecdt,
    collate_with_note_stats,
    apply_training_modality_dropout,
    compute_order_margin_loss,
    predict_ecdt,
    FocalBCE,
    stratified_split_3way,
)
from npj_runtime import configure_environment, set_seed
import benchmark_mimic_to_eicu_baselines as bm

configure_environment(require_data=False)
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

def setup_logging(log_path: Path) -> None:
    root = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in root.handlers):
        return
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(file_handler)

def add_note_signal(data: Sequence[Dict]) -> List[Dict]:
    from npj_external_validation_v3 import clone_samples
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

def run_multi_seed_pretraining(
    seeds: List[int],
    data_dir: str,
    out_base_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    resume: bool,
):
    for seed in seeds:
        log.info(f"\n{'='*20} STARTING SEED {seed} {'='*20}")
        output_model_path = str(out_base_dir / f"ecdt_v3_seed{seed}_best.pt")
        pretrain_ecdt_from_scratch(
            data_dir=data_dir,
            output_model_path=output_model_path,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            lr=lr,  # Aligned with v5_training run (was 5e-4 -> too low, AUROC capped at ~0.825)
            resume_from=output_model_path if resume else None,
        )

def pretrain_ecdt_from_scratch(
    data_dir: str,
    output_model_path: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1.2e-3,  # Aligned with v5_training run
    mixup_alpha: float = 0.4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    model_kwargs: dict = None,
    resume_from: str | None = None,
):
    log.info(f"Pre-training ECDT V3 | Seed: {seed} | Data: {data_dir}")
    if model_kwargs:
        log.info(f"Using custom model_kwargs (Ablation mode): {model_kwargs}")
    
    # Override MIMIC_DATA_DIR for bm.load_mimic_data
    os.environ["MIMIC_DATA_DIR"] = data_dir
    
    mimic_data_raw = bm.load_mimic_data(limit=0)
    mimic_data = add_note_signal(mimic_data_raw)
    
    set_seed(seed)
    
    mimic_train, mimic_val, mimic_test = stratified_split_3way(mimic_data, 0.20, 0.10, seed)
    log.info(f"Training set: {len(mimic_train)}, Validation set: {len(mimic_val)}, Test set: {len(mimic_test)}")
    
    if model_kwargs is None:
        model_kwargs = {}
    model = build_ecdt(device=device, model_impl="v4", **model_kwargs)
    if resume_from:
        ckpt_path = Path(resume_from)
        if ckpt_path.exists():
            state = torch.load(str(ckpt_path), map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            log.info("Loaded resume checkpoint for Seed %s from %s", seed, ckpt_path)
        else:
            log.warning("Resume checkpoint not found for Seed %s: %s", seed, ckpt_path)
    
    label_names = ['heart_failure', 'renal_failure', 'infection', 'pneumonia', 'cerebrovascular', 'diabetic_foot']
    
    criterion = FocalBCE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    best_auc = -1.0
    best_state = None
    patience = 15
    stale = 0
    
    steps_per_epoch = (len(mimic_train) + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.1, anneal_strategy='cos'
    )
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_list = list(mimic_train)
        random.shuffle(train_list)
        losses = []
        
        for start in tqdm(range(0, len(train_list), batch_size), desc=f"Seed {seed} Epoch {epoch}/{epochs}", file=sys.stdout):
            batch_samples = train_list[start:start + batch_size]
            batch = collate_with_note_stats(batch_samples, max_events=512)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Modality dropout restored to match requested regularization settings.
            batch = apply_training_modality_dropout(
                batch,
                static_dropout_p=0.3,
                notes_dropout_p=0.5,
            )

            optimizer.zero_grad(set_to_none=True)
            out = model(batch["events"], batch["event_mask"], batch["static"], batch["notes"], batch["note_stats"])
            loss = criterion(out["factual_logits"], batch["labels"])

            # MixUp: blend static/notes/note_stats and labels within the batch.
            if mixup_alpha > 0.0 and batch["static"].shape[0] > 1:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(batch["static"].shape[0], device=batch["static"].device)
                static_mix = lam * batch["static"] + (1.0 - lam) * batch["static"][perm]
                notes_mix = lam * batch["notes"] + (1.0 - lam) * batch["notes"][perm]
                note_stats_mix = lam * batch["note_stats"] + (1.0 - lam) * batch["note_stats"][perm]
                labels_mix = lam * batch["labels"] + (1.0 - lam) * batch["labels"][perm]

                out_mix = model(
                    batch["events"],
                    batch["event_mask"],
                    static_mix,
                    notes_mix,
                    note_stats_mix,
                )
                loss_mix = criterion(out_mix["factual_logits"], labels_mix)
                loss = 0.5 * loss + 0.5 * loss_mix

            # Temporal Order Margin Aux Loss - restore requested coefficient.
            shuffled_loss = compute_order_margin_loss(
                model, batch_samples, criterion, device=device, order_margin=0.05
            )
            order_aux = torch.relu(0.05 + loss - shuffled_loss)
            total_loss = loss + 0.1 * order_aux

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(total_loss.item()))
            
        # Validation
        val_result = predict_ecdt(
            model, mimic_val, device=device, batch_size=batch_size, threshold=0.5, label_names=label_names, max_events=512
        )
        val_auc = val_result["metrics"]["macro_auroc"]
        
        test_result = predict_ecdt(
            model, mimic_test, device=device, batch_size=batch_size, threshold=0.5, label_names=label_names, max_events=512
        )
        test_auc = test_result["metrics"]["macro_auroc"]
        log.info(f"Seed {seed} Epoch {epoch}: Loss = {np.mean(losses):.4f}, Val AUROC = {val_auc:.4f}, Test AUROC = {test_auc:.4f}")
        
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            torch.save({"state_dict": best_state}, output_model_path)
            log.info(f"--> Saved best model for Seed {seed} to {output_model_path}")
        else:
            stale += 1
            if stale >= patience:
                log.info(f"Early stopping for Seed {seed} at epoch {epoch}")
                break

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECDT V3 multi-seed pre-training")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_seed_list(text: str) -> List[int]:
    seeds: List[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            seeds.append(int(chunk))
        except ValueError:
            log.warning("Ignoring non-integer seed: %s", chunk)
    return seeds


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(__file__).resolve().parent / "models"
    out_dir.mkdir(exist_ok=True)

    log_path = out_dir / "train_ecdt_v3_scratch.log"
    setup_logging(log_path)

    log.info("Starting multi-seed pre-training...")

    seeds = parse_seed_list(args.seeds)
    run_multi_seed_pretraining(
        seeds=seeds,
        data_dir=os.environ["MIMIC_DATA_DIR"],
        out_base_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
    )
