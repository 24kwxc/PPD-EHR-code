import os
import sys
import json
import time
import types
import random
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

MODELS_CODE_DIR = os.environ.get('MODELS_CODE_DIR', '').strip()
if MODELS_CODE_DIR:
    MODELS_ROOT = Path(MODELS_CODE_DIR).expanduser()
else:
    MODELS_ROOT = PROJECT_ROOT / 'paper_experiments_v2'

if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from models.enhanced_cdt_v5 import (
    EnhancedCausalDigitalTwinV5,
    EnhancedCausalDigitalTwinV6,
    EnhancedCausalDigitalTwinV6Lite,
)
from models.data_utils import collate_events

# Import SOTA Transformer Baselines
try:
    if str(MODELS_ROOT) not in sys.path:
        sys.path.insert(0, str(MODELS_ROOT))
    from transformer_baselines import MedBERT, ClinicalBERTForSequence
except ImportError:
    # Fallback if not in path
    MedBERT = None
    ClinicalBERTForSequence = None

OUT_DIR = Path(os.environ.get('OUTPUT_DIR', str(PROJECT_ROOT / 'results'))).expanduser() / 'baseline_benchmark_mimic_eicu'
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    seed: int = int(os.environ.get('BASELINE_SEED', '42'))
    mimic_limit: int = int(os.environ.get('BASELINE_MIMIC_LIMIT', '0'))
    eicu_limit: int = int(os.environ.get('BASELINE_EICU_LIMIT', '0'))
    batch_size: int = int(os.environ.get('BASELINE_BATCH_SIZE', '64'))
    mimic_train_epochs: int = int(os.environ.get('BASELINE_MIMIC_EPOCHS', '3'))
    do_auto_finetune: bool = os.environ.get('BASELINE_AUTO_FINETUNE', '1').strip().lower() not in ('0', 'false', 'no', 'off')
    finetune_mode: str = os.environ.get('BASELINE_FINETUNE_MODE', 'two_stage').strip().lower()
    finetune_epochs: int = int(os.environ.get('BASELINE_FINETUNE_EPOCHS', '8'))
    finetune_lr: float = float(os.environ.get('BASELINE_FINETUNE_LR', '2e-4'))
    finetune_lr_probe: float = float(os.environ.get('BASELINE_FINETUNE_LR_PROBE', '8e-4'))
    finetune_probe_epochs: int = int(os.environ.get('BASELINE_FINETUNE_PROBE_EPOCHS', '2'))
    finetune_patience: int = int(os.environ.get('BASELINE_FINETUNE_PATIENCE', '3'))
    finetune_val_ratio: float = float(os.environ.get('BASELINE_FINETUNE_VAL_RATIO', '0.2'))
    replay_ratio: float = float(os.environ.get('BASELINE_REPLAY_RATIO', '0.3'))
    target_auc_floor: float = float(os.environ.get('TARGET_EICU_AUC', '0.70'))


CFG = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.asarray(x)


def _install_src_stub_module_for_eicu_unpickle():
    src_mod = types.ModuleType('src')
    prep_mod = types.ModuleType('src.preprocess')
    src_mod.__file__ = '<stub-src>'
    prep_mod.__file__ = '<stub-src-preprocess>'

    class _Stub:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.state = state

    def _safe_getattr(name):
        if name.startswith('__'):
            raise AttributeError(name)
        return _Stub

    prep_mod.__getattr__ = _safe_getattr
    sys.modules['src'] = src_mod
    sys.modules['src.preprocess'] = prep_mod


def _dense_to_sparse_events(matrix: np.ndarray, mask: Optional[np.ndarray], feature_offset: int = 0) -> Dict[str, List[float]]:
    arr = _to_numpy(matrix)
    if arr.ndim == 1:
        arr = arr[:, None]
    if mask is None:
        valid = np.ones_like(arr, dtype=bool)
    else:
        m = _to_numpy(mask)
        if m.shape != arr.shape:
            m = np.broadcast_to(m, arr.shape)
        valid = m > 0

    t_idx, f_idx = np.where(valid)
    vals = arr[t_idx, f_idx]
    keep = np.isfinite(vals) & (vals != 0)
    t_idx = t_idx[keep]
    f_idx = f_idx[keep]
    vals = vals[keep]
    return {
        't': t_idx.astype(np.int32).tolist(),
        'f': (f_idx.astype(np.int32) + int(feature_offset)).tolist(),
        'v': vals.astype(np.float32).tolist(),
    }


def _to_mimic_like_dict(sample_obj, default_static_dim: int = 18):
    id_like_keys = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "hospitalid",
        "hospitaldischargeyear",
        "stay_id",
        "subject_id",
        "hadm_id",
        "admission_year",
        "admit_year",
        "year",
    ]
    if isinstance(sample_obj, dict):
        out = {
            'timestamps': sample_obj.get('timestamps', np.arange(1)),
            'vitals': sample_obj.get('vitals', {'t': [], 'f': [], 'v': []}),
            'labs': sample_obj.get('labs', {'t': [], 'f': [], 'v': []}),
            'meds': sample_obj.get('meds', {'t': [], 'f': [], 'v': []}),
            'label': _to_numpy(sample_obj.get('label', np.zeros(6, dtype=np.float32))).astype(np.float32),
            'static': _to_numpy(sample_obj.get('static', np.zeros(default_static_dim, dtype=np.float32))).astype(np.float32),
            'note': _to_numpy(sample_obj.get('note', np.zeros(768, dtype=np.float32))).astype(np.float32),
        }
        for key in id_like_keys:
            if key in sample_obj:
                out[key] = sample_obj.get(key)
        return out

    timestamps = _to_numpy(getattr(sample_obj, 'timestamps', np.arange(1)))
    vitals = _dense_to_sparse_events(getattr(sample_obj, 'vitals', np.zeros((1, 1))), getattr(sample_obj, 'vitals_mask', None), 0)
    labs = _dense_to_sparse_events(getattr(sample_obj, 'labs', np.zeros((1, 1))), getattr(sample_obj, 'labs_mask', None), 0)
    meds = _dense_to_sparse_events(getattr(sample_obj, 'meds', np.zeros((1, 1))), getattr(sample_obj, 'meds_mask', None), 0)
    static = _to_numpy(getattr(sample_obj, 'static', np.zeros(default_static_dim)))
    if static.shape[0] < default_static_dim:
        static = np.pad(static, (0, default_static_dim - static.shape[0]))
    elif static.shape[0] > default_static_dim:
        static = static[:default_static_dim]
    note = _to_numpy(getattr(sample_obj, 'note', np.zeros(768, dtype=np.float32))).astype(np.float32)
    if note.shape[0] < 768:
        note = np.pad(note, (0, 768 - note.shape[0]))
    elif note.shape[0] > 768:
        note = note[:768]

    out = {
        'timestamps': timestamps,
        'vitals': vitals,
        'labs': labs,
        'meds': meds,
        'label': _to_numpy(getattr(sample_obj, 'label', np.zeros(6, dtype=np.float32))).astype(np.float32),
        'static': static.astype(np.float32),
        'note': note.astype(np.float32),
    }
    for key in id_like_keys:
        if hasattr(sample_obj, key):
            out[key] = getattr(sample_obj, key)
    return out


def load_mimic_data(limit: int = 0) -> List[Dict]:
    mimic_dir = Path(os.environ.get('MIMIC_DATA_DIR', str(PROJECT_ROOT / 'data' / 'mimic'))).expanduser()
    shard_files = sorted(mimic_dir.glob('shard_*.joblib'))
    data = []
    for shard in shard_files:
        shard_data = joblib.load(str(shard))
        data.extend(shard_data)
        if limit > 0 and len(data) >= limit:
            data = data[:limit]
            break
    return [_to_mimic_like_dict(x) for x in data]


def load_eicu_data(limit: int = 0) -> Tuple[List[Dict], Dict[str, List[str]], List[str]]:
    eicu_path = Path(os.environ.get('EICU_DATA_PATH', str(PROJECT_ROOT / 'data' / 'eicu' / 'eicu_bundle.joblib'))).expanduser()
    used_stub = False
    try:
        raw = joblib.load(str(eicu_path))
    except ModuleNotFoundError:
        _install_src_stub_module_for_eicu_unpickle()
        used_stub = True
        raw = joblib.load(str(eicu_path))
    finally:
        if used_stub:
            sys.modules.pop('src.preprocess', None)
            sys.modules.pop('src', None)

    if hasattr(raw, 'sequences'):
        seqs = list(raw.sequences)
        meta = {
            'vitals': list(getattr(getattr(raw, 'vitals_meta', None), 'feature_names', [])),
            'labs': list(getattr(getattr(raw, 'labs_meta', None), 'feature_names', [])),
            'meds': list(getattr(getattr(raw, 'meds_meta', None), 'feature_names', [])),
            'static': list(getattr(raw, 'static_feature_names', [])),
        }
        label_names = list(getattr(raw, 'label_names', [f'label_{i}' for i in range(6)]))
    elif isinstance(raw, dict) and 'sequences' in raw:
        seqs = list(raw['sequences'])
        meta = {'vitals': [], 'labs': [], 'meds': [], 'static': []}
        label_names = list(raw.get('label_names', [f'label_{i}' for i in range(6)]))
    else:
        seqs = list(raw)
        meta = {'vitals': [], 'labs': [], 'meds': [], 'static': []}
        label_names = [f'label_{i}' for i in range(6)]

    if limit > 0:
        seqs = seqs[:limit]
    data = [_to_mimic_like_dict(s) for s in seqs]
    return data, meta, label_names


def _safe_auc_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    aucs, aprs = [], []
    for j in range(y_true.shape[1]):
        pos = y_true[:, j].sum()
        if pos < 5 or pos >= len(y_true):
            continue
        aucs.append(roc_auc_score(y_true[:, j], y_prob[:, j]))
        aprs.append(average_precision_score(y_true[:, j], y_prob[:, j]))
    return {
        'auroc': float(np.mean(aucs)) if aucs else 0.0,
        'auprc': float(np.mean(aprs)) if aprs else 0.0,
        'n_labels_eval': int(len(aucs)),
    }


def _extract_stats_from_sparse(modality: Dict[str, List[float]]) -> np.ndarray:
    vals = np.asarray(modality.get('v', []), dtype=np.float32)
    if vals.size == 0:
        return np.zeros(5, dtype=np.float32)
    vals = np.clip(vals, -100, 1000)
    return np.array([vals.mean(), vals.std(), vals.min(), vals.max(), float(vals.size)], dtype=np.float32)


def make_tabular_features(data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for sample in data:
        static = np.asarray(sample['static'], dtype=np.float32)
        static = np.clip(static, -100, 1000)
        # Ensure fixed static dimension (18)
        if static.size < 18:
            static = np.pad(static, (0, 18 - static.size))
        static = static[:18]
        
        note = np.asarray(sample['note'], dtype=np.float32)
        if note.size < 768:
            note = np.pad(note, (0, 768 - note.size))
        note32 = note[:768].reshape(32, 24).mean(axis=1)
        
        v_stats = _extract_stats_from_sparse(sample['vitals'])
        l_stats = _extract_stats_from_sparse(sample['labs'])
        m_stats = _extract_stats_from_sparse(sample['meds'])
        
        feat = np.concatenate([static, v_stats, l_stats, m_stats, note32], axis=0)
        # Ensure consistent feature dimension: 18 + 5 + 5 + 5 + 32 = 65
        if feat.size < 65:
            feat = np.pad(feat, (0, 65 - feat.size))
        feat = feat[:65]
        
        xs.append(feat.astype(np.float32))
        ys.append(np.asarray(sample['label'], dtype=np.float32))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _feature_value_by_name_mimic(sample: Dict, name: str, mimic_index: Dict) -> Optional[float]:
    for modality_key, idx_key in [('vitals', 'vital_features'), ('labs', 'lab_features'), ('meds', 'med_features')]:
        names = mimic_index.get(idx_key, [])
        if not names:
            continue
        ids = [i for i, n in enumerate(names) if n == name]
        if not ids:
            continue
        target_ids = set(ids)
        mod = sample.get(modality_key, {})
        f = np.asarray(mod.get('f', []), dtype=np.int64)
        v = np.asarray(mod.get('v', []), dtype=np.float32)
        if f.size == 0:
            continue
        mask = np.isin(f, list(target_ids))
        if mask.any():
            vals = v[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                return float(np.nanmean(vals))
    return None


def _clinical_scores_mimic(sample: Dict, mimic_index: Dict) -> np.ndarray:
    apache_feats = [
        'apache_ii', 'apacheiii', 'apacheiv_mortality_prediction', 'aps', 'apache_ii_predecited_death_rate'
    ]
    sofa_feats = ['sofa_score', 'creatinine_serum', 'bilirubin_apacheiv', 'platelet_count', 'map_apacheiv']
    saps_proxy_feats = ['ageapacheiivalue', 'hr_apacheiv', 'rr_apacheiv', 'bun', 'wbc_apacheiv', 'sodium_apacheiv']

    def agg(names):
        vals = [
            _feature_value_by_name_mimic(sample, nm, mimic_index)
            for nm in names
        ]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            return 0.0
        return float(np.mean(vals))

    return np.array([agg(apache_feats), agg(sofa_feats), agg(saps_proxy_feats)], dtype=np.float32)


def _values_from_sparse_feature_id(sample: Dict, modality: str, feature_idx: int) -> np.ndarray:
    mod = sample.get(modality, {})
    f = np.asarray(mod.get('f', []), dtype=np.int64)
    v = np.asarray(mod.get('v', []), dtype=np.float32)
    if f.size == 0:
        return np.array([], dtype=np.float32)
    keep = f == int(feature_idx)
    vals = v[keep]
    vals = vals[np.isfinite(vals)]
    return vals


def _clinical_scores_eicu(sample: Dict, meta: Dict[str, List[str]]) -> np.ndarray:
    static_names = list(meta.get('static', []))
    s = np.asarray(sample['static'], dtype=np.float32)

    def static_get(name: str):
        if name in static_names:
            idx = static_names.index(name)
            if idx < s.size and np.isfinite(s[idx]):
                return float(s[idx])
        return None

    apache_vals = [static_get('apachescore'), static_get('predictedicumortality')]
    apache_vals = [v for v in apache_vals if v is not None]
    apache = float(np.mean(apache_vals)) if apache_vals else 0.0

    v_names = list(meta.get('vitals', []))
    l_names = list(meta.get('labs', []))
    m_names = list(meta.get('meds', []))

    def mod_mean(modality: str, names: List[str], target: str):
        if target not in names:
            return None
        idx = names.index(target)
        vals = _values_from_sparse_feature_id(sample, modality, idx)
        if vals.size == 0:
            return None
        return float(np.mean(vals))

    creat = mod_mean('labs', l_names, 'creatinine')
    bili = mod_mean('labs', l_names, 'bilirubin')
    plate = mod_mean('labs', l_names, 'platelets')
    mbp = mod_mean('vitals', v_names, 'mbp')
    vaso = mod_mean('meds', m_names, 'vasopressors')

    sofa_components = []
    if creat is not None:
        sofa_components.append(np.clip(creat / 2.0, 0, 4))
    if bili is not None:
        sofa_components.append(np.clip(bili / 2.0, 0, 4))
    if plate is not None and plate > 0:
        sofa_components.append(np.clip((150 - min(150, plate)) / 35.0, 0, 4))
    if mbp is not None:
        sofa_components.append(1.0 if mbp < 70 else 0.0)
    if vaso is not None and vaso > 0:
        sofa_components.append(2.0)
    sofa = float(np.sum(sofa_components)) if sofa_components else 0.0

    age = static_get('age') or 0.0
    hr = mod_mean('vitals', v_names, 'heart_rate') or 0.0
    rr = mod_mean('vitals', v_names, 'resp_rate') or 0.0
    bun = mod_mean('labs', l_names, 'bun') or 0.0
    wbc = mod_mean('labs', l_names, 'wbc') or 0.0
    saps = float(0.15 * age + 0.05 * hr + 0.05 * rr + 0.03 * bun + 0.02 * wbc)

    return np.array([apache, sofa, saps], dtype=np.float32)


def make_clinical_scores(data: List[Dict], domain: str, mimic_index: Optional[Dict] = None, eicu_meta: Optional[Dict] = None):
    xs, ys = [], []
    for s in data:
        if domain == 'mimic':
            x = _clinical_scores_mimic(s, mimic_index)
        else:
            x = _clinical_scores_eicu(s, eicu_meta)
        xs.append(x)
        ys.append(np.asarray(s['label'], dtype=np.float32))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_predict_one_vs_rest(model_ctor, x_train, y_train, x_test):
    models = []
    probs = []
    yb = (y_train > 0.5).astype(np.int32)
    for j in range(yb.shape[1]):
        yj = yb[:, j]
        if np.unique(yj).size < 2:
            probs.append(np.full(x_test.shape[0], float(yj[0]), dtype=np.float32))
            models.append(None)
            continue
        m = model_ctor()
        m.fit(x_train, yj)
        p = m.predict_proba(x_test)[:, 1]
        probs.append(p.astype(np.float32))
        models.append(m)
    return models, np.stack(probs, axis=1)


@torch.no_grad()
def eval_v6_model(model: nn.Module, data: List[Dict], device: str = 'cuda', batch_size: int = 64):
    model.eval()
    all_p, all_y = [], []
    is_transformer_baseline = hasattr(model, 'AVAILABLE_BASELINES') or 'MedBERT' in model.__class__.__name__ or 'ClinicalBERT' in model.__class__.__name__
    
    for i in range(0, len(data), batch_size):
        batch = collate_events(data[i:i + batch_size])
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        
        if is_transformer_baseline:
            # SOTA Baselines use 'events' as 'features'
            out = model(batch['events'], mask=batch['event_mask'])
            logits = out['logits']
        else:
            # ECDT Models
            out = model(
                batch['events'],
                batch['event_mask'],
                batch['static'],
                batch['notes'],
            )
            logits = out['factual_logits']
            
        p = torch.sigmoid(logits).cpu().numpy()
        y = batch['labels'].cpu().numpy()
        all_p.append(p)
        all_y.append(y)
    p = np.concatenate(all_p, axis=0)
    y = np.concatenate(all_y, axis=0)
    return _safe_auc_metrics((y > 0.5).astype(np.int32), p)


class FocalBCE(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, smoothing=0.02):
        super().__init__()
        self.alpha, self.gamma, self.s = alpha, gamma, smoothing

    def forward(self, logits, targets):
        targets_s = targets * (1 - self.s) + 0.5 * self.s
        bce = F.binary_cross_entropy_with_logits(logits, targets_s, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


def train_baseline_deep_model(model: nn.Module, train_data: List[Dict], val_data: List[Dict], device: str, epochs: int, lr: float):
    model.to(device)
    criterion = FocalBCE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    is_transformer_baseline = 'MedBERT' in model.__class__.__name__ or 'ClinicalBERT' in model.__class__.__name__

    best_auc = 0
    best_state = None
    
    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)
        bs = 2 if is_transformer_baseline else 32
        for i in range(0, len(train_data), bs):
            batch = collate_events(train_data[i:i+bs])
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            optimizer.zero_grad()
            
            if is_transformer_baseline:
                out = model(batch['events'], mask=batch['event_mask'])
                logits = out['logits']
            else:
                out = model(batch['events'], batch['event_mask'], batch['static'], batch['notes'])
                logits = out['factual_logits']
                
            loss = criterion(logits, batch['labels'])
            loss.backward()
            optimizer.step()
            
        metrics = eval_v6_model(model, val_data, device=device)
        if metrics['auroc'] > best_auc:
            best_auc = metrics['auroc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        log.info(f"[MIMIC Train] ep={ep}/{epochs} va_auc={metrics['auroc']:.4f}")
        
    if best_state:
        model.load_state_dict(best_state)
    return model


def finetune_v6(model: nn.Module, train_data: List[Dict], val_data: List[Dict], device: str):
    history = []
    is_transformer_baseline = 'MedBERT' in model.__class__.__name__ or 'ClinicalBERT' in model.__class__.__name__

    def _params_by_mode(mode: str):
        all_named = list(model.named_parameters())
        if mode == 'probe':
            picked = []
            for n, p in all_named:
                low = n.lower()
                if ('classifier' in low) or ('head' in low) or ('out' in low) or ('logit' in low):
                    picked.append(p)
            if len(picked) == 0:
                picked = [p for _, p in all_named]
            for _, p in all_named:
                p.requires_grad = False
            for p in picked:
                p.requires_grad = True
            return picked

        for _, p in all_named:
            p.requires_grad = True
        return [p for _, p in all_named]

    def _run_stage(stage_name: str, train_list: List[Dict], val_list: List[Dict], epochs: int, lr: float):
        if epochs <= 0:
            return
        params = _params_by_mode('probe' if stage_name == 'probe' else 'full')
        criterion = FocalBCE()
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        best_auc = -1.0
        best_state = None
        patience = 0

        for ep in range(1, epochs + 1):
            random.shuffle(train_list)
            model.train()
            losses = []
            for i in range(0, len(train_list), 32):
                batch = collate_events(train_list[i:i + 32])
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                
                if is_transformer_baseline:
                    out = model(batch['events'], mask=batch['event_mask'])
                    logits = out['logits']
                else:
                    out = model(batch['events'], batch['event_mask'], batch['static'], batch['notes'])
                    logits = out['factual_logits']
                    
                loss = criterion(logits, batch['labels'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.item()))

            va = eval_v6_model(model, val_list, device=device, batch_size=64)
            avg_loss = float(np.mean(losses)) if losses else 0.0
            log.info(f'[Finetune:{stage_name}] ep={ep}/{epochs} loss={avg_loss:.4f} va_auc={va["auroc"]:.4f}')
            history.append({
                'stage': stage_name,
                'epoch': ep,
                'loss': avg_loss,
                'val': va,
                'lr': lr,
            })
            if va['auroc'] > best_auc:
                best_auc = va['auroc']
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= CFG.finetune_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

    if CFG.finetune_mode == 'two_stage':
        _run_stage('probe', train_data, val_data, CFG.finetune_probe_epochs, CFG.finetune_lr_probe)
        _run_stage('full', train_data, val_data, CFG.finetune_epochs, CFG.finetune_lr)
    else:
        _run_stage('full', train_data, val_data, CFG.finetune_epochs, CFG.finetune_lr)

    return model, history


def _merge_with_replay(target_train: List[Dict], source_replay: Optional[List[Dict]]) -> List[Dict]:
    if not source_replay or CFG.replay_ratio <= 0:
        return list(target_train)
    n_replay = int(len(target_train) * CFG.replay_ratio)
    if n_replay <= 0:
        return list(target_train)
    picked = list(source_replay) if len(source_replay) <= n_replay else random.sample(source_replay, n_replay)
    mixed = list(target_train) + picked
    random.shuffle(mixed)
    return mixed


def _build_model_from_ckpt_name(ckpt_path: Path):
    name = ckpt_path.stem.lower()
    full = str(ckpt_path).lower()
    if 'v6_lite' in name or 'v6_lite' in full:
        return EnhancedCausalDigitalTwinV6Lite(n_outputs=6, d_model=128, use_domain_adaptation=False)
    if 'v5' in name or '/v5/' in full:
        return EnhancedCausalDigitalTwinV5(
            n_outputs=6,
            d_model=128,
            use_gnn=True,
            use_diffusion=True,
            use_domain_adaptation=False,
        )
    return EnhancedCausalDigitalTwinV6(
        n_outputs=6,
        d_model=128,
        use_gnn=True,
        use_diffusion=True,
        use_domain_adaptation=False,
        n_fusion_layers=2,
        n_heads=8,
    )


def evaluate_v6_checkpoint(
    ckpt_path: Path,
    mimic_test: List[Dict],
    eicu_data: List[Dict],
    eicu_tune: Optional[List[Dict]] = None,
    source_replay: Optional[List[Dict]] = None,
    model_tag: str = 'model',
):
    if not ckpt_path.exists():
        return {'error': f'checkpoint not found: {ckpt_path}'}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(str(ckpt_path), map_location=device)
    model = _build_model_from_ckpt_name(ckpt_path).to(device)
    model.load_state_dict(state, strict=False)
    zero_mimic = eval_v6_model(model, mimic_test, device=device, batch_size=CFG.batch_size)
    zero_eicu = eval_v6_model(model, eicu_data, device=device, batch_size=CFG.batch_size)

    result = {
        'mimic_test': zero_mimic,
        'eicu_zeroshot': zero_eicu,
    }

    if CFG.do_auto_finetune and eicu_tune is not None and len(eicu_tune) > 128:
        y = np.array([int(np.dot((np.asarray(s['label']) > 0.5).astype(np.int32), (1 << np.arange(6)))) for s in eicu_tune])
        idx = np.arange(len(eicu_tune))
        try:
            tr, va = train_test_split(
                idx,
                test_size=CFG.finetune_val_ratio,
                random_state=CFG.seed,
                stratify=y if np.min(np.bincount(y)) >= 2 else None,
            )
        except Exception:
            split = int(len(idx) * (1.0 - CFG.finetune_val_ratio))
            tr, va = idx[:split], idx[split:]

        tune_train = [eicu_tune[i] for i in tr]
        tune_val = [eicu_tune[i] for i in va] if len(va) > 0 else [eicu_tune[i] for i in tr[: max(1, len(tr)//5)]]
        mixed_train = _merge_with_replay(tune_train, source_replay)

        model, ft_history = finetune_v6(model, mixed_train, tune_val, device=device)
        ft_eicu = eval_v6_model(model, eicu_data, device=device, batch_size=CFG.batch_size)
        result['eicu_finetuned'] = ft_eicu
        save_dir = OUT_DIR / 'finetuned_models'
        save_dir.mkdir(parents=True, exist_ok=True)
        ft_ckpt = save_dir / f'{model_tag}_finetuned.pt'
        torch.save(model.state_dict(), ft_ckpt)
        hist_path = save_dir / f'{model_tag}_finetune_history.json'
        hist_path.write_text(json.dumps(ft_history, indent=2, ensure_ascii=False), encoding='utf-8')
        result['finetune'] = {
            'mode': CFG.finetune_mode,
            'replay_ratio': CFG.replay_ratio,
            'target_train_n': len(tune_train),
            'target_val_n': len(tune_val),
            'mixed_train_n': len(mixed_train),
            'finetuned_checkpoint': str(ft_ckpt),
            'history_json': str(hist_path),
        }
    return result


def main():
    set_seed(CFG.seed)
    t0 = time.time()

    log.info('Loading datasets...')
    mimic_data = load_mimic_data(limit=CFG.mimic_limit)
    eicu_data, eicu_meta, label_names = load_eicu_data(limit=CFG.eicu_limit)

    mimic_index = json.loads((PROJECT_ROOT / 'data' / 'mimic' / 'index.json').read_text(encoding='utf-8'))

    y_sig = np.array([int(np.dot((np.asarray(s['label']) > 0.5).astype(np.int32), (1 << np.arange(6)))) for s in mimic_data])
    idx = np.arange(len(mimic_data))
    tr_idx, te_idx = train_test_split(idx, test_size=0.30, random_state=CFG.seed, stratify=y_sig if np.min(np.bincount(y_sig)) >= 2 else None)
    mimic_train = [mimic_data[i] for i in tr_idx]
    mimic_test = [mimic_data[i] for i in te_idx]

    eicu_mid = len(eicu_data) // 2
    eicu_tune = eicu_data[:eicu_mid]
    eicu_test = eicu_data[eicu_mid:]

    log.info(f'MIMIC train/test = {len(mimic_train):,}/{len(mimic_test):,}; eICU tune/test = {len(eicu_tune):,}/{len(eicu_test):,}')

    # A) 临床评分系统（公式代理，基于现有字段）
    cm_train_x, cm_train_y = make_clinical_scores(mimic_train, domain='mimic', mimic_index=mimic_index)
    cm_test_x, cm_test_y = make_clinical_scores(mimic_test, domain='mimic', mimic_index=mimic_index)
    cm_eicu_x, cm_eicu_y = make_clinical_scores(eicu_test, domain='eicu', eicu_meta=eicu_meta)

    base_ckpt_dir = OUT_DIR / 'baseline_checkpoints'
    base_ckpt_dir.mkdir(parents=True, exist_ok=True)

    clinical_results = {}
    # clinical_names = ['APACHE_proxy', 'SOFA_proxy', 'SAPS_proxy']
    # ... (already run)

    # B) 强力表格模型
    x_train, y_train = make_tabular_features(mimic_train)
    x_test, y_test = make_tabular_features(mimic_test)
    x_eicu, y_eicu = make_tabular_features(eicu_test)

    y_train_bin = (y_train > 0.5).astype(np.int32)
    y_test_bin = (y_test > 0.5).astype(np.int32)
    y_eicu_bin = (y_eicu > 0.5).astype(np.int32)

    tabular_results = {}
    # xgb_ctor = lambda: XGBClassifier(...)
    # ... (already run)

    # C) DANN / 深度基线
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deep_results = {}
    # for ckpt in ckpts: ... (already run or about to retrain)

    # D) SOTA Transformer Baselines (现场训练并在相同条件下评测)
    # 只包含尚未成功的基线
    sota_baselines = {
        'Med-BERT': MedBERT(n_features=4, n_labels=6) if MedBERT else None,
        # 'ClinicalBERT': ClinicalBERTForSequence(n_features=4, n_labels=6) if ClinicalBERTForSequence else None,
    }

    clinical_results = {} 
    # tabular_results = {} # Already defined above

    for name, model in sota_baselines.items():
        if model is None: continue
        log.info(f'Training {name} from scratch on MIMIC No-Aug...')
        model = train_baseline_deep_model(model, mimic_train, mimic_test, device, CFG.mimic_train_epochs, 5e-4)
        
        # MIMIC Test performance
        zero_mimic = eval_v6_model(model, mimic_test, device=device, batch_size=CFG.batch_size)
        # eICU Zeroshot
        zero_eicu = eval_v6_model(model, eicu_test, device=device, batch_size=CFG.batch_size)
        
        # 相同微调逻辑
        log.info(f'Fine-tuning {name} on eICU (Same Protocol)...')
        # Setup tuning data
        idx_t = np.arange(len(eicu_tune))
        tr_t, va_t = train_test_split(idx_t, test_size=CFG.finetune_val_ratio, random_state=CFG.seed)
        tune_train = [eicu_tune[i] for i in tr_t]
        tune_val = [eicu_tune[i] for i in va_t]
        mixed_train = _merge_with_replay(tune_train, mimic_train)
        
        tuned_model, ft_history = finetune_v6(model, mixed_train, tune_val, device=device)
        ft_eicu = eval_v6_model(tuned_model, eicu_test, device=device, batch_size=CFG.batch_size)
        
        save_dir = OUT_DIR / 'finetuned_models'
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tuned_model.state_dict(), save_dir / f'baseline_{name}_finetuned.pt')
        
        deep_results[f'Baseline_{name}'] = {
            'mimic_test': zero_mimic,
            'eicu_zeroshot': zero_eicu,
            'eicu_finetuned': ft_eicu,
            'finetune': {
                'mode': CFG.finetune_mode,
                'history_json': str(save_dir / f'baseline_{name}_finetune_history.json')
            }
        }

    # 自动策略：如果V6外部不如最强表格或低于目标AUC，则建议使用微调后版本
    best_tab_eicu = max(tabular_results.items(), key=lambda kv: kv[1]['eicu_test']['auroc'])
    best_tab_auc = best_tab_eicu[1]['eicu_test']['auroc']
    decisions = {}
    for k, v in deep_results.items():
        z = v.get('eicu_zeroshot', {}).get('auroc', 0.0)
        f = v.get('eicu_finetuned', {}).get('auroc', z)
        use_ft = False
        if z < best_tab_auc or z < CFG.target_auc_floor:
            use_ft = f > z
        decisions[k] = {
            'best_tabular_eicu_auroc': float(best_tab_auc),
            'zeroshot_eicu_auroc': float(z),
            'finetuned_eicu_auroc': float(f),
            'recommend_finetuned': bool(use_ft),
        }

    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': CFG.__dict__,
        'data': {
            'mimic_total': len(mimic_data),
            'mimic_train': len(mimic_train),
            'mimic_test': len(mimic_test),
            'eicu_total': len(eicu_data),
            'eicu_tune': len(eicu_tune),
            'eicu_test': len(eicu_test),
            'labels': label_names,
        },
        'clinical_scoring': clinical_results,
        'tabular_ml': tabular_results,
        'deep_domain_adaptation': deep_results,
        'selection_decision': decisions,
        'runtime_sec': float(time.time() - t0),
        'notes': {
            'clinical_scores': 'SAPS/SOFA/APACHE 使用当前数据可观测字段构建 proxy（非官方逐条手工公式重算）。'
        }
    }

    out_json = OUT_DIR / 'mimic_to_eicu_baseline_report.json'
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = ['model_group,model,mimic_auroc,mimic_auprc,eicu_auroc,eicu_auprc']
    for k, v in clinical_results.items():
        lines.append(f'clinical,{k},{v["mimic_test"]["auroc"]:.6f},{v["mimic_test"]["auprc"]:.6f},{v["eicu_test"]["auroc"]:.6f},{v["eicu_test"]["auprc"]:.6f}')
    for k, v in tabular_results.items():
        lines.append(f'tabular,{k},{v["mimic_test"]["auroc"]:.6f},{v["mimic_test"]["auprc"]:.6f},{v["eicu_test"]["auroc"]:.6f},{v["eicu_test"]["auprc"]:.6f}')
    for k, v in deep_results.items():
        mz = v.get('mimic_test', {})
        ez = v.get('eicu_zeroshot', {})
        lines.append(f'deep,{k}_zeroshot,{mz.get("auroc",0):.6f},{mz.get("auprc",0):.6f},{ez.get("auroc",0):.6f},{ez.get("auprc",0):.6f}')
        ef = v.get('eicu_finetuned', None)
        if ef is not None:
            lines.append(f'deep,{k}_finetuned,{mz.get("auroc",0):.6f},{mz.get("auprc",0):.6f},{ef.get("auroc",0):.6f},{ef.get("auprc",0):.6f}')
    out_csv = OUT_DIR / 'mimic_to_eicu_baseline_summary.csv'
    out_csv.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    log.info(f'Baseline benchmark done. Report: {out_json}')
    log.info(f'Summary CSV: {out_csv}')


if __name__ == '__main__':
    main()
