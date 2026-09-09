"""
Data utilities for Enhanced CDT model V2
支持更多数据格式和特征映射可视化
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from joblib import load
from tqdm import tqdm
from scipy.sparse import csr_matrix


def load_event_data(bundle_dir: Path, limit: Optional[int] = None, verbose: bool = True):
    """加载事件序列数据"""
    bundle_dir = Path(bundle_dir)
    shard_files = sorted(bundle_dir.glob("shard_*.joblib"))
    all_data = []
    iterator = tqdm(shard_files, desc="Loading shards") if verbose else shard_files
    for shard_file in iterator:
        shard_data = load(shard_file)
        all_data.extend(shard_data)
        if limit and len(all_data) >= limit:
            all_data = all_data[:limit]
            break
    return all_data


def _sample_events_with_temporal_coverage(all_events: List[List[float]], max_events: int) -> List[List[float]]:
    if len(all_events) <= max_events:
        return all_events

    # Preserve coverage across the full trajectory instead of truncating to the earliest window.
    bins = [[], [], []]
    for ev in all_events:
        t = float(ev[0])
        if t < 1.0 / 3.0:
            bins[0].append(ev)
        elif t < 2.0 / 3.0:
            bins[1].append(ev)
        else:
            bins[2].append(ev)

    quotas = [max_events // 3, max_events // 3, max_events - 2 * (max_events // 3)]
    selected: List[List[float]] = []
    leftovers: List[List[List[float]]] = []

    for bucket, quota in zip(bins, quotas):
        if len(bucket) <= quota:
            selected.extend(bucket)
            leftovers.append([])
            continue
        idx = np.linspace(0, len(bucket) - 1, num=quota)
        picked = np.round(idx).astype(np.int64)
        picked_set = set(int(i) for i in picked.tolist())
        selected.extend([bucket[i] for i in picked])
        leftovers.append([bucket[i] for i in range(len(bucket)) if i not in picked_set])

    if len(selected) < max_events:
        remaining = max_events - len(selected)
        pool = [ev for bucket in leftovers for ev in bucket]
        if pool:
            idx = np.linspace(0, len(pool) - 1, num=min(remaining, len(pool)))
            picked = np.round(idx).astype(np.int64)
            selected.extend([pool[i] for i in picked])

    selected.sort(key=lambda x: x[0])
    return selected[:max_events]


def collate_events(batch_data: List[Dict], max_events: int = 512) -> Dict[str, torch.Tensor]:
    """批量整理事件数据 - 增加异常值裁剪以保证模型稳定性"""
    batch_size = len(batch_data)
    events = torch.zeros(batch_size, max_events, 4)
    event_mask = torch.zeros(batch_size, max_events)
    labels = []
    notes = []
    statics = []
    
    # 合理的生理值范围 (裁剪999999等异常值)
    CLIP_MIN, CLIP_MAX = -100, 500
    
    for i, seq in enumerate(batch_data):
        timestamps = seq.get('timestamps', [0])
        T = len(timestamps)
        time_norm = np.arange(T) / max(T - 1, 1) if T > 1 else np.array([0.5])
        all_events = []
        
        for m_idx, modality in enumerate(['vitals', 'labs', 'meds']):
            mod_data = seq.get(modality, {})
            if 't' in mod_data and len(mod_data['t']) > 0:
                for t, f, v in zip(mod_data['t'], mod_data['f'], mod_data['v']):
                    if 0 <= t < T:
                        # 裁剪异常值 - 非常重要！
                        v_clipped = np.clip(v, CLIP_MIN, CLIP_MAX)
                        all_events.append([time_norm[t], m_idx, f, v_clipped])
        
        if len(all_events) > 0:
            all_events.sort(key=lambda x: x[0])
            kept_events = _sample_events_with_temporal_coverage(all_events, max_events)
            n_evs = min(len(kept_events), max_events)
            events[i, :n_evs, :] = torch.tensor(kept_events)
            event_mask[i, :n_evs] = 1.0
        
        labels.append(seq['label'])
        notes.append(seq.get('note', np.zeros(768)))
        # 静态特征也需要裁剪
        static = np.clip(seq.get('static', np.zeros(18)), CLIP_MIN, CLIP_MAX)
        statics.append(static)
    
    return {
        'events': events,
        'event_mask': event_mask,
        'labels': torch.tensor(np.array(labels)).float(),
        'notes': torch.tensor(np.array(notes)).float(),
        'static': torch.tensor(np.array(statics)).float()
    }


def extract_flat_features(data: List[Dict], n_features: int = 4500) -> Tuple[np.ndarray, np.ndarray]:
    """
    公平特征提取：为ML模型提供与DL模型等价的信息量
    
    注意：对统计特征进行裁剪以避免999999等异常值影响模型
    """
    features_list = []
    labels_list = []
    
    offsets = {'vitals': 0, 'labs': 1000, 'meds': 2000}
    
    # 合理的生理值范围（用于裁剪异常值）
    CLIP_MIN, CLIP_MAX = -100, 1000
    
    for seq in tqdm(data, desc="Extracting fair features"):
        # 1. 静态特征 (18)
        static = seq.get('static', np.zeros(18))
        
        # 裁剪静态特征中的异常值
        static = np.clip(static, CLIP_MIN, CLIP_MAX)
        
        # 2. 统计特征 (3模态 * 6统计量 = 18) - 对值进行裁剪以避免999999等异常值
        stats_feat = []
        for modality in ['vitals', 'labs', 'meds']:
            mod_data = seq.get(modality, {})
            if 't' in mod_data and len(mod_data['t']) > 0:
                v = np.clip(np.array(mod_data['v']), CLIP_MIN, CLIP_MAX)  # 裁剪异常值
                # 注意：len(v)也需要裁剪，因为事件数量可能非常大（45000+）
                count = min(len(v), 1000)  # 限制事件数量特征
                stats_feat.extend([np.mean(v), np.std(v), np.min(v), np.max(v), count, np.median(v)])
            else:
                stats_feat.extend([0]*6)
        
        # 3. Bag of Events (BoE)
        counts = np.zeros(n_features)
        for m_idx, modality in enumerate(['vitals', 'labs', 'meds']):
            mod_data = seq.get(modality, {})
            if 'f' in mod_data:
                for f in mod_data['f']:
                    global_f = (int(f) + offsets[modality]) % n_features
                    counts[global_f] += 1
        
        # 4. Note Embedding (768 -> 32)
        note = seq.get('note', np.zeros(768))
        note_reduced = note.reshape(32, 24).mean(axis=1)
        
        full_feat = np.concatenate([static, stats_feat, counts, note_reduced])
        features_list.append(full_feat)
        labels_list.append(seq['label'])
    
    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.float32)


# =============================================================================
# 特征映射 - 用于可视化GNN邻接矩阵
# =============================================================================

import json

def load_feature_names(bundle_dir: Path, n_buckets: int = 512) -> Dict[int, str]:
    """从index.json加载真实的特征名称映射，并映射到GNN的bucket"""
    index_path = Path(bundle_dir) / "index.json"
    if not index_path.exists():
        print(f"Warning: {index_path} not found, using fallback names")
        return {i: f"Bucket_{i}" for i in range(n_buckets)}
    
    try:
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        # 1. 建立 global_id -> name 的映射
        global_to_name = {}
        offsets = {'vital_features': 0, 'labs_features': 1000, 'meds_features': 2000}
        
        for key, offset in offsets.items():
            if key in index:
                for f_idx, name in enumerate(index[key]):
                    global_to_name[f_idx + offset] = name
        
        # 2. 建立 bucket_id -> list of names 的映射 (因为有哈希冲突)
        bucket_to_names = {}
        for global_f, name in global_to_name.items():
            bucket_id = global_f % n_buckets
            if bucket_id not in bucket_to_names:
                bucket_to_names[bucket_id] = []
            bucket_to_names[bucket_id].append(name)
        
        # 3. 每个bucket取最简短或最具有代表性的名称
        final_mapping = {}
        for b_id, names in bucket_to_names.items():
            # 如果有多个，取最短的那个作为代表，或者拼接
            if len(names) == 1:
                final_mapping[b_id] = names[0]
            else:
                # 优先保留非数字结尾的名称
                clean_names = [n for n in names if not any(char.isdigit() for char in n)]
                if clean_names:
                    final_mapping[b_id] = clean_names[0]
                else:
                    final_mapping[b_id] = names[0]
        
        # 补充缺失的bucket
        for i in range(n_buckets):
            if i not in final_mapping:
                final_mapping[i] = f"Bucket_{i}"
                
        return final_mapping
    except Exception as e:
        print(f"Error loading feature names: {e}")
        return {i: f"Bucket_{i}" for i in range(n_buckets)}

# 默认特征名称 (Fallback)
FEATURE_NAMES = {
    # Vitals (0-999)
    0: 'Heart Rate',
    1: 'Systolic BP',
    2: 'Diastolic BP',
    3: 'Temperature',
    4: 'SpO2',
    5: 'Respiratory Rate',
    6: 'MAP',
    # Labs (1000-1999)
    1000: 'Glucose',
    1001: 'Creatinine',
    1002: 'BUN',
    1003: 'Potassium',
    1004: 'Sodium',
    1005: 'WBC',
    1006: 'Hemoglobin',
    1007: 'Platelets',
    1008: 'HbA1c',
    1009: 'eGFR',
    1010: 'ALT',
    1011: 'AST',
    1012: 'Bilirubin',
    1013: 'Albumin',
    1014: 'Lactate',
    1015: 'Troponin',
    1016: 'BNP',
    # Meds (2000-2999)
    2000: 'Insulin',
    2001: 'Metformin',
    2002: 'Lisinopril',
    2003: 'Amlodipine',
    2004: 'Furosemide',
    2005: 'Aspirin',
    2006: 'Atorvastatin',
    2007: 'Metoprolol',
    2008: 'Heparin',
    2009: 'Norepinephrine',
    2010: 'Vancomycin',
    2011: 'Piperacillin',
}


def get_feature_name(bucket_id: int, n_buckets: int = 512, custom_mapping: Optional[Dict[int, str]] = None) -> str:
    """获取bucket对应的代表性特征名"""
    mapping = custom_mapping if custom_mapping is not None else FEATURE_NAMES
    # 由于使用了哈希，需要反向映射
    # 优先寻找匹配的已知特征
    for fid, name in mapping.items():
        if fid % n_buckets == bucket_id:
            return name
    return f"Bucket_{bucket_id}"


def build_feature_id_mapping(data: List[Dict], n_buckets: int = 512) -> Dict[int, List[int]]:
    """
    构建bucket到原始特征ID的映射
    用于理解GNN学习的邻接矩阵
    """
    bucket_to_features = {i: [] for i in range(n_buckets)}
    offsets = {'vitals': 0, 'labs': 1000, 'meds': 2000}
    
    for seq in data[:1000]:  # 采样1000个样本
        for m_idx, modality in enumerate(['vitals', 'labs', 'meds']):
            mod_data = seq.get(modality, {})
            if 'f' in mod_data:
                for f in mod_data['f']:
                    global_f = int(f) + offsets[modality]
                    bucket = global_f % n_buckets
                    if global_f not in bucket_to_features[bucket]:
                        bucket_to_features[bucket].append(global_f)
    
    return bucket_to_features
