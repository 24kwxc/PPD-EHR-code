"""Create multimodal event bundles for model development.

The four former one-off scripts are exposed as explicit subcommands: ``extract``
for raw-table feature extraction, ``stream`` for sharded sequence construction,
``convert`` for event-stream conversion, and ``freeze-v3`` for the frozen V3 bundle.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace


# ==============================================================================
# Feature extraction
# ==============================================================================

def _load_extract():
    """Command-line utility to extract and persist multimodal MIMIC-IV features."""

    import argparse
    import json
    import logging
    import shutil
    import zipfile
    from pathlib import Path
    from typing import Dict

    import pandas as pd
    from joblib import dump

    import multimodal_features as config
    from multimodal_features import (
        extract_labs,
        extract_medications,
        extract_vitals,
        load_complication_labels,
        load_discharge_notes,
        load_icu_stays,
        load_static_features,
    )
    from multimodal_features import build_sequence_bundle
    from multimodal_features import build_clinicalbert_embeddings

    logger = logging.getLogger(__name__)


    def _setup_logging(verbose: bool) -> None:
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s - %(message)s")


    def extract_discharge_notes(archive: Path, output_dir: Path) -> Path:
        """Extract the single MIMIC-IV-Note member used by this preprocessor."""
        member = "note/discharge.csv.gz"
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "discharge.csv.gz"
        with zipfile.ZipFile(archive) as source:
            if member not in source.namelist():
                raise FileNotFoundError(f"Archive does not contain {member}: {archive}")
            with source.open(member) as raw, target.open("wb") as destination:
                shutil.copyfileobj(raw, destination)
        return target


    def _save_table(df: pd.DataFrame, path: Path, fmt: str) -> None:
        if df.empty:
            logger.info("%s is empty; skipping save.", path.name)
            return
        if fmt == "parquet":
            try:
                df.to_parquet(path, index=False)
                return
            except Exception as err:  # pragma: no cover - pyarrow may be missing
                logger.warning("Failed to save %s as parquet (%s); falling back to CSV.", path.name, err)
                fmt = "csv"
        if fmt == "csv":
            df.to_csv(path, index=False)
            return
        raise ValueError(f"Unsupported output format: {fmt}")


    def _summarise_counts(labels: pd.DataFrame) -> Dict[str, int]:
        keep_cols = [col for col in labels.columns if col not in {"subject_id", "hadm_id", "stay_id"}]
        if not keep_cols:
            return {}
        return labels[keep_cols].sum().astype(int).to_dict()


    def run_preprocessing(args: argparse.Namespace) -> None:
        config.ensure_output_dirs()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        diabetes_only = not args.all_stays
        logger.info("Loading ICU stays (diabetes_only=%s)...", diabetes_only)
        stays = load_icu_stays(max_records=args.max_stays, diabetes_only=diabetes_only)
        if stays.empty:
            logger.error("No ICU stays matched the selection criteria.")
            return

        labels = load_complication_labels(stays, config.COMPLICATION_CODE_GROUPS)
        static_df = load_static_features(stays)

        notes_df = load_discharge_notes(stays) if args.include_notes else pd.DataFrame(columns=["stay_id", "text"])
        note_embeddings = None
        note_embedding_meta = None
        if args.include_notes:
            note_embedding_result = build_clinicalbert_embeddings(
                notes_df,
                stay_id_column="stay_id",
                model_dir=args.clinicalbert_dir,
                batch_size=args.note_batch_size,
                max_length=args.note_max_length,
                pooling=args.note_pooling,
                standardize=args.note_standardize,
            )
            note_embeddings = note_embedding_result.matrix
            note_embedding_meta = note_embedding_result

        vitals = extract_vitals(stays, freq=args.vitals_frequency)
        labs = extract_labs(stays, freq=args.labs_frequency)
        meds = extract_medications(stays, freq=args.med_frequency)

        if args.save_intermediate:
            fmt = args.format
            tables: Dict[str, pd.DataFrame] = {
                "stays": stays,
                "labels": labels,
                "static": static_df,
                "notes": notes_df,
                "vitals": vitals,
                "labs": labs,
                "medications": meds,
            }
            if note_embeddings is not None:
                tables["note_embeddings"] = note_embeddings
            for name, df in tables.items():
                path = output_dir / f"{name}.{fmt}"
                logger.info("Saving %s (%d rows) to %s", name, len(df), path)
                _save_table(df, path, fmt)
            if note_embedding_meta is not None and note_embedding_meta.standardization_applied:
                stats_path = output_dir / "note_embedding_standardization_stats.json"
                stats_payload = {
                    "feature_names": list(note_embedding_meta.feature_names),
                    "mean": note_embedding_meta.mean.tolist() if note_embedding_meta.mean is not None else [],
                    "std": note_embedding_meta.std.tolist() if note_embedding_meta.std is not None else [],
                }
                stats_path.write_text(json.dumps(stats_payload))
                logger.info("Saved note embedding standardization stats to %s", stats_path)

        summary: Dict[str, object] = {
            "diabetes_only": diabetes_only,
            "max_stays": args.max_stays,
            "stays": len(stays),
            "vitals_rows": len(vitals),
            "labs_rows": len(labs),
            "med_rows": len(meds),
            "label_counts": _summarise_counts(labels),
            "vitals_frequency": args.vitals_frequency,
            "labs_frequency": args.labs_frequency,
            "med_frequency": args.med_frequency,
        }

        if note_embedding_meta is not None:
            summary["note_embedding_dim"] = len(note_embedding_meta.feature_names)
            summary["note_embedding_model"] = note_embedding_meta.model_name
            summary["note_embedding_pooling"] = note_embedding_meta.pooling
            summary["note_standardization_applied"] = note_embedding_meta.standardization_applied

        bundle_path: Path | None = None
        if args.build_sequences:
            logger.info("Building sequence bundle (min_seq_len=%d)...", args.min_seq_len)
            note_df = note_embeddings if note_embeddings is not None else None
            bundle = build_sequence_bundle(
                vitals,
                labs,
                meds,
                labels,
                static_df,
                note_df,
                min_sequence_length=args.min_seq_len,
            )
            summary["sequence_count"] = len(bundle.sequences)
            summary["vital_feature_count"] = len(bundle.vitals_meta.feature_names)
            summary["lab_feature_count"] = len(bundle.labs_meta.feature_names)
            summary["med_feature_count"] = len(bundle.meds_meta.feature_names)
            summary["label_names"] = list(bundle.label_names)
            if args.save_bundle:
                bundle_path = Path(args.save_bundle)
                bundle_path.parent.mkdir(parents=True, exist_ok=True)
                dump(bundle, bundle_path)
                logger.info("Saved sequence bundle to %s", bundle_path)
        else:
            summary["sequence_count"] = 0

        summary_path = output_dir / "preprocess_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("Preprocessing summary written to %s", summary_path)
        if bundle_path is not None:
            logger.info("Sequence bundle path: %s", bundle_path)


    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Extract multimodal features from MIMIC-IV")
        parser.add_argument("--max-stays", type=int, default=config.MAX_STAYS)
        parser.add_argument("--min-seq-len", type=int, default=config.MIN_SEQUENCE_LENGTH)
        parser.add_argument("--vitals-frequency", type=str, default=None)
        parser.add_argument("--labs-frequency", type=str, default=None)
        parser.add_argument("--med-frequency", type=str, default=config.TIME_FREQUENCY)
        parser.add_argument("--all-stays", action="store_true", help="Use all ICU stays without diabetes filtering")
        parser.add_argument("--include-notes", action="store_true", help="Compute discharge note embeddings")
        parser.add_argument("--note-archive", type=Path, default=None, help="Optional MIMIC-IV-Note ZIP to extract before preprocessing")
        parser.add_argument("--build-sequences", action="store_true", help="Assemble SequenceBundle objects")
        parser.add_argument("--save-bundle", type=str, default=None, help="Optional path to persist the sequence bundle via joblib")
        parser.add_argument("--save-intermediate", action="store_true", help="Persist raw tables to disk")
        parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
        parser.add_argument("--output-dir", type=str, default=str(config.OUTPUT_DIR / "preprocessed"))
        parser.add_argument("--clinicalbert-dir", type=str, default="data/models/ClinicalBERT", help="Local path to ClinicalBERT model directory")
        parser.add_argument("--note-batch-size", type=int, default=8, help="Batch size for ClinicalBERT embedding inference")
        parser.add_argument("--note-max-length", type=int, default=512, help="Maximum token length for ClinicalBERT encoding")
        parser.add_argument("--note-pooling", choices=["cls", "mean"], default="cls", help="Pooling strategy for ClinicalBERT embeddings")
        parser.add_argument("--note-standardize", action="store_true", help="Apply per-feature standardization to ClinicalBERT embeddings")
        parser.add_argument("--verbose", action="store_true")
        return parser.parse_args()

    return SimpleNamespace(**locals())

_extract = _load_extract()


# ==============================================================================
# Streaming sequence-bundle construction
# ==============================================================================

def _load_stream():
    """Streamed construction of SequenceBundle to avoid holding all long tables in memory."""

    import argparse
    import gc
    import json
    import logging
    from pathlib import Path
    from typing import Dict, Iterable, List, Optional, Sequence, Tuple

    import numpy as np
    import pandas as pd
    from joblib import dump

    from multimodal_features import SequenceExample

    logger = logging.getLogger(__name__)


    def _setup_logging(verbose: bool) -> None:
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s - %(message)s")


    def _load_feature_set(path: Path, chunk_size: int) -> Tuple[str, ...]:
        features: set[str] = set()
        for chunk in pd.read_csv(path, usecols=["feature"], chunksize=chunk_size):
            features.update(chunk["feature"].dropna().unique().tolist())
        return tuple(sorted(features))


    def _load_long_table_for_block(
        path: Path,
        stay_set: set[int],
        chunk_size: int,
        expected_columns: Sequence[str],
    ) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            filtered = chunk[chunk["stay_id"].astype(int).isin(stay_set)]
            if filtered.empty:
                continue
            filtered = filtered.copy()
            filtered["stay_id"] = filtered["stay_id"].astype(int)
            if "subject_id" in filtered.columns:
                filtered["subject_id"] = filtered["subject_id"].astype(int)
            if "hadm_id" in filtered.columns:
                filtered["hadm_id"] = filtered["hadm_id"].astype(int)
            if "time_bin" in filtered.columns:
                filtered["time_bin"] = pd.to_timedelta(filtered["time_bin"])
            frames.append(filtered)
        if not frames:
            return pd.DataFrame(columns=list(expected_columns))
        concatenated = pd.concat(frames, ignore_index=True)
        return concatenated[list(expected_columns)]


    def _prepare_long_frame(frame: pd.DataFrame | None, features: Sequence[str]) -> Optional[pd.DataFrame]:
        if frame is None or frame.empty or not features:
            return None
        pivot = (
            frame.pivot_table(
                index="time_bin",
                columns="feature",
                values="value",
                aggfunc="mean",
            )
            .reindex(columns=list(features))
            .astype(float)
        )
        pivot.columns.name = None
        return pivot


    def _ensure_feature_frame(
        frame: Optional[pd.DataFrame],
        columns: Sequence[str],
        timeline: Sequence[pd.Timedelta],
    ) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(index=pd.Index(timeline, name="time_bin"), columns=list(columns), dtype=float)
        aligned = frame.reindex(timeline)
        return aligned.astype(float)


    def _frame_to_tensor(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        values = frame.to_numpy(dtype=np.float32)
        mask = (~np.isnan(values)).astype(np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return values, mask


    def _assemble_sequences_for_block(
        labels_block: pd.DataFrame,
        vitals_block: pd.DataFrame,
        labs_block: pd.DataFrame,
        meds_block: pd.DataFrame,
        static_lookup: pd.DataFrame,
        note_lookup: pd.DataFrame,
        vitals_features: Tuple[str, ...],
        labs_features: Tuple[str, ...],
        meds_features: Tuple[str, ...],
        static_features: Tuple[str, ...],
        note_features: Tuple[str, ...],
        min_seq_len: int,
    ) -> List[SequenceExample]:
        sequences: List[SequenceExample] = []

        vitals_by_stay = {int(k): df.sort_values("time_bin").reset_index(drop=True) for k, df in vitals_block.groupby("stay_id", sort=True)} if not vitals_block.empty else {}
        labs_by_stay = {int(k): df.sort_values("time_bin").reset_index(drop=True) for k, df in labs_block.groupby("stay_id", sort=True)} if not labs_block.empty else {}
        meds_by_stay = {int(k): df.sort_values("time_bin").reset_index(drop=True) for k, df in meds_block.groupby("stay_id", sort=True)} if not meds_block.empty else {}

        for _, row in labels_block.iterrows():
            stay_id = int(row["stay_id"])
            vitals_frame = vitals_by_stay.get(stay_id)
            labs_frame = labs_by_stay.get(stay_id)
            meds_frame = meds_by_stay.get(stay_id)

            if vitals_frame is None and labs_frame is None and meds_frame is None:
                continue

            timeline_set: set[pd.Timedelta] = set()
            for frame in (vitals_frame, labs_frame, meds_frame):
                if frame is not None:
                    timeline_set.update(frame["time_bin"].tolist())
            if not timeline_set:
                continue
            timeline = sorted(timeline_set)
            if len(timeline) < min_seq_len:
                continue

            vitals_tensor = np.zeros((len(timeline), len(vitals_features)), dtype=np.float32)
            vitals_mask = np.zeros_like(vitals_tensor)
            if vitals_features:
                vitals_wide = _prepare_long_frame(vitals_frame, vitals_features)
                vitals_prepared = _ensure_feature_frame(vitals_wide, vitals_features, timeline)
                vitals_tensor, vitals_mask = _frame_to_tensor(vitals_prepared)

            labs_tensor = np.zeros((len(timeline), len(labs_features)), dtype=np.float32)
            labs_mask = np.zeros_like(labs_tensor)
            if labs_features:
                labs_wide = _prepare_long_frame(labs_frame, labs_features)
                labs_prepared = _ensure_feature_frame(labs_wide, labs_features, timeline)
                labs_tensor, labs_mask = _frame_to_tensor(labs_prepared)

            meds_tensor = np.zeros((len(timeline), len(meds_features)), dtype=np.float32)
            meds_mask = np.zeros_like(meds_tensor)
            if meds_features:
                meds_wide = _prepare_long_frame(meds_frame, meds_features)
                meds_prepared = _ensure_feature_frame(meds_wide, meds_features, timeline)
                meds_tensor, meds_mask = _frame_to_tensor(meds_prepared)

            timestamps = np.array(timeline, dtype="timedelta64[s]")
            label_names = [col for col in labels_block.columns if col not in {"subject_id", "hadm_id", "stay_id"}]
            label_values = row[label_names].to_numpy(dtype=np.float32)

            if static_features and stay_id in static_lookup.index:
                static_vec = static_lookup.loc[stay_id].to_numpy(dtype=np.float32)
            else:
                static_vec = np.zeros(len(static_features), dtype=np.float32)

            if note_features and stay_id in note_lookup.index:
                note_vec = note_lookup.loc[stay_id].to_numpy(dtype=np.float32)
            else:
                note_vec = np.zeros(len(note_features), dtype=np.float32)

            sequences.append(
                SequenceExample(
                    subject_id=int(row["subject_id"]),
                    hadm_id=int(row["hadm_id"]),
                    stay_id=stay_id,
                    timestamps=timestamps,
                    vitals=vitals_tensor,
                    vitals_mask=vitals_mask,
                    labs=labs_tensor,
                    labs_mask=labs_mask,
                    meds=meds_tensor,
                    meds_mask=meds_mask,
                    static=static_vec,
                    note=note_vec,
                    label=label_values,
                )
            )

        return sequences


    def run(args: argparse.Namespace) -> None:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory {input_dir} does not exist")

        labels_path = input_dir / "labels.csv"
        static_path = input_dir / "static.csv"
        note_embeddings_path = input_dir / "note_embeddings.csv"
        vitals_path = input_dir / "vitals.csv"
        labs_path = input_dir / "labs.csv"
        meds_path = input_dir / "medications.csv"

        logger.info("Loading label, static, and note embeddings tables")
        labels = pd.read_csv(labels_path)
        static_df = pd.read_csv(static_path)
        note_embeddings = pd.read_csv(note_embeddings_path)

        label_names = [col for col in labels.columns if col not in {"subject_id", "hadm_id", "stay_id"}]
        static_features = tuple(col for col in static_df.columns if col not in {"subject_id", "hadm_id", "stay_id"})
        note_features = tuple(col for col in note_embeddings.columns if col != "stay_id")

        static_lookup = (
            static_df.set_index("stay_id")[list(static_features)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            if static_features
            else pd.DataFrame()
        )
        note_lookup = (
            note_embeddings.set_index("stay_id")[list(note_features)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            if note_features
            else pd.DataFrame()
        )
        if not static_lookup.empty:
            static_lookup.index = static_lookup.index.astype(int)
        if not note_lookup.empty:
            note_lookup.index = note_lookup.index.astype(int)

        logger.info("Collecting feature sets for vitals, labs, and medications")
        vitals_features = _load_feature_set(vitals_path, args.chunk_size)
        labs_features = _load_feature_set(labs_path, args.chunk_size)
        meds_features = _load_feature_set(meds_path, args.chunk_size)

        logger.info(
            "Feature counts -> vitals: %d, labs: %d, meds: %d, static: %d, note: %d",
            len(vitals_features),
            len(labs_features),
            len(meds_features),
            len(static_features),
            len(note_features),
        )

        stay_ids = labels["stay_id"].astype(int).tolist()
        total_blocks = (len(stay_ids) + args.block_size - 1) // args.block_size
        total_sequences = 0
        shards: List[Dict[str, object]] = []

        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        for block_idx in range(total_blocks):
            start = block_idx * args.block_size
            end = min(start + args.block_size, len(stay_ids))
            block_stay_ids = stay_ids[start:end]
            stay_set = set(block_stay_ids)
            logger.info(
                "Processing block %d/%d (stay indices %d-%d)",
                block_idx + 1,
                total_blocks,
                start,
                end - 1,
            )
            labels_block = labels[labels["stay_id"].isin(stay_set)].reset_index(drop=True)

            vitals_block = _load_long_table_for_block(
                vitals_path,
                stay_set,
                args.chunk_size,
                ["subject_id", "hadm_id", "stay_id", "time_bin", "feature", "value"],
            )
            labs_block = _load_long_table_for_block(
                labs_path,
                stay_set,
                args.chunk_size,
                ["subject_id", "hadm_id", "stay_id", "time_bin", "feature", "value"],
            )
            meds_block = _load_long_table_for_block(
                meds_path,
                stay_set,
                args.chunk_size,
                ["subject_id", "hadm_id", "stay_id", "time_bin", "feature", "value"],
            )

            block_sequences = _assemble_sequences_for_block(
                labels_block,
                vitals_block,
                labs_block,
                meds_block,
                static_lookup,
                note_lookup,
                vitals_features,
                labs_features,
                meds_features,
                static_features,
                note_features,
                args.min_seq_len,
            )
            shard_name = f"shard_{block_idx:04d}.joblib"
            shard_path = output_path / shard_name
            dump(block_sequences, shard_path)

            total_sequences += len(block_sequences)
            shards.append(
                {
                    "shard": shard_name,
                    "sequence_count": len(block_sequences),
                    "stay_indices": [start, end - 1],
                    "stay_ids": [int(sid) for sid in block_stay_ids],
                }
            )

            logger.info(
                "Block %d produced %d sequences (saved to %s)",
                block_idx + 1,
                len(block_sequences),
                shard_name,
            )

            del vitals_block, labs_block, meds_block, block_sequences
            gc.collect()

        logger.info("Finished writing %d shards (%d sequences)", len(shards), total_sequences)

        index = {
            "input_dir": str(input_dir),
            "output_dir": str(output_path),
            "total_sequences": total_sequences,
            "num_shards": len(shards),
            "min_sequence_length": args.min_seq_len,
            "block_size": args.block_size,
            "chunk_size": args.chunk_size,
            "label_names": label_names,
            "vital_features": list(vitals_features),
            "lab_features": list(labs_features),
            "med_features": list(meds_features),
            "static_features": list(static_features),
            "note_features": list(note_features),
            "shards": shards,
        }
        index_path = output_path / "index.json"
        index_path.write_text(json.dumps(index, indent=2))
        logger.info("Shard index written to %s", index_path)



    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Stream construction of sharded SequenceBundle from preprocessed CSV tables")
        parser.add_argument("--input-dir", required=True, help="Directory containing preprocessed CSV tables")
        parser.add_argument("--output", required=True, help="Directory to write sharded SequenceBundle")
        parser.add_argument("--min-seq-len", type=int, default=6)
        parser.add_argument("--block-size", type=int, default=512, help="Number of stays to process per block")
        parser.add_argument("--chunk-size", type=int, default=250_000, help="Row chunksize to use when scanning long tables")
        parser.add_argument("--verbose", action="store_true")
        return parser.parse_args()

    return SimpleNamespace(**locals())

_stream = _load_stream()


# ==============================================================================
# Shards-to-event-stream conversion
# ==============================================================================

def _load_convert():
    """Convert existing SequenceExample shards into compact event-stream shards.

    Writes shards into a target directory with the same shard filenames but with
    entries stored as dicts containing sparse event arrays for vitals/labs/meds.

    Usage: python -m src.convert_shards_to_event_stream --src-dir <src> --dst-dir <dst> --shards N
    """

    import argparse
    import json
    import logging
    from pathlib import Path
    from typing import List

    import numpy as np
    from joblib import load, dump

    from multimodal_features import SequenceExample

    logger = logging.getLogger(__name__)


    def _example_to_event_dict(ex: SequenceExample) -> dict:
        # timestamps -> seconds from admission as int
        ts_seconds = ex.timestamps.astype("timedelta64[s]").astype(np.int64)

        def _sparsify(array: np.ndarray, mask: np.ndarray):
            if array.size == 0:
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)
            times_idx, feat_idx = np.nonzero(mask.astype(bool))
            values = array[times_idx, feat_idx].astype(np.float32)
            return times_idx.astype(np.int32), feat_idx.astype(np.int32), values

        vit_times, vit_feats, vit_vals = _sparsify(ex.vitals, ex.vitals_mask)
        lab_times, lab_feats, lab_vals = _sparsify(ex.labs, ex.labs_mask)
        med_times, med_feats, med_vals = _sparsify(ex.meds, ex.meds_mask)

        return {
            "format": "event_shard_v1",
            "subject_id": int(ex.subject_id),
            "hadm_id": int(ex.hadm_id),
            "stay_id": int(ex.stay_id),
            "timestamps": ts_seconds.astype(np.int64),
            "vitals": {"t": vit_times, "f": vit_feats, "v": vit_vals},
            "labs": {"t": lab_times, "f": lab_feats, "v": lab_vals},
            "meds": {"t": med_times, "f": med_feats, "v": med_vals},
            "static": ex.static.astype(np.float32),
            "note": ex.note.astype(np.float32),
            "label": ex.label.astype(np.float32),
        }


    def convert_shard(src_path: Path, dst_path: Path) -> int:
        obj = load(src_path)
        if not isinstance(obj, list):
            raise ValueError(f"Expected list of SequenceExample in {src_path}")
        out = []
        for ex in obj:
            out.append(_example_to_event_dict(ex))
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dump(out, dst_path, compress=("lz4", 3))
        return len(out)


    def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument("--src-dir", required=True)
        p.add_argument("--dst-dir", required=True)
        p.add_argument("--shards", type=int, default=0, help="Number of shards to convert (0 -> all)")
        return p.parse_args()


    def main():
        args = parse_args()
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
        src = Path(args.src_dir)
        dst = Path(args.dst_dir)
        index_path = src / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        index = json.loads(index_path.read_text())
        shards = index.get("shards", [])
        if args.shards > 0:
            shards = shards[: args.shards]

        for entry in shards:
            shard_name = entry["shard"]
            src_path = src / shard_name
            dst_path = dst / shard_name
            logger.info("Converting %s -> %s", src_path, dst_path)
            n = convert_shard(src_path, dst_path)
            logger.info("Wrote %d examples", n)

    return SimpleNamespace(**locals())

_convert = _load_convert()


# ==============================================================================
# Frozen V3 dataset construction
# ==============================================================================

def _load_v3():
    import argparse
    import json
    import joblib
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import re
    from tqdm import tqdm
    import os

    # Strict biological limits for known core features
    CORE_LIMITS = {
        'heart_rate': {'min': 20, 'max': 300},
        'sbp': {'min': 40, 'max': 300},
        'dbp': {'min': 20, 'max': 200},
        'mbp': {'min': 30, 'max': 250},
        'resp_rate': {'min': 4, 'max': 60},
        'temperature_c': {'min': 30, 'max': 45},
        'spo2': {'min': 30, 'max': 100},
        'glucose': {'min': 20, 'max': 1000},
        'creatinine': {'min': 0.1, 'max': 25},
        'bun': {'min': 1, 'max': 250},
        'wbc': {'min': 0.1, 'max': 200},
        'hemoglobin': {'min': 2, 'max': 25},
        'platelets': {'min': 5, 'max': 1500},
        'sodium': {'min': 100, 'max': 180},
        'potassium': {'min': 1.0, 'max': 9.0},
        'lactate': {'min': 0.1, 'max': 30},
        'ph': {'min': 6.8, 'max': 7.8},
        'pco2': {'min': 10, 'max': 150},
        'po2': {'min': 20, 'max': 500},
        'calcium': {'min': 4, 'max': 20},
        'gcs_motor': {'min': 1, 'max': 6},
        'gcs_verbal': {'min': 1, 'max': 5},
        'gcs_eyes': {'min': 1, 'max': 4},
        'urine_output': {'min': 0, 'max': 5000}
    }

    def clean_feature_name(name):
        clean_name = str(name).lower().replace(' ', '_').replace('-', '_').replace('/', '_')
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')
        return clean_name

    def semantic_map(raw_name):
        ln = raw_name.lower()
        if 'heart' in ln and 'rate' in ln and 'alarm' not in ln: return 'heart_rate'
        if 'systolic' in ln and 'alarm' not in ln and 'pulmonary' not in ln: return 'sbp'
        if 'diastolic' in ln and 'alarm' not in ln and 'pulmonary' not in ln: return 'dbp'
        if 'mean' in ln and 'blood_pressure' in ln and 'alarm' not in ln: return 'mbp'
        if 'temperature' in ln and 'alarm' not in ln and 'water' not in ln: return 'temperature_c' # assume all C eventually
        if ('spo2' in ln or 'o2 sat' in ln) and 'alarm' not in ln: return 'spo2'
        if 'respiratory_rate' in ln and 'alarm' not in ln: return 'resp_rate'
        if 'gcs' in ln and 'motor' in ln: return 'gcs_motor'
        if 'gcs' in ln and 'verbal' in ln: return 'gcs_verbal'
        if 'gcs' in ln and 'eye' in ln: return 'gcs_eyes'
        if 'urine' in ln and 'output' in ln: return 'urine_output'
        if 'glucose' in ln: return 'glucose'
        if 'creatinine' in ln and 'urine' not in ln: return 'creatinine'
        if 'wbc' in ln: return 'wbc'
        if 'hemoglobin' in ln: return 'hemoglobin'
        if 'platelet' in ln: return 'platelets'
    
        return clean_feature_name(raw_name)

    def build_mapping(source_dir: Path, output_dir: Path):
        with (source_dir / "index.json").open("r", encoding="utf-8") as f:
            idx_data = json.load(f)
        
        old_vital_features = idx_data['vital_features']
        old_lab_features = idx_data['lab_features']
    
        new_vitals = []
        old_to_new_v_idx = {}
        for i, f in enumerate(old_vital_features):
            nm = semantic_map(f)
            if nm not in new_vitals:
                new_vitals.append(nm)
            old_to_new_v_idx[i] = new_vitals.index(nm)
        
        new_labs = []
        old_to_new_l_idx = {}
        for i, f in enumerate(old_lab_features):
            nm = semantic_map(f)
            if nm not in new_labs:
                new_labs.append(nm)
            old_to_new_l_idx[i] = new_labs.index(nm)
        
        idx_data['vital_features'] = new_vitals
        idx_data['lab_features'] = new_labs
    
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "index.json").open("w", encoding="utf-8") as f:
            json.dump(idx_data, f, indent=2)
        
        print(f"Reduced {len(old_vital_features)} vitals -> {len(new_vitals)} unique features.")
        print(f"Reduced {len(old_lab_features)} labs -> {len(new_labs)} unique features.")
    
        return old_to_new_v_idx, old_to_new_l_idx, new_vitals, new_labs

    def compute_percentiles(source_dir: Path, old_to_new_v_idx, old_to_new_l_idx, num_shards_to_scan=5):
        print(f"Scanning first {num_shards_to_scan} shards to compute 1% and 99% quantiles...")
        v_values = {}
        l_values = {}
    
        shards = sorted(source_dir.glob("shard_*.joblib"))[:num_shards_to_scan]
        for sp in shards:
            data = joblib.load(sp)
            for seq in data:
                if 'vitals' in seq and 'f' in seq['vitals']:
                    fs = seq['vitals']['f']
                    vs = seq['vitals']['v']
                    for f, v in zip(fs, vs):
                        nf = old_to_new_v_idx[f]
                        if nf not in v_values: v_values[nf] = []
                        v_values[nf].append(v)
                if 'labs' in seq and 'f' in seq['labs']:
                    fs = seq['labs']['f']
                    vs = seq['labs']['v']
                    for f, v in zip(fs, vs):
                        nf = old_to_new_l_idx[f]
                        if nf not in l_values: l_values[nf] = []
                        l_values[nf].append(v)
                    
        v_bounds = {}
        for nf, vals in v_values.items():
            if len(vals) > 0:
                v_bounds[nf] = (np.percentile(vals, 1), np.percentile(vals, 99))
            
        l_bounds = {}
        for nf, vals in l_values.items():
            if len(vals) > 0:
                l_bounds[nf] = (np.percentile(vals, 1), np.percentile(vals, 99))
            
        return v_bounds, l_bounds

    def convert_shards(source_dir: Path, output_dir: Path, old_to_new_v_idx, old_to_new_l_idx, new_vitals, new_labs, v_bounds, l_bounds, max_shards=0):
        shards = sorted(source_dir.glob("shard_*.joblib"))
        if max_shards:
            shards = shards[:max_shards]
        print(f"Converting {len(shards)} shards...")
    
        total_raw_points = 0
        total_clean_points = 0
    
        for i, sp in enumerate(shards):
            data = joblib.load(sp)
            new_data = []
            for seq in data:
                new_seq = seq.copy() # Keeps timestamps, static, meds, note, label
            
                # Vitals
                if 'vitals' in seq and 'f' in seq['vitals']:
                    fs = seq['vitals']['f']
                    ts = seq['vitals']['t']
                    vs = seq['vitals']['v']
                
                    new_f, new_t, new_v = [], [], []
                    total_raw_points += len(fs)
                
                    for f, t, v in zip(fs, ts, vs):
                        nf = old_to_new_v_idx[f]
                        nf_name = new_vitals[nf]
                    
                        # Convert Temp F to C logic if needed here, but let's just bounds check for now
                        if nf_name == 'temperature_c' and v > 50:
                            v = (v - 32) * 5/9
                        
                        # Bounds Check
                        is_valid = True
                        if nf_name in CORE_LIMITS:
                            if not (CORE_LIMITS[nf_name]['min'] <= v <= CORE_LIMITS[nf_name]['max']):
                                is_valid = False
                        elif nf in v_bounds:
                            if not (v_bounds[nf][0] <= v <= v_bounds[nf][1]):
                                is_valid = False
                            
                        if is_valid:
                            new_f.append(nf)
                            new_t.append(t)
                            new_v.append(v)
                            total_clean_points += 1
                        
                    new_seq['vitals'] = {'t': np.array(new_t), 'f': np.array(new_f), 'v': np.array(new_v)}
                
                # Labs
                if 'labs' in seq and 'f' in seq['labs']:
                    fs = seq['labs']['f']
                    ts = seq['labs']['t']
                    vs = seq['labs']['v']
                
                    new_f, new_t, new_v = [], [], []
                    total_raw_points += len(fs)
                
                    for f, t, v in zip(fs, ts, vs):
                        nf = old_to_new_l_idx[f]
                        nf_name = new_labs[nf]
                    
                        is_valid = True
                        if nf_name in CORE_LIMITS:
                            if not (CORE_LIMITS[nf_name]['min'] <= v <= CORE_LIMITS[nf_name]['max']):
                                is_valid = False
                        elif nf in l_bounds:
                            if not (l_bounds[nf][0] <= v <= l_bounds[nf][1]):
                                is_valid = False
                            
                        if is_valid:
                            new_f.append(nf)
                            new_t.append(t)
                            new_v.append(v)
                            total_clean_points += 1
                        
                    new_seq['labs'] = {'t': np.array(new_t), 'f': np.array(new_f), 'v': np.array(new_v)}
                
                new_data.append(new_seq)
            
            out_path = output_dir / sp.name
            joblib.dump(new_data, out_path)
            print(f"  Processed {sp.name}")

        print(f"\nStats for processed shards:")
        print(f"Total Raw Points: {total_raw_points}")
        print(f"Total Clean Points Retained: {total_clean_points}")
        print(f"Outliers Removed: {total_raw_points - total_clean_points} ({((total_raw_points - total_clean_points)/total_raw_points)*100:.2f}%)")

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Create the frozen V3 event bundle from a V2 event bundle."
        )
        parser.add_argument("--source-bundle", type=Path, required=True)
        parser.add_argument("--output-bundle", type=Path, required=True)
        parser.add_argument("--quantile-shards", type=int, default=3)
        parser.add_argument(
            "--max-shards",
            type=int,
            default=0,
            help="Convert only the first N shards for a structural smoke test (0 converts all).",
        )
        return parser.parse_args()

    return SimpleNamespace(**locals())

_v3 = _load_v3()


def _run_extract() -> None:
    args = _extract.parse_args()
    _extract._setup_logging(args.verbose)
    if args.note_archive is not None:
        _extract.logger.info(
            "Extracted note file to %s",
            _extract.extract_discharge_notes(args.note_archive, _extract.config.NOTE_DIR),
        )
    _extract.run_preprocessing(args)


def _run_stream() -> None:
    args = _stream.parse_args()
    _stream._setup_logging(args.verbose)
    _stream.run(args)


def _run_convert() -> None:
    _convert.main()


def _run_v3() -> None:
    args = _v3.parse_args()
    if args.quantile_shards < 1:
        raise SystemExit("--quantile-shards must be positive")
    old_to_new_v_idx, old_to_new_l_idx, new_vitals, new_labs = _v3.build_mapping(
        args.source_bundle, args.output_bundle
    )
    v_bounds, l_bounds = _v3.compute_percentiles(
        args.source_bundle,
        old_to_new_v_idx,
        old_to_new_l_idx,
        num_shards_to_scan=args.quantile_shards,
    )
    _v3.convert_shards(
        args.source_bundle,
        args.output_bundle,
        old_to_new_v_idx,
        old_to_new_l_idx,
        new_vitals,
        new_labs,
        v_bounds,
        l_bounds,
        max_shards=args.max_shards,
    )
    if args.max_shards:
        index_path = args.output_bundle / "index.json"
        index = _v3.json.loads(index_path.read_text(encoding="utf-8"))
        index["shards"] = index["shards"][: args.max_shards]
        index["num_shards"] = len(index["shards"])
        index["total_sequences"] = sum(int(shard["sequence_count"]) for shard in index["shards"])
        index_path.write_text(_v3.json.dumps(index, indent=2), encoding="utf-8")


STAGES = {
    "extract": _run_extract,
    "stream": _run_stream,
    "convert": _run_convert,
    "freeze-v3": _run_v3,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multimodal event bundles")
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
