"""Export supplementary source-data tables from frozen downstream outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", choices=("export", "manifest", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=package / "source_data")
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifests(package: Path) -> dict[str, int]:
    """Regenerate source-data, figure and package SHA-256 manifests."""
    source_root = package / "source_data"
    source_rows = []
    for path in sorted(source_root.rglob("*.csv")):
        if path.name == "source_data_manifest.csv":
            continue
        try:
            rows, columns = pd.read_csv(path).shape
        except Exception:
            rows, columns = None, None
        relative = path.relative_to(source_root).as_posix()
        source_rows.append({
            "file": relative, "rows": rows, "columns": columns,
            "role": "legacy exploratory" if relative.startswith("legacy/") else "final live source data",
            "sha256": digest(path),
        })
    source_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(source_root / "source_data_manifest.csv", index=False)

    figure_root = package / "figures"
    figure_rows = []
    if figure_root.exists():
        for path in sorted(figure_root.rglob("*")):
            if not path.is_file() or path.name == "figure_export_manifest.csv":
                continue
            relative = path.relative_to(figure_root).as_posix()
            row = {
                "file": relative, "bytes": path.stat().st_size,
                "role": "legacy exploratory" if relative.startswith("legacy/") else "final live figure",
                "sha256": digest(path), "width_px": None, "height_px": None,
                "dpi_x": None, "dpi_y": None, "mode": None,
            }
            if path.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
                with Image.open(path) as image:
                    row["width_px"], row["height_px"] = image.size
                    row["dpi_x"], row["dpi_y"] = image.info.get("dpi", (None, None))
                    row["mode"] = image.mode
            figure_rows.append(row)
        pd.DataFrame(figure_rows).to_csv(figure_root / "figure_export_manifest.csv", index=False)

    manifest_dir = package / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    target = manifest_dir / "file_sha256_manifest.csv"
    package_rows = []
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.resolve() == target.resolve():
            continue
        stat = path.stat()
        package_rows.append({
            "relative_path": path.relative_to(package).as_posix(),
            "size_bytes": stat.st_size,
            "last_write_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "sha256": digest(path),
        })
    pd.DataFrame(package_rows).to_csv(target, index=False, quoting=1)
    return {"source_files": len(source_rows), "figure_files": len(figure_rows), "package_files": len(package_rows)}


def copy_outputs(package: Path, output_dir: Path) -> list[str]:
    mbf = package / "outputs" / "downstream" / "metabolic_buffering_failure"
    validation = package / "outputs" / "downstream" / "metabolic_buffering_failure_validation"
    copies = {
        "Supplement_mbf_absolute_mismatch_quartile_effects.csv": mbf / "mbf_absolute_mismatch_quartile_effects.csv",
        "Supplement_mbf_absolute_mismatch_quartile_outcomes.csv": mbf / "mbf_absolute_mismatch_quartile_outcomes.csv",
        "Supplement_mbf_external_eicu_model_metrics.csv": validation / "mbf_external_eicu_model_metrics.csv",
        "Supplement_mbf_incremental_value_multi_target.csv": mbf / "mbf_incremental_value_multi_target.csv",
        "Supplement_mbf_incremental_value_multi_target_lr_sensitivity.csv": mbf / "mbf_incremental_value_multi_target_lr_sensitivity.csv",
        "Supplement_mbf_landscape_empirical_support_with_ess.csv": mbf / "mbf_landscape_empirical_support_with_ess.csv",
        "Supplement_mbf_reliability_model_metrics_mimic.csv": validation / "mbf_reliability_model_metrics_mimic.csv",
        "Supplement_mbf_reliability_window_correlations_mimic.csv": validation / "mbf_reliability_window_correlations_mimic.csv",
        "Supplement_mbf_reliability_window_summary_mimic.csv": validation / "mbf_reliability_window_summary_mimic.csv",
        "Supplement_mbf_signed_directional_coefficients.csv": mbf / "mbf_signed_directional_coefficients.csv",
    }
    exported = []
    for target_name, source in copies.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, output_dir / target_name)
        exported.append(target_name)
    return exported


def build_weight_support_audit(package: Path, output_dir: Path) -> str:
    rows = []
    cohorts = [
        (
            "Primary pathway cohort: diabetes",
            package / "outputs" / "downstream" / "longitudinal_pathway",
            "longitudinal_dataset_audit.json",
            "causal_dataset",
            "stays",
            "pathway_landmark_stays",
            "strict_cam_assessed_in_pathway_landmark",
            "strict_cam_events_in_pathway_landmark",
        ),
        (
            "Secondary outcome-anchor cohort: all ICU",
            package / "outputs" / "downstream" / "longitudinal_pathway_full_icu",
            "full_icu_longitudinal_dataset_audit.json",
            None,
            "stays",
            "pathway_landmark",
            "strict_cam_assessed",
            "strict_cam_events",
        ),
    ]
    for label, root, audit_name, nested, n_source, n_landmark, n_assessed, n_events in cohorts:
        audit = json.loads((root / audit_name).read_text(encoding="utf-8"))
        audit = audit[nested] if nested else audit
        regime = json.loads((root / audit_name).read_text(encoding="utf-8")).get("regime_values", {})
        cam = pd.read_csv(root / "cam_observation_weight_audit.csv").iloc[0]
        selection = json.loads((root / "landmark_selection_competing_audit.json").read_text(encoding="utf-8"))
        rows.extend([
            {
                "cohort": label,
                "process": "CAM assessment",
                "n_source": int(audit[n_source]),
                "n_landmark": int(cam.n_landmark),
                "n_assessed": int(cam.assessed),
                "events": int(cam.events),
                "rate": float(cam.assessment_rate),
                "weight_p01": float(cam.weight_p01),
                "weight_p99": float(cam.weight_p99),
                "effective_sample_size": float(cam.effective_sample_size),
                "stable_energy_q25": float(regime["stable_energy_q25"]),
                "unstable_energy_q75": float(regime["unstable_energy_q75"]),
                "policy": "stabilized assessment weight truncated at assessed-arm p01/p99",
            },
            {
                "cohort": label,
                "process": "48-h landmark selection",
                "n_source": int(audit[n_source]),
                "n_landmark": int(selection["landmark_stays"]),
                "n_assessed": int(cam.assessed),
                "events": int(cam.events),
                "rate": float(selection["landmark_rate"]),
                "weight_p01": float(selection["weight_p01"]),
                "weight_p99": float(selection["weight_p99"]),
                "effective_sample_size": float(selection["selection_weight_ESS"]),
                "stable_energy_q25": float(regime["stable_energy_q25"]),
                "unstable_energy_q75": float(regime["unstable_energy_q75"]),
                "policy": "stabilized landmark IPCW truncated at selected-arm p01/p99",
            },
        ])
    name = "Supplement_longitudinal_weight_and_support_audit.csv"
    table = pd.DataFrame(rows)
    for column in [
        "rate", "weight_p01", "weight_p99",
        "stable_energy_q25", "unstable_energy_q75",
    ]:
        table[column] = table[column].map(lambda value: f"{value:.10f}")
    table["effective_sample_size"] = table["effective_sample_size"].map(
        lambda value: f"{value:.6f}"
    )
    table.to_csv(output_dir / name, index=False, lineterminator="\n")
    return name


def main() -> None:
    args = parse_args()
    package = Path(__file__).resolve().parents[2]
    report: dict[str, object] = {"status": "ok", "action": args.action}
    if args.action in {"export", "all"}:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        exported = copy_outputs(package, args.output_dir)
        exported.append(build_weight_support_audit(package, args.output_dir))
        report["exported"] = exported
    if args.action in {"manifest", "all"}:
        report["manifests"] = build_manifests(package)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
