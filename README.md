# Code map

The submission release contains 27 Python files. Preprocessing is limited to
five files, and the 13 former downstream scripts are organized into three
manuscript-section analysis entry points plus one release utility.

## Preprocessing: exactly five files

| File | Purpose |
| --- | --- |
| `preprocessing/clinical_definitions.py` | Shared item definitions, time windows, SOFA, and Charlson functions |
| `preprocessing/build_mimic_clinical_table.py` | Six auditable MIMIC-IV clinical-table stages |
| `preprocessing/build_eicu_clinical_table.py` | eICU external-validation clinical table |
| `preprocessing/multimodal_features.py` | Raw-table loading, sequence classes, and optional note features |
| `preprocessing/build_event_bundle.py` | Feature extraction, sharding, conversion, and frozen V3 bundle |

## Analysis–plot pairs

| Manuscript section | Analysis | Plotting |
| --- | --- | --- |
| Clinical states and transport | `analysis/analysis_clinical_states.py` | `plots/plot_clinical_states.py` |
| Longitudinal pathways | `analysis/analysis_longitudinal_pathways.py` | `plots/plot_longitudinal_pathways.py` |
| Metabolic buffering failure | `analysis/analysis_metabolic_buffering.py` | `plots/plot_metabolic_buffering.py` |

Each combined entry lists its retained stages with `--help`. For example:

```text
python analysis/analysis_clinical_states.py --help
python analysis/analysis_clinical_states.py eicu-transport --help
python plots/plot_metabolic_buffering.py --help
```

## Other current-paper entries

- `training/train_ecdt_v3_scratch.py`: main PPD-EHR training.
- `training/runtime/npj_external_validation_v3.py`: eICU external validation;
  the local-update analysis uses the prespecified 20% eICU development subset.
- `latent_export/export_fixed_time_latents_seed42_721.py`: fixed-time latent export.
- `latent_export/manage_latent_exports.py`: latent validation and finalization.
- `plots/plot_model_performance.py`: observation-density and performance figures.
- `plots/plot_external_validation.py`: external-validation figure.
- `plots/plot_patient_evidence.py`: patient-level evidence figure.
- `analysis/release_outputs.py`: source-data export and checksums.

The combined files use labelled internal sections and isolated namespaces. This
keeps the original calculations traceable without exposing dozens of date-stamped
one-off scripts as separate entry points.
