# EdgeVision

EdgeVision is a pill visual inspection project. The stage-1 goal is an end-to-end
pipeline that can detect pills, count them, identify their type when references
are available, and mark each pill as OK, NG, or REVIEW.

The project is intentionally structured as a system, not a single model:

```text
input image
 -> image precheck
 -> pill detector
 -> bbox post-processing and counting
 -> crop normalizer
 -> pill identifier
 -> quality inspector
 -> decision fusion
 -> annotated image + JSON report
```

## Current Implementation

This initial repo contains a dependency-light runnable core:

- `edgevision.detectors.heuristic`: PIL/Numpy connected-component detector for
  fast local pipeline testing.
- `edgevision.detectors.ultralytics_yolo`: optional adapter for a real YOLO
  detector when `ultralytics` is installed.
- `edgevision.identifiers.reference_gallery`: reference-gallery nearest-neighbor
  identifier based on deterministic color/shape features. It can read either a
  folder gallery or a CSV manifest such as ePillID's `all_labels.csv`.
- `edgevision.quality.rule_based`: rule-based OK/NG/REVIEW inspector.
- `edgevision.pipeline`: complete orchestration and report generation.
- `tools/audit_epillid_source.py`: source-data audit helper for the ePillID
  benchmark folder.
- `tools/check_yolo_dataset.py`: strict YOLO label audit.
- `tools/export_yolo_dataset.py`: create a sanitized YOLO training copy without
  modifying source data.

The heuristic detector and feature identifier are not the final production AI
models. They are a stable executable baseline so every downstream contract can
be built and tested before heavy model training starts.

## Quick Start

Use a Python environment with at least `numpy` and `Pillow`.

```powershell
python -m edgevision.cli infer --image path\to\image.jpg --output runs\demo
```

Use ePillID reference images as a gallery without copying the source data:

```powershell
python -m edgevision.cli infer ^
  --image path\to\image.jpg ^
  --output runs\demo ^
  --reference-manifest C:\Users\longn\PyCharmMiscProject\ePillID-benchmark-ePillID_data_v1.0\ePillID_data\ePillID_data\all_labels.csv ^
  --reference-image-root C:\Users\longn\PyCharmMiscProject\ePillID-benchmark-ePillID_data_v1.0\ePillID_data\ePillID_data\classification_data
```

If the local machine does not expose `python` on PATH, the Codex bundled Python
used during development is:

```powershell
C:\Users\longn\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m edgevision.cli infer --image path\to\image.jpg --output runs\demo
```

Run the synthetic pipeline test:

```powershell
python -m unittest discover -s tests
```

## Project Layout

```text
EdgeVision/
  configs/                  Runtime configuration examples
  docs/                     Architecture and algorithm notes
  edgevision/               Python package
    detectors/              Detector interfaces and implementations
    identifiers/            Pill type identification modules
    quality/                OK/NG inspection modules
    visualization/          Annotation utilities
  tests/                    Synthetic tests for the core contracts
  tools/                    Dataset audit and utility scripts
```

## Stage-1 Roadmap

1. Data audit and dataset contracts.
2. YOLO pill detector for detection/counting.
3. Crop normalizer and report schema.
4. Reference-gallery pill identifier.
5. Rule/anomaly based OK/NG baseline.
6. Evaluation harness for count, detection, identity, and quality.
7. Model export preparation for Edge AI in stage 2.
