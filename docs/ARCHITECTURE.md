# EdgeVision Architecture

## System Contract

The stage-1 system receives an image and returns:

- detected pill bounding boxes,
- total pill count,
- count by predicted type,
- per-pill identity prediction,
- per-pill OK/NG/REVIEW decision,
- annotated image,
- machine-readable JSON report.

The core design is modular because detection, identity, and quality inspection
have different data sources, metrics, and failure modes.

```text
ImagePrecheck
 -> Detector
 -> PostProcessor
 -> CropNormalizer
 -> Identifier
 -> QualityInspector
 -> DecisionFusion
 -> Reporter
```

## Data Mapping

The ePillID benchmark folder remains a source-data repository:

- `archive/`: YOLO-format one-class pill detection dataset.
- `ePillID_data/`: crop-level reference/consumer pill identity dataset.
- `MEDISEG/`: additional pill images and medication metadata.

EdgeVision should not mutate those source folders. Training scripts should read
from them, produce derived datasets under this repo's ignored `artifacts/` or
`runs/`, and record the source path in metadata.

## Module Boundaries

### Detector

Input: RGB image.

Output: list of `Detection` objects with `bbox`, `confidence`, and `label`.

The detector should only answer "where are pill instances?" It should not decide
the medication type. This keeps counting robust and allows the detector to be
trained with a single class.

### Identifier

Input: normalized crop.

Output: `IdentificationResult` with top-k candidates and confidence.

Two strategies are supported by design:

- closed-set classifier for fixed production lines,
- reference-gallery retrieval for low-shot and expandable medication sets.

The initial gallery loader supports two source contracts:

- folder gallery: `reference_root/<label>/*.jpg`,
- CSV manifest: columns `label`, `image_path`, optional `is_ref`.

### Quality Inspector

Input: normalized crop and optional identity result.

Output: `QualityResult` with status `OK`, `NG`, or `REVIEW`.

The initial implementation is rule based. It must be replaced or augmented by
supervised/anomaly models when real OK/NG data is available.

### Decision Fusion

The fused status should avoid forced binary decisions when evidence is weak:

- `OK`: confident identity and no strong quality anomaly.
- `NG`: strong defect evidence.
- `REVIEW`: low detector confidence, unknown identity, poor image quality, or
  weak quality score.

## Edge AI Preparation

Each model-facing component is isolated so stage 2 can export or replace it:

- detector: YOLO to ONNX/TensorRT/OpenVINO,
- identifier: embedding model to ONNX plus gallery index,
- inspector: rule engine or anomaly model,
- reporter: unchanged business logic.
