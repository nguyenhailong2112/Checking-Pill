# Dataset Strategy

EdgeVision should use all available data, but not as one mixed dataset. The data
must be separated by task because each model needs a different label contract.

## Source Data Policy

The benchmark/source folder is treated as immutable:

```text
C:/Users/longn/PyCharmMiscProject/ePillID-benchmark-ePillID_data_v1.0
```

Do not edit, rename, or relabel files in that folder directly. EdgeVision should
read from it and create derived manifests/configs under:

```text
EdgeVision/data/
EdgeVision/artifacts/
EdgeVision/runs/
```

These folders can be regenerated.

## Recommended Dataset Layout

```text
EdgeVision/
  data/
    manifests/
      source_inventory.json
      archive_yolo_audit.json
      epillid_identity_manifest.csv
      mediseg_unlabeled_manifest.csv

    detection_yolo/
      data.yaml
      README.md

    identity/
      epillid_all.csv
      epillid_reference_gallery.csv

    quality/
      README.md
      train/
        OK/
        NG_broken/
        NG_chipped/
        NG_color_abnormal/
        NG_stain/
      val/
      test/
```

The first stage does not need to copy images. The generated manifests can point
to the original absolute source paths.

## Task-Specific Datasets

### 1. Detection / Counting

Source:

```text
source/archive/images/train
source/archive/images/val
source/archive/labels/train
source/archive/labels/val
```

Label format:

```text
class_id x_center y_center width height
```

All values after `class_id` are normalized to `[0, 1]`. Class `0` means `pill`.
Empty label files are valid negative images.

Current audit result on the local source archive:

```text
images: 12,067
labels: 12,067
missing labels: 0
labels without images: 0
boxes: 144,111 total raw boxes
invalid/out-of-bound lines found by strict audit: 29
```

Those 29 boxes exceed the normalized image boundary slightly. The source data
should remain unchanged; use `tools/export_yolo_dataset.py` to create a
sanitized training copy that clips those boxes into valid YOLO coordinates.

Model:

```text
YOLO one-class pill detector
```

Outputs:

```text
models/pill_detector_yolo/best.pt
models/pill_detector_yolo/last.pt
```

### 2. Identity / Recognition

Source:

```text
source/ePillID_data/ePillID_data/all_labels.csv
source/ePillID_data/ePillID_data/classification_data
```

Main columns:

```text
label, pilltype_id, image_path, is_ref, is_front, is_new
```

Stage-1 use:

```text
reference-gallery retrieval
```

The current EdgeVision identifier can read `all_labels.csv` directly with:

```text
--reference-manifest <all_labels.csv>
--reference-image-root <classification_data>
```

Future model:

```text
embedding network trained with metric learning
```

### 3. Quality OK/NG

Current source data does not contain explicit OK/NG defect labels. Do not train a
supervised quality classifier until labels exist.

Initial stage:

```text
rule-based + anomaly baseline
```

Future supervised dataset:

```text
quality/train/OK
quality/train/NG_broken
quality/train/NG_chipped
quality/train/NG_color_abnormal
quality/train/NG_stain
```

## What "Use 100% Data" Means

Using all data does not mean mixing all images into one training set. It means:

- all YOLO-labeled images are used for detector training/evaluation,
- all ePillID crop images are used for identity gallery/training,
- all MEDISEG images are preserved as unlabeled/domain data until labeling or
  pseudo-labeling is intentionally added,
- negative images remain negative detection samples,
- train/val/test boundaries are never mixed.
