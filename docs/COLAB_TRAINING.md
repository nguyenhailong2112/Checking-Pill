# Google Colab Training Guide

This guide trains the stage-1 pill detector with the existing YOLO labels from
the source `archive/` dataset.

## 1. Prepare Files For Colab

Recommended Google Drive layout:

```text
MyDrive/EdgeVision/
  source/
    archive/
      images/
        train/
        val/
      labels/
        train/
        val/
      data.yaml
  repo/
    EdgeVision/
```

You can upload/copy the entire local source folder or at least:

```text
ePillID-benchmark-ePillID_data_v1.0/archive
```

Detection training only needs `archive`.

## 2. Colab Setup

In a Colab notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Clone or upload the `EdgeVision` repo under:

```text
/content/drive/MyDrive/EdgeVision/repo/EdgeVision
```

Install dependencies:

```python
!pip install -U ultralytics
```

## 3. Audit Labels Before Training

```python
%cd /content/drive/MyDrive/EdgeVision/repo/EdgeVision

!python tools/check_yolo_dataset.py \
  --dataset-root /content/drive/MyDrive/EdgeVision/source/archive \
  --output runs/colab_archive_yolo_audit.json
```

The audit should report:

- no missing image/label pairs,
- no malformed label lines,
- no out-of-range normalized coordinates, or only a very small number that will
  be sanitized before training,
- class id only `0`,
- empty label files counted as valid negative images.

If the audit reports boxes that slightly exceed the image boundary, export a
sanitized copy before training. This does not modify the original source data:

```python
!python tools/export_yolo_dataset.py \
  --source-root /content/drive/MyDrive/EdgeVision/source/archive \
  --output-root /content/drive/MyDrive/EdgeVision/datasets/archive_yolo_sanitized \
  --overwrite
```

## 4. Train YOLO

```python
!python training/train_yolo_detector.py \
  --dataset-root /content/drive/MyDrive/EdgeVision/datasets/archive_yolo_sanitized \
  --model yolo11s.pt \
  --epochs 80 \
  --imgsz 832 \
  --batch 16 \
  --project runs/train \
  --name pill_detector_yolo11s_832
```

For a lighter run:

```python
!python training/train_yolo_detector.py \
  --dataset-root /content/drive/MyDrive/EdgeVision/datasets/archive_yolo_sanitized \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --project runs/train \
  --name pill_detector_yolo11n_640
```

## 5. Download Weights

After training, the important files are:

```text
runs/train/pill_detector_yolo11s_832/weights/best.pt
runs/train/pill_detector_yolo11s_832/weights/last.pt
```

Copy `best.pt` back to local:

```text
C:/Users/longn/PyCharmMiscProject/EdgeVision/models/pill_detector_yolo/best.pt
```

Then run EdgeVision locally with:

```powershell
python -m edgevision.cli infer `
  --image path\to\real_image.jpg `
  --output runs\real_test `
  --detector-backend yolo `
  --yolo-weights models\pill_detector_yolo\best.pt
```

Local YOLO inference requires `ultralytics` and its PyTorch dependencies.
