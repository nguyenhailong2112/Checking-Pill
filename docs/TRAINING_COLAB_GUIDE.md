# Hướng Dẫn Train Model Trên Google Colab

Tài liệu này hướng dẫn rõ dữ liệu nào cần upload lên Colab, train model nào,
chạy script nào, output lấy ở đâu và copy về chương trình local như thế nào.

## 1. Tổng Quan: Dataset Nào Train Model Nào?

| Mục tiêu | Dataset | Model output | Dùng ở runtime |
| --- | --- | --- | --- |
| Detect/count viên thuốc | ePillID `archive/` YOLO | `best.pt` | detector pill |
| Detect object mới tại site | YOLO annotation của site | `best.pt` | detector multi-class |
| Nhận diện loại thuốc mạnh hơn | ePillID crop CSV | `identity_embedding.pt` | phase tiếp theo |
| OK/NG bằng model | crop OK/NG thật | `quality_classifier.pt` | phase tiếp theo |
| Thêm loại thuốc nhanh | reference images | gallery folder | đã dùng được hiện tại |

Hiện tại runtime đã dùng được:

- detector YOLO `best.pt`,
- reference gallery,
- quality rule-based.

Hai model `identity_embedding.pt` và `quality_classifier.pt` đã có script train,
nhưng runtime adapter trực tiếp cho hai artifact này là bước triển khai tiếp
theo.

## 2. Chuẩn Bị Google Drive

Trên Google Drive nên tạo:

```text
MyDrive/
  EdgeVision/
    datasets/
      epillid_source/
        archive/
        ePillID_data/
      site_demo_yolo/
        images/train/
        images/val/
        labels/train/
        labels/val/
      quality/
        train/
        val/
    models/
      pill_detector_yolo/
      site_detector_yolo/
      pill_identity_embedding/
      quality_classifier/
```

Nếu bạn train pill detector từ ePillID, cần upload/copy:

```text
ePillID-benchmark-ePillID_data_v1.0/archive/
```

Nếu bạn train identity embedding, cần:

```text
ePillID_data/ePillID_data/all_labels.csv
ePillID_data/ePillID_data/classification_data/
```

Nếu bạn train detector cho object mới, cần dataset YOLO site:

```text
site_demo_yolo/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

## 3. Chuẩn Bị Colab Notebook

Trong Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Đưa repo EdgeVision vào Colab. Có thể:

- upload folder repo,
- clone từ GitHub nếu đã push,
- copy từ Drive.

Ví dụ repo nằm ở:

```text
/content/EdgeVision
```

Chạy:

```python
%cd /content/EdgeVision
!pip install -U ultralytics torch torchvision
```

## 4. Train Pill Detector Từ ePillID `archive/`

Dùng khi chỉ cần detector class `pill`.

Command:

```python
!python training/colab_train_yolo_detector.py \
  --dataset-root "/content/drive/MyDrive/EdgeVision/datasets/epillid_source/archive" \
  --class-names pill \
  --model yolo11n.pt \
  --epochs 120 \
  --imgsz 832 \
  --batch -1 \
  --device 0 \
  --cache \
  --output-dir "/content/drive/MyDrive/EdgeVision/models/pill_detector_yolo"
```

Giải thích tham số:

- `--dataset-root`: thư mục YOLO dataset.
- `--class-names pill`: class id `0` là pill.
- `--model yolo11n.pt`: model nhỏ, phù hợp định hướng Edge.
- `--epochs 120`: số epoch train.
- `--imgsz 832`: kích thước ảnh train/infer.
- `--batch -1`: để Ultralytics tự chọn batch.
- `--device 0`: dùng GPU Colab.
- `--cache`: cache dataset nếu RAM đủ.
- `--output-dir`: nơi lưu output trên Drive.

Output:

```text
/content/drive/MyDrive/EdgeVision/models/pill_detector_yolo/
  best.pt
  last.pt
  data.yaml
  dataset_audit.json
  val_metrics.json
  training_summary.json
```

Copy `best.pt` về local:

```text
EdgeVision/models/pill_detector_yolo/best.pt
```

Chạy local:

```powershell
python -m edgevision.cli batch `
  --input data\real_test_images `
  --output runs\real_test_yolo `
  --config configs\runtime_yolo_config.json `
  --yolo-weights models\pill_detector_yolo\best.pt
```

## 5. Train Detector Multi-Object Cho Site

Dùng khi user muốn detect thêm object khác:

```text
pill
bottle
blister
thermometer
syringe
```

Dataset phải là YOLO format:

```text
site_demo_yolo/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Class id phải đúng thứ tự:

```text
0 pill
1 bottle
2 blister
3 thermometer
4 syringe
```

Command:

```python
!python training/colab_train_yolo_detector.py \
  --dataset-root "/content/drive/MyDrive/EdgeVision/datasets/site_demo_yolo" \
  --class-names pill bottle blister thermometer syringe \
  --model yolo11n.pt \
  --epochs 150 \
  --imgsz 832 \
  --batch -1 \
  --device 0 \
  --cache \
  --output-dir "/content/drive/MyDrive/EdgeVision/models/site_detector_yolo"
```

Copy về local:

```text
site_projects/demo_site/models/best.pt
```

Chạy local:

```powershell
python -m edgevision.cli batch `
  --input data\real_test_images `
  --output runs\site_test `
  --config configs\runtime_yolo_config.json `
  --site-project site_projects\demo_site `
  --yolo-weights site_projects\demo_site\models\best.pt
```

## 6. Train Identity Embedding

Dùng khi muốn model học embedding nhận diện thuốc tốt hơn baseline gallery.

Dữ liệu:

```text
all_labels.csv
classification_data/
```

Command:

```python
!python training/colab_train_identity_embedding.py \
  --manifest "/content/drive/MyDrive/EdgeVision/datasets/epillid_source/ePillID_data/ePillID_data/all_labels.csv" \
  --image-root "/content/drive/MyDrive/EdgeVision/datasets/epillid_source/ePillID_data/ePillID_data/classification_data" \
  --label-column label \
  --model mobilenet_v3_small \
  --embedding-dim 256 \
  --epochs 40 \
  --batch 96 \
  --output-dir "/content/drive/MyDrive/EdgeVision/models/pill_identity_embedding"
```

Output:

```text
identity_embedding.pt
history.json
training_summary.json
```

Ghi chú quan trọng:

- Runtime hiện tại vẫn dùng reference gallery.
- `identity_embedding.pt` là artifact chuẩn bị cho phase tiếp theo.
- Khi có adapter runtime, embedding sẽ giúp nhận diện low-shot tốt hơn.

## 7. Train OK/NG Quality Classifier

Chỉ train khi có dữ liệu OK/NG thật.

Cấu trúc:

```text
quality/
  train/
    OK/
    NG_broken/
    NG_chipped/
    NG_stain/
  val/
    OK/
    NG_broken/
    NG_chipped/
    NG_stain/
```

Command:

```python
!python training/colab_train_quality_classifier.py \
  --data-root "/content/drive/MyDrive/EdgeVision/datasets/quality" \
  --model mobilenet_v3_small \
  --epochs 60 \
  --batch 96 \
  --output-dir "/content/drive/MyDrive/EdgeVision/models/quality_classifier"
```

Output:

```text
quality_classifier.pt
history.json
training_summary.json
```

Ghi chú:

- Nếu data OK/NG chưa rõ, không train vội.
- Case mơ hồ nên để `REVIEW`.
- False OK cho NG là lỗi nguy hiểm nhất.

## 8. Kiểm Tra Sau Khi Train

Sau mỗi lần train, giữ lại:

```text
best.pt hoặc model .pt
training_summary.json
val_metrics.json hoặc history.json
dataset_audit.json
data.yaml
command đã chạy
```

Không overwrite model tốt cũ nếu chưa backup.

Khuyến nghị đặt tên folder theo ngày:

```text
models/
  pill_detector_yolo_2026_05_08/
  pill_detector_yolo_2026_05_09/
```

## 9. Lỗi Thường Gặp Khi Train YOLO

### Lỗi class id out of range

Nguyên nhân:

- label `.txt` có class id lớn hơn số class trong `--class-names`.

Cách sửa:

- kiểm tra lại class order,
- kiểm tra file label,
- kiểm tra `data.yaml`.

### Lỗi không thấy ảnh train

Nguyên nhân:

- sai `--dataset-root`,
- folder không đúng `images/train`,
- ảnh nằm sai cấp thư mục.

### Val metric cao nhưng test thực tế kém

Nguyên nhân:

- val quá giống train,
- thiếu ảnh hard case,
- ảnh site khác domain so với dataset train.

Cách xử lý:

- thêm ảnh thực tế site,
- tách val/test đúng hơn,
- bổ sung negative/background images.

## 10. Quy Tắc Chọn Model

Ưu tiên Edge:

```text
yolo11n.pt hoặc model nano/small tương đương
```

Nếu recall không đủ:

```text
thử model small
```

Không chọn model lớn ngay từ đầu nếu mục tiêu cuối là Edge AI Device.

## 11. Sau Khi Có `best.pt`

Copy về local đúng chỗ:

Pill detector:

```text
models/pill_detector_yolo/best.pt
```

Site detector:

```text
site_projects/demo_site/models/best.pt
```

Chạy test local ngay:

```powershell
python -m edgevision.cli batch `
  --input data\real_test_images `
  --output runs\model_check `
  --config configs\runtime_yolo_config.json `
  --yolo-weights models\pill_detector_yolo\best.pt
```

Sau đó review:

```text
runs/model_check/summary.csv
runs/model_check/*/annotated.jpg
runs/model_check/*/report.json
```
