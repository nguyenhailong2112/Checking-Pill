# Chiến Lược Dữ Liệu

Tài liệu này trả lời câu hỏi quan trọng:

```text
Dữ liệu nào dùng để train model nào?
```

Không được trộn tất cả ảnh thành một dataset duy nhất. Detection, identity và
OK/NG là ba bài toán khác nhau, cần label khác nhau và metric khác nhau.

## 1. Nguyên Tắc Dữ Liệu

Thư mục source ePillID được xem là dữ liệu gốc, không chỉnh sửa trực tiếp:

```text
C:/Users/longn/PyCharmMiscProject/ePillID-benchmark-ePillID_data_v1.0
```

Nếu cần tạo bản train sạch hơn, tạo bản copy/dẫn xuất dưới repo EdgeVision:

```text
EdgeVision/data/
EdgeVision/models/
EdgeVision/runs/
EdgeVision/site_projects/
```

Không rename, sửa label, xóa file trực tiếp trong source gốc.

## 2. Bảng Dữ Liệu -> Model

| Mục tiêu | Dữ liệu cần | Label cần có | Script train | Artifact |
| --- | --- | --- | --- | --- |
| Detect/count pill | ePillID `archive/` | YOLO bbox class `pill` | `colab_train_yolo_detector.py` | `best.pt` |
| Detect object mới | YOLO site annotations | bbox cho từng class user label | `colab_train_yolo_detector.py` | `best.pt` |
| Nhận diện loại thuốc | ePillID crop hoặc reference gallery | label loại thuốc | gallery hoặc `colab_train_identity_embedding.py` | gallery / `identity_embedding.pt` |
| OK/NG | crop lỗi thật | OK/NG hoặc defect subtype | `colab_train_quality_classifier.py` | `quality_classifier.pt` |

## 3. Source ePillID Hiện Có

Audit local source:

```text
YOLO archive:
  train images: 10,559
  val images: 1,508
  labels: 12,067
  boxes: 144,111
  empty label files: 456

Identity ePillID:
  rows: 13,532
  labels: 4,902
  reference rows: 9,804
  consumer rows: 3,728

MEDISEG:
  images: 10,594
```

Ý nghĩa:

- `archive/` dùng tốt cho detector/counting.
- `ePillID_data/all_labels.csv` dùng cho identity/reference/embedding.
- `MEDISEG` hiện nên xem là domain data, chưa dùng supervised nếu chưa có label
  phù hợp.

## 4. Dataset Cho Detector

Detector cần dữ liệu YOLO:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Mỗi file label `.txt`:

```text
class_id x_center y_center width height
```

Tất cả tọa độ normalize về `[0, 1]`.

### Pill-only detector

Class:

```text
0: pill
```

Train ra:

```text
models/pill_detector_yolo/best.pt
```

### Multi-object site detector

Ví dụ class:

```text
0: pill
1: bottle
2: blister
3: thermometer
4: syringe
```

Train ra:

```text
site_projects/demo_site/models/best.pt
```

Nếu user muốn object mới, ví dụ `cotton_swab`, thêm class mới vào dataset YOLO và
train lại detector site.

## 5. Dataset Cho Identity

Identity không cần ảnh full-scene, mà cần crop hoặc ảnh reference của từng loại
thuốc.

Gallery đơn giản:

```text
reference_gallery/
  RoundWhiteTablet/
    ref_001.jpg
    ref_002.jpg
  RedWhiteCapsule/
    ref_001.jpg
```

Site project gallery:

```text
site_projects/demo_site/reference_gallery/
  RoundWhiteTablet/
    ref_001.jpg
```

ePillID external gallery:

```text
all_labels.csv
classification_data/
```

Lệnh runtime:

```powershell
python -m edgevision.cli batch `
  --input data\real_test_images `
  --output runs\epillid_gallery_test `
  --config configs\runtime_yolo_config.json `
  --reference-manifest C:\Users\longn\PyCharmMiscProject\ePillID-benchmark-ePillID_data_v1.0\ePillID_data\ePillID_data\all_labels.csv `
  --reference-image-root C:\Users\longn\PyCharmMiscProject\ePillID-benchmark-ePillID_data_v1.0\ePillID_data\ePillID_data\classification_data
```

## 6. Dataset Cho OK/NG

OK/NG cần dữ liệu lỗi thật. Nếu chưa có ảnh lỗi thật thì không nên claim model
OK/NG supervised.

Cấu trúc đề xuất:

```text
data/quality/
  train/
    OK/
    NG_broken/
    NG_chipped/
    NG_stain/
    NG_color_abnormal/
  val/
    OK/
    NG_broken/
    NG_chipped/
    NG_stain/
    NG_color_abnormal/
```

Nguồn crop có thể lấy từ runtime:

```text
runs/<run_name>/<image_id>/crops/
```

Quy trình:

1. Chạy inference trên ảnh thật.
2. Mở crop từng object.
3. Người dùng hoặc engineer gán nhãn OK/NG.
4. Copy crop vào folder tương ứng.
5. Chỉ train quality classifier khi label đã đủ tin cậy.

## 7. Site Project Data

Mỗi site nên có project riêng:

```text
site_projects/demo_site/
  captures/                  ảnh/site frame gốc
  reference_gallery/          ảnh reference loại thuốc
  quality/                    ảnh OK/NG theo site
  annotations/yolo/           dữ liệu YOLO cho detector site
  models/                     weights của site
  runs/                       output test site
```

Tạo bằng:

```powershell
python -m edgevision.cli project-init `
  --path site_projects\demo_site `
  --name demo_site `
  --classes pill bottle blister thermometer syringe `
  --identity-labels pill `
  --quality-labels pill
```

## 8. Khi Nào Cần Train Lại?

### Không cần train lại detector nếu:

- detector đã tìm được viên thuốc tốt,
- bạn chỉ thêm loại thuốc mới,
- bạn chỉ thêm reference ảnh cho loại thuốc.

Khi đó chỉ cần:

```powershell
python -m edgevision.cli project-add-reference ...
```

### Cần train lại detector nếu:

- có object class mới cần detect,
- detector bỏ sót object,
- detector nhận nhầm nhiều,
- setup camera/nền/ánh sáng khác nhiều so với data train.

### Cần train quality model nếu:

- rule-based không đủ,
- có nhiều defect type,
- đã có crop OK/NG thật và label đáng tin cậy.

## 9. Nguyên Tắc Train/Val/Test

Không được để ảnh gần giống nhau rơi lung tung vào train và val nếu điều đó làm
metric ảo.

Khuyến nghị:

- train: ảnh dùng để học,
- val: ảnh dùng chọn model/threshold,
- test: ảnh giữ lại đến cuối mới đánh giá.

Với site thực tế, nên giữ một tập test riêng gồm ảnh khó:

- object chạm nhau,
- ánh sáng khác,
- nền khác,
- ảnh blur nhẹ,
- object gần biên ảnh,
- case OK/NG khó.

## 10. Checklist Trước Khi Train

Trước khi train detector:

- `images/train` có ảnh,
- `images/val` có ảnh,
- `labels/train` có `.txt`,
- `labels/val` có `.txt`,
- class id trong label khớp `data.yaml`,
- bbox normalize đúng,
- không thiếu label file ngoài ý muốn.

Trước khi train identity:

- ảnh crop/reference rõ,
- label loại thuốc đúng,
- mỗi loại có đủ ảnh nếu có thể,
- không trộn nhầm hai loại giống nhau.

Trước khi train quality:

- định nghĩa OK/NG rõ,
- defect type rõ,
- label đã được review,
- không đưa ảnh mơ hồ vào OK/NG cứng.
