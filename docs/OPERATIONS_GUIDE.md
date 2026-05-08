# Hướng Dẫn Vận Hành Thực Tế

Tài liệu này là checklist thao tác khi bạn muốn dùng EdgeVision như một chương
trình inspection thực tế: chuẩn bị ảnh, tạo site project, thêm reference, train
model, chạy inference, đọc kết quả, và cải thiện model.

## 1. Các Chế Độ Vận Hành

### Chế độ A: Smoke demo

Mục đích:

- kiểm tra code chạy được,
- kiểm tra output có sinh ra không,
- kiểm tra report schema không lỗi.

Lệnh:

```powershell
python -m edgevision.cli demo --output runs\demo_smoke
```

Khi nào chạy:

- sau khi pull/sửa code,
- sau khi cài môi trường mới,
- trước khi test ảnh thật.

### Chế độ B: Runtime pill cơ bản

Mục đích:

- detect/count viên thuốc,
- nhận diện loại thuốc nếu có reference,
- kiểm tra OK/NG baseline.

Lệnh:

```powershell
python -m edgevision.cli batch `
  --input data\real_test_images `
  --output runs\real_test `
  --config configs\runtime_yolo_config.json `
  --yolo-weights models\pill_detector_yolo\best.pt `
  --reference-root reference_gallery
```

### Chế độ C: Runtime site nhiều object

Mục đích:

- detect pill và object khác,
- dùng model site riêng,
- dùng reference gallery site riêng.

Lệnh:

```powershell
python -m edgevision.cli batch `
  --input data\real_test_images `
  --output runs\site_test `
  --config configs\runtime_yolo_config.json `
  --site-project site_projects\demo_site `
  --yolo-weights site_projects\demo_site\models\best.pt
```

## 2. Quy Trình Setup Một Site Mới

### Bước 1: Xác định object cần detect

Ví dụ site cần:

```text
pill
bottle
blister
thermometer
syringe
```

Nếu sau này có object mới, ví dụ `cotton_swab`, cần thêm class và train lại site
detector.

### Bước 2: Tạo site project

```powershell
python -m edgevision.cli project-init `
  --path site_projects\demo_site `
  --name demo_site `
  --classes pill bottle blister thermometer syringe `
  --identity-labels pill `
  --quality-labels pill
```

### Bước 3: Kiểm tra thư mục project

```powershell
python -m edgevision.cli project-summary --project site_projects\demo_site
```

Bạn sẽ thấy:

```text
reference_counts
yolo_counts
data_yaml
```

### Bước 4: Đặt ảnh test

Đặt ảnh vào:

```text
data/real_test_images/
```

hoặc:

```text
site_projects/demo_site/captures/
```

### Bước 5: Đặt model detector

Sau khi train trên Colab, copy:

```text
best.pt
```

vào:

```text
site_projects/demo_site/models/best.pt
```

### Bước 6: Thêm reference loại thuốc

```powershell
python -m edgevision.cli project-add-reference `
  --project site_projects\demo_site `
  --label RoundWhiteTablet `
  --image path\to\round_white_ref_001.jpg
```

Lặp lại cho nhiều ảnh reference.

## 3. Quy Trình User Label Object Mới

Khi người dùng muốn camera/chương trình học object mới:

### Bước 1: Chụp ảnh đại diện

Nên có:

- object ở nhiều vị trí,
- nhiều góc xoay,
- nhiều khoảng cách,
- nhiều điều kiện sáng,
- object chạm nhau,
- object sát biên ảnh,
- ảnh nền không có object.

### Bước 2: Gán bbox

Dùng tool annotation bất kỳ có export YOLO format.

Kết quả cần đưa vào:

```text
site_projects/demo_site/annotations/yolo/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
```

### Bước 3: Kiểm tra class id

Ví dụ:

```text
0 pill
1 bottle
2 blister
3 thermometer
4 syringe
```

File label phải dùng đúng class id này.

### Bước 4: Train detector trên Colab

Dùng:

```text
training/colab_train_yolo_detector.py
```

Chi tiết xem:

```text
docs/TRAINING_COLAB_GUIDE.md
```

### Bước 5: Copy model về local

Copy:

```text
/content/drive/MyDrive/EdgeVision/models/site_detector_yolo/best.pt
```

về:

```text
site_projects/demo_site/models/best.pt
```

### Bước 6: Chạy test site

```powershell
python -m edgevision.cli batch `
  --input data\real_test_images `
  --output runs\site_test `
  --config configs\runtime_yolo_config.json `
  --site-project site_projects\demo_site `
  --yolo-weights site_projects\demo_site\models\best.pt
```

### Bước 7: Review lỗi

Xem:

```text
runs/site_test/summary.csv
runs/site_test/*/annotated.jpg
runs/site_test/*/report.json
runs/site_test/*/crops/
```

Phân loại lỗi:

- miss object,
- duplicate bbox,
- wrong class,
- wrong identity,
- wrong OK/NG,
- ảnh chụp quá xấu.

Sau đó bổ sung data và train lại.

## 4. Quy Trình Thêm Loại Thuốc Mới

Nếu detector đã detect được pill tốt, nhưng cần thêm loại thuốc mới:

1. Không cần train lại detector ngay.
2. Thêm ảnh reference vào site project.
3. Chạy inference lại.
4. Review top-k candidates.

Lệnh thêm reference:

```powershell
python -m edgevision.cli project-add-reference `
  --project site_projects\demo_site `
  --label SitePillA `
  --image path\to\site_pill_a_ref_001.jpg
```

Khuyến nghị:

- ít nhất 3 ảnh/type nếu có,
- ảnh rõ,
- ánh sáng tương tự thực tế,
- có mặt trước/mặt sau nếu viên thuốc khác biệt theo mặt.

## 5. Quy Trình OK/NG

### Giai đoạn hiện tại

Chương trình dùng rule-based baseline.

Nó có thể bắt một số lỗi như:

- vùng tối bất thường,
- hình dạng bất thường,
- crop chạm biên,
- foreground quá nhỏ/lớn,
- identity unknown.

### Khi nào cần train quality classifier?

Khi bạn có dữ liệu crop OK/NG thật.

Cấu trúc:

```text
data/quality/
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

Train bằng:

```text
training/colab_train_quality_classifier.py
```

Không nên train quality nếu label chưa rõ. Ảnh mơ hồ nên để `REVIEW`.

## 6. Checklist Sau Mỗi Lần Chạy Inference

Luôn kiểm tra theo thứ tự:

1. `summary.csv`: tổng quan nhanh.
2. `annotated.jpg`: bbox có đúng không.
3. `crops/`: crop có sạch không.
4. `report.json`: confidence, flags, runtime.

Nếu bbox sai:

- kiểm tra detector,
- kiểm tra data train YOLO,
- thêm ảnh hard case.

Nếu type sai:

- thêm reference,
- kiểm tra crop,
- kiểm tra threshold unknown.

Nếu OK/NG sai:

- xem crop,
- xem flags,
- xem rule threshold,
- thu thập thêm OK/NG data thật.

## 7. Điều Kiện Trước Khi Đẩy Lên Edge Device

Chưa nên đóng gói Edge nếu local chưa đạt:

- detector không bỏ sót quá nhiều,
- count đúng trên ảnh thực tế,
- false OK cho NG được kiểm soát,
- report schema ổn định,
- runtime latency đo được,
- model path ổn định,
- site project workflow ổn định.

Edge deployment là bước sau. Local system phải chắc trước.

## 8. Lệnh Kiểm Tra Hằng Ngày

Test:

```powershell
python -m unittest discover -s tests
```

Demo:

```powershell
python -m edgevision.cli demo --output runs\demo_smoke
```

Compile:

```powershell
python -m compileall edgevision training
```

Git status:

```powershell
git status --short
```
