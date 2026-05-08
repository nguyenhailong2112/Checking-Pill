# EdgeVision

EdgeVision là chương trình kiểm tra hình ảnh định hướng cho bài toán thuốc và
triển khai Edge AI sau này. Mục tiêu hiện tại là xây dựng một chương trình local
chạy được, có cấu trúc rõ ràng, có thể test, có thể gắn model `best.pt`, có thể
xuất kết quả kiểm tra đầy đủ trước khi nghĩ tới Jetson, camera công nghiệp hoặc
Edge AI Device.

Chương trình hiện hỗ trợ:

- detect object bằng detector,
- đếm tổng số object,
- đếm theo nhãn detector,
- nhận diện loại thuốc bằng reference gallery,
- kiểm tra OK/NG/REVIEW baseline,
- xử lý detector nhiều class như `pill`, `bottle`, `blister`, `thermometer`,
  `syringe` hoặc object mới do user label,
- tạo site project riêng cho từng bài toán triển khai,
- xuất `report.json`, `summary.csv`, `annotated.jpg`, crop từng object,
- script train model trên Colab cho detector, identity embedding và quality
  classifier.

## 1. Tư Duy Hệ Thống

EdgeVision không được thiết kế như một model duy nhất làm tất cả. Bài toán được
tách thành nhiều khối:

```text
ảnh đầu vào hoặc frame camera
 -> kiểm tra chất lượng ảnh
 -> detector tìm object
 -> lọc bbox và đếm số lượng
 -> crop từng object
 -> nếu là pill thì nhận diện loại thuốc
 -> nếu cần kiểm tra chất lượng thì chạy OK/NG
 -> hợp nhất quyết định
 -> xuất ảnh annotate + crop + JSON + CSV
```

Lý do phải tách như vậy:

- Detection/counting cần dataset bbox dạng YOLO.
- Nhận diện loại thuốc cần crop/reference/embedding.
- OK/NG cần dữ liệu lỗi thật.
- Object mới tại site có thể cần train detector mới.
- Loại thuốc mới có thể chỉ cần thêm ảnh reference, chưa cần train lại detector.

## 2. Cấu Trúc Thư Mục Quan Trọng

```text
EdgeVision/
  configs/                  Config runtime
  data/                     Ảnh test và dữ liệu dẫn xuất
  docs/                     Tài liệu hướng dẫn
  edgevision/               Source code runtime
  models/                   Trọng số model local
  reference_gallery/        Gallery reference đơn giản
  runs/                     Kết quả chạy demo/inference/test
  site_projects/            Project riêng cho từng site/tác vụ
  tests/                    Test kiểm tra contract chương trình
  tools/                    Tool audit/export dataset
  training/                 Script train model trên Colab
```

## 3. Chạy Demo Nhanh Nhất

Mục đích: kiểm tra chương trình có chạy được không, có tạo report không, có
detect/count/classify/OK-NG baseline không.

Chạy lệnh:

```powershell
python -m edgevision.cli demo --output runs\demo_smoke
```

Nếu máy không nhận lệnh `python`, dùng Python bundled:

```powershell
C:\Users\longn\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m edgevision.cli demo --output runs\demo_smoke
```

Kết quả mong đợi:

```json
{
  "total_count": 3,
  "count_by_type": {
    "BlueCaplet": 1,
    "RoundCream": 2
  },
  "image_status": "NG"
}
```

Xem kết quả tại:

```text
runs/demo_smoke/inference/report.json
runs/demo_smoke/inference/annotated.jpg
runs/demo_smoke/inference/crops/
```

Lưu ý: demo này dùng ảnh synthetic và detector heuristic. Nó dùng để kiểm tra
pipeline, không phải để chứng minh độ chính xác model ngoài thực tế.

## 4. Chạy Một Ảnh Thật

Đặt ảnh test vào:

```text
data/real_test_images/
```

Ví dụ có ảnh:

```text
data/real_test_images/sample.jpg
```

Chạy:

```powershell
python -m edgevision.cli infer `
  --image data\real_test_images\sample.jpg `
  --output runs\single_test `
  --config configs\runtime_yolo_config.json
```

Nếu đã có trọng số YOLO `best.pt`, chạy:

```powershell
python -m edgevision.cli infer `
  --image data\real_test_images\sample.jpg `
  --output runs\single_test_yolo `
  --config configs\runtime_yolo_config.json `
  --yolo-weights models\pill_detector_yolo\best.pt
```

Kết quả nằm trong:

```text
runs/single_test_yolo/report.json
runs/single_test_yolo/annotated.jpg
runs/single_test_yolo/crops/
```

## 5. Chạy Cả Folder Ảnh

Đặt nhiều ảnh vào:

```text
data/real_test_images/
```

Chạy:

```powershell
python -m edgevision.cli batch `
  --input data\real_test_images `
  --output runs\real_test `
  --config configs\runtime_yolo_config.json `
  --yolo-weights models\pill_detector_yolo\best.pt
```

Kết quả:

```text
runs/real_test/
  summary.csv
  0001_ten_anh/
    report.json
    annotated.jpg
    crops/
  0002_ten_anh/
    report.json
    annotated.jpg
    crops/
```

File cần xem đầu tiên:

```text
runs/real_test/summary.csv
```

## 6. Tạo Site Project Cho Object Mới

Khi bạn muốn chương trình xử lý không chỉ viên thuốc mà còn object khác như chai
thuốc, vỉ thuốc, nhiệt kế, syringe hoặc object mới tại site, hãy tạo site
project.

Chạy:

```powershell
python -m edgevision.cli project-init `
  --path site_projects\demo_site `
  --name demo_site `
  --classes pill bottle blister thermometer syringe `
  --identity-labels pill `
  --quality-labels pill
```

Ý nghĩa:

- `--classes`: toàn bộ class detector cần detect.
- `--identity-labels pill`: chỉ object có nhãn detector `pill` mới chạy nhận
  diện loại thuốc.
- `--quality-labels pill`: chỉ object có nhãn detector `pill` mới chạy OK/NG.
- Object khác như `bottle`, `blister`, `thermometer`, `syringe` vẫn được detect,
  count, annotate, report, nhưng không bị ép qua pill identity.

Sau khi tạo, project có dạng:

```text
site_projects/demo_site/
  site_project.json
  captures/
  reference_gallery/
  quality/
  annotations/yolo/
  models/
  runs/
  exports/
```

## 7. Thêm Reference Cho Loại Thuốc

Ví dụ bạn có ảnh reference của loại `RoundWhiteTablet`:

```powershell
python -m edgevision.cli project-add-reference `
  --project site_projects\demo_site `
  --label RoundWhiteTablet `
  --image path\to\round_white_ref.jpg
```

Ảnh sẽ được copy vào:

```text
site_projects/demo_site/reference_gallery/RoundWhiteTablet/
```

Nên có từ 3 đến 10 ảnh reference cho mỗi loại thuốc nếu có thể. Ảnh nên rõ,
đủ sáng, crop/pose tương tự ảnh chạy thực tế.

## 8. Chạy Với Site Project

Sau khi bạn đã có site detector:

```text
site_projects/demo_site/models/best.pt
```

Chạy:

```powershell
python -m edgevision.cli batch `
  --input data\real_test_images `
  --output runs\site_test `
  --config configs\runtime_yolo_config.json `
  --site-project site_projects\demo_site `
  --yolo-weights site_projects\demo_site\models\best.pt
```

Kết quả:

- object nào detector thấy đều được count,
- `pill` được nhận diện loại thuốc nếu có reference,
- `pill` được kiểm tra OK/NG baseline,
- object khác vẫn được report theo detector label.

## 9. Train Model Ở Đâu?

Train trên Google Colab. Xem chi tiết tại:

[docs/TRAINING_COLAB_GUIDE.md](docs/TRAINING_COLAB_GUIDE.md)

Tóm tắt:

| Muốn làm gì | Dữ liệu cần | Script |
| --- | --- | --- |
| Detect/count pill | YOLO `archive/` | `training/colab_train_yolo_detector.py` |
| Detect object mới | YOLO site annotations | `training/colab_train_yolo_detector.py` |
| Nhận diện loại thuốc mạnh hơn | ePillID crop CSV | `training/colab_train_identity_embedding.py` |
| OK/NG bằng deep learning | crop OK/NG thật | `training/colab_train_quality_classifier.py` |

## 10. Kiểm Tra Chương Trình

Chạy test:

```powershell
python -m unittest discover -s tests
```

Compile check:

```powershell
python -m compileall edgevision training
```

## 11. Thứ Tự Tài Liệu Nên Đọc

1. [docs/RUNTIME_GUIDE.md](docs/RUNTIME_GUIDE.md): chạy chương trình.
2. [docs/OPERATIONS_GUIDE.md](docs/OPERATIONS_GUIDE.md): quy trình vận hành
   thực tế.
3. [docs/TRAINING_COLAB_GUIDE.md](docs/TRAINING_COLAB_GUIDE.md): train model.
4. [docs/DATASET_STRATEGY.md](docs/DATASET_STRATEGY.md): hiểu dataset nào
   dùng cho model nào.
5. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): hiểu kiến trúc hệ thống.
6. [docs/ALGORITHMS.md](docs/ALGORITHMS.md): hiểu thuật toán, metric và lỗi
   cần audit.

## 12. Roadmap Tiếp Theo

Đã có:

- runtime local,
- demo,
- batch inference,
- YOLO adapter,
- multi-class detector label,
- site project,
- reference gallery,
- OK/NG baseline,
- script train Colab.

Các bước tiếp theo:

1. Thêm runtime adapter cho `identity_embedding.pt`.
2. Thêm runtime adapter cho `quality_classifier.pt`.
3. Thêm evaluation harness tính metric detection/count/identity/quality.
4. Thêm webcam/camera loop.
5. Chuẩn bị export model và đóng gói Edge AI.
