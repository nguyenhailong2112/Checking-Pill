# Kiến Trúc Hệ Thống EdgeVision

Tài liệu này giải thích cách chương trình EdgeVision được thiết kế, vì sao phải
tách thành nhiều module, và mỗi module chịu trách nhiệm gì.

## 1. Mục Tiêu Kiến Trúc

Mục tiêu hiện tại chưa phải là đưa ngay lên Edge Device. Mục tiêu đúng ở giai
đoạn này là:

1. Chương trình local chạy được.
2. Có thể gắn model detector `best.pt`.
3. Có thể test trên ảnh thật.
4. Có thể xuất kết quả rõ ràng để kiểm tra đúng/sai.
5. Có thể mở rộng sang object mới do user label.
6. Có cấu trúc đủ sạch để sau này đóng gói lên Edge AI Device.

## 2. Luồng Tổng Thể

```text
ảnh đầu vào hoặc frame camera
 -> ImagePrecheck
 -> Detector
 -> PostProcessor
 -> CropNormalizer
 -> Identifier
 -> QualityInspector
 -> DecisionFusion
 -> Reporter
```

Diễn giải:

- `ImagePrecheck`: kiểm tra ảnh có dùng được không.
- `Detector`: tìm object trong ảnh.
- `PostProcessor`: lọc bbox, confidence, NMS.
- `CropNormalizer`: crop object và resize về kích thước chuẩn.
- `Identifier`: nhận diện loại thuốc nếu object là pill.
- `QualityInspector`: kiểm tra OK/NG nếu object cần inspection.
- `DecisionFusion`: hợp nhất kết quả từng object thành trạng thái ảnh.
- `Reporter`: lưu JSON, CSV, ảnh annotate, crop.

## 3. Vì Sao Không Làm Một Model Duy Nhất?

Không nên gom detection, classification, OK/NG vào một model duy nhất vì:

- Detection cần bbox label.
- Nhận diện loại thuốc cần crop/reference/embedding.
- OK/NG cần defect label thật.
- Object mới tại site có thể chỉ cần thêm class detector.
- Loại thuốc mới có thể chỉ cần thêm reference ảnh.
- Metric đánh giá của từng bài toán khác nhau.

Vì vậy kiến trúc đúng là nhiều module nhỏ, contract rõ ràng, có thể thay thế
từng module khi model tốt hơn.

## 4. Runtime Contract

Với một ảnh RGB đầu vào, chương trình trả ra:

- kích thước ảnh,
- tổng số object detect được,
- count theo nhãn detector,
- count theo type/identity,
- bbox từng object,
- confidence detector,
- nhãn detector,
- type nhận diện,
- confidence nhận diện,
- trạng thái quality `OK`, `NG`, hoặc `REVIEW`,
- flags giải thích lỗi,
- crop path,
- thông tin precheck,
- thời gian chạy từng stage,
- ảnh annotate,
- JSON report.

Ví dụ field chính trong `report.json`:

```json
{
  "total_count": 3,
  "count_by_detection_label": {
    "pill": 3
  },
  "count_by_type": {
    "RoundWhiteTablet": 2,
    "RedWhiteCapsule": 1
  },
  "image_status": "NG"
}
```

## 5. Detector

Detector trả lời câu hỏi:

```text
Object nằm ở đâu?
Object thuộc class detector nào?
```

Detector không nên quyết định loại thuốc chi tiết. Ví dụ detector có thể trả:

```text
pill
bottle
blister
thermometer
syringe
```

Sau đó nếu object là `pill`, module identifier mới nhận diện sâu hơn:

```text
RoundWhiteTablet
RedWhiteCapsule
BlueCaplet
...
```

Backend hiện có:

- `heuristic`: dùng để demo/smoke test.
- `yolo`: dùng trọng số YOLO thật.
- `auto`: nếu có weights thì dùng YOLO, nếu không có thì fallback heuristic.

## 6. Identifier

Identifier trả lời câu hỏi:

```text
Viên thuốc này là loại gì?
```

Hiện tại identifier dùng reference gallery:

```text
reference_gallery/
  RoundWhiteTablet/
    ref_001.jpg
    ref_002.jpg
  RedWhiteCapsule/
    ref_001.jpg
```

Chương trình chỉ chạy identifier cho detector label nằm trong:

```json
"identifier": {
  "apply_to_detection_labels": ["pill"]
}
```

Nghĩa là:

- `pill` -> chạy nhận diện loại thuốc,
- `bottle` -> giữ label `bottle`,
- `thermometer` -> giữ label `thermometer`,
- `syringe` -> giữ label `syringe`.

Đây là điểm quan trọng để chương trình xử lý object mới đúng hướng, không ép
mọi object qua pill classifier.

## 7. Quality Inspector

Quality inspector trả lời câu hỏi:

```text
Object này OK, NG hay cần REVIEW?
```

Hiện tại quality inspector là rule-based baseline, đo:

- foreground ratio,
- mask solidity,
- border touch ratio,
- dark spot ratio,
- identity unknown.

Chương trình chỉ chạy quality cho detector label nằm trong:

```json
"quality": {
  "apply_to_detection_labels": ["pill"]
}
```

Nếu sau này bạn muốn kiểm tra OK/NG cho object khác, ví dụ `blister`, có thể cấu
hình:

```json
"apply_to_detection_labels": ["pill", "blister"]
```

## 8. Decision Fusion

Logic hợp nhất trạng thái ảnh:

1. Ảnh không đạt precheck -> `REVIEW`.
2. Có bất kỳ item nào `NG` -> ảnh là `NG`.
3. Có bất kỳ item nào `REVIEW` -> ảnh là `REVIEW`.
4. Còn lại -> ảnh là `OK`.

Nguyên tắc là không được ép kết quả mơ hồ thành OK.

## 9. Site Project

Site project là gói cấu hình/dữ liệu riêng cho một bài toán triển khai cụ thể.

Ví dụ:

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

Trong `site_project.json` có:

- danh sách class detector,
- label nào cần chạy identity,
- label nào cần chạy quality.

Ví dụ:

```json
{
  "detector_class_names": ["pill", "bottle", "blister", "thermometer", "syringe"],
  "identity_detection_labels": ["pill"],
  "quality_detection_labels": ["pill"]
}
```

## 10. Chuẩn Bị Cho Edge AI

Khi local runtime đã ổn, từng module có thể được tối ưu/export:

- Detector YOLO -> ONNX/TensorRT/OpenVINO.
- Identifier -> embedding model + gallery index.
- Quality -> rule engine hoặc classifier/anomaly model.
- Site project -> gói tác vụ cụ thể.
- Reporter -> giữ nguyên contract để tích hợp hệ thống.

Điều quan trọng: trước khi Edge deployment, chương trình local phải chạy đúng,
report đúng, metric rõ ràng, failure case được kiểm soát.
