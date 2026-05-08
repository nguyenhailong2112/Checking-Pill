# Thuật Toán Và Luồng Xử Lý

Tài liệu này mô tả chi tiết chương trình xử lý một ảnh như thế nào, từng bước
được tính toán ra sao, và cần đánh giá bằng metric nào.

## 1. Luồng Inference Chi Tiết

Khi chạy lệnh:

```powershell
python -m edgevision.cli infer --image path\to\image.jpg --output runs\single_test
```

chương trình xử lý theo thứ tự:

```text
1. Load ảnh.
2. Convert ảnh sang RGB.
3. Chạy precheck.
4. Chạy detector.
5. Lọc detection.
6. Crop từng object.
7. Chạy identity nếu object nằm trong identity labels.
8. Chạy quality nếu object nằm trong quality labels.
9. Đếm object.
10. Fusion trạng thái ảnh.
11. Lưu report, crop, ảnh annotate.
```

## 2. Image Precheck

Precheck kiểm tra ảnh trước khi đưa vào model.

Hiện tại đo:

- chiều rộng,
- chiều cao,
- độ tương phản,
- sharpness proxy.

Nếu ảnh quá nhỏ hoặc gần như không có tương phản, chương trình không cố đoán.
Kết quả ảnh sẽ là `REVIEW`.

Lý do: ảnh xấu mà vẫn cố kết luận OK là rất nguy hiểm trong inspection.

## 3. Detector

Detector tạo danh sách bbox:

```json
{
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.91,
  "label": "pill",
  "source": "ultralytics_yolo"
}
```

### 3.1 Heuristic Detector

Heuristic detector dùng cho demo và test nhanh.

Cách hoạt động:

1. Ước lượng màu nền từ viền ảnh.
2. Tính khoảng cách màu giữa pixel và nền.
3. Threshold foreground.
4. Tìm connected components.
5. Mỗi component đủ lớn được xem là một object.

Không dùng heuristic detector làm production model.

### 3.2 YOLO Detector

YOLO detector dùng trọng số `best.pt`.

Nếu model được train multi-class, detector có thể trả:

```text
pill
bottle
blister
thermometer
syringe
```

Tên class được đọc từ model YOLO, không hard-code trong pipeline.

## 4. Postprocess Detection

Sau khi detector trả bbox, chương trình lọc:

1. Bỏ bbox confidence thấp hơn threshold.
2. Clip bbox vào biên ảnh.
3. Bỏ bbox có diện tích không hợp lệ.
4. Chạy Non-Maximum Suppression để giảm duplicate bbox.

Config liên quan:

```json
{
  "confidence_threshold": 0.25,
  "nms_iou_threshold": 0.45
}
```

Nếu nhiều duplicate box, tăng/giảm `nms_iou_threshold` cần được test bằng ảnh
thực tế.

## 5. Crop Normalizer

Mỗi bbox được crop thêm padding rồi resize về kích thước chuẩn.

Config:

```json
{
  "padding_ratio": 0.12,
  "output_size": 224,
  "save_crops": true
}
```

Crop được lưu tại:

```text
runs/<run_name>/<image_id>/crops/
```

Crop rất quan trọng vì:

- identity dùng crop,
- quality dùng crop,
- human review cũng nên nhìn crop.

## 6. Identity Branch

Identity branch chỉ chạy nếu detector label nằm trong:

```json
"apply_to_detection_labels": ["pill"]
```

Ví dụ:

| Detector label | Có chạy identity không? |
| --- | --- |
| `pill` | Có |
| `bottle` | Không |
| `blister` | Không |
| `thermometer` | Không |

Hiện tại identity baseline:

```text
crop -> extract feature màu/hình -> so với prototype trong gallery -> top-k
```

Nếu confidence thấp hơn `unknown_threshold`, kết quả là:

```text
Unknown
```

Điều này tốt hơn việc ép nhầm sang một loại thuốc gần giống.

## 7. Quality Branch

Quality branch chỉ chạy nếu detector label nằm trong:

```json
"quality": {
  "apply_to_detection_labels": ["pill"]
}
```

Hiện tại rule-based quality tính:

- `foreground_ratio`: object chiếm bao nhiêu trong crop,
- `mask_solidity`: hình dạng có đều không,
- `border_touch_ratio`: object có chạm biên crop không,
- `dark_spot_ratio`: có vùng tối bất thường không.

Kết quả:

- `OK`: không thấy dấu hiệu lỗi mạnh,
- `NG`: có lỗi rõ,
- `REVIEW`: chưa đủ chắc chắn hoặc identity không rõ.

## 8. Counting

Chương trình hiện có hai kiểu count:

### Count theo detector label

Ví dụ:

```json
"count_by_detection_label": {
  "pill": 12,
  "bottle": 1,
  "thermometer": 1
}
```

Ý nghĩa: detector thấy bao nhiêu object theo class.

### Count theo identity/type

Ví dụ:

```json
"count_by_type": {
  "RoundWhiteTablet": 8,
  "RedWhiteCapsule": 4,
  "bottle": 1
}
```

Ý nghĩa: sau identity branch, chương trình phân nhóm theo type cuối cùng.

## 9. Decision Fusion

Logic:

```text
nếu ảnh không usable -> REVIEW
nếu có item NG -> NG
nếu có item REVIEW -> REVIEW
ngược lại -> OK
```

Nguyên tắc inspection: khi không đủ chắc chắn, chọn `REVIEW`, không ép `OK`.

## 10. Output Artifacts

Với một ảnh:

```text
report.json
annotated.jpg
crops/
```

Với batch:

```text
summary.csv
0001_image_name/
0002_image_name/
...
```

File cần xem khi debug:

1. `annotated.jpg`: nhìn bbox đúng/sai.
2. `crops/`: xem crop có sạch không.
3. `report.json`: xem confidence, flags, runtime.
4. `summary.csv`: xem thống kê toàn bộ folder.

## 11. Metric Cần Theo Dõi

### Detection/counting

- `mAP50`,
- `mAP50-95`,
- precision,
- recall,
- false positive,
- false negative,
- count MAE,
- count exact-match accuracy.

### Identity

- top-1 accuracy,
- top-5 accuracy,
- unknown rejection precision,
- unknown rejection recall,
- confusion giữa thuốc giống nhau.

### Quality

- OK recall,
- NG recall,
- false OK: NG bị nhận thành OK,
- false NG: OK bị nhận thành NG,
- REVIEW rate,
- lỗi theo từng defect type.

### Runtime

- detector latency,
- identity latency,
- quality latency,
- total latency,
- model size,
- memory usage.

## 12. Failure Mode Cần Audit

Khi test thực tế, phân loại lỗi vào các nhóm:

- detector bỏ sót viên thuốc,
- detector duplicate bbox,
- detector nhận nhầm object,
- viên thuốc chạm nhau bị merge bbox,
- viên trắng trên nền sáng bị miss,
- identity nhầm giữa hai loại giống nhau,
- pill mới bị ép sang loại đã biết,
- crop bị cắt mất một phần object,
- defect rõ nhưng quality vẫn OK,
- ảnh blur/thiếu sáng nhưng vẫn bị kết luận OK.

Những lỗi này phải được đưa ngược lại vào dataset hoặc rule/model cải tiến.
