# Hướng Dẫn Chạy Chương Trình

Tài liệu này hướng dẫn cách chạy EdgeVision trên máy local: chạy demo, chạy một
ảnh, chạy folder ảnh, dùng YOLO weights, dùng reference gallery và dùng site
project.

## 1. Chuẩn Bị Môi Trường

Tối thiểu cần:

```text
numpy
Pillow
```

Nếu dùng YOLO `best.pt`, cần thêm:

```text
ultralytics
```

Cài runtime cơ bản:

```powershell
pip install -e .
```

Cài runtime có YOLO:

```powershell
pip install -e .[yolo]
```

Nếu máy không nhận `python`, dùng Python bundled:

```powershell
C:\Users\longn\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

Trong các lệnh bên dưới, nếu `python` không chạy, thay `python` bằng đường dẫn
Python bundled trên.

## 2. Chạy Demo Kiểm Tra Chương Trình

Lệnh:

```powershell
python -m edgevision.cli demo --output runs\demo_smoke
```

Kỳ vọng terminal in ra:

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

Mở các file:

```text
runs/demo_smoke/inference/report.json
runs/demo_smoke/inference/annotated.jpg
runs/demo_smoke/inference/crops/
```

Nếu có đủ các file này nghĩa là pipeline đang chạy được.

## 3. Cấu Hình Runtime

Config chính:

```text
configs/runtime_yolo_config.json
```

Các field quan trọng:

```json
{
  "detector": {
    "backend": "auto",
    "confidence_threshold": 0.25,
    "nms_iou_threshold": 0.45,
    "yolo": {
      "weights_path": "models/pill_detector_yolo/best.pt",
      "image_size": 832,
      "device": null
    }
  },
  "identifier": {
    "apply_to_detection_labels": ["pill"]
  },
  "quality": {
    "apply_to_detection_labels": ["pill"]
  }
}
```

Ý nghĩa:

- `backend: auto`: nếu có YOLO weight thì dùng YOLO, nếu không thì fallback
  heuristic.
- `confidence_threshold`: lọc detection yếu.
- `nms_iou_threshold`: lọc bbox trùng.
- `apply_to_detection_labels`: label nào mới chạy identity/quality.

## 4. Chạy Một Ảnh

Đặt ảnh vào:

```text
data/real_test_images/sample.jpg
```

Chạy không ép weights:

```powershell
python -m edgevision.cli infer `
  --image data\real_test_images\sample.jpg `
  --output runs\single_test `
  --config configs\runtime_yolo_config.json
```

Chạy với YOLO weights:

```powershell
python -m edgevision.cli infer `
  --image data\real_test_images\sample.jpg `
  --output runs\single_test_yolo `
  --config configs\runtime_yolo_config.json `
  --yolo-weights models\pill_detector_yolo\best.pt
```

Output:

```text
runs/single_test_yolo/
  report.json
  annotated.jpg
  crops/
```

## 5. Chạy Nhiều Ảnh

Đặt ảnh:

```text
data/real_test_images/
  image_001.jpg
  image_002.jpg
  image_003.jpg
```

Chạy:

```powershell
python -m edgevision.cli batch `
  --input data\real_test_images `
  --output runs\real_test `
  --config configs\runtime_yolo_config.json `
  --yolo-weights models\pill_detector_yolo\best.pt
```

Output:

```text
runs/real_test/
  summary.csv
  0001_image_001/
    report.json
    annotated.jpg
    crops/
  0002_image_002/
    report.json
    annotated.jpg
    crops/
```

Nên xem theo thứ tự:

1. `summary.csv`,
2. từng `annotated.jpg`,
3. từng `report.json`,
4. từng crop trong `crops/`.

## 6. Dùng Reference Gallery

Tạo thư mục:

```text
reference_gallery/
  RoundWhiteTablet/
    ref_001.jpg
    ref_002.jpg
  RedWhiteCapsule/
    ref_001.jpg
```

Chạy:

```powershell
python -m edgevision.cli batch `
  --input data\real_test_images `
  --output runs\gallery_test `
  --config configs\runtime_yolo_config.json `
  --yolo-weights models\pill_detector_yolo\best.pt `
  --reference-root reference_gallery
```

Kết quả identity sẽ nằm trong:

```text
report.json -> items -> identity
summary.csv -> type, type_conf
```

## 7. Dùng ePillID Làm External Gallery

Nếu muốn dùng ePillID source làm gallery:

```powershell
python -m edgevision.cli batch `
  --input data\real_test_images `
  --output runs\epillid_gallery_test `
  --config configs\runtime_yolo_config.json `
  --yolo-weights models\pill_detector_yolo\best.pt `
  --reference-manifest C:\Users\longn\PyCharmMiscProject\ePillID-benchmark-ePillID_data_v1.0\ePillID_data\ePillID_data\all_labels.csv `
  --reference-image-root C:\Users\longn\PyCharmMiscProject\ePillID-benchmark-ePillID_data_v1.0\ePillID_data\ePillID_data\classification_data
```

Lưu ý: gallery feature hiện tại là baseline, không phải embedding model mạnh
cuối cùng.

## 8. Tạo Site Project

Khi cần xử lý object khác ngoài thuốc, tạo site project:

```powershell
python -m edgevision.cli project-init `
  --path site_projects\demo_site `
  --name demo_site `
  --classes pill bottle blister thermometer syringe `
  --identity-labels pill `
  --quality-labels pill
```

Kiểm tra:

```powershell
python -m edgevision.cli project-summary --project site_projects\demo_site
```

## 9. Thêm Reference Vào Site Project

```powershell
python -m edgevision.cli project-add-reference `
  --project site_projects\demo_site `
  --label RoundWhiteTablet `
  --image data\real_test_images\round_white_ref.jpg
```

Kiểm tra lại:

```powershell
python -m edgevision.cli project-summary --project site_projects\demo_site
```

## 10. Chạy Với Site Project

Sau khi có:

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

Ý nghĩa:

- detector dùng `best.pt` của site,
- reference gallery lấy từ site project,
- identity chỉ áp dụng cho `pill`,
- quality chỉ áp dụng cho `pill`,
- object khác vẫn được count/report.

## 11. Đọc Report JSON

Ví dụ:

```json
{
  "total_count": 3,
  "count_by_detection_label": {
    "pill": 2,
    "bottle": 1
  },
  "count_by_type": {
    "RoundWhiteTablet": 2,
    "bottle": 1
  },
  "image_status": "OK"
}
```

Trong từng item:

```json
{
  "id": 1,
  "bbox": [10, 20, 100, 140],
  "det_label": "pill",
  "det_conf": 0.91,
  "type": "RoundWhiteTablet",
  "type_conf": 0.95,
  "quality": "OK",
  "flags": []
}
```

Nếu `det_label` đúng nhưng `type` sai, lỗi nằm ở identity/reference.

Nếu bbox sai, lỗi nằm ở detector.

Nếu `quality` sai, lỗi nằm ở quality rule/model hoặc crop.

## 12. Kiểm Tra Chương Trình Sau Khi Sửa Code

Chạy test:

```powershell
python -m unittest discover -s tests
```

Chạy demo:

```powershell
python -m edgevision.cli demo --output runs\demo_smoke
```

Compile check:

```powershell
python -m compileall edgevision training
```
