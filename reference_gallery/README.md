# Reference Gallery

Put project-specific reference images here when you want type recognition by
nearest reference matching.

Example:

```text
reference_gallery/
  Panadol_Extra/
    ref_1.jpg
    ref_2.jpg
  Aspirin/
    ref_1.jpg
```

Run with:

```powershell
python -m edgevision.cli batch --input data\real_test_images --output runs\real_test_gallery --config configs\runtime_yolo_config.json --reference-root reference_gallery
```

