# Models

Place trained model weights here for local inference.

Recommended detector path:

```text
models/pill_detector_yolo/best.pt
```

The runtime config [configs/runtime_yolo_config.json](../configs/runtime_yolo_config.json)
looks for that file. If it is missing and the detector backend is `auto`,
EdgeVision falls back to the heuristic detector so the pipeline remains runnable.

