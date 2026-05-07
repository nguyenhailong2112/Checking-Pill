# Algorithm Flows

## Inference Flow

```text
1. Load image and convert to RGB.
2. Run image precheck:
   - width/height sanity,
   - global contrast,
   - blur proxy.
3. Run detector.
4. Filter detections:
   - confidence threshold,
   - non-maximum suppression,
   - invalid geometry removal.
5. Crop each pill with padding.
6. Normalize crop to fixed size.
7. Identify each crop:
   - extract deterministic baseline features,
   - compare to gallery prototypes,
   - return top-k candidates,
   - mark UNKNOWN if confidence is below threshold.
8. Inspect quality:
   - estimate foreground mask,
   - compute foreground ratio,
   - compute mask solidity proxy,
   - compute border touch ratio,
   - compute dark spot ratio.
9. Fuse decisions.
10. Save JSON report, crops, and annotated image.
```

## Detection Metrics

- `mAP50`, `mAP50-95`
- precision and recall
- false positives on negative images
- count MAE
- count exact-match accuracy

## Identification Metrics

- top-1 accuracy
- top-5 accuracy
- mean average precision
- unknown rejection precision/recall
- confusion among visually similar types

## Quality Metrics

- OK recall
- NG recall
- false reject rate
- false accept rate
- review rate
- defect reason distribution

## Failure Modes To Audit

- merged boxes when pills touch,
- duplicate boxes on large tablets,
- capsule front/back asymmetry,
- unknown pill assigned to nearest known type,
- color shift from lighting,
- strong background color bias,
- broken pill cropped partly outside image,
- low-confidence NG forced to OK.

