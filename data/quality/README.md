# Quality Dataset

The current source folders do not provide reliable explicit OK/NG defect labels.

When real defect data is available, organize it as:

```text
quality/
  train/
    OK/
    NG_broken/
    NG_chipped/
    NG_color_abnormal/
    NG_stain/
  val/
    OK/
    NG_broken/
    NG_chipped/
    NG_color_abnormal/
    NG_stain/
  test/
    OK/
    NG_broken/
    NG_chipped/
    NG_color_abnormal/
    NG_stain/
```

Until then, EdgeVision uses the rule-based quality inspector and returns
`REVIEW` when evidence is weak.

