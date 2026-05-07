# Identity Dataset

The first identity implementation uses reference-gallery retrieval.

It can read ePillID directly:

```text
--reference-manifest C:/Users/longn/PyCharmMiscProject/ePillID-benchmark-ePillID_data_v1.0/ePillID_data/ePillID_data/all_labels.csv
--reference-image-root C:/Users/longn/PyCharmMiscProject/ePillID-benchmark-ePillID_data_v1.0/ePillID_data/ePillID_data/classification_data
```

For a project-specific medicine set, a folder gallery is also supported:

```text
reference_gallery/
  Panadol_Extra/
    ref_1.jpg
    ref_2.jpg
  Aspirin/
    ref_1.jpg
```

