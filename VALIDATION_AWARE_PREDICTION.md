# Validation-aware prediction

After you validate objects in EcoTaxa and export those classifications, copy the TSV file into the project's `data/` directory. Both the usual EcoTaxa **Object Export** and the **Classification Export** (EcoTaxa: *Jobs ? Create ? Identification Export*) are supported.

Run training with the Classification Export when it contains the validations you want to use:

```powershell
cytoprocess train <project> --export-tsv <path-to-classification-export.tsv>
```

After training, run prediction again with force so the old prediction parquet is rebuilt:

```powershell
cytoprocess predict_images <project> --force
```

By default, CytoProcess finds the newest usable TSV in `<project>/data/`, reads `object_id` and `object_annotation_status`, and skips objects whose status is `validated`. It ignores TSVs that do not contain both columns. If no usable export is present, it retains the previous behavior and predicts every image.

To deliberately predict validated images as well:

```powershell
cytoprocess predict_images <project> --force --predict-validated
```

The generated predictions contain only the objects selected for prediction. If a sample is entirely validated, CytoProcess writes an empty prediction file, so syncing will not overwrite validated classifications.
