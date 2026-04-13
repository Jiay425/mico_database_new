# Model Pipeline

This folder now contains the production-oriented model pipeline for the prediction workbench.

## Files

- `xgboost_multiclass.py`
  Trains on the full merged dataset, selects features, exports the final XGBoost model, and writes preprocessing artifacts.
- `inference_service.py`
  Loads exported artifacts and exposes `/predict`, `/abundance_curves`, and `/pcoa_data`.
- `import_standard_abundance.py`
  Imports the standardized abundance matrix into `microbe_abundance_standard`.
- `requirements-model.txt`
  Python dependencies for training and inference.
- `requirements-import.txt`
  Python dependency for MySQL import.

## Recommended flow

1. Install Python dependencies

```bash
pip install -r model/requirements-model.txt
```

2. Train the full model and export artifacts

```bash
python model/xgboost_multiclass.py --data-dir model --artifact-dir model/artifacts
```

3. Start the inference service

```bash
python model/inference_service.py
```

4. Enable the Spring proxy layer in `src/main/resources/application.properties`

```properties
app.model.enabled=true
app.model.service-url=http://localhost:5001
```

## Import standardized abundance into MySQL

Install the import dependency:

```bash
pip install -r model/requirements-import.txt
```

Dry run first:

```bash
python model/import_standard_abundance.py --csv model/merged_abundance.csv --dry-run --skip-zero
```

Then import:

```bash
python model/import_standard_abundance.py --csv model/merged_abundance.csv --skip-zero --feature-version v1 --truncate
```

Default assumption:

- CSV sample IDs map to `patients.patient_name`
- CSV tax column maps to `microbe_name_standard`

If your sample ID should map to another field, adjust `--match-field`.

## Request contract

The prediction page now sends both:

- `raw_data`
- `features`

`features` is preferred for real inference because the backend can align `microbeName -> abundanceValue` against the training feature table. This avoids prediction drift caused by mismatched feature order.
