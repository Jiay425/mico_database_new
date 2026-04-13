from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 2026
N_FOLDS = 5
PSEUDOCOUNT = 1e-6
VARIANCE_DROP_PERCENTILE = 20
MI_SAMPLE_LIMIT = 3000
MI_KEEP_LIMIT = 1500
MI_KEEP_RATIO = 0.60
CURVE_LENGTH = 100


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train the production multiclass XGBoost model and export artifacts."
    )
    parser.add_argument("--data-dir", default=str(root), help="Directory containing merged CSV files.")
    parser.add_argument(
        "--artifact-dir",
        default=str(root / "artifacts"),
        help="Directory where model artifacts will be written.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=N_FOLDS,
        help="Number of CV folds used to choose final boosting rounds. Use 1 for fast training.",
    )
    return parser.parse_args()


def log_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_dataset(data_dir: Path):
    meta = pd.read_csv(data_dir / "merged_metadata.csv")
    abundance = pd.read_csv(data_dir / "merged_abundance.csv", index_col=0).astype(np.float32)

    full_feature_names = abundance.index.to_numpy(dtype=str)
    sample_names = abundance.columns.to_numpy(dtype=str)
    sample_to_idx = {sample: idx for idx, sample in enumerate(sample_names)}

    meta = meta[meta["sampleID"].isin(sample_to_idx)].copy()
    meta["_sample_index"] = meta["sampleID"].map(sample_to_idx)
    meta = meta.sort_values("_sample_index").reset_index(drop=True)

    x_raw = abundance.values.T[meta["_sample_index"].to_numpy()].astype(np.float32)
    y = meta["Group"].astype(str).to_numpy()

    del abundance
    gc.collect()

    return meta, x_raw, y, full_feature_names


def clr_transform(matrix: np.ndarray, pseudocount: float) -> np.ndarray:
    matrix = np.maximum(matrix.astype(np.float64), 0.0) + pseudocount
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matrix /= row_sums
    log_matrix = np.log(matrix)
    return (log_matrix - log_matrix.mean(axis=1, keepdims=True)).astype(np.float32)


def compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    counts = Counter(labels)
    total = float(len(labels))
    class_count = float(len(counts))
    weights = {
        label: total / (class_count * count)
        for label, count in counts.items()
    }
    return np.array([weights[label] for label in labels], dtype=np.float32)


def fit_feature_pipeline(x_raw: np.ndarray, y: np.ndarray, full_feature_names: np.ndarray):
    log_section("1. Fit preprocessing and feature selection")

    feature_variances = np.var(x_raw, axis=0)
    variance_threshold = float(np.percentile(feature_variances, VARIANCE_DROP_PERCENTILE))
    stage1_mask = feature_variances >= variance_threshold
    stage1_indices = np.flatnonzero(stage1_mask)
    x_stage1_raw = x_raw[:, stage1_mask]
    stage1_feature_names = full_feature_names[stage1_mask]
    print(f"Stage 1 keep: {x_stage1_raw.shape[1]} / {x_raw.shape[1]}")

    x_stage1_clr = clr_transform(x_stage1_raw, PSEUDOCOUNT)

    if len(y) > MI_SAMPLE_LIMIT:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=MI_SAMPLE_LIMIT,
            random_state=RANDOM_STATE,
        )
        mi_index, _ = next(sss.split(x_stage1_clr, y))
    else:
        mi_index = np.arange(len(y))

    mi_scores = mutual_info_classif(
        x_stage1_clr[mi_index],
        y[mi_index],
        discrete_features=False,
        n_neighbors=5,
        random_state=RANDOM_STATE,
    )

    keep_count = max(32, min(MI_KEEP_LIMIT, int(x_stage1_clr.shape[1] * MI_KEEP_RATIO)))
    selected_stage1_indices = np.argsort(mi_scores)[::-1][:keep_count]
    selected_stage1_indices = np.sort(selected_stage1_indices)

    x_selected_raw = x_stage1_raw[:, selected_stage1_indices].astype(np.float32)
    x_selected_clr = x_stage1_clr[:, selected_stage1_indices].astype(np.float32)
    selected_feature_names = stage1_feature_names[selected_stage1_indices]
    print(f"Stage 2 keep: {x_selected_clr.shape[1]} / {x_stage1_clr.shape[1]}")

    config = {
        "artifact_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pseudocount": PSEUDOCOUNT,
        "full_feature_names": full_feature_names.tolist(),
        "stage1_indices": stage1_indices.tolist(),
        "selected_stage1_indices": selected_stage1_indices.tolist(),
        "selected_feature_names": selected_feature_names.tolist(),
        "variance_threshold": variance_threshold,
        "curve_length": CURVE_LENGTH,
    }

    return x_selected_raw, x_selected_clr, config


def cross_validate_model(x: np.ndarray, y: np.ndarray, n_folds: int):
    log_section("2. Cross validation and estimator selection")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    sample_weights = compute_sample_weights(y)

    params = dict(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        n_estimators=900,
        early_stopping_rounds=40,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.75,
        colsample_bytree=0.55,
        min_child_weight=5,
        gamma=0.25,
        reg_alpha=0.3,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss",
        tree_method="hist",
        verbosity=0,
    )

    if n_folds <= 1:
        print("Fast mode enabled: skip cross validation and use the configured n_estimators directly.")
        report = {
            "note": "cross validation skipped because --folds <= 1",
            "configured_n_estimators": params["n_estimators"],
        }
        return label_encoder, sample_weights, params, params["n_estimators"], report

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    best_rounds = []
    oof_predictions = np.zeros((len(y_encoded), len(label_encoder.classes_)), dtype=np.float32)

    for fold_id, (train_idx, valid_idx) in enumerate(skf.split(x, y_encoded), start=1):
        print(f"Fold {fold_id}/{n_folds}")
        model = XGBClassifier(**params)
        model.fit(
            x[train_idx],
            y_encoded[train_idx],
            sample_weight=sample_weights[train_idx],
            eval_set=[(x[valid_idx], y_encoded[valid_idx])],
            verbose=False,
        )

        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is None or best_iteration < 1:
            best_iteration = params["n_estimators"]
        else:
            best_iteration = int(best_iteration) + 1
        best_rounds.append(best_iteration)
        print(f"  best_rounds={best_iteration}")

        oof_predictions[valid_idx] = model.predict_proba(x[valid_idx])

    oof_labels = label_encoder.inverse_transform(np.argmax(oof_predictions, axis=1))
    report = classification_report(y, oof_labels, output_dict=True, zero_division=0)
    final_estimators = int(np.median(best_rounds))
    final_estimators = max(120, final_estimators)

    print(f"Chosen final n_estimators={final_estimators}")
    return label_encoder, sample_weights, params, final_estimators, report


def train_final_model(x: np.ndarray, y: np.ndarray, label_encoder: LabelEncoder, sample_weights: np.ndarray, params: dict, final_estimators: int):
    log_section("3. Train final production model")

    final_params = dict(params)
    final_params["n_estimators"] = final_estimators
    final_params.pop("early_stopping_rounds", None)

    model = XGBClassifier(**final_params)
    model.fit(x, label_encoder.transform(y), sample_weight=sample_weights, verbose=False)
    return model


def build_reference_bundle(meta: pd.DataFrame, y: np.ndarray, x_selected_raw: np.ndarray, x_selected_clr: np.ndarray, config: dict):
    log_section("4. Build visualization references")

    healthy_label = "0" if "0" in set(y.tolist()) else Counter(y).most_common(1)[0][0]
    healthy_mask = y == healthy_label

    healthy_selected_raw = x_selected_raw[healthy_mask]
    healthy_selected_clr = x_selected_clr[healthy_mask]

    if healthy_selected_raw.size == 0:
        healthy_curve = np.zeros(config["curve_length"], dtype=np.float32)
        healthy_coords = np.zeros((0, 2), dtype=np.float32)
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pca.fit(np.zeros((2, x_selected_clr.shape[1]), dtype=np.float32))
    else:
        sorted_healthy = np.sort(healthy_selected_raw, axis=1)[:, ::-1]
        curve_limit = min(config["curve_length"], sorted_healthy.shape[1])
        healthy_curve = sorted_healthy[:, :curve_limit].mean(axis=0).astype(np.float32)

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pca.fit(x_selected_clr)
        healthy_coords = pca.transform(healthy_selected_clr).astype(np.float32)

    bundle = {
        "healthy_group": healthy_label,
        "healthy_count": int(healthy_mask.sum()),
        "curve_length": int(len(healthy_curve)),
        "healthy_curve": healthy_curve.tolist(),
        "pca_mean": pca.mean_.astype(np.float32).tolist(),
        "pca_components": pca.components_.astype(np.float32).tolist(),
        "healthy_coords": healthy_coords.tolist(),
        "selected_feature_count": int(x_selected_clr.shape[1]),
        "sample_count": int(len(meta)),
    }
    return bundle


def save_artifacts(
    artifact_dir: Path,
    model: XGBClassifier,
    config: dict,
    label_encoder: LabelEncoder,
    evaluation_report: dict,
    reference_bundle: dict,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "xgb_model.json"
    model.save_model(model_path)

    with (artifact_dir / "preprocessing.json").open("w", encoding="utf-8") as fp:
        json.dump(config, fp, ensure_ascii=False, indent=2)

    with (artifact_dir / "label_classes.json").open("w", encoding="utf-8") as fp:
        json.dump({"label_classes": label_encoder.classes_.tolist()}, fp, ensure_ascii=False, indent=2)

    with (artifact_dir / "training_report.json").open("w", encoding="utf-8") as fp:
        json.dump({"oof_report": evaluation_report}, fp, ensure_ascii=False, indent=2)

    with (artifact_dir / "reference_bundle.json").open("w", encoding="utf-8") as fp:
        json.dump(reference_bundle, fp, ensure_ascii=False, indent=2)

    top_importances = pd.DataFrame(
        {
            "feature": config["selected_feature_names"],
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    top_importances.to_csv(artifact_dir / "feature_importance.csv", index=False)

    print(f"Artifacts written to: {artifact_dir}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()

    log_section("Load dataset")
    meta, x_raw, y, full_feature_names = load_dataset(data_dir)
    print(f"samples={x_raw.shape[0]} full_features={x_raw.shape[1]}")

    x_selected_raw, x_selected_clr, config = fit_feature_pipeline(x_raw, y, full_feature_names)
    label_encoder, sample_weights, params, final_estimators, evaluation_report = cross_validate_model(
        x_selected_clr,
        y,
        args.folds,
    )
    model = train_final_model(x_selected_clr, y, label_encoder, sample_weights, params, final_estimators)
    reference_bundle = build_reference_bundle(meta, y, x_selected_raw, x_selected_clr, config)
    save_artifacts(artifact_dir, model, config, label_encoder, evaluation_report, reference_bundle)

    log_section("Training complete")
    print(f"Selected features: {len(config['selected_feature_names'])}")
    print(f"Classes: {label_encoder.classes_.tolist()}")


if __name__ == "__main__":
    main()
