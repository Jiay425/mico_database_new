from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from flask import Flask, jsonify, request
from xgboost import XGBClassifier


ARTIFACT_DIR = Path(os.environ.get("MODEL_ARTIFACT_DIR", Path(__file__).resolve().parent / "artifacts"))
HOST = os.environ.get("MODEL_SERVICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("MODEL_SERVICE_PORT", "5001"))

GROUP_LABELS = {
    "0": "健康",
    "1": "2型糖尿病 (T2D)",
    "2": "炎症性肠病 (IBD)",
    "3": "结直肠癌 (CRC)",
    "4": "多发性硬化 (MS)",
    "5": "冠状动脉疾病 (CAD)",
    "6": "脂肪肝",
    "7": "肝硬化",
    "8": "肥胖",
    "9": "早期阿尔茨海默病",
    "10": "阿尔茨海默病",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


class ModelBundle:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        self.model = None
        self.ready = False
        self.error = None
        self._load()

    def _load(self) -> None:
        try:
            preprocessing = load_json(self.artifact_dir / "preprocessing.json")
            labels = load_json(self.artifact_dir / "label_classes.json")
            references = load_json(self.artifact_dir / "reference_bundle.json")

            model = XGBClassifier()
            model.load_model(str(self.artifact_dir / "xgb_model.json"))

            self.model = model
            self.pseudocount = float(preprocessing["pseudocount"])
            self.full_feature_names = preprocessing["full_feature_names"]
            self.full_feature_index = {
                name: idx for idx, name in enumerate(self.full_feature_names)
            }
            self.stage1_indices = np.asarray(preprocessing["stage1_indices"], dtype=np.int32)
            self.selected_stage1_indices = np.asarray(preprocessing["selected_stage1_indices"], dtype=np.int32)
            self.selected_feature_names = preprocessing["selected_feature_names"]
            self.label_classes = labels["label_classes"]
            self.healthy_group = str(references["healthy_group"])
            self.curve_length = int(references["curve_length"])
            self.healthy_curve = np.asarray(references["healthy_curve"], dtype=np.float32)
            self.healthy_coords = np.asarray(references["healthy_coords"], dtype=np.float32)
            self.pca_mean = np.asarray(references["pca_mean"], dtype=np.float32)
            self.pca_components = np.asarray(references["pca_components"], dtype=np.float32)
            self.ready = True
        except Exception as exc:  # pragma: no cover - bootstrap error path
            self.error = str(exc)
            self.ready = False

    def _parse_raw_vector(self, raw_data: str) -> np.ndarray:
        values = [
            float(token)
            for token in raw_data.replace("\n", ",").replace("\t", ",").split(",")
            if token.strip()
        ]
        return np.asarray(values, dtype=np.float32)

    def _feature_vector_from_payload(self, payload: dict) -> Tuple[np.ndarray, np.ndarray]:
        features = payload.get("features") or []
        if features:
            full_vector = np.zeros(len(self.full_feature_names), dtype=np.float32)
            for item in features:
                name = item.get("name")
                if not name:
                    continue
                idx = self.full_feature_index.get(name)
                if idx is None:
                    continue
                full_vector[idx] = max(0.0, float(item.get("value") or 0.0))
            return full_vector, self._select_raw_features(full_vector)

        raw_data = (payload.get("raw_data") or "").strip()
        if not raw_data:
            raise ValueError("missing raw_data or features")

        values = self._parse_raw_vector(raw_data)
        if len(values) == len(self.full_feature_names):
            return values, self._select_raw_features(values)
        if len(values) == len(self.selected_feature_names):
            return np.array([], dtype=np.float32), values
        raise ValueError(
            "raw_data length does not match training feature space; send named features instead"
        )

    def _select_raw_features(self, full_vector: np.ndarray) -> np.ndarray:
        stage1_vector = full_vector[self.stage1_indices]
        return stage1_vector[self.selected_stage1_indices]

    def _clr_single(self, raw_selected: np.ndarray) -> np.ndarray:
        vector = np.maximum(raw_selected.astype(np.float64), 0.0) + self.pseudocount
        total = float(vector.sum())
        if total <= 0:
            total = 1.0
        vector /= total
        log_vector = np.log(vector)
        clr = log_vector - log_vector.mean()
        return clr.astype(np.float32)

    def _format_label(self, label: str) -> str:
        label = str(label)
        return GROUP_LABELS.get(label, f"Group {label}")

    def predict(self, payload: dict) -> dict:
        if not self.ready:
            return self.not_ready_response("预测模型尚未加载")

        _, raw_selected = self._feature_vector_from_payload(payload)
        clr_selected = self._clr_single(raw_selected)
        probabilities = self.model.predict_proba(clr_selected.reshape(1, -1))[0]
        order = np.argsort(probabilities)[::-1]

        predictions = []
        for idx in order[:3]:
            predictions.append(
                {
                    "group": str(self.label_classes[idx]),
                    "label": self._format_label(str(self.label_classes[idx])),
                    "probability": round(float(probabilities[idx]) * 100.0, 2),
                }
            )

        return {
            "success": True,
            "model_ready": True,
            "sample_id": payload.get("sample_id"),
            "predictions": predictions,
        }

    def abundance_curves(self, payload: dict) -> dict:
        if not self.ready:
            return self.not_ready_response("丰度曲线服务尚未加载")

        _, raw_selected = self._feature_vector_from_payload(payload)
        sorted_user = np.sort(raw_selected)[::-1][: self.curve_length]
        user_curve = [[idx + 1, float(value)] for idx, value in enumerate(sorted_user)]
        healthy_curve = [[idx + 1, float(value)] for idx, value in enumerate(self.healthy_curve)]

        return {
            "success": True,
            "model_ready": True,
            "user_curve": user_curve,
            "healthy_curve": healthy_curve,
            "has_healthy_data": len(healthy_curve) > 0,
        }

    def pcoa_data(self, payload: dict) -> dict:
        if not self.ready:
            return self.not_ready_response("PCoA 服务尚未加载")

        _, raw_selected = self._feature_vector_from_payload(payload)
        clr_selected = self._clr_single(raw_selected)
        user_coord = np.dot(clr_selected - self.pca_mean, self.pca_components.T).astype(np.float32)
        ellipse_points = compute_confidence_ellipse(self.healthy_coords)

        return {
            "success": True,
            "model_ready": True,
            "user_coord": user_coord.tolist(),
            "healthy_coords": self.healthy_coords.tolist(),
            "ellipse_points": ellipse_points,
        }

    def not_ready_response(self, message: str) -> dict:
        return {
            "success": False,
            "model_ready": False,
            "error": message,
            "detail": self.error or f"artifacts not found in {self.artifact_dir}",
        }


def compute_confidence_ellipse(coords: np.ndarray, points: int = 120) -> List[List[float]]:
    if coords.shape[0] < 3:
        return []

    mean = coords.mean(axis=0)
    covariance = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # 95% chi-square for 2 DOF.
    scale = math.sqrt(5.991)
    radii = scale * np.sqrt(np.maximum(eigvals, 1e-8))

    circle = np.stack(
        [np.cos(np.linspace(0, 2 * math.pi, points)), np.sin(np.linspace(0, 2 * math.pi, points))],
        axis=1,
    )
    ellipse = circle * radii
    rotated = ellipse @ eigvecs.T
    translated = rotated + mean
    return translated.astype(np.float32).tolist()


bundle = ModelBundle(ARTIFACT_DIR)
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"ready": bundle.ready, "artifact_dir": str(ARTIFACT_DIR), "error": bundle.error})


@app.post("/predict")
def predict():
    return jsonify(bundle.predict(request.get_json(silent=True) or {}))


@app.post("/abundance_curves")
def abundance_curves():
    return jsonify(bundle.abundance_curves(request.get_json(silent=True) or {}))


@app.post("/pcoa_data")
def pcoa_data():
    return jsonify(bundle.pcoa_data(request.get_json(silent=True) or {}))


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
