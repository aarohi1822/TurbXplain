import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class DegradationFingerprinter:
    LABELS = {0: "fast-burn", 1: "slow-decay", 2: "sudden-failure"}

    def __init__(self, n_clusters: int = 3, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        self.feature_names: list[str] = []
        self.cluster_labels_: np.ndarray | None = None
        self.silhouette_: float | None = None

    def extract_vector(self, shap_matrix: pd.DataFrame) -> np.ndarray:
        features = [col for col in shap_matrix.columns if col != "cycle"]
        vector = []
        max_cycle = max(float(shap_matrix["cycle"].max()), 1.0)
        for feature in features:
            values = shap_matrix[feature].to_numpy(dtype=float)
            abs_values = np.abs(values)
            onset_idx = int(np.argmax(abs_values >= np.quantile(abs_values, 0.9))) if len(values) else 0
            slope = np.polyfit(np.arange(len(values)), values, 1)[0] if len(values) > 1 else 0.0
            acceleration = np.diff(values, n=2).mean() if len(values) > 2 else 0.0
            vector.extend([abs_values.max(), abs_values.mean(), onset_idx / max_cycle, slope, acceleration])
        return np.asarray(vector, dtype=float)

    def fit(self, engine_shap_matrices: dict[int, pd.DataFrame]) -> "DegradationFingerprinter":
        engine_ids = sorted(engine_shap_matrices)
        X = np.vstack([self.extract_vector(engine_shap_matrices[engine_id]) for engine_id in engine_ids])
        X_scaled = self.scaler.fit_transform(X)
        self.cluster_labels_ = self.model.fit_predict(X_scaled)
        self.silhouette_ = float(silhouette_score(X_scaled, self.cluster_labels_)) if self.n_clusters > 1 else None
        self.engine_ids_ = engine_ids
        return self

    def classify_vector(self, vector: np.ndarray) -> dict:
        scaled = self.scaler.transform([vector])
        distances = self.model.transform(scaled)[0]
        label = int(np.argmin(distances))
        confidence = float(1.0 / (1.0 + distances[label]))
        return {"cluster": label, "name": self.LABELS.get(label, f"cluster-{label}"), "confidence": confidence}

    def cluster_plot(self):
        if self.cluster_labels_ is None:
            raise ValueError("Fit the fingerprinter before plotting.")
        df = pd.DataFrame({"engine_id": self.engine_ids_, "cluster": self.cluster_labels_.astype(str)})
        return px.histogram(df, x="cluster", color="cluster", title="Degradation Fingerprint Distribution")

    def save(self, path: str | Path = "reports/fingerprints.json") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = [] if self.cluster_labels_ is None else self.cluster_labels_
        payload = {
            "n_clusters": self.n_clusters,
            "silhouette_score": self.silhouette_,
            "labels": self.LABELS,
            "engine_clusters": dict(zip(map(str, getattr(self, "engine_ids_", [])), map(int, labels))),
        }
        path.write_text(json.dumps(payload, indent=2))
