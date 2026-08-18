from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class InvertibleScaler:
    def __init__(self, scaler_type: str = "standard"):
        scaler_type = scaler_type.lower()

        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(
                f"Unsupported scaler type: {scaler_type}"
            )

        self.scaler_type = scaler_type
        self.fitted = False

    def fit(self, data: np.ndarray):
        data = np.asarray(data, dtype=np.float64)

        if data.ndim == 3:
            original_shape = data.shape
            data_2d = data.reshape(-1, original_shape[-1])
        elif data.ndim == 2:
            data_2d = data
        else:
            raise ValueError(
                "Scaler expects a 2D or 3D array."
            )

        self.scaler.fit(data_2d)
        self.fitted = True

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted.")

        data = np.asarray(data, dtype=np.float64)

        if data.ndim == 3:
            shape = data.shape
            transformed = self.scaler.transform(
                data.reshape(-1, shape[-1])
            )
            return transformed.reshape(shape).astype(np.float32)

        if data.ndim == 2:
            return self.scaler.transform(data).astype(np.float32)

        raise ValueError(
            "Scaler expects a 2D or 3D array."
        )

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted.")

        data = np.asarray(data, dtype=np.float64)

        if data.ndim == 3:
            shape = data.shape
            restored = self.scaler.inverse_transform(
                data.reshape(-1, shape[-1])
            )
            return restored.reshape(shape).astype(np.float32)

        if data.ndim == 2:
            return self.scaler.inverse_transform(data).astype(
                np.float32
            )

        raise ValueError(
            "Scaler expects a 2D or 3D array."
        )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "scaler_type": self.scaler_type,
                "scaler": self.scaler,
                "fitted": self.fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path):
        payload: Dict[str, Any] = joblib.load(path)

        obj = cls(payload["scaler_type"])
        obj.scaler = payload["scaler"]
        obj.fitted = payload["fitted"]

        return obj