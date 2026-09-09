"""src2/models/baseline.py — logistic regression baseline classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class LogisticRegressionBaseline:
    """Logistic Regression wrapper compatible with the pipeline interface.

    Parameters
    ----------
    C : float
        Regularisation strength (inverse; smaller = stronger).
    class_weight : str | dict
        ``'balanced'`` adjusts for imbalanced datasets.
    """

    def __init__(self, C: float = 1.0, class_weight: str = "balanced") -> None:
        self._model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            solver="lbfgs",
        )
        self._trained = False

    # ------------------------------------------------------------------
    @property
    def status(self) -> str:
        """Return ``'TRAINED'`` or ``'NOT_TRAINED'``."""
        return "TRAINED" if self._trained else "NOT_TRAINED"

    # ------------------------------------------------------------------
    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Flatten (N, seq, F) or (N, F) → (N, F)."""
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        """Train the model.

        Parameters
        ----------
        X : np.ndarray shape (N, F) or (N, seq, F)
        y : np.ndarray shape (N,)
        """
        self._model.fit(self._flatten(X), y)
        self._trained = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> dict:
        """Return dict with risk_prob."""
        if not self._trained:
            raise RuntimeError("Model has not been trained yet.")
        probs = self._model.predict_proba(self._flatten(X))[:, 1]
        return {
            'risk_prob': probs,
            'stage_idx': np.zeros_like(probs)  # LR doesn't predict stage
        }

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of class 1."""
        if not self._trained:
            raise RuntimeError("Model has not been trained yet.")
        probas = self._model.predict_proba(self._flatten(X))
        # Return probability for positive class
        return probas[:, 1] if probas.shape[1] == 2 else probas

    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Evaluate performance."""
        if not self._trained:
            raise RuntimeError("Model has not been trained yet.")
        preds_dict = self.predict(X)
        probs = preds_dict["risk_prob"]
        preds = (probs >= 0.5).astype(int)
        
        cm = confusion_matrix(y, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
            "fpr": fpr,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist model to disk using joblib."""
        import joblib  # lazy import

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "trained": self._trained}, path)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegressionBaseline":
        """Load a saved model from disk."""
        import joblib  # lazy import

        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._model = data["model"]
        obj._trained = data["trained"]
        return obj
