"""
Simple Logistic Regression from scratch using NumPy.
"""
from __future__ import annotations
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr: float = 0.1, epochs: int = 1000, l2: float = 0.0, verbose: bool = False, random_state: int | None = None):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.verbose = verbose
        self.random_state = random_state
        self.coef_ = None  # shape (n_features, )
        self.intercept_ = 0.0
        self.loss_history_ = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # stable sigmoid
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1 - eps)
        logloss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        if self.l2 > 0 and self.coef_ is not None:
            logloss += 0.5 * self.l2 * np.sum(self.coef_ ** 2)
        return float(logloss)

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.coef_ = rng.normal(0, 0.01, size=n_features)
        self.intercept_ = 0.0

        for epoch in range(self.epochs):
            logits = X @ self.coef_ + self.intercept_
            y_prob = self._sigmoid(logits)
            # gradients
            error = y_prob - y
            grad_w = (X.T @ error) / n_samples + self.l2 * self.coef_
            grad_b = float(np.mean(error))
            # update
            self.coef_ -= self.lr * grad_w
            self.intercept_ -= self.lr * grad_b
            # track loss
            loss = self._loss(y, y_prob)
            self.loss_history_.append(loss)
            if self.verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                print(f"epoch {epoch:4d} loss={loss:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        logits = X @ self.coef_ + self.intercept_
        return self._sigmoid(logits)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
