"""Error calculation module with multiple metrics."""

import numpy as np
from typing import Dict


class ErrorCalculator:
    """Calculate multiple error metrics for regression models."""

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(ErrorCalculator.mse(y_true, y_pred))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    def calculate_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all error metrics.

        Returns:
            Dictionary with all error metrics
        """
        return {
            'mse': round(self.mse(y_true, y_pred), 6),
            'rmse': round(self.rmse(y_true, y_pred), 6),
            'mae': round(self.mae(y_true, y_pred), 6),
            'r2': round(self.r2_score(y_true, y_pred), 6)
        }
