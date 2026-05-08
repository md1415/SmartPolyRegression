"""
SmartRegression: Auto-magic polynomial regression with intelligent degree selection
Author: Mehrshad Mahmoudi
"""

import json
import os
from typing import Dict, List, Union, Any
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from .error_calculator import ErrorCalculator
from .visualizer import Visualizer


class SmartRegression:
    """
    Automated polynomial regression with optimal degree selection.
    Evaluates multiple error metrics and saves results to JSON.
    """

    def __init__(self, max_degree: int = 5, cv_folds: int = 3):
        """
        Initialize SmartRegression model.

        Args:
            max_degree: Maximum polynomial degree to test
            cv_folds: Number of cross-validation folds (auto-adjusted if > n_samples)
        """
        self.max_degree = max_degree
        self.cv_folds = cv_folds
        self.best_model = None
        self.best_degree = None
        self.results = {}
        self.error_calculator = ErrorCalculator()
        self.visualizer = Visualizer()

    def _create_polynomial_model(self, degree: int) -> Pipeline:
        """Create polynomial regression pipeline for given degree."""
        return Pipeline([
            ('poly', PolynomialFeatures(degree)),
            ('linear', LinearRegression())
        ])

    def _find_best_degree(self, X: np.ndarray, y: np.ndarray) -> int:
        """
        Find optimal polynomial degree using cross-validation.

        Returns:
            Best degree based on negative MSE score
        """
        n_samples = len(X)
        # Adjust cv_folds if not enough samples
        actual_cv = min(self.cv_folds, n_samples)
        if actual_cv < 2:
            actual_cv = 2  # Need at least 2 folds

        best_score = float('-inf')
        best_degree = 1

        for degree in range(1, self.max_degree + 1):
            model = self._create_polynomial_model(degree)

            try:
                scores = cross_val_score(model, X, y, cv=actual_cv,
                                         scoring='neg_mean_squared_error')
                avg_score = scores.mean()
                std_score = scores.std()
            except Exception:
                # Fallback: use simple training score if CV fails
                model.fit(X, y)
                y_pred = model.predict(X)
                mse = np.mean((y - y_pred) ** 2)
                avg_score = -mse
                std_score = 0.0

            self.results[f'degree_{degree}'] = {
                'cv_score': float(avg_score),
                'cv_std': float(std_score),
                'cv_folds_used': actual_cv
            }

            if avg_score > best_score:
                best_score = avg_score
                best_degree = degree

        return best_degree

    def fit(self, X: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> 'SmartRegression':
        """
        Fit the model with automatic degree selection.

        Args:
            X: Feature matrix (list of lists or numpy array)
            y: Target values (list or numpy array)

        Returns:
            self
        """
        # Convert to numpy arrays
        X_array = np.array(X)
        y_array = np.array(y).ravel()

        # Reshape X if needed
        if len(X_array.shape) == 1:
            X_array = X_array.reshape(-1, 1)
        elif len(X_array.shape) == 2 and X_array.shape[1] != 1:
            # For now, only support 1 feature (polynomial regression)
            raise ValueError(f"Expected 1 feature, got {X_array.shape[1]} features")

        self.best_degree = self._find_best_degree(X_array, y_array)
        self.best_model = self._create_polynomial_model(self.best_degree)
        self.best_model.fit(X_array, y_array)

        # Calculate errors on training data
        y_pred = self.best_model.predict(X_array)
        self.error_metrics = self.error_calculator.calculate_all(y_array, y_pred)

        # Store basic info and training data
        self.results['best_degree'] = self.best_degree
        self.results['error_metrics'] = self.error_metrics
        self.results['training_samples'] = len(X_array)
        self.results['cv_folds_configured'] = self.cv_folds
        self._X_train = X_array  # Store for plotting
        self._y_train = y_array

        return self

    def predict(self, X: Union[List, np.ndarray]) -> np.ndarray:
        """
        Predict using the best model.

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """
        if self.best_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X_array = np.array(X)

        # Reshape if single number
        if len(X_array.shape) == 0 or (len(X_array.shape) == 1 and len(X_array) == 1):
            X_array = X_array.reshape(1, -1)
        elif len(X_array.shape) == 1:
            X_array = X_array.reshape(-1, 1)

        return self.best_model.predict(X_array)

    def save_results(self, filepath: str = 'outputs/model_results.json') -> None:
        """
        Save all results to JSON file.

        Args:
            filepath: Path to save JSON results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        serializable_results = self._make_serializable(self.results)

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)

        print(f"✅ Results saved to {filepath}")

    def plot_and_save(self, X: np.ndarray = None, y: np.ndarray = None,
                      save_path: str = 'outputs/plots/comparison.png') -> None:
        """
        Generate and save comparison plot without displaying.

        Args:
            X: Training features (optional, uses stored if not provided)
            y: Training targets (optional, uses stored if not provided)
            save_path: Path to save the plot
        """
        # Use stored data if not provided
        if X is None:
            X = self._X_train
        if y is None:
            y = self._y_train

        if X is None or y is None:
            raise ValueError("No training data available. Call fit() first or provide X, y.")

        X_plot = np.linspace(X.min() - 1, X.max() + 1, 100).reshape(-1, 1)
        y_pred_plot = self.predict(X_plot)

        self.visualizer.plot_comparison(
            X, y, X_plot, y_pred_plot,
            self.best_degree,
            self.error_metrics,
            save_path
        )

    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

    def get_summary(self) -> Dict:
        """Get model summary dictionary."""
        return {
            'best_degree': self.best_degree,
            'error_metrics': self.error_metrics,
            'max_degree_tested': self.max_degree,
            'cv_folds': self.cv_folds
        }
