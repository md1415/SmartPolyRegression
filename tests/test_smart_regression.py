"""
Unit tests for SmartRegression module
Run with: pytest tests/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import json
import tempfile
from src.smart_regression import SmartRegression
from src.error_calculator import ErrorCalculator
from src.data_handler import DataHandler


class TestErrorCalculator:
    """Test error calculation functions."""

    def test_mse_perfect_prediction(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert ErrorCalculator.mse(y_true, y_pred) == 0.0

    def test_mse_with_error(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        # MSE = ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 = (1+1+1)/3 = 1.0
        assert ErrorCalculator.mse(y_true, y_pred) == 1.0

    def test_rmse(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        assert ErrorCalculator.rmse(y_true, y_pred) == 1.0

    def test_r2_perfect(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert ErrorCalculator.r2_score(y_true, y_pred) == 1.0


class TestSmartRegression:
    """Test SmartRegression model."""

    def test_linear_data_chooses_degree_1(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        model = SmartRegression(max_degree=3, cv_folds=3)
        model.fit(X, y)

        # Linear data should prefer degree 1
        assert model.best_degree == 1
        assert model.error_metrics['r2'] == 1.0

    def test_quadratic_data(self):
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])  # Perfect quadratic

        model = SmartRegression(max_degree=3, cv_folds=3)
        model.fit(X, y)

        # With enough samples, should pick degree 2
        # But sometimes might pick degree 3 if scores are very close
        assert model.best_degree in [2, 3]  # Allow either
        assert model.error_metrics['r2'] > 0.99

    def test_prediction_shape(self):
        X = np.array([[1], [2], [3], [4], [5]])  # 5 samples minimum
        y = np.array([2, 4, 6, 8, 10])

        model = SmartRegression(max_degree=2, cv_folds=2)
        model.fit(X, y)

        # Test single prediction
        pred1 = model.predict([[4]])
        assert pred1.shape == (1,)

        # Test multiple predictions
        pred2 = model.predict([[4], [5], [6]])
        assert pred2.shape == (3,)

        # Test with list
        pred3 = model.predict([4])
        assert pred3.shape == (1,)

    def test_save_results_creates_file(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        model = SmartRegression(max_degree=2, cv_folds=2)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            model.save_results(tmp.name)

            # Verify file exists and contains valid JSON
            with open(tmp.name, 'r') as f:
                data = json.load(f)
                assert 'best_degree' in data
                assert 'error_metrics' in data

    def test_small_dataset(self):
        """Test with minimum samples (3 samples)"""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])

        model = SmartRegression(max_degree=2, cv_folds=2)  # Will auto-adjust
        model.fit(X, y)

        # Should not crash and produce some prediction
        pred = model.predict([[4]])
        assert pred is not None


class TestDataHandler:
    """Test data loading functionality."""

    def test_generate_sample_data(self):
        X, y = DataHandler.generate_sample_data(n_samples=50, noise=0.0)

        assert len(X) == 50
        assert len(y) == 50
        assert X.shape[1] == 1

    def test_save_and_load_json(self):
        X, y = DataHandler.generate_sample_data(n_samples=20)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            data = {'features': X.tolist(), 'target': y.tolist()}
            DataHandler.save_to_json(data, tmp.name)

            # Load back
            X_loaded, y_loaded = DataHandler.load_json(tmp.name)

            assert np.allclose(X, X_loaded)
            assert np.allclose(y, y_loaded)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])