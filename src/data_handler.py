"""
Data Handler Module for SmartRegression
Supports CSV, JSON, and direct data loading
"""

import json
import csv
import numpy as np
from typing import Tuple, Union, Optional, Dict, Any
from pathlib import Path


class DataHandler:
    """
    Load and preprocess data from various file formats.
    Supports CSV, JSON, and dictionary inputs.
    """

    @staticmethod
    def load_csv(filepath: str,
                 x_columns: Union[str, list],
                 y_column: str,
                 delimiter: str = ',') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file
            x_columns: Column name(s) for features
            y_column: Column name for target
            delimiter: CSV delimiter (default: ',')

        Returns:
            Tuple of (X, y) as numpy arrays

        Example:
            >>> X, y = DataHandler.load_csv('data.csv', ['temp', 'humidity'], 'sales')
        """
        data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                data.append(row)

        # Handle single or multiple feature columns
        if isinstance(x_columns, str):
            x_columns = [x_columns]

        X = np.array([[float(row[col]) for col in x_columns] for row in data])
        y = np.array([float(row[y_column]) for row in data])

        print(f"✅ Loaded CSV: {len(X)} samples, {len(x_columns)} features")
        return X, y

    @staticmethod
    def load_json(filepath: str,
                  x_key: str = 'features',
                  y_key: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from JSON file.

        Expected JSON format:
        {
            "features": [[1,2], [3,4], ...],
            "target": [10, 20, ...]
        }

        Args:
            filepath: Path to JSON file
            x_key: Key for features in JSON
            y_key: Key for target in JSON

        Returns:
            Tuple of (X, y) as numpy arrays
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        X = np.array(data[x_key])
        y = np.array(data[y_key])

        # Reshape if single feature
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        print(f"✅ Loaded JSON: {len(X)} samples, {X.shape[1]} features")
        return X, y

    @staticmethod
    def save_to_json(data: Dict[str, Any], filepath: str) -> None:
        """
        Save any data to JSON file.

        Args:
            data: Dictionary to save
            filepath: Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        serializable = convert(data)

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=4)

        print(f"✅ Data saved to {filepath}")

    @staticmethod
    def generate_sample_data(n_samples: int = 100,
                             noise: float = 0.1,
                             degree: int = 3,
                             save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic polynomial data for testing.

        Args:
            n_samples: Number of samples
            noise: Standard deviation of noise
            degree: Polynomial degree for underlying function
            save_path: If provided, saves data to CSV/JSON

        Returns:
            Tuple of (X, y)
        """
        np.random.seed(42)
        X = np.linspace(-3, 3, n_samples).reshape(-1, 1)

        # Generate polynomial function
        coeffs = np.random.randn(degree + 1) * 2
        y_poly = np.polyval(coeffs[::-1], X.flatten())

        # Add noise
        y = y_poly + noise * np.random.randn(n_samples)

        if save_path:
            data = {
                'features': X.tolist(),
                'target': y.tolist(),
                'metadata': {
                    'n_samples': n_samples,
                    'noise': noise,
                    'degree': degree,
                    'coeffs': coeffs.tolist()
                }
            }

            if save_path.endswith('.json'):
                DataHandler.save_to_json(data, save_path)
            elif save_path.endswith('.csv'):
                # Save as CSV
                with open(save_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['feature', 'target'])
                    for xi, yi in zip(X, y):
                        writer.writerow([xi[0], yi])
                print(f"✅ Sample data saved to {save_path}")

        return X, y
