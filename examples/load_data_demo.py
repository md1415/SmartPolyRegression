"""Demo: Loading data from different sources"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smart_regression import SmartRegression
from src.data_handler import DataHandler


def main():
    print("🚀 Data Loading Demo")
    print("=" * 50)

    # Method 1: Generate synthetic data
    print("\n1. Generating synthetic cubic data...")
    X, y = DataHandler.generate_sample_data(
        n_samples=100,
        noise=0.2,
        degree=3,
        save_path='outputs/sample_data.json'
    )

    # Method 2: Load from JSON (just saved)
    print("\n2. Loading from JSON file...")
    X_loaded, y_loaded = DataHandler.load_json('outputs/sample_data.json')

    # Train model
    print("\n3. Training SmartRegression...")
    model = SmartRegression(max_degree=5)
    model.fit(X_loaded, y_loaded)

    print(f"\n📊 Results:")
    print(f"   Best degree: {model.best_degree}")
    print(f"   R² score: {model.error_metrics['r2']}")

    # Save all results
    model.save_results('outputs/demo_results.json')
    model.plot_and_save(X_loaded, y_loaded, 'outputs/plots/demo_plot.png')

    print("\n✅ Demo complete! Check outputs/ folder")


if __name__ == '__main__':
    main()