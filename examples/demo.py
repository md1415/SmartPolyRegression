"""Demo script showing SmartRegression capabilities."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.smart_regression import SmartRegression


def generate_sample_data(n_samples=50, noise=0.2):
    """Generate sample non-linear data."""
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = 0.5 * X ** 3 - X ** 2 + 2 * X + 3 + noise * np.random.randn(n_samples, 1)
    y = y.ravel()
    return X, y


def main():
    print("🚀 SmartRegression Demo")
    print("=" * 50)

    # Generate data
    X, y = generate_sample_data(n_samples=80, noise=0.3)
    print(f"📊 Generated {len(X)} samples with cubic + quadratic + linear pattern")

    # Create and train model
    print("\n🤖 Training SmartRegression...")
    model = SmartRegression(max_degree=5, cv_folds=5)
    model.fit(X, y)

    # Get summary
    summary = model.get_summary()
    print(f"\n📈 Best polynomial degree: {summary['best_degree']}")
    print(f"\n📉 Error Metrics:")
    for metric, value in summary['error_metrics'].items():
        print(f"   {metric.upper()}: {value}")

    # Make predictions
    X_test = np.array([[-2.5], [0], [2.5]])
    predictions = model.predict(X_test)
    print(f"\n🔮 Predictions for [-2.5, 0, 2.5]:")
    for x, pred in zip(X_test, predictions):
        print(f"   x={x[0]:.1f} → y={pred:.3f}")

    # Save results
    model.save_results('outputs/model_results.json')

    # Generate and save plot
    model.plot_and_save(X, y, 'outputs/plots/smart_regression_plot.png')

    print("\n✅ Demo completed successfully!")
    print("📁 Check 'outputs/' folder for results and plots")


if __name__ == '__main__':
    main()