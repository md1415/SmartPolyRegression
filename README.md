# 🤖 SmartRegression

**Smart Polynomial Regression with Automatic Degree Selection**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI/CD Pipeline](https://github.com/md1415/SmartPolyRegression/actions/workflows/ci.yml/badge.svg)](https://github.com/md1415/SmartPolyRegression/actions)

> Never guess the polynomial degree again. SmartRegression automatically finds the optimal degree using cross-validation and provides comprehensive error metrics.

## ✨ Features

- 🎯 **Auto Degree Selection** - Finds optimal polynomial degree via k-fold CV
- 📊 **4 Error Metrics** - MSE, RMSE, MAE, R²
- 💾 **JSON Export** - Save all results
- 📈 **Plot Generation** - Publication-ready plots
- 🧪 **11 Unit Tests** - 100% pass rate
- 🔄 **CI/CD Pipeline** - Automated testing

## 🚀 Quick Start

```python
from src.smart_regression import SmartRegression
import numpy as np

# Your data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# One line to rule them all
model = SmartRegression(max_degree=5)
model.fit(X, y)

# Get predictions, save results, generate plots
print(model.predict([[6]]))  # [12]
model.save_results('results.json')
model.plot_and_save(X, y, 'perfect_fit.png')
```

## 📦 Installation

```bash
git clone https://github.com/md1415/SmartPolyRegression.git
cd SmartPolyRegression
pip install -r requirements.txt
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📁 Project Structure

```text
SmartPolyRegression/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── src/
│   ├── smart_regression.py     # Main model class
│   ├── error_calculator.py     # Error metrics
│   ├── visualizer.py           # Plot generation
│   └── data_handler.py         # Data loading
├── tests/
│   └── test_smart_regression.py # Unit tests (11 tests)
├── examples/
│   ├── demo.py                  # Basic demo
│   └── load_data_demo.py        # Data loading demo
├── outputs/                     # Results and plots
├── pyproject.toml               # Package config
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🤝 Contributing

Contributions are welcome! Open an issue or PR.

## 📄 License

MIT License

## ⭐ Show your support

Give a ⭐ if this project helped you!