# 🤖 SmartRegression

**Smart Polynomial Regression with Automatic Degree Selection**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org)
[![Tests](https://github.com/YOUR_USERNAME/SmartPolyRegression/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/SmartPolyRegression/actions)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/SmartPolyRegression/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/SmartPolyRegression)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> Never guess the polynomial degree again. SmartRegression automatically finds the optimal degree using cross-validation and provides comprehensive error metrics.

## ✨ Why SmartRegression?

Traditional linear regression forces you to guess the polynomial degree. SmartRegression eliminates this guesswork by:
- 🎯 **Auto-selecting** the best degree via k-fold cross-validation
- 📊 **Calculating** 4 different error metrics (MSE, RMSE, MAE, R²)
- 💾 **Exporting** all results to structured JSON
- 📈 **Generating** publication-ready plots without display

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