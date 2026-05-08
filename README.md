# SmartRegression 🤖

**Automated polynomial regression with intelligent degree selection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![sklearn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

## ✨ Features

- 🎯 **Auto Degree Selection**: Automatically finds optimal polynomial degree using cross-validation
- 📊 **Multiple Error Metrics**: MSE, RMSE, MAE, and R²
- 💾 **JSON Export**: Save all results to structured JSON files
- 📈 **Plot Generation**: Creates publication-ready plots without displaying them
- 🔄 **Pipeline Ready**: Works with scikit-learn pipelines

## 🚀 Quick Start

```python
from src.smart_regression import SmartRegression
import numpy as np

# Your data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])

# Train with auto degree selection
model = SmartRegression(max_degree=3)
model.fit(X, y)

# Predict
predictions = model.predict([[6]])

# Save results
model.save_results('results.json')
model.plot_and_save(X, y, 'plot.png')