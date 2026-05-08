"""Visualization module that saves plots without displaying."""

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict


class Visualizer:
    """Generate and save plots without GUI display."""

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize visualizer with specific style."""
        try:
            plt.style.use(style)
        except:
            pass

    def plot_comparison(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_plot: np.ndarray, y_pred: np.ndarray,
                        degree: int, error_metrics: Dict[str, float],
                        save_path: str) -> None:
        """
        Create comparison plot and save to file.

        Args:
            X_train: Training features
            y_train: Training targets
            X_plot: Points for smooth curve
            y_pred: Predictions for X_plot
            degree: Best polynomial degree
            error_metrics: Dictionary of error metrics
            save_path: Where to save the plot
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot training data
        ax.scatter(X_train, y_train, color='blue', alpha=0.6,
                   s=60, label='Training Data', zorder=5)

        # Plot prediction curve
        ax.plot(X_plot, y_pred, color='red', linewidth=3,
                label=f'Polynomial Degree {degree}', zorder=10)

        # Add error metrics text box
        error_text = f"Error Metrics:\n"
        error_text += f"MSE: {error_metrics['mse']}\n"
        error_text += f"RMSE: {error_metrics['rmse']}\n"
        error_text += f"MAE: {error_metrics['mae']}\n"
        error_text += f"R²: {error_metrics['r2']}"

        ax.text(0.05, 0.95, error_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Labels and title
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('y', fontsize=12, fontweight='bold')
        ax.set_title(f'Smart Regression: Best Degree = {degree} (Auto-Selected)',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Plot saved to {save_path}")