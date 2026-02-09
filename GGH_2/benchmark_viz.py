"""Benchmark visualization utilities for creating consistent plots across benchmarks."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime


def plot_r2_comparison(results_r2, dataset_name, results_path):
    """Plot R² score comparison bar chart.

    Args:
        results_r2: Dictionary mapping method name -> list of R² scores
        dataset_name: Name of dataset for title (e.g., "Wine Quality")
        results_path: Directory path to save the plot
    """
    # Sort methods by mean R2 (descending - higher is better)
    method_means = [(m, np.mean(results_r2[m])) for m in results_r2.keys() if results_r2[m]]
    method_means.sort(key=lambda x: x[1], reverse=True)

    # Prepare data for plotting
    methods_to_plot = [m for m, _ in method_means]
    means = [mean for _, mean in method_means]
    stds = [np.std(results_r2[m]) for m in methods_to_plot]

    # Generate viridis colors without yellow (use 0 to 0.85 range)
    n_methods = len(methods_to_plot)
    viridis_colors = [cm.viridis(i) for i in np.linspace(0, 0.85, n_methods)]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods_to_plot))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=viridis_colors,
                  edgecolor='black', linewidth=1.2, ecolor='dimgray', alpha=0.8)

    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(f'{dataset_name} Benchmark: R² Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_to_plot, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars with bold font
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        offset = 0.01 if height >= 0 else -0.03
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                f'{mean:.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Generate filename with dataset name and date (manuscript quality)
    date_str = datetime.now().strftime('%Y-%m-%d')
    dataset_safe = dataset_name.replace(' ', '_')
    save_path = f'{results_path}/{dataset_safe}_r2_{date_str}.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to: {save_path}")


def plot_mse_comparison(results_mse, dataset_name, results_path):
    """Plot MSE comparison bar chart.

    Args:
        results_mse: Dictionary mapping method name -> list of MSE values
        dataset_name: Name of dataset for title (e.g., "Wine Quality")
        results_path: Directory path to save the plot
    """
    # Sort methods by MSE (ascending - lower is better)
    method_means = [(m, np.mean(results_mse[m])) for m in results_mse.keys() if results_mse[m]]
    method_means.sort(key=lambda x: x[1])

    # Prepare data for plotting
    methods_to_plot = [m for m, _ in method_means]
    means = [mean for _, mean in method_means]
    stds = [np.std(results_mse[m]) for m in methods_to_plot]

    # Generate viridis colors without yellow (use 0 to 0.85 range)
    n_methods = len(methods_to_plot)
    viridis_colors = [cm.viridis(i) for i in np.linspace(0, 0.85, n_methods)]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods_to_plot))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=viridis_colors,
                  edgecolor='black', linewidth=1.2, ecolor='dimgray', alpha=0.8)

    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(f'{dataset_name} Benchmark: MSE Comparison (Lower is Better)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_to_plot, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars with bold font
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        offset = max(0.01 * max(means), 0.005)  # Dynamic offset based on scale
        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Generate filename with dataset name and date (manuscript quality)
    date_str = datetime.now().strftime('%Y-%m-%d')
    dataset_safe = dataset_name.replace(' ', '_')
    save_path = f'{results_path}/{dataset_safe}_mse_{date_str}.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to: {save_path}")


def plot_mae_comparison(results_mae, dataset_name, results_path):
    """Plot MAE comparison bar chart.

    Args:
        results_mae: Dictionary mapping method name -> list of MAE values
        dataset_name: Name of dataset for title (e.g., "Wine Quality")
        results_path: Directory path to save the plot
    """
    # Sort methods by MAE (ascending - lower is better)
    method_means = [(m, np.mean(results_mae[m])) for m in results_mae.keys() if results_mae[m]]
    method_means.sort(key=lambda x: x[1])

    # Prepare data for plotting
    methods_to_plot = [m for m, _ in method_means]
    means = [mean for _, mean in method_means]
    stds = [np.std(results_mae[m]) for m in methods_to_plot]

    # Generate viridis colors without yellow (use 0 to 0.85 range)
    n_methods = len(methods_to_plot)
    viridis_colors = [cm.viridis(i) for i in np.linspace(0, 0.85, n_methods)]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods_to_plot))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=viridis_colors,
                  edgecolor='black', linewidth=1.2, ecolor='dimgray', alpha=0.8)

    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(f'{dataset_name} Benchmark: MAE Comparison (Lower is Better)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_to_plot, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars with bold font
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        offset = max(0.01 * max(means), 0.005)  # Dynamic offset based on scale
        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Generate filename with dataset name and date (manuscript quality)
    date_str = datetime.now().strftime('%Y-%m-%d')
    dataset_safe = dataset_name.replace(' ', '_')
    save_path = f'{results_path}/{dataset_safe}_mae_{date_str}.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to: {save_path}")


def plot_all_metrics(results_r2, results_mse, results_mae, dataset_name, results_path):
    """Plot all three metric comparisons.

    Convenience function that plots R², MSE, and MAE in sequence.

    Args:
        results_r2: Dictionary mapping method name -> list of R² scores
        results_mse: Dictionary mapping method name -> list of MSE values
        results_mae: Dictionary mapping method name -> list of MAE values
        dataset_name: Name of dataset for title (e.g., "Wine Quality")
        results_path: Directory path to save the plots
    """
    plot_r2_comparison(results_r2, dataset_name, results_path)
    plot_mse_comparison(results_mse, dataset_name, results_path)
    plot_mae_comparison(results_mae, dataset_name, results_path)
