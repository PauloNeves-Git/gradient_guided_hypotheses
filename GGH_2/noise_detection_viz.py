"""Visualization utilities for noise detection benchmarks."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime


def plot_noise_detection_r2(all_results, dataset_name, results_path):
    """Plot R2 comparison bar chart for noise detection benchmark.

    Args:
        all_results: Dictionary mapping method name -> list of result dicts with 'test_r2'
        dataset_name: Name of dataset for title
        results_path: Directory path to save the plot
    """
    # Compute means and stds
    method_data = []
    for method, results in all_results.items():
        r2_list = [r['test_r2'] for r in results]
        method_data.append((method, np.mean(r2_list), np.std(r2_list)))

    # Sort by mean R2 (descending)
    method_data.sort(key=lambda x: x[1], reverse=True)

    methods = [m for m, _, _ in method_data]
    means = [m for _, m, _ in method_data]
    stds = [s for _, _, s in method_data]

    n_methods = len(methods)
    viridis_colors = [cm.viridis(i) for i in np.linspace(0, 0.85, n_methods)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_methods)
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=viridis_colors,
                  edgecolor='black', linewidth=1.2, ecolor='dimgray', alpha=0.8)

    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(f'{dataset_name} Noise Detection: R² Score Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for bar, mean in zip(bars, means):
        height = bar.get_height()
        offset = 0.01 if height >= 0 else -0.03
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                f'{mean:.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

    plt.tight_layout()
    date_str = datetime.now().strftime('%Y-%m-%d')
    dataset_safe = dataset_name.replace(' ', '_')
    save_path = f'{results_path}/{dataset_safe}_noise_r2_{date_str}.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to: {save_path}")


def plot_noise_detection_mse(all_results, dataset_name, results_path):
    """Plot MSE comparison bar chart for noise detection benchmark.

    Args:
        all_results: Dictionary mapping method name -> list of result dicts with 'test_mse'
        dataset_name: Name of dataset for title
        results_path: Directory path to save the plot
    """
    method_data = []
    for method, results in all_results.items():
        mse_list = [r['test_mse'] for r in results]
        method_data.append((method, np.mean(mse_list), np.std(mse_list)))

    # Sort by MSE (ascending - lower is better)
    method_data.sort(key=lambda x: x[1])

    methods = [m for m, _, _ in method_data]
    means = [m for _, m, _ in method_data]
    stds = [s for _, _, s in method_data]

    n_methods = len(methods)
    viridis_colors = [cm.viridis(i) for i in np.linspace(0, 0.85, n_methods)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_methods)
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=viridis_colors,
                  edgecolor='black', linewidth=1.2, ecolor='dimgray', alpha=0.8)

    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(f'{dataset_name} Noise Detection: MSE Comparison (Lower is Better)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for bar, mean in zip(bars, means):
        height = bar.get_height()
        offset = max(0.01 * max(means), 0.005)
        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    date_str = datetime.now().strftime('%Y-%m-%d')
    dataset_safe = dataset_name.replace(' ', '_')
    save_path = f'{results_path}/{dataset_safe}_noise_mse_{date_str}.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to: {save_path}")


def plot_noise_detection_mae(all_results, dataset_name, results_path):
    """Plot MAE comparison bar chart for noise detection benchmark.

    Args:
        all_results: Dictionary mapping method name -> list of result dicts with 'test_mae'
        dataset_name: Name of dataset for title
        results_path: Directory path to save the plot
    """
    method_data = []
    for method, results in all_results.items():
        mae_list = [r['test_mae'] for r in results]
        method_data.append((method, np.mean(mae_list), np.std(mae_list)))

    method_data.sort(key=lambda x: x[1])

    methods = [m for m, _, _ in method_data]
    means = [m for _, m, _ in method_data]
    stds = [s for _, _, s in method_data]

    n_methods = len(methods)
    viridis_colors = [cm.viridis(i) for i in np.linspace(0, 0.85, n_methods)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_methods)
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=viridis_colors,
                  edgecolor='black', linewidth=1.2, ecolor='dimgray', alpha=0.8)

    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(f'{dataset_name} Noise Detection: MAE Comparison (Lower is Better)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for bar, mean in zip(bars, means):
        height = bar.get_height()
        offset = max(0.01 * max(means), 0.005)
        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    date_str = datetime.now().strftime('%Y-%m-%d')
    dataset_safe = dataset_name.replace(' ', '_')
    save_path = f'{results_path}/{dataset_safe}_noise_mae_{date_str}.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to: {save_path}")


def plot_detection_metrics(all_results, dataset_name, results_path):
    """Plot detection precision and recall comparison.

    Only plots methods that have detection metrics (skips Full Info baselines).

    Args:
        all_results: Dictionary mapping method name -> list of result dicts
        dataset_name: Name of dataset for title
        results_path: Directory path to save the plot
    """
    # Filter methods with detection metrics
    detection_methods = {}
    for method, results in all_results.items():
        if results and 'detection' in results[0] and results[0]['detection'] is not None:
            detection_methods[method] = results

    if not detection_methods:
        print("No methods with detection metrics to plot.")
        return

    methods = list(detection_methods.keys())
    precision_means = [np.mean([r['detection']['precision'] for r in detection_methods[m]]) for m in methods]
    precision_stds = [np.std([r['detection']['precision'] for r in detection_methods[m]]) for m in methods]
    recall_means = [np.mean([r['detection']['recall'] for r in detection_methods[m]]) for m in methods]
    recall_stds = [np.std([r['detection']['recall'] for r in detection_methods[m]]) for m in methods]
    f1_means = [np.mean([r['detection']['f1'] for r in detection_methods[m]]) for m in methods]
    f1_stds = [np.std([r['detection']['f1'] for r in detection_methods[m]]) for m in methods]

    n_methods = len(methods)
    x = np.arange(n_methods)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision_means, width, yerr=precision_stds,
                   label='Precision', color=cm.viridis(0.1), edgecolor='black',
                   linewidth=1.2, ecolor='dimgray', alpha=0.8, capsize=4)
    bars2 = ax.bar(x, recall_means, width, yerr=recall_stds,
                   label='Recall', color=cm.viridis(0.4), edgecolor='black',
                   linewidth=1.2, ecolor='dimgray', alpha=0.8, capsize=4)
    bars3 = ax.bar(x + width, f1_means, width, yerr=f1_stds,
                   label='F1 Score', color=cm.viridis(0.7), edgecolor='black',
                   linewidth=1.2, ecolor='dimgray', alpha=0.8, capsize=4)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(f'{dataset_name} Noise Detection: Detection Metrics',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bars, vals in [(bars1, precision_means), (bars2, recall_means), (bars3, f1_means)]:
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    date_str = datetime.now().strftime('%Y-%m-%d')
    dataset_safe = dataset_name.replace(' ', '_')
    save_path = f'{results_path}/{dataset_safe}_detection_metrics_{date_str}.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to: {save_path}")


def plot_all_noise_detection_metrics(all_results, dataset_name, results_path):
    """Plot all noise detection visualizations.

    Args:
        all_results: Dictionary mapping method name -> list of result dicts
        dataset_name: Name of dataset for title
        results_path: Directory path to save the plots
    """
    plot_noise_detection_r2(all_results, dataset_name, results_path)
    plot_noise_detection_mse(all_results, dataset_name, results_path)
    plot_noise_detection_mae(all_results, dataset_name, results_path)
    plot_detection_metrics(all_results, dataset_name, results_path)
