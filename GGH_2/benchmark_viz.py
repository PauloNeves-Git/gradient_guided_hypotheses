"""Benchmark visualization utilities for creating consistent plots across benchmarks."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
from datetime import datetime


# =============================================================================
# SAVE / LOAD BENCHMARK RESULTS
# =============================================================================

def save_benchmark_results(results_r2, results_mse, results_mae, dataset_name,
                           results_path, timings=None, valid_r_states=None,
                           extra=None):
    """Save benchmark results to a JSON file.

    Args:
        results_r2: Dict mapping method name -> list of R² scores
        results_mse: Dict mapping method name -> list of MSE values
        results_mae: Dict mapping method name -> list of MAE values
        dataset_name: Name of dataset (used in filename)
        results_path: Directory to save the file
        timings: Optional dict mapping method name -> list of run times
        valid_r_states: Optional list of random states used
        extra: Optional dict with any additional metadata to save
    """
    data = {
        'dataset_name': dataset_name,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_runs': max((len(v) for v in results_r2.values() if v), default=0),
        'methods': list(results_r2.keys()),
        'results_r2': {k: [float(x) for x in v] for k, v in results_r2.items()},
        'results_mse': {k: [float(x) for x in v] for k, v in results_mse.items()},
        'results_mae': {k: [float(x) for x in v] for k, v in results_mae.items()},
    }
    if timings is not None:
        data['timings'] = {k: [float(x) for x in v] for k, v in timings.items()}
    if valid_r_states is not None:
        data['valid_r_states'] = [int(x) for x in valid_r_states]
    if extra is not None:
        data['extra'] = extra

    os.makedirs(results_path, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d')
    dataset_safe = dataset_name.replace(' ', '_')
    save_path = os.path.join(results_path, f'{dataset_safe}_results_{date_str}.json')
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {save_path}")
    return save_path


def load_benchmark_results(filepath):
    """Load benchmark results from a JSON file.

    Args:
        filepath: Path to the JSON results file

    Returns:
        tuple: (results_r2, results_mse, results_mae, metadata)
            where metadata is a dict with keys: dataset_name, date, n_runs,
            methods, timings (if saved), valid_r_states (if saved), extra (if saved)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    results_r2 = data['results_r2']
    results_mse = data['results_mse']
    results_mae = data['results_mae']

    metadata = {
        'dataset_name': data.get('dataset_name', ''),
        'date': data.get('date', ''),
        'n_runs': data.get('n_runs', 0),
        'methods': data.get('methods', []),
    }
    if 'timings' in data:
        metadata['timings'] = data['timings']
    if 'valid_r_states' in data:
        metadata['valid_r_states'] = data['valid_r_states']
    if 'extra' in data:
        metadata['extra'] = data['extra']

    print(f"Loaded results: {metadata['dataset_name']} ({metadata['date']})")
    print(f"  {metadata['n_runs']} runs, {len(metadata['methods'])} methods: {metadata['methods']}")
    return results_r2, results_mse, results_mae, metadata


def save_noise_detection_results(all_results, dataset_name, results_path,
                                 noise_profile=None, extra=None):
    """Save noise detection benchmark results to a JSON file.

    Args:
        all_results: Dict mapping method name -> list of result dicts
            Each result dict has keys like 'test_r2', 'test_mse', 'test_mae',
            and optionally 'detection' (with precision/recall/f1/accuracy)
        dataset_name: Name of dataset (used in filename)
        results_path: Directory to save the file
        noise_profile: Optional dict with noise simulation parameters
        extra: Optional dict with any additional metadata
    """
    # Convert to JSON-serializable format (drop non-serializable fields like sample_weights)
    serializable_results = {}
    for method, runs in all_results.items():
        serializable_runs = []
        for run in runs:
            clean_run = {}
            for k, v in run.items():
                if k == 'sample_weights':
                    continue  # skip large weight dicts
                if isinstance(v, dict):
                    clean_run[k] = {
                        sk: float(sv) if isinstance(sv, (int, float, np.integer, np.floating)) else sv
                        for sk, sv in v.items()
                    }
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    clean_run[k] = float(v)
                else:
                    clean_run[k] = v
            serializable_runs.append(clean_run)
        serializable_results[method] = serializable_runs

    data = {
        'dataset_name': dataset_name,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_runs': max((len(v) for v in all_results.values() if v), default=0),
        'methods': list(all_results.keys()),
        'all_results': serializable_results,
    }
    if noise_profile is not None:
        data['noise_profile'] = noise_profile
    if extra is not None:
        data['extra'] = extra

    os.makedirs(results_path, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d')
    dataset_safe = dataset_name.replace(' ', '_')
    save_path = os.path.join(results_path, f'{dataset_safe}_noise_results_{date_str}.json')
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {save_path}")
    return save_path


def load_noise_detection_results(filepath):
    """Load noise detection benchmark results from a JSON file.

    Args:
        filepath: Path to the JSON results file

    Returns:
        tuple: (all_results, metadata)
            where all_results maps method name -> list of result dicts,
            and metadata is a dict with dataset_name, date, noise_profile, etc.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    all_results = data['all_results']

    metadata = {
        'dataset_name': data.get('dataset_name', ''),
        'date': data.get('date', ''),
        'n_runs': data.get('n_runs', 0),
        'methods': data.get('methods', []),
    }
    if 'noise_profile' in data:
        metadata['noise_profile'] = data['noise_profile']
    if 'extra' in data:
        metadata['extra'] = data['extra']

    print(f"Loaded results: {metadata['dataset_name']} ({metadata['date']})")
    print(f"  {metadata['n_runs']} runs, {len(metadata['methods'])} methods: {metadata['methods']}")
    return all_results, metadata


def _sort_methods(results, descending=True):
    """Sort methods by mean value. Returns (methods, data_lists) in sorted order."""
    method_means = [(m, np.mean(results[m])) for m in results.keys() if results[m]]
    method_means.sort(key=lambda x: x[1], reverse=descending)
    methods = [m for m, _ in method_means]
    data = [results[m] for m in methods]
    return methods, data


def _make_boxplot(ax, methods, data, viridis_colors):
    """Create a styled box plot on the given axes."""
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='dimgray', linewidth=1.2),
                    capprops=dict(color='dimgray', linewidth=1.2),
                    flierprops=dict(marker='o', markerfacecolor='dimgray',
                                    markersize=4, alpha=0.6))
    for patch, color in zip(bp['boxes'], viridis_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=45, ha='right')

    # Add median labels
    for i, d in enumerate(data):
        median = np.median(d)
        ax.text(i + 1, median, f'{median:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    return bp


def plot_r2_comparison(results_r2, dataset_name, results_path, plot_type="bar"):
    """Plot R² score comparison.

    Args:
        results_r2: Dictionary mapping method name -> list of R² scores
        dataset_name: Name of dataset for title (e.g., "Wine Quality")
        results_path: Directory path to save the plot
        plot_type: "bar" for bar chart with error bars, "box" for box plot
    """
    methods, data = _sort_methods(results_r2, descending=True)
    means = [np.mean(d) for d in data]
    n_methods = len(methods)
    viridis_colors = [cm.viridis(i) for i in np.linspace(0, 0.80, n_methods)]

    fig, ax = plt.subplots(figsize=(12, 6))

    if plot_type == "box":
        _make_boxplot(ax, methods, data, viridis_colors)
    else:
        stds = [np.std(d) for d in data]
        x = np.arange(n_methods)
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=viridis_colors,
                      edgecolor='black', linewidth=1.2, ecolor='dimgray', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            offset = 0.01 if height >= 0 else -0.03
            va = 'bottom' if height >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                    f'{mean:.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(f'{dataset_name} Benchmark: R² Score Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    date_str = datetime.now().strftime('%Y-%m-%d')
    dataset_safe = dataset_name.replace(' ', '_')
    suffix = '_boxplot' if plot_type == 'box' else ''
    save_path = f'{results_path}/{dataset_safe}_r2{suffix}_{date_str}.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to: {save_path}")


def plot_mse_comparison(results_mse, dataset_name, results_path, plot_type="bar"):
    """Plot MSE comparison.

    Args:
        results_mse: Dictionary mapping method name -> list of MSE values
        dataset_name: Name of dataset for title (e.g., "Wine Quality")
        results_path: Directory path to save the plot
        plot_type: "bar" for bar chart with error bars, "box" for box plot
    """
    methods, data = _sort_methods(results_mse, descending=False)
    means = [np.mean(d) for d in data]
    n_methods = len(methods)
    viridis_colors = [cm.viridis(i) for i in np.linspace(0, 0.85, n_methods)]

    fig, ax = plt.subplots(figsize=(12, 6))

    if plot_type == "box":
        _make_boxplot(ax, methods, data, viridis_colors)
    else:
        stds = [np.std(d) for d in data]
        x = np.arange(n_methods)
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=viridis_colors,
                      edgecolor='black', linewidth=1.2, ecolor='dimgray', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            offset = max(0.01 * max(means), 0.005)
            ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(f'{dataset_name} Benchmark: MSE Comparison (Lower is Better)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    date_str = datetime.now().strftime('%Y-%m-%d')
    dataset_safe = dataset_name.replace(' ', '_')
    suffix = '_boxplot' if plot_type == 'box' else ''
    save_path = f'{results_path}/{dataset_safe}_mse{suffix}_{date_str}.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to: {save_path}")


def plot_mae_comparison(results_mae, dataset_name, results_path, plot_type="bar"):
    """Plot MAE comparison.

    Args:
        results_mae: Dictionary mapping method name -> list of MAE values
        dataset_name: Name of dataset for title (e.g., "Wine Quality")
        results_path: Directory path to save the plot
        plot_type: "bar" for bar chart with error bars, "box" for box plot
    """
    methods, data = _sort_methods(results_mae, descending=False)
    means = [np.mean(d) for d in data]
    n_methods = len(methods)
    viridis_colors = [cm.viridis(i) for i in np.linspace(0, 0.85, n_methods)]

    fig, ax = plt.subplots(figsize=(12, 6))

    if plot_type == "box":
        _make_boxplot(ax, methods, data, viridis_colors)
    else:
        stds = [np.std(d) for d in data]
        x = np.arange(n_methods)
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=viridis_colors,
                      edgecolor='black', linewidth=1.2, ecolor='dimgray', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            offset = max(0.01 * max(means), 0.005)
            ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title(f'{dataset_name} Benchmark: MAE Comparison (Lower is Better)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    date_str = datetime.now().strftime('%Y-%m-%d')
    dataset_safe = dataset_name.replace(' ', '_')
    suffix = '_boxplot' if plot_type == 'box' else ''
    save_path = f'{results_path}/{dataset_safe}_mae{suffix}_{date_str}.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to: {save_path}")


def plot_all_metrics(results_r2, results_mse, results_mae, dataset_name, results_path,
                     plot_type="bar"):
    """Plot all three metric comparisons.

    Convenience function that plots R², MSE, and MAE in sequence.

    Args:
        results_r2: Dictionary mapping method name -> list of R² scores
        results_mse: Dictionary mapping method name -> list of MSE values
        results_mae: Dictionary mapping method name -> list of MAE values
        dataset_name: Name of dataset for title (e.g., "Wine Quality")
        results_path: Directory path to save the plots
        plot_type: "bar" for bar charts with error bars, "box" for box plots
    """
    plot_r2_comparison(results_r2, dataset_name, results_path, plot_type)
    plot_mse_comparison(results_mse, dataset_name, results_path, plot_type)
    plot_mae_comparison(results_mae, dataset_name, results_path, plot_type)


def plot_combined_mse_r2(results_mse, results_r2, dataset_name, results_path):
    """Plot MSE boxplots with R² scores overlayed as a line on a secondary axis.

    Creates a single professional figure combining MSE distribution (boxplots)
    with mean R² scores (line + markers) on a dual y-axis layout.

    Args:
        results_mse: Dictionary mapping method name -> list of MSE values
        results_r2: Dictionary mapping method name -> list of R² scores
        dataset_name: Name of dataset for title (e.g., "Wine Quality")
        results_path: Directory path to save the plot
    """
    # Sort methods by median MSE (ascending = best first)
    methods, mse_data = _sort_methods(results_mse, descending=False)
    r2_data = [results_r2[m] for m in methods]
    n_methods = len(methods)

    # Color palette
    viridis_colors = [cm.viridis(i) for i in np.linspace(0.15, 0.85, n_methods)]
    r2_color = '#21536B'  # dark teal-blue from viridis range

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- MSE Boxplots (primary axis) ---
    bp = ax1.boxplot(
        mse_data, patch_artist=True, widths=0.55,
        medianprops=dict(color='black', linewidth=1.8),
        whiskerprops=dict(color='#555555', linewidth=1.2),
        capprops=dict(color='#555555', linewidth=1.2),
        flierprops=dict(marker='o', markerfacecolor='#999999',
                        markersize=4, alpha=0.5, markeredgecolor='none'),
    )
    for patch, color in zip(bp['boxes'], viridis_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1.2)
        patch.set_alpha(0.82)

    ax1.set_ylabel('Mean Squared Error', fontsize=12, color='#333333')
    ax1.tick_params(axis='y', labelcolor='#333333')

    # Median labels on boxplots (hovering just above the median line)
    for i, d in enumerate(mse_data):
        median = np.median(d)
        ax1.annotate(f'{median:.4f}', xy=(i + 1, median),
                     xytext=(0, 6), textcoords='offset points',
                     ha='center', va='bottom',
                     fontsize=8, fontweight='bold', color='#333333',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                               edgecolor='none', alpha=0.7))

    # --- R² Line (secondary axis) ---
    ax2 = ax1.twinx()
    r2_means = [np.mean(d) for d in r2_data]
    r2_stds = [np.std(d) for d in r2_data]
    x_positions = list(range(1, n_methods + 1))

    ax2.errorbar(
        x_positions, r2_means, yerr=r2_stds,
        color=r2_color, linewidth=2.2, marker='D', markersize=7,
        markerfacecolor='white', markeredgecolor=r2_color, markeredgewidth=2,
        capsize=0, elinewidth=1.0, ecolor='#21536B40', zorder=5,
        label='R\u00b2 Score',
    )

    # R² value labels
    for i, (mean, std) in enumerate(zip(r2_means, r2_stds)):
        ax2.annotate(
            f'{mean:.3f}', xy=(x_positions[i], mean),
            xytext=(0, 12), textcoords='offset points',
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            color=r2_color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=r2_color, alpha=0.85, linewidth=0.8),
        )

    # Expand R² axis to ensure annotations stay within bounds
    r2_max = max(m + s for m, s in zip(r2_means, r2_stds))
    r2_min = min(m - s for m, s in zip(r2_means, r2_stds))
    r2_range = r2_max - r2_min if r2_max > r2_min else 0.1
    ax2.set_ylim(r2_min - 0.15 * r2_range, r2_max + 0.20 * r2_range)

    ax2.set_ylabel('R\u00b2 Score', fontsize=12, color=r2_color)
    ax2.tick_params(axis='y', labelcolor=r2_color)

    # --- Shared formatting ---
    ax1.set_xticks(range(1, n_methods + 1))
    ax1.set_xticklabels(methods, rotation=35, ha='right', fontsize=10)
    ax1.set_title(f'{dataset_name}: Mean Squared Error Distribution & R\u00b2 Performance',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.2, linestyle='--')
    ax1.set_axisbelow(True)

    # Legend combining both axes
    from matplotlib.lines import Line2D
    box_patch = Rectangle((0, 0), 1, 1, facecolor=viridis_colors[0],
                           edgecolor='#333333', linewidth=1.2, alpha=0.82)
    r2_line = Line2D([0], [0], color=r2_color, linewidth=2.2, marker='D',
                     markersize=7, markerfacecolor='white',
                     markeredgecolor=r2_color, markeredgewidth=2)
    ax1.legend([box_patch, r2_line], ['Mean Squared Error', 'R\u00b2 Score'],
               loc='upper right', fontsize=10, framealpha=0.9)

    fig.tight_layout()

    date_str = datetime.now().strftime('%Y-%m-%d')
    dataset_safe = dataset_name.replace(' ', '_')
    save_path = f'{results_path}/{dataset_safe}_combined_mse_r2_{date_str}.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to: {save_path}")
    return save_path


def plot_weight_distributions(gid_weights, DO, dataset_name='', results_path=None,
                              bins=20, hist_alpha=0.3, kde=True):
    """Plot GGH V2 soft weight distributions for correct vs incorrect hypotheses.

    Equivalent of selection_histograms from the old GGH, but shows continuous
    weight values instead of discrete selection counts. Correct hypotheses
    (green) should cluster at higher weights, incorrect (red) at lower weights.

    Includes all hypotheses used for training:
    - Partial correct gids (known correct) at weight 1.0
    - Blacklisted gids (known incorrect of partial samples) at weight 0.0
    - Selected hypotheses from gid_weights at their assigned weight
    - Non-selected hypotheses (not chosen by GGH) at weight 0.0

    Args:
        gid_weights: Dict mapping global_id -> weight (from run_ggh_soft_refinement)
        DO: DataOperator instance with df_train_hypothesis
        dataset_name: Name of dataset for title
        results_path: Directory path to save the plot (optional)
        bins: Number of histogram bins
        hist_alpha: Transparency for histogram bars
        kde: Whether to overlay KDE curves
    """
    correct_weights = []
    incorrect_weights = []

    # Identify partial correct gids from DO (known correct, used at weight 1.0)
    partial_correct_gids = set(DO.df_train_hypothesis[
        (DO.df_train_hypothesis['partial_full_info'] == 1) &
        (DO.df_train_hypothesis['correct_hypothesis'] == True)
    ].index.tolist())

    # Add partial correct gids at weight 1.0
    for gid in partial_correct_gids:
        correct_weights.append(1.0)

    # Add gid_weights (GGH-selected hypotheses)
    for gid, weight in gid_weights.items():
        is_correct = DO.df_train_hypothesis.iloc[gid]['correct_hypothesis']
        if is_correct:
            correct_weights.append(weight)
        else:
            incorrect_weights.append(weight)

    if not correct_weights or not incorrect_weights:
        print("No correct or incorrect hypothesis weights to plot.")
        return

    data_combined = np.hstack((correct_weights, incorrect_weights))
    bin_edges = np.linspace(min(data_combined), max(data_combined), bins + 1)

    correct_hist, bin_edges = np.histogram(correct_weights, bins=bin_edges, density=False)
    incorrect_hist, _ = np.histogram(incorrect_weights, bins=bin_edges, density=False)

    correct_pct = (correct_hist / sum(correct_hist)) * 100
    incorrect_pct = (incorrect_hist / sum(incorrect_hist)) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(bin_edges[:-1], correct_pct, width=np.diff(bin_edges), align='edge',
           alpha=hist_alpha, color='green', label='Correct Hypothesis')
    ax.bar(bin_edges[:-1], incorrect_pct, width=np.diff(bin_edges), align='edge',
           alpha=hist_alpha, color='red', label='Incorrect Hypothesis')

    bin_width = np.diff(bin_edges)[0]
    for left, height in zip(bin_edges[:-1], correct_pct):
        rect = Rectangle((left, 0), bin_width, height, fill=None,
                         alpha=0.7, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
    for left, height in zip(bin_edges[:-1], incorrect_pct):
        rect = Rectangle((left, 0), bin_width, height, fill=None,
                         alpha=0.7, edgecolor='black', linewidth=1)
        ax.add_patch(rect)

    try:
        if kde and len(correct_weights) > 1 and len(incorrect_weights) > 1:
            x_d = np.linspace(min(data_combined), max(data_combined), 1000)
            kde_correct = gaussian_kde(correct_weights, bw_method=0.15)
            kde_incorrect = gaussian_kde(incorrect_weights, bw_method=0.15)

            kde_c_vals = kde_correct(x_d)
            kde_i_vals = kde_incorrect(x_d)

            kde_c_scale = max(correct_pct) / max(kde_c_vals) if max(kde_c_vals) > 0 else 1
            kde_i_scale = max(incorrect_pct) / max(kde_i_vals) if max(kde_i_vals) > 0 else 1

            ax.plot(x_d, kde_c_vals * kde_c_scale, color='green', alpha=0.7)
            ax.plot(x_d, kde_i_vals * kde_i_scale, color='red', alpha=0.7)
    except Exception:
        pass

    ax.tick_params(direction='out')
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=11)
    ax.set_xlabel('GGH Weight', fontsize=12)
    ax.set_ylabel('Hypothesis Probability (%)', fontsize=12)

    title = 'Hypothesis Weight Distributions'
    if dataset_name:
        title = f'{dataset_name}: {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if results_path:
        os.makedirs(results_path, exist_ok=True)
        date_str = datetime.now().strftime('%Y-%m-%d')
        dataset_safe = dataset_name.replace(' ', '_') if dataset_name else 'dataset'
        save_path = os.path.join(results_path, f'{dataset_safe}_weight_distributions_{date_str}.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()
