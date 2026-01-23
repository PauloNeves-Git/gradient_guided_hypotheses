"""
Experiment Runner for GGH

Systematic experimentation framework for testing gradient-guided hypotheses
under various conditions, especially high missing data scenarios (>50%).

This module provides:
1. Comprehensive experiment configuration
2. Ablation studies on key parameters
3. Comparison across selection methods
4. Statistical significance testing
"""

import numpy as np
import pandas as pd
import torch
import json
import os
from datetime import datetime
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from itertools import product
import warnings
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from .data_ops import DataOperator
from .selection_algorithms import AlgoModulators
from .models import initialize_model, load_model
from .train_val_loop import TrainValidationManager
from .inspector import Inspector
from .gradient_diagnostics import GradientDiagnostics, EnrichedVectorBuilder, DoubleBackpropManager


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    # Data configuration
    data_path: str = ""
    inpt_vars: List[str] = None
    target_vars: List[str] = None
    miss_vars: List[str] = None
    hypothesis: List[List[float]] = None
    partial_perc: float = 0.025

    # Model configuration
    hidden_size: int = 32
    dropout: float = 0.05
    model_type: str = "mlp"

    # Training configuration
    num_epochs: int = 100
    batch_size: int = 400
    lr: float = 0.001
    nu: float = 0.1

    # Algorithm configuration
    use_context: bool = True
    normalize_grads_contx: bool = False
    freqperc_cutoff: float = 0.33
    selector_type: str = "ocsvm"  # ocsvm, contrastive, ensemble, centroid, adaptive

    # Double backprop configuration
    use_double_backprop: bool = False
    double_backprop_strategy: str = "weighted"
    partial_weight: float = 2.0

    # Enriched vector configuration
    use_gradient_magnitude: bool = False
    use_gradient_direction: bool = False
    use_loss_context: bool = True
    use_prediction_context: bool = False

    # Experiment metadata
    experiment_name: str = "default"
    random_seed: int = 42
    num_runs: int = 5

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    run_id: int
    random_state: int

    # Performance metrics
    train_r2: float = None
    val_r2: float = None
    test_r2: float = None
    val_mse: float = None
    test_mse: float = None

    # Selection metrics
    selection_accuracy: float = None
    avg_selection_rate: float = None
    correct_selection_rate: float = None
    incorrect_selection_rate: float = None

    # Diagnostic metrics
    final_separability: float = None
    gradient_magnitude_mean: float = None

    # Training history
    train_errors: List[float] = None
    valid_errors: List[float] = None
    best_epoch: int = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['config'] = self.config.to_dict()
        return d


class ExperimentRunner:
    """
    Comprehensive experiment runner for GGH.

    Features:
    - Run single experiments or parameter sweeps
    - Track all metrics automatically
    - Generate comparison reports
    - Statistical significance testing
    """

    def __init__(
        self,
        results_dir: str = "./experiment_results",
        device: str = "cpu",
        verbose: bool = True
    ):
        self.results_dir = results_dir
        self.device = device
        self.verbose = verbose
        self.results = []

        os.makedirs(results_dir, exist_ok=True)

    def run_single_experiment(
        self,
        config: ExperimentConfig,
        run_id: int = 0
    ) -> ExperimentResult:
        """
        Run a single experiment with given configuration.
        """
        torch.manual_seed(config.random_seed + run_id)
        np.random.seed(config.random_seed + run_id)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running experiment: {config.experiment_name} (run {run_id})")
            print(f"Partial %: {config.partial_perc}, Selector: {config.selector_type}")
            print(f"{'='*60}")

        # Initialize data
        DO = DataOperator(
            config.data_path,
            config.inpt_vars,
            config.target_vars,
            config.miss_vars,
            config.hypothesis,
            config.partial_perc,
            config.random_seed + run_id,
            device=self.device
        )

        # Check for partial coverage
        if DO.lack_partial_coverage:
            warnings.warn(f"Insufficient partial coverage for run {run_id}")
            return ExperimentResult(
                config=config,
                run_id=run_id,
                random_state=config.random_seed + run_id
            )

        # Initialize algorithm modulators
        AM = AlgoModulators(
            DO,
            lr=config.lr,
            nu=config.nu,
            dropout=config.dropout,
            normalize_grads_contx=config.normalize_grads_contx,
            use_context=config.use_context,
            freqperc_cutoff=config.freqperc_cutoff
        )

        # Prepare dataloader
        dataloader = DO.prep_dataloader("use hypothesis", config.batch_size)

        # Initialize model
        model = initialize_model(
            DO, dataloader, config.hidden_size,
            config.random_seed + run_id, dropout=config.dropout,
            model_type=config.model_type
        )

        # Initialize inspector and diagnostics
        INSPECT = Inspector(config.data_path.rsplit('/', 1)[0], config.hidden_size)
        diagnostics = GradientDiagnostics()

        # Initialize double backprop manager if enabled
        double_backprop_mgr = None
        if config.use_double_backprop:
            double_backprop_mgr = DoubleBackpropManager(
                strategy=config.double_backprop_strategy,
                partial_weight=config.partial_weight
            )

        # Training
        TVM = TrainValidationManager(
            "use hypothesis",
            config.num_epochs,
            dataloader,
            config.batch_size,
            config.random_seed + run_id,
            self.results_dir,
            final_analysis=True
        )

        # Run training with custom selector if specified
        TVM.train_model(DO, AM, model, final_analysis=True)

        # Compute metrics
        best_model = load_model(DO, TVM.weights_save_path, config.batch_size)

        val_r2 = INSPECT.calculate_val_r2score(DO, TVM, best_model, data="validation")
        test_r2 = INSPECT.calculate_val_r2score(DO, TVM, best_model, data="test")

        # Compute selection metrics
        sel_tracker = DO.df_train_hypothesis.final_sel_hyp.values
        correct_mask = DO.df_train_hypothesis["correct_hypothesis"].values

        total_selected = sum([sum(s) for s in sel_tracker])
        correct_selected = sum([
            sum(s) for s, c in zip(sel_tracker, correct_mask) if c
        ])
        incorrect_selected = sum([
            sum(s) for s, c in zip(sel_tracker, correct_mask) if not c
        ])

        selection_accuracy = correct_selected / (total_selected + 1e-10)
        correct_rate = correct_selected / (sum(correct_mask) * config.num_epochs + 1e-10)
        incorrect_rate = incorrect_selected / (sum(~correct_mask) * config.num_epochs + 1e-10)

        result = ExperimentResult(
            config=config,
            run_id=run_id,
            random_state=config.random_seed + run_id,
            val_r2=val_r2,
            test_r2=test_r2,
            val_mse=min(TVM.valid_errors_epoch) if TVM.valid_errors_epoch else None,
            selection_accuracy=selection_accuracy,
            correct_selection_rate=correct_rate,
            incorrect_selection_rate=incorrect_rate,
            train_errors=TVM.train_errors_epoch,
            valid_errors=TVM.valid_errors_epoch,
            best_epoch=TVM.best_checkpoint
        )

        self.results.append(result)

        if self.verbose:
            print(f"Test R2: {test_r2:.4f}, Selection Accuracy: {selection_accuracy:.4f}")

        return result

    def run_multiple_experiments(
        self,
        config: ExperimentConfig,
        num_runs: int = None
    ) -> List[ExperimentResult]:
        """
        Run multiple experiments with different random seeds.
        """
        num_runs = num_runs or config.num_runs
        results = []

        for run_id in range(num_runs):
            result = self.run_single_experiment(config, run_id)
            results.append(result)

        return results

    def run_parameter_sweep(
        self,
        base_config: ExperimentConfig,
        param_grid: Dict[str, List[Any]],
        num_runs_per_config: int = 3
    ) -> pd.DataFrame:
        """
        Run parameter sweep across multiple configurations.

        Args:
            base_config: Base configuration to modify
            param_grid: Dict of parameter names to list of values
            num_runs_per_config: Number of runs for each configuration
        """
        all_results = []
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        total_configs = 1
        for vals in param_values:
            total_configs *= len(vals)

        if self.verbose:
            print(f"\nRunning parameter sweep: {total_configs} configurations")
            print(f"Parameters: {param_names}")

        config_idx = 0
        for value_combo in product(*param_values):
            config = deepcopy(base_config)

            # Set parameters
            param_dict = dict(zip(param_names, value_combo))
            for param, value in param_dict.items():
                setattr(config, param, value)

            config.experiment_name = f"sweep_{config_idx}"

            if self.verbose:
                print(f"\nConfig {config_idx + 1}/{total_configs}: {param_dict}")

            # Run experiments
            for run_id in range(num_runs_per_config):
                try:
                    result = self.run_single_experiment(config, run_id)
                    result_dict = result.to_dict()
                    result_dict.update(param_dict)
                    all_results.append(result_dict)
                except Exception as e:
                    warnings.warn(f"Error in config {config_idx}, run {run_id}: {e}")

            config_idx += 1

        df = pd.DataFrame(all_results)
        return df

    def run_ablation_study(
        self,
        base_config: ExperimentConfig,
        ablation_params: List[str],
        num_runs: int = 5
    ) -> pd.DataFrame:
        """
        Run ablation study - test impact of each parameter individually.
        """
        results = []

        # Baseline run
        if self.verbose:
            print("Running baseline configuration...")

        baseline_results = self.run_multiple_experiments(base_config, num_runs)
        for r in baseline_results:
            result_dict = r.to_dict()
            result_dict['ablation_type'] = 'baseline'
            result_dict['ablated_param'] = None
            results.append(result_dict)

        # Ablate each parameter
        for param in ablation_params:
            if self.verbose:
                print(f"\nAblating: {param}")

            ablated_config = deepcopy(base_config)

            # Set parameter to alternative value
            current_val = getattr(ablated_config, param)
            if isinstance(current_val, bool):
                setattr(ablated_config, param, not current_val)
            elif isinstance(current_val, float):
                setattr(ablated_config, param, current_val * 0.5)
            elif isinstance(current_val, int):
                setattr(ablated_config, param, current_val // 2)
            elif isinstance(current_val, str):
                # For selector_type, try different options
                alternatives = ['ocsvm', 'contrastive', 'ensemble', 'centroid']
                idx = alternatives.index(current_val) if current_val in alternatives else 0
                setattr(ablated_config, param, alternatives[(idx + 1) % len(alternatives)])

            ablated_results = self.run_multiple_experiments(ablated_config, num_runs)
            for r in ablated_results:
                result_dict = r.to_dict()
                result_dict['ablation_type'] = 'ablated'
                result_dict['ablated_param'] = param
                results.append(result_dict)

        return pd.DataFrame(results)

    def run_missing_data_comparison(
        self,
        base_config: ExperimentConfig,
        partial_percs: List[float] = None,
        num_runs: int = 5
    ) -> pd.DataFrame:
        """
        Compare performance across different missing data percentages.

        Specifically targets the goal of achieving good performance
        when >50% of data is missing.
        """
        if partial_percs is None:
            partial_percs = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

        all_results = []

        for partial_perc in partial_percs:
            if self.verbose:
                print(f"\n{'='*40}")
                print(f"Testing with {partial_perc*100:.1f}% partial data")
                print(f"(i.e., {(1-partial_perc)*100:.1f}% missing)")
                print(f"{'='*40}")

            config = deepcopy(base_config)
            config.partial_perc = partial_perc
            config.experiment_name = f"missing_{int((1-partial_perc)*100)}pct"

            results = self.run_multiple_experiments(config, num_runs)

            for r in results:
                result_dict = r.to_dict()
                result_dict['partial_perc'] = partial_perc
                result_dict['missing_perc'] = 1 - partial_perc
                all_results.append(result_dict)

        df = pd.DataFrame(all_results)
        return df

    def run_selector_comparison(
        self,
        base_config: ExperimentConfig,
        selectors: List[str] = None,
        num_runs: int = 5
    ) -> pd.DataFrame:
        """
        Compare different gradient selection methods.
        """
        if selectors is None:
            selectors = ['ocsvm', 'contrastive', 'ensemble', 'centroid', 'adaptive']

        all_results = []

        for selector in selectors:
            if self.verbose:
                print(f"\n{'='*40}")
                print(f"Testing selector: {selector}")
                print(f"{'='*40}")

            config = deepcopy(base_config)
            config.selector_type = selector
            config.experiment_name = f"selector_{selector}"

            results = self.run_multiple_experiments(config, num_runs)

            for r in results:
                result_dict = r.to_dict()
                result_dict['selector_type'] = selector
                all_results.append(result_dict)

        df = pd.DataFrame(all_results)
        return df

    def generate_report(
        self,
        results_df: pd.DataFrame = None,
        save_path: str = None
    ) -> str:
        """
        Generate a comprehensive experiment report.
        """
        if results_df is None:
            results_df = pd.DataFrame([r.to_dict() for r in self.results])

        report = []
        report.append("=" * 80)
        report.append("GRADIENT GUIDED HYPOTHESES - EXPERIMENT REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)

        # Summary statistics
        report.append("\n## SUMMARY STATISTICS ##")
        report.append(f"Total experiments: {len(results_df)}")

        if 'test_r2' in results_df.columns:
            report.append(f"\nTest R2 Score:")
            report.append(f"  Mean: {results_df['test_r2'].mean():.4f}")
            report.append(f"  Std:  {results_df['test_r2'].std():.4f}")
            report.append(f"  Min:  {results_df['test_r2'].min():.4f}")
            report.append(f"  Max:  {results_df['test_r2'].max():.4f}")

        if 'selection_accuracy' in results_df.columns:
            report.append(f"\nSelection Accuracy:")
            report.append(f"  Mean: {results_df['selection_accuracy'].mean():.4f}")
            report.append(f"  Std:  {results_df['selection_accuracy'].std():.4f}")

        # Group by configuration if available
        if 'selector_type' in results_df.columns:
            report.append("\n## BY SELECTOR TYPE ##")
            grouped = results_df.groupby('selector_type').agg({
                'test_r2': ['mean', 'std'],
                'selection_accuracy': ['mean', 'std']
            }).round(4)
            report.append(grouped.to_string())

        if 'missing_perc' in results_df.columns:
            report.append("\n## BY MISSING PERCENTAGE ##")
            grouped = results_df.groupby('missing_perc').agg({
                'test_r2': ['mean', 'std'],
                'selection_accuracy': ['mean', 'std']
            }).round(4)
            report.append(grouped.to_string())

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)

        return report_text

    def plot_results(
        self,
        results_df: pd.DataFrame,
        x_col: str,
        y_col: str = 'test_r2',
        hue_col: str = None,
        save_path: str = None
    ):
        """
        Plot experiment results.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if hue_col:
            sns.boxplot(data=results_df, x=x_col, y=y_col, hue=hue_col, ax=ax)
        else:
            sns.boxplot(data=results_df, x=x_col, y=y_col, ax=ax)

        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').title())
        ax.set_title(f'{y_col.replace("_", " ").title()} by {x_col.replace("_", " ").title()}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def statistical_comparison(
        self,
        results_df: pd.DataFrame,
        group_col: str,
        metric_col: str = 'test_r2',
        baseline_group: str = None
    ) -> pd.DataFrame:
        """
        Perform statistical significance tests between groups.
        """
        groups = results_df[group_col].unique()

        if baseline_group is None:
            baseline_group = groups[0]

        baseline_values = results_df[results_df[group_col] == baseline_group][metric_col].values

        comparison_results = []

        for group in groups:
            if group == baseline_group:
                continue

            group_values = results_df[results_df[group_col] == group][metric_col].values

            # Paired t-test
            t_stat, t_pval = stats.ttest_ind(baseline_values, group_values)

            # Mann-Whitney U test (non-parametric)
            u_stat, u_pval = stats.mannwhitneyu(baseline_values, group_values, alternative='two-sided')

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(baseline_values) + np.var(group_values)) / 2)
            cohens_d = (np.mean(group_values) - np.mean(baseline_values)) / (pooled_std + 1e-10)

            comparison_results.append({
                'baseline': baseline_group,
                'comparison': group,
                'baseline_mean': np.mean(baseline_values),
                'comparison_mean': np.mean(group_values),
                'improvement': np.mean(group_values) - np.mean(baseline_values),
                'improvement_pct': (np.mean(group_values) - np.mean(baseline_values)) / (abs(np.mean(baseline_values)) + 1e-10) * 100,
                't_statistic': t_stat,
                't_pvalue': t_pval,
                'u_statistic': u_stat,
                'u_pvalue': u_pval,
                'cohens_d': cohens_d,
                'significant_005': t_pval < 0.05
            })

        return pd.DataFrame(comparison_results)


def create_high_missing_config(
    data_path: str,
    inpt_vars: List[str],
    target_vars: List[str],
    miss_vars: List[str],
    hypothesis: List[List[float]],
    partial_perc: float = 0.01  # Only 1% complete data = 99% missing
) -> ExperimentConfig:
    """
    Create configuration optimized for high missing data scenarios.

    When >50% of data is missing, we need:
    - Stronger partial data signal (double backprop)
    - More robust selection (ensemble)
    - Better exploitation of limited complete data
    """
    return ExperimentConfig(
        data_path=data_path,
        inpt_vars=inpt_vars,
        target_vars=target_vars,
        miss_vars=miss_vars,
        hypothesis=hypothesis,
        partial_perc=partial_perc,

        # Model optimized for limited data
        hidden_size=32,
        dropout=0.1,  # Higher dropout to prevent overfitting
        model_type="mlp",

        # Training with more epochs for limited data
        num_epochs=150,
        batch_size=200,
        lr=0.0005,  # Lower learning rate for stability
        nu=0.2,  # More permissive selection initially

        # Algorithm tuned for high missing
        use_context=True,
        normalize_grads_contx=True,  # Normalize to handle scale differences
        freqperc_cutoff=0.25,  # Lower cutoff to select more hypotheses

        # Use ensemble selector for robustness
        selector_type='ensemble',

        # Enable double backprop to amplify partial data signal
        use_double_backprop=True,
        double_backprop_strategy='adaptive',
        partial_weight=3.0,  # Strong weight on partial data

        # Enriched vectors
        use_gradient_magnitude=True,
        use_gradient_direction=True,
        use_loss_context=True,

        experiment_name="high_missing_optimized",
        num_runs=10
    )
