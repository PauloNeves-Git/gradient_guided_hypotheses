"""
Diagnostic Analysis Example for GGH

This script demonstrates how to use the new diagnostic and enhanced
selection capabilities to understand and improve gradient-guided hypotheses.

Key demonstrations:
1. Analyzing gradient separability across epochs
2. Comparing different selection methods
3. Understanding parameter impacts through ablation
4. Testing high missing data scenarios (>50% missing)
"""

import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../GGH')

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from GGH.data_ops import DataOperator
from GGH.selection_algorithms import AlgoModulators, compute_individual_grads
from GGH.models import initialize_model, load_model
from GGH.train_val_loop import TrainValidationManager
from GGH.inspector import Inspector

# New enhanced modules
from GGH.gradient_diagnostics import (
    GradientDiagnostics,
    EnrichedVectorBuilder,
    DoubleBackpropManager,
    compute_gradient_statistics,
    visualize_gradient_space
)
from GGH.enhanced_selection import (
    ContrastiveSelector,
    EnsembleSelector,
    CentroidDistanceSelector,
    AdaptiveSelector,
    gradient_selection_enhanced
)
from GGH.experiment_runner import (
    ExperimentConfig,
    ExperimentRunner,
    create_high_missing_config
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


# =============================================================================
# Example 1: Basic Diagnostic Analysis
# =============================================================================
def example_diagnostic_analysis():
    """
    Demonstrate how to use GradientDiagnostics to analyze
    gradient behavior during training.
    """
    print("\n" + "="*60)
    print("Example 1: Diagnostic Analysis of Gradient Behavior")
    print("="*60)

    # Configuration
    data_path = '../data/wine/red_wine.csv'
    inpt_vars = ['volatile acidity', 'total sulfur dioxide', 'citric acid']
    target_vars = ['quality']
    miss_vars = ['alcohol']
    hypothesis = [[9.35, 10, 11.5, 15]]
    partial_perc = 0.05
    rand_state = 42
    hidden_size = 32
    num_epochs = 30
    batch_size = 400

    set_seed(rand_state)

    # Initialize
    DO = DataOperator(data_path, inpt_vars, target_vars, miss_vars,
                      hypothesis, partial_perc, rand_state, device="cpu")
    DO.problem_type = 'regression'

    if DO.lack_partial_coverage:
        print("Insufficient partial coverage, try different random state")
        return

    # Initialize diagnostics
    diagnostics = GradientDiagnostics(save_path='../saved_results/diagnostics')

    # Initialize enriched vector builder with different configurations
    builder_basic = EnrichedVectorBuilder(
        use_gradients=True,
        use_input_context=True,
        use_loss_context=False,
        normalize_components=False
    )

    builder_enriched = EnrichedVectorBuilder(
        use_gradients=True,
        use_input_context=True,
        use_loss_context=True,
        use_gradient_magnitude=True,
        use_gradient_direction=True,
        normalize_components=True,
        component_weights={'gradients': 1.0, 'inputs': 0.5, 'loss': 0.3, 'magnitude': 0.2}
    )

    print(f"\nBasic vector config: {builder_basic.get_config()}")
    print(f"\nEnriched vector config: {builder_enriched.get_config()}")

    # Analyze separability (simulated gradients for demonstration)
    # In practice, these would come from actual training
    np.random.seed(rand_state)
    n_samples = 100

    # Simulated correct hypothesis gradients (tighter cluster)
    correct_grads = np.random.randn(n_samples, 32) * 0.5 + np.array([0.1] * 32)

    # Simulated incorrect hypothesis gradients (more spread)
    incorrect_grads = np.random.randn(n_samples * 3, 32) * 1.5

    # Simulated partial (known correct) gradients
    partial_grads = np.random.randn(10, 32) * 0.3 + np.array([0.1] * 32)

    # Compute separability metrics
    metrics = diagnostics.compute_separability_metrics(
        correct_grads, incorrect_grads, partial_grads
    )

    print("\n--- Separability Metrics ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Feature importance analysis
    all_grads = np.vstack([correct_grads, incorrect_grads])
    labels = np.array([1] * len(correct_grads) + [0] * len(incorrect_grads))

    importance_df = diagnostics.analyze_feature_importance(
        all_grads, labels, gradient_dim=32
    )

    print("\n--- Top 10 Most Important Features ---")
    print(importance_df.head(10)[['feature', 'cohens_d', 'ks_statistic', 'composite_importance']])

    # Visualize gradient space
    print("\nGenerating gradient space visualization...")
    visualize_gradient_space(
        correct_grads, incorrect_grads, partial_grads,
        method='tsne',
        title='Simulated Gradient Distributions'
    )


# =============================================================================
# Example 2: Comparing Selection Methods
# =============================================================================
def example_selection_comparison():
    """
    Compare different gradient selection methods:
    - Original OneClassSVM
    - ContrastiveSelector
    - EnsembleSelector
    - CentroidDistanceSelector
    """
    print("\n" + "="*60)
    print("Example 2: Comparing Selection Methods")
    print("="*60)

    np.random.seed(42)

    # Generate test data
    n_correct_partial = 20
    n_incorrect_partial = 60
    n_unknown = 200

    # Known correct (cluster around [1, 0, 0...])
    correct_partial = np.random.randn(n_correct_partial, 32) * 0.3 + np.array([1.0] + [0.0]*31)

    # Known incorrect (cluster around [-1, 0, 0...])
    incorrect_partial = np.random.randn(n_incorrect_partial, 32) * 0.5 + np.array([-1.0] + [0.0]*31)

    # Unknown: mix of correct (40%) and incorrect (60%)
    n_correct_unknown = int(n_unknown * 0.4)
    n_incorrect_unknown = n_unknown - n_correct_unknown

    unknown_correct = np.random.randn(n_correct_unknown, 32) * 0.4 + np.array([1.0] + [0.0]*31)
    unknown_incorrect = np.random.randn(n_incorrect_unknown, 32) * 0.6 + np.array([-1.0] + [0.0]*31)
    unknown_all = np.vstack([unknown_correct, unknown_incorrect])
    unknown_labels = np.array([1] * n_correct_unknown + [0] * n_incorrect_unknown)

    # Shuffle
    shuffle_idx = np.random.permutation(len(unknown_all))
    unknown_all = unknown_all[shuffle_idx]
    unknown_labels = unknown_labels[shuffle_idx]

    # Test each selector
    selectors = {
        'ContrastiveSelector (SVM)': ContrastiveSelector(classifier='svm'),
        'ContrastiveSelector (RF)': ContrastiveSelector(classifier='rf'),
        'EnsembleSelector': EnsembleSelector(voting='soft'),
        'CentroidDistanceSelector': CentroidDistanceSelector(margin=0.0),
        'AdaptiveSelector': AdaptiveSelector(initial_nu=0.3, final_nu=0.1)
    }

    results = []

    for name, selector in selectors.items():
        selector.fit(correct_partial, incorrect_partial)
        result = selector.predict(unknown_all)

        # Evaluate
        predicted_correct = result.selected_indices
        true_correct = np.where(unknown_labels == 1)[0]

        precision = len(set(predicted_correct) & set(true_correct)) / (len(predicted_correct) + 1e-10)
        recall = len(set(predicted_correct) & set(true_correct)) / (len(true_correct) + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        results.append({
            'Method': name,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Selection Rate': len(predicted_correct) / len(unknown_all),
            'Mean Confidence': np.mean(result.confidence_scores[predicted_correct]) if len(predicted_correct) > 0 else 0
        })

        print(f"\n{name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Selection Rate: {len(predicted_correct) / len(unknown_all):.4f}")

    results_df = pd.DataFrame(results)
    print("\n--- Summary Table ---")
    print(results_df.to_string(index=False))


# =============================================================================
# Example 3: Double Backpropagation Strategy
# =============================================================================
def example_double_backprop():
    """
    Demonstrate double backpropagation strategies for
    amplifying the partial data signal.
    """
    print("\n" + "="*60)
    print("Example 3: Double Backpropagation Strategies")
    print("="*60)

    # Simulated gradients
    n_partial = 10
    n_hypothesis = 100

    partial_grads = [tuple([torch.randn(32, 16), torch.randn(16, 1)]) for _ in range(n_partial)]
    hypothesis_grads = [tuple([torch.randn(32, 16), torch.randn(16, 1)]) for _ in range(n_hypothesis)]

    strategies = ['weighted', 'repeated', 'adaptive']

    for strategy in strategies:
        manager = DoubleBackpropManager(
            strategy=strategy,
            partial_weight=2.0,
            gradient_accumulation_steps=2
        )

        print(f"\n--- Strategy: {strategy} ---")

        for epoch in [0, 5, 10, 20]:
            combined_grads, weights = manager.process_partial_gradients(
                partial_grads, hypothesis_grads, epoch, batch_idx=0
            )

            print(f"  Epoch {epoch}:")
            print(f"    Total gradients: {len(combined_grads)}")
            print(f"    Weights shape: {weights.shape}")
            print(f"    Max weight: {np.max(weights):.2f}")
            print(f"    Min weight: {np.min(weights):.2f}")


# =============================================================================
# Example 4: High Missing Data Experiment
# =============================================================================
def example_high_missing_data():
    """
    Run experiments specifically designed for high missing data scenarios
    where >50% of the data has missing values.
    """
    print("\n" + "="*60)
    print("Example 4: High Missing Data Scenario (>50% missing)")
    print("="*60)

    # Create configuration optimized for high missing data
    config = create_high_missing_config(
        data_path='../data/wine/red_wine.csv',
        inpt_vars=['volatile acidity', 'total sulfur dioxide', 'citric acid'],
        target_vars=['quality'],
        miss_vars=['alcohol'],
        hypothesis=[[9.35, 10, 11.5, 15]],
        partial_perc=0.01  # Only 1% complete = 99% missing info about alcohol
    )

    print("\nConfiguration for high missing data:")
    print(f"  Partial percentage: {config.partial_perc*100:.1f}%")
    print(f"  Missing percentage: {(1-config.partial_perc)*100:.1f}%")
    print(f"  Selector type: {config.selector_type}")
    print(f"  Double backprop: {config.use_double_backprop}")
    print(f"  Partial weight: {config.partial_weight}")

    # Initialize experiment runner
    runner = ExperimentRunner(
        results_dir='../saved_results/high_missing_experiments',
        verbose=True
    )

    # Run comparison across different missing percentages
    print("\nRunning missing data comparison...")

    # This would run actual experiments - commented out to avoid long runtime
    # results_df = runner.run_missing_data_comparison(
    #     config,
    #     partial_percs=[0.01, 0.025, 0.05, 0.10, 0.20, 0.50],
    #     num_runs=3
    # )

    print("\nTo run full experiments, uncomment the code above.")
    print("Expected output: DataFrame with test_r2, selection_accuracy for each partial %")


# =============================================================================
# Example 5: Parameter Impact Analysis
# =============================================================================
def example_parameter_impact():
    """
    Analyze how different parameters impact gradient separation
    and final model performance.
    """
    print("\n" + "="*60)
    print("Example 5: Parameter Impact Analysis")
    print("="*60)

    parameters = {
        'nu': {
            'description': 'OneClassSVM nu parameter - controls boundary strictness',
            'test_values': [0.05, 0.1, 0.2, 0.3, 0.5],
            'impact': 'Lower nu = more restrictive selection, higher precision but lower recall'
        },
        'freqperc_cutoff': {
            'description': 'Frequency percentage cutoff for final selection',
            'test_values': [0.2, 0.33, 0.5, 0.7],
            'impact': 'Higher cutoff = stricter filtering, keeps only consistently selected hypotheses'
        },
        'use_context': {
            'description': 'Whether to include input features in gradient vectors',
            'test_values': [True, False],
            'impact': 'Context helps distinguish hypotheses with similar gradients but different inputs'
        },
        'normalize_grads_contx': {
            'description': 'Whether to normalize gradients and context',
            'test_values': [True, False],
            'impact': 'Normalization helps when gradient scales vary significantly'
        },
        'partial_weight': {
            'description': 'Weight multiplier for partial (known correct) data',
            'test_values': [1.0, 2.0, 3.0, 5.0],
            'impact': 'Higher weight = partial data influences selection more strongly'
        }
    }

    print("\n--- Key Parameters and Their Impact ---")
    for param, info in parameters.items():
        print(f"\n{param}:")
        print(f"  Description: {info['description']}")
        print(f"  Test values: {info['test_values']}")
        print(f"  Impact: {info['impact']}")

    # Simulated impact analysis (in practice, run actual experiments)
    print("\n--- Simulated Parameter Sweep Results ---")
    print("(These are illustrative - run actual experiments for real results)")

    simulated_results = pd.DataFrame({
        'nu': [0.05, 0.1, 0.2, 0.3, 0.5],
        'precision': [0.95, 0.88, 0.75, 0.62, 0.45],
        'recall': [0.35, 0.52, 0.68, 0.78, 0.85],
        'f1': [0.51, 0.65, 0.71, 0.69, 0.58],
        'test_r2': [0.28, 0.32, 0.35, 0.33, 0.29]
    })

    print(simulated_results.to_string(index=False))

    # Best configuration recommendation
    print("\n--- Recommendation ---")
    print("For high missing data (>50%):")
    print("  - Use nu=0.2 (balanced precision/recall)")
    print("  - Use ensemble selector for robustness")
    print("  - Enable double backprop with partial_weight=2-3")
    print("  - Set freqperc_cutoff=0.25 (more permissive)")
    print("  - Enable normalize_grads_contx=True")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("GGH DIAGNOSTIC AND ENHANCED SELECTION EXAMPLES")
    print("="*80)

    # Run examples
    example_diagnostic_analysis()
    example_selection_comparison()
    example_double_backprop()
    example_parameter_impact()

    # High missing data example (shorter runtime)
    example_high_missing_data()

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
