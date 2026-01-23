"""
Gradient Diagnostics Module for GGH

This module provides comprehensive diagnostic tools to understand how gradients
and enriched vectors behave, and how parameters impact the separation between
correct and incorrect hypotheses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import cdist
import torch
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime


class GradientDiagnostics:
    """
    Comprehensive diagnostic tools for analyzing gradient behavior in GGH.

    This class provides methods to:
    1. Measure separability between correct/incorrect hypothesis gradients
    2. Analyze feature importance in enriched vectors
    3. Track gradient evolution across epochs
    4. Perform ablation studies on parameters
    """

    def __init__(self, save_path: str = None):
        self.save_path = save_path
        self.diagnostic_history = []
        self.epoch_metrics = {}

    def compute_separability_metrics(
        self,
        correct_grads: np.ndarray,
        incorrect_grads: np.ndarray,
        partial_grads: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute multiple metrics to quantify how well correct and incorrect
        hypothesis gradients can be separated.

        Args:
            correct_grads: Gradients from correct hypotheses (unknown)
            incorrect_grads: Gradients from incorrect hypotheses (unknown)
            partial_grads: Gradients from known partial data (ground truth)

        Returns:
            Dictionary of separability metrics
        """
        metrics = {}

        # Combine for clustering metrics
        all_grads = np.vstack([correct_grads, incorrect_grads])
        labels = np.array([1] * len(correct_grads) + [0] * len(incorrect_grads))

        # 1. Silhouette Score (-1 to 1, higher is better)
        try:
            metrics['silhouette_score'] = silhouette_score(all_grads, labels)
        except:
            metrics['silhouette_score'] = 0.0

        # 2. Calinski-Harabasz Index (higher is better)
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(all_grads, labels)
        except:
            metrics['calinski_harabasz'] = 0.0

        # 3. Davies-Bouldin Index (lower is better)
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(all_grads, labels)
        except:
            metrics['davies_bouldin'] = float('inf')

        # 4. Inter-class distance (higher is better)
        correct_centroid = np.mean(correct_grads, axis=0)
        incorrect_centroid = np.mean(incorrect_grads, axis=0)
        metrics['centroid_distance'] = np.linalg.norm(correct_centroid - incorrect_centroid)

        # 5. Intra-class variance ratio
        correct_var = np.mean(np.var(correct_grads, axis=0))
        incorrect_var = np.mean(np.var(incorrect_grads, axis=0))
        metrics['correct_variance'] = correct_var
        metrics['incorrect_variance'] = incorrect_var
        metrics['variance_ratio'] = correct_var / (incorrect_var + 1e-10)

        # 6. Kolmogorov-Smirnov test on first principal component
        pca = PCA(n_components=1)
        all_pca = pca.fit_transform(all_grads)
        correct_pca = all_pca[:len(correct_grads)]
        incorrect_pca = all_pca[len(correct_grads):]
        ks_stat, ks_pval = ks_2samp(correct_pca.flatten(), incorrect_pca.flatten())
        metrics['ks_statistic'] = ks_stat
        metrics['ks_pvalue'] = ks_pval

        # 7. Wasserstein distance on first PC
        metrics['wasserstein_distance'] = wasserstein_distance(
            correct_pca.flatten(), incorrect_pca.flatten()
        )

        # 8. If partial grads provided, compute alignment
        if partial_grads is not None and len(partial_grads) > 0:
            partial_centroid = np.mean(partial_grads, axis=0)
            metrics['correct_to_partial_dist'] = np.linalg.norm(correct_centroid - partial_centroid)
            metrics['incorrect_to_partial_dist'] = np.linalg.norm(incorrect_centroid - partial_centroid)
            metrics['partial_alignment_ratio'] = (
                metrics['incorrect_to_partial_dist'] /
                (metrics['correct_to_partial_dist'] + 1e-10)
            )

        return metrics

    def analyze_feature_importance(
        self,
        enriched_vectors: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str] = None,
        gradient_dim: int = None
    ) -> pd.DataFrame:
        """
        Analyze which features in enriched vectors contribute most to separation.

        Args:
            enriched_vectors: Combined gradient + context vectors
            labels: Binary labels (1=correct, 0=incorrect)
            feature_names: Optional names for features
            gradient_dim: Dimension where gradients end and context begins

        Returns:
            DataFrame with feature importance scores
        """
        n_features = enriched_vectors.shape[1]

        if feature_names is None:
            if gradient_dim:
                feature_names = (
                    [f'grad_{i}' for i in range(gradient_dim)] +
                    [f'context_{i}' for i in range(n_features - gradient_dim)]
                )
            else:
                feature_names = [f'feature_{i}' for i in range(n_features)]

        importance_scores = []

        correct_mask = labels == 1
        incorrect_mask = labels == 0

        for i in range(n_features):
            feat_correct = enriched_vectors[correct_mask, i]
            feat_incorrect = enriched_vectors[incorrect_mask, i]

            # Mean difference
            mean_diff = abs(np.mean(feat_correct) - np.mean(feat_incorrect))

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(feat_correct) + np.var(feat_incorrect)) / 2)
            cohens_d = mean_diff / (pooled_std + 1e-10)

            # KS statistic
            ks_stat, _ = ks_2samp(feat_correct, feat_incorrect)

            # Variance ratio
            var_ratio = np.var(feat_correct) / (np.var(feat_incorrect) + 1e-10)

            importance_scores.append({
                'feature': feature_names[i] if i < len(feature_names) else f'feature_{i}',
                'mean_difference': mean_diff,
                'cohens_d': cohens_d,
                'ks_statistic': ks_stat,
                'variance_ratio': var_ratio,
                'is_gradient': i < gradient_dim if gradient_dim else True
            })

        df = pd.DataFrame(importance_scores)
        df['composite_importance'] = (
            df['cohens_d'].rank(pct=True) * 0.4 +
            df['ks_statistic'].rank(pct=True) * 0.4 +
            df['mean_difference'].rank(pct=True) * 0.2
        )

        return df.sort_values('composite_importance', ascending=False)

    def track_epoch_metrics(
        self,
        epoch: int,
        correct_grads: np.ndarray,
        incorrect_grads: np.ndarray,
        partial_grads: np.ndarray = None,
        selection_accuracy: float = None
    ):
        """
        Track gradient separability metrics across epochs.

        Args:
            epoch: Current epoch number
            correct_grads: Gradients from correct hypotheses
            incorrect_grads: Gradients from incorrect hypotheses
            partial_grads: Gradients from partial (known) data
            selection_accuracy: Optional accuracy of hypothesis selection
        """
        metrics = self.compute_separability_metrics(
            correct_grads, incorrect_grads, partial_grads
        )

        if selection_accuracy is not None:
            metrics['selection_accuracy'] = selection_accuracy

        self.epoch_metrics[epoch] = metrics

    def plot_epoch_evolution(self, metric_names: List[str] = None, save_fig: bool = True):
        """
        Plot how separability metrics evolve across training epochs.
        """
        if not self.epoch_metrics:
            print("No epoch metrics tracked yet.")
            return

        if metric_names is None:
            metric_names = ['silhouette_score', 'centroid_distance', 'wasserstein_distance']

        epochs = sorted(self.epoch_metrics.keys())

        fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 3*len(metric_names)))
        if len(metric_names) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metric_names):
            values = [self.epoch_metrics[e].get(metric, 0) for e in epochs]
            ax.plot(epochs, values, 'b-o', linewidth=2, markersize=6)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Evolution')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_fig and self.save_path:
            fig_path = os.path.join(self.save_path, 'epoch_evolution.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')

        plt.show()

    def run_ablation_study(
        self,
        DO,
        AM,
        param_name: str,
        param_values: List[Any],
        num_runs: int = 3
    ) -> pd.DataFrame:
        """
        Run ablation study on a specific parameter.

        Args:
            DO: DataOperator instance
            AM: AlgoModulators instance
            param_name: Name of parameter to vary
            param_values: List of values to test
            num_runs: Number of runs per value

        Returns:
            DataFrame with results for each parameter value
        """
        results = []

        for val in param_values:
            setattr(AM, param_name, val)

            for run in range(num_runs):
                # This would need to be integrated with training loop
                # Placeholder for actual training run
                result = {
                    'param_name': param_name,
                    'param_value': val,
                    'run': run,
                    # These would be filled from actual training
                    'final_separability': None,
                    'selection_accuracy': None,
                    'test_r2': None
                }
                results.append(result)

        return pd.DataFrame(results)


class EnrichedVectorBuilder:
    """
    Build enriched gradient vectors with configurable components.

    This class allows systematic experimentation with what information
    to include in the enriched vectors used for hypothesis selection.
    """

    def __init__(
        self,
        use_gradients: bool = True,
        gradient_layers: List[int] = [-2],
        use_input_context: bool = True,
        use_loss_context: bool = True,
        use_prediction_context: bool = False,
        use_gradient_magnitude: bool = False,
        use_gradient_direction: bool = False,
        use_second_order: bool = False,
        normalize_components: bool = False,
        component_weights: Dict[str, float] = None
    ):
        """
        Initialize enriched vector builder.

        Args:
            use_gradients: Include raw gradient values
            gradient_layers: Which layers to extract gradients from
            use_input_context: Include input features
            use_loss_context: Include loss values
            use_prediction_context: Include model predictions
            use_gradient_magnitude: Include gradient L2 norm
            use_gradient_direction: Include unit direction vector
            use_second_order: Include gradient of gradient (Hessian diagonal)
            normalize_components: Normalize each component separately
            component_weights: Weights for different components
        """
        self.use_gradients = use_gradients
        self.gradient_layers = gradient_layers
        self.use_input_context = use_input_context
        self.use_loss_context = use_loss_context
        self.use_prediction_context = use_prediction_context
        self.use_gradient_magnitude = use_gradient_magnitude
        self.use_gradient_direction = use_gradient_direction
        self.use_second_order = use_second_order
        self.normalize_components = normalize_components
        self.component_weights = component_weights or {}

        self.component_dims = {}

    def build_enriched_vector(
        self,
        gradients: tuple,
        inputs: torch.Tensor,
        loss: torch.Tensor = None,
        predictions: torch.Tensor = None,
        hessian_diag: torch.Tensor = None
    ) -> np.ndarray:
        """
        Build a single enriched vector from gradient and context.

        Args:
            gradients: Tuple of gradient tensors for each layer
            inputs: Input features tensor
            loss: Loss value tensor
            predictions: Model predictions tensor
            hessian_diag: Diagonal of Hessian (second order)

        Returns:
            Numpy array of enriched vector
        """
        components = []

        # 1. Raw gradients from specified layers
        if self.use_gradients:
            for layer_idx in self.gradient_layers:
                grad = gradients[layer_idx]
                if isinstance(grad, torch.Tensor):
                    grad = grad.detach().cpu().numpy()
                grad_flat = grad.flatten()

                weight = self.component_weights.get('gradients', 1.0)
                if self.normalize_components:
                    grad_flat = self._normalize(grad_flat)
                components.append(grad_flat * weight)

        # 2. Gradient magnitude
        if self.use_gradient_magnitude:
            for layer_idx in self.gradient_layers:
                grad = gradients[layer_idx]
                if isinstance(grad, torch.Tensor):
                    grad = grad.detach().cpu().numpy()
                magnitude = np.array([np.linalg.norm(grad.flatten())])

                weight = self.component_weights.get('magnitude', 1.0)
                components.append(magnitude * weight)

        # 3. Gradient direction (unit vector)
        if self.use_gradient_direction:
            for layer_idx in self.gradient_layers:
                grad = gradients[layer_idx]
                if isinstance(grad, torch.Tensor):
                    grad = grad.detach().cpu().numpy()
                grad_flat = grad.flatten()
                norm = np.linalg.norm(grad_flat)
                direction = grad_flat / (norm + 1e-10)

                weight = self.component_weights.get('direction', 1.0)
                components.append(direction * weight)

        # 4. Input context
        if self.use_input_context and inputs is not None:
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.detach().cpu().numpy()
            input_flat = inputs.flatten()

            weight = self.component_weights.get('inputs', 1.0)
            if self.normalize_components:
                input_flat = self._normalize(input_flat)
            components.append(input_flat * weight)

        # 5. Loss context
        if self.use_loss_context and loss is not None:
            if isinstance(loss, torch.Tensor):
                loss = loss.detach().cpu().numpy()
            loss_flat = np.atleast_1d(loss).flatten()

            weight = self.component_weights.get('loss', 1.0)
            components.append(loss_flat * weight)

        # 6. Prediction context
        if self.use_prediction_context and predictions is not None:
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu().numpy()
            pred_flat = predictions.flatten()

            weight = self.component_weights.get('predictions', 1.0)
            components.append(pred_flat * weight)

        # 7. Second order (Hessian diagonal)
        if self.use_second_order and hessian_diag is not None:
            if isinstance(hessian_diag, torch.Tensor):
                hessian_diag = hessian_diag.detach().cpu().numpy()
            hess_flat = hessian_diag.flatten()

            weight = self.component_weights.get('hessian', 0.5)
            if self.normalize_components:
                hess_flat = self._normalize(hess_flat)
            components.append(hess_flat * weight)

        return np.concatenate(components)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to zero mean and unit variance."""
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-10:
            return arr - mean
        return (arr - mean) / std

    def get_config(self) -> Dict:
        """Return current configuration as dictionary."""
        return {
            'use_gradients': self.use_gradients,
            'gradient_layers': self.gradient_layers,
            'use_input_context': self.use_input_context,
            'use_loss_context': self.use_loss_context,
            'use_prediction_context': self.use_prediction_context,
            'use_gradient_magnitude': self.use_gradient_magnitude,
            'use_gradient_direction': self.use_gradient_direction,
            'use_second_order': self.use_second_order,
            'normalize_components': self.normalize_components,
            'component_weights': self.component_weights
        }


class DoubleBackpropManager:
    """
    Manages double backpropagation strategy for partial data samples.

    The hypothesis is that partial data (where we know the correct answer)
    should contribute more strongly to learning. This class implements
    strategies to amplify their signal.
    """

    def __init__(
        self,
        strategy: str = 'weighted',
        partial_weight: float = 2.0,
        use_accumulated_gradients: bool = True,
        gradient_accumulation_steps: int = 2
    ):
        """
        Initialize double backprop manager.

        Args:
            strategy: 'weighted', 'repeated', 'accumulated', or 'adaptive'
            partial_weight: Weight multiplier for partial data gradients
            use_accumulated_gradients: Accumulate gradients before update
            gradient_accumulation_steps: Number of accumulation steps
        """
        self.strategy = strategy
        self.partial_weight = partial_weight
        self.use_accumulated_gradients = use_accumulated_gradients
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.accumulated_grads = None
        self.accumulation_count = 0

    def process_partial_gradients(
        self,
        partial_grads: List[tuple],
        hypothesis_grads: List[tuple],
        epoch: int,
        batch_idx: int
    ) -> Tuple[List[tuple], np.ndarray]:
        """
        Process partial data gradients according to strategy.

        Args:
            partial_grads: Gradients from partial (known correct) data
            hypothesis_grads: Gradients from hypothesis data
            epoch: Current epoch
            batch_idx: Current batch index

        Returns:
            Tuple of (processed gradients, confidence weights)
        """
        if self.strategy == 'weighted':
            return self._weighted_strategy(partial_grads, hypothesis_grads)
        elif self.strategy == 'repeated':
            return self._repeated_strategy(partial_grads, hypothesis_grads)
        elif self.strategy == 'accumulated':
            return self._accumulated_strategy(partial_grads, hypothesis_grads, batch_idx)
        elif self.strategy == 'adaptive':
            return self._adaptive_strategy(partial_grads, hypothesis_grads, epoch)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _weighted_strategy(
        self,
        partial_grads: List[tuple],
        hypothesis_grads: List[tuple]
    ) -> Tuple[List[tuple], np.ndarray]:
        """
        Weight partial gradients more heavily.
        """
        # Create weights: higher for partial data
        n_partial = len(partial_grads)
        n_hypothesis = len(hypothesis_grads)

        weights = np.concatenate([
            np.ones(n_hypothesis),  # Normal weight for hypothesis
            np.ones(n_partial) * self.partial_weight  # Higher weight for partial
        ])

        combined_grads = hypothesis_grads + partial_grads
        return combined_grads, weights

    def _repeated_strategy(
        self,
        partial_grads: List[tuple],
        hypothesis_grads: List[tuple]
    ) -> Tuple[List[tuple], np.ndarray]:
        """
        Repeat partial gradients multiple times (effective double backprop).
        """
        n_repeats = int(self.partial_weight)
        repeated_partial = partial_grads * n_repeats

        combined_grads = hypothesis_grads + repeated_partial
        weights = np.ones(len(combined_grads))

        return combined_grads, weights

    def _accumulated_strategy(
        self,
        partial_grads: List[tuple],
        hypothesis_grads: List[tuple],
        batch_idx: int
    ) -> Tuple[List[tuple], np.ndarray]:
        """
        Accumulate partial gradients across batches before applying.
        """
        # Initialize accumulator
        if self.accumulated_grads is None:
            self.accumulated_grads = [
                [torch.zeros_like(g) for g in grad_tuple]
                for grad_tuple in partial_grads
            ]

        # Accumulate
        for i, grad_tuple in enumerate(partial_grads):
            for j, g in enumerate(grad_tuple):
                if i < len(self.accumulated_grads):
                    self.accumulated_grads[i][j] = self.accumulated_grads[i][j] + g

        self.accumulation_count += 1

        # Apply accumulated gradients periodically
        if self.accumulation_count >= self.gradient_accumulation_steps:
            # Average the accumulated gradients
            averaged_grads = [
                tuple(g / self.accumulation_count for g in grad_tuple)
                for grad_tuple in self.accumulated_grads
            ]

            # Reset accumulator
            self.accumulated_grads = None
            self.accumulation_count = 0

            combined_grads = hypothesis_grads + averaged_grads
            weights = np.ones(len(combined_grads))
            return combined_grads, weights
        else:
            # Don't include partial grads yet
            weights = np.ones(len(hypothesis_grads))
            return hypothesis_grads, weights

    def _adaptive_strategy(
        self,
        partial_grads: List[tuple],
        hypothesis_grads: List[tuple],
        epoch: int
    ) -> Tuple[List[tuple], np.ndarray]:
        """
        Adaptively adjust partial weight based on epoch.

        Early epochs: Higher partial weight to establish strong signal
        Later epochs: Lower weight to allow hypothesis selection to dominate
        """
        # Decay partial weight over epochs
        decay_rate = 0.95
        current_weight = self.partial_weight * (decay_rate ** epoch)
        current_weight = max(1.0, current_weight)  # Never go below 1

        n_partial = len(partial_grads)
        n_hypothesis = len(hypothesis_grads)

        weights = np.concatenate([
            np.ones(n_hypothesis),
            np.ones(n_partial) * current_weight
        ])

        combined_grads = hypothesis_grads + partial_grads
        return combined_grads, weights


def compute_gradient_statistics(
    grads: List[tuple],
    layer_idx: int = -2
) -> Dict[str, float]:
    """
    Compute comprehensive statistics about a set of gradients.

    Args:
        grads: List of gradient tuples
        layer_idx: Which layer to analyze

    Returns:
        Dictionary of gradient statistics
    """
    if not grads:
        return {}

    # Extract specified layer gradients
    layer_grads = []
    for grad_tuple in grads:
        g = grad_tuple[layer_idx]
        if isinstance(g, torch.Tensor):
            g = g.detach().cpu().numpy()
        layer_grads.append(g.flatten())

    layer_grads = np.array(layer_grads)

    stats = {
        'mean_magnitude': np.mean([np.linalg.norm(g) for g in layer_grads]),
        'std_magnitude': np.std([np.linalg.norm(g) for g in layer_grads]),
        'mean_per_dim': np.mean(layer_grads, axis=0).tolist()[:10],  # First 10
        'std_per_dim': np.std(layer_grads, axis=0).tolist()[:10],
        'max_gradient': np.max(np.abs(layer_grads)),
        'gradient_sparsity': np.mean(np.abs(layer_grads) < 1e-6),
        'gradient_dim': layer_grads.shape[1]
    }

    return stats


def visualize_gradient_space(
    correct_grads: np.ndarray,
    incorrect_grads: np.ndarray,
    partial_grads: np.ndarray = None,
    method: str = 'tsne',
    save_path: str = None,
    title: str = None
):
    """
    Visualize gradient distributions in 2D space.

    Args:
        correct_grads: Gradients from correct hypotheses
        incorrect_grads: Gradients from incorrect hypotheses
        partial_grads: Gradients from partial (known) data
        method: 'tsne' or 'pca'
        save_path: Path to save figure
        title: Plot title
    """
    # Combine all gradients
    all_grads = [correct_grads, incorrect_grads]
    labels = ['Correct (unknown)', 'Incorrect (unknown)']
    colors = ['green', 'red']

    if partial_grads is not None and len(partial_grads) > 0:
        all_grads.append(partial_grads)
        labels.append('Partial (known)')
        colors.append('blue')

    combined = np.vstack(all_grads)

    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined)-1))
    else:
        reducer = PCA(n_components=2)

    reduced = reducer.fit_transform(combined)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    start_idx = 0
    for grads, label, color in zip(all_grads, labels, colors):
        end_idx = start_idx + len(grads)
        ax.scatter(
            reduced[start_idx:end_idx, 0],
            reduced[start_idx:end_idx, 1],
            c=color,
            label=label,
            alpha=0.6,
            s=50
        )
        start_idx = end_idx

    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(title or f'Gradient Space Visualization ({method.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    plt.close()
