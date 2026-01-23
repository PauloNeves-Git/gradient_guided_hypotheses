"""
Enhanced Gradient Selection Methods for GGH

This module provides multiple strategies for separating correct from incorrect
hypothesis gradients, going beyond the original OneClassSVM approach.

Key methods:
1. ContrastiveSelector: Uses both correct and incorrect partial data
2. EnsembleSelector: Combines multiple selection algorithms
3. MetricLearningSelector: Learns a distance metric from partial data
4. AdaptiveSelector: Adjusts selection strategy based on epoch/performance
"""

import numpy as np
import torch
from sklearn.svm import OneClassSVM, SVC
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import warnings


@dataclass
class SelectionResult:
    """Container for selection results."""
    selected_indices: np.ndarray
    confidence_scores: np.ndarray
    selection_method: str
    diagnostic_info: Dict[str, Any]


class BaseSelector:
    """Base class for gradient selectors."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.is_fitted = False

    def fit(
        self,
        partial_correct_grads: np.ndarray,
        partial_incorrect_grads: np.ndarray = None
    ):
        """Fit the selector on partial (known) data."""
        raise NotImplementedError

    def predict(self, unknown_grads: np.ndarray) -> SelectionResult:
        """Predict which gradients are from correct hypotheses."""
        raise NotImplementedError


class ContrastiveSelector(BaseSelector):
    """
    Uses both correct AND incorrect partial data for contrastive learning.

    Unlike OneClassSVM which only learns from positive examples,
    this method learns the boundary between correct and incorrect.
    """

    def __init__(
        self,
        classifier: str = 'svm',
        kernel: str = 'rbf',
        C: float = 1.0,
        class_weight: str = 'balanced',
        device: str = "cpu"
    ):
        super().__init__(device)
        self.classifier_type = classifier
        self.kernel = kernel
        self.C = C
        self.class_weight = class_weight
        self.scaler = StandardScaler()
        self.classifier = None

    def fit(
        self,
        partial_correct_grads: np.ndarray,
        partial_incorrect_grads: np.ndarray = None
    ):
        """
        Fit contrastive classifier.

        Args:
            partial_correct_grads: Gradients from known correct hypotheses
            partial_incorrect_grads: Gradients from known incorrect hypotheses
        """
        if partial_incorrect_grads is None or len(partial_incorrect_grads) == 0:
            warnings.warn("ContrastiveSelector requires incorrect examples. Falling back to OneClassSVM.")
            self.classifier = OneClassSVM(kernel=self.kernel, nu=0.1)
            X_scaled = self.scaler.fit_transform(partial_correct_grads)
            self.classifier.fit(X_scaled)
        else:
            # Combine correct and incorrect with labels
            X = np.vstack([partial_correct_grads, partial_incorrect_grads])
            y = np.array([1] * len(partial_correct_grads) + [0] * len(partial_incorrect_grads))

            X_scaled = self.scaler.fit_transform(X)

            if self.classifier_type == 'svm':
                self.classifier = SVC(
                    kernel=self.kernel,
                    C=self.C,
                    class_weight=self.class_weight,
                    probability=True
                )
            elif self.classifier_type == 'rf':
                self.classifier = RandomForestClassifier(
                    n_estimators=100,
                    class_weight=self.class_weight,
                    random_state=42
                )
            elif self.classifier_type == 'knn':
                self.classifier = KNeighborsClassifier(
                    n_neighbors=min(5, len(X) - 1),
                    weights='distance'
                )
            else:
                raise ValueError(f"Unknown classifier: {self.classifier_type}")

            self.classifier.fit(X_scaled, y)

        self.is_fitted = True

    def predict(self, unknown_grads: np.ndarray) -> SelectionResult:
        """
        Predict correct/incorrect for unknown gradients.
        """
        if not self.is_fitted:
            raise RuntimeError("Selector not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(unknown_grads)

        # Get predictions
        if isinstance(self.classifier, OneClassSVM):
            predictions = self.classifier.predict(X_scaled)
            predictions = (predictions == 1).astype(int)
            # Decision function as confidence
            decision = self.classifier.decision_function(X_scaled)
            confidence = 1 / (1 + np.exp(-decision))
        else:
            predictions = self.classifier.predict(X_scaled)
            if hasattr(self.classifier, 'predict_proba'):
                confidence = self.classifier.predict_proba(X_scaled)[:, 1]
            else:
                confidence = predictions.astype(float)

        selected_indices = np.where(predictions == 1)[0]

        return SelectionResult(
            selected_indices=selected_indices,
            confidence_scores=confidence,
            selection_method='contrastive',
            diagnostic_info={
                'n_selected': len(selected_indices),
                'selection_rate': len(selected_indices) / len(unknown_grads),
                'mean_confidence': np.mean(confidence[selected_indices]) if len(selected_indices) > 0 else 0
            }
        )


class EnsembleSelector(BaseSelector):
    """
    Combines multiple selection algorithms through voting or averaging.

    Methods included:
    - OneClassSVM
    - Isolation Forest
    - Local Outlier Factor
    - Elliptic Envelope (Mahalanobis distance)
    """

    def __init__(
        self,
        methods: List[str] = None,
        voting: str = 'soft',
        threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Args:
            methods: List of methods to ensemble. Default: all available
            voting: 'hard' (majority) or 'soft' (probability averaging)
            threshold: Threshold for soft voting
        """
        super().__init__(device)
        self.methods = methods or ['ocsvm', 'iforest', 'lof', 'elliptic']
        self.voting = voting
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.estimators = {}

    def fit(
        self,
        partial_correct_grads: np.ndarray,
        partial_incorrect_grads: np.ndarray = None
    ):
        """Fit all ensemble members on correct partial data."""
        X_scaled = self.scaler.fit_transform(partial_correct_grads)

        # Estimate contamination from incorrect samples if available
        if partial_incorrect_grads is not None and len(partial_incorrect_grads) > 0:
            contamination = len(partial_incorrect_grads) / (
                len(partial_correct_grads) + len(partial_incorrect_grads)
            )
            contamination = min(max(contamination, 0.01), 0.5)
        else:
            contamination = 0.1

        for method in self.methods:
            if method == 'ocsvm':
                est = OneClassSVM(kernel='rbf', nu=contamination)
            elif method == 'iforest':
                est = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=100
                )
            elif method == 'lof':
                est = LocalOutlierFactor(
                    contamination=contamination,
                    novelty=True,
                    n_neighbors=min(20, len(X_scaled) - 1)
                )
            elif method == 'elliptic':
                try:
                    est = EllipticEnvelope(contamination=contamination, random_state=42)
                except:
                    continue
            else:
                continue

            try:
                est.fit(X_scaled)
                self.estimators[method] = est
            except Exception as e:
                warnings.warn(f"Failed to fit {method}: {e}")

        self.is_fitted = True

    def predict(self, unknown_grads: np.ndarray) -> SelectionResult:
        """Ensemble prediction with voting."""
        if not self.is_fitted:
            raise RuntimeError("Selector not fitted.")

        X_scaled = self.scaler.transform(unknown_grads)
        n_samples = len(X_scaled)

        # Collect predictions and scores from each method
        all_predictions = []
        all_scores = []

        for method, est in self.estimators.items():
            pred = est.predict(X_scaled)
            pred = (pred == 1).astype(int)
            all_predictions.append(pred)

            # Get anomaly score
            if hasattr(est, 'decision_function'):
                score = est.decision_function(X_scaled)
            elif hasattr(est, 'score_samples'):
                score = est.score_samples(X_scaled)
            else:
                score = pred.astype(float)

            # Normalize to 0-1
            score = (score - score.min()) / (score.max() - score.min() + 1e-10)
            all_scores.append(score)

        all_predictions = np.array(all_predictions)
        all_scores = np.array(all_scores)

        if self.voting == 'hard':
            # Majority voting
            votes = np.sum(all_predictions, axis=0)
            final_predictions = (votes > len(self.estimators) / 2).astype(int)
            confidence = votes / len(self.estimators)
        else:
            # Soft voting (average probabilities)
            avg_scores = np.mean(all_scores, axis=0)
            final_predictions = (avg_scores >= self.threshold).astype(int)
            confidence = avg_scores

        selected_indices = np.where(final_predictions == 1)[0]

        return SelectionResult(
            selected_indices=selected_indices,
            confidence_scores=confidence,
            selection_method='ensemble',
            diagnostic_info={
                'n_selected': len(selected_indices),
                'methods_used': list(self.estimators.keys()),
                'individual_selection_rates': {
                    m: np.mean(all_predictions[i])
                    for i, m in enumerate(self.estimators.keys())
                }
            }
        )


class CentroidDistanceSelector(BaseSelector):
    """
    Selects based on distance to correct vs incorrect centroids.

    Simple but interpretable: closer to correct centroid = more likely correct.
    """

    def __init__(
        self,
        metric: str = 'euclidean',
        use_mahalanobis: bool = False,
        margin: float = 0.0,
        device: str = "cpu"
    ):
        super().__init__(device)
        self.metric = metric
        self.use_mahalanobis = use_mahalanobis
        self.margin = margin
        self.scaler = StandardScaler()
        self.correct_centroid = None
        self.incorrect_centroid = None
        self.covariance_inv = None

    def fit(
        self,
        partial_correct_grads: np.ndarray,
        partial_incorrect_grads: np.ndarray = None
    ):
        """Compute centroids."""
        X_scaled = self.scaler.fit_transform(partial_correct_grads)
        self.correct_centroid = np.mean(X_scaled, axis=0)

        if partial_incorrect_grads is not None and len(partial_incorrect_grads) > 0:
            inc_scaled = self.scaler.transform(partial_incorrect_grads)
            self.incorrect_centroid = np.mean(inc_scaled, axis=0)
        else:
            # If no incorrect examples, use a point far from correct centroid
            self.incorrect_centroid = None

        if self.use_mahalanobis:
            cov = np.cov(X_scaled.T)
            try:
                self.covariance_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
            except:
                self.covariance_inv = None

        self.is_fitted = True

    def predict(self, unknown_grads: np.ndarray) -> SelectionResult:
        """Select based on centroid distances."""
        X_scaled = self.scaler.transform(unknown_grads)

        # Distance to correct centroid
        if self.use_mahalanobis and self.covariance_inv is not None:
            diff = X_scaled - self.correct_centroid
            dist_correct = np.sqrt(np.sum(diff @ self.covariance_inv * diff, axis=1))
        else:
            dist_correct = np.linalg.norm(X_scaled - self.correct_centroid, axis=1)

        if self.incorrect_centroid is not None:
            if self.use_mahalanobis and self.covariance_inv is not None:
                diff = X_scaled - self.incorrect_centroid
                dist_incorrect = np.sqrt(np.sum(diff @ self.covariance_inv * diff, axis=1))
            else:
                dist_incorrect = np.linalg.norm(X_scaled - self.incorrect_centroid, axis=1)

            # Select if closer to correct than incorrect (with margin)
            predictions = (dist_incorrect > dist_correct + self.margin).astype(int)
            # Confidence based on relative distance
            total_dist = dist_correct + dist_incorrect + 1e-10
            confidence = dist_incorrect / total_dist
        else:
            # Select based on threshold distance to correct centroid
            threshold = np.percentile(dist_correct, 75)
            predictions = (dist_correct <= threshold).astype(int)
            confidence = 1 - (dist_correct / (np.max(dist_correct) + 1e-10))

        selected_indices = np.where(predictions == 1)[0]

        return SelectionResult(
            selected_indices=selected_indices,
            confidence_scores=confidence,
            selection_method='centroid_distance',
            diagnostic_info={
                'n_selected': len(selected_indices),
                'mean_dist_to_correct': np.mean(dist_correct[selected_indices]) if len(selected_indices) > 0 else 0,
                'selection_rate': len(selected_indices) / len(unknown_grads)
            }
        )


class AdaptiveSelector(BaseSelector):
    """
    Adapts selection strategy based on training dynamics.

    Early epochs: Permissive selection (more exploration)
    Later epochs: Stricter selection (more exploitation)

    Also monitors selection quality and adjusts parameters.
    """

    def __init__(
        self,
        base_selector: str = 'ocsvm',
        initial_nu: float = 0.3,
        final_nu: float = 0.1,
        warmup_epochs: int = 5,
        decay_type: str = 'linear',
        device: str = "cpu"
    ):
        super().__init__(device)
        self.base_selector = base_selector
        self.initial_nu = initial_nu
        self.final_nu = final_nu
        self.warmup_epochs = warmup_epochs
        self.decay_type = decay_type
        self.current_epoch = 0
        self.scaler = StandardScaler()
        self.selector = None
        self.selection_history = []

    def fit(
        self,
        partial_correct_grads: np.ndarray,
        partial_incorrect_grads: np.ndarray = None,
        epoch: int = 0
    ):
        """Fit selector with epoch-dependent parameters."""
        self.current_epoch = epoch
        X_scaled = self.scaler.fit_transform(partial_correct_grads)

        # Compute current nu based on epoch
        current_nu = self._compute_current_nu(epoch)

        if self.base_selector == 'ocsvm':
            self.selector = OneClassSVM(kernel='rbf', nu=current_nu)
        elif self.base_selector == 'iforest':
            contamination = 1 - current_nu  # Inverse relationship
            self.selector = IsolationForest(contamination=max(0.01, min(0.5, contamination)))
        else:
            self.selector = OneClassSVM(kernel='rbf', nu=current_nu)

        self.selector.fit(X_scaled)
        self.is_fitted = True

    def _compute_current_nu(self, epoch: int) -> float:
        """Compute nu value for current epoch."""
        if epoch < self.warmup_epochs:
            # Use initial (more permissive) nu during warmup
            return self.initial_nu

        # After warmup, decay towards final nu
        progress = min(1.0, (epoch - self.warmup_epochs) / (50 - self.warmup_epochs))

        if self.decay_type == 'linear':
            return self.initial_nu + progress * (self.final_nu - self.initial_nu)
        elif self.decay_type == 'exponential':
            return self.final_nu + (self.initial_nu - self.final_nu) * np.exp(-3 * progress)
        elif self.decay_type == 'cosine':
            return self.final_nu + 0.5 * (self.initial_nu - self.final_nu) * (1 + np.cos(np.pi * progress))
        else:
            return self.initial_nu + progress * (self.final_nu - self.initial_nu)

    def predict(self, unknown_grads: np.ndarray) -> SelectionResult:
        """Predict with current adaptive settings."""
        if not self.is_fitted:
            raise RuntimeError("Selector not fitted.")

        X_scaled = self.scaler.transform(unknown_grads)

        predictions = self.selector.predict(X_scaled)
        predictions = (predictions == 1).astype(int)

        if hasattr(self.selector, 'decision_function'):
            decision = self.selector.decision_function(X_scaled)
            confidence = 1 / (1 + np.exp(-decision))
        else:
            confidence = predictions.astype(float)

        selected_indices = np.where(predictions == 1)[0]
        selection_rate = len(selected_indices) / len(unknown_grads)

        # Track selection rate for monitoring
        self.selection_history.append({
            'epoch': self.current_epoch,
            'selection_rate': selection_rate,
            'nu': self._compute_current_nu(self.current_epoch)
        })

        return SelectionResult(
            selected_indices=selected_indices,
            confidence_scores=confidence,
            selection_method='adaptive',
            diagnostic_info={
                'n_selected': len(selected_indices),
                'selection_rate': selection_rate,
                'current_nu': self._compute_current_nu(self.current_epoch),
                'epoch': self.current_epoch
            }
        )


class MultiScaleSelector(BaseSelector):
    """
    Performs selection at multiple gradient scales/layers.

    Different layers capture different aspects of the learning signal.
    This selector combines signals from multiple layers.
    """

    def __init__(
        self,
        layer_indices: List[int] = [-2, -1],
        layer_weights: List[float] = None,
        aggregation: str = 'weighted_vote',
        device: str = "cpu"
    ):
        super().__init__(device)
        self.layer_indices = layer_indices
        self.layer_weights = layer_weights or [1.0] * len(layer_indices)
        self.aggregation = aggregation
        self.layer_selectors = {}
        self.layer_scalers = {}

    def fit_layer(
        self,
        layer_idx: int,
        partial_correct_grads: np.ndarray,
        partial_incorrect_grads: np.ndarray = None
    ):
        """Fit selector for a specific layer."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(partial_correct_grads)

        selector = OneClassSVM(kernel='rbf', nu=0.1)
        selector.fit(X_scaled)

        self.layer_selectors[layer_idx] = selector
        self.layer_scalers[layer_idx] = scaler

    def predict_multilayer(
        self,
        layer_grads: Dict[int, np.ndarray]
    ) -> SelectionResult:
        """
        Predict using multiple layers.

        Args:
            layer_grads: Dict mapping layer index to gradient array
        """
        n_samples = len(list(layer_grads.values())[0])
        all_scores = []

        for layer_idx, grads in layer_grads.items():
            if layer_idx not in self.layer_selectors:
                continue

            scaler = self.layer_scalers[layer_idx]
            selector = self.layer_selectors[layer_idx]
            weight = self.layer_weights[self.layer_indices.index(layer_idx)]

            X_scaled = scaler.transform(grads)
            if hasattr(selector, 'decision_function'):
                scores = selector.decision_function(X_scaled)
            else:
                scores = selector.predict(X_scaled).astype(float)

            # Normalize scores
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            all_scores.append(scores * weight)

        # Aggregate scores
        combined_scores = np.sum(all_scores, axis=0) / sum(self.layer_weights)

        # Threshold at 0.5
        predictions = (combined_scores >= 0.5).astype(int)
        selected_indices = np.where(predictions == 1)[0]

        return SelectionResult(
            selected_indices=selected_indices,
            confidence_scores=combined_scores,
            selection_method='multiscale',
            diagnostic_info={
                'n_selected': len(selected_indices),
                'layers_used': list(layer_grads.keys()),
                'selection_rate': len(selected_indices) / n_samples
            }
        )


def gradient_selection_enhanced(
    DO,
    AM,
    epoch: int,
    hypothesis_grads: List[tuple],
    partial_correct_grads: List[tuple],
    partial_incorrect_grads: List[tuple] = None,
    batch_size: int = 100,
    hyp_per_sample: int = 4,
    batch_i: int = 0,
    inputs: torch.Tensor = None,
    labels: torch.Tensor = None,
    individual_losses: torch.Tensor = None,
    selector_type: str = 'ensemble',
    selector_config: Dict = None
) -> Tuple[List[tuple], List[int], np.ndarray]:
    """
    Enhanced gradient selection with multiple methods.

    Args:
        DO: DataOperator
        AM: AlgoModulators
        epoch: Current epoch
        hypothesis_grads: Gradients from all hypotheses
        partial_correct_grads: Gradients from known correct partial data
        partial_incorrect_grads: Gradients from known incorrect partial data
        batch_size: Batch size
        hyp_per_sample: Number of hypotheses per sample
        batch_i: Batch index
        inputs: Input tensor
        labels: Label tensor
        individual_losses: Individual loss values
        selector_type: 'ocsvm', 'contrastive', 'ensemble', 'centroid', 'adaptive'
        selector_config: Configuration dict for selector

    Returns:
        Tuple of (selected_gradients, selected_indices, confidence_weights)
    """
    selector_config = selector_config or {}

    # Extract gradient vectors for specified layer
    layer = AM.layer if hasattr(AM, 'layer') else -2

    def extract_layer_grads(grads_list):
        """Extract and flatten gradients from specified layer."""
        vectors = []
        for grad_tuple in grads_list:
            g = grad_tuple[layer]
            if isinstance(g, torch.Tensor):
                g = g.detach().cpu().numpy()
            vectors.append(g.flatten())
        return np.array(vectors) if vectors else np.array([]).reshape(0, 0)

    # Build enriched vectors with context
    def build_enriched_vectors(grads_list, inputs_list=None, losses_list=None):
        """Build enriched vectors with gradients + context."""
        vectors = extract_layer_grads(grads_list)
        if len(vectors) == 0:
            return vectors

        # Add input context if available
        if inputs_list is not None and AM.gradwcontext:
            if isinstance(inputs_list, torch.Tensor):
                inputs_np = inputs_list.detach().cpu().numpy()
            else:
                inputs_np = np.array(inputs_list)
            if len(inputs_np) == len(vectors):
                vectors = np.hstack([vectors, inputs_np])

        # Add loss context if available and epoch > threshold
        if losses_list is not None and epoch > AM.epoch_loss_in_contxt:
            if isinstance(losses_list, torch.Tensor):
                losses_np = losses_list.detach().cpu().numpy()
            else:
                losses_np = np.array(losses_list)
            if len(losses_np) == len(vectors):
                if losses_np.ndim == 1:
                    losses_np = losses_np.reshape(-1, 1)
                vectors = np.hstack([vectors, losses_np])

        return vectors

    # Organize gradients by hypothesis class
    selected_gradients = []
    selected_global_ids = []
    all_confidence_weights = []

    for class_id in range(hyp_per_sample):
        # Extract gradients for this hypothesis class
        class_grads = hypothesis_grads[class_id::hyp_per_sample]
        class_inputs = inputs[class_id::hyp_per_sample] if inputs is not None else None
        class_losses = individual_losses[class_id::hyp_per_sample] if individual_losses is not None else None

        # Extract partial grads for this class
        partial_correct_class = [
            g for g, cls in zip(partial_correct_grads, DO.true_partial_hyp_class)
            if cls == class_id
        ]
        partial_incorrect_class = []
        if partial_incorrect_grads:
            partial_incorrect_class = [
                g for g, cls in zip(partial_incorrect_grads, DO.inc_partial_hyp_class)
                if cls == class_id
            ]

        # Skip if no partial data for this class
        if len(partial_correct_class) == 0:
            continue

        # Build enriched vectors
        unknown_vectors = build_enriched_vectors(class_grads, class_inputs, class_losses)
        partial_correct_vectors = extract_layer_grads(partial_correct_class)
        partial_incorrect_vectors = extract_layer_grads(partial_incorrect_class) if partial_incorrect_class else None

        if len(unknown_vectors) == 0:
            continue

        # Normalize if configured
        if AM.normalize_grads_contx:
            combined = np.vstack([partial_correct_vectors, unknown_vectors])
            mean = np.mean(combined, axis=0)
            std = np.std(combined, axis=0) + 1e-10
            partial_correct_vectors = (partial_correct_vectors - mean) / std
            unknown_vectors = (unknown_vectors - mean) / std
            if partial_incorrect_vectors is not None and len(partial_incorrect_vectors) > 0:
                partial_incorrect_vectors = (partial_incorrect_vectors - mean) / std

        # Initialize selector based on type
        if selector_type == 'contrastive':
            selector = ContrastiveSelector(**selector_config)
        elif selector_type == 'ensemble':
            selector = EnsembleSelector(**selector_config)
        elif selector_type == 'centroid':
            selector = CentroidDistanceSelector(**selector_config)
        elif selector_type == 'adaptive':
            selector = AdaptiveSelector(**selector_config)
        else:  # Default to enhanced OneClassSVM
            selector = ContrastiveSelector(classifier='svm', **selector_config)

        # Fit and predict
        selector.fit(partial_correct_vectors, partial_incorrect_vectors)
        result = selector.predict(unknown_vectors)

        # Map selected indices back to global IDs and full gradients
        for local_idx in result.selected_indices:
            global_hyp_id = batch_i * batch_size + local_idx * hyp_per_sample + class_id
            full_grad = class_grads[local_idx]

            # Apply frequency filtering if enabled
            if epoch > AM.sel_freq_crit_start_after:
                DO.df_train_hypothesis.at[global_hyp_id, 'sel_hyp_tracker'].append(1)
                freq_sum = sum(DO.df_train_hypothesis.sel_hyp_tracker.iloc[global_hyp_id])
                if freq_sum >= epoch * AM.freqperc_cutoff:
                    selected_gradients.append(full_grad)
                    selected_global_ids.append(global_hyp_id)
                    all_confidence_weights.append(result.confidence_scores[local_idx])
                    DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(1)
                else:
                    DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(0)
            else:
                DO.df_train_hypothesis.at[global_hyp_id, 'sel_hyp_tracker'].append(1)
                DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(1)
                selected_gradients.append(full_grad)
                selected_global_ids.append(global_hyp_id)
                all_confidence_weights.append(result.confidence_scores[local_idx])

        # Mark non-selected as 0
        for local_idx in range(len(class_grads)):
            if local_idx not in result.selected_indices:
                global_hyp_id = batch_i * batch_size + local_idx * hyp_per_sample + class_id
                DO.df_train_hypothesis.at[global_hyp_id, 'sel_hyp_tracker'].append(0)
                DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(0)

    confidence_weights = np.array(all_confidence_weights) if all_confidence_weights else None

    return selected_gradients, selected_global_ids, confidence_weights
