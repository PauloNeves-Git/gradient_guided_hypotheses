# GGH_2: Gradient-Guided Hypotheses Package
# Refactored from benchmark notebooks for code reuse

from .models import HypothesisAmplifyingModel
from .trainers import UnbiasedTrainer, WeightedTrainer, RemainingDataScorer
from .utils import set_to_deterministic, create_dataloader_with_gids, evaluate_on_test
from .scoring import compute_anchor_data, compute_enriched_score, compute_soft_weights
from .training import train_with_soft_weights
from .ggh import run_ggh_soft_refinement
from .tabpfn import get_tabpfn_probabilities
from .benchmark_utils import (
    SimpleRegressionMLP,
    run_imputation_method,
    train_ggh_model,
    train_full_info_model,
    train_partial_model,
)
from .benchmark_viz import (
    plot_r2_comparison,
    plot_mse_comparison,
    plot_mae_comparison,
    plot_all_metrics,
)
from .noise_detection import (
    SimpleNoiseDetectionModel,
    evaluate_detection,
    train_on_cleaned_data,
    run_full_info,
    run_full_info_noisy,
    run_old_ggh_dbscan,
    run_new_ggh_unsupervised,
)
from .noise_detection_viz import (
    plot_noise_detection_r2,
    plot_noise_detection_mse,
    plot_noise_detection_mae,
    plot_detection_metrics,
    plot_all_noise_detection_metrics,
)

__all__ = [
    'HypothesisAmplifyingModel',
    'UnbiasedTrainer',
    'WeightedTrainer',
    'RemainingDataScorer',
    'set_to_deterministic',
    'create_dataloader_with_gids',
    'evaluate_on_test',
    'compute_anchor_data',
    'compute_enriched_score',
    'compute_soft_weights',
    'train_with_soft_weights',
    'run_ggh_soft_refinement',
    'get_tabpfn_probabilities',
    # Benchmark utilities
    'SimpleRegressionMLP',
    'run_imputation_method',
    'train_ggh_model',
    'train_full_info_model',
    'train_partial_model',
    # Benchmark visualization
    'plot_r2_comparison',
    'plot_mse_comparison',
    'plot_mae_comparison',
    'plot_all_metrics',
    # Noise detection utilities
    'SimpleNoiseDetectionModel',
    'evaluate_detection',
    'train_on_cleaned_data',
    'run_full_info',
    'run_full_info_noisy',
    'run_old_ggh_dbscan',
    'run_new_ggh_unsupervised',
    # Noise detection visualization
    'plot_noise_detection_r2',
    'plot_noise_detection_mse',
    'plot_noise_detection_mae',
    'plot_detection_metrics',
    'plot_all_noise_detection_metrics',
]
