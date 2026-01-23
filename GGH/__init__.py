"""
GGH: Gradient Guided Hypotheses

A framework for handling missing data through gradient-based hypothesis selection.

Modules:
- data_ops: Data loading and preprocessing
- selection_algorithms: Core gradient selection algorithms
- enhanced_selection: Advanced selection methods (contrastive, ensemble, etc.)
- gradient_diagnostics: Diagnostic tools for gradient analysis
- experiment_runner: Systematic experimentation framework
- models: Neural network architectures
- train_val_loop: Training and validation management
- inspector: Results inspection and visualization
- custom_optimizer: Custom optimizers for gradient-guided learning
- imputation_methods: Traditional imputation baselines
"""

from .data_ops import DataOperator
from .selection_algorithms import AlgoModulators, gradient_selection, compute_individual_grads
from .models import initialize_model, load_model, MLP, TabPFNWrapper
from .train_val_loop import TrainValidationManager
from .inspector import Inspector
from .custom_optimizer import CustomAdam

# New enhanced modules
from .gradient_diagnostics import (
    GradientDiagnostics,
    EnrichedVectorBuilder,
    DoubleBackpropManager,
    compute_gradient_statistics,
    visualize_gradient_space
)

from .enhanced_selection import (
    ContrastiveSelector,
    EnsembleSelector,
    CentroidDistanceSelector,
    AdaptiveSelector,
    MultiScaleSelector,
    gradient_selection_enhanced,
    SelectionResult
)

from .experiment_runner import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    create_high_missing_config
)

__version__ = "1.1.0"
__author__ = "GGH Team"
