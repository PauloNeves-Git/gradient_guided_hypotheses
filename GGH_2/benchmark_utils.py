"""Benchmark utilities for GGH vs imputation method comparisons."""

import contextlib
import io
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .models import HypothesisAmplifyingModel
from .utils import set_to_deterministic
from .training import train_with_soft_weights
from .utils import evaluate_on_test


class SimpleRegressionMLP(nn.Module):
    """Simple MLP for regression - used for imputation baselines."""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


def run_imputation_method(DO, method_name, r_state, hypothesis, lr, n_epochs=200,
                          batch_size=64, hidden_size=32, results_path=""):
    """Run a single imputation method and return performance metrics.

    This function replicates the original Wine_Benchmark.ipynb imputation pipeline exactly:
    1. Use Imputer.impute_w_sel() to create dataloader with imputed data
    2. Use initialize_model() to create MLP (same as original)
    3. Use TrainValidationManager for training (same as original)
    4. Use Inspector for evaluation (same as original)

    Args:
        DO: DataOperator instance
        method_name: Name of imputation method (e.g., "Soft Impute", "Miss Forest")
        r_state: Random state for reproducibility
        hypothesis: List of hypothesis values (e.g., [[9.4, 10.5, 12.0]])
        lr: Learning rate for training
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        hidden_size: Hidden layer size for MLP
        results_path: Path for saving results (can be empty string)

    Returns:
        tuple: ((mse, mae, r2), error_message) - error_message is None if successful
    """
    # Import here to avoid circular imports - using same classes as original benchmark
    from GGH.imputation_methods import Imputer
    from GGH.models import initialize_model
    from GGH.train_val_loop import TrainValidationManager
    from GGH.selection_algorithms import AlgoModulators
    from GGH.inspector import Inspector
    import os

    try:
        # Create Imputer and check if method is available
        imputer = Imputer(DO)

        if method_name not in imputer.imputers:
            return None, f"Method {method_name} not available"

        # Get imputed dataloader - same as original: IMP.impute_w_sel(DO_imp, imput_method, batch_size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                dataloader_imp = imputer.impute_w_sel(DO, method_name, batch_size)

        # Create AlgoModulators - same as original: AM_imp = AlgoModulators(DO_imp, lr=lr)
        AM_imp = AlgoModulators(DO, lr=lr)

        # Initialize model - same as original: initialize_model(DO_imp, dataloader_imp, hidden_size, r_state, dropout=0.05)
        set_to_deterministic(r_state)
        model_imp = initialize_model(DO, dataloader_imp, hidden_size, r_state, dropout=0.05)

        # Create results directory if needed
        if results_path and not os.path.exists(f"{results_path}/use imputation"):
            os.makedirs(f"{results_path}/use imputation", exist_ok=True)

        # Train with TrainValidationManager - same as original
        TVM_imp = TrainValidationManager(
            "use imputation", n_epochs, dataloader_imp, batch_size, r_state,
            results_path if results_path else ".",
            imput_method=method_name, final_analysis=False
        )
        TVM_imp.train_model(DO, AM_imp, model_imp, final_analysis=False)

        # Evaluate - calculate R2, MAE, and MSE on test set
        INSPECT_imp = Inspector(results_path if results_path else ".", hidden_size)
        imp_r2 = INSPECT_imp.calculate_val_r2score(DO, TVM_imp, model_imp, data="test")
        imp_mae = INSPECT_imp.calculate_val_mse(DO, TVM_imp, model_imp, data="test")  # Note: this actually returns MAE

        # Calculate MSE manually from predictions
        from GGH.inspector import model_predict
        val_tensors = INSPECT_imp._get_val_tensors(DO, TVM_imp, "test")
        val_pred = model_predict(TVM_imp, model_imp, val_tensors)
        imp_mse = mean_squared_error(DO.df_test[DO.target_vars].values, val_pred)

        return (imp_mse, imp_mae, imp_r2), None

    except Exception as e:
        import traceback
        return None, f"{str(e)}\n{traceback.format_exc()}"


def train_ggh_model(DO, ggh_weights, partial_gids, partial_weight, r_state, ggh_config, lr, n_epochs, device):
    """Train model with GGH weights.

    Args:
        DO: DataOperator instance
        ggh_weights: Dictionary mapping GID -> weight
        partial_gids: Set of partial (known correct) GIDs
        partial_weight: Weight multiplier for partial samples
        r_state: Random state for reproducibility
        ggh_config: GGH configuration dictionary with model architecture params
        lr: Learning rate
        n_epochs: Number of training epochs
        device: PyTorch device

    Returns:
        tuple: (mse, mae, r2) - Test set metrics
    """
    n_shared = len(DO.inpt_vars)
    n_hyp = len(DO.miss_vars)
    out_size = len(DO.target_vars)

    set_to_deterministic(r_state + 200)
    model = HypothesisAmplifyingModel(
        n_shared, n_hyp,
        ggh_config['shared_hidden'], ggh_config['hypothesis_hidden'],
        ggh_config['final_hidden'], out_size
    ).to(device)

    model, _, _ = train_with_soft_weights(
        DO, model, ggh_weights, partial_gids,
        partial_weight, lr, n_epochs
    )

    mse, mae, r2 = evaluate_on_test(DO, model)
    return mse, mae, r2


def train_full_info_model(DO, r_state, ggh_config, lr, n_epochs, device):
    """Train model with full information (oracle upper bound).

    Args:
        DO: DataOperator instance
        r_state: Random state
        ggh_config: GGH config with model architecture
        lr: Learning rate
        n_epochs: Number of epochs
        device: PyTorch device

    Returns:
        tuple: (mse, mae, r2) - Test set metrics
    """
    hyp_per_sample = DO.num_hyp_comb
    n_samples = len(DO.df_train_hypothesis) // hyp_per_sample

    full_weights = {}
    for sample_idx in range(n_samples):
        for hyp_idx in range(hyp_per_sample):
            gid = sample_idx * hyp_per_sample + hyp_idx
            if DO.df_train_hypothesis.iloc[gid]['correct_hypothesis']:
                full_weights[gid] = 1.0

    n_shared = len(DO.inpt_vars)
    n_hyp = len(DO.miss_vars)
    out_size = len(DO.target_vars)

    set_to_deterministic(r_state + 600)
    model = HypothesisAmplifyingModel(
        n_shared, n_hyp,
        ggh_config['shared_hidden'], ggh_config['hypothesis_hidden'],
        ggh_config['final_hidden'], out_size
    ).to(device)

    model, _, _ = train_with_soft_weights(DO, model, full_weights, set(), 1.0, lr, n_epochs)
    mse, mae, r2 = evaluate_on_test(DO, model)
    return mse, mae, r2


def train_partial_model(DO, partial_gids, r_state, ggh_config, lr, n_epochs, device):
    """Train model with only partial information (baseline lower bound).

    Args:
        DO: DataOperator instance
        partial_gids: Set of partial (known correct) GIDs
        r_state: Random state
        ggh_config: GGH config with model architecture
        lr: Learning rate
        n_epochs: Number of epochs
        device: PyTorch device

    Returns:
        tuple: (mse, mae, r2) - Test set metrics
    """
    n_shared = len(DO.inpt_vars)
    n_hyp = len(DO.miss_vars)
    out_size = len(DO.target_vars)

    set_to_deterministic(r_state + 400)
    model = HypothesisAmplifyingModel(
        n_shared, n_hyp,
        ggh_config['shared_hidden'], ggh_config['hypothesis_hidden'],
        ggh_config['final_hidden'], out_size
    ).to(device)

    model, _, _ = train_with_soft_weights(DO, model, {}, partial_gids, 1.0, lr, n_epochs)
    mse, mae, r2 = evaluate_on_test(DO, model)
    return mse, mae, r2
