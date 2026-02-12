"""Noise detection utilities for GGH benchmark.

Provides functions for:
- Old GGH: DBSCAN-based noise detection via gradient clustering
- New GGH: Unsupervised soft refinement with bootstrapped anchors
- Helper functions: detection evaluation, model training on cleaned data
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import grad
from copy import deepcopy
from sklearn.cluster import DBSCAN

from GGH.data_ops import DataOperator
from GGH.selection_algorithms import AlgoModulators, compute_individual_grads, MSEIndividualLosses
from GGH.models import initialize_model
from GGH.train_val_loop import TrainValidationManager
from GGH.inspector import get_gradarrays_n_labels


def _ensure_noise_detection_compat(DO):
    """Ensure DataOperator is compatible with noise detection.

    When miss_vars=[] (e.g., noise detection), DataOperator skips hypothesis
    processing and never sets num_hyp_comb. AlgoModulators requires it, so
    we default to 1.

    Also forces regression mode since noise detection always predicts a
    continuous target. Without this, datasets with few unique target values
    (e.g., Wine Quality with integer scores 3-8) get misclassified as
    multi-class by determine_problem_type, causing the MLP to use softmax.
    """
    if not hasattr(DO, 'num_hyp_comb'):
        DO.num_hyp_comb = 1
    DO.problem_type = 'regression'


class SimpleNoiseDetectionModel(nn.Module):
    """Simple MLP for noise detection (used by New GGH)."""
    def __init__(self, n_features, hidden_size=32, output_size=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)


def set_to_deterministic(rand_state):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(rand_state)
    np.random.seed(rand_state)
    torch.manual_seed(rand_state)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def sigmoid_stable(x):
    """Numerically stable sigmoid."""
    x = np.array(x, dtype=np.float64)
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def compute_soft_weights(scores, min_weight=0.1, temperature=1.0):
    """Convert scores to soft weights using sigmoid."""
    scores = np.array(scores, dtype=np.float64)
    if len(scores) == 0:
        return np.array([])

    mean_s = np.mean(scores)
    std_s = np.std(scores) + 1e-8
    normalized = (scores - mean_s) / std_s

    raw_weights = sigmoid_stable(normalized / temperature)
    weights = min_weight + (1 - min_weight) * raw_weights

    return weights


def evaluate_detection(detected_noisy_indices, true_noisy_indices, n_total):
    """Evaluate noise detection performance.

    Args:
        detected_noisy_indices: Indices predicted as noisy
        true_noisy_indices: Ground truth noisy indices
        n_total: Total number of samples

    Returns:
        dict: Detection metrics (precision, recall, accuracy, f1, TP, FP, FN, TN)
    """
    detected_set = set(detected_noisy_indices)
    true_noisy_set = set(true_noisy_indices)
    true_clean_set = set(range(n_total)) - true_noisy_set

    TP = len(detected_set & true_noisy_set)
    FP = len(detected_set & true_clean_set)
    FN = len(true_noisy_set - detected_set)
    TN = len(true_clean_set - detected_set)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    }


def train_on_cleaned_data(DO, df_train_cleaned, r_state, data_path, inpt_vars,
                          target_vars, miss_vars, hypothesis, partial_perc,
                          batch_size, hidden_size, lr, num_epochs, dropout=0.05):
    """Train model on cleaned data (after noise removal).

    Args:
        DO: Original DataOperator (for test data access)
        df_train_cleaned: Cleaned training dataframe
        r_state: Random state
        data_path: Path to dataset
        inpt_vars: Input variable names
        target_vars: Target variable names
        miss_vars: Missing variable names
        hypothesis: Hypothesis values
        partial_perc: Partial percentage
        batch_size: Batch size
        hidden_size: Hidden layer size
        lr: Learning rate
        num_epochs: Number of training epochs
        dropout: Dropout rate

    Returns:
        tuple: (DO_clean, TVM, AM, model)
    """
    use_info = "full info"
    _ensure_noise_detection_compat(DO)
    AM = AlgoModulators(DO, lr=lr)
    DO_clean = DataOperator(data_path, inpt_vars, target_vars, miss_vars, hypothesis,
                            partial_perc, r_state, pre_defined_train=df_train_cleaned,
                            device="cpu")
    _ensure_noise_detection_compat(DO_clean)
    dataloader = DO_clean.prep_dataloader(use_info, batch_size)
    model = initialize_model(DO_clean, dataloader, hidden_size, r_state, dropout=dropout)
    TVM = TrainValidationManager(use_info, num_epochs, dataloader, batch_size, r_state,
                                 ".", best_valid_error=np.inf)
    TVM.train_model(DO_clean, AM, model, final_analysis=False)

    return DO_clean, TVM, AM, model


def evaluate_on_test(DO, model, use_info="full info"):
    """Evaluate model on test set.

    Args:
        DO: DataOperator instance
        model: Trained model
        use_info: Information type for test tensors

    Returns:
        tuple: (r2, mse, mae)
    """
    model.eval()
    with torch.no_grad():
        test_inputs, test_targets = DO.get_test_tensors(use_info=use_info)
        test_preds = model(test_inputs)
        ss_res = torch.sum((test_targets - test_preds) ** 2).item()
        ss_tot = torch.sum((test_targets - test_targets.mean()) ** 2).item()
        test_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        test_mse = torch.nn.functional.mse_loss(test_preds, test_targets).item()
        test_mae = torch.nn.functional.l1_loss(test_preds, test_targets).item()
    return test_r2, test_mse, test_mae


def run_full_info(DO_clean, r_state, batch_size, hidden_size, lr, num_epochs,
                  dropout=0.05, results_path="."):
    """Train on clean data (oracle upper bound).

    Args:
        DO_clean: DataOperator with clean data (no noise)
        r_state: Random state
        batch_size: Batch size
        hidden_size: Hidden layer size
        lr: Learning rate
        num_epochs: Number of epochs
        dropout: Dropout rate
        results_path: Path for saving results

    Returns:
        dict: {'test_r2', 'test_mse', 'test_mae'}
    """
    set_to_deterministic(r_state)
    _ensure_noise_detection_compat(DO_clean)
    AM = AlgoModulators(DO_clean, lr=lr)
    dataloader = DO_clean.prep_dataloader("full info", batch_size)
    model = initialize_model(DO_clean, dataloader, hidden_size, r_state, dropout=dropout)
    TVM = TrainValidationManager("full info", num_epochs, dataloader, batch_size,
                                 r_state, results_path, select_gradients=True,
                                 final_analysis=False)
    TVM.train_model(DO_clean, AM, model, final_analysis=False)

    r2, mse, mae = evaluate_on_test(DO_clean, model)
    return {'test_r2': r2, 'test_mse': mse, 'test_mae': mae}


def run_full_info_noisy(DO, r_state, batch_size, hidden_size, lr, num_epochs,
                        dropout=0.05, results_path="."):
    """Train on noisy data without removal (baseline).

    Args:
        DO: DataOperator with noise simulated
        r_state: Random state
        batch_size: Batch size
        hidden_size: Hidden layer size
        lr: Learning rate
        num_epochs: Number of epochs
        dropout: Dropout rate
        results_path: Path for saving results

    Returns:
        dict: {'test_r2', 'test_mse', 'test_mae'}
    """
    set_to_deterministic(r_state)
    _ensure_noise_detection_compat(DO)
    AM = AlgoModulators(DO, lr=lr)
    dataloader = DO.prep_dataloader("full info noisy", batch_size)
    model = initialize_model(DO, dataloader, hidden_size, r_state, dropout=dropout)
    TVM = TrainValidationManager("full info noisy", num_epochs, dataloader, batch_size,
                                 r_state, results_path, select_gradients=True,
                                 final_analysis=False)
    TVM.train_model(DO, AM, model, final_analysis=False)

    r2, mse, mae = evaluate_on_test(DO, model)
    return {'test_r2': r2, 'test_mse': mse, 'test_mae': mae}


def run_old_ggh_dbscan(DO, r_state, config):
    """Old GGH noise detection using DBSCAN clustering on gradients.

    Pipeline:
    1. Train on noisy data with gradient tracking
    2. Extract gradients at best checkpoint
    3. DBSCAN clustering - outliers flagged as noisy
    4. Grid search over eps and min_samples
    5. Retrain on cleaned data

    Args:
        DO: DataOperator with noise simulated (used for test evaluation)
        r_state: Random state
        config: Dictionary with keys:
            - data_path, inpt_vars, target_vars, miss_vars, hypothesis, partial_perc
            - batch_size, hidden_size, lr, dropout
            - old_ggh_epochs, old_ggh_end_epochs
            - old_ggh_eps_values, old_ggh_min_samples_ratios
            - final_epochs
            - noise_perc, noise_min, noise_max (noise simulation parameters)

    Returns:
        dict: {'test_r2', 'test_mse', 'test_mae', 'detection', 'n_detected', 'eps', 'min_samples_ratio'}
    """
    # Must use "known info noisy simulation" — the training loop only stores
    # per-sample gradients to DO.df_train_noisy under this use_info value.
    use_info = "known info noisy simulation"
    num_epochs = config['old_ggh_epochs'] + config['old_ggh_end_epochs']
    noise_perc = config['noise_perc']
    noise_min = config['noise_min']
    noise_max = config['noise_max']

    best_result = None
    best_val_error = np.inf

    for eps in config['old_ggh_eps_values']:
        for min_samples_ratio in config['old_ggh_min_samples_ratios']:
            set_to_deterministic(r_state)

            # Re-create DO each iteration — training appends gradients to
            # df_train_noisy in place, so we need a fresh copy each time.
            DO = DataOperator(config['data_path'], config['inpt_vars'],
                              config['target_vars'], config['miss_vars'],
                              config['hypothesis'], config['partial_perc'],
                              r_state, device="cpu", use_case="noise detection")
            DO.simulate_noise(noise_perc, noise_min, noise_max)
            _ensure_noise_detection_compat(DO)

            AM = AlgoModulators(DO, lr=config['lr'], eps_value=eps,
                                min_samples_ratio=min_samples_ratio,
                                save_results=True)
            dataloader = DO.prep_dataloader(use_info, config['batch_size'])
            model = initialize_model(DO, dataloader, config['hidden_size'],
                                     r_state, dropout=config['dropout'])

            TVM = TrainValidationManager(
                use_info, num_epochs, dataloader, config['batch_size'], r_state,
                ".", select_gradients=True,
                end_epochs_noise_detection=config['old_ggh_end_epochs'],
                best_valid_error=np.inf, final_analysis=True
            )
            TVM.train_model(DO, AM, model, final_analysis=True)

            # Extract gradients
            array_grads_context, do_hyp_class = get_gradarrays_n_labels(
                DO, 0, layer=-2, remov_avg=False, include_context=False,
                normalize_grads_context=False, loss_in_context=True,
                only_loss_context=True,
                num_batches=math.ceil(len(DO.df_train_noisy) / config['batch_size']),
                epoch=TVM.best_checkpoint, use_case="noise_detection"
            )

            # DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=int(config['batch_size'] * min_samples_ratio))
            pred_labels = dbscan.fit_predict(array_grads_context)
            pred_labels = pred_labels * -1  # Invert: 1=noisy (outliers), 0=clean

            detected_noisy_indices = [i for i, label in enumerate(pred_labels) if label == 1]

            # Remove detected noisy and retrain
            DO.df_train_noisy["noise_detected"] = pred_labels
            df_cleaned = deepcopy(DO.df_train_noisy[DO.df_train_noisy["noise_detected"] == 0])

            DO_clean, TVM_clean, AM_clean, model_clean = train_on_cleaned_data(
                DO, df_cleaned, r_state, config['data_path'], config['inpt_vars'],
                config['target_vars'], config['miss_vars'], config['hypothesis'],
                config['partial_perc'], config['batch_size'], config['hidden_size'],
                config['lr'], config['final_epochs'], config['dropout']
            )

            if TVM_clean.best_valid_error < best_val_error:
                best_val_error = TVM_clean.best_valid_error

                true_noisy_indices = DO.df_train_noisy[
                    DO.df_train_noisy['noise_added'] == 1
                ].index.tolist()
                detection_metrics = evaluate_detection(
                    detected_noisy_indices, true_noisy_indices, len(DO.df_train_noisy)
                )

                r2, mse, mae = evaluate_on_test(DO, model_clean)
                best_result = {
                    'eps': eps,
                    'min_samples_ratio': min_samples_ratio,
                    'test_r2': r2,
                    'test_mse': mse,
                    'test_mae': mae,
                    'detection': detection_metrics,
                    'n_detected': len(detected_noisy_indices),
                }

    return best_result


def run_old_ggh_dbscan_fast(DO_original, r_state, config):
    """Fast Old GGH noise detection: Adam training + single gradient extraction + DBSCAN.

    Unlike run_old_ggh_dbscan (which computes per-sample gradients every epoch),
    this variant trains normally with Adam, then computes gradients ONCE on the
    best model for DBSCAN clustering.

    Pipeline:
    1. Train normally with Adam on noisy data (no gradient selection)
    2. Load best model, compute per-sample gradients once
    3. DBSCAN clustering on gradients - outliers flagged as noisy
    4. Grid search over eps and min_samples
    5. Retrain on cleaned data

    Args:
        DO_original: DataOperator with noise simulated (used for test evaluation)
        r_state: Random state
        config: Same config dict as run_old_ggh_dbscan

    Returns:
        dict: {'test_r2', 'test_mse', 'test_mae', 'detection', 'n_detected', 'eps', 'min_samples_ratio'}
    """
    num_epochs = config['old_ggh_epochs'] + config['old_ggh_end_epochs']
    noise_perc = config['noise_perc']
    noise_min = config['noise_min']
    noise_max = config['noise_max']

    best_result = None
    best_val_error = np.inf

    for eps in config['old_ggh_eps_values']:
        for min_samples_ratio in config['old_ggh_min_samples_ratios']:
            set_to_deterministic(r_state)

            # Fresh DO for each combo (gradient storage modifies df in place)
            DO = DataOperator(config['data_path'], config['inpt_vars'],
                              config['target_vars'], config['miss_vars'],
                              config['hypothesis'], config['partial_perc'],
                              r_state, device="cpu", use_case="noise detection")
            DO.simulate_noise(noise_perc, noise_min, noise_max)
            _ensure_noise_detection_compat(DO)

            AM = AlgoModulators(DO, lr=config['lr'], eps_value=eps,
                                min_samples_ratio=min_samples_ratio)

            # --- Phase 1: Train normally with Adam on noisy data ---
            dataloader = DO.prep_dataloader("full info noisy", config['batch_size'])
            model = initialize_model(DO, dataloader, config['hidden_size'],
                                     r_state, dropout=config['dropout'])
            TVM = TrainValidationManager(
                "full info noisy", num_epochs, dataloader, config['batch_size'],
                r_state, ".", select_gradients=False, final_analysis=True
            )
            TVM.train_model(DO, AM, model, final_analysis=True)

            # --- Phase 2: Compute gradients ONCE on best model ---
            model = initialize_model(DO, dataloader, config['hidden_size'],
                                     r_state, dropout=config['dropout'])
            model.load_state_dict(torch.load(TVM.weights_save_path))
            model.eval()

            loss_fn = MSEIndividualLosses()
            for batch_i, (inputs, labels) in enumerate(dataloader):
                labels = labels.view(-1, 1)
                predictions = model(inputs)
                _, individual_losses = loss_fn(predictions, labels)
                grads = compute_individual_grads(model, individual_losses, "cpu")
                DO.append2hyp_df(batch_i, grads, "gradients", layer=AM.layer)
                ind_loss_array = individual_losses.detach().numpy()
                DO.append2hyp_df(batch_i, ind_loss_array, "loss")

            # --- Phase 3: DBSCAN on stored gradients ---
            array_grads_context, do_hyp_class = get_gradarrays_n_labels(
                DO, 0, layer=-2, remov_avg=False, include_context=False,
                normalize_grads_context=False, loss_in_context=True,
                only_loss_context=True,
                num_batches=math.ceil(len(DO.df_train_noisy) / config['batch_size']),
                epoch=0,  # Only one epoch of gradients stored
                use_case="noise_detection"
            )

            dbscan = DBSCAN(eps=eps, min_samples=int(config['batch_size'] * min_samples_ratio))
            pred_labels = dbscan.fit_predict(array_grads_context)
            pred_labels = pred_labels * -1  # Invert: 1=noisy (outliers), 0=clean

            detected_noisy_indices = [i for i, label in enumerate(pred_labels) if label == 1]

            # --- Phase 4: Remove detected noisy and retrain from scratch ---
            DO.df_train_noisy["noise_detected"] = pred_labels
            df_cleaned = deepcopy(DO.df_train_noisy[DO.df_train_noisy["noise_detected"] == 0])

            DO_clean, TVM_clean, AM_clean, model_clean = train_on_cleaned_data(
                DO, df_cleaned, r_state, config['data_path'], config['inpt_vars'],
                config['target_vars'], config['miss_vars'], config['hypothesis'],
                config['partial_perc'], config['batch_size'], config['hidden_size'],
                config['lr'], config['final_epochs'], config['dropout']
            )

            if TVM_clean.best_valid_error < best_val_error:
                best_val_error = TVM_clean.best_valid_error

                true_noisy_indices = DO.df_train_noisy[
                    DO.df_train_noisy['noise_added'] == 1
                ].index.tolist()
                detection_metrics = evaluate_detection(
                    detected_noisy_indices, true_noisy_indices, len(DO.df_train_noisy)
                )

                r2, mse, mae = evaluate_on_test(DO, model_clean)
                best_result = {
                    'eps': eps,
                    'min_samples_ratio': min_samples_ratio,
                    'test_r2': r2,
                    'test_mse': mse,
                    'test_mae': mae,
                    'detection': detection_metrics,
                    'n_detected': len(detected_noisy_indices),
                }

    return best_result


def run_new_ggh_unsupervised(DO, r_state, config):
    """New GGH unsupervised noise detection using soft refinement.

    Bootstrap clean/noisy anchors from loss distribution, then iteratively refine.
    NO LABELS USED during detection - fully unsupervised.

    Pipeline:
    1. Iter1: Unbiased training + bootstrap anchors from loss distribution
    2. Iter2: Weighted training (high weight = likely clean)
    3. Iter3: Refined anchors from biased model + rescoring
    4. Detection: Threshold on final weights
    5. Retrain on cleaned data

    Args:
        DO: DataOperator with noise simulated
        r_state: Random state
        config: Dictionary with keys:
            - data_path, inpt_vars, target_vars, miss_vars, hypothesis, partial_perc
            - batch_size, hidden_size, lr, dropout
            - ggh_iter1_epochs, ggh_iter1_analysis_epochs
            - ggh_iter2_epochs, final_epochs
            - ggh_min_weight, ggh_temperature
            - ggh_noise_threshold, ggh_clean_percentile

    Returns:
        dict: {'test_r2', 'test_mse', 'test_mae', 'detection', 'n_detected', 'sample_weights'}
    """
    inpt_vars = config['inpt_vars']
    target_vars = config['target_vars']
    n_features = len(inpt_vars)
    n_samples = len(DO.df_train_noisy)

    # Create dataloader
    train_features = torch.tensor(DO.df_train_noisy[inpt_vars].values, dtype=torch.float32)
    train_targets = torch.tensor(DO.df_train_noisy[target_vars].values, dtype=torch.float32)
    sample_indices = torch.arange(n_samples)
    dataset = TensorDataset(train_features, train_targets, sample_indices)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # =========================================================================
    # ITERATION 1: Unbiased training + Bootstrap anchors
    # =========================================================================
    set_to_deterministic(r_state)
    model_iter1 = SimpleNoiseDetectionModel(n_features, config['hidden_size'], len(target_vars))
    optimizer = torch.optim.Adam(model_iter1.parameters(), lr=config['lr'])
    criterion = nn.MSELoss(reduction='none')

    sample_losses = {i: [] for i in range(n_samples)}
    sample_gradients = {i: [] for i in range(n_samples)}

    total_iter1_epochs = config['ggh_iter1_epochs']
    track_start = total_iter1_epochs - config['ggh_iter1_analysis_epochs']

    for epoch in range(total_iter1_epochs):
        model_iter1.train()
        track_this_epoch = (epoch >= track_start)

        for features, targets, indices in dataloader:
            predictions = model_iter1(features)
            losses = criterion(predictions, targets).squeeze()
            batch_loss = losses.mean()

            if track_this_epoch:
                for i, idx in enumerate(indices):
                    sample_losses[idx.item()].append(losses[i].item())

                for i, idx in enumerate(indices):
                    feat = features[i:i+1].clone().requires_grad_(True)
                    pred = model_iter1(feat)
                    loss = criterion(pred, targets[i:i+1]).mean()
                    params = list(model_iter1.parameters())
                    grad_param = grad(loss, params[-2], retain_graph=False)[0]
                    grad_vec = grad_param.flatten().detach().cpu().numpy()
                    sample_gradients[idx.item()].append(grad_vec)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    avg_losses = {i: np.mean(sample_losses[i]) for i in range(n_samples) if sample_losses[i]}
    avg_gradients = {i: np.mean(sample_gradients[i], axis=0)
                     for i in range(n_samples) if sample_gradients[i]}

    loss_values = np.array([avg_losses[i] for i in range(n_samples)])
    loss_threshold = np.percentile(loss_values, config['ggh_clean_percentile'] * 100)

    clean_candidates = [i for i in range(n_samples) if avg_losses[i] <= loss_threshold]
    noisy_candidates = [i for i in range(n_samples) if avg_losses[i] > loss_threshold]

    if clean_candidates and noisy_candidates:
        clean_anchor_grad = np.mean(
            [avg_gradients[i] for i in clean_candidates if i in avg_gradients], axis=0
        )
        noisy_anchor_grad = np.mean(
            [avg_gradients[i] for i in noisy_candidates if i in avg_gradients], axis=0
        )

        sample_scores = {}
        for i in range(n_samples):
            if i not in avg_gradients:
                sample_scores[i] = 0.0
                continue
            g = avg_gradients[i]
            sim_clean = np.dot(g, clean_anchor_grad) / (
                np.linalg.norm(g) * np.linalg.norm(clean_anchor_grad) + 1e-8
            )
            sim_noisy = np.dot(g, noisy_anchor_grad) / (
                np.linalg.norm(g) * np.linalg.norm(noisy_anchor_grad) + 1e-8
            )
            sample_scores[i] = float(sim_clean - sim_noisy)

        scores_list = [sample_scores[i] for i in range(n_samples)]
        sample_weights = compute_soft_weights(
            scores_list, config['ggh_min_weight'], config['ggh_temperature']
        )
        sample_weights_dict = {i: float(sample_weights[i]) for i in range(n_samples)}
    else:
        sample_weights_dict = {i: 0.5 for i in range(n_samples)}

    # =========================================================================
    # ITERATION 2: Weighted training
    # =========================================================================
    set_to_deterministic(r_state + 100)
    model_iter2 = SimpleNoiseDetectionModel(n_features, config['hidden_size'], len(target_vars))
    optimizer2 = torch.optim.Adam(model_iter2.parameters(), lr=config['lr'])

    for epoch in range(config['ggh_iter2_epochs']):
        model_iter2.train()
        for features, targets, indices in dataloader:
            predictions = model_iter2(features)
            losses = criterion(predictions, targets).squeeze()
            weights = torch.tensor(
                [sample_weights_dict[idx.item()] for idx in indices], dtype=torch.float32
            )
            weighted_loss = (losses * weights).sum() / weights.sum()
            optimizer2.zero_grad()
            weighted_loss.backward()
            optimizer2.step()

    # =========================================================================
    # ITERATION 3: Refined anchors
    # =========================================================================
    model_iter2.eval()
    iter3_losses = {i: [] for i in range(n_samples)}
    iter3_gradients = {i: [] for i in range(n_samples)}

    for _ in range(3):
        for features, targets, indices in dataloader:
            for i, idx in enumerate(indices):
                feat = features[i:i+1].clone().requires_grad_(True)
                pred = model_iter2(feat)
                loss = criterion(pred, targets[i:i+1]).mean()
                iter3_losses[idx.item()].append(loss.item())
                params = list(model_iter2.parameters())
                grad_param = grad(loss, params[-2], retain_graph=False)[0]
                grad_vec = grad_param.flatten().detach().cpu().numpy()
                iter3_gradients[idx.item()].append(grad_vec)

    avg_iter3_gradients = {i: np.mean(iter3_gradients[i], axis=0)
                           for i in range(n_samples) if iter3_gradients[i]}

    weight_threshold_top = np.percentile(list(sample_weights_dict.values()), 70)
    weight_threshold_bottom = np.percentile(list(sample_weights_dict.values()), 30)

    refined_clean = [i for i in range(n_samples) if sample_weights_dict[i] >= weight_threshold_top]
    refined_noisy = [i for i in range(n_samples) if sample_weights_dict[i] <= weight_threshold_bottom]

    if refined_clean and refined_noisy:
        refined_clean_anchor = np.mean(
            [avg_iter3_gradients[i] for i in refined_clean if i in avg_iter3_gradients], axis=0
        )
        refined_noisy_anchor = np.mean(
            [avg_iter3_gradients[i] for i in refined_noisy if i in avg_iter3_gradients], axis=0
        )

        iter3_scores = {}
        for i in range(n_samples):
            if i not in avg_iter3_gradients:
                iter3_scores[i] = 0.0
                continue
            g = avg_iter3_gradients[i]
            sim_clean = np.dot(g, refined_clean_anchor) / (
                np.linalg.norm(g) * np.linalg.norm(refined_clean_anchor) + 1e-8
            )
            sim_noisy = np.dot(g, refined_noisy_anchor) / (
                np.linalg.norm(g) * np.linalg.norm(refined_noisy_anchor) + 1e-8
            )
            iter3_scores[i] = float(sim_clean - sim_noisy)

        scores_list_iter3 = [iter3_scores[i] for i in range(n_samples)]
        weights_iter3 = compute_soft_weights(
            scores_list_iter3, config['ggh_min_weight'], config['ggh_temperature']
        )

        for i in range(n_samples):
            sample_weights_dict[i] = sample_weights_dict[i] * weights_iter3[i]

        max_weight = max(sample_weights_dict.values())
        if max_weight > 0:
            for i in range(n_samples):
                sample_weights_dict[i] = (
                    config['ggh_min_weight'] +
                    (sample_weights_dict[i] / max_weight) * (1 - config['ggh_min_weight'])
                )

    # =========================================================================
    # DETECTION: Threshold on final weights
    # =========================================================================
    detected_noisy_indices = [
        i for i in range(n_samples)
        if sample_weights_dict[i] < config['ggh_noise_threshold']
    ]

    # Retrain on cleaned data
    df_cleaned = DO.df_train_noisy.drop(index=detected_noisy_indices).reset_index(drop=True)

    DO_clean, TVM_clean, AM_clean, model_clean = train_on_cleaned_data(
        DO, df_cleaned, r_state, config['data_path'], config['inpt_vars'],
        config['target_vars'], config['miss_vars'], config['hypothesis'],
        config['partial_perc'], config['batch_size'], config['hidden_size'],
        config['lr'], config['final_epochs'], config['dropout']
    )

    true_noisy_indices = DO.df_train_noisy[
        DO.df_train_noisy['noise_added'] == 1
    ].index.tolist()
    detection_metrics = evaluate_detection(
        detected_noisy_indices, true_noisy_indices, n_samples
    )

    r2, mse, mae = evaluate_on_test(DO, model_clean)

    return {
        'test_r2': r2,
        'test_mse': mse,
        'test_mae': mae,
        'detection': detection_metrics,
        'n_detected': len(detected_noisy_indices),
        'sample_weights': sample_weights_dict,
    }
