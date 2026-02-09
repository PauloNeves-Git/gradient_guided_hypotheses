"""GGH (Gradient-Guided Hypotheses) main algorithm."""

import numpy as np

from .models import HypothesisAmplifyingModel
from .trainers import UnbiasedTrainer, WeightedTrainer, RemainingDataScorer
from .utils import set_to_deterministic, create_dataloader_with_gids
from .scoring import compute_anchor_data, compute_enriched_score, compute_soft_weights, sigmoid_stable


# Default hyperparameters
DEFAULT_CONFIG = {
    'iter1_epochs': 60,
    'iter1_analysis_epochs': 5,
    'iter1_lr': 0.01,
    'iter2_epochs': 30,
    'iter2_lr': 0.01,
    'scoring_passes': 5,
    'min_weight': 0.1,
    'temperature_iter1': 1.0,
    'temperature_iter3': 0.8,
    'loss_influence': 0.25,
    'partial_base_weight': 2.0,
    'shared_hidden': 16,
    'hypothesis_hidden': 32,
    'final_hidden': 32,
}


def run_ggh_soft_refinement(DO, rand_state, config=None):
    """Standard GGH soft refinement (without TabPFN prior).

    Full 4-iteration implementation:
    1. Unbiased training + initial soft weights
    2. Weighted training
    3. Biased rescoring with weight multiplication
    4. Loss-based adjustment

    Args:
        DO: DataOperator instance
        rand_state: Random seed for reproducibility
        config: Optional dict with hyperparameters (uses DEFAULT_CONFIG if not provided)

    Returns:
        tuple: (gid_weights, effective_precision, partial_correct_gids, partial_weight_dynamic)
    """
    # Merge config with defaults
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    set_to_deterministic(rand_state)

    hyp_per_sample = DO.num_hyp_comb
    n_samples = len(DO.df_train_hypothesis) // hyp_per_sample
    n_shared = len(DO.inpt_vars)
    n_hyp = len(DO.miss_vars)
    out_size = len(DO.target_vars)

    partial_correct_gids = set(DO.df_train_hypothesis[
        (DO.df_train_hypothesis['partial_full_info'] == 1) &
        (DO.df_train_hypothesis['correct_hypothesis'] == True)
    ].index.tolist())
    blacklisted_gids = set(DO.df_train_hypothesis[
        (DO.df_train_hypothesis['partial_full_info'] == 1) &
        (DO.df_train_hypothesis['correct_hypothesis'] == False)
    ].index.tolist())
    partial_sample_indices = set(gid // hyp_per_sample for gid in partial_correct_gids)

    dataloader = create_dataloader_with_gids(DO, batch_size=32)

    # === ITERATION 1: Unbiased training + Initial soft weights ===
    model_unbiased = HypothesisAmplifyingModel(
        n_shared, n_hyp,
        cfg['shared_hidden'], cfg['hypothesis_hidden'],
        cfg['final_hidden'], out_size
    ).to(DO.device)
    trainer_unbiased = UnbiasedTrainer(DO, model_unbiased, lr=cfg['iter1_lr'], device=DO.device)

    for epoch in range(cfg['iter1_epochs'] - cfg['iter1_analysis_epochs']):
        trainer_unbiased.train_epoch(dataloader, epoch, track_data=False)
    for epoch in range(cfg['iter1_epochs'] - cfg['iter1_analysis_epochs'], cfg['iter1_epochs']):
        trainer_unbiased.train_epoch(dataloader, epoch, track_data=True)

    anchor_data = compute_anchor_data(trainer_unbiased, DO)
    analysis = trainer_unbiased.get_hypothesis_analysis()
    input_cols = anchor_data['input_cols']

    # Standard scoring (no TabPFN)
    sample_scores = {}
    for sample_idx in range(n_samples):
        if sample_idx in partial_sample_indices:
            continue

        start = sample_idx * hyp_per_sample
        best_score, best_gid, best_is_correct = -np.inf, None, False

        for hyp_idx in range(hyp_per_sample):
            gid = start + hyp_idx
            if gid in blacklisted_gids or gid not in analysis or analysis[gid]['avg_gradient'] is None:
                continue

            gradient = analysis[gid]['avg_gradient']
            class_id = DO.df_train_hypothesis.iloc[gid]['hyp_class_id']
            features = DO.df_train_hypothesis.loc[gid, input_cols].values.astype(np.float64)
            score = compute_enriched_score(gradient, features, class_id, anchor_data)

            if score > best_score:
                best_score = score
                best_gid = gid
                best_is_correct = DO.df_train_hypothesis.iloc[gid]['correct_hypothesis']

        if best_gid is not None:
            sample_scores[sample_idx] = (best_score, best_gid, best_is_correct)

    # Convert to soft weights
    scores_list = [s[0] for s in sample_scores.values()]
    weights_iter1 = compute_soft_weights(scores_list, cfg['min_weight'], cfg['temperature_iter1'])

    gid_weights = {}
    for i, (sample_idx, (score, gid, is_correct)) in enumerate(sample_scores.items()):
        gid_weights[gid] = float(weights_iter1[i])

    # === ITERATION 2: Weighted training ===
    set_to_deterministic(rand_state + 100)
    model_weighted = HypothesisAmplifyingModel(
        n_shared, n_hyp,
        cfg['shared_hidden'], cfg['hypothesis_hidden'],
        cfg['final_hidden'], out_size
    ).to(DO.device)

    trainer_weighted = WeightedTrainer(
        DO, model_weighted, sample_weights=gid_weights,
        partial_gids=partial_correct_gids,
        partial_weight=cfg['partial_base_weight'], lr=cfg['iter2_lr'],
        device=DO.device
    )

    for epoch in range(cfg['iter2_epochs']):
        trainer_weighted.train_epoch(dataloader, epoch)

    # === ITERATION 3: Biased rescoring -> Multiply weights ===
    selected_sample_indices = set(sample_scores.keys())
    scorer = RemainingDataScorer(DO, model_weighted, selected_sample_indices | partial_sample_indices, device=DO.device)
    scorer.compute_scores(dataloader, n_passes=cfg['scoring_passes'])
    biased_analysis = scorer.get_analysis()

    # Build biased anchor data
    anchor_data_biased = {
        'anchor_correct_grad': {},
        'anchor_incorrect_grad': {},
        'anchor_correct_enriched': {},
        'anchor_incorrect_enriched': {},
        'feature_norm_params': {},
        'loss_norm_params': {},
    }

    all_grads = [biased_analysis[gid]['avg_gradient'] for gid in partial_correct_gids | blacklisted_gids
                 if gid in biased_analysis and biased_analysis[gid]['avg_gradient'] is not None]
    grad_scale = np.mean([np.linalg.norm(g) for g in all_grads]) if all_grads else 1.0
    anchor_data_biased['grad_scale'] = grad_scale

    inpt_vars_list = DO.inpt_vars

    for class_id in range(hyp_per_sample):
        correct_grads, incorrect_grads = [], []
        correct_features, incorrect_features = [], []
        correct_losses, incorrect_losses = [], []

        for gid in partial_correct_gids:
            if gid in biased_analysis and DO.df_train_hypothesis.iloc[gid]['hyp_class_id'] == class_id:
                if biased_analysis[gid]['avg_gradient'] is not None:
                    correct_grads.append(biased_analysis[gid]['avg_gradient'])
                    correct_features.append(DO.df_train_hypothesis.loc[gid, inpt_vars_list].values.astype(np.float64))
                    correct_losses.append(biased_analysis[gid]['avg_loss'])

        for gid in blacklisted_gids:
            if gid in biased_analysis and DO.df_train_hypothesis.iloc[gid]['hyp_class_id'] == class_id:
                if biased_analysis[gid]['avg_gradient'] is not None:
                    incorrect_grads.append(biased_analysis[gid]['avg_gradient'])
                    incorrect_features.append(DO.df_train_hypothesis.loc[gid, inpt_vars_list].values.astype(np.float64))
                    incorrect_losses.append(biased_analysis[gid]['avg_loss'])

        if correct_grads and incorrect_grads:
            anchor_data_biased['anchor_correct_grad'][class_id] = np.mean(correct_grads, axis=0)
            anchor_data_biased['anchor_incorrect_grad'][class_id] = np.mean(incorrect_grads, axis=0)

            all_features = correct_features + incorrect_features
            feat_mean = np.mean(all_features, axis=0)
            feat_std = np.std(all_features, axis=0) + 1e-8
            anchor_data_biased['feature_norm_params'][class_id] = {'mean': feat_mean, 'std': feat_std, 'scale': grad_scale}

            correct_features_norm = [(f - feat_mean) / feat_std * grad_scale for f in correct_features]
            incorrect_features_norm = [(f - feat_mean) / feat_std * grad_scale for f in incorrect_features]

            all_losses = correct_losses + incorrect_losses
            loss_mean = np.mean(all_losses)
            loss_std = np.std(all_losses) + 1e-8
            anchor_data_biased['loss_norm_params'][class_id] = {'mean': loss_mean, 'std': loss_std, 'scale': grad_scale}

            correct_losses_norm = [-(l - loss_mean) / loss_std * grad_scale for l in correct_losses]
            incorrect_losses_norm = [-(l - loss_mean) / loss_std * grad_scale for l in incorrect_losses]

            correct_enriched = [np.concatenate([g, f, [l]])
                               for g, f, l in zip(correct_grads, correct_features_norm, correct_losses_norm)]
            incorrect_enriched = [np.concatenate([g, f, [l]])
                                 for g, f, l in zip(incorrect_grads, incorrect_features_norm, incorrect_losses_norm)]

            anchor_data_biased['anchor_correct_enriched'][class_id] = np.mean(correct_enriched, axis=0)
            anchor_data_biased['anchor_incorrect_enriched'][class_id] = np.mean(incorrect_enriched, axis=0)

    # Rescore with biased model
    iter3_scores = {}
    for sample_idx, (_, gid, _) in sample_scores.items():
        if gid in biased_analysis and biased_analysis[gid]['avg_gradient'] is not None:
            gradient = biased_analysis[gid]['avg_gradient']
            loss = biased_analysis[gid]['avg_loss']
            class_id = DO.df_train_hypothesis.iloc[gid]['hyp_class_id']
            features = DO.df_train_hypothesis.loc[gid, inpt_vars_list].values.astype(np.float64)

            norm_params = anchor_data_biased.get('feature_norm_params', {}).get(class_id)
            loss_params = anchor_data_biased.get('loss_norm_params', {}).get(class_id)

            if norm_params:
                features_norm = (features - norm_params['mean']) / norm_params['std'] * norm_params['scale']
            else:
                features_norm = features * grad_scale / (np.linalg.norm(features) + 1e-8)

            if loss_params:
                loss_norm = -((loss - loss_params['mean']) / loss_params['std']) * loss_params['scale']
            else:
                loss_norm = -loss * grad_scale

            enriched = np.concatenate([gradient, features_norm, [loss_norm]])

            anchor_c = anchor_data_biased.get('anchor_correct_enriched', {}).get(class_id)
            anchor_i = anchor_data_biased.get('anchor_incorrect_enriched', {}).get(class_id)

            if anchor_c is not None:
                sim_c = float(np.dot(enriched, anchor_c) / (np.linalg.norm(enriched) * np.linalg.norm(anchor_c) + 1e-8))
                sim_i = float(np.dot(enriched, anchor_i) / (np.linalg.norm(enriched) * np.linalg.norm(anchor_i) + 1e-8)) if anchor_i is not None else 0.0
                iter3_scores[gid] = sim_c - sim_i
            else:
                iter3_scores[gid] = 0.0

    # Multiply weights
    scores_list_iter3 = list(iter3_scores.values())
    gids_iter3 = list(iter3_scores.keys())
    weights_iter3_raw = compute_soft_weights(scores_list_iter3, cfg['min_weight'], cfg['temperature_iter3'])

    for i, gid in enumerate(gids_iter3):
        gid_weights[gid] = gid_weights[gid] * weights_iter3_raw[i]

    # Renormalize
    if gid_weights:
        max_w = max(gid_weights.values())
        if max_w > 0:
            for gid in gid_weights:
                gid_weights[gid] = cfg['min_weight'] + (gid_weights[gid] / max_w) * (1 - cfg['min_weight'])

    # === ITERATION 4: Loss-based adjustment ===
    losses = {gid: biased_analysis[gid]['avg_loss']
              for gid in gid_weights if gid in biased_analysis}

    if losses:
        loss_values = list(losses.values())
        loss_mean = np.mean(loss_values)
        loss_std = np.std(loss_values) + 1e-8

        for gid in gid_weights:
            if gid in losses:
                norm_loss = (losses[gid] - loss_mean) / loss_std
                loss_factor = 1 - cfg['loss_influence'] * sigmoid_stable(norm_loss)
                gid_weights[gid] = max(cfg['min_weight'], gid_weights[gid] * loss_factor)

    # Calculate effective precision
    correct_weights_final = [gid_weights[s[1]] for s in sample_scores.values() if s[2] and s[1] in gid_weights]
    total_weight_correct = sum(correct_weights_final) if correct_weights_final else 0
    total_weight_all = sum(gid_weights.values()) if gid_weights else 1
    effective_precision = total_weight_correct / total_weight_all * 100 if total_weight_all > 0 else 0

    avg_final_weight = np.mean(list(gid_weights.values())) if gid_weights else 0.5
    partial_weight_dynamic = cfg['partial_base_weight'] * (1 + (1 - avg_final_weight))

    return gid_weights, effective_precision, partial_correct_gids, partial_weight_dynamic
