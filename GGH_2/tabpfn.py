"""TabPFN integration for hypothesis classification."""

import numpy as np

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False


def get_tabpfn_probabilities(DO, rand_state, verbose=False):
    """Use TabPFN to predict hypothesis class probabilities for all samples.

    Args:
        DO: DataOperator instance
        rand_state: Random seed (unused, but kept for API consistency)
        verbose: Whether to print diagnostic info

    Returns:
        tuple: (tabpfn_probs, diagnostics)
            - tabpfn_probs: dict mapping sample_idx -> array of probabilities for each hypothesis class
            - diagnostics: dict with diagnostic info (classes seen, confidence stats, etc.)
    """
    if not TABPFN_AVAILABLE:
        print("    Warning: TabPFN not installed. Install with: pip install tabpfn")
        return None, {'error': 'TabPFN not installed'}

    hyp_per_sample = DO.num_hyp_comb
    n_samples = len(DO.df_train_hypothesis) // hyp_per_sample

    # Get partial data for training TabPFN
    partial_correct_gids = DO.df_train_hypothesis[
        (DO.df_train_hypothesis['partial_full_info'] == 1) &
        (DO.df_train_hypothesis['correct_hypothesis'] == True)
    ].index.tolist()

    partial_sample_indices = set(gid // hyp_per_sample for gid in partial_correct_gids)

    # Prepare training data: input features -> hypothesis class
    X_train = []
    y_train = []

    for gid in partial_correct_gids:
        row = DO.df_train_hypothesis.iloc[gid]
        features = row[DO.inpt_vars].values.astype(np.float64)
        class_id = int(row['hyp_class_id'])
        X_train.append(features)
        y_train.append(class_id)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # DIAGNOSTIC: Class coverage
    unique_classes = np.unique(y_train)
    class_counts = {c: np.sum(y_train == c) for c in unique_classes}

    diagnostics = {
        'n_partial_samples': len(X_train),
        'classes_seen': unique_classes.tolist(),
        'n_classes_seen': len(unique_classes),
        'class_counts': class_counts,
        'missing_classes': [c for c in range(hyp_per_sample) if c not in unique_classes],
    }

    if verbose:
        print(f"    TabPFN Training: {len(X_train)} samples, {len(unique_classes)}/{hyp_per_sample} classes")
        print(f"    Classes seen: {unique_classes.tolist()}, Missing: {diagnostics['missing_classes']}")

    if len(X_train) < 2:
        print("    Warning: Not enough partial data for TabPFN")
        return None, diagnostics

    # Prepare test data: all non-partial samples
    X_test = []
    test_sample_indices = []

    for sample_idx in range(n_samples):
        if sample_idx in partial_sample_indices:
            continue

        gid = sample_idx * hyp_per_sample
        row = DO.df_train_hypothesis.iloc[gid]
        features = row[DO.inpt_vars].values.astype(np.float64)
        X_test.append(features)
        test_sample_indices.append(sample_idx)

    X_test = np.array(X_test)

    if len(X_test) == 0:
        return {}, diagnostics

    # Train TabPFN and get probabilities
    try:
        # Use DO.device for proper device handling
        tabpfn = TabPFNClassifier(device=DO.device)
        tabpfn.fit(X_train, y_train)
        probs = tabpfn.predict_proba(X_test)

        # Map to sample indices
        tabpfn_probs = {}
        confidence_scores = []

        for i, sample_idx in enumerate(test_sample_indices):
            if probs.shape[1] < hyp_per_sample:
                # Pad with uniform probabilities for missing classes
                full_probs = np.ones(hyp_per_sample) / hyp_per_sample
                for j, cls in enumerate(tabpfn.classes_):
                    full_probs[cls] = probs[i, j]
                tabpfn_probs[sample_idx] = full_probs
            else:
                tabpfn_probs[sample_idx] = probs[i]

            # Track confidence (max probability)
            confidence_scores.append(np.max(tabpfn_probs[sample_idx]))

        # DIAGNOSTIC: Confidence statistics
        diagnostics['avg_confidence'] = np.mean(confidence_scores)
        diagnostics['std_confidence'] = np.std(confidence_scores)
        diagnostics['min_confidence'] = np.min(confidence_scores)
        diagnostics['max_confidence'] = np.max(confidence_scores)

        if verbose:
            print(f"    TabPFN Confidence: avg={diagnostics['avg_confidence']:.3f}, "
                  f"std={diagnostics['std_confidence']:.3f}, "
                  f"range=[{diagnostics['min_confidence']:.3f}, {diagnostics['max_confidence']:.3f}]")

        return tabpfn_probs, diagnostics

    except Exception as e:
        print(f"    TabPFN error: {e}")
        diagnostics['error'] = str(e)
        return None, diagnostics


def analyze_tabpfn_vs_ggh_decisions(DO, tabpfn_probs, ggh_scores, sample_scores, verbose=True):
    """Analyze where TabPFN and GGH agree/disagree.

    Args:
        DO: DataOperator instance
        tabpfn_probs: dict from get_tabpfn_probabilities
        ggh_scores: GGH scores (unused in current implementation)
        sample_scores: dict mapping sample_idx -> (score, gid, is_correct)
        verbose: Whether to print analysis

    Returns:
        dict: Agreement/disagreement statistics
    """
    hyp_per_sample = DO.num_hyp_comb

    agreement_stats = {
        'total_samples': 0,
        'both_correct': 0,
        'both_wrong': 0,
        'ggh_correct_tabpfn_wrong': 0,
        'ggh_wrong_tabpfn_correct': 0,
        'tabpfn_predictions': [],
        'ggh_predictions': [],
        'true_classes': [],
    }

    for sample_idx, (ggh_score, ggh_gid, ggh_is_correct) in sample_scores.items():
        if tabpfn_probs is None or sample_idx not in tabpfn_probs:
            continue

        agreement_stats['total_samples'] += 1

        # GGH's choice
        ggh_class = DO.df_train_hypothesis.iloc[ggh_gid]['hyp_class_id']

        # TabPFN's choice (argmax of probabilities)
        tabpfn_class = np.argmax(tabpfn_probs[sample_idx])

        # True class
        start = sample_idx * hyp_per_sample
        true_gid = None
        for hyp_idx in range(hyp_per_sample):
            gid = start + hyp_idx
            if DO.df_train_hypothesis.iloc[gid]['correct_hypothesis']:
                true_gid = gid
                break

        if true_gid is None:
            continue

        true_class = DO.df_train_hypothesis.iloc[true_gid]['hyp_class_id']

        tabpfn_correct = (tabpfn_class == true_class)

        agreement_stats['tabpfn_predictions'].append(tabpfn_class)
        agreement_stats['ggh_predictions'].append(ggh_class)
        agreement_stats['true_classes'].append(true_class)

        if ggh_is_correct and tabpfn_correct:
            agreement_stats['both_correct'] += 1
        elif not ggh_is_correct and not tabpfn_correct:
            agreement_stats['both_wrong'] += 1
        elif ggh_is_correct and not tabpfn_correct:
            agreement_stats['ggh_correct_tabpfn_wrong'] += 1
        else:
            agreement_stats['ggh_wrong_tabpfn_correct'] += 1

    if verbose and agreement_stats['total_samples'] > 0:
        total = agreement_stats['total_samples']
        print(f"\n    === TabPFN vs GGH Agreement Analysis ===")
        print(f"    Both correct:           {agreement_stats['both_correct']:4d} ({agreement_stats['both_correct']/total*100:5.1f}%)")
        print(f"    Both wrong:             {agreement_stats['both_wrong']:4d} ({agreement_stats['both_wrong']/total*100:5.1f}%)")
        print(f"    GGH correct, TabPFN wrong: {agreement_stats['ggh_correct_tabpfn_wrong']:4d} ({agreement_stats['ggh_correct_tabpfn_wrong']/total*100:5.1f}%)")
        print(f"    GGH wrong, TabPFN correct: {agreement_stats['ggh_wrong_tabpfn_correct']:4d} ({agreement_stats['ggh_wrong_tabpfn_correct']/total*100:5.1f}%)")

        # TabPFN standalone accuracy
        tabpfn_accuracy = (agreement_stats['both_correct'] + agreement_stats['ggh_wrong_tabpfn_correct']) / total * 100
        ggh_accuracy = (agreement_stats['both_correct'] + agreement_stats['ggh_correct_tabpfn_wrong']) / total * 100
        print(f"    TabPFN standalone accuracy: {tabpfn_accuracy:.1f}%")
        print(f"    GGH standalone accuracy:    {ggh_accuracy:.1f}%")

    return agreement_stats
