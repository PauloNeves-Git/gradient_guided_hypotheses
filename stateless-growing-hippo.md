# GGH Soft Weight Iterative Refinement - Algorithm Documentation

## Overview

The GGH (Gradient-Guided Hypothesis) Soft Weight Iterative Refinement algorithm is a method for selecting the correct hypothesis value for samples with missing data. Unlike traditional imputation methods that predict a single value, GGH uses gradient patterns from neural network training to identify which hypothesis is most likely correct for each sample.

---

## Core Components

### 1. HypothesisAmplifyingModel Architecture

A specialized neural network designed to amplify gradient sensitivity to hypothesis features:

```
Input: [shared_features (n), hypothesis_feature (1)]
       ↓                        ↓
  Shared Path              Hypothesis Path
  (smaller: 16 hidden)     (larger: 32→32 hidden, 2 layers)
       ↓                        ↓
       └──────── Concat ────────┘
                   ↓
              Final Path
           (32 hidden → output)
```

**Key insight**: The larger hypothesis path creates stronger gradient signals for hypothesis features, making it easier to distinguish correct vs incorrect hypotheses.

### 2. Anchors (Reference Gradient Patterns)

Built from **partial data** (samples with known correct hypothesis values):

| Anchor Type | Description |
|-------------|-------------|
| `anchor_correct_grad[class_id]` | Mean gradient of known correct hypotheses for each class |
| `anchor_incorrect_grad[class_id]` | Mean gradient of known incorrect hypotheses for each class |
| `anchor_correct_enriched[class_id]` | Mean enriched vector (gradient + normalized features) of correct |
| `anchor_incorrect_enriched[class_id]` | Mean enriched vector of incorrect |

### 3. Enriched Vectors

Combine multiple signals for better discrimination:

- **Iteration 1**: `enriched = [gradient, normalized_features]`
- **Iteration 3+**: `enriched = [gradient, normalized_features, normalized_loss]`

Feature normalization scales features to match gradient magnitude:
```python
features_norm = (features - feat_mean) / feat_std * grad_scale
```

### 4. Scoring Function

Cosine similarity-based differential scoring:
```python
score = cosine_sim(enriched, anchor_correct) - cosine_sim(enriched, anchor_incorrect)
```

- **High score** → gradient pattern resembles correct hypotheses
- **Low score** → gradient pattern resembles incorrect hypotheses

### 5. Soft Weights via Sigmoid

Convert scores to continuous weights in range `[min_weight, 1.0]`:
```python
normalized = (score - mean) / std
raw_weight = sigmoid(normalized / temperature)
weight = min_weight + (1 - min_weight) * raw_weight
```

---

## Algorithm Flow (4 Iterations + Final Training)

### Iteration 1: Unbiased Training + Initial Scoring

**Purpose**: Learn basic patterns and compute initial hypothesis scores

1. **Train** `HypothesisAmplifyingModel` on ALL hypotheses equally (60 epochs)
2. **Track** gradients and losses during last 5 epochs
3. **Build anchors** from partial data (known correct/incorrect per class)
4. **Score** each sample's hypotheses using enriched vectors
5. **Select** best-scoring hypothesis per sample
6. **Convert** scores to soft weights via sigmoid

**Output**: Initial `gid_weights` dictionary mapping global_id → weight

```
Iter1: 847 samples, precision: 23.5%
  Avg weight correct: 0.612, incorrect: 0.487
```

### Iteration 2: Weighted Training

**Purpose**: Bias model toward likely-correct hypotheses

1. **Create** new `HypothesisAmplifyingModel`
2. **Train** with weighted loss:
   - Partial data (known correct): weight = 2.0
   - Other samples: weight from Iter1
3. **Train** for 30 epochs

**Effect**: Model becomes biased toward likely-correct data patterns

### Iteration 3: Biased Rescoring

**Purpose**: Refine weights using improved model

1. **Recompute** gradients and losses for all samples using biased model
2. **Rebuild** anchors from partial data (now with biased gradients)
3. **Build** enriched vectors with **loss included**:
   ```python
   enriched = [gradient, features_norm, -loss_norm]  # Negated: low loss = good
   ```
4. **Rescore** all samples
5. **Multiply** new weights with Iter1 weights
6. **Renormalize** to `[min_weight, 1.0]`

```
Iter3: Avg weight correct: 0.658, incorrect: 0.441
```

### Iteration 4: Loss-based Adjustment

**Purpose**: Further penalize high-loss samples

```python
for gid in gid_weights:
    norm_loss = (loss[gid] - loss_mean) / loss_std
    loss_factor = 1 - loss_influence * sigmoid(norm_loss)
    gid_weights[gid] = max(min_weight, gid_weights[gid] * loss_factor)
```

**Effect**: High loss → weight reduced; Low loss → weight preserved

```
Iter4: Avg weight correct: 0.701, incorrect: 0.398
```

### Final Training

1. **Compute** dynamic partial weight based on average final weights
2. **Train** final model with all soft weights
3. **Use** validation-based epoch selection
4. **Evaluate** on test set

---

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `GGH_MIN_WEIGHT` | 0.1 | Minimum weight (no sample below this) |
| `GGH_TEMPERATURE_ITER1` | 1.0 | Sharpness of Iter1 weight distribution |
| `GGH_TEMPERATURE_ITER3` | 0.8 | Sharper weights in Iter3 |
| `GGH_LOSS_INFLUENCE` | 0.25 | How much loss affects final weights |
| `GGH_PARTIAL_BASE_WEIGHT` | 2.0 | Base weight for known correct data |
| `GGH_ITER1_EPOCHS` | 60 | Unbiased training epochs |
| `GGH_ITER2_EPOCHS` | 30 | Weighted training epochs |
| `GGH_FINAL_EPOCHS` | 200 | Final training epochs |

---

## Effective Precision Metric

Instead of binary correct/incorrect counting, GGH uses weight-based precision:

```python
effective_precision = sum(weights of correct hypotheses) / sum(all weights) * 100%
```

This accounts for the fact that incorrect hypotheses receive lower weights, so their negative impact on training is reduced.

**Example**:
- Unweighted precision: 23.5% (raw selection accuracy)
- Effective precision: 42.1% (accounting for soft weights)

---

## Supporting Classes

### UnbiasedTrainer
Trains on ALL hypotheses equally, tracks per-sample losses and gradients.

### WeightedTrainer
Trains with continuous sample weights. Key feature: weighted loss computation:
```python
weighted_loss = (individual_losses * weights).sum() / weights.sum()
```

### RemainingDataScorer
Scores samples using a biased model. Computes multiple passes for stable gradient estimates.

---

## Why GGH Works

1. **Gradient patterns are discriminative**: Correct hypotheses produce different gradient patterns than incorrect ones
2. **Anchors provide reference**: Partial data gives us ground truth to compare against
3. **Iterative refinement improves estimates**: Each iteration biases the model more toward correct patterns
4. **Soft weights are robust**: Even imperfect selection is useful when weights reflect confidence

---

# Research Plan: Adapting New GGH Soft Refinement for Noise Detection

## User Goal

Demonstrate that the new GGH soft refinement method is a **unified algorithm** that works for both:
1. **Missing data** (hypothesis selection / imputation) - already demonstrated
2. **Incorrect data** (noise detection) - needs implementation

Create benchmark notebooks comparing new GGH vs old GGH (DBSCAN-based) for noise detection on:
- Photoredox Yield dataset
- Photocell Degradation dataset

## Understanding Old GGH Noise Detection

### How Old GGH Worked for Noise Detection

From `Photoredox Yield Noise Detection.ipynb` and `Photocell Deg Noise Detection.ipynb`:

1. **Noise Simulation**:
   - `DO.simulate_noise(DATA_NOISE_PERC=0.3-0.4, NOISE_MINRANGE=0.4, NOISE_MAXRANGE=0.6)`
   - Corrupts 30-40% of target values: `target_noisy = target * (1 + noise_factor)`
   - Creates `df_train_noisy` with both clean and noisy samples
   - True labels tracked: `label=0` for clean, `label=1` for noisy

2. **Training on Noisy Data**:
   - `use_info = "known info noisy simulation"` or `"full info noisy"`
   - Train model on noisy data for ~600 epochs
   - Extract gradients during last `end_epochs_nd = 10-15` epochs
   - Use enriched context: `loss_in_context = True` (gradient + loss value)

3. **DBSCAN Clustering for Detection**:
   ```python
   array_grads_context = get_gradarrays_n_labels(DO, layer=-2,
                                                   loss_in_context=True)
   dbscan = DBSCAN(eps=0.15-0.25, min_samples=int(batch_size*0.15-0.25))
   pred_labels = dbscan.labels_ * -1  # Invert: 1=noisy, 0=clean
   ```
   - Outliers (label=-1 from DBSCAN) detected as noisy
   - Hyperparameter grid search over eps and min_samples_ratio

4. **Post-Detection Retraining**:
   - Remove detected noisy samples
   - Retrain model on cleaned data
   - Select best hyperparameters based on validation error

5. **Evaluation Metrics**:
   - **Detection**: Accuracy, Precision on noise classification
   - **Model performance**: R2, MSE, MAE on test set
   - Compare against:
     - Full info (no noise) - oracle
     - Full info noisy (with noise, no removal) - baseline with noise
     - Known info noisy simulation (with detection + removal) - GGH method

### Performance Benchmarks

**Photoredox Yield** (30% noise):
- Full Info: R2 = 0.858
- Full Info Noisy: R2 = 0.608 (baseline with noise)
- Old GGH (DBSCAN): R2 = 0.674 (improvement of 0.066)

**Photocell Degradation** (40% noise):
- Full Info: R2 = 0.845
- Full Info Noisy: R2 = 0.674 (baseline with noise)
- Old GGH (DBSCAN): R2 = 0.799 (improvement of 0.125)

## How New GGH Soft Refinement Can Be Adapted

### Core Insight

The new GGH soft refinement already has all the components needed for noise detection:

| Component | Missing Data Use | Noise Detection Use |
|-----------|------------------|---------------------|
| **Anchors** | Partial data (known correct hypotheses) | Clean samples (ground truth known) |
| **Target samples** | Unknown hypotheses to score | All training samples to classify |
| **Scoring** | Gradient similarity to correct anchors | Gradient similarity to clean anchors |
| **Output** | Soft weights (high = likely correct) | Soft weights (low = likely noisy) |
| **Decision** | Weight training by confidence | Threshold to detect noise |

### Adaptation Algorithm (Unsupervised)

**Key Difference from Imputation**: No labeled "partial data" available. Must bootstrap anchors from data itself using gradient patterns.

**Stage 1: Simulate Noise (Benchmark Only)**
```python
DO.simulate_noise(DATA_NOISE_PERC=0.3, NOISE_MINRANGE=0.4, NOISE_MAXRANGE=0.6)
# Labels (clean vs noisy) are tracked but NOT used for detection
# Labels only used for evaluation metrics (precision, recall, accuracy)
```

**Stage 2: Bootstrap Clean/Noisy Anchors (Unsupervised)**
1. **Iter1 - Unbiased Training + Bootstrap Anchors**:
   - Train on ALL data (both clean and noisy) - no labels used
   - Track gradients and losses for all samples
   - **Bootstrap clean candidates**: Bottom 60-70% by loss → likely clean
   - **Bootstrap noisy candidates**: Top 30-40% by loss → likely noisy
   - Build initial anchors from these candidates:
     - `clean_anchor = mean(gradients of low-loss samples)`
     - `noisy_anchor = mean(gradients of high-loss samples)`
   - Score all samples: `enriched_score = sim_to_clean_anchor - sim_to_noisy_anchor`
   - Convert to soft weights using sigmoid

2. **Iter2 - Weighted Training**:
   - Weight high-scoring samples (likely clean) more heavily
   - Weight low-scoring samples (likely noisy) less
   - Train for brief epochs
   - Model becomes biased toward likely-clean data

3. **Iter3 - Refined Anchors + Rescoring**:
   - Model now biased toward clean data
   - Recompute anchors from top-weighted samples:
     - `refined_clean_anchor = mean(gradients of top 60% by weight)`
     - `refined_noisy_anchor = mean(gradients of bottom 30% by weight)`
   - Rescore all samples with refined anchors
   - Multiply weights iteratively

4. **Iter4 - Loss-based Adjustment**:
   - High loss → likely noisy → reduce weight further
   - `final_weight = iter3_weight * (1 - loss_influence * sigmoid(normalized_loss))`

**Stage 3: Noise Detection (Thresholding)**
```python
# Method 1: Fixed threshold
NOISE_THRESHOLD = 0.3  # Samples with weight < 0.3 detected as noisy
detected_noisy = [idx for idx, w in sample_weights.items() if w < NOISE_THRESHOLD]

# Method 2: Percentage-based (if we know expected noise rate)
sorted_samples = sorted(sample_weights.items(), key=lambda x: x[1])
n_noisy_expected = int(len(samples) * DATA_NOISE_PERC)
detected_noisy = [idx for idx, _ in sorted_samples[:n_noisy_expected]]
```

**Stage 4: Retraining on Cleaned Data**
```python
cleaned_data = df_train[~df_train.index.isin(detected_noisy)]
# Retrain model on cleaned data (removes detected noisy samples)
# Evaluate on test set
```

**Stage 5: Evaluation (Labels Used Only Here)**
```python
true_noisy = set(df_train[df_train['label'] == 1].index)
true_clean = set(df_train[df_train['label'] == 0].index)
detected_noisy_set = set(detected_noisy)

# Confusion matrix
TP = len(detected_noisy_set & true_noisy)  # Correctly detected noisy
FP = len(detected_noisy_set & true_clean)  # Clean wrongly labeled as noisy
FN = len(true_noisy - detected_noisy_set)  # Noisy missed
TN = len(true_clean - detected_noisy_set)  # Clean correctly kept

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Key Differences from Old GGH

| Aspect | Old GGH (DBSCAN) | New GGH (Soft Refinement) |
|--------|------------------|---------------------------|
| Classification | Hard (binary: clean/noisy) | Soft (continuous weights) |
| Anchor type | Implicit (DBSCAN clusters) | Explicit (clean vs noisy) |
| Context | Gradient + loss only | Gradient + features + loss |
| Iterations | Single pass | Iterative refinement (4 stages) |
| Hyperparameters | eps, min_samples (grid search) | min_weight, temperature, threshold |
| Scoring | DBSCAN outlier detection | Anchor cosine similarity |

## Implementation Plan: New GGH for Noise Detection

### Function: `run_ggh_noise_detection()`

Adapt the existing `run_ggh_soft_refinement()` function for noise detection:

```python
def run_ggh_noise_detection(DO, r_state, clean_sample_indices, noisy_sample_indices,
                            noise_threshold=0.3):
    """
    GGH soft refinement adapted for noise detection.

    Args:
        DO: DataOperator with noisy data (df_train_noisy)
        r_state: Random state
        clean_sample_indices: Known clean sample indices (for anchor building)
        noisy_sample_indices: Known noisy sample indices (for anchor building)
        noise_threshold: Weight threshold below which samples detected as noisy

    Returns:
        sample_weights: Dict mapping sample_idx to weight (0.1-1.0)
        detected_noisy: List of detected noisy sample indices
        detection_precision: Precision of noise detection
        detection_recall: Recall of noise detection
    """

    # === ITERATION 1: Unbiased training + Initial soft weights ===
    # Train on ALL samples (clean + noisy) without discrimination
    # Track gradients for clean samples → "clean anchors"
    # Track gradients for noisy samples → "noisy anchors"
    # Score all samples: enriched_score = sim_to_clean - sim_to_noisy
    # Convert to soft weights

    # === ITERATION 2: Weighted training ===
    # Weight known clean samples highly (weight=2.0)
    # Weight other samples by Iter1 scores
    # Brief training (30 epochs)

    # === ITERATION 3: Biased rescoring ===
    # Model now biased toward clean data
    # Recompute anchors from clean samples only
    # Rescore all samples → multiply weights

    # === ITERATION 4: Loss-based adjustment ===
    # High loss → likely noisy → reduce weight

    # === DETECTION ===
    # Threshold: samples with weight < noise_threshold detected as noisy
    # Or: bottom K% by weight (K = expected noise percentage)

    return sample_weights, detected_noisy, detection_precision, detection_recall
```

### Benchmark Notebook Structure

Create TWO notebooks:
1. **`notebooks/Photoredox_Yield_Noise_Detection_Benchmark.ipynb`**
2. **`notebooks/Photocell_Deg_Noise_Detection_Benchmark.ipynb`**

Each notebook follows this structure:

```
1. Header & Configuration
   - Dataset info (Photoredox Yield or Photocell Deg)
   - Noise simulation parameters (30-40% noise, range 0.4-0.6)
   - Model parameters (same as old notebooks)

2. Noise Simulation
   - Use DO.simulate_noise() to corrupt target values
   - Track true labels (clean vs noisy)

3. Baseline Methods
   a. Full Info (no noise) - oracle upper bound
   b. Full Info Noisy (with noise, no removal) - baseline lower bound
   c. Old GGH (DBSCAN) - existing method to beat

4. New GGH Soft Refinement for Noise Detection
   - Implement run_ggh_noise_detection()
   - Run iterative refinement (4 stages)
   - Detect noisy samples via weight thresholding
   - Remove detected noisy samples
   - Retrain on cleaned data

5. Evaluation Metrics
   a. Detection Metrics:
      - Accuracy: (TP + TN) / (TP + TN + FP + FN)
      - Precision: TP / (TP + FP)
      - Recall: TP / (TP + FN)
      - Confusion matrix
   b. Model Performance Metrics:
      - R2, MSE, MAE on test set
      - Compare against Full Info and Full Info Noisy

6. Statistical Analysis
   - Paired t-tests across 15 runs
   - Win rate (New GGH vs Old GGH)
   - Improvement over baseline

7. Visualization
   - R2 comparison bar plot
   - Detection precision/recall plots
   - Weight distribution (clean vs noisy samples)
   - Confusion matrices
```

### Files to Create

1. **NEW: `notebooks/Photoredox_Yield_Noise_Detection_Benchmark.ipynb`**
   - Photoredox Yield dataset
   - 30% noise simulation
   - Compare: Full Info, Full Info Noisy, Old GGH (DBSCAN), New GGH (Soft Refinement)

2. **NEW: `notebooks/Photocell_Deg_Noise_Detection_Benchmark.ipynb`**
   - Photocell Degradation dataset
   - 40% noise simulation
   - Same comparison structure

### Key Parameters

| Parameter | Photoredox Yield | Photocell Deg |
|-----------|------------------|---------------|
| DATA_NOISE_PERC | 0.30 | 0.40 |
| NOISE_MINRANGE | 0.40 | 0.40 |
| NOISE_MAXRANGE | 0.60 | 0.60 |
| Epochs | 300 | 600 |
| Batch size | 299 | 250 |
| NOISE_THRESHOLD | 0.30 | 0.30 |

### Expected Performance Targets

Based on old GGH results, new GGH should achieve:

**Photoredox Yield** (30% noise):
- Old GGH: R2 = 0.674 (improvement of 0.066 over noisy baseline)
- Target: R2 > 0.70 (beat old GGH by at least 0.03)

**Photocell Degradation** (40% noise):
- Old GGH: R2 = 0.799 (improvement of 0.125 over noisy baseline)
- Target: R2 > 0.82 (beat old GGH by at least 0.02)

### Advantages of New GGH Over Old GGH

1. **Soft weights** instead of hard binary classification
   - More nuanced: uncertain samples get intermediate weights
   - Can adjust detection threshold post-hoc

2. **Enriched context** (gradients + features + loss)
   - Old GGH: gradient + loss only
   - New GGH: gradient + input features + loss (more discriminative)

3. **Iterative refinement**
   - Old GGH: Single-pass DBSCAN clustering
   - New GGH: 4-stage iterative scoring and reweighting

4. **Explicit anchors**
   - Old GGH: Implicit clusters from DBSCAN
   - New GGH: Explicit clean vs noisy anchors

5. **No hyperparameter grid search**
   - Old GGH: Grid search over eps and min_samples_ratio
   - New GGH: Fixed parameters (min_weight, temperature, threshold)

## Verification Plan

1. **Create Photoredox Yield Noise Detection Benchmark**:
   - Simulate 30% noise on target values
   - Run 15 trials with different random states
   - Compare: Full Info, Full Info Noisy, Old GGH (DBSCAN), New GGH (Soft Refinement)
   - Metrics: Detection (Accuracy, Precision, Recall), Performance (R2, MSE, MAE)
   - Target: Beat Old GGH (R2 > 0.70 vs 0.674)

2. **Create Photocell Deg Noise Detection Benchmark**:
   - Simulate 40% noise on target values
   - Run 15 trials with different random states
   - Same comparison and metrics
   - Target: Beat Old GGH (R2 > 0.82 vs 0.799)

3. **Verify Detection Quality**:
   - Check confusion matrices for both datasets
   - Ensure precision > 70% (low false positive rate)
   - Ensure recall > 60% (reasonable true positive rate)
   - Visualize weight distributions (clean vs noisy samples should separate)

4. **Compare Against Old GGH**:
   - Statistical significance via paired t-tests
   - Win rate across 15 runs
   - Detection metrics comparison (precision/recall/accuracy)

## Implementation Notes

### Code Reuse from Soft Refinement

The noise detection function will reuse most components from `run_ggh_soft_refinement()`:
- `HypothesisAmplifyingModel` (same architecture)
- `UnbiasedTrainer`, `WeightedTrainer` (same training logic)
- `compute_anchor_data()` (adapt for clean vs noisy anchors)
- `compute_enriched_score()` (same scoring logic)
- `compute_soft_weights()` (same weight conversion)

### Key Adaptation Points (Unsupervised)

1. **Anchor building**: Bootstrap from data itself (no labeled anchors)
   - Use loss distribution to identify likely clean (low loss) vs likely noisy (high loss)
   - Build initial anchors from these candidates
   - Refine anchors iteratively as model improves

2. **Sample identification**: NO labeled data used during detection
   - Only gradient patterns and loss values guide detection
   - Labels only used post-hoc for evaluation metrics

3. **Detection threshold**: Weight-based binary classification
   - Fixed threshold: `weight < 0.3` → noisy
   - Or percentage-based if noise rate known: bottom K% by weight

4. **Evaluation**: Detection metrics computed using ground truth labels
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix visualization

### Comparison: Supervised (Imputation) vs Unsupervised (Noise Detection)

| Aspect | Imputation (Supervised) | Noise Detection (Unsupervised) |
|--------|-------------------------|-------------------------------|
| **Input** | Partial data (known correct) | No labeled data |
| **Anchors** | From partial correct samples | Bootstrapped from low-loss samples |
| **Target** | Select correct hypothesis | Detect noisy samples |
| **Weights** | High = likely correct | High = likely clean, Low = likely noisy |
| **Evaluation** | Test R2 after training | Detection metrics + Test R2 |

### Notebook Self-Containment

Both notebooks will be fully self-contained:
- All functions defined in notebook (no external dependencies beyond GGH)
- Copy relevant functions from Photocell_Benchmark.ipynb GGH implementation
- Easy to run independently and share results
