# Repository Review: Gradient Guided Hypotheses (GGH)

## What the project does
- Implements the **Gradient Guided Hypotheses (GGH)** algorithm to detect noise and recover missing information by clustering enriched gradients from deep learning models. It is designed to work with any SGD-based architecture and treats missingness and label/feature noise within the same gradient-selection framework.【F:README.md†L1-L32】

## Why it exists
- Standard imputation or noise-handling techniques break down when data are extremely sparse or corrupted. GGH augments incomplete records with plausible hypotheses, then uses gradient patterns to decide which hypotheses (or data points) are trustworthy enough to train on. This enables learning even when only a handful of ground-truth examples per missing-value class are available.【F:README.md†L13-L21】【F:GGH/train_val_loop.py†L52-L191】

## How it works
1. **Training manager** – `TrainValidationManager` orchestrates model training for different scenarios (full data, imputation, hypotheses, or noise simulation). When hypotheses are used, it computes per-sample losses and gradients, optionally including partial ground-truth examples, and substitutes the usual optimizer step with a custom gradient-selection pipeline.【F:GGH/train_val_loop.py†L13-L206】
2. **Gradient enrichment & selection** – Gradients are computed per hypothesis; optional context (input features and losses) is concatenated to form enriched vectors. A One-Class SVM is trained on gradients from known-correct examples to admit similar hypotheses, while selection frequency thresholds prevent unstable choices.【F:GGH/selection_algorithms.py†L17-L355】
3. **Noise handling** – For noise-focused runs, enriched gradients are clustered with DBSCAN to isolate outlier gradients; only gradients from dense clusters are kept for backpropagation, helping the model prioritize clean data.【F:GGH/selection_algorithms.py†L482-L527】【F:GGH/train_val_loop.py†L109-L197】
4. **Optimization** – Selected gradients are averaged before a custom Adam optimizer applies them, allowing hypothesis-aware updates. Conventional Adam remains available for baseline training paths.【F:GGH/train_val_loop.py†L69-L205】

## Key takeaways
- GGH extends training data via hypothesis generation and filters updates using gradient distribution patterns, offering a unified approach to imputation and noise reduction.
- The framework is modular: swapping model architectures or selection hyperparameters (e.g., SVM `nu`, DBSCAN `eps`) adapts it to different scarcity/noise regimes without altering core training logic.【F:GGH/selection_algorithms.py†L17-L345】【F:GGH/selection_algorithms.py†L482-L527】
