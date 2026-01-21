# Quick Start: Using TabPFN-Style Transformer with GGH

## Requirements

- Python 3.8+
- PyTorch 1.8.0+ (tested with 1.8.2 and 2.0+)
- Existing GGH dependencies

## Installation

The TabPFN package is already installed. No additional setup needed.

## Basic Usage

### 1. Import and Setup (Same as before)

```python
from GGH.data_ops import DataOperator
from GGH.selection_algorithms import AlgoModulators
from GGH.models import initialize_model
from GGH.train_val_loop import TrainValidationManager
from GGH.inspector import Inspector

# Setup parameters
data_path = '../data/photoredox_yield/photo_redox_merck2021_1649reactions.csv'
results_path = "../saved_results/Photoredox Yield"
inpt_vars = ['aryl_halides', 'photocalysts', 'piperidines_moles']
target_vars = ['uplcms']
miss_vars = ['photocalysts_moles']
hypothesis = [[0.02, 0.05, 0.50, 5.0]]
partial_perc = 0.3
rand_state = 42
```

### 2. Run Experiment with Transformer

```python
# Initialize data
DO = DataOperator(data_path, inpt_vars, target_vars, miss_vars, 
                  hypothesis, partial_perc, rand_state, device="cpu")

# Run with transformer (NEW: model_type="tabpfn")
DO, TVM, model = full_experiment(
    use_info="use hypothesis",
    DO=DO,
    INSPECT=INSPECT,
    batch_size=400,
    hidden_size=64,        # Larger for transformer
    output_size=1,
    num_epochs=60,
    rand_state=42,
    results_path=results_path,
    dropout=0.05,
    lr=0.001,
    nu=0.1,
    normalize_grads_contx=False,
    use_context=True,
    final_analysis=False,
    model_type="tabpfn"    # ← KEY CHANGE: Use transformer
)
```

### 3. Compare with MLP

```python
# Run with standard MLP for comparison
DO_mlp, TVM_mlp, model_mlp = full_experiment(
    use_info="use hypothesis",
    DO=DO,
    INSPECT=INSPECT,
    batch_size=400,
    hidden_size=32,        # Smaller for MLP
    output_size=1,
    num_epochs=60,
    rand_state=42,
    results_path=results_path,
    dropout=0.05,
    lr=0.001,
    nu=0.1,
    normalize_grads_contx=False,
    use_context=True,
    final_analysis=False,
    model_type="mlp"       # ← Use MLP
)
```

## What Changed?

**Two additions to function calls:**

1. **`model_type` parameter in `full_experiment()`**
   - `"tabpfn"` → Use transformer-based model
   - `"mlp"` → Use standard 2-layer MLP (default)

2. **Larger `hidden_size` for transformer**
   - MLP: 32 neurons (typical)
   - Transformer: 64-128 neurons (recommended)

## Checking Results

```python
# Evaluate performance
print("Transformer Results:")
print(f"Val R²: {INSPECT.calculate_val_r2score(DO, TVM, model, data='validation'):.4f}")
print(f"Test R²: {INSPECT.calculate_val_r2score(DO, TVM, model, data='test'):.4f}")

# Check model complexity
print(f"Transformer params: {sum(p.numel() for p in model.parameters()):,}")
print(f"MLP params: {sum(p.numel() for p in model_mlp.parameters()):,}")

# Visualize hypothesis selection
selection_histograms(DO, TVM, num_epochs, rand_state, partial_perc)
```

## Key Differences to Expect

### Performance
- **Better gradient informativeness** → Better hypothesis selection
- **Potential better final R²** due to training on better-selected hypotheses
- **More stable training** due to residual connections and layer normalization

### Computational Cost
- ~15-20x more parameters than MLP
- ~2-3x slower training per epoch
- Worth it if gradient quality matters more than speed

### Selection Quality
Look at selection histograms:
- Should see **higher selection rates for correct hypotheses**
- Should see **lower selection rates for incorrect hypotheses**
- Should see **more consistent selection across epochs**

## Troubleshooting

### "Model too large for GPU"
```python
# Use CPU instead
DO = DataOperator(..., device="cpu")
```

### "Training is slow"
```python
# Reduce hidden_size or use MLP
model_type="mlp"  # Faster but less informed gradients
```

### "Selection is too aggressive/conservative"
```python
# Adjust nu parameter (SVM encompassing parameter)
nu=0.05  # More selective (fewer hypotheses)
nu=0.2   # Less selective (more hypotheses)
```

## Default Behavior

If you don't specify `model_type`, it defaults to `"mlp"`:

```python
# These are equivalent:
full_experiment(..., model_type="mlp")
full_experiment(...)  # Default is MLP
```

## Files Modified

- `GGH/models.py` - Added `TabPFNWrapper` class
- `GGH/inspector.py` - Saves model type in results
- `notebooks/Photoredox Yield.ipynb` - Added demo cells

All changes are backward compatible!
