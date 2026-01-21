# TabPFN-Style Transformer Integration for GGH

## Overview

This integration adds a transformer-based architecture (inspired by TabPFN) to the GGH framework, enabling the use of more sophisticated models that produce more informed gradients for hypothesis selection.

**PyTorch Compatibility:** Works with PyTorch 1.8.0+ (tested with PyTorch 1.8.2 and 2.0+)

## Motivation

The original hypothesis: **Better models produce more informed gradients, which benefit gradient-guided hypothesis selection more than model performance alone.**

Simple 2-layer MLPs produce gradients that capture only basic feature interactions. Transformer-based models with attention mechanisms can:
1. Capture complex feature relationships through self-attention
2. Maintain better gradient flow via residual connections
3. Produce more discriminative gradient patterns for clustering
4. Create richer internal representations

## Implementation

### New Model: `TabPFNWrapper`

Located in `GGH/models.py`, this transformer-inspired architecture includes:

- **Feature Projection**: Linear layer to project inputs to hidden dimension
- **Multi-Head Attention**: 4-head self-attention for feature interactions
- **Feed-Forward Network**: 2-layer FFN with GELU activation (transformer-style)
- **Layer Normalization**: Stabilizes training and gradient flow
- **Residual Connections**: Preserves gradient signal through the network
- **Output Head**: Final projection to target dimension

### Architecture Comparison

| Feature | MLP | TabPFN-Style Transformer |
|---------|-----|--------------------------|
| Hidden Layers | 2 | Multi-layer with attention |
| Typical Hidden Size | 32 | 64-128 |
| Parameters (~) | 2,000 | 34,000+ |
| Attention | No | Yes (4 heads) |
| Residuals | No | Yes |
| LayerNorm | No | Yes |
| Activation | ReLU | GELU |

### Changes Made

1. **`GGH/models.py`**:
   - Added `TabPFNWrapper` class with transformer architecture
   - Updated `initialize_model()` to accept `model_type` parameter
   - Updated `load_model()` to restore model type from saved JSON

2. **`GGH/inspector.py`**:
   - Modified `save_train_val_logs()` to save `model_type` in results JSON
   - Added backward compatibility for loading old models

3. **`notebooks/Photoredox Yield.ipynb`**:
   - Updated `full_experiment()` to accept `model_type` parameter
   - Added demonstration cells showing transformer vs MLP comparison
   - Added test cell to verify model initialization

## Usage

### Basic Usage

```python
from GGH.models import initialize_model
from GGH.data_ops import DataOperator

# Initialize with transformer
model = initialize_model(
    DO, dataloader, 
    hidden_size=64,  # Larger for transformer
    rand_state=42, 
    dropout=0.05, 
    model_type="tabpfn"  # Use transformer
)

# Or use standard MLP (default)
model = initialize_model(
    DO, dataloader, 
    hidden_size=32, 
    rand_state=42, 
    dropout=0.05, 
    model_type="mlp"  # Use MLP
)
```

### In GGH Experiments

```python
DO, TVM, model = full_experiment(
    use_info="use hypothesis",
    DO=DO,
    INSPECT=INSPECT,
    batch_size=batch_size,
    hidden_size=64,  # Increase for transformer
    output_size=output_size,
    num_epochs=60,
    rand_state=42,
    results_path=results_path,
    dropout=0.05,
    lr=0.001,
    nu=0.1,
    normalize_grads_contx=False,
    use_context=True,
    final_analysis=False,
    model_type="tabpfn"  # NEW: Specify model type
)
```

## Expected Benefits

### For Gradient Quality

1. **Richer Gradient Patterns**: Attention mechanisms create diverse gradient signatures across different hypotheses
2. **Better Separability**: One-Class SVM in GGH should find clearer boundaries between correct/incorrect hypotheses
3. **Stable Gradient Flow**: LayerNorm + residuals prevent gradient degradation

### For Overall Performance

1. **Better Hypothesis Selection**: More informed gradients → more accurate clustering
2. **Improved Final Model**: Training on better-selected hypotheses → better predictions
3. **Multiplicative Effect**: Better gradients → better selection → better data → better model (positive feedback loop)

## Testing

Run the test cells in the notebook to:

1. Verify model initialization works correctly
2. Compare transformer vs MLP performance on the same dataset
3. Examine selection histograms to see if transformers produce better hypothesis selection

## Recommendations

### Hyperparameters

- **Hidden Size**: Use 64-128 for transformers (vs 32 for MLP)
- **Learning Rate**: May need slight adjustment (try 0.0005-0.001)
- **Dropout**: 0.05-0.1 works well with transformers
- **Nu (SVM parameter)**: May need retuning as gradient distributions change

### When to Use Each Model

**Use TabPFN-Style Transformer when**:
- Dataset is medium to large (>500 samples)
- Feature interactions are complex
- You want maximum gradient informativeness
- Computational resources allow

**Use MLP when**:
- Dataset is very small (<200 samples)
- Features are relatively independent
- Fast training is critical
- Limited computational resources

## Backward Compatibility

All existing code continues to work without changes. The default model type is "mlp", so:

```python
# This still works and uses MLP
model = initialize_model(DO, dataloader, hidden_size, rand_state, dropout=0.05)
```

## Future Extensions

Potential improvements:
1. Add more transformer layers for very large datasets
2. Implement cross-attention between features and hypotheses
3. Add pre-training on similar datasets
4. Experiment with different attention patterns (e.g., causal, sparse)
5. Implement actual TabPFN classifier integration (currently using custom transformer)

## Notes

- The implementation is called "TabPFN-style" because it uses transformer architecture inspired by TabPFN, not the actual TabPFN pre-trained model
- The `tabpfn` package is installed but not directly used (custom implementation instead)
- This provides full gradient access needed for GGH, which pre-trained TabPFN wouldn't easily provide

## References

- GGH paper: [Add reference when available]
- TabPFN: Hollmann et al., "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"
- Attention mechanism: Vaswani et al., "Attention Is All You Need"
