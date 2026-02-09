"""Training functions for GGH."""

import torch

from .trainers import WeightedTrainer
from .utils import create_dataloader_with_gids


def train_with_soft_weights(DO, model, sample_weights, partial_gids, partial_weight, lr, n_epochs=200, batch_size=32):
    """Train model with soft weights and validation-based epoch selection.

    Args:
        DO: DataOperator instance
        model: PyTorch model to train
        sample_weights: dict mapping gid -> weight
        partial_gids: set of GIDs that are partial (known correct)
        partial_weight: weight to use for partial samples
        lr: learning rate
        n_epochs: number of training epochs
        batch_size: batch size for dataloader

    Returns:
        tuple: (trained_model, best_epoch, best_val_loss)
    """
    dataloader = create_dataloader_with_gids(DO, batch_size)

    # Use DO.device for proper device handling
    trainer = WeightedTrainer(DO, model, sample_weights=sample_weights,
                             partial_gids=partial_gids, partial_weight=partial_weight, lr=lr,
                             device=DO.device)

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None

    for epoch in range(n_epochs):
        trainer.train_epoch(dataloader, epoch, track_data=False)

        model.eval()
        with torch.no_grad():
            val_inputs, val_targets = DO.get_validation_tensors(use_info="full info")
            val_preds = model(val_inputs)
            val_loss = torch.nn.functional.mse_loss(val_preds, val_targets).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_epoch, best_val_loss
