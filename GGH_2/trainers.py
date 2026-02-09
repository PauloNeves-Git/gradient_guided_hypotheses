"""Trainer classes for GGH."""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad


class UnbiasedTrainer:
    """Train on ALL hypotheses equally. Track per-hypothesis losses and gradients."""

    def __init__(self, DO, model, lr=0.001, device=None):
        self.DO = DO
        self.model = model
        self.device = device if device is not None else DO.device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none')
        self.hyp_per_sample = DO.num_hyp_comb

        self.loss_history = {}
        self.gradient_history = {}

    def train_epoch(self, dataloader, epoch, track_data=False):
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (inputs, targets, global_ids) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(inputs)
            individual_losses = self.criterion(predictions, targets).mean(dim=1)
            batch_loss = individual_losses.mean()

            if track_data:
                self._track_hypothesis_data(inputs, targets, global_ids, individual_losses)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _track_hypothesis_data(self, inputs, targets, global_ids, losses):
        self.model.eval()

        for i in range(len(inputs)):
            gid = global_ids[i].item()

            if gid not in self.loss_history:
                self.loss_history[gid] = []
            self.loss_history[gid].append(losses[i].item())

            inp = inputs[i:i+1].clone().requires_grad_(True)
            pred = self.model(inp)
            loss = nn.MSELoss()(pred, targets[i:i+1])

            params = list(self.model.parameters())
            grad_param = grad(loss, params[-2], retain_graph=False)[0]
            grad_vec = grad_param.flatten().detach().cpu().numpy()

            if gid not in self.gradient_history:
                self.gradient_history[gid] = []
            self.gradient_history[gid].append(grad_vec)

        self.model.train()

    def get_hypothesis_analysis(self):
        analysis = {}
        for gid in self.loss_history:
            analysis[gid] = {
                'avg_loss': np.mean(self.loss_history[gid]),
                'loss_std': np.std(self.loss_history[gid]),
                'avg_gradient': np.mean(self.gradient_history[gid], axis=0) if gid in self.gradient_history else None,
                'gradient_magnitude': np.mean([np.linalg.norm(g) for g in self.gradient_history.get(gid, [])]),
            }
        return analysis


class WeightedTrainer:
    """Train on ALL samples with continuous weights."""

    def __init__(self, DO, model, sample_weights, partial_gids, partial_weight, lr=0.001, device=None):
        self.DO = DO
        self.model = model
        self.device = device if device is not None else DO.device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none')
        self.hyp_per_sample = DO.num_hyp_comb

        self.sample_weights = sample_weights
        self.partial_gids = set(partial_gids)
        self.partial_weight = partial_weight

    def train_epoch(self, dataloader, epoch, track_data=False):
        self.model.train()
        total_loss = 0
        total_weight = 0

        for batch_idx, (inputs, targets, global_ids) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(inputs)
            individual_losses = self.criterion(predictions, targets).mean(dim=1)

            weights = torch.zeros(len(inputs), device=self.device)

            for i, gid in enumerate(global_ids):
                gid = gid.item()
                if gid in self.partial_gids:
                    weights[i] = self.partial_weight
                elif gid in self.sample_weights:
                    weights[i] = self.sample_weights[gid]

            if weights.sum() == 0:
                continue

            weighted_loss = (individual_losses * weights).sum() / weights.sum()

            self.optimizer.zero_grad()
            weighted_loss.backward()
            self.optimizer.step()

            total_loss += weighted_loss.item() * weights.sum().item()
            total_weight += weights.sum().item()

        return total_loss / total_weight if total_weight > 0 else 0


class RemainingDataScorer:
    """Score remaining data using biased model."""

    def __init__(self, DO, model, remaining_sample_indices, device=None):
        self.DO = DO
        self.model = model
        self.device = device if device is not None else DO.device
        self.hyp_per_sample = DO.num_hyp_comb
        self.remaining_sample_indices = set(remaining_sample_indices)

        self.loss_scores = {}
        self.gradient_history = {}

    def compute_scores(self, dataloader, n_passes=5):
        self.model.eval()

        for pass_idx in range(n_passes):
            for inputs, targets, global_ids in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                for i in range(len(inputs)):
                    gid = global_ids[i].item()
                    sample_idx = gid // self.hyp_per_sample

                    if sample_idx not in self.remaining_sample_indices:
                        continue

                    inp = inputs[i:i+1].clone().requires_grad_(True)
                    pred = self.model(inp)
                    loss = nn.MSELoss()(pred, targets[i:i+1])

                    if gid not in self.loss_scores:
                        self.loss_scores[gid] = []
                    self.loss_scores[gid].append(loss.item())

                    params = list(self.model.parameters())
                    grad_param = grad(loss, params[-2], retain_graph=False)[0]
                    grad_vec = grad_param.flatten().detach().cpu().numpy()

                    if gid not in self.gradient_history:
                        self.gradient_history[gid] = []
                    self.gradient_history[gid].append(grad_vec)

    def get_analysis(self):
        analysis = {}
        for gid in self.loss_scores:
            analysis[gid] = {
                'avg_loss': np.mean(self.loss_scores[gid]),
                'loss_std': np.std(self.loss_scores[gid]),
                'avg_gradient': np.mean(self.gradient_history[gid], axis=0) if gid in self.gradient_history else None,
                'gradient_magnitude': np.mean([np.linalg.norm(g) for g in self.gradient_history.get(gid, [])]),
            }
        return analysis
