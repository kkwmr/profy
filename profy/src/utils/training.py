"""Training utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LabelSmoothing(nn.Module):
    """Label smoothing for better generalization."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing."""
        n_classes = pred.size(1)
        
        # One-hot encode targets
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        
        # Smooth labels
        smooth_target = one_hot * self.confidence + self.smoothing / n_classes
        
        # Compute loss
        log_probs = F.log_softmax(pred, dim=1)
        loss = -(smooth_target * log_probs).sum(dim=1).mean()
        
        return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for better feature learning."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss."""
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Exclude diagonal
        mask.fill_diagonal_(0)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Mean log-likelihood for positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss


class BalancedBatchSampler:
    """Balanced batch sampler for handling class imbalance."""
    
    def __init__(self, labels: np.ndarray, batch_size: int, oversample: bool = True):
        self.labels = labels
        self.batch_size = batch_size
        self.oversample = oversample
        
        # Get class indices
        self.class_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.n_classes = len(self.class_indices)
        self.samples_per_class = batch_size // self.n_classes
        
    def __iter__(self):
        """Generate balanced batches."""
        # Shuffle indices within each class
        for label in self.class_indices:
            np.random.shuffle(self.class_indices[label])
        
        # Create batches
        n_batches = len(self.labels) // self.batch_size
        
        for _ in range(n_batches):
            batch_indices = []
            
            for label in self.class_indices:
                indices = self.class_indices[label]
                
                if self.oversample:
                    # Sample with replacement if needed
                    selected = np.random.choice(
                        indices, 
                        size=self.samples_per_class, 
                        replace=len(indices) < self.samples_per_class
                    )
                else:
                    # Sample without replacement
                    selected = np.random.choice(
                        indices,
                        size=min(self.samples_per_class, len(indices)),
                        replace=False
                    )
                
                batch_indices.extend(selected)
            
            # Shuffle batch
            np.random.shuffle(batch_indices)
            yield batch_indices[:self.batch_size]
    
    def __len__(self):
        return len(self.labels) // self.batch_size


class EarlyStopping:
    """Early stopping with model restoration."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max', restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module, epoch: int) -> bool:
        """Check if should stop training."""
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.restore_best:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        
        # Check improvement
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        # Check if should stop
        should_stop = self.counter >= self.patience
        
        if should_stop and self.restore_best and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"Restored best weights from epoch {self.best_epoch}")
        
        return should_stop


class MixupCutmix:
    """Combined Mixup and Cutmix augmentation."""
    
    def __init__(self, mixup_alpha: float = 1.0, cutmix_alpha: float = 1.0, prob: float = 0.5, switch_prob: float = 0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup or cutmix."""
        if not training or np.random.random() > self.prob:
            return x, y, y, 1.0
        
        batch_size = x.size(0)
        
        # Get random permutation
        index = torch.randperm(batch_size).to(x.device)
        
        # Choose mixup or cutmix
        if np.random.random() < self.switch_prob:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixed_x = lam * x + (1 - lam) * x[index]
        else:
            # Cutmix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            
            # Get random box
            W = x.size(-1)
            cut_ratio = np.sqrt(1. - lam)
            cut_w = int(W * cut_ratio)
            
            # Random position
            cx = np.random.randint(W)
            
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            
            # Apply cutmix
            mixed_x = x.clone()
            mixed_x[..., bbx1:bbx2] = x[index][..., bbx1:bbx2]
            
            # Adjust lambda
            lam = 1 - (bbx2 - bbx1) / W
        
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


def compute_adaptive_loss_weights(predictions: torch.Tensor, targets: torch.Tensor, 
                                current_weights: Optional[torch.Tensor] = None, 
                                momentum: float = 0.9) -> torch.Tensor:
    """Compute adaptive loss weights based on class performance."""
    with torch.no_grad():
        # Get predictions
        preds = predictions.argmax(dim=1)
        
        # Compute per-class accuracy
        n_classes = predictions.size(1)
        class_correct = torch.zeros(n_classes)
        class_total = torch.zeros(n_classes)
        
        for c in range(n_classes):
            mask = targets == c
            if mask.sum() > 0:
                class_correct[c] = (preds[mask] == c).float().sum()
                class_total[c] = mask.sum()
        
        # Compute class accuracy
        class_acc = class_correct / (class_total + 1e-8)
        
        # Compute weights (inverse of accuracy)
        weights = 1.0 / (class_acc + 0.1)
        weights = weights / weights.mean()
        
        # Apply momentum if we have previous weights
        if current_weights is not None:
            weights = momentum * current_weights + (1 - momentum) * weights
        
    return weights


def setup_logger(output_dir: str, name: str = 'training') -> logging.Logger:
    """Setup logger for training."""
    import os
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int, 
                   metrics: Dict, filepath: str, **kwargs) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        **kwargs
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model: nn.Module, optimizer=None, 
                   scheduler=None, device='cpu') -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint