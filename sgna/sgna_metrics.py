
from typing import Dict, Any
import torch
import torch.nn.functional as F

class MetricsLogger:
    def __init__(self):
        self.metrics: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "mean_insep": [],
            "temporal_violations": [],
            "epoch_times": [],
        }

    def log_epoch(self, **kwargs: Any):
        if "epoch_time" in kwargs and "epoch_times" not in kwargs:
            kwargs["epoch_times"] = kwargs.pop("epoch_time")
        for k, v in kwargs.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

@torch.no_grad()
def compute_inseparability(z: torch.Tensor) -> float:
    if z.size(0) < 2:
        return 0.0
    z = F.normalize(z, dim=1)
    sims = torch.matmul(z, z.t()).abs()
    sims = sims - torch.eye(sims.size(0), device=sims.device) * sims
    n = sims.numel() - sims.size(0)
    return sims.sum().div(max(n, 1)).item()

def check_temporal_violations(adj: torch.Tensor) -> int:
    upper = torch.triu(adj, diagonal=1)
    return int((upper > 0).sum().item())
