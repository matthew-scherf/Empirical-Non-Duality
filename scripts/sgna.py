"""
TUOS-SGNA: Substrate-Grounded Neural Architecture
==================================================
Machine-verified non-dual metaphysics as neural architecture.

Based on the Isabelle/HOL formalization:
"Complete Formal Axiomatization of Empirical Non-Duality"
Scherf, M. (2025). DOI: https://doi.org/10.5281/zenodo.17388701

This implementation operationalizes axioms A1-A5, C1-C3, S1-S2, and Time_monotone
as runtime-enforced architectural constraints in PyTorch.

License: BSD-3-Clause
Version: 2.1.0 - Improved temporal constraint enforcement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Set, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np


__version__ = "2.1.0-sgna"
__all__ = [
    "SubstrateRegistry", "SubstrateLayer", "GaugeInvariantClassifier",
    "PhenomenonGraph", "TUOSConfig", "AdaptiveTrainer", "MetricsLogger",
    "substrate_context"
]


# =============================================================================
# CONFIGURATION SYSTEM
# =============================================================================

@dataclass
class TUOSConfig:
    """Configuration for TUOS-SGNA architecture."""
    
    substrate_dim: int = 256
    num_classes: int = 2
    input_dim: int = 64
    num_epochs: int = 40
    learning_rate: float = 0.001
    batch_size: int = 32
    
    n_phenomena: int = 200
    edge_probability: float = 0.03
    
    use_adaptive_weights: bool = True
    use_parametric: bool = False
    use_harmonic: bool = False
    use_dependent_arising: bool = False
    
    lambda_insep: float = 0.1
    lambda_time: float = 5.0  # Increased from 2.0 for stronger temporal enforcement
    target_inseparability: float = 0.85
    target_temporal_violation_rate: float = 0.0
    
    log_interval: int = 5


# =============================================================================
# SUBSTRATE REGISTRY (A1, A2: Uniqueness)
# =============================================================================

class SubstrateRegistry:
    """
    Global registry ensuring substrate uniqueness (A1, A2).
    Only one substrate Omega can exist per context.
    """
    _instance = None
    _omega = None
    _context_depth = 0
    
    @classmethod
    def get_substrate(cls, substrate_dim: int) -> torch.Tensor:
        """Get or create the unique substrate."""
        if cls._omega is None:
            cls._omega = nn.Parameter(torch.randn(substrate_dim) * 0.01)
        return cls._omega
    
    @classmethod
    def reset(cls):
        """Reset substrate (for testing only)."""
        cls._omega = None
        cls._context_depth = 0
    
    @classmethod
    def enter_context(cls):
        """Enter substrate context."""
        cls._context_depth += 1
    
    @classmethod
    def exit_context(cls):
        """Exit substrate context."""
        cls._context_depth -= 1
        if cls._context_depth == 0:
            cls.reset()


@contextmanager
def substrate_context():
    """Context manager for substrate scope."""
    SubstrateRegistry.enter_context()
    try:
        yield
    finally:
        SubstrateRegistry.exit_context()


# =============================================================================
# CORE SUBSTRATE LAYER (A3, A4: Presentation)
# =============================================================================

class SubstrateLayer(nn.Module):
    """
    The unique ontic substrate Omega.
    
    Implements:
    - A1, A2: Substrate uniqueness via registry
    - A3, A4: Phenomena as presentations of Omega
    - A5: Inseparability maintained through architecture
    """
    
    def __init__(self, config: TUOSConfig):
        super().__init__()
        self.substrate_dim = config.substrate_dim
        self.omega = SubstrateRegistry.get_substrate(config.substrate_dim)
        self.presentation_ops = nn.ModuleDict()
    
    def register_presentation_mode(self, mode: str, input_dim: int, **kwargs):
        """Register a presentation mode (modality)."""
        self.presentation_ops[mode] = nn.Sequential(
            nn.Linear(self.substrate_dim + input_dim, self.substrate_dim * 2),
            nn.LayerNorm(self.substrate_dim * 2),
            nn.GELU(),
            nn.Linear(self.substrate_dim * 2, self.substrate_dim),
            nn.LayerNorm(self.substrate_dim)
        )
    
    def present(self, input_data: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Create presentations of phenomena from Omega (A4).
        All phenomena are presentations of the substrate, not independent entities.
        """
        if mode not in self.presentation_ops:
            raise ValueError(f"Presentation mode '{mode}' not registered")
        
        batch_size = input_data.shape[0]
        omega_expanded = self.omega.unsqueeze(0).expand(batch_size, -1)
        combined = torch.cat([omega_expanded, input_data], dim=-1)
        
        presentation = self.presentation_ops[mode](combined)
        return presentation
    
    def verify_inseparability(self, presentation: torch.Tensor) -> torch.Tensor:
        """
        Verify phenomenon is inseparable from Omega (A5).
        Returns cosine similarity scores.
        """
        omega_expanded = self.omega.unsqueeze(0).expand(presentation.shape[0], -1)
        return F.cosine_similarity(presentation, omega_expanded, dim=-1)
    
    def functional_dependence(self, presentation: torch.Tensor, 
                             input_data: torch.Tensor) -> torch.Tensor:
        """
        Measure functional dependence on Omega.
        Computes gradient of presentation with respect to omega.
        """
        if not input_data.requires_grad:
            input_data = input_data.clone().detach().requires_grad_(True)
        
        jacobian_norm = torch.autograd.grad(
            presentation.sum(), self.omega,
            create_graph=True, retain_graph=True
        )[0].norm()
        
        return jacobian_norm


# =============================================================================
# GAUGE-INVARIANT CLASSIFIER (S1, S2: Gauge Structure)
# =============================================================================

class GaugeInvariantClassifier(nn.Module):
    """
    Classifier that separates substrate predictions from coordinate representations.
    
    Implements:
    - S1: Substrate predictions are gauge-invariant
    - S2: Coordinate representations are frame-dependent
    """
    
    def __init__(self, substrate: SubstrateLayer, config: TUOSConfig = None):
        super().__init__()
        self.substrate = substrate
        
        if config is None:
            # Default fallback
            num_classes = 2
            substrate_dim = substrate.substrate_dim
        else:
            num_classes = config.num_classes
            substrate_dim = config.substrate_dim
        
        self.num_classes = num_classes
        
        self.substrate_classifier = nn.Sequential(
            nn.Linear(substrate_dim, substrate_dim),
            nn.LayerNorm(substrate_dim),
            nn.GELU(),
            nn.Linear(substrate_dim, num_classes)
        )
        
        self.frames = nn.ModuleDict({
            'default': nn.Identity(),
            'rotated': nn.Linear(num_classes, num_classes, bias=False),
        })
    
    def predict(self, substrate_presentation: torch.Tensor, 
                frame: str = 'default') -> Dict[str, torch.Tensor]:
        """
        Make predictions maintaining gauge invariance.
        
        Returns:
            substrate_prediction: Frame-invariant logits
            coordinate_prediction: Frame-dependent logits
            inseparability: A5 verification scores
        """
        substrate_logits = self.substrate_classifier(substrate_presentation)
        coordinate_logits = self.frames[frame](substrate_logits)
        
        insep = self.substrate.verify_inseparability(substrate_presentation)
        
        return {
            'substrate_prediction': substrate_logits,
            'coordinate_prediction': coordinate_logits,
            'inseparability': insep
        }


# =============================================================================
# PHENOMENON GRAPH (C1-C3: Causal Structure, Time_monotone)
# =============================================================================

class PhenomenonGraph(nn.Module):
    """
    Learnable DAG representing discovered causal structure and emergent time.
    
    Implements:
    - C1-C3: Causal relation properties
    - Time_monotone: Temporal ordering of phenomena
    - Learnable edges: Network discovers which causal relationships exist
    """
    
    def __init__(self, config: TUOSConfig):
        super().__init__()
        self.n = config.n_phenomena
        self.T = nn.Parameter(torch.zeros(self.n))
        self.edge_margin = 0.1
        
        # Learnable edge weights: positive = causal edge exists
        # Initialize with small random values, will be pruned during training
        self.edge_logits = nn.Parameter(torch.randn(self.n, self.n) * 0.1)
        
        # Mask to enforce DAG structure (only i->j where i<j)
        self.register_buffer('dag_mask', torch.triu(torch.ones(self.n, self.n), diagonal=1))
        
        # Temperature for soft gating (starts high, anneals down)
        self.temperature = 5.0
        self.min_temperature = 0.5
        
    def get_edge_weights(self) -> torch.Tensor:
        """
        Get soft edge weights using sigmoid gating.
        Returns values in [0,1] indicating edge strength.
        """
        # Apply DAG mask and sigmoid
        masked_logits = self.edge_logits * self.dag_mask
        edge_weights = torch.sigmoid(masked_logits / self.temperature)
        return edge_weights
    
    def get_active_edges(self, threshold: float = 0.5) -> Set[Tuple[int, int]]:
        """Get edges with weight above threshold."""
        edge_weights = self.get_edge_weights()
        edges = set()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if edge_weights[i, j].item() > threshold:
                    edges.add((i, j))
        return edges
    
    @property
    def edges(self) -> Set[Tuple[int, int]]:
        """Compatibility property for existing code."""
        return self.get_active_edges()
    
    def anneal_temperature(self, epoch: int, total_epochs: int):
        """Gradually reduce temperature to make edges more discrete."""
        progress = epoch / total_epochs
        self.temperature = max(
            self.min_temperature,
            5.0 * (1 - progress)
        )
    
    def build_random_dag(self):
        """
        Initialize edge logits to encourage sparsity.
        Negative bias means most edges start "off" and must be learned.
        """
        # Bias towards no edges (network must learn to activate them)
        with torch.no_grad():
            self.edge_logits.data = torch.randn(self.n, self.n) * 0.5 - 2.0
            self.edge_logits.data *= self.dag_mask
    
    def time_monotone_loss(self) -> torch.Tensor:
        """
        Compute weighted loss for time monotonicity violations.
        Only penalize violations on edges that are "active" (high weight).
        Vectorized for efficiency.
        """
        edge_weights = self.get_edge_weights()
        
        # Vectorized: compute all potential violations at once
        # For edge i->j, violation if T[i] >= T[j]
        T_i = self.T.unsqueeze(1)  # (n, 1)
        T_j = self.T.unsqueeze(0)  # (1, n)
        
        # Violation matrix: how much T[i] exceeds T[j] + margin
        violations = F.relu(T_i - T_j + self.edge_margin)
        
        # Weight by edge strength and DAG mask
        weighted_violations = violations * edge_weights * self.dag_mask
        
        # Sum over all edges, normalize by total edge weight
        total_weight = (edge_weights * self.dag_mask).sum()
        
        if total_weight < 0.01:
            return torch.tensor(0.0, device=self.T.device)
        
        return weighted_violations.sum() / (total_weight + 1e-8)
    
    def get_violation_details(self) -> List[Tuple[int, int, float]]:
        """Get detailed violation information for active edges."""
        violations = []
        active_edges = self.get_active_edges(threshold=0.5)
        
        for i, j in active_edges:
            if self.T[i].item() >= self.T[j].item():
                violations.append((i, j, self.T[i].item() - self.T[j].item()))
        
        return violations
    
    def get_causal_regularization(self) -> torch.Tensor:
        """
        Regularization to encourage:
        1. Temporal spread (avoid all times being similar)
        2. Sparse but meaningful edges (L1 penalty on edge weights)
        """
        # Encourage time standard deviation
        time_std_loss = F.relu(1.0 - self.T.std())
        
        # Encourage sparsity: L1 penalty on active edges only (upper triangle)
        edge_weights = self.get_edge_weights()
        active_weights = edge_weights * self.dag_mask
        sparsity_loss = active_weights.sum() / (self.n * (self.n - 1) / 2)  # Normalize by max possible edges
        
        # Combined regularization
        return time_std_loss + 0.05 * sparsity_loss  # Reduced weight from 0.1
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the learned graph structure."""
        edge_weights = self.get_edge_weights()
        active_edges = self.get_active_edges(threshold=0.5)
        
        return {
            'n_active_edges': len(active_edges),
            'mean_edge_weight': edge_weights[edge_weights > 0.01].mean().item() if (edge_weights > 0.01).any() else 0.0,
            'edge_sparsity': 1.0 - (len(active_edges) / (self.n * (self.n - 1) / 2)),
            'temperature': self.temperature,
            'time_std': self.T.std().item(),
            'time_range': (self.T.min().item(), self.T.max().item())
        }


# =============================================================================
# ADAPTIVE TRAINING SYSTEM
# =============================================================================

class MetricsLogger:
    """Logger for tracking training metrics."""
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {
            'accuracy': [],
            'task_loss': [],
            'insep_loss': [],
            'time_loss': [],
            'total_loss': [],
            'mean_insep': [],
            'num_violations': [],
            'lambda_insep': [],
            'lambda_time': []
        }
    
    def log(self, metrics: Dict[str, float]):
        """Log metrics for current epoch."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of final metrics."""
        return {key: values[-1] if values else 0.0 
                for key, values in self.history.items()}


class AdaptiveTrainer:
    """
    Adaptive training system with dynamic weight adjustment.
    """
    
    def __init__(self, substrate: SubstrateLayer, 
                 classifier: GaugeInvariantClassifier,
                 graph: PhenomenonGraph,
                 config: TUOSConfig):
        self.substrate = substrate
        self.classifier = classifier
        self.graph = graph
        self.config = config
        
        if config.use_adaptive_weights:
            self.lambda_insep = nn.Parameter(torch.tensor(config.lambda_insep))
            self.lambda_time = nn.Parameter(torch.tensor(config.lambda_time))
        else:
            self.lambda_insep = config.lambda_insep
            self.lambda_time = config.lambda_time
        
        self.metrics_logger = MetricsLogger()
    
    def compute_loss(self, presentations: torch.Tensor,
                    predictions: Dict[str, torch.Tensor],
                    labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss with all constraints."""
        
        task_loss = F.cross_entropy(
            predictions['substrate_prediction'], labels
        )
        
        insep_scores = predictions['inseparability']
        target = torch.ones_like(insep_scores) * self.config.target_inseparability
        insep_loss = F.mse_loss(insep_scores, target)
        
        time_loss = self.graph.time_monotone_loss()
        causal_reg = self.graph.get_causal_regularization()
        
        lambda_insep = self.lambda_insep if isinstance(self.lambda_insep, float) else self.lambda_insep.abs()
        lambda_time = self.lambda_time if isinstance(self.lambda_time, float) else self.lambda_time.abs()
        
        # Combined loss with both time components
        total_loss = (task_loss + 
                     lambda_insep * insep_loss + 
                     lambda_time * (time_loss + 0.1 * causal_reg))
        
        metrics = {
            'task_loss': task_loss.item(),
            'insep_loss': insep_loss.item(),
            'time_loss': time_loss.item(),
            'causal_reg': causal_reg.item(),
            'total_loss': total_loss.item(),
            'mean_inseparability': insep_scores.mean().item(),
            'lambda_insep': lambda_insep if isinstance(lambda_insep, float) else lambda_insep.item(),
            'lambda_time': lambda_time if isinstance(lambda_time, float) else lambda_time.item()
        }
        
        return total_loss, metrics
    
    def adapt_weights(self, metrics: Dict[str, float]):
        """Adapt loss weights based on current performance."""
        if not self.config.use_adaptive_weights:
            return
        
        if not isinstance(self.lambda_insep, nn.Parameter):
            return
        
        # Adapt inseparability weight
        insep_gap = self.config.target_inseparability - metrics['mean_inseparability']
        
        if insep_gap > 0.05:
            self.lambda_insep.data += 0.05
        elif insep_gap < -0.05:
            self.lambda_insep.data = torch.max(
                self.lambda_insep.data - 0.05,
                torch.tensor(0.01)
            )
        
        # More aggressive adaptation for temporal violations
        violation_rate = metrics.get('violation_rate', 0.0)
        
        if violation_rate > self.config.target_temporal_violation_rate + 0.1:
            # High violations: increase weight significantly
            self.lambda_time.data += 0.5
        elif violation_rate > self.config.target_temporal_violation_rate:
            # Moderate violations: increase moderately
            self.lambda_time.data += 0.1
        elif violation_rate == 0.0:
            # Zero violations: can reduce slightly but keep some pressure
            self.lambda_time.data = torch.max(
                self.lambda_time.data - 0.05,
                torch.tensor(1.0)  # Keep minimum at 1.0, not 0.01
            )
        
        # Cap maximum to prevent instability
        self.lambda_time.data = torch.min(
            self.lambda_time.data,
            torch.tensor(50.0)
        )


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_tuos(config: TUOSConfig, X: torch.Tensor, y: torch.Tensor,
               device: str = 'cpu') -> Dict[str, Any]:
    """
    Train TUOS-SGNA on provided data.
    
    Args:
        config: Configuration object
        X: Input data tensor
        y: Label tensor
        device: Device to train on
    
    Returns:
        Dictionary with trained models and metrics
    """
    
    with substrate_context():
        substrate = SubstrateLayer(config).to(device)
        substrate.register_presentation_mode('data', config.input_dim)
        
        classifier = GaugeInvariantClassifier(substrate, config).to(device)
        
        graph = PhenomenonGraph(config)
        graph.build_random_dag()  # Initialize before moving to device
        graph = graph.to(device)   # Now move everything to device
        
        trainer = AdaptiveTrainer(substrate, classifier, graph, config)
        
        param_dict = {}
        for p in substrate.parameters():
            param_dict[id(p)] = p
        for p in classifier.parameters():
            param_dict[id(p)] = p
        for p in graph.parameters():  # Get all graph parameters
            param_dict[id(p)] = p
        
        if config.use_adaptive_weights:
            if isinstance(trainer.lambda_insep, nn.Parameter):
                param_dict[id(trainer.lambda_insep)] = trainer.lambda_insep
            if isinstance(trainer.lambda_time, nn.Parameter):
                param_dict[id(trainer.lambda_time)] = trainer.lambda_time
        
        params = [p for p in param_dict.values() if p.is_leaf]
        
        optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
        
        X = X.to(device)
        y = y.to(device)
        
        print("Training with integrated axiom constraints...")
        
        for epoch in range(config.num_epochs):
            substrate.train()
            classifier.train()
            
            # Anneal temperature for edge gating
            graph.anneal_temperature(epoch, config.num_epochs)
            
            optimizer.zero_grad()
            
            presentations = substrate.present(X, mode='data')
            predictions = classifier.predict(presentations)
            
            loss, metrics = trainer.compute_loss(presentations, predictions, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            _, predicted = predictions['substrate_prediction'].max(1)
            accuracy = (predicted == y).float().mean().item()
            
            num_violations = len(graph.get_violation_details())
            violation_rate = num_violations / len(graph.edges) if len(graph.edges) > 0 else 0.0
            
            trainer.adapt_weights({
                'mean_inseparability': metrics['mean_inseparability'],
                'num_violations': num_violations,
                'violation_rate': violation_rate
            })
            
            metrics['accuracy'] = accuracy
            metrics['num_violations'] = num_violations
            trainer.metrics_logger.log(metrics)
            
            if epoch % config.log_interval == 0:
                print(f"Epoch {epoch:3d}: Acc={accuracy:.3f}, "
                      f"Insep={metrics['mean_inseparability']:.3f}, "
                      f"Viol={num_violations}")
        
        print(f"\nFinal: Acc={accuracy:.3f}, "
              f"Insep={metrics['mean_inseparability']:.3f}, "
              f"Violations={num_violations}/{len(graph.edges)}")
        
        return {
            'substrate': substrate,
            'classifier': classifier,
            'graph': graph,
            'trainer': trainer,
            'config': config
        }


if __name__ == "__main__":
    print("\nTUOS-SGNA: Substrate-Grounded Neural Architecture")
    print("="*70)
    
    config = TUOSConfig()
    
    torch.manual_seed(42)
    X = torch.randn(200, 64)
    y = (X[:, 0] > 0).long()
    
    results = train_tuos(config, X, y)
    
    print("\nDemonstration complete!")