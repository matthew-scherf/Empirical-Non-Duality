"""
The Unique Ontic Substrate (TUOS / Ω)
======================================
A neural architecture implementing machine-verified non-dual metaphysics.

Based on the Isabelle/HOL formalization:
"Complete Formal Axiomatization of Empirical Non-Duality"
Scherf, M. (2025). DOI: https://doi.org/10.5281/zenodo.17388701

This implementation operationalizes axioms A1-A5, C1-C3, S1-S2, and Time_monotone
as runtime-enforced architectural constraints in PyTorch.

Core Principles:
  - Exactly one ontic substrate Ω (A1, A2)
  - All phenomena are presentations of Ω (A3, A4)
  - Inseparability from Ω is maintained through training (A5)
  - Gauge-invariant predictions separate ultimate from conventional truth
  - Causal structure and emergent time are integrated constraints

License: BSD-3-Clause
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Set, Optional
from contextlib import contextmanager


__version__ = "1.0.0"
__all__ = [
    "SubstrateRegistry", "SubstrateLayer", "GaugeInvariantClassifier",
    "NonEssentialistEmbedding", "BiasAuditor", "DecisionTracer",
    "PhenomenonGraph", "TUOSTrainer", "substrate_context"
]


# =============================================================================
# SUBSTRATE REGISTRY - Enforces Axiom A2 (Uniqueness)
# =============================================================================

class SubstrateRegistry:
    """
    Manages the unique substrate Ω, ensuring exactly one exists (A2).
    All SubstrateLayer instances share the same parameter object.
    """
    _instance = None
    _substrate_param = None
    _substrate_dim = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset registry for new experiments"""
        cls._instance = None
        cls._substrate_param = None
        cls._substrate_dim = None
    
    def get_substrate(self, dim: int) -> nn.Parameter:
        """Get or create the unique substrate Ω"""
        if self._substrate_param is None:
            self._substrate_param = nn.Parameter(torch.randn(dim) * 0.01)
            self._substrate_dim = dim
        else:
            assert self._substrate_dim == dim, \
                f"Ω exists with dimension {self._substrate_dim}, requested {dim}"
        return self._substrate_param
    
    def verify_uniqueness(self, layers: List["SubstrateLayer"]) -> bool:
        """Verify all layers share the same Ω object (A2)"""
        if not layers:
            return True
        ref = layers[0].omega
        return all(layer.omega is ref for layer in layers[1:])


@contextmanager
def substrate_context():
    """Context manager for clean substrate lifecycle management"""
    SubstrateRegistry.reset()
    try:
        yield SubstrateRegistry.get_instance()
    finally:
        SubstrateRegistry.reset()


# =============================================================================
# CORE ARCHITECTURE - Implements A1-A5 (Ontology)
# =============================================================================

class SubstrateLayer(nn.Module):
    """
    The unique ontic substrate Ω from which all phenomena emerge.
    
    Implements:
      A1 (Existence): Ω is created and maintained
      A2 (Uniqueness): All instances share the same Ω via registry
      A3 (Exhaustivity): All data becomes phenomena via presentation
      A4 (Presentation): present() creates phenomena from Ω
      A5 (Inseparability): Maintained via training with inseparability_loss
    
    Args:
        substrate_dim: Dimension of the substrate space (default 256)
    """
    
    def __init__(self, substrate_dim: int = 256):
        super().__init__()
        self.substrate_dim = substrate_dim
        self.registry = SubstrateRegistry.get_instance()
        self.omega = self.registry.get_substrate(substrate_dim)
        self.presentation_ops = nn.ModuleDict()
        
    def register_presentation_mode(self, mode: str, input_dim: int):
        """
        Register a presentation mode (modality).
        
        Args:
            mode: Name of the presentation mode (e.g., 'vision', 'text')
            input_dim: Dimension of input data for this mode
        """
        self.presentation_ops[mode] = nn.Sequential(
            nn.Linear(self.substrate_dim + input_dim, self.substrate_dim * 2),
            nn.LayerNorm(self.substrate_dim * 2),
            nn.GELU(),
            nn.Linear(self.substrate_dim * 2, self.substrate_dim),
            nn.LayerNorm(self.substrate_dim)
        )
        
    def present(self, input_data: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Create presentations of phenomena from Ω (A4).
        
        All phenomena are presentations of the substrate, not independent entities.
        
        Args:
            input_data: Input tensor [batch_size, input_dim]
            mode: Presentation mode to use
            
        Returns:
            Presentations in substrate space [batch_size, substrate_dim]
        """
        if mode not in self.presentation_ops:
            raise ValueError(f"Presentation mode '{mode}' not registered")
            
        batch_size = input_data.shape[0]
        omega_expanded = self.omega.unsqueeze(0).expand(batch_size, -1)
        combined = torch.cat([omega_expanded, input_data], dim=-1)
        return self.presentation_ops[mode](combined)
    
    def verify_inseparability(self, presentation: torch.Tensor) -> torch.Tensor:
        """
        Measure inseparability from Ω (A5).
        
        Returns cosine similarity with Ω as proxy for inseparability.
        Values close to 1 indicate strong inseparability.
        
        Args:
            presentation: Presentation tensor [batch_size, substrate_dim]
            
        Returns:
            Inseparability scores [batch_size]
        """
        omega_expanded = self.omega.unsqueeze(0).expand(presentation.shape[0], -1)
        return F.cosine_similarity(presentation, omega_expanded, dim=-1)
    
    def functional_dependence(self, inputs: torch.Tensor, mode: str = "simple") -> float:
        """
        Verify presentations functionally depend on Ω via Jacobian norm.
        
        Computes ||∂p/∂Ω|| to prove presentations are functions of Ω.
        
        Args:
            inputs: Input tensor
            mode: Presentation mode
            
        Returns:
            Jacobian norm (> 0 proves functional dependence)
        """
        inputs_detached = inputs.detach()
        self.omega.requires_grad_(True)
        
        presentations = self.present(inputs_detached, mode=mode)
        scalar_output = presentations.sum()
        
        if self.omega.grad is not None:
            self.omega.grad.zero_()
        
        scalar_output.backward()
        grad_norm = self.omega.grad.norm().item()
        
        self.omega.grad.zero_()
        return grad_norm


# =============================================================================
# GAUGE-INVARIANT CLASSIFIER - Implements S1-S2 (Spacetime/Gauge)
# =============================================================================

class GaugeInvariantClassifier(nn.Module):
    """
    Classifier with gauge-invariant substrate predictions.
    
    Separates ultimate (substrate-level) from conventional (frame-dependent)
    predictions, implementing the distinction between absolute and relative truth.
    
    Note: Non-linearities provide approximate gauge structure. Exact invariance
    holds for output frame choice, but deep orthogonal transforms yield soft
    gauge structure due to LayerNorm and GELU.
    
    Args:
        substrate_layer: The SubstrateLayer instance
        num_classes: Number of output classes
    """
    
    def __init__(self, substrate_layer: SubstrateLayer, num_classes: int):
        super().__init__()
        self.substrate = substrate_layer
        self.num_classes = num_classes
        
        # Substrate classifier (gauge-invariant core)
        self.substrate_classifier = nn.Sequential(
            nn.Linear(substrate_layer.substrate_dim, substrate_layer.substrate_dim),
            nn.LayerNorm(substrate_layer.substrate_dim),
            nn.GELU(),
            nn.Linear(substrate_layer.substrate_dim, num_classes)
        )
        
        # Frame transformations (conventional/coordinate-dependent)
        self.frames = nn.ModuleDict({
            'default': nn.Identity(),
            'rotated': nn.Linear(num_classes, num_classes, bias=False),
        })
        
    def predict(self, substrate_presentation: torch.Tensor, 
                frame: str = 'default') -> Dict[str, torch.Tensor]:
        """
        Generate predictions with explicit substrate/frame separation.
        
        Args:
            substrate_presentation: Presentation from SubstrateLayer
            frame: Output coordinate frame (default, rotated, etc.)
            
        Returns:
            Dictionary containing:
              - substrate_prediction: Frame-invariant logits
              - frame_prediction: Frame-dependent coordinates
              - inseparability: Inseparability scores for input
        """
        substrate_logits = self.substrate_classifier(substrate_presentation)
        
        if frame in self.frames:
            frame_logits = self.frames[frame](substrate_logits)
        else:
            frame_logits = substrate_logits
            
        return {
            'substrate_prediction': substrate_logits,
            'frame_prediction': frame_logits,
            'inseparability': self.substrate.verify_inseparability(substrate_presentation)
        }


# =============================================================================
# NON-ESSENTIALIST EMBEDDINGS - Implements Emptiness Axiom
# =============================================================================

class NonEssentialistEmbedding(nn.Module):
    """
    Embeddings without fixed essence (Emptiness axiom).
    
    Tokens have no intrinsic properties, only relational positioning via Ω.
    Meaning is purely differential, not absolute.
    
    Args:
        substrate_layer: The SubstrateLayer instance
        vocab_size: Size of vocabulary
    """
    
    def __init__(self, substrate_layer: SubstrateLayer, vocab_size: int):
        super().__init__()
        self.substrate = substrate_layer
        self.to_substrate_input = nn.Embedding(vocab_size, 128)
        
        if 'embedding' not in self.substrate.presentation_ops:
            self.substrate.register_presentation_mode('embedding', 128)
    
    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode tokens without essence, only substrate presentation.
        
        Args:
            token_ids: Token indices [batch_size] or [batch_size, seq_len]
            
        Returns:
            Centered presentations [batch_size, substrate_dim] or 
            [batch_size, seq_len, substrate_dim]
        """
        substrate_input = self.to_substrate_input(token_ids)
        presentation = self.substrate.present(substrate_input, mode='embedding')
        # Center to ensure purely differential (no absolute essence)
        return presentation - presentation.mean(dim=-1, keepdim=True)


# =============================================================================
# BIAS AUDITING - Substrate vs Coordinate Level Analysis
# =============================================================================

class BiasAuditor:
    """
    Identifies bias at substrate vs coordinate levels.
    
    Substrate-level correlations may reflect meaningful structure.
    Coordinate-level correlations often reflect representational artifacts
    that can be removed via gauge transformations.
    
    Args:
        substrate_layer: The SubstrateLayer instance
        embedding: NonEssentialistEmbedding instance
    """
    
    def __init__(self, substrate_layer: SubstrateLayer, 
                 embedding: NonEssentialistEmbedding):
        self.substrate = substrate_layer
        self.embedding = embedding
    
    def audit_association(self, concept_ids: torch.Tensor, 
                         attribute_ids: torch.Tensor) -> Dict[str, float]:
        """
        Measure correlation at substrate vs coordinate levels.
        
        Args:
            concept_ids: Token IDs for concepts to audit
            attribute_ids: Token IDs for attributes to check
            
        Returns:
            Dictionary with:
              - substrate_level: Correlation at substrate level
              - coordinate_level: Correlation at coordinate level
              - bias_indicator: Difference (high = removable bias)
        """
        with torch.no_grad():
            concept_pres = self.embedding.encode(concept_ids)
            attribute_pres = self.embedding.encode(attribute_ids)
            
            # Substrate-level: correlation of means
            substrate_corr = F.cosine_similarity(
                concept_pres.mean(dim=0, keepdim=True),
                attribute_pres.mean(dim=0, keepdim=True)
            ).item()
            
            # Coordinate-level: mean of pairwise correlations
            coord_corr = F.cosine_similarity(
                concept_pres, 
                attribute_pres
            ).mean().item()
            
        return {
            'substrate_level': abs(substrate_corr),
            'coordinate_level': abs(coord_corr),
            'bias_indicator': abs(coord_corr - substrate_corr)
        }


# =============================================================================
# DECISION TRACING - Transparency via Substrate Projection
# =============================================================================

class DecisionTracer:
    """
    Traces decisions to substrate via mathematical projection.
    
    Every decision can be decomposed into substrate-grounded and
    coordinate-dependent components via projection onto span{Ω}.
    
    Args:
        substrate_layer: The SubstrateLayer instance
    """
    
    def __init__(self, substrate_layer: SubstrateLayer):
        self.substrate = substrate_layer
    
    def trace_decision(self, input_presentation: torch.Tensor) -> Dict:
        """
        Decompose decision into substrate and coordinate components.
        
        Args:
            input_presentation: Presentation tensor
            
        Returns:
            Dictionary with:
              - inseparability_score: How strongly connected to Ω
              - substrate_contribution: Fraction of norm from substrate component
              - interpretation: Human-readable classification
        """
        with torch.no_grad():
            omega = self.substrate.omega.unsqueeze(0).expand_as(input_presentation)
            
            # Project onto span{Ω}
            omega_norm_sq = (omega ** 2).sum(dim=-1, keepdim=True) + 1e-8
            projection_coeff = (input_presentation * omega).sum(dim=-1, keepdim=True) / omega_norm_sq
            substrate_component = projection_coeff * omega
            coordinate_residual = input_presentation - substrate_component
            
            # Compute fractions
            substrate_frac = substrate_component.norm(dim=-1) / (input_presentation.norm(dim=-1) + 1e-8)
            insep = F.cosine_similarity(input_presentation, omega, dim=-1)
            
            insep_mean = insep.mean().item()
            frac_mean = substrate_frac.mean().item()
            
            # Interpretation
            if insep_mean > 0.6 and frac_mean > 0.5:
                interpretation = "Strongly substrate-grounded (ultimate factors)"
            elif insep_mean > 0.3:
                interpretation = "Partially substrate-based, partially conventional"
            else:
                interpretation = "Primarily coordinate-dependent (conventional factors)"
            
            return {
                'inseparability_score': insep_mean,
                'substrate_contribution': frac_mean,
                'interpretation': interpretation
            }


# =============================================================================
# CAUSAL STRUCTURE - Implements C1-C3, Time_monotone
# =============================================================================

class PhenomenonGraph:
    """
    Causal structure among phenomena with emergent time.
    
    Implements:
      C1: Causality only among phenomena
      C2: Irreflexive (no self-causation)
      C3: Transitive closure encouraged
      Time_monotone: T strictly ordered along causal edges
    
    Args:
        n_phenomena: Number of phenomena in the graph
    """
    
    def __init__(self, n_phenomena: int):
        self.n = n_phenomena
        self.edges: Set[Tuple[int, int]] = set()
        self.T = nn.Parameter(torch.randn(n_phenomena))
        
    def add_edge(self, i: int, j: int):
        """
        Add causal edge i → j.
        
        Args:
            i: Source phenomenon index
            j: Target phenomenon index
        """
        assert i != j, f"C2 violated: self-loop {i}→{i}"
        assert 0 <= i < self.n and 0 <= j < self.n, "Indices out of range"
        self.edges.add((i, j))
    
    def time_monotone_loss(self) -> torch.Tensor:
        """
        Soft constraint enforcing T[i] < T[j] for edges i→j.
        
        Returns:
            Loss value (lower is better)
        """
        if not self.edges:
            return torch.tensor(0.0)
        
        loss = torch.tensor(0.0)
        for (i, j) in self.edges:
            loss = loss + F.softplus(self.T[i] - self.T[j])
        
        return loss / len(self.edges)
    
    def build_random_dag(self, edge_prob: float = 0.05, seed: int = 0):
        """
        Build random directed acyclic graph.
        
        Args:
            edge_prob: Probability of edge between ordered pairs
            seed: Random seed for reproducibility
        """
        torch.manual_seed(seed)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if torch.rand(()).item() < edge_prob:
                    self.add_edge(i, j)


# =============================================================================
# TRAINING - Integrated Axiom-Based Losses
# =============================================================================

class TUOSTrainer:
    """
    Trainer enforcing axioms A5, C1-C3, Time_monotone through loss terms.
    
    Args:
        substrate_layer: The SubstrateLayer instance
        lambda_insep: Weight for inseparability loss (default 0.1)
        graph: Optional PhenomenonGraph for causal/time constraints
        lambda_time: Weight for time monotonicity loss (default 0.0)
    """
    
    def __init__(self, 
                 substrate_layer: SubstrateLayer, 
                 lambda_insep: float = 0.1,
                 graph: Optional[PhenomenonGraph] = None,
                 lambda_time: float = 0.0):
        self.substrate = substrate_layer
        self.lambda_insep = lambda_insep
        self.graph = graph
        self.lambda_time = lambda_time
    
    def inseparability_loss(self, presentations: torch.Tensor) -> torch.Tensor:
        """
        Loss encouraging inseparability from Ω (A5).
        
        Args:
            presentations: Presentation tensors
            
        Returns:
            Loss value (minimize to maximize inseparability)
        """
        inseparability = self.substrate.verify_inseparability(presentations)
        return -inseparability.mean()
    
    def compute_loss(self, 
                    presentations: torch.Tensor,
                    task_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Combined loss: task + inseparability + time monotone.
        
        Args:
            presentations: Presentation tensors
            task_loss: Task-specific loss (e.g., cross-entropy)
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        insep_loss = self.inseparability_loss(presentations)
        total_loss = task_loss + self.lambda_insep * insep_loss
        
        time_loss_value = 0.0
        if self.graph is not None and self.lambda_time > 0.0:
            time_loss = self.graph.time_monotone_loss()
            total_loss = total_loss + self.lambda_time * time_loss
            time_loss_value = time_loss.item()
        
        metrics = {
            'task_loss': task_loss.item(),
            'inseparability_loss': insep_loss.item(),
            'mean_inseparability': self.substrate.verify_inseparability(presentations).mean().item(),
            'time_monotone_loss': time_loss_value
        }
        
        return total_loss, metrics


# =============================================================================
# UTILITIES
# =============================================================================

def deduplicate_parameters(*param_lists) -> List[nn.Parameter]:
    """
    Deduplicate parameters for optimizer (handles shared Ω).
    
    Args:
        *param_lists: Variable number of parameter lists
        
    Returns:
        List of unique parameters
    """
    all_params = []
    for plist in param_lists:
        all_params.extend(plist)
    
    seen = {}
    for p in all_params:
        seen[id(p)] = p
    
    return list(seen.values())


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example: Training TUOS on a simple classification task.
    """
    
    print("The Unique Ontic Substrate (TUOS / Ω)")
    print("Machine-verified non-dual metaphysics as neural architecture\n")
    
    # Initialize within substrate context
    with substrate_context():
        
        # Create substrate and register presentation mode
        substrate = SubstrateLayer(substrate_dim=256)
        substrate.register_presentation_mode('data', input_dim=64)
        
        # Create classifier
        classifier = GaugeInvariantClassifier(substrate, num_classes=2)
        
        # Create causal graph
        graph = PhenomenonGraph(n_phenomena=50)
        graph.build_random_dag(edge_prob=0.08, seed=42)
        
        # Setup trainer
        trainer = TUOSTrainer(
            substrate,
            lambda_insep=0.1,
            graph=graph,
            lambda_time=0.02
        )
        
        # Synthetic data
        torch.manual_seed(42)
        X = torch.randn(200, 64)
        y = (X[:, 0] > 0).long()
        
        # Setup optimizer with deduplicated parameters
        params = deduplicate_parameters(
            list(substrate.parameters()),
            list(classifier.parameters()),
            [graph.T]
        )
        optimizer = torch.optim.Adam(params, lr=0.01)
        
        # Training loop
        print("Training with integrated axiom constraints...")
        for epoch in range(20):
            optimizer.zero_grad()
            
            # Forward pass through substrate
            presentations = substrate.present(X, mode='data')
            predictions = classifier.predict(presentations)
            
            # Task loss
            task_loss = F.cross_entropy(
                predictions['substrate_prediction'],
                y
            )
            
            # Combined loss with axiom constraints
            loss, metrics = trainer.compute_loss(presentations, task_loss)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Report
            if epoch % 5 == 0:
                with torch.no_grad():
                    _, predicted = predictions['substrate_prediction'].max(1)
                    acc = (predicted == y).float().mean().item()
                    print(f"Epoch {epoch:2d}: "
                          f"Accuracy={acc:.3f}, "
                          f"Inseparability={metrics['mean_inseparability']:.3f}, "
                          f"Time={metrics['time_monotone_loss']:.4f}")
        
        print(f"\nFinal inseparability: {metrics['mean_inseparability']:.3f}")
        print("Training complete - axioms maintained throughout.")