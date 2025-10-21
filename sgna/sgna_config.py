
from dataclasses import dataclass
from typing import Optional

"""TUOS/SGNA configuration (Option B: Ω is reference-only)

Suggested ranges (typical):
- epochs: 1–200 (common: 5–50)
- batch_size: 16–1024 (common: 64–256; adjust for RAM/GPU)
- substrate_dim: 16–512 (common: 64–256)
- num_phenomena: 1–64 (common: 4–16)
- num_classes: task dependent (MNIST: 10)
- task_weight: 0.5–2.0
- insep_weight: 0.0–1.0
- temporal_weight: 0.0–1.0 (use >0 only with graph)
- gauge_weight: 0.0–0.5 (read-only regulariser)
"""

@dataclass
class TUOSConfig:
    # Data / training
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"
    seed: int = 42
    early_stopping_patience: Optional[int] = 5

    # Model
    input_dim: int = 784         # MNIST flattened
    substrate_dim: int = 64
    num_phenomena: int = 8
    num_classes: int = 10

    # Loss weights
    task_weight: float = 1.0
    insep_weight: float = 0.2
    temporal_weight: float = 0.1
    gauge_weight: float = 0.0

    # Graph
    graph_mode: str = "learnable_causal"  # or "none"
    temporal_threshold: float = 0.0

    # Logging / diagnostics
    log_dir: str = "runs"
    verbose: bool = True
    print_every: int = 1

    # Export options
    save_state: bool = False
