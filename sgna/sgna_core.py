
from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class SubstrateLayer(nn.Module):
    """Shared ontic substrate Ω (reference-only; does not act)."""
    def __init__(self, input_dim: int, substrate_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, substrate_dim)
        )
        self.omega = nn.Parameter(torch.zeros(substrate_dim))
        nn.init.normal_(self.omega, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)  # [B, D]
        return z  # Ω is reference-only; not applied to z

    def get_omega(self) -> torch.Tensor:
        return self.omega

class PhenomenonGraph(nn.Module):
    def __init__(self, n_phenomena: int, threshold: float = 0.0):
        super().__init__()
        self.n = n_phenomena
        self.threshold = threshold
        self.adj_logits = nn.Parameter(torch.zeros(n_phenomena, n_phenomena))
        nn.init.xavier_uniform_(self.adj_logits)

    def adjacency(self) -> torch.Tensor:
        adj = F.relu(self.adj_logits)
        adj = adj * (1 - torch.eye(self.n, device=adj.device))
        return adj

    def temporal_penalty(self) -> torch.Tensor:
        adj = self.adjacency()
        upper = torch.triu(adj, diagonal=1)
        return upper.sum()

    def forward(self, phi_list: List[torch.Tensor]) -> List[torch.Tensor]:
        adj = self.adjacency()
        lower = torch.tril(adj, diagonal=-1)
        Phi = torch.stack(phi_list, dim=1)  # [B, N, D]
        influence = torch.einsum("ij,bjd->bid", lower, Phi)
        Phi_new = Phi + influence
        return [Phi_new[:, i, :] for i in range(Phi_new.size(1))]

class GaugeInvariantClassifier(nn.Module):
    def __init__(self, substrate_dim: int, num_phenomena: int, num_classes: int):
        super().__init__()
        self.num_phenomena = num_phenomena
        self.proj = nn.Parameter(torch.randn(num_phenomena, substrate_dim, substrate_dim))
        nn.init.orthogonal_(self.proj.view(-1, substrate_dim))
        self.head = nn.Linear(substrate_dim, num_classes)

    def forward(self, z: torch.Tensor, graph: Optional[PhenomenonGraph] = None) -> Dict[str, torch.Tensor]:
        Phi = torch.einsum("bd,nsd->bns", z, self.proj)  # [B, N, D]
        phi_list = [Phi[:, i, :] for i in range(Phi.size(1))]
        if graph is not None:
            phi_list = graph(phi_list)

        phi_stack = torch.stack(phi_list, dim=1)  # [B, N, D]
        phi_mean = phi_stack.mean(dim=1)          # [B, D]
        logits = self.head(phi_mean)              # [B, C]
        return {"logits": logits, "phi": phi_stack, "phi_mean": phi_mean}

class SGNAModel(nn.Module):
    def __init__(self, input_dim: int, substrate_dim: int, num_phenomena: int, num_classes: int, graph_mode: str = "learnable_causal", temporal_threshold: float = 0.0):
        super().__init__()
        self.substrate = SubstrateLayer(input_dim, substrate_dim)
        self.graph = None
        if graph_mode == "learnable_causal":
            self.graph = PhenomenonGraph(num_phenomena, threshold=temporal_threshold)
        self.clf = GaugeInvariantClassifier(substrate_dim, num_phenomena, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.substrate(x)
        out = self.clf(z, self.graph)
        out["z"] = z
        out["omega"] = self.substrate.get_omega()
        return out
