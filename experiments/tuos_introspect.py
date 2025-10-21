
import argparse, sys, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd

from sgna.sgna_config import TUOSConfig
from sgna.sgna_core import SGNAModel
from sgna.sgna_metrics import check_temporal_violations

def flatten(t): return t.view(-1)

import io, sys  # keep this near the imports

class Tee:
    def __init__(self, filename):
        # UTF-8 log file so Ω/φ/⟨⟩ are safe on Windows
        self.file = open(filename, "a", buffering=1, encoding="utf-8", errors="replace")
        # Try to switch console to UTF-8 (Py 3.7+). If not supported, ignore.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        self.stdout = sys.stdout

    def write(self, data):
        # Write to console (fallback if console can’t encode)
        try:
            self.stdout.write(data)
        except UnicodeEncodeError:
            self.stdout.write(data.encode("utf-8", "replace").decode("utf-8"))
        # Always safe to log file (encoding="utf-8", errors="replace")
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def parse_args():
    p = argparse.ArgumentParser(
        description="Introspect SGNA/TUOS internal states (Option B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--graph", choices=["learnable_causal", "none"], default=None, help="Enable/disable phenomenon graph.")
    p.add_argument("--batch", type=int, default=16, help="1–256 (batch size for introspection).")
    p.add_argument("--random-only", action="store_true", help="Skip MNIST sample; use random inputs only.")
    p.add_argument("--device", default=None, help="cuda|cpu")
    p.add_argument("--log-file", default=None, help="If set, tee console output to this file.")
    p.add_argument("--save-state", action="store_true", help="Export CSV/PT for adjacency, omega, phi_mean, probs, and summary.")
    return p.parse_args()

def load_data(cfg: TUOSConfig, batch_size: int = 16):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(flatten)])
    test = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    num_workers = 0 if os.name == "nt" else 2
    pin = torch.cuda.is_available()
    return DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)

@torch.no_grad()
def print_thoughts(model: SGNAModel, x: torch.Tensor, y: torch.Tensor | None, title: str, run_dir: str | None = None):
    device = next(model.parameters()).device
    out = model(x.to(device))
    logits = out["logits"]
    probs  = F.softmax(logits, dim=1)
    z      = out["z"]
    phi    = out["phi"]
    omega  = out["omega"]

    print(f"\n=== {title} ===")
    print(f"Batch shape: x={tuple(x.shape)}  z={tuple(z.shape)}  phi={tuple(phi.shape)}  logits={tuple(logits.shape)}")

    z_mean = z.mean(dim=0)
    z_norm = z.norm(dim=1).mean().item()
    omega_norm = omega.norm().item()
    cos_omega = F.cosine_similarity(F.normalize(z_mean, dim=0), F.normalize(omega, dim=0), dim=0).item()
    print(f"Substrate Ω stats: ||ω||={omega_norm:.4f}  ⟨z⟩_norm~={z_norm:.4f}  cos(⟨z⟩, ω)={cos_omega:.4f} (reference-only)")

    if getattr(model, "graph", None) is not None:
        adj = model.graph.adjacency()
        violations = check_temporal_violations(adj)
        fro = torch.linalg.matrix_norm(adj).item()
        print(f"Graph: adj shape={tuple(adj.shape)}  fro={fro:.4f}  violations={violations}")
        n = min(adj.size(0), 8)
        small = adj[:n,:n].cpu()
        print("Adj[0:8,0:8] (rounded):")
        for i in range(n):
            row = " ".join(f"{small[i,j].item():.2f}" for j in range(n))
            print("  ", row)
    else:
        print("Graph: disabled (graph_mode='none')")

    i = 0
    pvals, pidx = probs[i].topk(5)
    pred = int(pidx[0].item())
    true_str = f"  true={int(y[i])}" if y is not None else ""
    print(f"Sample[0]: pred={pred} top5={(pidx.tolist(), [round(v,4) for v in pvals.tolist()])}{true_str}")

    W = model.clf.head.weight[pred]
    contribs = torch.einsum("d,nd->n", W, phi[i]) / phi.size(1)
    order = torch.argsort(contribs, descending=True)
    print("Per-phenomenon contribution to predicted class (top 10):")
    for k in range(min(10, contribs.numel())):
        j = int(order[k])
        print(f"  φ[{j:02d}] contrib={contribs[j].item():+.4f}  ||φ[{j}]||={phi[i,j].norm().item():.3f}")

    # --- Optional exports ---
    if run_dir is not None:
        # Save omega
        torch.save(omega.detach().cpu(), os.path.join(run_dir, "omega.pt"))
        pd.DataFrame([omega.detach().cpu().numpy()], columns=[f"omega_{i}" for i in range(omega.numel())]).to_csv(os.path.join(run_dir, "omega.csv"), index=False)

        # Save adjacency
        if getattr(model, "graph", None) is not None:
            adj = model.graph.adjacency().detach().cpu()
            torch.save(adj, os.path.join(run_dir, "adjacency.pt"))
            cols = [f"phi_{j}" for j in range(adj.size(1))]
            idx  = [f"phi_{i}" for i in range(adj.size(0))]
            pd.DataFrame(adj.numpy(), columns=cols, index=idx).to_csv(os.path.join(run_dir, "adjacency.csv"))

        # Save phi_mean and probs for this batch
        phi_mean = phi.mean(dim=1).detach().cpu()
        df_phi = pd.DataFrame(phi_mean.numpy(), columns=[f"dim_{i}" for i in range(phi_mean.size(1))])
        df_phi.insert(0, "sample", list(range(phi_mean.size(0))))
        df_phi.to_csv(os.path.join(run_dir, "phi_mean.csv"), index=False)
        torch.save(phi_mean, os.path.join(run_dir, "phi_mean.pt"))

        df_probs = pd.DataFrame(probs.detach().cpu().numpy(), columns=[f"class_{i}" for i in range(probs.size(1))])
        df_probs.insert(0, "sample", list(range(probs.size(0))))
        df_probs.to_csv(os.path.join(run_dir, "introspect_probs.csv"), index=False)

        # Contributions for sample[0]
        df_contrib = pd.DataFrame({
            "sample": [0]*contribs.numel(),
            "phenomenon": [f"phi_{j}" for j in range(contribs.numel())],
            "contrib": contribs.detach().cpu().numpy(),
            "norm": [phi[i,j].norm().item() for j in range(contribs.numel())],
        })
        df_contrib.to_csv(os.path.join(run_dir, "introspect_phi.csv"), index=False)

        # Summary row
        ent = -(probs[0] * probs[0].log()).sum().item()
        fro = None
        vio = None
        if getattr(model, "graph", None) is not None:
            M = model.graph.adjacency()
            fro = float(torch.linalg.matrix_norm(M).item())
            vio = int((torch.triu(M, diagonal=1) > 0).sum().item())
        pd.DataFrame([{
            "batch": x.size(0),
            "entropy": ent,
            "omega_norm": omega.norm().item(),
            "mean_z_norm": z.norm(dim=1).mean().item(),
            "cos_z_omega": float(torch.nn.functional.cosine_similarity(z.mean(dim=0).float(), omega.float(), dim=0).item()),
            "adj_fro": fro,
            "violations": vio,
            "title": title,
        }]).to_csv(os.path.join(run_dir, "introspect_summary.csv"), index=False)

    # Random 'thoughts'
    rand_x = torch.randn_like(x)
    out_r = model(rand_x.to(device))
    probs_r = F.softmax(out_r["logits"], dim=1)
    pred_r = int(probs_r[0].argmax().item())
    ent = -(probs_r[0] * probs_r[0].log()).sum().item()
    print(f"\nRandom input: pred={pred_r}  entropy={ent:.3f}  top5={(probs_r[0].topk(5).indices.tolist(), [round(v,4) for v in probs_r[0].topk(5).values.tolist()])}")

def main():
    args = parse_args()
    if args.log_file:
        sys.stdout = Tee(args.log_file)

    cfg = TUOSConfig()
    if args.graph is not None: cfg.graph_mode = args.graph
    if args.device is not None: cfg.device = args.device
    assert 1 <= args.batch <= 256, "batch 1–256 suggested"
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = SGNAModel(cfg.input_dim, cfg.substrate_dim, cfg.num_phenomena, cfg.num_classes, cfg.graph_mode, cfg.temporal_threshold).to(device)
    model.eval()

    run_dir = None
    if args.save_state:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(cfg.log_dir, f"full_dump_{ts}")
        os.makedirs(run_dir, exist_ok=True)

    if not args.random_only:
        try:
            test_loader = load_data(cfg, batch_size=args.batch)
            x, y = next(iter(test_loader))
            print_thoughts(model, x, y, title="Test-batch thoughts", run_dir=run_dir)
        except Exception as e:
            print(f"[WARN] Could not load MNIST for introspection: {e}")

    x_rand = torch.randn(args.batch, cfg.input_dim, device=device)
    print_thoughts(model, x_rand.cpu(), None, title="Random-input thoughts", run_dir=run_dir)

if __name__ == "__main__":
    main()
