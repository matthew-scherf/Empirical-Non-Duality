
import argparse, sys, os, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sgna.sgna_config import TUOSConfig
from sgna.sgna_core import SGNAModel
from sgna.sgna_trainer import TUOSTrainer

def flatten(t):  # picklable
    return t.view(-1)

class Tee:
    def __init__(self, filename):
        self.file = open(filename, "a", buffering=1)
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()

def parse_args():
    p = argparse.ArgumentParser(
        description="Train SGNA/TUOS on MNIST (Option B compliant).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--graph", choices=["learnable_causal", "none"], default=None, help="Enable/disable causal graph.")
    p.add_argument("--epochs", type=int, default=None, help="1–200 (typical 5–50).")
    p.add_argument("--batch-size", type=int, default=None, help="16–1024 (typical 64–256).")
    p.add_argument("--substrate-dim", type=int, default=None, help="16–512 (typical 64–256).")
    p.add_argument("--num-phenomena", type=int, default=None, help="1–64 (typical 4–16).")
    p.add_argument("--gauge-weight", type=float, default=None, help="0.0–0.5; read-only invariance penalty.")
    p.add_argument("--temporal-weight", type=float, default=None, help="0.0–1.0; only meaningful if graph enabled.")
    p.add_argument("--device", default=None, help="cuda|cpu")
    p.add_argument("--verbose", action="store_true", help="Print extra diagnostics each epoch.")
    p.add_argument("--print-every", type=int, default=None, help="Print every N epochs.")
    p.add_argument("--log-file", default=None, help="If set, tee console output to this file.")
    p.add_argument("--save-state", action="store_true", help="Export best.pt, omega/adjacency CSV/PT, train_log.json")
    return p.parse_args()

def make_loaders(cfg: TUOSConfig):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(flatten)])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    # Windows-safe defaults
    num_workers = 0 if os.name == "nt" else 2
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test,  batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader

def main():
    args = parse_args()
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        sys.stdout = Tee(args.log_file)

    cfg = TUOSConfig()
    if args.graph is not None: cfg.graph_mode = args.graph
    if args.device is not None: cfg.device = args.device
    if args.epochs is not None:
        assert 1 <= args.epochs <= 200, "epochs out of suggested range (1–200)"
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        assert 16 <= args.batch_size <= 4096, "batch_size 16–4096 suggested"
        cfg.batch_size = args.batch_size
    if args.substrate_dim is not None:
        assert 8 <= args.substrate_dim <= 1024, "substrate_dim 8–1024 suggested"
        cfg.substrate_dim = args.substrate_dim
    if args.num_phenomena is not None:
        assert 1 <= args.num_phenomena <= 128, "num_phenomena 1–128 suggested"
        cfg.num_phenomena = args.num_phenomena
    if args.gauge_weight is not None:
        assert 0.0 <= args.gauge_weight <= 1.0, "gauge_weight 0.0–1.0"
        cfg.gauge_weight = args.gauge_weight
    if args.temporal_weight is not None:
        assert 0.0 <= args.temporal_weight <= 5.0, "temporal_weight 0.0–5.0"
        cfg.temporal_weight = args.temporal_weight
    if args.verbose: cfg.verbose = True
    if args.print_every is not None and args.print_every > 0: cfg.print_every = args.print_every
    if args.save_state: cfg.save_state = True

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = SGNAModel(cfg.input_dim, cfg.substrate_dim, cfg.num_phenomena, cfg.num_classes, cfg.graph_mode, cfg.temporal_threshold).to(device)

    trainer = TUOSTrainer(model, cfg)
    train_loader, test_loader = make_loaders(cfg)
    logger = trainer.fit(train_loader, test_loader)
    print("Best val acc:", max(logger.metrics["val_acc"]) if logger.metrics["val_acc"] else None)

if __name__ == "__main__":
    main()
