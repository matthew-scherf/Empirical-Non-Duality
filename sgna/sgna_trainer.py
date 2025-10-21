
import time, os, json
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sgna_config import TUOSConfig
from .sgna_metrics import MetricsLogger, accuracy_from_logits, compute_inseparability, check_temporal_violations

class TUOSTrainer:
    def __init__(self, model: nn.Module, config: TUOSConfig):
        self.model = model
        self.cfg = config
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logger = MetricsLogger()

        self.task_weight = self.cfg.task_weight
        self.insep_weight = self.cfg.insep_weight
        self.temporal_weight = self.cfg.temporal_weight
        self.gauge_weight = self.cfg.gauge_weight

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Save-state
        self.run_dir = None
        if self.cfg.save_state:
            ts = time.strftime('%Y%m%d_%H%M%S')
            self.run_dir = os.path.join(self.cfg.log_dir, f'full_dump_{ts}')
            os.makedirs(self.run_dir, exist_ok=True)

    def _save_tensor_csv(self, tensor: torch.Tensor, path_csv: str, row_prefix: str):
        import pandas as pd
        x = tensor.detach().float().cpu()
        if x.dim() == 1:
            df = pd.DataFrame([x.tolist()], columns=[f"{row_prefix}{i}" for i in range(x.numel())])
        elif x.dim() == 2:
            df = pd.DataFrame(x.numpy(), columns=[f"{row_prefix}{i}" for i in range(x.size(1))])
            df.index = [f"phi_{i}" for i in range(x.size(0))]
        else:
            df = pd.DataFrame(x.view(x.size(0), -1).numpy())
        df.to_csv(path_csv)

    def _loss_components(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        out = self.model(x)
        logits = out["logits"]
        z = out["z"]

        task_loss = F.cross_entropy(logits, y)
        z_centered = z - z.mean(dim=0, keepdim=True)
        insep_surrogate = z_centered.pow(2).mean()

        temporal_loss = torch.tensor(0.0, device=self.device)
        if getattr(self.model, "graph", None) is not None:
            temporal_loss = self.model.graph.temporal_penalty()

        gauge_pen = torch.tensor(0.0, device=self.device)
        if self.gauge_weight > 0.0:
            with torch.no_grad():
                D = z.size(1)
                A = torch.randn(D, D, device=self.device)
                Q, _ = torch.linalg.qr(A)
                z_g = z @ Q
                logits_g = self.model.clf(z_g, getattr(self.model, "graph", None))["logits"]
                gauge_pen = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(logits_g, dim=1), reduction="batchmean")

        total = (
            self.task_weight * task_loss +
            self.insep_weight * insep_surrogate +
            self.temporal_weight * temporal_loss +
            self.gauge_weight * gauge_pen
        )

        with torch.no_grad():
            acc = accuracy_from_logits(logits, y)
            mean_insep = compute_inseparability(z)
            violations = 0
            if getattr(self.model, "graph", None) is not None:
                violations = check_temporal_violations(self.model.graph.adjacency())

        scalars = dict(
            task_loss=task_loss.item(),
            insep_loss=insep_surrogate.item(),
            temporal_loss=temporal_loss.item(),
            gauge_pen=float(gauge_pen.item()) if isinstance(gauge_pen, torch.Tensor) else float(gauge_pen),
            acc=acc,
            mean_insep=mean_insep,
            violations=int(violations),
        )
        return total, scalars

    def fit(self, train_loader, val_loader=None):
        best_val = -1.0
        last_best_epoch = -1

        for epoch in range(self.cfg.epochs):
            t0 = time.time()
            self.model.train()
            train_losses = []
            train_accs = []

            for batch in train_loader:
                self.opt.zero_grad(set_to_none=True)
                loss, scalars = self._loss_components(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                if self.cfg.verbose:
                    bad = False
                    for n,p in self.model.named_parameters():
                        if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
                            print(f"[WARN] Param {n} has NaN/Inf")
                            bad = True
                    if bad:
                        print("[WARN] Detected NaN/Inf in parameters this step")

                train_losses.append(loss.item())
                train_accs.append(scalars["acc"])

            val_loss = 0.0
            val_acc = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    cnt = 0
                    for batch in val_loader:
                        l, s = self._loss_components(batch)
                        val_loss += l.item()
                        val_acc += s["acc"]
                        cnt += 1
                    if cnt > 0:
                        val_loss /= cnt
                        val_acc /= cnt

            epoch_time = time.time() - t0
            self.logger.log_epoch(
                train_loss=sum(train_losses)/max(1, len(train_losses)),
                train_acc=sum(train_accs)/max(1, len(train_accs)),
                val_loss=val_loss,
                val_acc=val_acc,
                epoch_times=epoch_time,
            )

            adj_upper_sum = None
            adj_fro = None
            violations = None
            if getattr(self.model, "graph", None) is not None:
                with torch.no_grad():
                    adj = self.model.graph.adjacency()
                    upper = torch.triu(adj, diagonal=1)
                    adj_upper_sum = float(upper.sum().item())
                    adj_fro = float(torch.norm(adj).item())
                    violations = int((upper > 0).sum().item())

            if (epoch % max(1, self.cfg.print_every) == 0) and self.cfg.verbose:
                msg = [
                    f"Epoch {epoch+1}/{self.cfg.epochs}",
                    f"  train_loss={self.logger.metrics['train_loss'][-1]:.4f}",
                    f"  train_acc={self.logger.metrics['train_acc'][-1]:.4f}",
                    f"  val_loss={self.logger.metrics['val_loss'][-1]:.4f}",
                    f"  val_acc={self.logger.metrics['val_acc'][-1]:.4f}",
                    f"  epoch_time={epoch_time:.2f}s"
                ]
                if violations is not None:
                    msg += [
                        f"  temporal_upper_sum={adj_upper_sum:.4f}",
                        f"  temporal_violations={violations}",
                        f"  adj_fro={adj_fro:.4f}",
                    ]
                print("\n".join(msg))

            if self.cfg.early_stopping_patience and self.logger.metrics['val_acc']:
                current_val = self.logger.metrics['val_acc'][-1]
                if current_val > best_val:
                    best_val = current_val
                    last_best_epoch = epoch
                    if self.cfg.verbose:
                        print(f"New best val_acc: {best_val:.4f} at epoch {epoch+1}")
                    if self.run_dir:
                        # Save model weights
                        torch.save(self.model.state_dict(), os.path.join(self.run_dir, "best.pt"))
                        # Save omega
                        omega = self.model.substrate.get_omega()
                        torch.save(omega.detach().cpu(), os.path.join(self.run_dir, "omega.pt"))
                        self._save_tensor_csv(omega, os.path.join(self.run_dir, "omega.csv"), row_prefix="omega_")
                        # Save adjacency
                        if getattr(self.model, "graph", None) is not None:
                            adj = self.model.graph.adjacency().detach().cpu()
                            torch.save(adj, os.path.join(self.run_dir, "adjacency.pt"))
                            self._save_tensor_csv(adj, os.path.join(self.run_dir, "adjacency.csv"), row_prefix="phi_")
                if (epoch - last_best_epoch) > self.cfg.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if self.run_dir:
            with open(os.path.join(self.run_dir, 'train_log.json'), 'w') as f:
                json.dump(self.logger.metrics, f)
        return self.logger
