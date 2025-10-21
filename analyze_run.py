# analyze_run.py
# Plots:
#  - plot_accuracy.png (if train_log.json present)
#  - plot_adjacency.png (from adjacency.csv)
#  - plot_entropy_hist.png (from introspect_probs.csv)
#  - plot_phi_contrib.png (from introspect_phi.csv)
# Exports:
#  - analysis_metrics.json (rich dict of stats)
#  - analysis_summary.csv  (one-row CSV of highlights)

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _latest_full_dump(base="runs"):
    if not os.path.isdir(base):
        return None
    cands = [d for d in os.listdir(base) if d.startswith("full_dump_")]
    if not cands:
        return None
    cands.sort()
    return os.path.join(base, cands[-1])

def _savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def analyze(run_dir=None, show=True):
    # resolve run_dir
    if run_dir is None:
        run_dir = _latest_full_dump("runs")
    if run_dir is None or not os.path.isdir(run_dir):
        print("No run folder found. Provide a path like: py analyze_run.py runs\\full_dump_YYYYMMDD_HHMM")
        return
    print(f"Analyzing: {run_dir}\n")

    metrics = {"run_dir": run_dir}

    # ---------- 1) Accuracy curves
    tl_path = os.path.join(run_dir, "train_log.json")
    if os.path.exists(tl_path):
        with open(tl_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        train_acc = log.get("train_acc", [])
        val_acc   = log.get("val_acc", [])
        if train_acc or val_acc:
            plt.figure()
            if train_acc: plt.plot(train_acc, label="train_acc")
            if val_acc:   plt.plot(val_acc,   label="val_acc")
            plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy curves")
            out = os.path.join(run_dir, "plot_accuracy.png")
            _savefig(out)
            if show: print(f"Saved {out}")
        if val_acc:
            metrics["epochs"]        = len(val_acc)
            metrics["best_val_acc"]  = float(np.max(val_acc))
            metrics["final_val_acc"] = float(val_acc[-1])
        if train_acc:
            metrics["final_train_acc"] = float(train_acc[-1])

    # ---------- 2) Adjacency heatmap + stats
    adj_csv = os.path.join(run_dir, "adjacency.csv")
    if os.path.exists(adj_csv):
        adj_df = pd.read_csv(adj_csv, index_col=0)
        adj = adj_df.values
        plt.figure()
        plt.imshow(adj, cmap="viridis", aspect="auto")
        plt.title("Adjacency (φ→φ)")
        plt.colorbar()
        out = os.path.join(run_dir, "plot_adjacency.png")
        _savefig(out)
        if show: print(f"Saved {out}")

        fro = float(np.linalg.norm(adj))
        upper = np.triu(adj, 1); lower = np.tril(adj, -1)
        vio_cnt = int((upper > 0).sum())
        metrics.update({
            "adj_shape_rows": int(adj.shape[0]),
            "adj_shape_cols": int(adj.shape[1]),
            "adj_fro": fro,
            "adj_density": float((adj > 0).sum() / adj.size),
            "adj_upper_sum": float(upper.sum()),
            "adj_lower_sum": float(lower.sum()),
            "temporal_violations": vio_cnt
        })
        if show:
            print(f"Adjacency stats: shape={adj.shape}, fro={fro:.4f}, "
                  f"upper_sum={upper.sum():.4f}, lower_sum={lower.sum():.4f}, violations={vio_cnt}")

    # ---------- 3) Entropy histogram + stats
    probs_csv = os.path.join(run_dir, "introspect_probs.csv")
    if os.path.exists(probs_csv):
        probs_df = pd.read_csv(probs_csv)
        prob_cols = [c for c in probs_df.columns if c.startswith("class_")]
        if prob_cols:
            P = probs_df[prob_cols].clip(1e-12, 1.0).to_numpy()
            ent = -(P * np.log(P)).sum(axis=1)
            plt.figure()
            plt.hist(ent, bins=20)
            plt.axvline(np.log(P.shape[1]), linestyle="--")
            plt.title("Softmax Entropy (samples)")
            plt.xlabel("entropy"); plt.ylabel("count")
            out = os.path.join(run_dir, "plot_entropy_hist.png")
            _savefig(out)
            if show:
                print(f"Saved {out}")
                print(f"Entropy: mean={ent.mean():.4f}, std={ent.std():.4f}, ln(K)={np.log(P.shape[1]):.4f}")
            metrics.update({
                "entropy_mean": float(ent.mean()),
                "entropy_std": float(ent.std()),
                "entropy_lnK": float(np.log(P.shape[1])),
                "entropy_n": int(len(ent))
            })

    # ---------- 4) Top/bottom φ-contributions + stats
    contrib_csv = os.path.join(run_dir, "introspect_phi.csv")
    if os.path.exists(contrib_csv):
        dfc = pd.read_csv(contrib_csv)
        if {"contrib", "phenomenon"}.issubset(dfc.columns):
            vals = dfc["contrib"].to_numpy()
            idx_sorted = np.argsort(vals)
            topk = min(10, len(vals))
            top_idx = idx_sorted[-topk:]
            bot_idx = idx_sorted[:topk]
            labels = list(dfc.iloc[top_idx]["phenomenon"]) + list(dfc.iloc[bot_idx]["phenomenon"])
            y = list(dfc.iloc[top_idx]["contrib"]) + list(dfc.iloc[bot_idx]["contrib"])
            x = np.arange(len(labels))
            plt.figure()
            plt.bar(x, y)
            plt.xticks(x, labels, rotation=90)
            plt.title("Per-phenomenon contributions (top & bottom)")
            plt.tight_layout()
            out = os.path.join(run_dir, "plot_phi_contrib.png")
            _savefig(out)
            if show:
                print(f"Saved {out}")
            metrics.update({
                "contrib_mean": float(vals.mean()),
                "contrib_std": float(vals.std()),
                "contrib_pos_frac": float((vals > 0).mean()),
                "contrib_n": int(len(vals))
            })

    # ---------- 5) Omega quick stats (if present)
    omega_csv = os.path.join(run_dir, "omega.csv")
    if os.path.exists(omega_csv):
        omega = pd.read_csv(omega_csv)
        if not omega.empty:
            w = omega.values.flatten().astype(float)
            metrics.update({
                "omega_norm": float(np.linalg.norm(w)),
                "omega_dim": int(w.size)
            })

    # ---------- 6) Write summaries
    # JSON (full detail)
    json_path = os.path.join(run_dir, "analysis_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if show: print(f"Wrote {json_path}")

    # CSV (one-row highlights)
    csv_rows = {k: [v] for k, v in metrics.items()
                if isinstance(v, (int, float, str))}
    if csv_rows:
        csv_path = os.path.join(run_dir, "analysis_summary.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        if show: print(f"Wrote {csv_path}")

if __name__ == "__main__":
    # Usage:
    #   py analyze_run.py                -> analyzes latest runs/full_dump_*
    #   py analyze_run.py <run_dir>      -> analyzes the given folder
    # Tip for headless save-only: set MPLBACKEND=Agg
    run_dir = sys.argv[1] if len(sys.argv) > 1 else None
    analyze(run_dir)
