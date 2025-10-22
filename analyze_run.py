#!/usr/bin/env python3
"""
Analyze a UOS run directory and produce:
- Metrics summary (stdout + CSV/JSON)
- Plots: plot_accuracy.png, plot_adjacency.png, plot_entropy_hist.png, plot_phi_contrib.png
- REPORT.md (Markdown with findings and image links)
- REPORT.html (self-contained HTML with embedded images)

Usage:
    py analyze_run.py [RUN_DIR] [--title "My Report"] [--out REPORT_BASENAME]

If RUN_DIR is omitted, the newest runs/full_dump_* is used.
"""

import os, sys, json, base64, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------- basic utils ----------------------

def latest_full_dump(root="runs"):
    paths = sorted(glob.glob(os.path.join(root, "full_dump_*")), key=os.path.getmtime)
    return paths[-1] if paths else None

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def b64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def safe_exists(run_dir, name):
    path = os.path.join(run_dir, name)
    return path if os.path.exists(path) else None

def note_missing(run_dir, expected):
    missing = [n for n in expected if not os.path.exists(os.path.join(run_dir, n))]
    if missing:
        print("Note: missing files in", run_dir, "→", ", ".join(missing))
    return missing

# ---------------------- robust CSV loader ----------------------

def _read_numeric_matrix(path):
    """
    Robustly read a numeric matrix from CSV:
    - handles accidental index columns and headers
    - coerces non-numeric to NaN and drops empty rows/cols
    - returns a float numpy array (2D)
    """
    # try with index_col=0 first (common when saved with df.to_csv(...))
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        df = pd.read_csv(path)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    A = df.to_numpy(dtype=float)

    if A.size == 0:
        # fallback: read with no header assumptions
        df = pd.read_csv(path, header=None)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").dropna(axis=1, how="all")
        A = df.to_numpy(dtype=float)

    if A.ndim == 1:
        # if flattened, try to square it; else keep as 1xN
        n = int(np.sqrt(A.size))
        if n * n == A.size:
            A = A.reshape(n, n)
        else:
            A = A.reshape(1, -1)
    return A

# ---------------------- metric computations ----------------------

def compute_nondual_metrics(run_dir):
    m = {}
    A_path = safe_exists(run_dir, "adjacency.csv")
    Om_path = safe_exists(run_dir, "omega.csv")

    if A_path:
        A = _read_numeric_matrix(A_path)
        n = A.shape[0]
        total_nonzero = float(np.count_nonzero(A))
        density = total_nonzero / (n * n) if n > 0 else 0.0
        diag_mean = float(np.mean(np.diag(A))) if n > 0 else 0.0
        fro = float(np.linalg.norm(A))
        lower_mass = float(np.sum(np.tril(A)))
        all_mass = float(np.sum(A)) if np.sum(A) != 0 else 1.0
        acyclicity = lower_mass / all_mass

        m.update({
            "phi_phi_density": density,
            "phi_self_mean": diag_mean,
            "phi_acyclicity": acyclicity,
            "adj_fro": fro,
        })

    if Om_path:
        Om = _read_numeric_matrix(Om_path).reshape(-1)
        m["omega_norm"] = float(np.linalg.norm(Om))
        m["omega_mean"] = float(np.mean(Om))
        m["omega_std"]  = float(np.std(Om))

    # Heuristic “emptiness” & “inseparability”
    emptiness = 1.0 - min(1.0, abs(m.get("omega_mean", 0.0)))
    m["emptiness"] = float(max(0.0, emptiness))
    density = m.get("phi_phi_density", 0.0)
    self_mean = max(0.0, min(1.0, m.get("phi_self_mean", 0.0)))
    m["inseparability"] = float(density * (1.0 - self_mean))
    return m

def compute_entropy_stats(run_dir):
    p_path = safe_exists(run_dir, "introspect_probs.csv")
    if not p_path:
        return {}
    df = pd.read_csv(p_path)
    P = df.values
    P = np.clip(P, 1e-12, 1.0)
    ent = -(P * np.log(P)).sum(axis=1)
    return {
        "entropy_mean": float(ent.mean()),
        "entropy_std": float(ent.std()),
        "entropy_min": float(ent.min()),
        "entropy_max": float(ent.max())
    }

def compute_phi_contrib_stats(run_dir):
    c_path = safe_exists(run_dir, "introspect_phi.csv")
    if not c_path:
        return {}
    df = pd.read_csv(c_path)
    if "contrib" in df.columns:
        contrib = df["contrib"].values
        pos = float((contrib > 0).mean())
        return {
            "phi_contrib_mean": float(np.mean(contrib)),
            "phi_contrib_std": float(np.std(contrib)),
            "phi_contrib_pos_frac": pos
        }
    return {}

def load_train_log(run_dir):
    tl_path = safe_exists(run_dir, "train_log.json")
    if not tl_path:
        return None
    try:
        return load_json(tl_path)
    except Exception:
        return None

# ---------------------- plotting (1 figure per plot) ----------------------

def plot_accuracy(run_dir, train_log):
    if not train_log:
        return None
    train = train_log.get("train_acc", [])
    val = train_log.get("val_acc", [])
    if not train and not val:
        return None
    plt.figure()
    if train: plt.plot(train, label="train")
    if val:   plt.plot(val,   label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.title("Accuracy curves")
    plt.legend()
    out = os.path.join(run_dir, "plot_accuracy.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

def plot_adjacency(run_dir):
    A_path = safe_exists(run_dir, "adjacency.csv")
    if not A_path:
        return None
    A = _read_numeric_matrix(A_path)
    plt.figure()
    plt.imshow(A, aspect="auto")
    plt.colorbar()
    plt.title("Adjacency (φ→φ)")
    out = os.path.join(run_dir, "plot_adjacency.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

def plot_entropy_hist(run_dir):
    p_path = safe_exists(run_dir, "introspect_probs.csv")
    if not p_path:
        return None
    P = pd.read_csv(p_path).values
    P = np.clip(P, 1e-12, 1.0)
    ent = -(P * np.log(P)).sum(axis=1)
    plt.figure()
    plt.hist(ent, bins=30)
    plt.xlabel("entropy"); plt.ylabel("count")
    plt.title("Entropy histogram (softmax)")
    out = os.path.join(run_dir, "plot_entropy_hist.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

def plot_phi_contrib(run_dir):
    c_path = safe_exists(run_dir, "introspect_phi.csv")
    if not c_path:
        return None
    df = pd.read_csv(c_path)
    if "phi" not in df.columns or "contrib" not in df.columns:
        return None
    agg = df.groupby("phi")["contrib"].mean()
    plt.figure()
    plt.bar(agg.index.astype(str), agg.values)
    plt.xlabel("φ index"); plt.ylabel("mean contribution")
    plt.title("Per-φ mean contribution to predicted class")
    out = os.path.join(run_dir, "plot_phi_contrib.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

# ---------------------- report writers ----------------------

def write_markdown_report(run_dir, title, metrics, images, table_axioms=True, out_base="REPORT"):
    md_path = os.path.join(run_dir, f"{out_base}.md")
    lines = []
    lines += [f"# {title}", "",
              f"_Run directory:_ `{run_dir}`  ",
              f"_Generated:_ {datetime.now().isoformat(timespec='seconds')}", "",
              "---", ""]
    # table
    lines += ["## Summary metrics", "", "| Metric | Value |", "|---|---:|"]
    def fmt(v):
        if v is None: return ""
        if isinstance(v, float): return f"{v:.4f}"
        if isinstance(v, (list, tuple)): return " / ".join(f"{x:.4f}" for x in v)
        return str(v)

    show = [
        ("Validation accuracy (best)", metrics.get("best_val_acc")),
        ("Ω norm", metrics.get("omega_norm")),
        ("φ→φ Frobenius", metrics.get("adj_fro")),
        ("φ→φ density", metrics.get("phi_phi_density")),
        ("Acyclicity (lower-tri mass / total)", metrics.get("phi_acyclicity")),
        ("Random-input entropy (mean)", metrics.get("entropy_mean")),
        ("φ contrib mean / std", (metrics.get("phi_contrib_mean"), metrics.get("phi_contrib_std")) if ("phi_contrib_mean" in metrics) else None),
        ("Emptiness", metrics.get("emptiness")),
        ("Inseparability", metrics.get("inseparability")),
    ]
    for k, v in show:
        if v is None: continue
        lines.append(f"| {k} | {fmt(v)} |")
    lines += ["", "---", ""]

    # plots
    lines += ["## Plots", ""]
    if images.get("accuracy"):     lines += [f"![Accuracy]({os.path.basename(images['accuracy'])})", ""]
    if images.get("adjacency"):    lines += [f"![Adjacency]({os.path.basename(images['adjacency'])})", ""]
    if images.get("entropy_hist"): lines += [f"![Entropy]({os.path.basename(images['entropy_hist'])})", ""]
    if images.get("phi_contrib"):  lines += [f"![Phi contributions]({os.path.basename(images['phi_contrib'])})", ""]
    lines += ["---", ""]

    if table_axioms:
        lines += [
            "## Axioms and empirical correspondence", "",
            "| Axiom | Empirical correspondence | Interpretation |",
            "| --- | --- | --- |",
            "| **U1: ∃! Ω (Y(Ω) ∧ A(Ω))** | One stable vector Ω (‖Ω‖ small), non-acting | Ω underlies all φ without causal agency. |",
            "| **U2: ¬ Acts(Ω)** | Weak Ω coupling; emptiness ≈ 1 | Substrate is background condition, not actor. |",
            "| **U3: ∀φ₁ φ₂ . Relates(φ₁, φ₂)** | φ→φ density moderate; structured | Appearances arise in mutual relation. |",
            "| **U4: ¬Cyclic(φ)** | Acyclicity high (lower-tri mass) | Self-causation minimal as system stabilizes. |",
            "| **U5: Random(p) → Equanimity(output)** | Entropy ~ ln(10) on noise | Unbiased awareness under unstructured input. |",
            "", "---", ""
        ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path

def write_html_report(run_dir, title, metrics, images, out_base="REPORT"):
    html_path = os.path.join(run_dir, f"{out_base}.html")
    def img_tag(label, path):
        if not path: return ""
        mime = "image/png"
        b64 = b64_image(path)
        return f'<h3>{label}</h3><img alt="{label}" src="data:{mime};base64,{b64}" style="max-width: 900px;">'

    # metrics table
    rows = []
    rows.append("<table>")
    for k, v in metrics.items():
        if isinstance(v, float):
            val = f"{v:.4f}"
        elif isinstance(v, (list, tuple)):
            val = " / ".join(f"{x:.4f}" for x in v)
        else:
            val = str(v)
        rows.append(f"<tr><td>{k}</td><td style='text-align:right'>{val}</td></tr>")
    rows.append("</table>")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif; padding:20px;}}
table{{border-collapse:collapse; margin-bottom:12px;}}
td,th{{border:1px solid #ddd; padding:6px 10px;}}
</style></head>
<body>
<h1>{title}</h1>
<p><em>Run directory:</em> {run_dir}<br><em>Generated:</em> {datetime.now().isoformat(timespec='seconds')}</p>
<h2>Summary metrics</h2>
{''.join(rows)}
<h2>Plots</h2>
{img_tag('Accuracy', images.get('accuracy'))}
{img_tag('Adjacency', images.get('adjacency'))}
{img_tag('Entropy', images.get('entropy_hist'))}
{img_tag('Phi contributions', images.get('phi_contrib'))}
</body></html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path

# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", help="Path to runs/full_dump_* folder; defaults to latest")
    ap.add_argument("--title", default="UOS Data Analysis Report")
    ap.add_argument("--out", default="REPORT", help="Base filename (without extension) for the report files")
    args = ap.parse_args()

    run_dir = args.run_dir or latest_full_dump()
    if not run_dir or not os.path.isdir(run_dir):
        print("Could not locate a runs/full_dump_* directory. Provide a path explicitly.")
        sys.exit(1)

    print(f"Analyzing: {run_dir}")

    expected = ["adjacency.csv", "introspect_probs.csv", "introspect_phi.csv", "train_log.json", "omega.csv"]
    note_missing(run_dir, expected)

    # Metrics
    train_log = load_train_log(run_dir)
    nondual = compute_nondual_metrics(run_dir)
    entropy = compute_entropy_stats(run_dir)
    contrib = compute_phi_contrib_stats(run_dir)

    metrics = {}
    if train_log and "best_val_acc" in train_log:
        metrics["best_val_acc"] = train_log["best_val_acc"]
    metrics.update(nondual)
    metrics.update(entropy)
    metrics.update(contrib)

    # Persist metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(run_dir, "analysis_summary.csv"), index=False)
    with open(os.path.join(run_dir, "analysis_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    images = {}
    images["accuracy"]     = plot_accuracy(run_dir, train_log)
    images["adjacency"]    = plot_adjacency(run_dir)
    images["entropy_hist"] = plot_entropy_hist(run_dir)
    images["phi_contrib"]  = plot_phi_contrib(run_dir)

    # Reports
    md_path   = write_markdown_report(run_dir, args.title, metrics, images, table_axioms=True, out_base=args.out)
    html_path = write_html_report(run_dir, args.title, metrics, images, out_base=args.out)

    print("\n=== Outputs ===")
    print("Metrics CSV:      ", os.path.join(run_dir, "analysis_summary.csv"))
    print("Metrics JSON:     ", os.path.join(run_dir, "analysis_metrics.json"))
    if images["accuracy"]:     print("plot_accuracy.png:    ", images["accuracy"])
    if images["adjacency"]:    print("plot_adjacency.png:   ", images["adjacency"])
    if images["entropy_hist"]: print("plot_entropy_hist.png:", images["entropy_hist"])
    if images["phi_contrib"]:  print("plot_phi_contrib.png: ", images["phi_contrib"])
    print("Markdown report:  ", md_path)
    print("HTML report:      ", html_path)

if __name__ == "__main__":
    main()
