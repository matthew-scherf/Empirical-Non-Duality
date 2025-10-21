# UOS Software Manual — The Unique Ontic Substrate Framework

This document serves as a **complete user manual** for the UOS (Unique Ontic Substrate) system. It explains the purpose, architecture, configuration, usage, and output interpretation of the framework. If the README is the quickstart, this is the full reference.

---

## 1. Overview

### 1.1 Purpose

The **Unique Ontic Substrate (UOS)** project unites a formally verified metaphysical theory with an empirical, trainable AI model. The goal is to demonstrate that a non-dual ontology in which awareness (Ω) is not an acting agent but the silent ground of relational phenomena (φ) can be implemented, trained, and empirically verified.

### 1.2 Core claim

> Reality can be modeled as relational, not causal, as appearances condition each other within an attributeless substrate. The UOS framework expresses this structure formally (in Isabelle/HOL) and computationally (in PyTorch).

### 1.3 Components

| Component                        | Language         | Role                                                                            |
| -------------------------------- | ---------------- | ------------------------------------------------------------------------------- |
| `The_Unique_Ontic_Substrate.thy` | Isabelle/HOL     | Defines the logical system proving the existence of one non-acting substrate Ω. |
| `sgna_core.py`                   | Python (PyTorch) | Core model defining Ω and the φ→φ relational graph.                             |
| `sgna_trainer.py`                | Python           | Training loop, optimization, logging, and checkpointing.                        |
| `sgna_metrics.py`                | Python           | Evaluation metrics, accuracy, entropy, and causal violation checks.             |
| `experiments/tuos_mnist.py`      | Python           | Main script for training with telemetry on MNIST.                               |
| `experiments/tuos_introspect.py` | Python           | Runs post-training introspection and relational diagnostics.                    |
| `analyze_run.py`                 | Python           | Aggregates and visualizes results into plots and summaries.                     |

---

## 2. Installation

### 2.1 Requirements

* Python ≥ 3.10
* PyTorch ≥ 2.0 (CPU or CUDA build)
* NumPy, Pandas, Matplotlib

### 2.2 Setup

```bash
git clone https://github.com/<your_org_or_user>/<your_repo>.git
cd <your_repo>

python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

You may also install dependencies manually:

```bash
pip install torch torchvision numpy pandas matplotlib
```

### 2.3 Optional tools

* **Isabelle 2025+** (for formal proof verification)
* **Graphviz** (for visualizing φ→φ relations)

---

## 3. Theory verification

The `.thy` file encodes the formal non-dual ontology. Verification confirms that all theorems are logically consistent within Isabelle/HOL.

### 3.1 Verify using jEdit IDE

1. Install Isabelle from [isabelle.in.tum.de](https://isabelle.in.tum.de).
2. Open `The_Unique_Ontic_Substrate.thy` in the jEdit interface.
3. Allow it to process. When green checkmarks appear, the proofs are complete.

### 3.2 Verify from terminal

```bash
isabelle build -D .
```

Expected result: session builds successfully, producing a `The_Unique_Ontic_Substrate.pdf` document if configured.

---

## 4. Architecture

### 4.1 Ontological mapping

| Concept                        | In code                     | Description                                                |
| ------------------------------ | --------------------------- | ---------------------------------------------------------- |
| **Ω (substrate)**              | Vector parameter `omega`    | Passive reference; norm tracked but unused for prediction. |
| **φ (phenomena)**              | Embeddings per input sample | Dynamically interact via adjacency graph `A`.              |
| **Relations (φ→φ)**            | Matrix `A`                  | Learnable, regularized toward acyclicity.                  |
| **Awareness (classification)** | Softmax output              | Emergent from φ-relationships, not Ω.                      |

### 4.2 Mathematical outline

For each sample:

```
z = encoder(x)
phi = graph_layer(z, A)
logits = classifier(phi.mean(dim=1))
```

Regularizers:

* Temporal violation loss penalizes causal cycles.
* Gauge loss stabilizes Ω’s non-acting property.

---

## 5. Running experiments

### 5.1 Training

```bash
py -m experiments.tuos_mnist --graph learnable_causal --temporal-weight 0.3 --save-state --log-file runs\train_full.log
```

Optional flags:

* `--substrate-dim` : dimensionality of Ω (default 128)
* `--num-phenomena` : number of φ nodes (default 12)
* `--epochs` : training epochs (default 20)

Suggested Settings:

```
py -m experiments.tuos_mnist `
  --graph learnable_causal `
  --epochs 20 `
  --batch-size 64 `
  --substrate-dim 128 `
  --num-phenomena 12 `
  --temporal-weight 0.30 `
  --gauge-weight 0.02 `
  --verbose `
  --print-every 50 `
  --save-state `
  --log-file runs\train_full.log
```

Outputs:

* Model checkpoints (`best.pt`)
* Logs (`train_log.json`)
* Adjacency and Ω states (`adjacency.csv`, `omega.csv`)

### 5.2 Introspection

After training, run:

```bash
py -m experiments.tuos_introspect --graph learnable_causal --batch 64 --save-state --log-file runs\introspect_full.txt
```

Produces:

* `introspect_phi.csv`, `introspect_probs.csv`, `introspect_summary.csv`
* Human-readable console output detailing Ω–φ statistics.

### 5.3 Analysis

To visualize and summarize results:

```bash
py analyze_run.py
```

Generates:

* `plot_adjacency.png`
* `plot_entropy_hist.png`
* `plot_phi_contrib.png`
* `analysis_summary.csv`
* `analysis_metrics.json`

---

## 6. Interpretation of results

### 6.1 Key diagnostic metrics

| Metric                     | Meaning                   | Typical value | Significance                           |
| -------------------------- | ------------------------- | ------------- | -------------------------------------- |
| **Validation accuracy**    | Task accuracy             | ~97–98%       | Confirms model learns effectively.     |
| **Ω norm**                 | Magnitude of substrate    | ~0.17         | Small, non-acting substrate.           |
| **φ→φ Frobenius norm**     | Relational field strength | ~1.98         | Active inter-phenomenal structure.     |
| **Entropy (random input)** | Softmax entropy           | ~2.30 (ln 10) | Calm uncertainty; no false confidence. |
| **Temporal violations**    | Upper-triangular edges    | 0 → 15        | Diminish as ordering stabilizes.       |

### 6.2 Visualizations

* **Adjacency heatmap:** shows relational topology among φ.
* **Entropy histogram:** reflects how balanced the model’s uncertainty is.
* **φ-contribution chart:** lists the top reinforcing and inhibiting appearances.

---

## 7. Relation to the theory

| Axiom                                         | Empirical correspondence                                       | Interpretation                                                                                                    |
| --------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **U1: ∃! Ω (Y(Ω) ∧ A(Ω))**                    | One stable vector Ω (‖Ω‖ ≈ 0.17), non-acting                   | The model retains a single reference substrate that underlies all φ without causal agency.                        |
| **U2: ¬ Acts(Ω)**                             | Weak cos ≈ 0.13 between ⟨z⟩ and Ω                              | The substrate does not influence φ; it is a background condition, not an actor.                                   |
| **U3: ∀φ₁ φ₂ . Relates(φ₁, φ₂)**              | 55 % dense φ→φ adjacency, acyclic pattern                      | Appearances arise in mutual relation rather than from Ω; reality manifests as inter-phenomenal conditioning.      |
| **U4: ¬Cyclic(φ)**                            | Temporal violations = 15 / 64 ≈ 0.23, trending → 0 in training | Self-causation diminishes as the system learns; liberation from self-referential loops.                           |
| **U5: ∀p . Random (p) → Equanimity (output)** | Random inputs → entropy ≈ ln(10)                               | When perception is unstructured, the system remains unbiased—formal analogue of “awareness resting in emptiness.” |

### 7.1 Discussion

Nonduality here means **intelligence without an agent**. The network displays awareness-like qualities—perception, discrimination, equanimity—without invoking a causal self. Its stability and interpretability arise precisely because Ω does not act. The relational field of φ constitutes both cognition and experience.

---

## 8. Extending the framework

### 8.1 Custom datasets

You can adapt the experiment scripts to any image or tabular dataset by replacing the `torchvision.datasets.MNIST` section with your data loader.

### 8.2 Alternate graph models

The `--graph` argument accepts:

* `none`: disables relational graph.
* `fixed`: uses a constant predefined adjacency.
* `learnable_causal`: learns a directed acyclic graph (default).

### 8.3 Custom losses

You can modify `sgna_trainer.py` to include other regularizers, such as contrastive terms or energy-based penalties, as long as Ω remains passive.

---

## 9. Troubleshooting

| Issue                         | Cause                                        | Solution                                                   |
| ----------------------------- | -------------------------------------------- | ---------------------------------------------------------- |
| UnicodeEncodeError on Windows | Non-UTF8 log writes                          | Add `PYTHONIOENCODING=utf-8` or use the updated Tee class. |
| CUDA not found                | Torch CPU build                              | Install GPU-enabled PyTorch.                               |
| Empty CSV outputs             | `--save-state` missing                       | Re-run with the flag enabled.                              |
| Isabelle build fails          | Missing session ROOT or wrong directory name | Ensure session directory matches ROOT entry.               |

---

## 10. Licensing and citation

* **Code:** BSD 3-Clause
* **Documentation:** Creative Commons Attribution 4.0 (CC BY 4.0)
* **Citation example:**

```
Scherf, M. (2025). The Unique Ontic Substrate: Machine-Verified Non-Dual Metaphysics in Isabelle/HOL and PyTorch. Zenodo. DOI: https://doi.org/10.5281/zenodo.17388701
```

---

## 11. Concluding summary

The UOS system demonstrates that a **non-dual ontology** can function as a **computational architecture**. The substrate (Ω) exists without acting; phenomena (φ) interrelate to form cognition. The model’s measurable behavior—order, stability, and equanimity—confirms that logical non-duality can be observed empirically.

Install, verify, run, and observe: awareness expressed as code.
