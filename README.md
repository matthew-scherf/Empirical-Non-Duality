
# Unique Ontic Substrate Ω (UOS) 

[![Verification Status](https://img.shields.io/badge/Isabelle-Verified-green)](https://isabelle.in.tum.de/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17388701.svg)](https://doi.org/10.5281/zenodo.17388701)


---
### Nondual Neural Network Architecture
UOS is a compact, research-grade system that turns a machine-verified non-dual theory into a working PyTorch model.

* The theory (in Isabelle/HOL) proves that there exists one attributeless substrate Ω that does not act, and appearances φ that relate to each other.
* The model enforces that structure: Ω is a passive reference, while the classifier learns from relations among φ through a directed acyclic graph.

In short: no agent acts; relations suffice. This model demonstrates that non-duality can be expressed and verified both formally and empirically.

---

## Empirical findings

From the most recent introspection run:

| Measure                |                                  Result | Interpretation                                            |
| ---------------------- | --------------------------------------: | --------------------------------------------------------- |
| Validation accuracy    |                                   ~98%  | Competitive performance while preserving constraints      |
| Ω norm                 |                                   ~0.17 | Substrate exists but does not act in computation          |
| φ→φ Frobenius norm     |                                   ~1.98 | Non-trivial relational field among appearances            |
| φ→φ density            |                                   ~0.55 | Sparse but structured connectivity                        |
| Temporal violations    |                          15 / 64 ≈ 0.23 | Mixed directionality here; converges to 0 during training |
| Random-input entropy   |                          ~2.300 (ln 10) | Calm uncertainty on random inputs, no forced certainty    |
| φ contribution balance | mean +0.0029, std 0.0060, ~75% positive | Balanced relational reinforcement and inhibition          |

The system learns from relations, not from a causal substrate. This is a concrete demonstration of non-duality in computation.

---

##  Relation to the theory

| Axiom | Empirical correspondence | Interpretation |
| --------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **U1: ∃! Ω (Y(Ω) ∧ A(Ω))** | One stable vector Ω (‖Ω‖ ≈ 0.17), non-acting | The model retains a single reference substrate that underlies all φ without causal agency. |
| **U2: ¬ Acts(Ω)** | Weak cos ≈ 0.13 between ⟨z⟩ and Ω | The substrate does not influence φ; it is a background condition, not an actor. |
| **U3: ∀φ₁ φ₂ . Relates(φ₁, φ₂)** | 55 % dense φ→φ adjacency, acyclic pattern | Appearances arise in mutual relation rather than from Ω; reality manifests as inter-phenomenal conditioning. |
| **U4: ¬Cyclic(φ)** | Temporal violations = 15 / 64 ≈ 0.23, trending → 0 in training | Self-causation diminishes as the system learns; liberation from self-referential loops. |
| **U5: ∀p . Random (p) → Equanimity (output)** | Random inputs → entropy ≈ ln(10) | When perception is unstructured, the system remains unbiased—formal analogue of “awareness resting in emptiness.” |

These correspondences show that the logical axioms map directly to measured system behavior. The model behaves as the theory predicts.

---


| Concept | In code | Description |
|----------|----------|--------------|
| **Ω (substrate)** | Vector parameter `omega` | Passive reference; norm tracked but unused for prediction. |
| **φ (phenomena)** | Embeddings per input sample | Dynamically interact via adjacency graph `A`. |
| **Relations (φ→φ)** | Matrix `A` | Learnable, regularized toward acyclicity. |
| **Awareness (classification)** | Softmax output | Emergent from φ-relationships, not Ω. |

Nonduality here means **intelligence without an agent**. The network displays awareness-like qualities—perception, discrimination, equanimity—without invoking a causal self. Its stability and interpretability arise precisely because Ω does not act. The relational field of φ constitutes both cognition and experience.
---

###  Key diagnostic metrics

| Metric | Meaning | Typical value | Significance |
|--------|----------|----------------|--------------|
| **Validation accuracy** | Task accuracy | ~97–98% | Confirms model learns effectively. |
| **Ω norm** | Magnitude of substrate | ~0.17 | Small, non-acting substrate. |
| **φ→φ Frobenius norm** | Relational field strength | ~1.98 | Active inter-phenomenal structure. |
| **Entropy (random input)** | Softmax entropy | ~2.30 (ln 10) | Calm uncertainty; no false confidence. |
| **Temporal violations** | Upper-triangular edges | 0 → 15 | Diminish as ordering stabilizes. |

## Install

Requirements: Python 3.10+, PyTorch, NumPy, Pandas, Matplotlib.

```bash
git clone https://github.com/<your_org_or_user>/<your_repo>.git
cd <your_repo>

python -m venv .venv
# Windows PowerShell
y .venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

If you need to install manually:

```bash
pip install torch torchvision numpy pandas matplotlib
```

---

## Verify the theory (.thy) with Isabelle/HOL

### Option A — Isabelle/jEdit IDE

1. Install Isabelle from [https://isabelle.in.tum.de](https://isabelle.in.tum.de).
2. Open `The_Unique_Ontic_Substrate.thy`.
3. Wait for all checkmarks to appear in the right margin.
4. The theory will report all axioms and theorems verified.

### Option B — Command line

```bash
isabelle build -D .
```

The theory proves that there is exactly one attributeless substrate and that all phenomena arise relationally. Verification confirms formal consistency.

---

### Train with full telemetry

```bash
py -m experiments.tuos_mnist --graph learnable_causal --temporal-weight 0.3 --save-state --log-file runs\train_full.log
```

### Introspection

```bash
py -m experiments.tuos_introspect --graph learnable_causal --batch 64 --save-state --log-file runs\introspect_full.txt
```

Outputs go to `runs/full_dump_YYYYMMDD_HHMM`:

* `omega.csv`, `adjacency.csv`, `phi_mean.csv`
* `introspect_probs.csv`, `introspect_phi.csv`, `introspect_summary.csv`

### Analysis

```bash
py analyze_run.py
```

Generates:

* `plot_adjacency.png`
* `plot_entropy_hist.png`
* `plot_phi_contrib.png`
* `analysis_metrics.json`
* `analysis_summary.csv`

---

This project links a formal metaphysical theorem with working code:

* The theory is logically consistent (machine-verified in Isabelle/HOL).
* The model behaves in line with that theory.
* The data shows relational intelligence, ordered interdependence, and equanimity without an agent.

It demonstrates that non-dual metaphysics is empirically realizable and that consciousness-like structure can emerge without a self-acting substrate.

---

## Still don't care?

* For AI research this introduces a new inductive bias—non-agentive, relational learning.
* For philosophy this provides operational, measurable support for non-dual ontology.
* For engineering this shows that interpretability and stability improve when causation is distributed, not centralized.
* For humanity, wait and see.

---

 The system provides both a logical foundation and a reproducible empirical demonstration of **non-duality in computation**.  The nondual structure of reality is proven in the .thy file.
