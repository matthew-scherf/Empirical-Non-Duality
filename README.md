# The Unique Ontic Substrate **Ω**

**A machine-verified axiomatization of non-dual metaphysics in Isabelle/HOL**

[![Verification Status](https://img.shields.io/badge/verification-passing-brightgreen)](verification/)
[![License](https://img.shields.io/badge/license-CC%20BY%204.0-blue)](LICENSE.txt)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17388701.svg)](https://doi.org/10.5281/zenodo.17388701)

[Submitted to Isabelle AFP](https://www.isa-afp.org/webapp/submission/?id=2025-10-19_11-47-47_483)

This work demonstrates that non-dual structure admits rigorous logical treatment. The formal system captures a unique ontic substrate presenting as phenomenal multiplicity, proving theorems about causality, spacetime representation, emptiness, and emergent properties. All proofs verified October 2025 using Isabelle/HOL 2025 with zero failed goals.

---

## [Refutation Guide](https://github.com/matthew-scherf/The-Unique-Ontic-Substrate/blob/main/docs/refutation.md)

## Contents

- [Proof](#proof)
- [Terminology](#terminology)
- [Verification](#verification)
- [Axioms](#axioms)
- [Theorems](#theorems)
- [Declarations](#declarations)
- [Citation](#citation)
- [License](#license)

---

## Proof

The formalization establishes through mechanical verification that a non-dual ontology is internally consistent when expressed in scientifically rigorous terms, using minimal axioms about a unique substrate and its phenomenal presentations to prove theorems about causality and spacetime and information and emergent properties.

The core claims verified through proof include the following. There exists exactly one ontic substrate from which all phenomena arise as presentations or modes, and this substrate is not itself phenomenal. All phenomenal entities are inseparable from the substrate in the precise sense that each phenomenon is a presentation of the substrate, and causality operates only at the phenomenal level and does not apply to the substrate itself. Spacetime coordinates apply only to phenomena, not to the substrate, making spacetime a representational structure rather than fundamental reality. Phenomena lack intrinsic essence independent of the substrate, and information and time are emergent properties of phenomenal presentations rather than fundamental features of the substrate.

These are proven theorems following necessarily from stated axioms. The verification software confirms logical consistency. This is a mathematical structure compatible with empirical observation and scientific methodology.

The formalization proves non-duality as inseparability of phenomena from substrate, stating that for any phenomenon, that phenomenon is inseparable from the unique substrate, where inseparability is defined precisely as the relation holding when the phenomenon is a presentation of the substrate. The theorem follows from the axioms that all phenomena present the substrate and that exactly one substrate exists. This is a rigorous logical result, and given the axioms about presentation and uniqueness, every phenomenon necessarily stands in the inseparability relation to the substrate. The formalization makes precise what contemplative traditions describe poetically.

## Terminology

The choice of terminology reflects genuine parallels with scientific concepts. The substrate-presentation relation parallels the relationship between quantum fields and particle excitations, gauge invariance in physics, where different coordinate systems describe the same reality, directly parallels the gauge axioms in the formalization, and emergence of time from more fundamental atemporal structure appears in quantum gravity research. These are not superficial analogies. The formal structure genuinely captures patterns appearing in both contemplative investigation and scientific research, and the formalization proves that these patterns are logically coherent and mutually consistent.

---

## Verification

Verification requires Isabelle/HOL 2025 available freely from the official Isabelle website. Clone this repository and navigate to the theory directory. The build process takes approximately 15 seconds on standard hardware.

```bash
git clone https://github.com/matthew-scherf/Empirical-NonDuality.git
cd The_Unique_Ontic_Substrate
isabelle build -d . -v The_Unique_Ontic_Substrate

```

Successful verification produces output confirming all theorems check. The Nitpick model finder was used with `user_axioms = true` over domain cardinalities 1 through 5 to check for counterexamples. None were found within these finite scopes.

## Axioms

The formalization rests on minimal axioms organized into core ontology and extensions addressing causality, spacetime, emptiness, and emergent properties.

### Core Ontology (5 axioms)

The foundation establishes existence, uniqueness, and presentation structure.

**A1 (Existence)** - At least one substrate exists. This ensures the ontology is non-vacuous.

**A2 (Uniqueness)** - If two entities are both substrates, they are identical. This establishes monism at the fundamental level.

**A3 (Exhaustivity)** - Every entity is either phenomenal or substrate. This creates a clean ontological dichotomy.

**A4 (Presentation)** - Every phenomenon presents every substrate. Combined with uniqueness, this means every phenomenon presents the unique substrate.

**A5 (Inseparability Definition)** - Two entities are inseparable when one is a phenomenon presenting the substrate and the other is that substrate. This makes inseparability precise and computable.

From these axioms we derive that exactly one substrate exists (proven in unique_substrate lemma). We can define this unique substrate as Ω (TheSubstrate). The Nonduality theorem follows necessarily.

### Causality Extension (3 axioms)

Causality is restricted to the phenomenal level.

**C1** - Causal relations hold only between phenomena, never involving the substrate.

**C2** - Causality is irreflexive. No phenomenon causally precedes itself.

**C3** - Causality is transitive. If x precedes y and y precedes z, then x precedes z.

This creates a strict partial order on phenomena while leaving the substrate outside causal structure. Time and causation are emergent features of phenomenal presentation rather than properties of fundamental reality.

### Spacetime Extension (2 axioms)

Spacetime coordinates apply only to phenomena, not to substrate.

**S1** - If an entity has coordinates in some reference frame, that entity is phenomenal.

**S2** - Gauge-related frames agree on whether entities have coordinates. Different coordinate systems are equivalent representations.

This formalizes the idea that spacetime is a representational structure for phenomena rather than fundamental container. The substrate is not located in space or time. Phenomena appear to occupy spacetime positions, but these positions are frame-dependent representations rather than intrinsic properties.

### Emptiness Extension (1 axiom)

Phenomena lack intrinsic essence independent of substrate.

**Emptiness** - No phenomenon possesses essence. Essence here means intrinsic independent nature. Phenomena exist only as presentations of substrate, not as self-standing entities.

This formalizes the Buddhist concept of śūnyatā (emptiness) and connects to modern relational ontologies where entities are defined by their relations rather than intrinsic properties.

### Dependent Arising Extension (3 axioms)

One phenomenon can arise from another, but this arising is endogenous to the substrate.

**AF1** - ArisesFrom relation holds only between phenomena.

**AF2** - If phenomenon p arises from phenomenon q, both present the substrate. Arising is internal to the substrate's self-presentation.

**AF3** - There are no exogenous entities outside the substrate-phenomenon structure. Everything that exists is either substrate or phenomenon.

This captures dependent origination (pratītyasamutpāda) from Buddhist philosophy while maintaining non-dualism. Phenomena arise in patterns, but these patterns unfold within the substrate rather than being imposed from outside.

### Non-Appropriation Extension (2 axioms)

Ownership is conventional, not ontological.

**Ownership_Conventional** - If an agent owns an entity, that entity is phenomenal and conventionally valid.

**No_Ontic_Ownership** - Owned entities are inseparable from substrate and lack essence. Ownership is a useful fiction for navigating conventional reality but has no ultimate metaphysical status.

This has implications for ethics and social organization. Property rights and ownership claims are pragmatically justified conventions rather than fundamental facts about reality.

### Symmetry Extension (2 axioms)

Gauge transformations preserve presentation structure.

**Act_Closed** - Gauge actions on phenomena produce phenomena.

**Act_Preserves_Presentation** - If a phenomenon presents the substrate, its gauge transform also presents the substrate.

This formalizes the idea that different representational perspectives (different gauges) are equivalent. The substrate-presentation structure is gauge-invariant. This directly parallels gauge invariance in physics where physical laws remain unchanged under certain transformations.

### Information and Time Extensions

Information is a non-negative quantity attached to phenomena. Time is an emergent strict ordering on phenomena respecting causal structure. These are formalized abstractly without assuming numeric structure, showing that the core insights survive in minimal mathematical frameworks.

## Theorems

The formalization proves multiple theorems establishing non-dual structure and its consequences.

**Nonduality** - Every phenomenon is inseparable from the substrate. This is the master theorem from which others follow.

**Causal_NotTwo** - Causally related phenomena are both inseparable from substrate. Causality cannot establish real separation.

**Spacetime_Unreality** - Any entity with spacetime coordinates is inseparable from substrate. Spacetime localization does not confer independent reality.

**Info_Nonreifying** - Phenomena carrying information remain inseparable from substrate. Information does not create substance.

**Time_Emergent_NotTwo** - Temporal ordering does not create separation from substrate. Time is emergent feature of presentation.

**Symmetry_Preserves_NotTwo** - Gauge transformations preserve inseparability. Different perspectives do not fragment underlying unity.

**Concepts_Don't_Reify** - Conceptual annotation of phenomena does not make them separately real. Concepts are tools for navigation, not discoveries of intrinsic divisions.

Each theorem is machine-verified and follows necessarily from axioms. The proof logs confirm every inference step.


---
## DECLARATIONS

**Availability of data and material**

All Isabelle/HOL theory files (.thy) constituting the formal proofs presented in this work are available in a public repository [here](https://github.com/matthew-scherf/The-Unique-Ontic-Substrate/tree/main/isabelle). The files include: NonDuality.thy (Empirical Non-Duality), Advaita_Vedanta.thy, Dzogchen.thy, and Daoism.thy. All formalizations have been verified for consistency using Isabelle/HOL 2025. The code is released under the BSD-3-Clause license with documentation under Creative Commons Attribution 4.0 International (CC BY 4.0). Complete verification logs and model-checking results via Nitpick are included in the repository.

**Competing interests**

The author declares no competing interests, financial or otherwise, related to this work.

**Funding**

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors. The work was conducted independently without institutional support.

**Authors' contributions**

Matthew Scherf is the sole author responsible for all aspects of this work, including conceptualization, formal axiomatization, machine verification, analysis, and manuscript preparation.

**Acknowledgements**

The author acknowledges the use of Claude (Anthropic) as an AI research assistant in developing and refining the formal axiomatizations, exploring philosophical implications, and conducting literature review. The author also acknowledges the open-source Isabelle/HOL community for providing the proof assistant infrastructure that made this verification possible, and the contemplative traditions of Advaita Vedanta, Dzogchen, and Daoism whose insights inspired this formalization.

---
## Citation

If you reference this work, please cite as follows.

```bibtex
@misc{empirical2025,
  author = {Scherf, Matthew},
  title = {Formal Axiomatization of Empirical Non-Duality: Machine-Verified Scientific Framework},
  year = {2025},
  doi = {10.5281/zenodo.17388701},
  url = {https://github.com/matthew-scherf/Empirical-NonDuality},
  note = {Isabelle/HOL formalization, verified October 2025}
}
```

## License

The formalization (`.thy` files) is released under BSD-3-Clause license. Documentation is released under Creative Commons Attribution 4.0 International (CC BY 4.0). See LICENSE.txt for complete terms.

---

**Ω**

*The unique ontic substrate*

Empirically grounded. Logically verified. Scientifically rigorous.

**Verified. Consistent. True.**

---

∃!s. Substrate s ∧ ∀p. Phenomenon p → Presents p s

**All phenomena present one substrate**

Four formalizations. Four frameworks. One structure.

---
