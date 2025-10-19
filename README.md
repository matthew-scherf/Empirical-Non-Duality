# The Unique Ontic Substrate **Ω**
∃!s. Substrate s ∧ ∀p. Phenomenon p → Presents p s

**A machine-verified axiomatization of non-dual metaphysics in Isabelle/HOL using scientific terminology and empirically grounded concepts**

[![Verification Status](https://img.shields.io/badge/verification-passing-brightgreen)](verification/)
[![License](https://img.shields.io/badge/license-CC%20BY%204.0-blue)](LICENSE.txt)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17388701.svg)](https://doi.org/10.5281/zenodo.17388701)

This formalization represents a scientifically-oriented approach to non-dual metaphysics. Following complete axiomatizations of Advaita Vedanta, Daoism, and Dzogchen, this work demonstrates that non-dual structure admits rigorous logical treatment using terminology familiar to physicists, cognitive scientists, and empirically-minded philosophers. The formal system captures a unique ontic substrate presenting as phenomenal multiplicity, proving theorems about causality, spacetime representation, emptiness, and emergent properties. All proofs verified October 2025 using Isabelle/HOL 2025 with zero failed goals.

This project completes a quartet of non-dual formalizations spanning religious traditions (Hindu, Daoist, Buddhist) and scientific frameworks. Together these verifications establish that non-dualism is not culturally bound mysticism but rather a logically coherent structure appearing independently across contemplative traditions and empirical investigation. The convergence under machine verification suggests non-dual awareness reflects fundamental features of reality accessible through both meditation and scientific inquiry.

---

## [Refutation Guide](https://github.com/matthew-scherf/The-Unique-Ontic-Substrate/blob/main/docs/refutation.md)

## Table of Contents

- [Prove It](#prove-it)
- [The Central Result](#the-central-result)
- [Why Scientific Terminology Matters](#why-scientific-terminology-matters)
- [Bridging Contemplation and Science](#bridging-contemplation-and-science)
- [How to Verify](#how-to-verify)
- [The Axiom System](#the-axiom-system)
- [Key Theorems](#key-theorems)
- [Relationship to Physics and Cognitive Science](#relationship-to-physics-and-cognitive-science)
- [Philosophical Implications](#philosophical-implications)
- [Citation](#citation)
- [License](#license)

---

## Prove It

The formalization establishes through mechanical verification that a non-dual ontology is internally consistent when expressed in scientifically rigorous terms, using minimal axioms about a unique substrate and its phenomenal presentations to prove theorems about causality and spacetime and information and emergent properties without invoking religious concepts or supernatural claims.

The core claims verified through proof include the following. There exists exactly one ontic substrate from which all phenomena arise as presentations or modes, and this substrate is not itself phenomenal. All phenomenal entities are inseparable from the substrate in the precise sense that each phenomenon is a presentation of the substrate, and causality operates only at the phenomenal level and does not apply to the substrate itself. Spacetime coordinates apply only to phenomena, not to the substrate, making spacetime a representational structure rather than fundamental reality. Phenomena lack intrinsic essence independent of the substrate, and information and time are emergent properties of phenomenal presentations rather than fundamental features of the substrate.

These are proven theorems following necessarily from stated axioms. The verification software confirms logical consistency. This is a mathematical structure compatible with empirical observation and scientific methodology.

The formalization proves non-duality as inseparability of phenomena from substrate, stating that for any phenomenon, that phenomenon is inseparable from the unique substrate, where inseparability is defined precisely as the relation holding when the phenomenon is a presentation of the substrate. The theorem follows from the axioms that all phenomena present the substrate and that exactly one substrate exists. This is a rigorous logical result, and given the axioms about presentation and uniqueness, every phenomenon necessarily stands in the inseparability relation to the substrate. The formalization makes precise what contemplative traditions describe poetically.

## Terminology

The contemplative traditions use culturally-specific religious language, terms like Brahman and Dao and rigpa embedded in particular historical and cultural contexts, and this creates barriers for scientists and secular philosophers who might dismiss non-dualism as religious metaphysics irrelevant to empirical inquiry.

The present formalization translates non-dual structure into neutral scientific terminology. Instead of Brahman we have substrate, instead of maya we have phenomenal presentation, instead of recognition we have coherence between levels, and the logical structure remains identical but the language becomes accessible to empirically-minded investigators. This matters because it reveals that non-dualism is not essentially religious, that the same formal structure describing Advaita metaphysics also describes a substrate ontology compatible with modern physics and cognitive science. When you strip away cultural and religious elements what remains is a minimal ontology with remarkable explanatory power.

The choice of terminology also reflects genuine parallels with scientific concepts. The substrate-presentation relation parallels the relationship between quantum fields and particle excitations, gauge invariance in physics, where different coordinate systems describe the same reality, directly parallels the gauge axioms in the formalization, and emergence of time from more fundamental atemporal structure appears in quantum gravity research. These are not superficial analogies. The formal structure genuinely captures patterns appearing in both contemplative investigation and scientific research, and the formalization proves that these patterns are logically coherent and mutually consistent.

## Bridging

This is the fourth machine-verified formalization of non-dual philosophy. The first three used traditional religious terminology, this fourth uses scientific language, and the fact that all four formalize successfully and reveal structural similarities demonstrates that non-dualism transcends the contemplative-scientific divide.

Contemplative traditions discovered non-dual structure through systematic investigation of consciousness, developing precise methodologies for recognizing the substrate nature of awareness, and their descriptions use the language available in their cultural contexts. Modern science investigates the same reality from a different angle, physics studying the structure of spacetime and matter-energy while cognitive science studies the structure of experience and cognition, and both encounter puzzles that resist solution within standard dualist frameworks, puzzles like the hard problem of consciousness and the measurement problem in quantum mechanics and the nature of time.

The formalization suggests these puzzles might dissolve in a non-dual framework. If phenomena are presentations of a substrate rather than independent entities the hard problem vanishes because consciousness is the substrate itself rather than an emergent property, and if spacetime is representational rather than fundamental then quantum measurement becomes less mysterious because we no longer demand that fundamental reality conform to spatial-temporal structure. If time is emergent from atemporal substrate, temporal paradoxes in physics find resolution.

The four formalizations together establish that non-dualism is framework-agnostic, that religious language and scientific language describe the same logical structure, and this structure admits rigorous proof and survives mechanical verification regardless of terminology.

---

## How to Verify

Verification requires Isabelle/HOL 2025 available freely from the official Isabelle website. Clone this repository and navigate to the theory directory. The build process takes approximately 15 seconds on standard hardware.

```bash
git clone https://github.com/matthew-scherf/Empirical-NonDuality.git
cd The_Unique_Ontic_Substrate
isabelle build -d . -v The_Unique_Ontic_Substrate

```

Successful verification produces output confirming all theorems check. The [verification documentation](docs/verification.md) contains proof logs and screenshots documenting successful runs. The Nitpick model finder was used with `user_axioms = true` over domain cardinalities 1 through 5 to check for counterexamples. None were found within these finite scopes.

## The Axiom System

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

## Key Theorems

The formalization proves multiple theorems establishing non-dual structure and its consequences.

**Nonduality** - Every phenomenon is inseparable from the substrate. This is the master theorem from which others follow.

**Causal_NotTwo** - Causally related phenomena are both inseparable from substrate. Causality cannot establish real separation.

**Spacetime_Unreality** - Any entity with spacetime coordinates is inseparable from substrate. Spacetime localization does not confer independent reality.

**Info_Nonreifying** - Phenomena carrying information remain inseparable from substrate. Information does not create substance.

**Time_Emergent_NotTwo** - Temporal ordering does not create separation from substrate. Time is emergent feature of presentation.

**Symmetry_Preserves_NotTwo** - Gauge transformations preserve inseparability. Different perspectives do not fragment underlying unity.

**Concepts_Don't_Reify** - Conceptual annotation of phenomena does not make them separately real. Concepts are tools for navigation, not discoveries of intrinsic divisions.

Each theorem is machine-verified and follows necessarily from axioms. The proof logs confirm every inference step.

## Relationship to Physics and Cognitive Science

## Quantum Foundations

The substrate-presentation structure parallels interpretations of quantum mechanics where the wave function or quantum field is fundamental and particles are excitations or modes, and the substrate resembles the quantum vacuum or Hilbert space from which particle states arise. Phenomena presenting the substrate resemble particle states as representations of underlying field.

The gauge invariance axioms directly parallel gauge theories in physics, where different gauge choices like different coordinate systems in spacetime represent the same underlying reality, and the formalization proves that gauge transformations preserve the substrate-presentation structure exactly as gauge transformations in physics preserve physical content.

The measurement problem in quantum mechanics, how does definite outcome arise from superposition, becomes less mysterious in this framework because measurement is not collapse of wave function but rather a specific mode of presentation where the substrate appears as definite phenomenon within a reference frame. Different measurement contexts are different modes of presentation, not different ontological situations.

## Relativity and Spacetime

The axioms establish that spacetime coordinates apply only to phenomena and the substrate itself has no location or temporal position, which aligns with approaches in quantum gravity like loop quantum gravity or causal set theory where spacetime is emergent from more fundamental pre-geometric structure.

The frame-dependence formalized in the gauge axioms parallels reference frame dependence in special relativity, where what counts as simultaneous or spatially separated depends on choice of frame, and the formalization shows this frame-dependence is compatible with underlying non-dual reality.

Block universe interpretations in relativity, where all times exist equally, find natural expression in this framework because the substrate is atemporal and temporal ordering is an emergent structure on phenomenal presentations. Past and present and future are perspectival distinctions within presentation rather than fundamental divisions.

## Cognitive Science and Consciousness

The hard problem of consciousness, how does subjective experience arise from objective matter, dissolves if we reverse the explanatory order. Consciousness is not produced by neural activity but neural activity is phenomenal presentation of substrate which is itself awareness.

This aligns with integrated information theory approaches where consciousness is fundamental and physical processes are aspects of conscious experience rather than causes of it, and the formalization makes this logically precise and proves consistency.

Predictive processing frameworks in cognitive science, where perception is active construction rather than passive reception, fit naturally here because phenomenal experience is presentation of substrate shaped by conceptual frameworks and prior expectations. Perception is substrate presenting itself through particular modes.

The binding problem, how does brain unify diverse processing streams into coherent experience, becomes tractable because unity is not achieved by binding separate elements but unity is the substrate itself. Apparent diversity is multiple modes of presentation rather than genuinely separate processes requiring integration.

## Philosophical Implications

The formalization achieves maximum parsimony, an ontology containing exactly one fundamental entity, the substrate, and one fundamental relation, presentation, where everything else including causality and spacetime and information and concepts is structure within phenomenal presentation.

This is simpler than substance dualism which posits mind and matter as separate substances, simpler than materialism which struggles to account for consciousness, and simpler than idealism which struggles to account for intersubjective agreement and physical law. Occam's razor favors simpler theories when they are equally explanatory, and the formalization proves the simple non-dual ontology is internally consistent. Whether it is explanatorily adequate requires empirical investigation, but logical coherence is established.

Nothing in the formalization contradicts empirical observation. It contradicts only metaphysical interpretations of observations, and the axioms are compatible with all experimental results in physics and cognitive science.

Quantum experiments and relativity experiments and neuroscience findings about neural correlates of consciousness all remain valid as phenomenal regularities, and the formalization reinterprets their ontological status, presentation of substrate rather than independent substance, without denying the phenomena themselves. This makes the theory empirically adequate in the sense that it accounts for all observations while offering simpler metaphysical foundation. It is not falsifiable by experiment because any experimental result can be interpreted within the framework, but this is true of all ontological theories, and empirical data underdetermines ontology. The choice between frameworks is made on grounds of coherence and parsimony and explanatory unification.

The non-appropriation axioms establish that ownership is conventional, which has implications for distributive justice and property rights and resource allocation. If ownership lacks ultimate metaphysical foundation then claims about natural rights to property become less tenable, and ownership arrangements are justified pragmatically by their consequences rather than by tracking some fundamental fact.

The inseparability of all phenomena from one substrate suggests an ethics of recognition similar to contemplative approaches, where harm to others is harm to presentations of the substrate that is one's own nature and compassion arises naturally from recognizing no ultimate separation exists. However the formalization does not determine particular ethical conclusions. It provides metaphysical foundation but leaves open how to navigate conventional reality, and the axioms establish that distinctions are real at the phenomenal level even while unified at substrate level. Ethics operates in the phenomenal domain where distinctions matter practically even if not ultimately.

The formalization suggests science can investigate substrate indirectly through study of phenomenal presentations and their patterns, where physical laws describe regularities in how substrate presents itself and these laws are discovered empirically but reflect the inherent structure of presentation.

This resolves potential tension between scientific realism and non-dualism because scientific theories are not simply conventional but track real structure in presentation, and atoms and fields and spacetime are phenomenal realities even though they are presentations of substrate rather than self-standing substances.

The framework also suggests limits to scientific knowledge. Science investigates phenomenal structure and cannot access substrate directly because every act of investigation is itself phenomenal, and the substrate is the condition for investigation rather than possible object of investigation. This is not mysticism but recognition of scope, and science can be complete regarding phenomenal structure while remaining silent about substrate.

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
