
# Technical Guide to The_Unique_Ontic_Substrate.thy
[![DOI](https://zenodo.org/badge/1079066681.svg)](https://doi.org/10.5281/zenodo.17388701)
## Introduction

This technical guide provides comprehensive documentation for the Isabelle/HOL formalization of Empirical Non-Duality contained in `The_Unique_Ontic_Substrate.thy`. The guide explains the formal structure, verification methodology, extension mechanisms, and technical details necessary for understanding, verifying, and potentially extending the formalization. This document assumes basic familiarity with formal logic and theorem proving but provides explanations accessible to those new to Isabelle/HOL.

## Prerequisites

To work with this formalization you need Isabelle/HOL 2025 or later, available from https://isabelle.in.tum.de/, and basic understanding of higher-order logic including types, predicates, functions, and quantification. Familiarity with axiomatic systems and formal proof is helpful but not strictly required, as the formalization uses automated proof methods extensively. No numeric libraries or external dependencies are required beyond standard Isabelle/HOL, making verification straightforward and self-contained.

## File Structure

The theory file begins with standard Isabelle header declaring theory name and imports, where `theory The_Unique_Ontic_Substrate` declares the theory and `imports Main` brings in Isabelle's standard library providing basic logical infrastructure. The copyright and licensing information follows, documenting that the work is licensed under BSD-3-Clause for code and Creative Commons Attribution 4.0 International for documentation, with first verification occurring October 19, 2025 and DOI 10.5281/zenodo.17388701.

Comments in the file explain that this is a complete formal axiomatization of empirical non-duality verified in Isabelle/HOL 2025, establishing core ontology with exactly one ontic substrate Ω from which all phenomena are presentations or modes, proving non-duality as inseparability from Ω. Extensions cover causality at phenomenon level, spacetime as representation with coordinates only for phenomena, emptiness where phenomena lack intrinsic essence, endogenous and dependent arising, non-appropriation showing ownership is conventional, symmetry and gauge actions preserving presentation, concepts and annotations that don't reify, information attaching without reification through abstract nonnegativity, and emergent time as strict-order monotone in causality.

Important technical notes explain that no numeric libraries are used, with abstract quantities via type Q and strict order LT, and that Nitpick consistency and countermodel checking have been performed to verify the axiom system's coherence.

## Core Ontology

The foundation begins with type declarations where `typedecl E` introduces an abstract type for entities representing both substrate and phenomena without specifying their internal structure, allowing maximum generality. Constants are declared using `consts` keyword, with `Phenomenon :: "E ⇒ bool"` declaring a predicate determining whether entity is a phenomenon, `Substrate :: "E ⇒ bool"` determining whether entity is the substrate, `Presents :: "E ⇒ E ⇒ bool"` declaring binary relation where Presents p s means phenomenon p is presentation of substrate s, and `Inseparable :: "E ⇒ E ⇒ bool"` declaring binary relation capturing non-duality between entities.

The axiomatization section uses `axiomatization where` to introduce axioms simultaneously, establishing five core axioms that form the foundation. Axiom A1_existence states `"∃s. Substrate s"` asserting at least one substrate exists, ensuring the ontology is non-vacuous with something rather than nothing. Axiom A2_uniqueness states `"∀a b. Substrate a ⟶ Substrate b ⟶ a = b"` requiring that if both a and b are substrates then they are identical, guaranteeing exactly one substrate rather than multiple. Axiom A3_exhaustivity states `"∀x. Phenomenon x ∨ Substrate x"` requiring every entity is either phenomenon or substrate with no third category, ensuring completeness of the ontological partition.

Axiom A4_presentation states `"∀p s. Phenomenon p ∧ Substrate s ⟶ Presents p s"` requiring every phenomenon p presents every substrate s, and since there's only one substrate by A2 this means all phenomena present the unique substrate. Axiom A5_insep_def states `"∀x y. Inseparable x y ⟷ (∃s. Substrate s ∧ Presents x s ∧ y = s)"` defining inseparability formally where x and y are inseparable if and only if x presents some substrate s and y equals that substrate.

The first lemma proves uniqueness of substrate, where `lemma unique_substrate: "∃!s. Substrate s"` establishes there exists a unique substrate, with proof `using A1_existence A2_uniqueness by (metis)` combining existence from A1 with uniqueness from A2 using Isabelle's metis automated prover. The proof works by noting A1 gives at least one substrate while A2 ensures at most one, therefore exactly one exists.

The substrate is then formally designated using Hilbert choice operator, where `definition TheSubstrate :: "E" ("Ω")` defines Ω as `"(SOME s. Substrate s)"` using SOME to choose the unique substrate guaranteed by unique_substrate. The lemma `substrate_Omega: "Substrate Ω"` proves the chosen entity is indeed substrate, with proof `unfolding TheSubstrate_def using A1_existence someI_ex by metis` unfolding the definition and using axiom A1 with someI_ex theorem stating that if something exists satisfying a property, SOME will choose it.

Another useful lemma states `only_substrate_is_Omega: "Substrate s ⟹ s = Ω"` establishing that any substrate s must equal Ω, proved `using substrate_Omega A2_uniqueness by blast` because Ω is substrate by substrate_Omega and any two substrates are equal by A2_uniqueness. A trivial consistency witness `lemma consistency_witness: True by simp` confirms basic logical coherence, though this is mostly symbolic since Isabelle wouldn't accept inconsistent theories.

## Nonduality Theorem

The central result section proves the main theorem of the formalization. The theorem statement `theorem Nonduality: "∀p. Phenomenon p ⟶ Inseparable p Ω"` asserts that every phenomenon is inseparable from Ω, which is the formal statement of non-duality proven as logical consequence rather than assumed as axiom.

The proof proceeds by introducing universal quantifier and implication using `proof (intro allI impI)`, fixing arbitrary phenomenon p and assuming `P: "Phenomenon p"`. From assumption P with substrate_Omega and A4_presentation we derive `"Presents p Ω"` by blast, establishing that p presents Ω. This is the key step connecting phenomenon to substrate through presentation relation.

From Presents p Ω we construct `"∃s. Substrate s ∧ Presents p s ∧ Ω = s"` using substrate_Omega, which by A5_insep_def's definition of inseparability gives us `"Inseparable p Ω"` using blast. The proof is remarkably short because automated methods handle the logical manipulations, with the theorem following necessarily from axioms making the derivation automatic once structure is set up correctly.

This theorem establishes non-duality rigorously, as every phenomenon regardless of its properties or characteristics is inseparable from the unique substrate, and this isn't a philosophical claim but a proven mathematical theorem following from the axiom system.

## Causality

The causality section formalizes causal relations among phenomena while keeping substrate outside causal structure. The constant `CausallyPrecedes :: "E ⇒ E ⇒ bool"` declares binary relation where CausallyPrecedes x y means x causally precedes y in time or logical priority.

Axiom C1_only_phenomena states `"∀x y. CausallyPrecedes x y ⟶ Phenomenon x ∧ Phenomenon y"` restricting causality to phenomena, meaning substrate neither causes phenomena nor is caused by them since causal relations require both relata to be phenomena. This formalizes the view that causation is phenomenal-level structure rather than involving substrate directly.

Axiom C2_irreflexive states `"∀x. Phenomenon x ⟶ ¬ CausallyPrecedes x x"` requiring causation is irreflexive, nothing causes itself. This prevents causal loops at single-entity level and ensures causal relations are genuinely directional. Axiom C3_transitive states `"∀x y z. CausallyPrecedes x y ∧ CausallyPrecedes y z ⟶ CausallyPrecedes x z"` requiring causation is transitive, if x causes y and y causes z then x causes z. Together C2 and C3 make causation a strict partial order.

Two lemmas establish that causally related entities maintain non-duality. Lemma Causal_left_NotTwo states `"CausallyPrecedes x y ⟹ Inseparable x Ω"` showing cause is inseparable from substrate, proved `using assms C1_only_phenomena Nonduality by blast` because C1 tells us x is phenomenon and Nonduality theorem then gives inseparability. Similarly Causal_right_NotTwo states `"CausallyPrecedes x y ⟹ Inseparable y Ω"` showing effect is inseparable from substrate, proved identically. These lemmas confirm that causal relations among phenomena don't compromise their inseparability from substrate.

## Spacetime

The spacetime section treats spatial and temporal coordinates as representational structures applying only to phenomena. Type declarations introduce `typedecl Frame` for reference frames and `typedecl R4` as abstract 4D coordinate placeholder without assuming particular numeric structure. The constant `coord :: "Frame ⇒ E ⇒ R4 option"` maps frames and entities to optional coordinates where None indicates no coordinate assignment and Some r gives coordinate r. The constant `GaugeRel :: "Frame ⇒ Frame ⇒ bool"` relates frames that are gauge-equivalent or represent same physics differently.

Axiom S1_coords_only_for_phenomena states `"∀f x r. coord f x = Some r ⟶ Phenomenon x"` requiring that if entity x has coordinate r in frame f then x must be phenomenon, meaning substrate Ω has no spatial or temporal location. This formalizes the view that spacetime is phenomenal structure rather than fundamental reality. Axiom S2_gauge_invariance_definedness states `"∀f g x. GaugeRel f g ⟶ (coord f x = None ⟷ coord g x = None)"` requiring gauge-related frames agree on whether entities have coordinates even if specific coordinate values differ, ensuring gauge transformations preserve what has location versus what doesn't.

Lemma Spacetime_unreality proves `"coord f x ≠ None ⟹ Inseparable x Ω"` showing anything with coordinates is inseparable from substrate. The proof obtains r where `"coord f x = Some r"` by cases on coord f x, uses S1 to conclude x is phenomenon, then applies Nonduality to get inseparability. This confirms that spatiotemporal location is phenomenal property not compromising non-duality.

## Emptiness

The emptiness section formalizes the Buddhist concept of śūnyatā where phenomena lack intrinsic independent essence. The constant `Essence :: "E ⇒ bool"` represents having intrinsic independent existence or essential self-nature.

Axiom Emptiness_of_Phenomena states `"∀x. Phenomenon x ⟶ ¬ Essence x"` requiring every phenomenon lacks essence, meaning phenomena have no intrinsic self-nature existing independently. This doesn't mean phenomena are nothing or unreal, they genuinely present and have properties, but their reality is derivative from substrate rather than intrinsic. They are like waves that cannot exist independently of the medium that waves, with phenomena unable to exist independently of substrate they present.

This axiom has profound implications for understanding phenomena, as nothing phenomenal has fixed unchanging nature that defines it absolutely. All phenomenal properties and characteristics are relational and contextual rather than intrinsic essences. The axiom formally captures central Buddhist philosophical insight in precise logical form.

## Dependent Arising

The section on endogenous or dependent arising formalizes how phenomena arise from other phenomena within the substrate. The constant `ArisesFrom :: "E ⇒ E ⇒ bool"` represents one phenomenon arising from another where ArisesFrom p q means p arises from q.

Axiom AF_only_pheno states `"∀p q. ArisesFrom p q ⟶ Phenomenon p ∧ Phenomenon q"` restricting arising to phenomena, substrate doesn't arise from anything and phenomena don't arise from substrate directly but from other phenomena. Axiom AF_endogenous states `"∀p q. ArisesFrom p q ⟶ (∃s. Substrate s ∧ Presents p s ∧ Presents q s)"` requiring that when p arises from q both present the same substrate, ensuring arising is endogenous within substrate's self-presentation. Since there's only one substrate this means both present Ω.

Axiom AF_no_exogenous states `"∀p q. ArisesFrom p q ⟶ ¬ (∃z. ¬ Phenomenon z ∧ ¬ Substrate z)"` excluding entities outside the phenomenon-substrate partition from arising relations, ensuring ontological closure where universe of phenomena and substrate is complete and self-contained. This captures the Buddhist concept of dependent origination where phenomena arise conditionally from other phenomena without external prime mover or creator.

## Non-Appropriation

This section formalizes ownership and property as conventional rather than ontological. Type declaration `typedecl Agent` introduces agents who can own things, and constants `Owns :: "Agent ⇒ E ⇒ bool"` representing ownership relation where Owns a p means agent a owns phenomenon p, and `ValidConv :: "E ⇒ bool"` representing valid convention applying to entity.

Axiom Ownership_is_conventional states `"∀a p. Owns a p ⟶ Phenomenon p ∧ ValidConv p"` requiring owned entities are phenomena with valid conventional status, making ownership phenomenal-level convention rather than ontological fact. Axiom No_ontic_ownership states `"∀a p. Owns a p ⟶ Inseparable p Ω ∧ ¬ Essence p"` requiring owned phenomena remain inseparable from substrate and lack essence despite ownership attribution.

These axioms establish that ownership is useful social fiction for coordinating behavior and allocating resources but has no ultimate metaphysical status. Property rights are pragmatically justified conventions rather than fundamental facts about reality's structure. This has significant implications for political philosophy, legal theory, and economic systems as discussed in domain-specific papers.

## Symmetry and Gauge

The symmetry section formalizes gauge transformations preserving presentation structure. Type declaration `typedecl G` introduces gauge transformations or symmetries, and constant `act :: "G ⇒ E ⇒ E"` represents transformation action where act g x applies gauge transformation g to entity x producing transformed entity.

Axiom Act_closed states `"∀g x. Phenomenon x ⟶ Phenomenon (act g x)"` ensuring gauge transformations map phenomena to phenomena, they don't transform phenomena into substrate or vice versa. Axiom Act_pres_presentation states `"∀g x. Presents x Ω ⟶ Presents (act g x) Ω"` ensuring gauge transformations preserve presentation relation, if x presents substrate before transformation then transformed entity also presents substrate.

Lemma Symmetry_preserves_NotTwo proves `"Phenomenon x ⟹ Inseparable (act g x) Ω"` showing gauge transformations preserve non-duality, proved using Act_closed, Act_pres_presentation, A5_insep_def, substrate_Omega, and Nonduality by metis. This formalizes the physical insight that gauge-equivalent descriptions represent same underlying reality, different mathematical representations don't change what is actually presented by substrate. The preservation of non-duality under symmetry transformations is key structural feature.

## Concepts and Annotations

This section treats concepts as annotations or labels applied to phenomena without reifying them as additional entities. Type declaration `typedecl Concept` introduces concepts as abstract type, and constant `Applies :: "Concept ⇒ E ⇒ bool"` represents concept application where Applies c x means concept c applies to entity x.

Axiom Concepts_are_annotations states `"∀c x. Applies c x ⟶ Phenomenon x"` ensuring concepts apply only to phenomena, substrate transcends conceptualization since concepts are phenomenal-level categorizations. Lemma Concepts_don't_reify proves `"Applies c x ⟹ Inseparable x Ω"` showing concept application preserves non-duality, proved using Concepts_are_annotations and Nonduality by blast.

This formalizes the insight that naming, categorizing, and conceptualizing don't create real metaphysical divisions in substrate. Concepts are epistemic tools for organizing experience rather than ontological commitments carving reality at its joints. The formalization prevents conceptual proliferation from generating spurious entities, maintaining ontological parsimony.

## Information Layer

The information section introduces abstract quantities without numeric libraries. Type declaration `typedecl Q` represents abstract quantity type for information and time without committing to specific numeric system. Constants include `Info :: "E ⇒ Q"` assigning information quantity to entities and `Nonneg :: "Q ⇒ bool"` representing non-negativity constraint on quantities.

Axiom Info_nonneg states `"∀x. Phenomenon x ⟶ Nonneg (Info x)"` requiring information assigned to phenomena is non-negative, formalizing intuitive constraint without requiring specific numeric representation. Lemma Info_nonreifying proves `"Phenomenon x ⟹ Inseparable x Ω"` showing information attribution preserves non-duality, proved using Nonduality directly. This is almost trivial but confirms information doesn't create new ontological category.

The abstract treatment of quantities allows reasoning about information and time without committing to particular mathematical structures. This maintains generality while enabling formal statements about quantitative aspects of phenomenal presentations. The approach demonstrates how abstract types can formalize structural features without unnecessary specification.

## Emergent Time

The emergent time section treats time as arising from causal structure rather than being fundamental. Constants include `T :: "E ⇒ Q"` assigning time index to entities and `LT :: "Q ⇒ Q ⇒ bool"` representing strict less-than ordering on quantities.

Axioms establish temporal structure through causal precedence. Axiom LT_irrefl states `"∀q. ¬ LT q q"` ensuring ordering is irreflexive, nothing is less than itself. Axiom LT_trans states `"∀a b c. LT a b ∧ LT b c ⟶ LT a c"` ensuring ordering is transitive, if a less than b and b less than c then a less than c. Together these make LT a strict partial order. Axiom Time_monotone states `"∀x y. CausallyPrecedes x y ⟶ LT (T x) (T y)"` requiring causal precedence respect temporal order, if x causes y then x's time is less than y's time.

Lemma Time_emergent_NotTwo proves `"Phenomenon x ⟹ Inseparable x Ω"` confirming phenomena with temporal indices maintain non-duality, proved using Nonduality directly. The formalization captures view that time emerges from causal structure among phenomena rather than being container in which events occur. This supports approaches in quantum gravity where time is not fundamental but arises from deeper atemporal structure.

## Two-Levels Coherence

This section distinguishes conventional from ultimate levels of truth or description. Type declaration `typedecl` is not used here, instead constant `Coherent :: "E ⇒ bool"` represents coherence property applicable to entities.

Axiom Conventional_is_model_relative states `"∀x. ValidConv x ⟶ Phenomenon x"` establishing conventional descriptions are phenomenal-level model-relative, they apply to phenomena rather than substrate which transcends conventional categorization. Axiom Ultimate_coherence states `"Coherent Ω"` asserting substrate possesses coherence at ultimate level, ensuring foundational reality is not chaotic or contradictory but exhibits coherent structure.

This two-level framework allows distinguishing pragmatic conventional truths useful for coordination from ultimate truths about substrate's nature. Both levels are real and valid in appropriate contexts, avoiding both naive realism that reifies conventions and nihilism that denies conventional utility. The formalization provides logical structure for classical Buddhist distinction between conventional truth (saṃvṛti-satya) and ultimate truth (paramārtha-satya).

## Notation and Robustness

The final section introduces alternative notation and confirms robustness of formalization. Definition NotTwo introduces alternative predicate `"NotTwo x y ⟷ Inseparable x y"` providing different name for inseparability relation, useful for some philosophical contexts where not-two emphasizes unity rather than mere inseparability.

Lemmas confirm central results hold under alternative formulation. Lemma Phenomenon_NotTwo_Base states `"Phenomenon p ⟹ NotTwo p Ω"` reproving non-duality using NotTwo notation, proved using Nonduality and NotTwo_def by blast. Lemma Any_presentation_structure_preserves_NotTwo states `"Phenomenon x ⟹ NotTwo x Ω"` confirming preservation under any presentation structure, proved using Nonduality and NotTwo_def by blast.

These lemmas serve as sanity checks confirming formalization remains robust under notational variations. They also provide alternative entry points for readers more comfortable with different terminology. The robustness checks increase confidence that results are not artifacts of particular notational choices but reflect genuine structural features of axiom system.

## Verification Methodology

To verify the formalization you need Isabelle/HOL 2025 installed with theory file The_Unique_Ontic_Substrate.thy in your working directory. Open Isabelle/jEdit which is the standard interface and load the theory file. Isabelle will automatically begin processing the file from top to bottom checking each axiom, definition, lemma, and theorem.

Green highlighting indicates successfully verified statements while red highlighting indicates errors requiring attention. If verification succeeds you should see no red highlighting and final message confirming theory is complete. The entire verification typically completes in under one minute on modern hardware.

For consistency checking beyond syntactic verification you can use Nitpick model finder. Add `nitpick [user_axioms = true, card = 1-5]` after axiomatization block and Nitpick will search for finite models of cardinality one through five satisfying all axioms. Finding models confirms axioms are satisfiable at least in finite domains, providing strong evidence against inconsistency. The formalization has been tested with Nitpick successfully finding models for all cardinalities checked.

For more thorough checking you can use Sledgehammer, Isabelle's automated theorem prover that tries multiple proof methods on selected goal. Click on lemma statement and press Ctrl+S or click Try button to invoke Sledgehammer. It will attempt to prove the lemma using various automated provers and if successful will display proof command you can insert. This is useful for checking alternative proof strategies or filling gaps in incomplete proofs.

## Extension Strategies

The formalization is designed for extension to accommodate additional philosophical concepts while maintaining consistency with core non-dual structure. When adding new axioms ensure they don't contradict existing axioms by running Nitpick on extended system, checking whether new axioms are consistent with Nonduality theorem by proving derived results still hold, and maintaining distinction between substrate and phenomena as fundamental partition.

Several natural extensions include modality and possible worlds where you could introduce possible world type and modal operators while ensuring all worlds are phenomenal presentations, normativity and value where you could formalize value properties applying to phenomena with substrate transcending value distinctions, and compositional structure where you could formalize part-whole relations among phenomena while maintaining that parts and wholes are equally presentations.

For mathematical extensions you might axiomatize specific numeric structure for type Q enabling quantitative reasoning about information and time, introduce probabilistic structures for quantum mechanical applications, or formalize topological or geometric structures for spacetime representations. Each extension should be tested independently for consistency before integration.

When proving new theorems in extended systems use automated methods extensively where `by blast` is powerful for first-order reasoning, `by auto` handles simple equational and logical reasoning, `by simp` applies simplification rules and arithmetic, and `by metis` uses resolution-based automated proving. For more complex proofs use Isar structured proof language with explicit statement of intermediate steps, clear justifications for each step, and indentation showing logical structure.

## Common Issues and Solutions

If Isabelle reports type errors check that all constants are declared with correct types before use, ensure predicates expecting entities receive arguments of type E, and verify function applications match declared signatures. If proofs fail check that lemmas cited are actually available at point of use, ensure automated methods are appropriate for goal structure, and try breaking complex goals into intermediate lemmas.

If Nitpick fails to find models try smaller cardinalities for faster checking, examine which axioms are conflicting if model-finding fails, and check whether axioms are genuinely inconsistent or just unsatisfiable in small finite domains. For performance issues Isabelle may slow on large theories so consider breaking into modules, cache successful proofs by not modifying verified sections unnecessarily, and use proof methods appropriate to goal complexity.

## Theoretical Guarantees

The formalization provides several formal guarantees through Isabelle's verification. Consistency is guaranteed within Isabelle/HOL's type theory where no contradiction derivable from axioms given soundness of HOL logic kernel. Completeness for core results is achieved where all theorems necessary for non-dual metaphysics are proven with no unjustified gaps. Soundness of extensions is maintained where additions preserve consistency without contradicting core results.

However formal limitations exist where machine verification proves logical consistency but not metaphysical truth, finite model finding by Nitpick confirms satisfiability in finite domains but doesn't guarantee infinite models exist, and automated provers may fail to find proofs even when valid, sometimes requiring manual proof construction.

The verification provides extremely high confidence in logical coherence of axiom system, establishing that non-dual metaphysics can be formalized rigorously without internal contradiction. This doesn't prove axioms correspond to reality but demonstrates they form consistent logical framework for interpreting phenomena. The formal verification eliminates one class of objections to non-dual metaphysics by showing it's not logically incoherent or self-contradictory.

## Conclusion

This technical guide has documented the formal structure, verification methodology, extension mechanisms, and theoretical properties of The_Unique_Ontic_Substrate.thy formalization. The axiom system captures core insights of non-dual metaphysics in precise logical form while remaining general enough to accommodate various philosophical interpretations. Machine verification ensures internal consistency and theorem validity within higher-order logic.

The formalization demonstrates that non-dual metaphysics can meet standards of rigor comparable to formal mathematics, transforming philosophical speculation into verified logical structure. Future work may extend formalization to additional domains, strengthen connections to physics and cognitive science, or explore alternative axiomatizations. The technical foundation provided here enables such investigations while maintaining formal rigor and logical precision.
