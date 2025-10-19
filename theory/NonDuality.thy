theory NonDuality
  imports Main
begin

(*
  SECULAR NON-DUALITY — COMPLETE (Prefix-Only, Build-Robust, No-Measurement)
  Isabelle/HOL 2025

  Core:
    - Exactly one ontic substrate \<Omega>.
    - All phenomena are presentations (modes) of \<Omega>.
    - Non-duality = inseparability from \<Omega>.

  Extensions kept:
    - Causality (phenomenon-level).
    - Spacetime as representation (coordinates only for phenomena).
    - Emptiness: phenomena lack intrinsic essence.
    - Endogenous/Dependent arising.
    - Non-appropriation (ownership is conventional).
    - Symmetry/Gauge actions preserve presentation.
    - Concepts/annotations don’t reify.
    - Information attaches without reification (abstract nonnegativity).
    - Emergent time: strict-order monotone in causality.

  Notes:
    - No numeric libraries; abstract quantities via type Q and strict order LT.
    - Nitpick consistency & countermodel checks included.
*)

nitpick_params [user_axioms = true, show_all, format = 3, card = 1,2,3,4,5, max_threads = 2]

section \<open>Core Ontology\<close>

typedecl E  (* entities: substrate and phenomena *)

consts
  Phenomenon :: "E \<Rightarrow> bool"
  Substrate  :: "E \<Rightarrow> bool"
  Presents   :: "E \<Rightarrow> E \<Rightarrow> bool"   (* Presents p s  =  p is a presentation of s *)
  Inseparable :: "E \<Rightarrow> E \<Rightarrow> bool"

axiomatization where
  A1_existence:     "\<exists>s. Substrate s" and
  A2_uniqueness:    "\<forall>a b. Substrate a \<longrightarrow> Substrate b \<longrightarrow> a = b" and
  A3_exhaustivity:  "\<forall>x. Phenomenon x \<or> Substrate x" and
  A4_presentation:  "\<forall>p s. Phenomenon p \<and> Substrate s \<longrightarrow> Presents p s" and
  A5_insep_def:     "\<forall>x y. Inseparable x y \<longleftrightarrow> (\<exists>s. Substrate s \<and> Presents x s \<and> y = s)"

lemma unique_substrate: "\<exists>!s. Substrate s"
  using A1_existence A2_uniqueness by (metis)

definition TheSubstrate :: "E"  ("\<Omega>")
  where "\<Omega> = (SOME s. Substrate s)"

lemma substrate_Omega: "Substrate \<Omega>"
  unfolding TheSubstrate_def using A1_existence someI_ex by blast

lemma only_substrate_is_Omega: "Substrate s \<Longrightarrow> s = \<Omega>"
  using substrate_Omega A2_uniqueness by blast

lemma consistency_witness: True by simp
nitpick [satisfy, expect = genuine]  (* find a finite model of the axioms *)

section \<open>Non-Duality\<close>

theorem Nonduality:
  "\<forall>p. Phenomenon p \<longrightarrow> Inseparable p \<Omega>"
proof (intro allI impI)
  fix p assume P: "Phenomenon p"
  from P substrate_Omega A4_presentation have "Presents p \<Omega>" by blast
  hence "\<exists>s. Substrate s \<and> Presents p s \<and> \<Omega> = s"
    using substrate_Omega by blast
  thus "Inseparable p \<Omega>"
    using A5_insep_def by blast
qed

nitpick [expect = none]  (* try to refute Nonduality; should fail *)

section \<open>Causality (Phenomenon-Level)\<close>

consts CausallyPrecedes :: "E \<Rightarrow> E \<Rightarrow> bool"   (* CausallyPrecedes x y *)

axiomatization where
  C1_only_phenomena: "\<forall>x y. CausallyPrecedes x y \<longrightarrow> Phenomenon x \<and> Phenomenon y" and
  C2_irreflexive:    "\<forall>x. Phenomenon x \<longrightarrow> \<not> CausallyPrecedes x x" and
  C3_transitive:     "\<forall>x y z. CausallyPrecedes x y \<and> CausallyPrecedes y z \<longrightarrow> CausallyPrecedes x z"

lemma Causal_left_NotTwo:
  assumes "CausallyPrecedes x y" shows "Inseparable x \<Omega>"
  using assms C1_only_phenomena Nonduality by blast

lemma Causal_right_NotTwo:
  assumes "CausallyPrecedes x y" shows "Inseparable y \<Omega>"
  using assms C1_only_phenomena Nonduality by blast

section \<open>Spacetime as Representation (Coordinates only for Phenomena)\<close>

typedecl Frame
typedecl R4     (* abstract 4D coordinate placeholder *)

consts
  coord    :: "Frame \<Rightarrow> E \<Rightarrow> R4 option"
  GaugeRel :: "Frame \<Rightarrow> Frame \<Rightarrow> bool"

axiomatization where
  S1_coords_only_for_phenomena:
    "\<forall>f x r. coord f x = Some r \<longrightarrow> Phenomenon x" and
  S2_gauge_invariance_definedness:
    "\<forall>f g x. GaugeRel f g \<longrightarrow> (coord f x = None \<longleftrightarrow> coord g x = None)"

lemma Spacetime_unreality:
  assumes "coord f x \<noteq> None"
  shows "Inseparable x \<Omega>"
proof -
  from assms obtain r where "coord f x = Some r" by (cases "coord f x") auto
  hence "Phenomenon x" using S1_coords_only_for_phenomena by blast
  thus "Inseparable x \<Omega>" using Nonduality by blast
qed

section \<open>Emptiness: No Intrinsic Essence of Phenomena\<close>

consts Essence :: "E \<Rightarrow> bool"

axiomatization where
  Emptiness_of_Phenomena: "\<forall>x. Phenomenon x \<longrightarrow> \<not> Essence x"

section \<open>Endogenous / Dependent Arising\<close>

consts ArisesFrom :: "E \<Rightarrow> E \<Rightarrow> bool"   (* ArisesFrom p q *)

axiomatization where
  AF_only_pheno:   "\<forall>p q. ArisesFrom p q \<longrightarrow> Phenomenon p \<and> Phenomenon q" and
  AF_endogenous:   "\<forall>p q. ArisesFrom p q \<longrightarrow> (\<exists>s. Substrate s \<and> Presents p s \<and> Presents q s)" and
  AF_no_exogenous: "\<forall>p q. ArisesFrom p q \<longrightarrow> \<not> (\<exists>z. \<not> Phenomenon z \<and> \<not> Substrate z)"

section \<open>Non-Appropriation (Ownership is Conventional)\<close>

typedecl Agent
consts Owns :: "Agent \<Rightarrow> E \<Rightarrow> bool"
consts ValidConv :: "E \<Rightarrow> bool"

axiomatization where
  Ownership_is_conventional:
    "\<forall>a p. Owns a p \<longrightarrow> Phenomenon p \<and> ValidConv p" and
  No_ontic_ownership:
    "\<forall>a p. Owns a p \<longrightarrow> Inseparable p \<Omega> \<and> \<not> Essence p"

section \<open>Symmetry / Gauge on Phenomena\<close>

typedecl G
consts act :: "G \<Rightarrow> E \<Rightarrow> E"   (* act g x *)

axiomatization where
  Act_closed:            "\<forall>g x. Phenomenon x \<longrightarrow> Phenomenon (act g x)" and
  Act_pres_presentation: "\<forall>g x. Presents x \<Omega> \<longrightarrow> Presents (act g x) \<Omega>"

lemma Symmetry_preserves_NotTwo:
  assumes "Phenomenon x" shows "Inseparable (act g x) \<Omega>"
  using assms Act_closed Act_pres_presentation A5_insep_def substrate_Omega by blast

section \<open>Concepts / Annotations\<close>

typedecl Concept
consts Applies :: "Concept \<Rightarrow> E \<Rightarrow> bool"

axiomatization where
  Concepts_are_annotations:
    "\<forall>c x. Applies c x \<longrightarrow> Phenomenon x"

lemma Concepts_don't_reify:
  assumes "Applies c x" shows "Inseparable x \<Omega>"
  using assms Concepts_are_annotations Nonduality by blast

section \<open>Quantities for Information and Time\<close>

typedecl Q      (* abstract quantity type *)

section \<open>Information Layer (Abstract Nonnegativity)\<close>

consts
  Info   :: "E \<Rightarrow> Q"
  Nonneg :: "Q \<Rightarrow> bool"

axiomatization where
  Info_nonneg: "\<forall>x. Phenomenon x \<longrightarrow> Nonneg (Info x)"

lemma Info_nonreifying:
  assumes "Phenomenon x" shows "Inseparable x \<Omega>"
  using assms Nonduality by blast

section \<open>Emergent Time (Abstract Strict Order on Q)\<close>

consts
  T  :: "E \<Rightarrow> Q"           (* time index *)
  LT :: "Q \<Rightarrow> Q \<Rightarrow> bool"   (* strict order on Q *)

axiomatization where
  LT_irrefl:     "\<forall>q. \<not> LT q q" and
  LT_trans:      "\<forall>a b c. LT a b \<and> LT b c \<longrightarrow> LT a c" and
  Time_monotone: "\<forall>x y. CausallyPrecedes x y \<longrightarrow> LT (T x) (T y)"

lemma Time_emergent_NotTwo:
  assumes "Phenomenon x" shows "Inseparable x \<Omega>"
  using assms Nonduality by blast

section \<open>Two-Levels Coherence\<close>

consts Coherent :: "E \<Rightarrow> bool"

axiomatization where
  Conventional_is_model_relative: "\<forall>x. ValidConv x \<longrightarrow> Phenomenon x" and
  Ultimate_coherence:             "Coherent \<Omega>"

section \<open>Notation and Robustness\<close>

definition NotTwo :: "E \<Rightarrow> E \<Rightarrow> bool"
  where "NotTwo x y \<longleftrightarrow> Inseparable x y"

lemma Phenomenon_NotTwo_Base: "Phenomenon p \<Longrightarrow> NotTwo p \<Omega>"
  using Nonduality NotTwo_def by blast

lemma Any_presentation_structure_preserves_NotTwo:
  assumes "Phenomenon x" shows "NotTwo x \<Omega>"
  using assms Nonduality NotTwo_def by blast

end
