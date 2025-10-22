theory Daoism
  imports Main
begin

(* Global finite-scope model search discipline for all goals *)
nitpick_params [user_axioms = true, format = 3, show_all, max_threads = 2, card = 1,2,3,4,5]

(* ========================================================================= *)
(* FORMAL AXIOMATIZATION OF DAOISM                                           *)
(* ========================================================================= *)
(*
  Copyright (C) 2025 Matthew Scherf
  CC BY 4.0 (docs) + BSD-3-Clause (code)
*)

section \<open>Domain and Core Predicates\<close>

typedecl entity

consts
  Dao :: "entity \<Rightarrow> bool"
  TenThousandThings :: "entity \<Rightarrow> bool"
  Formless :: "entity \<Rightarrow> bool"
  Nameless :: "entity \<Rightarrow> bool"
  HasForm :: "entity \<Rightarrow> bool"
  Exists :: "entity \<Rightarrow> bool"

consts
  TrueMan :: "entity \<Rightarrow> bool"

consts
  ArisesFr :: "entity \<Rightarrow> entity \<Rightarrow> bool"
  ReturnsTo :: "entity \<Rightarrow> entity \<Rightarrow> bool"

section \<open>Fundamental Axioms\<close>

axiomatization where
  A1_dao_exists: "\<exists>!d. Dao d"

axiomatization where
  A2a_dao_formless: "\<forall>d. Dao d \<longrightarrow> Formless d" and
  A2b_dao_nameless: "\<forall>d. Dao d \<longrightarrow> Nameless d"

axiomatization where
  A3_things_have_form: "\<forall>x. TenThousandThings x \<longrightarrow> HasForm x"

axiomatization where
  A4_form_exclusion: "\<forall>x. HasForm x \<longrightarrow> \<not>Formless x"

axiomatization where
  A5_dao_not_thing: "\<forall>d. Dao d \<longrightarrow> \<not>TenThousandThings d"

axiomatization where
  A6_things_arise_from_dao: "\<forall>x. TenThousandThings x \<longrightarrow> (\<exists>d. Dao d \<and> ArisesFr x d)"

axiomatization where
  A6b_only_from_dao: "\<forall>x y. ArisesFr x y \<longrightarrow> Dao y"

axiomatization where
  A7_things_return_to_dao: "\<forall>x. TenThousandThings x \<longrightarrow> (\<exists>d. Dao d \<and> ReturnsTo x d)"

axiomatization where
  A7b_only_to_dao: "\<forall>x y. ReturnsTo x y \<longrightarrow> Dao y"

axiomatization where
  A8_same_dao:
    "\<forall>x d1 d2. (ArisesFr x d1 \<and> ReturnsTo x d2 \<and> Dao d1 \<and> Dao d2) \<longrightarrow> d1 = d2"

axiomatization where
  A9_you_are_trueman: "\<exists>!u. TrueMan u"

axiomatization where
  A10_trueman_is_dao: "\<forall>u d. (TrueMan u \<and> Dao d) \<longrightarrow> u = d"

section \<open>Non-vacuity of Manifestation\<close>

(* NEW: rule out the degenerate 1-element world (no manifested things). *)
axiomatization where
  NV1_exists_thing: "\<exists>x. TenThousandThings x"

section \<open>Core Theorems\<close>

theorem T1_dao_unique: "\<exists>!d. Dao d"
  using A1_dao_exists by blast

theorem T2_dao_no_form: "\<forall>d. Dao d \<longrightarrow> \<not>HasForm d"
  using A2a_dao_formless A4_form_exclusion by blast

theorem T3_things_distinct_in_form:
  "\<forall>x d. (TenThousandThings x \<and> Dao d) \<longrightarrow> x \<noteq> d"
  using A3_things_have_form A2a_dao_formless A4_form_exclusion by blast

theorem T4_one_source_and_return:
  "\<forall>x y d1 d2. (TenThousandThings x \<and> TenThousandThings y \<and> ArisesFr x d1 \<and> ArisesFr y d2) \<longrightarrow> d1 = d2"
proof (intro allI impI)
  fix x y d1 d2
  assume "TenThousandThings x \<and> TenThousandThings y \<and> ArisesFr x d1 \<and> ArisesFr y d2"
  then have "ArisesFr x d1" "ArisesFr y d2" by auto
  have "Dao d1" using \<open>ArisesFr x d1\<close> A6b_only_from_dao by blast
  have "Dao d2" using \<open>ArisesFr y d2\<close> A6b_only_from_dao by blast
  from A1_dao_exists \<open>Dao d1\<close> \<open>Dao d2\<close> show "d1 = d2" by auto
qed

theorem T5_you_are_dao: "\<forall>u d. (TrueMan u \<and> Dao d) \<longrightarrow> u = d"
  using A10_trueman_is_dao by blast

theorem T6_you_formless: "\<forall>u. TrueMan u \<longrightarrow> Formless u"
  by (metis A2a_dao_formless A10_trueman_is_dao A1_dao_exists)

theorem T7_you_nameless: "\<forall>u. TrueMan u \<longrightarrow> Nameless u"
  by (metis A2b_dao_nameless A10_trueman_is_dao A1_dao_exists)

theorem T8_you_not_thing: "\<forall>u. TrueMan u \<longrightarrow> \<not>TenThousandThings u"
  by (metis A5_dao_not_thing A10_trueman_is_dao A1_dao_exists)

theorem T9_things_arise_from_you:
  "\<forall>u x. (TrueMan u \<and> TenThousandThings x) \<longrightarrow> ArisesFr x u"
  by (metis A6_things_arise_from_dao A10_trueman_is_dao A1_dao_exists)

theorem T10_things_return_to_you:
  "\<forall>u x. (TrueMan u \<and> TenThousandThings x) \<longrightarrow> ReturnsTo x u"
  by (metis A7_things_return_to_dao A10_trueman_is_dao A1_dao_exists)

section \<open>Extension 1: Spontaneity and Non-Causation\<close>

consts
  Spontaneous :: "entity \<Rightarrow> bool"
  Caused :: "entity \<Rightarrow> entity \<Rightarrow> bool"

axiomatization where
  S1_spontaneous_arising: "\<forall>x. TenThousandThings x \<longrightarrow> Spontaneous x"

axiomatization where
  S2_no_causation: "\<forall>x y. Spontaneous x \<longrightarrow> \<not>Caused y x"

axiomatization where
  S3_dao_wu_wei: "\<forall>d x. (Dao d \<and> ArisesFr x d) \<longrightarrow> \<not>Caused d x"

theorem TS1_no_real_causation:
  "\<forall>x y. TenThousandThings x \<longrightarrow> \<not>Caused y x"
  using S1_spontaneous_arising S2_no_causation by blast

theorem TS2_dao_not_cause:
  "\<forall>d x. (Dao d \<and> TenThousandThings x) \<longrightarrow> \<not>Caused d x"
proof (intro allI impI)
  fix d x
  assume "Dao d \<and> TenThousandThings x"
  then have "Dao d" "TenThousandThings x" by auto
  from \<open>TenThousandThings x\<close> A6_things_arise_from_dao obtain d' where "Dao d' \<and> ArisesFr x d'" by blast
  then have "Dao d'" "ArisesFr x d'" by auto
  from A1_dao_exists \<open>Dao d\<close> \<open>Dao d'\<close> have "d = d'" by auto
  from \<open>Dao d'\<close> \<open>ArisesFr x d'\<close> S3_dao_wu_wei have "\<not>Caused d' x" by blast
  with \<open>d = d'\<close> show "\<not>Caused d x" by simp
qed

section \<open>Extension 2: The Uncarved Block and Original Nature\<close>

consts
  UncarvedBlock :: "entity \<Rightarrow> bool"
  Artificial :: "entity \<Rightarrow> bool"

axiomatization where
  U1_trueman_uncarved: "\<forall>u. TrueMan u \<longrightarrow> UncarvedBlock u"

axiomatization where
  U2_uncarved_not_artificial: "\<forall>x. UncarvedBlock x \<longrightarrow> \<not>Artificial x"

axiomatization where
  U3_original_uncarved: "\<forall>x. (TenThousandThings x \<and> \<not>Artificial x) \<longrightarrow> UncarvedBlock x"

theorem TU1_you_uncarved: "\<forall>u. TrueMan u \<longrightarrow> UncarvedBlock u"
  using U1_trueman_uncarved by blast

theorem TU2_you_not_artificial: "\<forall>u. TrueMan u \<longrightarrow> \<not>Artificial u"
  using U1_trueman_uncarved U2_uncarved_not_artificial by blast

section \<open>Extension 3: Emptiness and Non-Being\<close>

consts
  Empty :: "entity \<Rightarrow> bool"
  Being :: "entity \<Rightarrow> bool"

axiomatization where
  E1_formless_empty: "\<forall>x. Formless x \<longrightarrow> Empty x"

axiomatization where
  E2_form_being: "\<forall>x. HasForm x \<longrightarrow> Being x"

axiomatization where
  E3_empty_being_exclusive: "\<forall>x. Empty x \<longrightarrow> \<not>Being x"

axiomatization where
  E4_being_from_emptiness: "\<forall>x d. (Being x \<and> Dao d \<and> ArisesFr x d) \<longrightarrow> Empty d"

theorem TE1_dao_empty: "\<forall>d. Dao d \<longrightarrow> Empty d"
  using A2a_dao_formless E1_formless_empty by blast

theorem TE2_you_empty: "\<forall>u. TrueMan u \<longrightarrow> Empty u"
  by (metis TE1_dao_empty A10_trueman_is_dao A1_dao_exists)

theorem TE3_things_being: "\<forall>x. TenThousandThings x \<longrightarrow> Being x"
  using A3_things_have_form E2_form_being by blast

theorem TE4_being_from_you:
  "\<forall>u x. (TrueMan u \<and> TenThousandThings x) \<longrightarrow> (Being x \<and> Empty u \<and> ArisesFr x u)"
  by (metis TE2_you_empty TE3_things_being T9_things_arise_from_you)

(* ------------------------------------------------------------------------- *)
(* T4b: Each manifested thing both arises from and returns to the same Dao   *)
(* Uses A6 (arising), A7 (returning), and A8 (identity of Dao)               *)
(* ------------------------------------------------------------------------- *)

theorem T_arise_and_return_each_thing:
  "\<forall>x. TenThousandThings x \<longrightarrow> (\<exists>d. Dao d \<and> ArisesFr x d \<and> ReturnsTo x d)"
proof (intro allI impI)
  fix x
  assume Tx: "TenThousandThings x"

  (* From A6, obtain a Dao that x arises from *)
  have ex_arise: "\<exists>d. Dao d \<and> ArisesFr x d"
    using A6_things_arise_from_dao Tx by blast
  obtain d1 where D1: "Dao d1" and A1: "ArisesFr x d1"
    using ex_arise by blast

  (* From A7, obtain a (possibly different) Dao that x returns to *)
  have ex_return: "\<exists>d. Dao d \<and> ReturnsTo x d"
    using A7_things_return_to_dao Tx by blast
  obtain d2 where D2: "Dao d2" and R2: "ReturnsTo x d2"
    using ex_return by blast

  (* By A8, the Dao of arising and returning must be the same *)
  have eq: "d1 = d2"
    using A8_same_dao A1 R2 D1 D2 by blast

  (* Assemble the witness using d1 *)
  have "Dao d1 \<and> ArisesFr x d1 \<and> ReturnsTo x d1"
    using D1 A1 R2 eq by simp
  thus "\<exists>d. Dao d \<and> ArisesFr x d \<and> ReturnsTo x d"
    by blast
qed

lemma T_arise_and_return_each_thing_Check:
  "\<forall>x. TenThousandThings x \<longrightarrow> (\<exists>d. Dao d \<and> ArisesFr x d \<and> ReturnsTo x d)"
  nitpick [expect = none]
  oops


section \<open>Ultimate Non-Dual Realization\<close>

theorem Complete_Daoist_NonDuality:
  "\<exists>!u. TrueMan u \<and> Dao u \<and> Formless u \<and> Nameless u \<and>
        Empty u \<and> UncarvedBlock u \<and> \<not>TenThousandThings u \<and> \<not>Artificial u \<and>
        (\<forall>x. TenThousandThings x \<longrightarrow> (ArisesFr x u \<and> ReturnsTo x u \<and> Spontaneous x))"
proof -
  obtain u where u_props: "TrueMan u" using A9_you_are_trueman by blast
  obtain d where d_props: "Dao d" using A1_dao_exists by blast
  have u_is_d: "u = d" using u_props d_props A10_trueman_is_dao by blast
  have "Dao u" using u_is_d d_props by simp
  moreover have "Formless u" using u_props T6_you_formless by blast
  moreover have "Nameless u" using u_props T7_you_nameless by blast
  moreover have "Empty u" using u_props TE2_you_empty by blast
  moreover have "UncarvedBlock u" using u_props TU1_you_uncarved by blast
  moreover have "\<not>TenThousandThings u" using u_props T8_you_not_thing by blast
  moreover have "\<not>Artificial u" using u_props TU2_you_not_artificial by blast
  moreover have "\<forall>x. TenThousandThings x \<longrightarrow> (ArisesFr x u \<and> ReturnsTo x u \<and> Spontaneous x)"
    using u_props T9_things_arise_from_you T10_things_return_to_you S1_spontaneous_arising by blast
  ultimately show ?thesis using u_props A9_you_are_trueman by blast
qed

(* ------------------------------------------------------------------------- *)
(* Nitpick model checks                                                      *)
(* ------------------------------------------------------------------------- *)

section \<open>Nitpick model checks\<close>

(* With NV1, there must be at least one manifested thing and (by A5) it is
   distinct from Dao, so |entity| \<ge> 2. We ask Nitpick to find such a model. *)
lemma Nonvacuous_Model:
  "\<exists>x d. TenThousandThings x \<and> Dao d \<and> x \<noteq> d"
  nitpick [satisfy, expect = genuine]
  oops

(* Check arising/returning obligations are satisfyable in a concrete model. *)
lemma Arising_Return_Model:
  "\<exists>x d. TenThousandThings x \<and> Dao d \<and> ArisesFr x d \<and> ReturnsTo x d"
  nitpick [satisfy, expect = genuine]
  oops

(* Sanity: no entity can be both Empty and Being. *)
lemma No_Empty_and_Being_Same: "\<not> (\<exists>x. Empty x \<and> Being x)"
  nitpick [expect = none]
  oops

(* Theorems: look for counter-models; expect none within finite scopes. *)
lemma T2_Check: "\<forall>d. Dao d \<longrightarrow> \<not>HasForm d"
  nitpick [expect = none]
  oops

lemma T3_Check: "\<forall>x d. (TenThousandThings x \<and> Dao d) \<longrightarrow> x \<noteq> d"
  nitpick [expect = none]
  oops

lemma T4_Check:
  "\<forall>x y d1 d2. (TenThousandThings x \<and> TenThousandThings y \<and> ArisesFr x d1 \<and> ArisesFr y d2) \<longrightarrow> d1 = d2"
  nitpick [expect = none]
  oops

lemma TS2_Check: "\<forall>d x. (Dao d \<and> TenThousandThings x) \<longrightarrow> \<not>Caused d x"
  nitpick [expect = none]
  oops

lemma TE1_Check: "\<forall>d. Dao d \<longrightarrow> Empty d"
  nitpick [expect = none]
  oops

end
