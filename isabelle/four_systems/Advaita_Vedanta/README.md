# Advaita_Vedanta (AFP submission candidate)

A machine-checked formalization of a non-dual Advaita Vedānta axiom system in Isabelle/HOL.
We derive the main result `Tat_Tvam_Asi_Complete`, stating the uniqueness of `u` such that
`You u` and `Absolute u`, with the expected non-phenomenal and nirguṇa properties, and the
appearance relation to all conditioned entities.

DOI: https://doi.org/10.5281/zenodo.17369575 
Git: https://github.com/matthew-scherf/Only-One 

**Build** (with Isabelle2025 or current release):

```sh
isabelle build -D .
```

**Nitpick regimen (finite-scope countermodel search):**
- Global defaults: `nitpick_params [user_axioms = true, format = 3, show_all, timeout = 60, max_threads = 2, card = 1,2,3,4,5]`
- Per-goal checks: `nitpick [user_axioms = true, card = 1,2,3,4,5]` before proofs of key lemmas/theorems.
- Result: No counterexamples found within these scopes on our machine.

**Licensing:**
- Code: BSD-3-Clause
- Documentation: CC BY 4.0

See headers in the theory and `document/intro.tex` for details.