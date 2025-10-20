# The Unique Ontic Substrate **Ω**

**A machine-verified axiomatization of non-dual metaphysics in Isabelle/HOL, implemented as a working neural network architecture.**

[![Verification Status](https://img.shields.io/badge/verification-passing-brightgreen)](verification/)
[![License](https://img.shields.io/badge/license-CC%20BY%204.0-blue)](LICENSE.txt)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17388701.svg)](https://doi.org/10.5281/zenodo.17388701)

[Submitted to Isabelle AFP](https://www.isa-afp.org/webapp/submission/?id=2025-10-19_11-47-47_483)


[Substrate Grounded Neural Architecture](scripts/sgna.py)



---

## Refutation 

This formal system is intentionally structured to be self-consistent and closed. Every theorem follows as logical consequence of clearly stated axioms. The system has been mechanically verified using Isabelle/HOL 2025 and checked for countermodels using Nitpick across domain cardinalities 1 through 5. No contradictions were found and no countermodels exist within the finite scopes tested.

Nevertheless, in principle the theory could be refuted through several paths. This document examines each potential path and explains why refutation is unlikely to succeed.

[Refutation Guide](docs/refutation_guide.md)

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
  title = {The Unique Ontic Substrate},
  year = {2025},
  doi = {10.5281/zenodo.17388701},
  url = {https://github.com/matthew-scherf/The-Unique-Ontic-Substrate},
  note = {Isabelle/HOL formalization, verified October 2025}
}
```

## License

The formalization (`.thy` files) is released under BSD-3-Clause license. Documentation is released under Creative Commons Attribution 4.0 International (CC BY 4.0). See LICENSE.txt for complete terms.

---

**Ω**

*The unique ontic substrate*

Empirically grounded. Logically verified. Scientifically rigorous.

---

∃!s. Substrate s ∧ ∀p. Phenomenon p → Presents p s

**All phenomena present one substrate**

Four formalizations. Four frameworks. One structure.

---
