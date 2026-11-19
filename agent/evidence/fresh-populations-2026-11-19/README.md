# Fresh populations C and D: the single pre-registered evaluation

Pre-registration: `agent/PREREGISTRATION_2026-11-19.md`, commit e32dcfd2a7f0be7a2994d14b68d62e632f62d2a9.

    python -m agent.evaluate_predictive --workload-csv data_pipeline/profiles/profile_diurnal_long_popC.csv --seeds 20 --eval-fleet \
        --hazard-error-sigma 0.25 --q-model agent/evidence/residual-q-2026-11-14/training/main/seed0 .../seed1 .../seed2 --out popC
    (and the same with profile_diurnal_long_popD.csv, out popD)
    python -m agent.preregistered_verdict popC/episodes.csv popD/episodes.csv --out verdict.md

`popC/`, `popD/`: `episodes.csv`, `gaps.csv`, `summary.md`. `verdict.md`: the verdict against the registered criterion.
Profiles (sha256 in `data_pipeline/manifest.json` and `PREPROCESSING.md` B5c): `profile_diurnal_long_popC.csv`,
`profile_diurnal_long_popD.csv`. Reading: `agent/FRESH_POPULATIONS_DECISION.md`.
