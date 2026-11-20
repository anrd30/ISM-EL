# Objective sensitivity

    python -m agent.objective_sensitivity run --workload-csv data_pipeline/profiles/<profile>.csv --label <A|B|C|D> --out runs [--worker i --workers n]
    python -m agent.objective_sensitivity analyse --dir runs --out .

Profiles: A `profile_diurnal_long_holdout.csv`, B `profile_diurnal_long_altB.csv`, C `profile_diurnal_long_popC.csv`, D `profile_diurnal_long_popD.csv`
(sha256 in `data_pipeline/manifest.json`). `runs/episodes_<population>_<setting>.csv`: one file per population and setting
(20 scenarios; the policies, the optimum re-solved under the setting, and the fixed-0.004 variants for beta other than 0.1).
`analysis.csv`, `headline.md`, `exceptions.md`: the analysis. `dp_validation/`: grid convergence of the optimum
(`grid_convergence.py`, `grid_convergence.csv`). Reading: `agent/OBJECTIVE_SENSITIVITY_DECISION.md`, `agent/OPTIMUM_DECISION.md`.
