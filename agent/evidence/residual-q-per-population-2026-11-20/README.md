# Residual Q retrained per population (post-registration comparator)

    python -m data_pipeline.tencent.split_long_profile --source data_pipeline/profiles/profile_diurnal_long_pop<C|D>.csv
    python -m agent.residual_q --seeds 0 1 2 --workload-csv data_pipeline/profiles/profile_diurnal_long_pop<C|D>_train.csv --hazard-error-sigma 0.25 --out runs/rq_pop<C|D>
    python -m agent.evaluate_residual_q_per_population --holdout-csv data_pipeline/profiles/profile_diurnal_long_pop<C|D>_holdout.csv \
        --retrained training/pop<C|D> --zero-shot agent/evidence/residual-q-2026-11-14/training/main --out eval_pop<C|D>

`training/<population>/seed<k>/`: `model.pkl`, `meta.json`, `validation.csv`. `eval_pop<C|D>/`: `episodes.csv`, `gaps.csv`, `summary.md`.
Post-registration: does not alter the verdict of `agent/PREREGISTRATION_2026-11-19.md`. Reading: `agent/RESIDUAL_Q_PER_POPULATION_DECISION.md`.
