# Completion time against latency violations

    python -m agent.frontier run --workload-csv data_pipeline/profiles/<profile>.csv --label <A|B|C|D> --kind dp|rule|controller --out points [--worker i --workers n]
    python -m agent.frontier analyse --dir points --out .

`points/points_<population>_<kind>_<index>.csv`: executed (hours, violation fraction) per scenario for one price `mu` (kind dp,
31 values from 1e-2 to 1e4) or one `progress_value` (kinds rule and controller, 12 values from 1e-4 to 1e-1).
`frontier_by_scenario.csv`, `frontier_summary.csv`, `frontier_area_0_to_0.1.csv`, `frontiers.png`. Reading: `agent/FRONTIER_DECISION.md`.
