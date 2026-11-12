# daily_strength of the long-window candidate pool

- window start 1594094400, volumes with the window 6469, ranked candidates 6054
- pool: median 0.0270, p90 0.1551, p99 0.7608, max 0.9965
- selected 44: min 0.8321, median 0.9310, max 0.9965; lowest selected is at percentile rank 99.29

| workload | rule | n | daily_strength | RMSE last value (MB/s) | RMSE same time yesterday (MB/s) | mean MB/s | p95 MB/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| diurnal_long | highest daily_strength (variance share of the time-of-day mean profile) | 44 | 0.9932 | 10.17 | 2.57 | 69.9 | 99.8 |
| bursty_long | highest window-local p99/median of total bandwidth | 41 | 0.0451 | 8.33 | 35.12 | 53.6 | 119.8 |
| idle_long | lowest mean total bandwidth | 599 | 0.3803 | 2.36 | 2.35 | 17.3 | 20.1 |
