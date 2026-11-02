# Root causes of the flaky tests found in November (Progress Report 2)

One record per flaky test in `labels.json`. Categories follow Luo et al. (FSE 2014) and Gruber et al. (ICST 2021).
Each label was checked a second time independently (`second_check_agrees`, all true).
The guard prompt test in data-partitioner is parametrized; its three flaky cases share one record.

| Study | Kept by the one run filter | Category | Tests |
|---|---|---|---|
| E1 | yes | floating point | 2 |
| E1 | yes | time | 2 |
| E2 guard | no | randomness | 3 |
| E2 guard | yes | randomness | 7 |
| E2 plain | no | randomness | 10 |
| E2 plain | yes | randomness | 13 |
| E2 plain | yes | test order dependency | 2 |

Fields: `study`, `kept_by_one_run_filter`, `project`, `suite` (condition or suite id), `test`, `category`,
`order_dependent`, `same_source_as_human_flaky_test` (E2 only: the generated test flips for the same reason as
the known flaky human test of that project), `mechanism`, `evidence` (quoted lines), `confidence`,
`second_check_agrees`, `second_check_note`. The 10 PL-G4 humanize tests and the PR1 tests are labeled in the
report and in `results/why_analysis.md` rather than here.
