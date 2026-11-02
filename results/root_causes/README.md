# Root causes of the flaky tests found in November (Progress Report 2)

One record per flaky test in `labels.json`. Categories follow Luo et al. (FSE 2014) and Gruber et al. (ICST 2021).
The guard prompt test in data-partitioner is parametrized; its three flaky cases share one record.

| Study | Kept by the one run filter | Category | Records |
|---|---|---|---|
| E1 | yes | floating point | 2 |
| E1 | yes | time | 2 |
| E2 guard | no | randomness | 3 |
| E2 guard | yes | randomness | 7 |
| E2 plain | no | randomness | 10 |
| E2 plain | no | test order dependency | 2 |
| E2 plain | yes | randomness | 13 |

Fields: `study`, `kept_by_one_run_filter`, `flips_only_after_rejected_polluter` (true for the two pylabeledrf
victims, which are not rejected themselves but flip only after a test the filter rejects, so they never flip in
a filtered suite and are counted as removed), `project`, `suite` (condition or suite id), `test`, `category`,
`order_dependent`, `same_source_as_human_flaky_test` (E2 only: the generated test flips for the same reason as
the known flaky human test of that project), `mechanism`, `evidence` (quoted lines), `confidence`,
`second_check_agrees`, `second_check_note`. The E1 rows include one human test (pybreaker). The 10 PL-G4
humanize tests and the two Progress Report 1 OS-H tests are explained in `results/why_analysis.md`.

How the labels were made: an LLM assistant (Claude) drafted each label from the test, the code under test, and a
reproduction of the failure, and a second, separate LLM pass checked each draft against the code
(`second_check_note`). Every draft was confirmed.
