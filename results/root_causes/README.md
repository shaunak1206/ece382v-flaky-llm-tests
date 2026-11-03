# Root causes of the flaky tests found in November (Progress Report 2)

One record per flaky test in `labels.json`. Categories follow Luo et al. (FSE 2014) and Gruber et al. (ICST 2021).
The guard prompt test in data-partitioner is parametrized; its three flaky cases share one record. `E2 guard v3` is the
third guard prompt version. The replication batch records (added November 3) also have `subcategory`, `polluters`
(from `results/od_diagnose/` and the labeler's own pair runs) and `reproduction` (what was run, with counts). The
`other` category holds the kevinarpe-rambutan3 tests, whose code under test has an unstable `__hash__`.

| Study | Kept by the one run filter | Category | Records |
|---|---|---|---|
| E1 | yes | floating point | 2 |
| E1 | yes | time | 2 |
| E2 guard | no | randomness | 3 |
| E2 guard | yes | randomness | 7 |
| E2 guard v3 | yes | randomness | 1 |
| E2 plain | no | other | 1 |
| E2 plain | no | randomness | 10 |
| E2 plain | no | test order dependency | 3 |
| E2 plain | yes | other | 6 |
| E2 plain | yes | randomness | 15 |
| E2 plain | yes | test order dependency | 24 |
| E2 plain | yes | time | 4 |

Fields: `study`, `kept_by_one_run_filter`, `flips_only_after_rejected_polluter` (true for the two pylabeledrf
victims, which are not rejected themselves but flip only after a test the filter rejects, so they never flip in
a filtered suite and are counted as removed), `project`, `suite` (condition or suite id), `test`, `category`,
`order_dependent`, `same_source_as_human_flaky_test` (E2 only: the generated test flips for the same reason as
the known flaky human test of that project), `mechanism`, `evidence` (quoted lines), `confidence`,
`second_check_agrees`, `second_check_note`. The E1 rows include one human test (pybreaker). The 10 PL-G4
humanize tests and the two Progress Report 1 OS-H tests are explained in `results/why_analysis.md`.

How the labels were made: Manas Pathak labeled each test from the test, the code under test, and a reproduction
of the failure, and each label was checked a second time against the code (`second_check_note`). Every label
held up.
