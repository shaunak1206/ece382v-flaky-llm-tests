# Data collected so far and how to read it

All results live under `results/` in this repository. Everything is plain JSON, CSV, or Markdown.
Condition names used everywhere:

| Code | Directory | Who wrote the tests | Prompt | Feedback |
|---|---|---|---|---|
| Human | the library's own `tests/` | the library developers | none | none |
| AG-S | `llm_tests/<lib>` | Claude Sonnet as an agent | feature prompt | runs its suite and fixes or deletes failing tests, up to 3 rounds |
| OS-S | `gen_oneshot_sonnet/<lib>` | Claude Sonnet, one shot | feature prompt | none, never runs the tests |
| OS-H | `gen_oneshot_haiku/<lib>` | Claude Haiku 4.5, one shot | feature prompt | none |
| TP-S | `gen_tp_sonnet/<lib>` | Claude Sonnet, one shot | TestPilot style (API docs only) | none |
| PL-S | `gen_plain_sonnet/<lib>` | Claude Sonnet, one shot | plain "write pytest tests" | none |

The **feature prompt** names the time, randomness, thread, and file features of each library so the model
tests them. The exact prompt texts are in `prompts/README.md`. In the matched study (below) suite ids have
the form `<model>_<prompt>_<sample>`, for example `sonnet_plain_1`.

## Rerun results: `results/runs/<cond>/<lib>.<mode>.json`, `results/human/<lib>.<mode>.json`, `results/llm/<lib>.<mode>.json`, `results/runs_scale/<cond>/<lib>.<mode>.json`

Written by `scripts/rerun.py`. One file per suite and perturbation mode. `results/llm/` holds the first six
AG-S suites from the pilot, `results/runs_scale/` holds the November large rerun campaign (GitHub Actions).

| Field | Meaning |
|---|---|
| `name`, `mode`, `runs` | suite label, perturbation mode, number of runs |
| `num_tests` | test ids collected in the last run |
| `runs_with_failure` | runs in which at least one test failed or errored |
| `avg_run_seconds` | mean wall time of one run of the whole suite |
| `num_flaky`, `num_always_fail`, `flaky_tests` | summary of the per test kinds below |
| `tests[id].counts` | `{pass, fail, error, skip}` counts over all runs |
| `tests[id].kind` | `flaky` (passed and failed), `always-fail`, `always-skip`, or `stable` |
| `tests[id].messages` | up to 3 distinct failure messages |
| `tests[id].fail_runs` | indices of the runs in which a flaky test failed |
| `noise_per_run` | stressng mode only, the stress-ng configuration used in each run |

Modes. `both` random test order (pytest-randomly) and a new `PYTHONHASHSEED` per run. `tz` the same plus
`TZ` cycling through 10 time zones. `stress` the same plus one Python busy loop per core. `stressng` the same
plus stress-ng CPU and memory load from 10 Shaker style configurations.

Machines. `both`, `tz`, `stress` for Human, AG-S, OS-S, OS-H, and 8 TP-S suites ran on an 8 core Apple M3.
`stressng`, the PL-S suites, 4 TP-S suites, and everything in `runs_scale/` ran on 2 vCPU Ubuntu GitHub
runners. `scripts/analyze.py` only calls a test flaky if it passed and failed on the same machine.

## One run filter: `results/filter/<cond>/<lib>.json`

Written by `scripts/filter_oneshot.py`. What a generate and filter pipeline would keep after one run in file
order. `accepted_ids` passed, `rejected_ids` failed or errored, `collection_error_files` are files that did not
import, `collected`, `accepted`, `rejected` are counts.

## Study tables: `results/summary.json`, `results/analysis.md`

Written by `scripts/analyze.py`. `pooled[cond]` has `tests`, `flaky`, `always_fail`, `machine_dependent`,
`ci` (95% Wilson interval for the flaky rate, in percent) and `flaky_failure_rates` (failures over runs for each
flaky test, in percent). `flaky` lists every flaky test with its counts per machine and the modes where it flipped.

## Static risk scan: `results/smells.json`, `results/smells.md`

Written by `scripts/run_smells.py` using `scripts/smells.py`. Per suite: `n_tests`, `any` (tests with at least
one risky pattern), `counts` per pattern, and `tests` (file, test name, list of patterns). Patterns are
SLEEP, WALLCLOCK, TIMING_ASSERT, RANDOM, CONCURRENCY, FIXED_PATH, SUBPROCESS, UNORDERED.

## Scanner validation: `results/scanner_validation/validation.json`

Written by `scripts/validate_scanner.py` against the Gruber et al. hand labeled flaky tests
(`NOD_manual_classification.csv`, Zenodo 10.5281/zenodo.4450435). `flaky_8_rules` and `controls_8_rules` give
`n` and `flagged`, `per_category` breaks recall down by the labeled cause.

## Coverage: `results/coverage/summary.json`

Written by `scripts/coverage.py`. Line coverage in percent of the library under test, per library and condition.
One shot suites are measured on the tests the one run filter keeps.

## Mixed suites: `results/mixed/<cond>__<lib>.json`, `results/mixed_summary.json`

Written by `scripts/mixed.py` and `scripts/analyze_mixed.py`. The generated tests are copied into the
library's own test folder and run in the same pytest session as the human tests, 30 runs in random order.
Each file has the rerun format above. The summary lists, per condition and library, `flaky_generated`,
`flaky_human`, `human_flaky_only_when_mixed`, and `human_broken_by_mix`.

## Extended runs of flagged tests: `results/extended_gha/`, `results/extended_gha_stress/`

Written by `scripts/extended.py`. Each `<cond>__<lib>__tz<i>.json` holds one time zone chunk with
`{tz, seed, count, seconds, tests: {id: {pass, fail, msg}}}`. `summary.json` and `summary.txt` aggregate them.

## Subject funnel: `results/subject_funnel/static.json`, `s6_sample.json`

Written by `scripts/select_subjects.py`. `counts` is the number of packages left after each rule, `S5` lists
the survivors with their stratum (ND imports a nondeterminism module, D does not), `current_subjects` traces
our 12 libraries through the rules. `s6_sample.json` is the run check (passes offline in under 60 s).

## FlaPy cross check: `results/flapy/tests_overview.csv`

FlaPy's own per test overview (columns documented in the FlaPy README). `flaky?` is FlaPy's verdict.

## Known flaky Python tests (Gruber et al.): `results/gruber/`

| File | Content |
|---|---|
| `TestsOverview.csv` | the full dataset from Zenodo (not committed, 249 MB, download with the URL in `scripts/gruber_reproduce.py`) |
| `candidates.csv` | the 58 hand labeled non order dependent flaky tests whose cause is not network. Columns `Project_Name, Project_URL, Project_Hash, Test_filename, Test_classname, Test_funcname, Test_parametrization, flakiness category, fail_rate` (failures over runs in the original study), `project_tests` |
| `repro/<project>.json` | output of `scripts/gruber_reproduce.py`, fields documented at the top of that script |

## Matched study: `matched/<project>/<suite_id>/`, `results/matched/<project>.json`

Generated suites that target the same code as the known flaky human test, and their rerun results from
`scripts/matched_run.py` (fields documented at the top of that script).
