# Data collected so far and how to read it

All results live under `results/` in this repository. Everything is plain JSON, CSV, or Markdown.
Condition names used everywhere:

| Code | Directory | Who wrote the tests | Prompt | Feedback |
|---|---|---|---|---|
| Human | the library's own `tests/` | the library developers | none | none |
| AG-S | `llm_tests/<lib>` | Claude Sonnet 5 as an agent | feature prompt | runs its suite and fixes or deletes failing tests, up to 3 rounds |
| OS-S | `gen_oneshot_sonnet/<lib>` | Claude Sonnet 5.5, one shot | feature prompt | none, never runs the tests |
| OS-H | `gen_oneshot_haiku/<lib>` | Claude Haiku 4.5, one shot | feature prompt | none |
| TP-S | `gen_tp_sonnet/<lib>` | Claude Sonnet 5.5, one shot | TestPilot style (API docs only) | none |
| PL-S | `gen_plain_sonnet/<lib>` | Claude Sonnet 5.5, one shot | plain "write pytest tests" | none |
| PL-G5 | `gen_plain_gpt54mini/<lib>` | GPT 5.4 mini (OpenAI API, reasoning effort low), one shot | plain, source pasted into the prompt | none |
| PL-G4 | `gen_plain_gpt41mini/<lib>` | GPT 4.1 mini (OpenAI API), one shot | plain, source pasted into the prompt | none |

Claude suites were written by Claude Code subagents that read the source files themselves and were told not to
open any test file. OpenAI suites were generated through the API with the source pasted into the prompt, so they
cannot see test files. A *test* in the result tables is one pytest test case (each parametrized case counts); a
*test function* in `results/why_analysis.md` is one function in the source.

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

## Large rerun campaign tables: `results/scale_summary.json`, `results/scale_analysis.md`

Written by `scripts/analyze_scale.py` from `results/runs_scale/`. Generated suites ran 300 times in `both` mode
and 150 times each in `tz` and `stressng` (600 runs per test), human suites 100, 50, 50 (200 runs per test).
All modes of one suite are pooled, which is safe because every run was on the same runner type.
`pooled[cond]` has `suites`, `tests` (generated tests only if the one run filter kept them), `flaky`,
`always_fail`, and `ci` (95% Wilson interval in percent). `flaky[]` has `cond`, `proj`, `test`, `pass`, `fail`
(fail plus error), `rate` (fail over runs), `modes` (the modes in which the test both passed and failed), and
`runs` (runs per mode). `executions` is runs times tests per condition.

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
| `nod_projects.csv` | projects with a flaky test that is not order dependent (`Project_Name, n, url, sha, tests, fail_rate_med`) |
| `candidates_ext.csv` | the replication batch: one unlabeled flaky test per remaining project, same columns as `candidates.csv` with `flakiness category` = `unlabeled`; rebuilt by `scripts/select_ext.py`, whose docstring gives the rule |
| `repro_ext/<project>.json` | `scripts/gruber_reproduce.py` output for `candidates_ext.csv` |
| `matched_projects.txt` | the 41 E2 projects, space separated; the first 17 are the hand labeled batch, the other 24 the replication batch |
| `human_root_causes_ext.json` | root cause of the human flaky test of each replication project (`project`, `category`, `mechanism`, `evidence`, `confidence`, second check fields) |
| `data_genie_recheck.json` | every rerun of the data-genie human test, with where it ran and the failures |

## Matched study: `matched/<project>/<suite_id>/`, `results/matched/<project>.json`

Generated suites that target the same code as the known flaky human test, and their rerun results from
`scripts/matched_run.py` (fields documented at the top of that script). `matched/targets.json` names the code
under test per project. Suite ids are `<model>_<prompt>_<sample>`: model `sonnet`, `haiku`, `gpt54mini`,
`gpt41mini`, prompt `plain`, `guard`, `guard2`, `guard3` (the guard prompts in `prompts/README.md`), `plainseed` (the
plain suite plus `mitigations/seed_conftest.py` as `conftest.py`, fixture M1) or `plainiso` (plus
`mitigations/isolate_conftest.py`, fixture M2). `scripts/make_seed_suites.py` makes the last two.

Per project file: `project`, `sha`, `category` (Gruber et al. label), `install_ok`, `seconds`, `human` (the known
flaky test as a control: `runs`, `passed`, `failed`, `errors`, `reproduced`, `messages`), and `suites[suite_id]`
with `model`, `prompt`, `sample`, `runs`, `n_tests`, `first_run_failed` (ids that failed in run 1, which a one run
filter rejects), `tests[id]` = {`pass`, `fail`, `error`, `skip`, `kind`: stable, flaky, always-fail,
always-skip, `fails_file_order`, `fails_shuffled` (failures in each half of the runs, recorded from the
replication batch on; absent in older files)}, and the `flaky` and `always_fail` id lists. Even runs keep file order, odd runs use
pytest-randomly with seed 7000 plus the run index. An id of the form `::module` means the file did not import.

The other folders have the same format: `results/matched_guard/` holds the `gpt41mini_guard_*` suites (the
first mitigation trial), `results/matched_mitig/` the `*_plainseed_*` suites and `gpt41mini_guard2_*`,
`results/matched_guard3/` the `gpt41mini_guard3_*` suites, and `results/matched_iso/` the `*_plainiso_*` suites.
`scripts/analyze_matched.py` reads all of them and writes
`results/matched_summary.json` (`models` per model and prompt, `projects` with the human control, `flaky` list)
and `results/matched_analysis.md`.

## Model comparisons and filter estimate: `results/matched_stats.json`, `results/krun_estimate.json`

`scripts/matched_stats.py` writes, for `original_17`, `new_24` and `all` projects (and the same with `_seeded` and
`_isolated` for the fixture suites): `models[m]` = {`suites`, `suites_with_flaky`, `projects_with_flaky`, `kept`,
`flaky_kept`} and `pairs["a vs b"]` = {`p_suite_fisher`, `projects_only_first`, `projects_only_second`,
`p_project_sign`}. `scripts/krun_estimate.py` writes the expected share of kept flaky LLM tests that k extra
filter runs catch, in file order and shuffled (`file_order[k]`, `shuffled[k]`), plus `tests`,
`fail_under_5pct` (larger of the file order and shuffled rate under 5%), `fail_under_5pct_all_runs` (rate over all runs under 5%), and `never_fail_in_file_order`.

## Order dependence diagnosis: `results/od_diagnose/<project>.json`

`scripts/od_diagnose.py` output for every flaky generated test of an E2 project: `tests` = [{`suite`, `test`,
`alone_varied_fail` (of 20 alone runs with a new hash seed), `alone_fixed_fail` (of 10 with PYTHONHASHSEED=0),
`pairs_tried`, `polluters` (tests whose run just before it makes it fail in one of 3 tries), `verdict`: od,
nod, nod-hash, or unknown}]. Read the verdict together with the root cause label. For a test that draws from a shared
generator, an earlier test can shift the generator stream and so show up as a polluter (pelicun,
spreading_dye_sampler), and a rare failure can be missed in 30 alone runs (unknown, nod-hash with 1 failure).

## Root causes: `results/root_causes/labels.json`

One record per flaky test found in November (E1 campaign, E2 plain prompt, E2 guard prompts), with category
(Luo et al. and Gruber et al. categories), order dependence, whether it flips for the same reason as the known
flaky human test, the mechanism, quoted evidence, and a second independent check. Field list and counts in
`results/root_causes/README.md`.

## Test characteristics: `results/why_analysis.md`, `results/characterize.json`

`scripts/characterize.py` scans every test function for shared state, isolation helpers, clock and sleep use,
timing bounds, randomness, and concurrency. The field table is at the top of `results/why_analysis.md`.
`characterize.json` is about 14 MB and is not committed; rerun the script to rebuild it (about 2 minutes).

## results/generation_audit.json

One entry per Claude E2 plain suite (`<project>/<model>_plain_<k>`): `models` (model ids in the generating transcript),
`shell_commands` (number of shell commands the subagent ran), and `test_files_opened` (any project test file the
subagent opened or searched, with the tool; empty for 163 of 164 suites, and the one entry is a `grep -rl` that
printed file names only). Made by `scripts/audit_generation_reads.py` from the Claude Code transcripts.

## results/matched_habits.json

From `scripts/matched_habits.py`. Keys `all` (41 projects) and `first_17` map each model to `suites`
(plain suites generated, including any that did not import), `seeds` (suites that seed random, NumPy, or PyTorch),
`monkeypatch` (suites that use `monkeypatch` or `mock.patch`), and `reloads` (suites that call `importlib.reload`
or read or edit `sys.modules`).

## results/humanize_reset_check.json

From `scripts/humanize_reset_check.py`. `as_generated` and `with_reset_fixture` each give `runs` (shuffled runs),
`tests`, and `flaky` (the PL-G4 humanize tests that flipped), with and without a fixture that calls
`humanize.i18n.deactivate()` around every test.

