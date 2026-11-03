# Are LLM generated unit tests flaky?

ECE 382V (Fall 2026) course project, Shaunak Kapur & Manas Pathak. We generate pytest suites with LLMs,
rerun them many times under controlled nondeterminism (test order, hash seed, time zone, machine load), and
compare their flakiness with developer written tests, both on 12 libraries we chose and on code that is
already known to cause flaky tests (the Gruber et al. ICST 2021 dataset).

- `REPORT.md` the running write up, `report1/` and `report2/` the progress reports (LaTeX and PDF)
- `docs/DATA.md` what every result file contains and how to read it (start here for the data)
- `prompts/README.md` every generation prompt, verbatim, and every deviation during generation
- `results/root_causes/` the root cause label and evidence for every flaky test found in November
- `SUBJECTS.txt` exact commits of the 12 libraries under test

## Layout

| Path | Content |
|---|---|
| `llm_tests/`, `gen_oneshot_sonnet/`, `gen_oneshot_haiku/`, `gen_tp_sonnet/`, `gen_plain_sonnet/`, `gen_plain_gpt54mini/`, `gen_plain_gpt41mini/` | generated suites for the 12 libraries, one folder per condition (codes in `docs/DATA.md`) |
| `matched/<project>/<model>_<prompt>_<k>/test_gen.py` | generated suites for the matched study, `matched/targets.json` lists the code under test per project |
| `mitigations/` | the M1 (seeding) and M2 (seeding plus state restore) `conftest.py` fixtures |
| `scripts/` | everything that generates, runs, and analyzes (table below) |
| `results/` | raw results and tables (see `docs/DATA.md`) |
| `.github/workflows/` | the experiments that run on GitHub Actions |

## Setup

Python 3.12 for the 12 library study, Python 3.8 (installed automatically by `uv`) for the matched study.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r colab/requirements-mixed.txt pytest-cov
for lib in cachetools schedule tinydb humanize toolz tabulate tenacity filelock boltons more_itertools arrow pybreaker; do
  bash scripts/fetch_subject.sh $lib        # clones the pinned commit into subjects/ and installs it
done
brew install stress-ng uv                   # or apt-get install stress-ng; pip install uv
```

## Running the experiments

| What | Command | Output |
|---|---|---|
| Rerun one suite in one mode | `python scripts/rerun.py --name X --cwd <empty dir> --target <suite dir> --runs 30 --mode both --out out.json -- --timeout=15` | rerun JSON |
| One run generate and filter step | `python scripts/filter_oneshot.py <cond> <lib>` | `results/filter/<cond>/<lib>.json` |
| All three laptop modes for one generated suite | `bash scripts/llm_suite_reruns.sh <cond> <lib>` | `results/runs/<cond>/<lib>.{both,tz,stress}.json` |
| Large rerun campaign (GitHub Actions) | `gh workflow run scale-reruns` (`-f conds=...` to pick conditions), or one suite locally with `bash scripts/scale_reruns.sh <cond> <lib> 300 150 150`; tables with `python scripts/analyze_scale.py > results/scale_analysis.md` | `results/runs_scale/`, `results/scale_summary.json` |
| Generated tests inside the human suite | `python scripts/mixed.py <cond> <lib> --runs 30 --out out.json`, or `gh workflow run mixed-suites` | `results/mixed/` |
| 1,000 reruns of scanner flagged tests | `gh workflow run extended-rerun -f count=100` (`-f stress=true` for stress-ng) | `results/extended_gha*/` |
| Reproduce a known flaky test from Gruber et al. | `python scripts/gruber_reproduce.py <Project_Name> --runs 150 --out out.json`, or `gh workflow run gruber-reproduce` | `results/gruber/repro/` |
| Pick the replication batch from the unlabeled data | `python scripts/select_ext.py` | `results/gruber/candidates_ext.csv` |
| Find the code under test of a known flaky test | `python scripts/matched_targets.py <Project_Name> --repo <checkout>` | JSON on stdout |
| Generate OpenAI suites | `python scripts/gen_openai.py matched <Project_Name> gpt-5.4-mini 1` or `... library <lib> gpt-4.1-mini 1` | `matched/...`, `gen_plain_<model>/...` |
| Run matched suites next to the known flaky test | `python scripts/matched_run.py <Project_Name> --runs 200 --out out.json` (`--suites '*_guard_*'` for one prompt), or `gh workflow run matched-study -f runs=200` | `results/matched/` |
| Add the M1 and M2 fixtures to every plain matched suite | `python scripts/make_seed_suites.py` | `matched/*/*_plainseed_*/`, `matched/*/*_plainiso_*/` |
| Rerun the fixture suites | `gh workflow run matched-study -f runs=200 -f suites='*_plainseed_*'` (M1) or `-f suites='*_plainiso_*'` (M2) | `results/matched_mitig/`, `results/matched_iso/` |
| Audit the Claude E2 generation transcripts for test file reads | `python3 -I scripts/audit_generation_reads.py --json report.json` (needs the local Claude Code transcripts) | `results/generation_audit.json` |
| Alone reruns and polluter search for flaky matched tests | `python scripts/od_diagnose.py <Project_Name> --out out.json`, or `gh workflow run od-diagnose -f projects=a,b` | `results/od_diagnose/` |
| Matched study model comparisons, per suite and per project | `python scripts/matched_stats.py` | `results/matched_stats.json` |
| Expected catch of a longer filter | `python scripts/krun_estimate.py` | `results/krun_estimate.json` |
| Static risk scan | `python scripts/run_smells.py > results/smells.md` | `results/smells.json` |
| Scanner check against Gruber et al. labels | `python scripts/validate_scanner.py` | `results/scanner_validation/validation.json` |
| Test characteristics (why LLM tests rarely flip) | `python scripts/characterize.py` | `results/why_analysis.md`, `results/characterize.json` |
| Reset fixture check for the PL-G4 humanize tests | `python scripts/humanize_reset_check.py --runs 50` | `results/humanize_reset_check.json` |
| Count how each model's E2 plain suites seed, monkeypatch, and reload or touch `sys.modules` | `python scripts/matched_habits.py` | `results/matched_habits.json` |
| Coverage | `python scripts/coverage.py` (`COV_CONDS=...` to pick conditions) | `results/coverage/summary.json` |
| Study tables | `python scripts/analyze.py > results/analysis.md`, `python scripts/analyze_mixed.py`, `python scripts/analyze_matched.py` | `results/summary.json`, `results/mixed_summary.json`, `results/matched_summary.json` |
| Subject selection funnel | `python scripts/select_subjects.py --top 5000 --static` then `--run-sample 6` | `results/subject_funnel/` |

Claude suites were generated by Claude Code subagents (no API key), with the prompts in `prompts/README.md`.
OpenAI suites were generated through the API by `scripts/gen_openai.py`; every call is logged in
`results/openai_calls.jsonl` (token counts only).
