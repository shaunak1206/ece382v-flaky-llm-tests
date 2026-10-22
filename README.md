# Are LLM generated unit tests flaky?

ECE 382V (Fall 2026) course project, Shaunak Kapur & Manas Pathak. Rerun study of LLM generated pytest suites against the human suites of 12 Python libraries.

- `REPORT.md` the write up
- `scripts/` rerun harness and analysis (`rerun.py`, `analyze.py`, `smells.py`, `extended.py`)
- `llm_tests/`, `gen_oneshot_sonnet/`, `gen_oneshot_haiku/` generated suites
- `results/` raw rerun data and tables
- `.github/workflows/extended.yml` the extended rerun experiment on GitHub Actions
- `SUBJECTS.txt` exact commits of the libraries under test
