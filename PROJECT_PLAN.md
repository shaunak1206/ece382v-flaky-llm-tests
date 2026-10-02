# Are LLM-Generated Unit Tests Flaky? An Empirical Study for ECE 382V

Shaunak Kapur — ECE 382V (Software Testing in the Era of Nondeterminism), Fall 2026

## Initial project idea (for the Oct 5 submission)

Developers increasingly let large language models (LLMs) write their unit tests, through tools such as
GitHub Copilot, ChatGPT, and research systems like TestPilot. These pipelines typically generate a test,
run it once, and keep it if it passes. A test that passes once, however, is not necessarily a
deterministic test: it may depend on wall-clock time, test execution order, hash-seed-dependent
iteration order of sets, random number generators, or leftover filesystem and global state. Such flaky
tests erode developers' trust in their test suites, and prior work (Luo et al., FSE 2014; Lam et al.,
ICST 2019; Gruber et al., ICST 2021) has characterized them only for human-written tests. This project
asks a simple, unanswered question: **are LLM-generated tests more (or less) flaky than the human-written
tests for the same code, and why?** We will take a set of small, well-tested open-source Python
libraries, have an LLM generate a test suite for each one using a TestPilot-style generate-and-filter
pipeline, and then rerun both the LLM-generated and the original human-written suites many times under
controlled perturbations (random test order via pytest-randomly, varying `PYTHONHASHSEED`, and
repeated plain reruns). We will measure the fraction of flaky tests in each suite, manually classify
each flaky test's root cause using the Luo et al. taxonomy, and compare the distributions. As a second
contribution, we will evaluate cheap mitigations, such as prompt instructions that discourage
time- and order-dependent assertions and a rerun-based filter in the generation pipeline, and measure
how much flakiness they remove and what test coverage they cost. The expected outcome is the first
quantified comparison of flakiness between LLM-generated and human-written tests, a labeled dataset of
flaky LLM-generated tests, and practical guidance for developers and tool builders who adopt LLM test
generation.

## Research questions

- **RQ1 (Prevalence):** What fraction of LLM-generated tests are flaky, versus human-written tests for
  the same projects, under the same rerun budget and perturbations?
- **RQ2 (Root causes):** What are the root causes of flaky LLM-generated tests, and how does the
  distribution differ from human-written flaky tests (Luo et al. categories: async wait, concurrency,
  test order dependency, resource leak, network, time, IO, randomness, floating point, unordered
  collections, platform)?
- **RQ3 (Perturbation sensitivity):** Which perturbation exposes each flaky test: plain reruns, random
  order, or hash-seed changes? How many reruns are needed to reach a stable flaky set?
- **RQ4 (Mitigation):** Do prompt-level guidance and a rerun-based filter reduce flakiness, and at
  what cost in test count and coverage?

## Methodology

1. **Subjects.** 15 to 25 pure-Python libraries from PyPI/GitHub with (a) an existing pytest suite,
   (b) no external services, (c) total suite runtime under two minutes. Stratify so that some
   libraries touch time, randomness, filesystem, or caching and some are pure computation.
   Pilot subjects: cachetools, schedule, tinydb, humanize, toolz, tabulate.
2. **Test generation.** For each library, prompt an LLM with the source module(s) and ask for a
   pytest suite covering the public API (TestPilot-style prompt). Keep tests that pass on a single
   run; give the model up to three repair iterations for tests that fail on that run. Do not show
   the model the existing human tests. Record model, prompt, and generation date. Use at least two
   models (e.g. Claude Sonnet and GPT-4o class) so results are not model-specific.
3. **Rerun harness** (`scripts/rerun.py`, already written). Run each suite N times (pilot N=30,
   full study N=100) in four modes: plain, random order, hash seed, and both. Record per-test
   outcomes from JUnit XML. A test is flaky if it both passes and fails across reruns.
4. **Root-cause classification.** Manually label each flaky test (two labelers, resolve
   disagreements) using the Luo et al. taxonomy plus an "LLM-specific" bucket for causes such as
   asserting on a hallucinated API detail that only holds in some orders.
5. **Mitigation.** Regenerate with (a) a prompt that lists flakiness pitfalls and (b) a pipeline
   that reruns 5 times before accepting a test. Measure flaky rate, number of tests, and line
   coverage (pytest-cov) versus the baseline pipeline.
6. **Analysis.** Report per-project and pooled flaky rates with confidence intervals, a category
   histogram, and the rerun-count curve (fraction of flaky tests found after k reruns).

## Timeline (course deadlines)

| Date | Deliverable | Work |
|---|---|---|
| Sep 1 - Oct 5 | Initial idea (Oct 5) | Pilot on 6 subjects (done below), pick full subject list, lock prompts |
| Oct 9 | Proposal | RQs, related work (Luo 2014, Lam 2019/2020, Gruber 2021, Schäfer 2023, Parry 2021), pilot numbers |
| Oct 24 | Progress report 1 | All subjects generated with model 1; reruns N=100 in all 4 modes; RQ1 numbers |
| Nov 7 | Progress report 2 | Model 2 generation; manual root-cause labeling (RQ2); RQ3 rerun curve |
| Nov 21 | Progress report 3 | Mitigation experiments (RQ4); coverage measurements; draft figures |
| after Thanksgiving | Presentation | 15-minute talk with the pilot + full results |
| Dec 12 | Final report | Workshop-style paper (6-8 pages), dataset and scripts released |

## Papers to present (two related papers)

1. Luo, Hariri, Eloussi, Marinov. "An Empirical Analysis of Flaky Tests." FSE 2014.
   (root-cause taxonomy that this project reuses)
2. Schäfer, Nadi, Eghbali, Tip. "An Empirical Evaluation of Using Large Language Models for
   Automated Unit Test Generation." TSE 2023. (TestPilot; the generate-and-filter pipeline this
   project mirrors, and which never checks for flakiness)

Backup pair: Gruber et al., "An Empirical Study of Flaky Tests in Python," ICST 2021, with
Lam et al., "Understanding Reproducibility and Characteristics of Flaky Tests Through Test Reruns in
Java Projects," ISSRE 2020.

## Risks and mitigations

- *Few flaky tests found.* Increase reruns, add perturbations (CPU load, timezone, locale), or add
  subjects that use time/IO. Even a low rate with a clear root-cause breakdown is a publishable
  negative result.
- *LLM cost.* Small libraries and a fixed prompt keep it to a few dollars per model.
- *Manual labeling effort.* Bounded by the flaky set size; typically a few dozen tests.

## Repository layout

```
scripts/rerun.py    rerun harness (modes: plain | order | hashseed | both), writes JSON per suite
scripts/baseline.sh reruns human-written suites of all subjects
scripts/compare.py  summary table + flaky list with crude root-cause keywords
subjects/           cloned subject libraries (pinned commits)
llm_tests/<proj>/   LLM-generated suites
results/human/, results/llm/   per-suite JSON results
```

## Pilot results

(filled in below once the reruns finish)
