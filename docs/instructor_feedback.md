# Instructor feedback on the project (received October 11, 2026)

Overall project feedback from Prof. Shi, to address in the progress reports. Not a proposal revision.

1. **Generate tests early.** Try generating tests for some planned subjects now, to prove we know how and do not get stuck on tooling later.
2. **Prompting and context.** Say whether we do anything fancy (prompting tricks, context collection) or keep generation basic.
3. **Root causing.** How will we root cause flaky tests? Will we run existing flaky test detection tools to confirm a test is flaky, and have we tried running them so we know how to use them?
4. **stress-ng.** Can simulate machine load for flaky test detection, see Shaker (Silva et al., ICSME 2020): https://damorim.github.io/publications/Silva_ETAL_ICSME20.pdf
5. **Metrics.** List every metric. Only flaky or not, or also flaky failure rates?
6. **Mitigation prompts.** Have an initial idea of the mitigation prompts now. Basing everything on eventual findings is risky, so do some preparation up front.
7. **Contingency.** What if the generated tests are never flaky? Can we still proceed, and if not, how would we shift direction?

## Where things stand (notes for us)
- 1: suites already generated for 12 libraries under three setups (prompts/README.md).
- 2: generation is basic by design (one prompt with the module source and a list of public APIs); the agentic setup adds execution feedback. A richer context variant would be new work.
- 3: root causing so far is targeted reruns (scripts/diagnose.py) plus manual Luo et al. labels. Existing tools not yet tried: FlaPy (Gruber et al., needs Docker, so run it on GitHub Actions), pytest-flakefinder (installed, untested), iDFlakies is Java only.
- 4: stress-ng 0.22.01 installed on the Mac via Homebrew (October 11). Current stress mode uses python CPU burners, so a stress-ng mode is still to add.
- 5: we already record per test pass and fail counts per mode, so flaky failure rates are available, not only a flaky yes or no.
- 6: no mitigation prompt drafted yet.
- 7: the pilot found few flips, so the contingency matters. Options include latent risk via the smell scanner, stronger perturbations (stress-ng, slower machines, FlakeRake style delays), and switching to LLM based detection or repair of known flaky tests.

# Proposal feedback (Canvas, attempt 1, October 12, 2026)

Questions from Prof. Shi on the submitted proposal, with where we stand (checked October 22).

| # | Question | Covered already? | Gap and plan |
|---|---|---|---|
| 1 | How were subjects chosen, systematically and without bias, so the set can be expanded the same way? | Partly. Criteria are written down (pure Python, own pytest suite, offline, under a minute), but the first six were hand picked and the second six were picked because they use time, randomness, threads, or files. | Write `scripts/select_subjects.py` as a funnel: top PyPI packages by downloads, then has a GitHub repo, pure Python, pytest suite passes offline in under 60 s. Log the count at every step. Then split into two strata by whether the library source imports time, datetime, random, threading, asyncio, or does file IO, and sample from both on purpose. Check the current 12 against the funnel and replace any that do not pass. |
| 2 | What prompts, are they based on prior work, are they what developers really use? | No. One custom prompt that names the time and random features of each library. Threats section already says this makes the results an upper bound. | Three prompt styles. (a) TestPilot style (Schäfer et al.): function signature, docstring, usage snippets. (b) Plain developer prompt, "write pytest unit tests for this module" plus the file, the way people use ChatGPT or Copilot chat. (c) The current feature naming prompt, kept as the stress condition. Report every result per prompt style. |
| 3 | Run generated tests with the existing ones? Individually or as a whole suite? | Partly. Each suite runs as a whole in random orders (pytest-randomly), so order dependence inside a suite can show. Flaky tests are then rerun alone by diagnose.py. Generated and human tests are never run together. | Add a mixed condition where the generated tests go into the existing test suite, which is how a developer would use them, so either side can pollute the other. Add an iDFlakies style polluter and victim search (already step 2). State clearly that alone runs are for diagnosis only. |
| 4 | How is CPU load created, why only that, what are the limits? | Weak. Python busy loops, one per core, CPU only, with no justification for the intensity. stress-ng is installed but not used. | Add a stress-ng mode with Shaker (Silva et al., ICSME 2020) style configurations (CPU and memory workers at several levels), run it on the Mac and the GitHub Actions runners. Justify from Shaker. Limits: no disk IO, network, or GC pressure, only two machine types. |
| 5 | How is the static scanner built and why these rules? | Partly. AST to find test functions, regular expressions for eight patterns. The proposal says the patterns come from prior studies but no rule cites a source and the rules are not validated. | Add a table that maps every rule to a root cause category in Luo et al. and Gruber et al. Validate precision and recall on known flaky Python tests (Gruber et al. / FlaPy dataset) as well as on our own flips. Current numbers on our data: 3 of 3 LLM flaky tests flagged, precision under 1%, 2 of 6 human flaky tests flagged (it cannot see order dependence). |
| 6 | Are human tests flaky at all? If not, what is the comparison? | Yes. 6 of 5,587 human tests flipped (4 order dependent in filelock and arrow, 2 pybreaker race under stress). | The baseline is thin, 0.11% vs 0.19% for one shot Haiku with overlapping CIs. Add per test flaky failure rates and risky pattern rates as comparisons, and consider adding subjects with known flaky human tests (picked from FlaPy, documented as a purposeful choice in the funnel). |
| 7 | What if no generated test is flaky? Does that break intermediate step 3 (mitigation)? | Partly. Not zero: 3 one shot Haiku tests flipped, and both pilot tests were LLM written. But mitigation measured only as "flaky tests kept" would have almost nothing to count. | Measure the mitigation on risky pattern rate (scanner), on failures that can be induced (FlakeRake style delays, stress-ng, slow runners), and on flaky failure rate, not only on observed flips. If flips stay near zero, the finding becomes "LLM tests carry about 2x latent timing risk that a single run filter cannot see". |
