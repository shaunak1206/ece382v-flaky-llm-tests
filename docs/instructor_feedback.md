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
