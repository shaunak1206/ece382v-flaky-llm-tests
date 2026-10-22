# Mitigation prompt (draft, October 22, 2026)

Drafted up front, as the instructor asked, instead of waiting for the final flaky test list. It targets the
patterns behind every LLM flaky test so far (real time, timing asserts, jitter) and the rules of the static
scanner, each of which maps to a root cause category of Luo et al. (see scripts/validate_scanner.py).

It is appended to the generation prompt of a condition (first the feature prompt, OS-S and OS-H, where the
risk is highest), and the result is compared with the same condition without it on: scanner risk rate,
flaky tests after the rerun modes, tests rejected by the one run filter, and line coverage
(so the prompt cannot win by just testing less).

> Write tests that give the same result on every run, on any machine, in any order, in any time zone.
> 1. Never call time.sleep or asyncio.sleep with a real delay. If code under test waits, inject or patch its sleep or clock.
> 2. Never read the real clock (now, utcnow, today, time.time, monotonic) in a test. Freeze time with freezegun, or pass a fixed datetime.
> 3. Never assert on elapsed wall clock time or on how long a call took.
> 4. Seed or patch every source of randomness, including random jitter in retry or wait strategies, or assert only on properties that hold for every random value.
> 5. Do not start threads or processes unless the behavior under test is concurrency, and then join them and assert on the final state, not on timing.
> 6. Write files only under pytest's tmp_path. Do not use fixed file paths.
> 7. Do not depend on the iteration order of sets or dict views. Compare with sets or sorted lists.
> 8. Do not change module level or global state (default schedulers, registries, locale, environment variables) without restoring it, for example with monkeypatch.
> 9. Each test must pass when run alone and in any order with the other tests.

Second mitigation (pipeline side, no prompt change): replace the one run filter with a k run filter
(k = 3 and 5) under random order, a new hash seed, and a different time zone each run, and measure how many
flaky tests it removes and how much extra run time it costs.
