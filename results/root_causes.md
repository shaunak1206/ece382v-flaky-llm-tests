# Root causes of every flaky test (manual, October 1, 2026)

Categories follow Luo et al. FSE 2014. "Exposed by" is the rerun mode(s) where the test both passed and failed.

| # | Suite | Test | Fails | Exposed by | Category | What happens |
|---|---|---|---|---|---|---|
| 1 | Human filelock | test_singleton_locks_are_deleted_when_no_external_references_exist[UnixFileLock] | 13/70 | both, tz, stress | Test order dependency | Passes 20/20 alone. Fails whenever test_singleton_locks_must_be_initialized_with_the_same_args runs first, because that test leaves its lock alive in the class level WeakValueDictionary registry. |
| 2 | Human filelock | same test [SoftFileLock] | 10/70 | both, tz, stress | Test order dependency | Same polluter and same registry. |
| 3 | Human arrow | TestLocaleValidation::test_locale_name_validation | 42/70 | both, tz, stress | Test order dependency | Passes 10/10 alone. Fails when TestCustomLocale::test_custom_locale_subclass runs first, because that test registers CustomLocale classes in the global locale registry and never removes them. |
| 4 | Human arrow | TestLocaleValidation::test_locale_validation | 42/70 | both, tz, stress | Test order dependency | Same polluter, the leftover CustomLocale1 has no day names. |
| 5 | Human pybreaker | CircuitBreakerThreadsTestCase::test_half_open_thread_safety | 1/70 | stress only | Concurrency | Several threads race to move the breaker from open to half open and the listener sees two transitions instead of one. Only shows up when every core is busy. |
| 6 | Human pybreaker | CircuitBreakerRedisConcurrencyTestCase::test_half_open_thread_safety | 2/70 | stress only | Concurrency | Same race through the Redis storage backend (fakeredis). |
| 7 | One shot Haiku tenacity | TestWaitRandomExponential::test_wait_random_exponential_in_timing_test | 7/70 | both, tz, stress | Randomness + time | wait_random_exponential uses full jitter, so two retries can sleep less than 10 ms in total, and the test asserts elapsed >= 0.01 s. Lowest failing value seen was 9.96 ms. Passed the single run filter. |
| 8 | One shot Haiku tenacity | TestStopBeforeDelay::test_stop_before_delay_prevents_exceeding_limit | 1/70 | stress only | Async wait / time | Asserts the whole retry loop finishes within 0.12 s of real time. Under load it took 0.17 s. Passed the single run filter. |
| 9 | One shot Haiku schedule | TestUntilConfiguration::test_until_with_string_time_format | 2/70 | tz only | Time (time of day) | Builds until() from now + 2 hours formatted as HH:MM:SS. When local time is after 22:00 that clock time is earlier today, so schedule raises "Cannot schedule a job to run until a time in the past". Both failures were in America/St_Johns where it was 22:53. Fails for any developer running the suite between 22:00 and midnight. Passed the single run filter. |

Summary
- Human tests: 4 of 6 flaky tests are order dependent through global registries, 2 are a real thread race only visible under CPU stress.
- LLM tests: all 3 flaky tests are about real time (random jitter vs a timing assert, a wall clock deadline under load, and time of day). None are order dependent. All 3 came from the one shot Haiku condition and all 3 passed the single run filter, so a TestPilot style pipeline would have kept them.
- Agentic Sonnet and one shot Sonnet produced 0 flaky tests across 1,511 and 1,406 tests.

Also observed, not counted as flaky
- One shot Haiku schedule test_zero_interval hangs forever (every(0) never finds a next run). It always fails, so the filter would reject it, but under random order the job it leaves in the global default scheduler can hang later tests. This made one run of that suite take about 5 minutes before we lowered the timeout.
