## Large rerun campaign (2 vCPU Ubuntu runners)

| Condition | Suites | Tests | Runs per test | Flaky | Flaky % (95% CI) | Always fail |
|---|---|---|---|---|---|---|
| Human | 12 | 5587 | 200 | 5 | 0.09 (0.04 to 0.21) | 1 |
| AG-S | 12 | 1511 | 600 | 1 | 0.07 (0.01 to 0.37) | 0 |
| OS-S | 12 | 1406 | 600 | 0 | 0.00 (0.00 to 0.27) | 0 |
| OS-H | 12 | 1595 | 600 | 3 | 0.19 (0.06 to 0.55) | 0 |
| TP-S | 12 | 913 | 600 | 0 | 0.00 (0.00 to 0.42) | 0 |
| PL-S | 12 | 1191 | 600 | 0 | 0.00 (0.00 to 0.32) | 0 |
| PL-G5 | 12 | 852 | 600 | 1 | 0.12 (0.02 to 0.66) | 0 |
| PL-G4 | 12 | 900 | 600 | 10 | 1.11 (0.60 to 2.03) | 1 |

Test executions: 6,301,400

## Flaky tests in the campaign

- Human / arrow / `tests.test_locales.TestLocaleValidation::test_locale_name_validation` pass=99 fail=101 (50.5%) modes=['both', 'tz', 'stressng']
- Human / arrow / `tests.test_locales.TestLocaleValidation::test_locale_validation` pass=99 fail=101 (50.5%) modes=['both', 'tz', 'stressng']
- Human / pybreaker / `tests.pybreaker_test.CircuitBreakerRedisConcurrencyTestCase::test_half_open_thread_safety` pass=198 fail=2 (1.0%) modes=['stressng']
- Human / pybreaker / `tests.pybreaker_test.CircuitBreakerRedisTestCase::test_successful_after_timeout` pass=199 fail=1 (0.5%) modes=['stressng']
- Human / pybreaker / `tests.pybreaker_test.CircuitBreakerThreadsTestCase::test_half_open_thread_safety` pass=193 fail=7 (3.5%) modes=['stressng']
- AG-S / tenacity / `llm_tests.tenacity.test_llm_stop_wait::test_stop_after_delay_threshold[5.0-True]` pass=594 fail=6 (1.0%) modes=['both']
- OS-H / schedule / `gen_oneshot_haiku.schedule.test_osh_job_configuration.TestUntilConfiguration::test_until_with_time_object` pass=599 fail=1 (0.2%) modes=['tz']
- OS-H / schedule / `gen_oneshot_haiku.schedule.test_osh_job_configuration.TestUntilConfiguration::test_until_with_string_time_format` pass=585 fail=15 (2.5%) modes=['tz']
- OS-H / tenacity / `gen_oneshot_haiku.tenacity.test_osh_wait_strategies.TestWaitRandomExponential::test_wait_random_exponential_in_timing_test` pass=470 fail=130 (21.7%) modes=['both', 'tz', 'stressng']
- PL-G5 / tenacity / `gen_plain_gpt54mini.tenacity.test_pgpt54mini_tenacity::test_run_stop_sets_delay_since_first_attempt_and_result` pass=598 fail=2 (0.3%) modes=['both']
- PL-G4 / humanize / `gen_plain_gpt41mini.humanize.test_pgpt41mini_humanize::test_intcomma_custom_separators` pass=327 fail=273 (45.5%) modes=['both', 'tz', 'stressng']
- PL-G4 / humanize / `gen_plain_gpt41mini.humanize.test_pgpt41mini_humanize::test_thousands_and_decimal_separators[hu_HU- -,]` pass=306 fail=294 (49.0%) modes=['both', 'tz', 'stressng']
- PL-G4 / humanize / `gen_plain_gpt41mini.humanize.test_pgpt41mini_humanize::test_thousands_and_decimal_separators[fr_FR- -.]` pass=304 fail=296 (49.3%) modes=['both', 'tz', 'stressng']
- PL-G4 / humanize / `gen_plain_gpt41mini.humanize.test_pgpt41mini_humanize::test_thousands_and_decimal_separators[None-,-.]` pass=314 fail=286 (47.7%) modes=['both', 'tz', 'stressng']
- PL-G4 / humanize / `gen_plain_gpt41mini.humanize.test_pgpt41mini_humanize::test_thousands_and_decimal_separators[pt_BR-.-,]` pass=311 fail=289 (48.2%) modes=['both', 'tz', 'stressng']
- PL-G4 / humanize / `gen_plain_gpt41mini.humanize.test_pgpt41mini_humanize::test_thousands_and_decimal_separators[en_US-,-.]` pass=320 fail=280 (46.7%) modes=['both', 'tz', 'stressng']
- PL-G4 / humanize / `gen_plain_gpt41mini.humanize.test_pgpt41mini_humanize::test_thousands_and_decimal_separators[it_IT-.-,]` pass=319 fail=281 (46.8%) modes=['both', 'tz', 'stressng']
- PL-G4 / humanize / `gen_plain_gpt41mini.humanize.test_pgpt41mini_humanize::test_thousands_and_decimal_separators[lv- -,]` pass=303 fail=297 (49.5%) modes=['both', 'tz', 'stressng']
- PL-G4 / humanize / `gen_plain_gpt41mini.humanize.test_pgpt41mini_humanize::test_thousands_and_decimal_separators[de_DE-.-,]` pass=313 fail=287 (47.8%) modes=['both', 'tz', 'stressng']
- PL-G4 / humanize / `gen_plain_gpt41mini.humanize.test_pgpt41mini_humanize::test_gettext_functions_basic` pass=313 fail=287 (47.8%) modes=['both', 'tz', 'stressng']
