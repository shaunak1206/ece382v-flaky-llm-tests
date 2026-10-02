## Table A. Tests and flakiness per suite (all modes pooled)

| Subject | Human tests | Human flaky | AG-S tests | AG-S flaky | OS-S tests | OS-S flaky | OS-H tests | OS-H flaky |
|---|---|---|---|---|---|---|---|---|
| cachetools | 333 | 0 | 151 | 0 | 87 | 0 | 95 | 0 |
| schedule | 81 | 0 | 113 | 0 | 124 | 0 | 179 | 1 |
| tinydb | 222 | 0 | 109 | 0 | 97 | 0 | 121 | 0 |
| humanize | 783 | 0 | 135 | 0 | 217 | 0 | 122 | 0 |
| toolz | 186 | 0 | 90 | 0 | 77 | 0 | 243 | 0 |
| tabulate | 366 | 0 | 100 | 0 | 130 | 0 | 136 | 0 |
| tenacity | 184 | 0 | 77 | 0 | 86 | 0 | 90 | 2 |
| filelock | 155 | 2 | 140 | 0 | 129 | 0 | 90 | 0 |
| boltons | 525 | 0 | 270 | 0 | 88 | 0 | 134 | 0 |
| more_itertools | 767 | 0 | 76 | 0 | 76 | 0 | 95 | 0 |
| arrow | 1904 | 2 | 169 | 0 | 221 | 0 | 154 | 0 |
| pybreaker | 81 | 2 | 81 | 0 | 74 | 0 | 136 | 0 |

## Table B. Pooled flaky rate by condition (one-shot: accepted tests only)

| Condition | Suites | Tests | Flaky | Flaky % | 95% CI | Always fail |
|---|---|---|---|---|---|---|
| Human | 12 | 5587 | 6 | 0.11 | 0.05 to 0.23 | 3 |
| AG-S | 12 | 1511 | 0 | 0.00 | 0.00 to 0.25 | 0 |
| OS-S | 12 | 1406 | 0 | 0.00 | 0.00 to 0.27 | 0 |
| OS-H | 12 | 1595 | 3 | 0.19 | 0.06 to 0.55 | 0 |

## Table C. Single-run generate-and-filter on one-shot suites

| Condition | Collected tests | Files that do not import (test functions lost) | Accepted | Rejected | Rejected but flaky | Accepted but flaky | Accepted then always fail |
|---|---|---|---|---|---|---|---|
| OS-S | 1406 | 0 (0) | 1406 | 0 | 0 | 0 | 0 |
| OS-H | 1696 | 2 (54) | 1595 | 101 | 0 | 3 | 0 |

## Table D. Flaky tests and the modes in which they flipped

- OS-H / schedule / `gen_oneshot_haiku.schedule.test_osh_job_configuration.TestUntilConfiguration::test_until_with_string_time_format` pass=68 fail=2 flipped in ['tz'] filter=accepted :: schedule.ScheduleValueError: Cannot schedule a job to run until a time in the past
- OS-H / tenacity / `gen_oneshot_haiku.tenacity.test_osh_stop_strategies.TestStopBeforeDelay::test_stop_before_delay_prevents_exceeding_limit` pass=69 fail=1 flipped in ['stress'] filter=accepted :: assert 0.17367887496948242 <= 0.12
- OS-H / tenacity / `gen_oneshot_haiku.tenacity.test_osh_wait_strategies.TestWaitRandomExponential::test_wait_random_exponential_in_timing_test` pass=63 fail=7 flipped in ['both', 'tz', 'stress'] filter=accepted :: assert 0.009955167770385742 >= 0.01
- Human / arrow / `tests.test_locales.TestLocaleValidation::test_locale_name_validation` pass=28 fail=42 flipped in ['both', 'tz', 'stress'] filter=None :: AssertionError: assert None
 +  where None = <function match at 0x1008a9b20>('^[a-z]{2}(-[a-z]{2})?(?:-latn|-cyrl)?$', 'foo')
 +    where <f
- Human / arrow / `tests.test_locales.TestLocaleValidation::test_locale_validation` pass=28 fail=42 flipped in ['both', 'tz', 'stress'] filter=None :: AssertionError: assert 0 == 8
 +  where 0 = len([])
 +    where [] = <class 'tests.test_locales.TestCustomLocale.test_custom_locale_subclass
- Human / filelock / `tests.test_filelock::test_singleton_locks_are_deleted_when_no_external_references_exist[UnixFileLock]` pass=57 fail=13 flipped in ['both', 'tz', 'stress'] filter=None :: AssertionError: assert <WeakValueDic...t 0x106706870> == {'/private/va... 0x107bb06e0>}
  
  Omitting 1 identical items, use -vv to show
  L
- Human / filelock / `tests.test_filelock::test_singleton_locks_are_deleted_when_no_external_references_exist[SoftFileLock]` pass=60 fail=10 flipped in ['both', 'tz', 'stress'] filter=None :: AssertionError: assert <WeakValueDic...t 0x10414c3e0> == {'/private/va... 0x1045a7710>}
  
  Omitting 1 identical items, use -vv to show
  L
- Human / pybreaker / `tests.pybreaker_test.CircuitBreakerRedisConcurrencyTestCase::test_half_open_thread_safety` pass=68 fail=2 flipped in ['stress'] filter=None :: assert 2 == 1
 +  where 2 = <tests.pybreaker_test.CircuitBreakerRedisConcurrencyTestCase.test_half_open_thread_safety.<locals>.StateListener
- Human / pybreaker / `tests.pybreaker_test.CircuitBreakerThreadsTestCase::test_half_open_thread_safety` pass=69 fail=1 flipped in ['stress'] filter=None :: assert 2 == 1
 +  where 2 = <tests.pybreaker_test.CircuitBreakerThreadsTestCase.test_half_open_thread_safety.<locals>.StateListener object a
