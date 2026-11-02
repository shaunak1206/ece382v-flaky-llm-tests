# Why are the LLM tests rarely flaky? A static comparison with the human tests

Generated from `results/characterize.json`, which `scripts/characterize.py` writes (`.venv/bin/python scripts/characterize.py`, stdlib only, about 20 s). Every number below comes from that file, from `results/summary.json`, or from `results/filter/`. The unit is a test function as written in the source (the same unit as `results/smells.json`), so a parametrized function counts once.

## characterize.json fields

Top level

| Key | Meaning |
|---|---|
| `per_condition` | share (%) and count (`<feature>_count`) of test functions with each feature, per condition and pooled over the five LLM conditions. Also means and medians of size fields, timing assertion totals, upper and lower slack lists in ms, and `tests_per_suite` |
| `per_condition_macro` | the same shares averaged over the 12 libraries, so large suites (arrow, more-itertools) do not dominate |
| `per_condition_excluding_arrow` | pooled shares without arrow, whose 708 human test functions use class scoped fixtures |
| `per_condition_library` | the same aggregates for every condition x library suite |
| `suites` | test functions found, plus for one shot suites the one run filter result (`collected`, `rejected`, `collection_error_files`, `functions_in_error_files`) and `collected_in_reruns` from summary.json |
| `case_studies` | `schedule_until_args` and `schedule_at_args` (how each condition builds the time it passes to `until()` and `at()`), `tenacity_random_wait_with_timing_assert`, `tenacity_stop_delay_upper_bounds` |
| `flaky_matched` | each flaky test id from summary.json and whether it was matched to a test function |
| `tests` | one record per test function (below) |

Per test record (`tests[]`). Flags are 0 or 1 unless noted. "Via fixture" means the flag is also set when a fixture the test requests (argument, `usefixtures`, autouse in the file, class, or a conftest.py) or a setUp/tearDown method has it. Concurrency and sleep flags are also taken from helper functions the test calls (`self._start_threads`).

| Field | Meaning |
|---|---|
| `cond`, `label`, `proj`, `file`, `class`, `test`, `line` | where the test is |
| `flaky`, `flaky_ids` | flaky in the rerun study (summary.json), and the pytest ids it covers |
| `in_collection_error_file` | the file did not import in the one run filter, so the test never ran |
| `lines`, `n_asserts`, `n_raises` | function length; `assert` statements plus `self.assertX` calls; `pytest.raises`/`warns` and `assertRaises` |
| `parametrized`, `param_cases`, `in_class`, `unittest_style`, `is_async` | style |
| `fixtures`, `builtin_fixtures` | user fixtures reached (transitively) and isolation fixtures used (tmp_path, monkeypatch, ...) |
| `global_stmt` | uses `global` |
| `module_lib_object` | uses a module level library instance or a module level function decorated by the library (`@retry`, `@cached`), which carries state across tests |
| `module_data_read` | reads a module level list/dict/set literal (test data, normally read only) |
| `module_object_write` | mutates one of those module level objects |
| `class_attr_read`, `class_attr_write` | reads a mutable class attribute, writes `Class.x`, `cls.x`, `type(self).x` |
| `lib_singleton_use` | uses a library singleton or registry: schedule module functions on the default scheduler (`schedule.every`, `run_pending`, ... but not `clear`), `filelock(..., is_singleton=True)` or `_instances`, arrow `locales._locale_map` or a `Locale` subclass, `humanize.activate` (via fixture too) |
| `lib_global_reset`, `lib_global_reset_after` | calls `schedule.clear()` or `humanize.deactivate()` anywhere / after the test (after `yield` in a fixture, in tearDown, or in `finally`) |
| `lib_singleton_unreset` | `lib_singleton_use` and no reset after the test |
| `lib_global_write` | assigns to an attribute of the library module or class (module flags like `tabulate.PRESERVE_WHITESPACE`, `TinyDB.default_table_name`), `setattr` on it, or defines an arrow `Locale` subclass (which registers itself) |
| `env_mutation_unguarded` | writes `os.environ`, `sys.*`, `os.chdir`, `locale.setlocale`, `time.tzset` directly (monkeypatch calls are not counted) |
| `setup_without_teardown` | the class has setUp/setup_method/setup_class with no matching teardown, or the module has setup_module without teardown_module |
| `wide_scope_fixture` | requests a fixture with scope module, class, package, or session |
| `private_attr_access` | touches `x._name` on anything other than self/cls (via fixture too) |
| `global_state_write`, `shared_state_any` | summary: any write above; any write, library object, class attribute read, singleton, or wide scope fixture |
| `iso_tmp_path`, `iso_tempfile`, `iso_monkeypatch`, `iso_freezegun`, `iso_time_machine`, `iso_mock_patch`, `iso_injected_sleep` | isolation helpers; `iso_injected_sleep` is a tenacity `sleep=` callable that does not really sleep |
| `fresh_lib_objects`, `fresh_object` | number of library constructor calls in the test or a function scoped fixture (count, and >0) |
| `yield_fixture` | a requested fixture has teardown code |
| `clock_reads`, `clock_frozen_or_patched`, `real_clock_read` | explicit clock reads (`time.time`, `monotonic`, `perf_counter`, `datetime.now/utcnow/today`, `arrow.now/utcnow`, helpers that return one); freezegun, time_machine, or a patched clock is active; reads with no frozen clock |
| `sleep_calls`, `sleep_seconds_literal`, `real_sleep` | non zero `time.sleep`/`asyncio.sleep` calls and their literal total (times literal loop counts), 0 when sleep is patched |
| `clock_derived_string`, `clock_time_of_day_string` | `strftime` on a real clock value; with a format that has `%H` but no date part |
| `timing_asserts` | list of timing assertions, each with `direction` (upper: elapsed < T, lower: elapsed >= T), `threshold` T in seconds, `planned` delay [min, nominal, max] in seconds, `slack_ms`, `delay_method`, `window` (lines), `expr`. Slack is T - planned max for upper bounds and planned min - T for lower bounds. Planned delay comes from literal sleeps in the measured window, a simulation of the tenacity stop/wait config with the library's own rules, filelock timeouts that are expected to expire, `time_limited` limits, `run_all(delay_seconds)`, or a hand derivation listed in `MANUAL_DELAY` in the script (9 assertions) |
| `timing_assert`, `timing_asserts_on_frozen_clock` | has at least one; comparisons dropped because the clock was frozen |
| `random_calls`, `random_seeded`, `random_unseeded` | calls to `random.*` or library random APIs (random_product, sample, RRCache, wait_random, ...), and whether a seed, `random_state=`, `choice=`, or a patch is visible |
| `random_assert_exact`, `random_assert_property`, `random_unseeded_exact` | an `==`/`is` assertion of a raw random value against something not drawn from it; any other assertion on random values (ranges, membership, len/set/sorted) |
| `threads`, `processes`, `asyncio`, `subprocess`, `concurrency`, `joins`, `join_timeout`, `threads_or_procs_without_join` | concurrency, and whether threads are joined (`.join()`, executor `with` block) and with a timeout |
| `via_fixture_or_setup` | which flags came from a fixture, setUp, or helper rather than the test body |

The scan is static and heuristic. It can miss state shared through objects it cannot resolve, and it does not know which library calls read the clock internally.

## Short answer

Yes, the finding holds for these 12 libraries. 3 of 5,592 LLM test functions were flaky against 6 of 3,077 human test functions (7 flaky test ids, since one is parametrized), and the reason is visible in how the tests are built.

1. LLM tests almost never share state between tests. 1.0 to 2.3% of LLM test functions touch anything shared, against 23.9% of human ones (5.8% without arrow). Human tests use class scoped fixtures, setUp without tearDown, module level decorated functions, and private registries. LLM tests build their own objects, use tmp_path and yield fixtures, and reset library globals after every test. That removes order dependence, which caused 4 of the 7 human flaky test ids.
2. LLM tests do carry more time risk (real sleeps, real clocks, timing bounds), which is where all 3 LLM flaky tests came from. Whether that risk turns into a flaky test depends on how the bound is written. Sonnet freezes or injects the clock and leaves about a second of headroom on upper bounds. Haiku does neither.
3. Haiku is the only flaky generator because, with the same prompt as one shot Sonnet, it writes twice as many tests, does not freeze the schedule clock or inject fake sleeps, writes tighter one sided timing bounds, and once builds a time of day string from the real clock. With 3 flaky tests this is a pattern, not a significant difference.

## Per condition table

Share (%) of test functions with each feature. Pooled over the 12 suites of each condition.

| Feature | Human | AG-S | OS-S | OS-H | TP-S | PL-S |
|---|---|---|---|---|---|---|
| Test functions | 3,077 | 1,074 | 907 | 1,750 | 934 | 927 |
| **Shared state** | | | | | | |
| any shared state | 23.9 | 2.3 | 2.2 | 1.8 | 1.5 | 1.0 |
| any shared state, without arrow | 5.8 | 2.5 | 2.4 | 2.0 | 1.6 | 1.1 |
| writes global state | 1.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| uses module level library object | 0.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| library singleton or registry | 2.9 | 2.3 | 2.2 | 1.8 | 1.5 | 1.0 |
| singleton or registry, no reset after the test | 2.6 | 0.4 | 0.4 | 1.0 | 0.4 | 0.5 |
| resets library global after the test | 0.3 | 8.9 | 9.6 | 1.0 | 3.0 | 8.6 |
| setup without teardown | 4.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| module/class/session scoped fixture | 19.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| touches a private attribute (`x._name`) | 12.3 | 0.1 | 0.3 | 0.2 | 0.0 | 0.9 |
| environ/locale/sys/cwd mutation | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| reads module level list/dict data (not counted as shared) | 4.3 | 9.5 | 2.8 | 0.0 | 1.7 | 1.7 |
| **Isolation** | | | | | | |
| any isolation helper | 6.1 | 18.1 | 24.0 | 8.9 | 17.7 | 20.5 |
| tmp_path/tmpdir | 4.5 | 15.0 | 11.5 | 0.0 | 8.9 | 9.2 |
| tempfile | 0.1 | 0.0 | 0.0 | 6.9 | 0.0 | 0.0 |
| freezegun | 0.4 | 0.9 | 8.8 | 1.9 | 6.4 | 7.7 |
| monkeypatch | 0.2 | 0.0 | 0.6 | 0.0 | 0.0 | 0.8 |
| mock.patch | 0.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| fake sleep injected (tenacity `sleep=`) | 0.1 | 2.1 | 3.6 | 0.0 | 2.4 | 3.0 |
| builds a fresh library object | 18.5 | 35.1 | 40.2 | 43.9 | 43.8 | 48.1 |
| yield fixture (has teardown) | 2.3 | 12.6 | 20.0 | 8.1 | 3.0 | 8.6 |
| **Time** | | | | | | |
| real clock read | 1.9 | 4.0 | 4.2 | 3.5 | 0.2 | 0.4 |
| real sleep | 1.1 | 2.4 | 2.3 | 2.6 | 0.2 | 0.1 |
| timing assertion | 0.1 | 0.7 | 1.5 | 0.9 | 0.1 | 0.3 |
| time of day string from the real clock | 0.0 | 0.0 | 0.0 | 0.1 (1 test) | 0.0 | 0.0 |
| **Randomness** | | | | | | |
| random call | 1.5 | 2.2 | 3.4 | 1.9 | 1.1 | 1.6 |
| unseeded random | 1.1 | 1.3 | 2.3 | 1.4 | 0.6 | 0.5 |
| unseeded and an exact assertion on the value | 0.1 | 0.0 | 1.0 | 0.1 | 0.3 | 0.4 |
| **Concurrency** | | | | | | |
| threads, processes, or asyncio | 1.8 | 5.1 | 5.1 | 3.1 | 2.5 | 2.5 |
| threads | 0.7 | 1.2 | 1.8 | 0.8 | 0.5 | 0.5 |
| threads without a join | 0.1 | 0.1 | 0.2 | 0.1 | 0.0 | 0.1 |
| **Size and style** | | | | | | |
| test functions per suite, mean (min to max) | 256 (49 to 763) | 90 (71 to 205) | 76 (72 to 79) | 146 (91 to 250) | 78 (74 to 80) | 77 (75 to 80) |
| lines per test, mean | 12.1 | 6.0 | 10.4 | 7.7 | 6.0 | 9.1 |
| asserts per test, mean | 2.96 | 1.76 | 3.69 | 1.34 | 2.07 | 3.58 |
| parametrized | 2.5 | 10.2 | 6.6 | 0.0 | 1.1 | 5.3 |
| in a class | 66.5 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| unittest.TestCase | 37.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

The human asserts per test is a lower bound, since several human suites assert through helper functions the scan does not count.

Unseeded random with an exact assertion looks like a risk, but all 21 such tests (4 human, 17 LLM) assert on forced outputs: empty input, `k=0`, sampling the whole input, `jitter=0`, or a wait that has hit its cap. Every other random assertion is a range, membership, or len/set/sorted check. Randomness alone did not produce a flaky test in any condition, and the one random flaky test mixes it with a timing bound (below).

## LLM vs human: shared state and isolation

The 4 order dependent human flaky test ids (3 functions) all sit on library global state that persists between tests. The scan flags every victim and both polluters.

- arrow `test_locale_validation` and `test_locale_name_validation` read `locales._locale_map` through a class scoped fixture (`wide_scope_fixture`, `lib_singleton_use`, `private_attr_access`). The polluter `test_custom_locale_subclass` defines `Locale` subclasses, which register themselves (`lib_global_write`).
- filelock `test_singleton_locks_are_deleted_when_no_external_references_exist` asserts `lock_type._instances == {}`, the whole class level registry, so any lock another test left alive breaks it.

LLM suites use the same singletons but differently.

- filelock. 4 tests in each of AG-S, OS-S, OS-H, and PL-S (1 in TP-S) use `is_singleton=True`, and none of them reads `_instances`. They compare their own two lock objects, inside their own tmp_path or temporary directory, so a leftover lock from another test cannot change the result. Across all LLM tests only 0.0 to 0.9% touch any private attribute, against 12.3% of human tests (3.8% without arrow).
- schedule. All 81 human schedule tests are in a class whose setUp calls `schedule.clear()` with no tearDown, and 65 of them use the default scheduler. So each test cleans before itself but leaves its jobs behind, which is exactly what broke the AG-S schedule tests in the mixed suites. In AG-S all 80 tests run under an autouse conftest fixture that clears the default scheduler before and after. In OS-S 62 of 78 run under a freezegun fixture, and 15 reset after. Most LLM schedule tests also build their own `Scheduler()` (fresh library object in 89% of AG-S, 83% of OS-S, and 90% of OS-H schedule tests, against 2 of 81 human ones).
- humanize. OS-S (72 of 72) and PL-S (78 of 78) put an autouse fixture in every file that calls `humanize.deactivate()` before and after each test.
- arrow. 596 of 708 human arrow test functions get their objects from class scoped fixtures that write `request.cls`. No LLM suite uses a fixture wider than function scope, and no LLM test has setup without teardown or writes a library global.

Pooled over all LLM conditions, 42.3% of tests build a fresh library object and 16.5% use an isolation helper, against 18.5% and 6.1% of human tests. Resetting a library global after the test is 0.3% of human tests and 8.6 to 9.6% of AG-S, OS-S, and PL-S.

Why the difference is plausible. The generated tests come from one prompt per library that asks for public API tests, written in one sitting, and each test is a small self contained example (6 to 10 lines on average) that builds what it needs. Human suites grow over years, test internals (registries, caches, private counters), and share setup to keep hundreds of tests short. Order dependence lives in exactly that shared setup and those internals. The prompt also told the model to use real temporary files for tinydb and boltons and named the module level default scheduler, which likely pushed the models toward tmp_path and explicit resets.

Thread use does not separate the groups. 0.5 to 1.8% of tests start threads in every condition and almost all join them (threads without a join is 0.0 to 0.2% everywhere). The two human pybreaker races are flagged only as real sleep plus threads, which many LLM tests also have, so the scan cannot say why only the human ones raced.

## OS-H vs OS-S (same feature prompt, different model)

| | OS-S | OS-H |
|---|---|---|
| Test functions written | 907 | 1,750 |
| Per suite, mean (min to max) | 76 (72 to 79) | 146 (91 to 250) |
| Suites over the 80 test limit | 0 of 12 | 12 of 12 |
| Files that did not import | 0 | 2 (cachetools `test_osh_func.py`, 38 functions; tenacity `test_osh_async_retry.py`, 16 functions). Both import names the package does not export (`lru_cache` etc. from `cachetools` instead of `cachetools.func`, and `async_retry`, which tenacity does not have) |
| Lines per test, mean | 10.4 | 7.7 |
| Asserts per test, mean | 3.69 | 1.34 |
| In a class / parametrized | 0% / 6.6% | 100% / 0% |
| freezegun | 8.8% | 1.9% |
| schedule tests under freezegun | 62 of 78 | 0 of 182 |
| schedule tests with a real clock read | 0 | 25 |
| tenacity tests with a fake injected sleep | 33 of 74 | 0 of 108 |
| tmp_path / tempfile | 11.5% / 0% | 0% / 6.9% |
| yield fixtures | 20.0% | 8.1% |
| Real sleep | 2.3% (21) | 2.6% (45) |
| Tests with a timing assertion | 14 (1.5%) | 16 (0.9%) |
| Timing assertions | 25, two sided in 9 tests | 16, never two sided |
| Upper bounds: count, slack | 13; min 100 ms, median 1,000 ms; 12 of 13 at 800 ms or more | 6; 70, 90, 100, 100, 500, 1,000 ms |
| Lower bounds: count, slack | 12; all 0 or +10 ms (never below the planned delay) | 10; one at -10 ms (below the guaranteed minimum), others +1 to +140 ms |

Slack is the planned delay against the bound (see the field table). A lower bound with slack 0 or more cannot fail from load, because sleeps do not return early. An upper bound fails when scheduling overhead exceeds its slack.

Sonnet's habit, visible in nearly every timing test it wrote, is to measure the actual sleeps or the planned delay, assert a lower bound 10 ms under it, and an upper bound about one second over it. Haiku writes one sided bounds, closer to the planned delay, and in one case below the minimum the code guarantees. Haiku also wrote twice as many tests from the same prompt. A smaller share of them are timing tests (0.9% against 1.5%), but the count is similar (16 against 14), and each Haiku bound is tighter. Sonnet removes most of the time risk in schedule and tenacity by freezing the clock or injecting a fake sleep, which Haiku never does in those two libraries.

## The 3 OS-H flaky tests next to OS-S

**schedule `test_until_with_string_time_format`.** It passes `(datetime.now() + timedelta(hours=2)).strftime("%H:%M:%S")` to `until()`. It is the only time of day string built from the real clock (`strftime` with `%H` and no date) that the scan found in all 5,592 LLM test functions. How each condition builds the `until()` argument (`case_studies.schedule_until_args`):

| Condition | From the real clock | Other |
|---|---|---|
| OS-H | 9 datetimes, 3 strings (only this one drops the date) | 1 literal string, 1 timedelta, 1 invalid value |
| OS-S | 0 | 5 literal strings under a frozen clock, 2 fixed datetimes, 1 timedelta, 2 invalid values |
| AG-S | 2 datetimes, 1 string with a full date | 4 timedeltas, 1 literal string, 1 invalid value |
| Human | 0 (5 datetimes under the suite's `mock_datetime`) | 4 literal strings, 3 fixed datetimes, 1 timedelta |

OS-S never reads the real clock in schedule (62 of 78 tests use a `frozen` freezegun fixture). AG-S does read it, but its time only test (`test_until_with_time_object`) calls `pytest.skip` when now plus one hour crosses midnight, which is the exact case that broke the Haiku test.

**tenacity `test_wait_random_exponential_in_timing_test`.** `wait_random_exponential(multiplier=0.01, max=0.2)` with two failures sleeps U(0, 0.01) + U(0, 0.02), so the planned delay is 0 to 30 ms with a minimum of 0, and the test asserts `elapsed >= 0.01`. Slack is -10 ms. From those two uniform draws the chance that the sum is under 10 ms is 25%, close to the 14% (laptop) and 24% (Linux) failure rates measured. OS-S has one test that combines random waits with timing (`test_wait_random_sleeps_stay_within_bounds`). It records the sleeps actually taken and asserts `elapsed >= sum(recorded) - 0.01` (+10 ms) and `elapsed < 1.45` for at most 0.45 s of sleep (+1,000 ms). Its other random wait tests call the strategy directly without sleeping and check ranges.

**tenacity `test_stop_before_delay_prevents_exceeding_limit`.** `stop_before_delay(0.1)` with `wait_fixed(0.05)` stops before the second sleep, so the planned delay is 50 ms, and the test asserts `elapsed <= 0.12`, 70 ms of slack. Its one failure took 174 ms under overlapping CPU stress. The OS-S counterpart (`test_stop_before_delay_does_not_overshoot_the_limit`, `stop_before_delay(0.5)`, `wait_fixed(0.2)`, planned 400 ms, bound `< 0.5`) has 100 ms of slack and never failed. The two bounds are only 30 ms apart, so this one is closer to luck at our rerun budget than to a model difference. The other OS-S stop delay test has 1,200 ms of slack.

So two of the three Haiku flaky tests follow from habits that Sonnet does not have (an unfrozen time of day string, a lower bound below the guaranteed minimum). The third is a tight upper bound that Sonnet also wrote once.

## Limits

- The scan is static and per test. It sees shared state only through names it can resolve (module globals, class attributes, known library singletons, private attributes), and it does not follow library internals, so implicit clock reads inside `humanize.naturaltime` or `schedule.next_run` are not counted.
- Planned delays for timing assertions come from a model of tenacity, filelock, more-itertools, and schedule. 9 assertions use a hand derivation listed in `MANUAL_DELAY` in the script.
- Human suites differ in size (49 to 763 test functions per suite). `per_condition_macro` and `per_condition_excluding_arrow` in the JSON show the shares without that weighting. The ordering of the shared state rows does not change.
- 3 flaky LLM tests against 0 is too few for a significance claim (see REPORT.md section 4.1). The features explain how the flaky tests happened, not a measured rate difference between models.

## Added November 2: the two OpenAI conditions (PL-G5, PL-G4)

`scripts/characterize.py` now also scans `gen_plain_gpt54mini` (PL-G5, 897 test functions) and
`gen_plain_gpt41mini` (PL-G4, 985). Same plain prompt as PL-S (927), different model. Shares in %.

| Feature | Human | PL-S | PL-G5 | PL-G4 |
|---|---|---|---|---|
| any shared state | 23.9 | 1.0 | 6.5 | 1.2 |
| touches a private attribute (`x._name`) | 12.3 | 0.9 | 14.2 | 15.5 |
| resets library global after the test | 0.3 | 8.6 | 9.8 | 0.0 |
| monkeypatch | 0.2 | 0.8 | 2.7 | 9.9 |
| builds a fresh library object | 18.5 | 48.1 | 34.0 | 46.2 |
| real clock read | 1.9 | 0.4 | 0.2 | 5.6 |
| timing assertion | 0.1 | 0.3 | 0.0 | 0.1 |

The GPT suites reach into private state about as often as the human suites do, unlike every Claude suite (0
to 0.9%). In humanize, PL-S resets the translation state after all 78 tests, PL-G5 after 2 of 78, and PL-G4
after none of 65, while 9 PL-G4 humanize tests monkeypatch `i18n._CURRENT.locale`. That attribute of a
`threading.local` only exists after `activate()` or `deactivate()` has run in the process, so these tests pass
when an earlier test has activated a locale and fail when they run first. All 10 flaky PL-G4 tests in the
large rerun campaign are this case. It is the first order dependence we have seen in an LLM suite, and it comes
from the same habit (private module state, no reset) that makes the human suites order dependent.
