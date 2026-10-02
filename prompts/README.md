# Generation prompts

Three generation conditions, all prompted with the same per-library API description (below).

| Condition | Model | Execution feedback | Output dir | File prefix |
|---|---|---|---|---|
| AG-S (agentic) | Claude Sonnet (subagent) | runs suite once, may fix or delete failing tests up to 3 iterations | llm_tests/<proj> | test_llm_ |
| OS-S (one-shot) | Claude Sonnet (subagent) | none, must not execute anything | gen_oneshot_sonnet/<proj> | test_oss_ |
| OS-H (one-shot) | Claude Haiku 4.5 (subagent) | none, must not execute anything | gen_oneshot_haiku/<proj> | test_osh_ |

All conditions: tests from source and docstrings only (library tests dir is off limits), 50 to 80 test functions,
plain pytest, allowed packages pytest, pytest-asyncio, freezegun, stdlib. Generated in September and October 2026.
For one-shot suites the harness applies a single-run generate-and-filter step afterwards (scripts/filter_oneshot.py).

## Per-library API descriptions
- cachetools (src/cachetools): Cache, LRUCache, LFUCache, FIFOCache, RRCache, TTLCache, TLRUCache, @cached and @cachedmethod (lock and key args, cache_info, cache_clear), cachetools.func, cachetools.keys. Include TTL/TLRU expiry and RRCache random eviction tests the way you naturally would.
- schedule (schedule/__init__.py): every().seconds..weeks and weekday chaining, at, until, to, tag, do, Scheduler (run_pending, run_all, clear, cancel_job, get_jobs, next_run, idle_seconds), Job next_run/last_run/should_run, CancelJob, module level default scheduler, exceptions. Include tests that jobs actually run when due.
- tinydb (tinydb/): TinyDB with JSONStorage on real files and MemoryStorage, tables, Query/where operators, operations, CachingMiddleware. Use real temporary files.
- humanize (src/humanize/): number, time (naturaltime/naturalday relative to now), filesize, lists, i18n activate/deactivate with fr_FR or de_DE.
- toolz (toolz/): itertoolz, functoolz, dicttoolz, recipes, curried, sandbox; assertions on dict/set returning functions.
- tabulate (tabulate/): tabulate() across tablefmts, headers, showindex, input types, alignment, floatfmt/intfmt, missingval, maxcolwidths, SEPARATING_LINE, CLI via subprocess; exact output strings.
- tenacity (tenacity/): @retry, Retrying/AsyncRetrying, stop_*, wait_* (fixed, random, exponential, random_exponential, incrementing, chain, +), retry_if_*, before/after/before_sleep, reraise, RetryError, statistics, retry_with, async. Include real retry timing and random jitter tests.
- filelock (src/filelock/): FileLock, SoftFileLock, context manager/decorator, reentrancy and lock counter, is_locked, timeout and Timeout, blocking=False, poll_interval, thread_local, multiple threads, AsyncFileLock. Include threads competing for one lock.
- boltons (cacheutils, iterutils, timeutils, queueutils, fileutils, dictutils, setutils, strutils): LRU, LRI, cached, ThresholdCounter, chunked, windowed, remap, bucketize, relative_time, parse_timedelta, isoparse, daterange, priority queues, atomic_save, mkdir_p, iter_find_files, OrderedMultiDict, OneToOne, FrozenDict, IndexedSet, slugify, bytes2human. Include relative_time vs current time and real temp files.
- more-itertools (more.py, recipes.py): broad sample incl. peekable, seekable, spy, bucket, distinct_permutations, powerset, unique_everseen, roundrobin, collapse, map_reduce, sample, random_product, random_permutation, random_combination(_with_replacement), nth_product, time_limited. Include random_* and time_limited tests.
- arrow (api, factory, arrow, parser, formatter, locales): get, now, utcnow, shift, replace, to, span, floor, ceil, range, span_range, interval, format/parse tokens, humanize/dehumanize (relative to now, a couple of locales), arithmetic, isoformat, timestamp. Include now/utcnow/humanize vs current time.
- pybreaker (src/pybreaker/__init__.py): CircuitBreaker fail_max, reset_timeout, closed/open/half-open transitions, call/call_async, decorator, exclude, success_threshold, listeners, CircuitMemoryStorage, counters, open/close/half_open, CircuitBreakerError. Include waiting for reset_timeout and checking the state change.

## Deviations recorded during generation (October 1, 2026)
- One-shot agents never ran pytest. Some one-shot Haiku agents ran `python -m py_compile` on their files (tabulate, boltons, toolz, more_itertools) and two imported their test modules (tinydb, toolz). This only gives syntax and import feedback, never test outcomes. One one-shot Sonnet agent used a python one-liner to do a string replacement in its own file.
- One-shot Haiku created empty `__init__.py` files in the humanize and arrow test folders. Because our folders are named after the library, those files made the test folder shadow the real package. We deleted the empty files before any run. This is a harness artifact, not a property of the tests.
- The agentic more-itertools agent wrote 121 tests and then deleted 44 passing ones to meet the 50 to 80 target.
- Per test timeout: the first 9 generated suites were rerun with a 60 second pytest-timeout. One-shot Haiku schedule contains `test_zero_interval`, which hangs forever (every(0) never finds a next run), and under random order the leftover job in the global default scheduler also hangs later module level tests, so one run took about 5 minutes. From that suite on, all reruns use a 15 second timeout. No passing test in any suite takes more than about 1 second, so this changes run time and not outcomes.
- Execution: suites were rerun one at a time until 18:45 on October 1. From then on two queue workers ran suites in parallel (plus the long one-shot Haiku schedule job and the human baseline job), so the machine was busier during the later suites. Extra load can only make timing problems more likely to show, so this does not hide flakiness.
- Stress lock: from 20:25 on October 1, stress runs take a machine wide file lock so only one stress run burns the CPUs at a time. Before that, up to two stress runs could overlap (16 burners on 8 cores), which only made the stress harder. Each run also has a 900 second wall clock cap.
