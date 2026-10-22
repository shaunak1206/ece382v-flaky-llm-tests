# tenacity: public API (signatures and documentation)


## module `tenacity/__init__.py`

```python
class TryAgain(Exception)
```
Always retry the executed function when raised.

```python
class DoAttempt()
```


```python
class DoSleep(float)
```


```python
class BaseAction()
```
Base class for representing actions to take by retry object.

Concrete implementations must define:
- __init__: to initialize all necessary fields
- REPR_FIELDS: class variable specifying attributes to include in repr(self)
- NAME: for identification in retry object methods and callbacks

```python
class RetryAction(BaseAction)
```


```python
# method of RetryAction
def __init__(self, sleep: t.SupportsFloat) -> None
```


```python
class RetryError(Exception)
```
Encapsulates the last attempt instance right before giving up.

```python
# method of RetryError
def __init__(self, last_attempt: 'Future') -> None
```


```python
# method of RetryError
def reraise(self) -> t.NoReturn
```


```python
class AttemptManager()
```
Manage attempt context.

```python
# method of AttemptManager
def __init__(self, retry_state: 'RetryCallState') -> None
```


```python
# method of AttemptManager
def __enter__(self) -> None
```


```python
# method of AttemptManager
def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: 'types.TracebackType | None') -> bool | None
```


```python
class BaseRetrying(ABC)
```


```python
# method of BaseRetrying
def __init__(self, sleep: t.Callable[[int | float], None]=sleep, stop: 'StopBaseT'=stop_never, wait: 'WaitBaseT'=wait_none(), retry: 'RetryBaseT'=retry_if_exception_type(), before: t.Callable[['RetryCallState'], None]=before_nothing, after: t.Callable[['RetryCallState'], None]=after_nothing, before_sleep: t.Callable[['RetryCallState'], None] | None=None, reraise: bool=False, retry_error_cls: type[RetryError]=RetryError, retry_error_callback: t.Callable[['RetryCallState'], t.Any] | None=None, name: str | None=None, enabled: bool=True) -> None
```


```python
# method of BaseRetrying
def copy(self, sleep: t.Callable[[int | float], None] | object=_unset, stop: 'StopBaseT | object'=_unset, wait: 'WaitBaseT | object'=_unset, retry: retry_base | object=_unset, before: t.Callable[['RetryCallState'], None] | object=_unset, after: t.Callable[['RetryCallState'], None] | object=_unset, before_sleep: t.Callable[['RetryCallState'], None] | object | None=_unset, reraise: bool | object=_unset, retry_error_cls: type[RetryError] | object=_unset, retry_error_callback: t.Callable[['RetryCallState'], t.Any] | object | None=_unset, name: str | object | None=_unset, enabled: bool | object=_unset) -> 'Self'
```
Copy this object with some parameters changed if needed.

```python
# method of BaseRetrying
def statistics(self) -> dict[str, t.Any]
```
Return a dictionary of runtime statistics.

This dictionary will be empty when the controller has never been
ran. When it is running or has ran previously it should have (but
may not) have useful and/or informational keys and values when
running is underway and/or completed.

.. warning:: The keys in this dictionary **should** be somewhat
             stable (not changing), but their existence **m ...

```python
# method of BaseRetrying
def iter_state(self) -> IterState
```


```python
# method of BaseRetrying
def wraps(self, f: t.Callable[P, R]) -> '_RetryDecorated[P, R]'
```
Wrap a function for retrying.

:param f: A function to wrap for retrying.

```python
# method of BaseRetrying
def begin(self) -> None
```


```python
# method of BaseRetrying
def iter(self, retry_state: 'RetryCallState') -> DoAttempt | DoSleep | t.Any
```


```python
# method of BaseRetrying
def __call__(self, fn: t.Callable[..., WrappedFnReturnT], *args: t.Any, **kwargs: t.Any) -> WrappedFnReturnT
```


```python
class Retrying(BaseRetrying)
```
Retrying controller.

```python
# method of Retrying
def __call__(self, fn: t.Callable[..., WrappedFnReturnT], *args: t.Any, **kwargs: t.Any) -> WrappedFnReturnT
```


```python
class Future(futures.Future[t.Any])
```
Encapsulates a (future or past) attempted call to a target function.

```python
# method of Future
def __init__(self, attempt_number: int) -> None
```


```python
# method of Future
def failed(self) -> bool
```
Return whether a exception is being held in this future.

```python
# method of Future
def construct(cls, attempt_number: int, value: t.Any, has_exception: bool) -> 'Future'
```
Construct a new Future object.

```python
class RetryCallState()
```
State related to a single call wrapped with Retrying.

```python
# method of RetryCallState
def __init__(self, retry_object: BaseRetrying, fn: WrappedFn | None, args: t.Any, kwargs: t.Any) -> None
```


```python
# method of RetryCallState
def get_fn_name(self) -> str
```
Get the name of the function being retried.

Returns the fully-qualified name of the wrapped function when used as a
decorator, the ``name`` passed to the retrying object when used as a
context manager, or ``"<unknown>"`` if neither is available.

```python
# method of RetryCallState
def seconds_since_start(self) -> float | None
```


```python
# method of RetryCallState
def prepare_for_next_attempt(self) -> None
```


```python
# method of RetryCallState
def set_result(self, val: t.Any) -> None
```


```python
# method of RetryCallState
def set_exception(self, exc_info: tuple[type[BaseException], BaseException, 'types.TracebackType | None']) -> None
```


```python
def retry(func: t.Callable[P, R]) -> _RetryDecorated[P, R]
```


```python
def retry(*, sleep: t.Callable[[int | float], t.Awaitable[None]], stop: 'StopBaseT'=..., wait: 'WaitBaseT'=..., retry: 'RetryBaseT | tasyncio.retry.RetryBaseT'=..., before: t.Callable[['RetryCallState'], t.Awaitable[None] | None]=..., after: t.Callable[['RetryCallState'], t.Awaitable[None] | None]=..., before_sleep: t.Callable[['RetryCallState'], t.Awaitable[None] | None] | None=..., reraise: bool=..., retry_error_cls: type['RetryError']=..., retry_error_callback: t.Callable[['RetryCallState'], t.Any | t.Awaitable[t.Any]] | None=..., enabled: bool=...) -> _AsyncRetryDecorator
```


```python
def retry(sleep: t.Callable[[int | float], None]=sleep, stop: 'StopBaseT'=stop_never, wait: 'WaitBaseT'=wait_none(), retry: 'RetryBaseT | tasyncio.retry.RetryBaseT'=retry_if_exception_type(), before: t.Callable[['RetryCallState'], t.Awaitable[None] | None]=before_nothing, after: t.Callable[['RetryCallState'], t.Awaitable[None] | None]=after_nothing, before_sleep: t.Callable[['RetryCallState'], t.Awaitable[None] | None] | None=None, reraise: bool=False, retry_error_cls: type['RetryError']=RetryError, retry_error_callback: t.Callable[['RetryCallState'], t.Any | t.Awaitable[t.Any]] | None=None, enabled: bool=True) -> t.Callable[[t.Callable[P, R]], _RetryDecorated[P, R]]
```


```python
def retry(*dargs: t.Any, **dkw: t.Any) -> t.Any
```
Wrap a function with a new `Retrying` object.

:param dargs: positional arguments passed to Retrying object
:param dkw: keyword arguments passed to the Retrying object


## module `tenacity/_utils.py`

```python
class LoggerProtocol(typing.Protocol)
```
Protocol used by utils expecting a logger (eg: before_log).

Compatible with logging, structlog, loguru, etc...

```python
# method of LoggerProtocol
def log(self, level: int, msg: str, *args: typing.Any) -> typing.Any
```


```python
def find_ordinal(pos_num: int) -> str
```


```python
def to_ordinal(pos_num: int) -> str
```


```python
def get_callback_name(cb: typing.Callable[..., typing.Any]) -> str
```
Get a callback fully-qualified name.

If no name can be produced ``repr(cb)`` is called and returned.

```python
def to_seconds(time_unit: time_unit_type) -> float
```


```python
def is_coroutine_callable(call: typing.Callable[..., typing.Any]) -> bool
```


```python
def wrap_to_async_func(call: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Awaitable[typing.Any]]
```



## module `tenacity/after.py`

```python
def after_nothing(retry_state: 'RetryCallState') -> None
```
After call strategy that does nothing.

```python
def after_log(logger: _utils.LoggerProtocol, log_level: int, sec_format: str='%.3g') -> typing.Callable[['RetryCallState'], None]
```
After call strategy that logs to some logger the finished attempt.


## module `tenacity/asyncio/__init__.py`

```python
class AsyncRetrying(BaseRetrying)
```


```python
# method of AsyncRetrying
def __init__(self, sleep: t.Callable[[int | float], t.Awaitable[None] | None]=_portable_async_sleep, stop: 'StopBaseT'=tenacity.stop.stop_never, wait: 'WaitBaseT'=tenacity.wait.wait_none(), retry: 'SyncRetryBaseT | RetryBaseT'=tenacity.retry_if_exception_type(), before: t.Callable[['RetryCallState'], t.Awaitable[None] | None]=before_nothing, after: t.Callable[['RetryCallState'], t.Awaitable[None] | None]=after_nothing, before_sleep: t.Callable[['RetryCallState'], t.Awaitable[None] | None] | None=None, reraise: bool=False, retry_error_cls: type['RetryError']=RetryError, retry_error_callback: t.Callable[['RetryCallState'], t.Any | t.Awaitable[t.Any]] | None=None, name: str | None=None, enabled: bool=True) -> None
```


```python
# method of AsyncRetrying
async def __call__(self, fn: WrappedFn, *args: t.Any, **kwargs: t.Any) -> WrappedFnReturnT
```


```python
# method of AsyncRetrying
async def iter(self, retry_state: 'RetryCallState') -> DoAttempt | DoSleep | t.Any
```


```python
# method of AsyncRetrying
def wraps(self, fn: t.Callable[P, R]) -> _RetryDecorated[P, R]
```



## module `tenacity/asyncio/retry.py`

```python
class async_retry_base(retry_base)
```
Abstract base class for async retry strategies.

```python
# method of async_retry_base
async def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class retry_if_exception(async_retry_base)
```
Retry strategy that retries if an exception verifies a predicate.

```python
# method of retry_if_exception
def __init__(self, predicate: typing.Callable[[BaseException], typing.Awaitable[bool]]) -> None
```


```python
# method of retry_if_exception
async def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class retry_if_result(async_retry_base)
```
Retries if the result verifies a predicate.

```python
# method of retry_if_result
def __init__(self, predicate: typing.Callable[[typing.Any], typing.Awaitable[bool]]) -> None
```


```python
# method of retry_if_result
async def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class retry_any(async_retry_base)
```
Retries if any of the retries condition is valid.

```python
# method of retry_any
def __init__(self, *retries: retry_base | async_retry_base) -> None
```


```python
# method of retry_any
async def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class retry_all(async_retry_base)
```
Retries if all the retries condition are valid.

```python
# method of retry_all
def __init__(self, *retries: retry_base | async_retry_base) -> None
```


```python
# method of retry_all
async def __call__(self, retry_state: 'RetryCallState') -> bool
```



## module `tenacity/before.py`

```python
def before_nothing(retry_state: 'RetryCallState') -> None
```
Before call strategy that does nothing.

```python
def before_log(logger: _utils.LoggerProtocol, log_level: int) -> typing.Callable[['RetryCallState'], None]
```
Before call strategy that logs to some logger the attempt.


## module `tenacity/before_sleep.py`

```python
def before_sleep_nothing(retry_state: 'RetryCallState') -> None
```
Before sleep strategy that does nothing.

```python
def before_sleep_log(logger: _utils.LoggerProtocol, log_level: int, exc_info: bool=False, sec_format: str='%.3g') -> typing.Callable[['RetryCallState'], None]
```
Before sleep strategy that logs to some logger the attempt.


## module `tenacity/nap.py`

```python
def sleep(seconds: float) -> None
```
Sleep strategy that delays execution for a given number of seconds.

This is the default strategy, and may be mocked out for unit testing.

```python
class sleep_using_event()
```
Sleep strategy that waits on an event to be set.

```python
# method of sleep_using_event
def __init__(self, event: 'threading.Event') -> None
```


```python
# method of sleep_using_event
def __call__(self, timeout: float | None) -> None
```



## module `tenacity/retry.py`

```python
class retry_base(abc.ABC)
```
Abstract base class for retry strategies.

```python
# method of retry_base
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class retry_if_exception(retry_base)
```
Retry strategy that retries if an exception verifies a predicate.

```python
# method of retry_if_exception
def __init__(self, predicate: typing.Callable[[BaseException], bool]) -> None
```


```python
# method of retry_if_exception
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class retry_if_exception_type(retry_if_exception)
```
Retries if an exception has been raised of one or more types.

```python
# method of retry_if_exception_type
def __init__(self, exception_types: type[BaseException] | tuple[type[BaseException], ...]=Exception) -> None
```


```python
class retry_if_not_exception_type(retry_if_exception)
```
Retries except an exception has been raised of one or more types.

```python
# method of retry_if_not_exception_type
def __init__(self, exception_types: type[BaseException] | tuple[type[BaseException], ...]=Exception) -> None
```


```python
class retry_unless_exception_type(retry_if_exception)
```
Retries until an exception is raised of one or more types.

```python
# method of retry_unless_exception_type
def __init__(self, exception_types: type[BaseException] | tuple[type[BaseException], ...]=Exception) -> None
```


```python
# method of retry_unless_exception_type
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class retry_if_exception_cause_type(retry_base)
```
Retries if any of the causes of the raised exception is of one or more types.

The check on the type of the cause of the exception is done recursively (until finding
an exception in the chain that has no ``__cause__``, or a cycle is detected).

```python
# method of retry_if_exception_cause_type
def __init__(self, exception_types: type[BaseException] | tuple[type[BaseException], ...]=Exception) -> None
```


```python
# method of retry_if_exception_cause_type
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class retry_if_result(retry_base)
```
Retries if the result verifies a predicate.

```python
# method of retry_if_result
def __init__(self, predicate: typing.Callable[[typing.Any], bool]) -> None
```


```python
# method of retry_if_result
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class retry_if_not_result(retry_base)
```
Retries if the result refutes a predicate.

```python
# method of retry_if_not_result
def __init__(self, predicate: typing.Callable[[typing.Any], bool]) -> None
```


```python
# method of retry_if_not_result
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class retry_if_exception_message(retry_if_exception)
```
Retries if an exception message equals or matches.

```python
# method of retry_if_exception_message
def __init__(self, message: str | None=None, match: str | re.Pattern[str] | None=None) -> None
```


```python
class retry_if_not_exception_message(retry_if_exception_message)
```
Retries until an exception message equals or matches.

```python
# method of retry_if_not_exception_message
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class retry_any(retry_base)
```
Retries if any of the retries condition is valid.

```python
# method of retry_any
def __init__(self, *retries: 'RetryBaseT') -> None
```


```python
# method of retry_any
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class retry_all(retry_base)
```
Retries if all the retries condition are valid.

```python
# method of retry_all
def __init__(self, *retries: 'RetryBaseT') -> None
```


```python
# method of retry_all
def __call__(self, retry_state: 'RetryCallState') -> bool
```



## module `tenacity/stop.py`

```python
class stop_base(abc.ABC)
```
Abstract base class for stop strategies.

```python
# method of stop_base
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class stop_any(stop_base)
```
Stop if any of the stop condition is valid.

```python
# method of stop_any
def __init__(self, *stops: stop_base) -> None
```


```python
# method of stop_any
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class stop_all(stop_base)
```
Stop if all the stop conditions are valid.

```python
# method of stop_all
def __init__(self, *stops: stop_base) -> None
```


```python
# method of stop_all
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class stop_when_event_set(stop_base)
```
Stop when the given event is set.

```python
# method of stop_when_event_set
def __init__(self, event: 'threading.Event') -> None
```


```python
# method of stop_when_event_set
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class stop_after_attempt(stop_base)
```
Stop when the previous attempt >= max_attempt.

```python
# method of stop_after_attempt
def __init__(self, max_attempt_number: int) -> None
```


```python
# method of stop_after_attempt
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class stop_after_delay(stop_base)
```
Stop when the time from the first attempt >= limit.

Note: `max_delay` will be exceeded, so when used with a `wait`, the actual total delay will be greater
than `max_delay` by some of the final sleep period before `max_delay` is exceeded.

If you need stricter timing with waits, consider `stop_before_delay` instead.

```python
# method of stop_after_delay
def __init__(self, max_delay: _utils.time_unit_type) -> None
```


```python
# method of stop_after_delay
def __call__(self, retry_state: 'RetryCallState') -> bool
```


```python
class stop_before_delay(stop_base)
```
Stop right before the next attempt would take place after the time from the first attempt >= limit.

Most useful when you are using with a `wait` function like wait_random_exponential, but need to make
sure that the max_delay is not exceeded.

```python
# method of stop_before_delay
def __init__(self, max_delay: _utils.time_unit_type) -> None
```


```python
# method of stop_before_delay
def __call__(self, retry_state: 'RetryCallState') -> bool
```



## module `tenacity/tornadoweb.py`

```python
class TornadoRetrying(BaseRetrying)
```


```python
# method of TornadoRetrying
def __init__(self, sleep: 'typing.Callable[[float], Future[None]]'=gen.sleep, **kwargs: typing.Any) -> None
```


```python
# method of TornadoRetrying
def __call__(self, fn: 'typing.Callable[..., typing.Generator[typing.Any, typing.Any, _RetValT] | Future[_RetValT]]', *args: typing.Any, **kwargs: typing.Any) -> 'typing.Generator[typing.Any, typing.Any, _RetValT]'
```



## module `tenacity/wait.py`

```python
class wait_base(abc.ABC)
```
Abstract base class for wait strategies.

```python
# method of wait_base
def __call__(self, retry_state: 'RetryCallState') -> float
```


```python
class wait_fixed(wait_base)
```
Wait strategy that waits a fixed amount of time between each retry.

```python
# method of wait_fixed
def __init__(self, wait: _utils.time_unit_type) -> None
```


```python
# method of wait_fixed
def __call__(self, retry_state: 'RetryCallState') -> float
```


```python
class wait_none(wait_fixed)
```
Wait strategy that doesn't wait at all before retrying.

```python
# method of wait_none
def __init__(self) -> None
```


```python
class wait_random(wait_base)
```
Wait strategy that waits a random amount of time between min/max.

```python
# method of wait_random
def __init__(self, min: _utils.time_unit_type=0, max: _utils.time_unit_type=1) -> None
```


```python
# method of wait_random
def __call__(self, retry_state: 'RetryCallState') -> float
```


```python
class wait_combine(wait_base)
```
Combine several waiting strategies.

```python
# method of wait_combine
def __init__(self, *strategies: 'WaitBaseT') -> None
```


```python
# method of wait_combine
def __call__(self, retry_state: 'RetryCallState') -> float
```


```python
class wait_chain(wait_base)
```
Chain two or more waiting strategies.

If all strategies are exhausted, the very last strategy is used
thereafter.

For example::

    @retry(wait=wait_chain(*[wait_fixed(1) for i in range(3)] +
                           [wait_fixed(2) for j in range(5)] +
                           [wait_fixed(5) for k in range(4)]))
    def wait_chained():
        print("Wait 1s for 3 attempts, 2s for 5 attempts and 5s "
              "thereafter.")

```python
# method of wait_chain
def __init__(self, *strategies: wait_base) -> None
```


```python
# method of wait_chain
def __call__(self, retry_state: 'RetryCallState') -> float
```


```python
class wait_exception(wait_base)
```
Wait strategy that waits the amount of time returned by the predicate.

The predicate is passed the exception object. Based on the exception, the
user can decide how much time to wait before retrying.

For example::

    def http_error(exception: BaseException) -> float:
        if (
            isinstance(exception, requests.HTTPError)
            and exception.response.status_code == requests.codes.too_many_requests
        ):
            return float(exception.response.headers.get("Retry-After", "1"))
        return 60.0


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exception(http_error),
    )
    def http_get_request(url: str) -> None:
        response = requests.g ...

```python
# method of wait_exception
def __init__(self, predicate: typing.Callable[[BaseException], float]) -> None
```


```python
# method of wait_exception
def __call__(self, retry_state: 'RetryCallState') -> float
```


```python
class wait_incrementing(wait_base)
```
Wait an incremental amount of time after each attempt.

Starting at a starting value and incrementing by a value for each attempt
(and restricting the upper limit to some maximum value).

```python
# method of wait_incrementing
def __init__(self, start: _utils.time_unit_type=0, increment: _utils.time_unit_type=100, max: _utils.time_unit_type=_utils.MAX_WAIT) -> None
```


```python
# method of wait_incrementing
def __call__(self, retry_state: 'RetryCallState') -> float
```


```python
class wait_exponential(wait_base)
```
Wait strategy that applies exponential backoff.

It allows for a customized multiplier and an ability to restrict the
upper and lower limits to some maximum and minimum value.

The intervals are fixed (i.e. there is no jitter), so this strategy is
suitable for balancing retries against latency when a required resource is
unavailable for an unknown duration, but *not* suitable for resolving
contention between multiple processes for a shared resource. Use
wait_random_exponential for the latter case.

```python
# method of wait_exponential
def __init__(self, multiplier: float=1, max: _utils.time_unit_type=_utils.MAX_WAIT, exp_base: float=2, min: _utils.time_unit_type=0) -> None
```


```python
# method of wait_exponential
def __call__(self, retry_state: 'RetryCallState') -> float
```


```python
class wait_random_exponential(wait_exponential)
```
Random wait with exponentially widening window.

An exponential backoff strategy used to mediate contention between multiple
uncoordinated processes for a shared resource in distributed systems. This
is the sense in which "exponential backoff" is meant in e.g. Ethernet
networking, and corresponds to the "Full Jitter" algorithm described in
this blog post:

https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/

Each retry occurs at a random time in a geometrically expanding interval.
It allows for a custom multiplier and an ability to restrict the upper
limit of the random interval to some maximum value.

Example::

    wait_random_exponential(multiplier=0.5,  # initial wi ...

```python
# method of wait_random_exponential
def __call__(self, retry_state: 'RetryCallState') -> float
```


```python
class wait_exponential_jitter(wait_base)
```
Wait strategy that applies exponential backoff and jitter.

It allows for a customized multiplier, maximum wait, jitter and minimum.

This implements the strategy described here:
https://cloud.google.com/storage/docs/retry-strategy

The wait time is max(min, min(multiplier * 2**n + random.uniform(0, jitter), maximum))
where n is the retry count.

```python
# method of wait_exponential_jitter
def __init__(self, initial: float=1, max: _utils.time_unit_type=_utils.MAX_WAIT, exp_base: float=2, jitter: _utils.time_unit_type=1, min: _utils.time_unit_type=0, multiplier: float=1) -> None
```


```python
# method of wait_exponential_jitter
def __call__(self, retry_state: 'RetryCallState') -> float
```



# Usage examples from the project README

```python
import random
from tenacity import retry

@retry
def do_something_unreliable():
    if random.randint(0, 10) > 1:
        raise IOError("Broken sauce, everything is hosed!!!111one")
    else:
        return "Awesome sauce!"

print(do_something_unreliable())
```
```python
import logging
#
# Note the following import is used for demonstration convenience only.
# Production code should always explicitly import the names it needs.
#
from tenacity import *

class MyException(Exception):
    pass
```
```python
@retry
def never_gonna_give_you_up():
    print("Retry forever ignoring Exceptions, don't wait between retries")
    raise Exception
```
```python
@retry(stop=stop_after_attempt(7))
def stop_after_7_attempts():
    print("Stopping after 7 attempts")
    raise Exception
```
```python
@retry(stop=stop_after_delay(10))
def stop_after_10_s():
    print("Stopping after 10 seconds")
    raise Exception
```
```python
@retry(stop=stop_before_delay(10))
def stop_before_10_s():
    print("Stopping 1 attempt before 10 seconds")
    raise Exception
```
```python
@retry(stop=(stop_after_delay(10) | stop_after_attempt(5)))
def stop_after_10_s_or_5_retries():
    print("Stopping after 10 seconds or 5 retries")
    raise Exception
```
```python
@retry(wait=wait_fixed(2))
def wait_2_s():
    print("Wait 2 second between retries")
    raise Exception
```
```python
@retry(wait=wait_random(min=1, max=2))
def wait_random_1_to_2_s():
    print("Randomly wait 1 to 2 seconds between retries")
    raise Exception
```
```python
@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def wait_exponential_1():
    print("Wait 2^x * 1 second between each retry starting with 4 seconds, then up to 10 seconds, then 10 seconds afterwards")
    raise Exception
```
```python
@retry(wait=wait_fixed(3) + wait_random(0, 2))
def wait_fixed_jitter():
    print("Wait at least 3 seconds, and add up to 2 seconds of random delay")
    raise Exception
```
```python
@retry(wait=wait_random_exponential(multiplier=1, max=60))
def wait_exponential_jitter():
    print("Randomly wait up to 2^x * 1 seconds between each retry until the range reaches 60 seconds, then randomly up to 60 seconds afterwards")
    raise Exception
```
```python
@retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                       [wait_fixed(7) for i in range(2)] +
                       [wait_fixed(9)]))
def wait_fixed_chained():
    print("Wait 3s for 3 attempts, 7s for the next 2 attempts and 9s for all attempts thereafter")
    raise Exception
```
```python
class ClientError(Exception):
    """Some type of client error."""

@retry(retry=retry_if_exception_type(IOError))
def might_io_error():
    print("Retry forever with no wait if an IOError occurs, raise any other errors")
    raise Exception

@retry(retry=retry_if_not_exception_type(ClientError))
def might_client_error():
    print("Retry forever with no wait if any error other than ClientError occurs. Immediately raise ClientError.")
    raise Exception
```
```python
def is_none_p(value):
    """Return True if value is None"""
    return value is None

@retry(retry=retry_if_result(is_none_p))
def might_return_none():
    print("Retry with no wait if return value is None")
```
```python
def is_none_p(value):
    """Return True if value is None"""
    return value is None

@retry(retry=(retry_if_result(is_none_p) | retry_if_exception_type()))
def might_return_none():
    print("Retry forever ignoring Exceptions with no wait if return value is None")
```
```python
@retry
def do_something():
   result = something_else()
   if result == 23:
      raise TryAgain
```
```python
@retry(reraise=True, stop=stop_after_attempt(3))
def raise_my_exception():
    raise MyException("Fail")

try:
    raise_my_exception()
except MyException:
    # timed out retrying
    pass
```
```python
import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), before=before_log(logger, logging.DEBUG))
def raise_my_exception():
    raise MyException("Fail")
```
```python
import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), after=after_log(logger, logging.DEBUG))
def raise_my_exception():
    raise MyException("Fail")
```
```python
import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3),
       before_sleep=before_sleep_log(logger, logging.DEBUG))
def raise_my_exception():
    raise MyException("Fail")
```
```python
@retry(stop=stop_after_attempt(3))
def raise_my_exception():
    raise MyException("Fail")

try:
    raise_my_exception()
except Exception:
    pass

print(raise_my_exception.statistics)
```
```python
def return_last_value(retry_state):
    """return the result of the last call attempt"""
    return retry_state.outcome.result()

def is_false(value):
    """Return True if value is False"""
    return value is False

# will return False after trying 3 times to get a different result
@retry(stop=stop_after_attempt(3),
       retry_error_callback=return_last_value,
       retry=retry_if_result(is_false))
def eventually_return_false():
    return False
```
```python
import logging

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logger = logging.getLogger(__name__)

def log_elapsed(retry_state):
    logger.info('Finished after %.3fs', retry_state.seconds_since_start)

@retry(stop=stop_after_attempt(3), after=log_elapsed)
def raise_my_exception():
    raise MyException("Fail")

try:
    raise_my_exception()
except RetryError:
    pass
```
```python
import logging

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

logger = logging.getLogger(__name__)

def my_before_sleep(retry_state):
    if retry_state.attempt_number < 1:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING
    logger.log(
        loglevel, 'Retrying %s: attempt %s ended with: %s',
        retry_state.fn, retry_state.attempt_number, retry_state.outcome)

@retry(stop=stop_after_attempt(3), before_sleep=my_before_sleep)
def raise_my_exception():
    raise MyException("Fail")

try:
    raise_my_exception()
except RetryError:
    pass
```