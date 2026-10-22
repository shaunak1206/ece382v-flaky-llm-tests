# pybreaker: public API (signatures and documentation)


## module `src/pybreaker/__init__.py`

```python
class CircuitBreaker()
```
More abstractly, circuit breakers exists to allow one subsystem to fail
without destroying the entire system.

This is done by wrapping dangerous operations (typically integration points)
with a component that can circumvent calls when the system is not healthy.

This pattern is described by Michael T. Nygard in his book 'Release It!'.

```python
# method of CircuitBreaker
def __init__(self, fail_max: int=5, reset_timeout: float=60, success_threshold: int=1, exclude: Iterable[type[ExceptionType] | Callable[[Any], bool]] | None=None, listeners: Sequence[CBListenerType] | None=None, state_storage: CircuitBreakerStorage | None=None, name: str | None=None, throw_new_error_on_trip: bool=True) -> None
```
Create a new circuit breaker with the given parameters.

```python
# method of CircuitBreaker
def fail_counter(self) -> int
```
Return the current number of consecutive failures.

```python
# method of CircuitBreaker
def success_counter(self) -> int
```
Return the current number of consecutive successes in half-open state.

```python
# method of CircuitBreaker
def fail_max(self) -> int
```
Return the maximum number of failures tolerated before the circuit is opened.

```python
# method of CircuitBreaker
def fail_max(self, number: int) -> None
```
Set the maximum `number` of failures tolerated before the circuit is opened.

```python
# method of CircuitBreaker
def reset_timeout(self) -> float
```
Once this circuit breaker is opened, it should remain opened until the
timeout period, in seconds, elapses.

```python
# method of CircuitBreaker
def reset_timeout(self, timeout: float) -> None
```
Set the `timeout` period, in seconds, this circuit breaker should be kept open.

```python
# method of CircuitBreaker
def success_threshold(self) -> int
```
Return the number of successful requests required before transitioning from half-open to closed state.

```python
# method of CircuitBreaker
def success_threshold(self, threshold: int) -> None
```
Set the number of successful requests required before transitioning from half-open to closed state.

```python
# method of CircuitBreaker
def state(self) -> CBStateType
```
Update (if needed) and returns the cached state object.

```python
# method of CircuitBreaker
def state(self, state_str: str) -> None
```
Set cached state and notify listeners of newly cached state.

```python
# method of CircuitBreaker
def current_state(self) -> str
```
Return a string that identifies the state of the circuit breaker as
reported by the _state_storage. i.e., 'closed', 'open', 'half-open'.

```python
# method of CircuitBreaker
def excluded_exceptions(self) -> tuple[type[ExceptionType] | Callable[[Any], bool], ...]
```
Return the list of excluded exceptions, e.g., exceptions that should
not be considered system errors by this circuit breaker.

```python
# method of CircuitBreaker
def add_excluded_exception(self, exception: type[ExceptionType]) -> None
```
Add an exception to the list of excluded exceptions.

```python
# method of CircuitBreaker
def add_excluded_exceptions(self, *exceptions: type[ExceptionType]) -> None
```
Add exceptions to the list of excluded exceptions.

```python
# method of CircuitBreaker
def remove_excluded_exception(self, exception: type[ExceptionType]) -> None
```
Remove an exception from the list of excluded exceptions.

```python
# method of CircuitBreaker
def is_system_error(self, exception: ExceptionType) -> bool
```
Return whether the exception `exception` is considered a signal of
system malfunction. Business exceptions should not cause this circuit
breaker to open.

```python
# method of CircuitBreaker
def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T
```
Call `func` with the given `args` and `kwargs` according to the rules
implemented by the current state of this circuit breaker.

```python
# method of CircuitBreaker
def calling(self) -> Any
```
Return a context manager, enabling the circuit breaker to be used with a
`with` statement. The block of code inside the `with` statement will be
executed according to the rules implemented by the current state of this
circuit breaker.

```python
# method of CircuitBreaker
def call_async(self, func, *args, **kwargs)
```
Call async `func` with the given `args` and `kwargs` according to the rules
implemented by the current state of this circuit breaker.

Return a closure to prevent import errors when using without tornado present

```python
# method of CircuitBreaker
def open(self) -> bool
```
Open the circuit, e.g., the following calls will immediately fail until timeout elapses.

```python
# method of CircuitBreaker
def half_open(self) -> None
```
Half-open the circuit, e.g. lets the following call pass through and
opens the circuit if the call fails (or closes the circuit if the call
succeeds).

```python
# method of CircuitBreaker
def close(self) -> None
```
Close the circuit, e.g. lets the following calls execute as usual.

```python
# method of CircuitBreaker
def __call__(self, *call_args: Any, **call_kwargs: bool) -> Callable
```
Return a wrapper that calls the function `func` according to the rules
implemented by the current state of this circuit breaker.

Optionally takes the keyword argument `__pybreaker_call_coroutine`,
which will will call `func` as a Tornado co-routine.

```python
# method of CircuitBreaker
def listeners(self) -> tuple[CBListenerType, ...]
```
Return the registered listeners as a tuple.

```python
# method of CircuitBreaker
def add_listener(self, listener: CBListenerType) -> None
```
Register a listener for this circuit breaker.

```python
# method of CircuitBreaker
def add_listeners(self, *listeners: CBListenerType) -> None
```
Register listeners for this circuit breaker.

```python
# method of CircuitBreaker
def remove_listener(self, listener: CBListenerType) -> None
```
Unregister a listener of this circuit breaker.

```python
# method of CircuitBreaker
def name(self) -> str | None
```
Return the name of this circuit breaker. Useful for logging.

```python
# method of CircuitBreaker
def name(self, name: str) -> None
```
Set the name of this circuit breaker.

```python
class CircuitMemoryStorage(CircuitBreakerStorage)
```
Implement a `CircuitBreakerStorage` in local memory.

```python
# method of CircuitMemoryStorage
def __init__(self, state: str) -> None
```
Create a new instance with the given `state`.

```python
# method of CircuitMemoryStorage
def state(self) -> str
```
Return the current circuit breaker state.

```python
# method of CircuitMemoryStorage
def state(self, state: str) -> None
```
Set the current circuit breaker state to `state`.

```python
# method of CircuitMemoryStorage
def increment_counter(self) -> None
```
Increase the failure counter by one.

```python
# method of CircuitMemoryStorage
def reset_counter(self) -> None
```
Set the failure counter to zero.

```python
# method of CircuitMemoryStorage
def increment_success_counter(self) -> None
```
Increase the success counter by one.

```python
# method of CircuitMemoryStorage
def reset_success_counter(self) -> None
```
Set the success counter to zero.

```python
# method of CircuitMemoryStorage
def counter(self) -> int
```
Return the current value of the failure counter.

```python
# method of CircuitMemoryStorage
def success_counter(self) -> int
```
Return the current value of the success counter.

```python
# method of CircuitMemoryStorage
def opened_at(self) -> datetime | None
```
Return the most recent value of when the circuit was opened.

```python
# method of CircuitMemoryStorage
def opened_at(self, datetime: datetime) -> None
```
Set the most recent value of when the circuit was opened to `datetime`.

```python
class CircuitRedisStorage(CircuitBreakerStorage)
```
Implement a `CircuitBreakerStorage` using redis.

```python
# method of CircuitRedisStorage
def __init__(self, state: str, redis_object: Redis, namespace: str | None=None, fallback_circuit_state: str=STATE_CLOSED, cluster_mode: bool=False)
```
Create a new instance with the given `state` and `redis` object. The
redis object should be similar to pyredis' StrictRedis class. If there
are any connection issues with redis, the `fallback_circuit_state` is
used to determine the state of the circuit.

```python
# method of CircuitRedisStorage
def state(self) -> str
```
Return the current circuit breaker state.

If the circuit breaker state on Redis is missing, re-initialize it
with the fallback circuit state and reset the fail counter.

```python
# method of CircuitRedisStorage
def state(self, state: str) -> None
```
Set the current circuit breaker state to `state`.

```python
# method of CircuitRedisStorage
def increment_counter(self) -> None
```
Increase the failure counter by one.

```python
# method of CircuitRedisStorage
def reset_counter(self) -> None
```
Set the failure counter to zero.

```python
# method of CircuitRedisStorage
def increment_success_counter(self) -> None
```
Increase the success counter by one.

```python
# method of CircuitRedisStorage
def reset_success_counter(self) -> None
```
Set the success counter to zero.

```python
# method of CircuitRedisStorage
def counter(self) -> int
```
Return the current value of the failure counter.

```python
# method of CircuitRedisStorage
def success_counter(self) -> int
```
Return the current value of the success counter.

```python
# method of CircuitRedisStorage
def opened_at(self) -> datetime | None
```
Returns a datetime object of the most recent value of when the circuit was opened.

```python
# method of CircuitRedisStorage
def opened_at(self, now: datetime) -> None
```
Atomically set the most recent value of when the circuit was opened
to `now`. Stored in redis as a simple integer of unix epoch time.
To avoid timezone issues between different systems, the passed in
datetime should be in UTC.

```python
class CircuitBreakerListener()
```
Listener class used to plug code to a ``CircuitBreaker`` instance when certain events happen.

```python
# method of CircuitBreakerListener
def before_call(self, cb: CircuitBreaker, func: Callable[..., T], *args: Any, **kwargs: Any) -> None
```
This callback function is called before the circuit breaker `cb` calls `fn`.

```python
# method of CircuitBreakerListener
def failure(self, cb: CircuitBreaker, exc: BaseException) -> None
```
This callback function is called when a function called by the circuit breaker `cb` fails.

```python
# method of CircuitBreakerListener
def success(self, cb: CircuitBreaker) -> None
```
This callback function is called when a function called by the circuit breaker `cb` succeeds.

```python
# method of CircuitBreakerListener
def state_change(self, cb: CircuitBreaker, old_state: CircuitBreakerState | None, new_state: CircuitBreakerState) -> None
```
This callback function is called when the state of the circuit breaker `cb` state changes.

```python
class CircuitBreakerError(Exception)
```
When calls to a service fails because the circuit is open, this error is
raised to allow the caller to handle this type of exception differently.


# Usage examples from the project README

```python
There's no need to subclass ``CircuitBreaker`` if you just want to take action
when certain events occur. In that case, it's better to subclass
``CircuitBreakerListener`` instead:

.. code:: python

    class DBListener(pybreaker.CircuitBreakerListener):
        "Listener used by circuit breakers that execute database operations."

        def before_call(self, cb, func, *args, **kwargs):
            "Called before the circuit breaker `cb` calls `func`."
            pass

        def state_change(self, cb, old_state, new_state):
            "Called when the circuit breaker `cb` state changes."
            pass

        def failure(self, cb, exc):
            "Called when a function invocation raises a system error."
            pass

        def success(self, cb):
            "Called when a function invocation succeeds."
            pass

    class LogListener(pybreaker.CircuitBreakerListener):
        "Listener used to log circuit breaker events."

        def state_change(self, cb, old_state, new_state):
            msg = "State Change: CB: {0}, New State: {1}".format(cb.name, new_state)
            logging.info(msg)


To add listeners to a circuit breaker:

.. code:: python

    # At creation time...
    db_breaker = pybreaker.CircuitBreaker(listeners=[DBListener(), LogListener()])

    # ...or later
    db_breaker.add_listeners(OneListener(), AnotherListener())


What Does a Circuit Breaker Do?
```
```python
Let's say you want to use a circuit breaker on a function that updates a row
in the ``customer`` database table:

.. code:: python

    @db_breaker
    def update_customer(cust):
        # Do stuff here...
        pass

    # Will trigger the circuit breaker
    updated_customer = update_customer(my_customer)


Or if you don't want to use the decorator syntax:

.. code:: python

    def update_customer(cust):
        # Do stuff here...
        pass

    # Will trigger the circuit breaker
    updated_customer = db_breaker.call(update_customer, my_customer)

Or use it as a context manager and a `with` statement:

.. code:: python

    # Will trigger the circuit breaker
    with db_breaker.calling():
        # Do stuff here...
        pass



According to the default parameters, the circuit breaker ``db_breaker`` will
automatically open the circuit after 5 consecutive failures in
``update_customer``.

When the circuit is open, all calls to ``update_customer`` will fail immediately
(raising ``CircuitBreakerError``) without any attempt to execute the real
operation. If you want the original error to be thrown when the circuit trips,
set the ``throw_new_error_on_trip`` option to ``False``:

.. code:: python

    pybreaker.CircuitBreaker(..., throw_new_error_on_trip=False)


After 60 seconds, the circuit breaker will allow the next call to
``update_customer`` pass through. If that call succeeds, the circuit is closed;
if it fails, however, the circuit is opened again until another timeout elapses.

Optional Tornado Support
```
```python
A circuit breaker can (optionally) be used to call asynchronous Tornado functions:

.. code:: python

    from tornado import gen

    @db_breaker(__pybreaker_call_async=True)
    @gen.coroutine
    def async_update(cust):
        # Do async stuff here...
        pass

Or if you don't want to use the decorator syntax:

.. code:: python

    @gen.coroutine
    def async_update(cust):
        # Do async stuff here...
        pass

    updated_customer = db_breaker.call_async(async_update, my_customer)


Excluding Exceptions
```
```python
By default, a failed call is any call that raises an exception. However, it's
common to raise exceptions to also indicate business exceptions, and those
exceptions should be ignored by the circuit breaker as they don't indicate
system errors:

.. code:: python

    # At creation time...
    db_breaker = CircuitBreaker(exclude=[CustomerValidationError])

    # ...or later
    db_breaker.add_excluded_exception(CustomerValidationError)


In that case, when any function guarded by that circuit breaker raises
``CustomerValidationError`` (or any exception derived from
``CustomerValidationError``), that call won't be considered a system failure.

So as to cover cases where the exception class alone is not enough to determine
whether it represents a system error, you may also pass a callable rather than
a type:

.. code:: python

    db_breaker = CircuitBreaker(exclude=[lambda e: type(e) == HTTPError and e.status_code < 500])

You may mix types and filter callables freely.


Monitoring and Management
```