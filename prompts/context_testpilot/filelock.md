# filelock: public API (signatures and documentation)


## module `src/filelock/_api.py`

```python
class AcquireReturnProxy()
```
A context-aware object that will release the lock file when exiting.

```python
# method of AcquireReturnProxy
def __init__(self, lock: BaseFileLock) -> None
```


```python
# method of AcquireReturnProxy
def __enter__(self) -> BaseFileLock
```


```python
# method of AcquireReturnProxy
def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None
```


```python
class BaseFileLock(contextlib.ContextDecorator)
```
Abstract base class for a file lock object.

```python
# method of BaseFileLock
def __init__(self, lock_file: str | os.PathLike[str], timeout: float=-1, mode: int=420, thread_local: bool=True, *, blocking: bool=True, is_singleton: bool=False) -> None
```
Create a new lock object.

:param lock_file: path to the file
:param timeout: default timeout when acquiring the lock, in seconds. It will be used as fallback value in             the acquire method, if no timeout value (``None``) is given. If you want to disable the timeout, set it             to a negative value. A timeout of 0 means that there is exactly one attempt to acquire the file lock.
:p ...

```python
# method of BaseFileLock
def is_thread_local(self) -> bool
```
:return: a flag indicating if this lock is thread local or not

```python
# method of BaseFileLock
def is_singleton(self) -> bool
```
:return: a flag indicating if this lock is singleton or not

```python
# method of BaseFileLock
def lock_file(self) -> str
```
:return: path to the lock file

```python
# method of BaseFileLock
def timeout(self) -> float
```
:return: the default timeout value, in seconds

.. versionadded:: 2.0.0

```python
# method of BaseFileLock
def timeout(self, value: float | str) -> None
```
Change the default timeout value.

:param value: the new value, in seconds

```python
# method of BaseFileLock
def blocking(self) -> bool
```
:return: whether the locking is blocking or not

```python
# method of BaseFileLock
def blocking(self, value: bool) -> None
```
Change the default blocking value.

:param value: the new value as bool

```python
# method of BaseFileLock
def mode(self) -> int
```
:return: the file permissions for the lockfile

```python
# method of BaseFileLock
def is_locked(self) -> bool
```
:return: A boolean indicating if the lock file is holding the lock currently.

.. versionchanged:: 2.0.0

    This was previously a method and is now a property.

```python
# method of BaseFileLock
def lock_counter(self) -> int
```
:return: The number of times this lock has been acquired (but not yet released).

```python
# method of BaseFileLock
def acquire(self, timeout: float | None=None, poll_interval: float=0.05, *, poll_intervall: float | None=None, blocking: bool | None=None) -> AcquireReturnProxy
```
Try to acquire the file lock.

:param timeout: maximum wait time for acquiring the lock, ``None`` means use the default :attr:`~timeout` is and
 if ``timeout < 0``, there is no timeout and this method will block until the lock could be acquired
:param poll_interval: interval of trying to acquire the lock file
:param poll_intervall: deprecated, kept for backwards compatibility, use ``poll_interval` ...

```python
# method of BaseFileLock
def release(self, force: bool=False) -> None
```
Releases the file lock. Please note, that the lock is only completely released, if the lock counter is 0.
Also note, that the lock file itself is not automatically deleted.

:param force: If true, the lock counter is ignored and the lock is released in every case/

```python
# method of BaseFileLock
def __enter__(self) -> Self
```
Acquire the lock.

:return: the lock object

```python
# method of BaseFileLock
def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None
```
Release the lock.

:param exc_type: the exception type if raised
:param exc_value: the exception value if raised
:param traceback: the exception traceback if raised


## module `src/filelock/_error.py`

```python
class Timeout(TimeoutError)
```
Raised when the lock could not be acquired in *timeout* seconds.

```python
# method of Timeout
def __init__(self, lock_file: str) -> None
```


```python
# method of Timeout
def lock_file(self) -> str
```
:return: The path of the file lock.


## module `src/filelock/_soft.py`

```python
class SoftFileLock(BaseFileLock)
```
Simply watches the existence of the lock file.


## module `src/filelock/_util.py`

```python
def raise_on_not_writable_file(filename: str) -> None
```
Raise an exception if attempting to open the file for writing would fail.

This is done so files that will never be writable can be separated from files that are writable but currently
locked.

:param filename: file to check
:raises OSError: as if the file was opened for writing.

```python
def ensure_directory_exists(filename: Path | str) -> None
```
Ensure the directory containing the file exists (create it if necessary).

:param filename: file.


## module `src/filelock/asyncio.py`

```python
class AsyncAcquireReturnProxy()
```
A context-aware object that will release the lock file when exiting.

```python
# method of AsyncAcquireReturnProxy
def __init__(self, lock: BaseAsyncFileLock) -> None
```


```python
class BaseAsyncFileLock(BaseFileLock)
```
Base class for asynchronous file locks.

```python
# method of BaseAsyncFileLock
def __init__(self, lock_file: str | os.PathLike[str], timeout: float=-1, mode: int=420, thread_local: bool=False, *, blocking: bool=True, is_singleton: bool=False, loop: asyncio.AbstractEventLoop | None=None, run_in_executor: bool=True, executor: futures.Executor | None=None) -> None
```
Create a new lock object.

:param lock_file: path to the file
:param timeout: default timeout when acquiring the lock, in seconds. It will be used as fallback value in             the acquire method, if no timeout value (``None``) is given. If you want to disable the timeout, set it             to a negative value. A timeout of 0 means that there is exactly one attempt to acquire the file lock.
:p ...

```python
# method of BaseAsyncFileLock
def run_in_executor(self) -> bool
```
::return: whether run in executor.

```python
# method of BaseAsyncFileLock
def executor(self) -> futures.Executor | None
```
::return: the executor.

```python
# method of BaseAsyncFileLock
def executor(self, value: futures.Executor | None) -> None
```
Change the executor.

:param value: the new executor or ``None``
:type value: futures.Executor | None

```python
# method of BaseAsyncFileLock
def loop(self) -> asyncio.AbstractEventLoop | None
```
::return: the event loop.

```python
# method of BaseAsyncFileLock
async def acquire(self, timeout: float | None=None, poll_interval: float=0.05, *, blocking: bool | None=None) -> AsyncAcquireReturnProxy
```
Try to acquire the file lock.

:param timeout: maximum wait time for acquiring the lock, ``None`` means use the default
    :attr:`~BaseFileLock.timeout` is and if ``timeout < 0``, there is no timeout and
    this method will block until the lock could be acquired
:param poll_interval: interval of trying to acquire the lock file
:param blocking: defaults to True. If False, function will return imm ...

```python
# method of BaseAsyncFileLock
async def release(self, force: bool=False) -> None
```
Releases the file lock. Please note, that the lock is only completely released, if the lock counter is 0.
Also note, that the lock file itself is not automatically deleted.

:param force: If true, the lock counter is ignored and the lock is released in every case/

```python
# method of BaseAsyncFileLock
def __enter__(self) -> NoReturn
```
Replace old __enter__ method to avoid using it.

NOTE: DO NOT USE `with` FOR ASYNCIO LOCKS, USE `async with` INSTEAD.

:return: none
:rtype: NoReturn

```python
class AsyncSoftFileLock(SoftFileLock, BaseAsyncFileLock)
```
Simply watches the existence of the lock file.

```python
class AsyncUnixFileLock(UnixFileLock, BaseAsyncFileLock)
```
Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems.

```python
class AsyncWindowsFileLock(WindowsFileLock, BaseAsyncFileLock)
```
Uses the :func:`msvcrt.locking` to hard lock the lock file on windows systems.
