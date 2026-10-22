import os
import pickle
import stat

import pytest

from filelock import SoftFileLock, Timeout
from filelock._util import ensure_directory_exists, raise_on_not_writable_file


def test_timeout_is_timeout_error():
    assert issubclass(Timeout, TimeoutError)


def test_timeout_is_oserror():
    assert issubclass(Timeout, OSError)


def test_timeout_lock_file_property():
    assert Timeout("/tmp/some.lock").lock_file == "/tmp/some.lock"


def test_timeout_message_mentions_lock_file():
    assert "/tmp/some.lock" in str(Timeout("/tmp/some.lock"))


def test_timeout_can_be_caught_as_timeout_error():
    with pytest.raises(TimeoutError):
        raise Timeout("x.lock")


def test_timeout_pickle_roundtrip():
    restored = pickle.loads(pickle.dumps(Timeout("/tmp/pick.lock")))
    assert isinstance(restored, Timeout)
    assert restored.lock_file == "/tmp/pick.lock"


def test_acquire_failure_reports_lock_file(tmp_path):
    path = str(tmp_path / "t.lock")
    first = SoftFileLock(path)
    second = SoftFileLock(path)
    with first:
        with pytest.raises(Timeout) as excinfo:
            second.acquire(timeout=0)
    assert excinfo.value.lock_file == path


def test_failed_acquire_does_not_change_counter(tmp_path):
    path = str(tmp_path / "t.lock")
    first = SoftFileLock(path)
    second = SoftFileLock(path)
    with first:
        with pytest.raises(Timeout):
            second.acquire(timeout=0)
        assert second.lock_counter == 0
        assert first.lock_counter == 1


def test_failed_acquire_does_not_release_holder(tmp_path):
    path = str(tmp_path / "t.lock")
    first = SoftFileLock(path)
    second = SoftFileLock(path)
    with first:
        with pytest.raises(Timeout):
            second.acquire(timeout=0)
        assert first.is_locked


def test_timeout_zero_succeeds_on_free_lock(tmp_path):
    lock = SoftFileLock(str(tmp_path / "free.lock"))
    lock.acquire(timeout=0)
    try:
        assert lock.is_locked
    finally:
        lock.release()


def test_ensure_directory_exists_creates_nested_dirs(tmp_path):
    target = tmp_path / "a" / "b" / "c" / "file.lock"
    ensure_directory_exists(target)
    assert (tmp_path / "a" / "b" / "c").is_dir()


def test_ensure_directory_exists_accepts_str(tmp_path):
    target = str(tmp_path / "x" / "y" / "file.lock")
    ensure_directory_exists(target)
    assert os.path.isdir(str(tmp_path / "x" / "y"))


def test_ensure_directory_exists_is_idempotent(tmp_path):
    target = tmp_path / "d" / "file.lock"
    ensure_directory_exists(target)
    ensure_directory_exists(target)
    assert (tmp_path / "d").is_dir()


def test_ensure_directory_exists_does_not_create_file(tmp_path):
    target = tmp_path / "d2" / "file.lock"
    ensure_directory_exists(target)
    assert not target.exists()


def test_raise_on_not_writable_file_nonexistent_ok(tmp_path):
    assert raise_on_not_writable_file(str(tmp_path / "missing.lock")) is None


def test_raise_on_not_writable_file_writable_ok(tmp_path):
    path = tmp_path / "w.lock"
    path.write_text("")
    assert raise_on_not_writable_file(str(path)) is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_raise_on_not_writable_file_readonly_raises(tmp_path):
    path = tmp_path / "ro.lock"
    path.write_text("")
    os.chmod(path, stat.S_IRUSR)
    try:
        with pytest.raises(OSError):
            raise_on_not_writable_file(str(path))
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
