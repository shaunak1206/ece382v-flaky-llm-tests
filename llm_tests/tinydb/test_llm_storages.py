"""
Unit tests for tinydb.storages: JSONStorage (using real temporary files)
and MemoryStorage.
"""
import json
import os

import pytest

from tinydb.storages import JSONStorage, MemoryStorage, touch


def test_memory_storage_read_returns_none_when_empty():
    storage = MemoryStorage()
    assert storage.read() is None


def test_memory_storage_write_then_read_roundtrip():
    storage = MemoryStorage()
    data = {"_default": {"1": {"a": 1}}}
    storage.write(data)
    assert storage.read() == data


def test_touch_creates_file(tmp_path):
    path = str(tmp_path / "newfile.json")
    assert not os.path.exists(path)
    touch(path, create_dirs=False)
    assert os.path.exists(path)


def test_json_storage_read_empty_file_returns_none(tmp_path):
    path = str(tmp_path / "empty.json")
    open(path, "a").close()
    storage = JSONStorage(path)
    try:
        assert storage.read() is None
    finally:
        storage.close()


def test_json_storage_write_and_read(tmp_path):
    path = str(tmp_path / "data.json")
    storage = JSONStorage(path)
    try:
        data = {"_default": {"1": {"name": "Alice"}}}
        storage.write(data)
        assert storage.read() == data
    finally:
        storage.close()


def test_json_storage_creates_file_if_missing(tmp_path):
    path = str(tmp_path / "auto.json")
    assert not os.path.exists(path)
    storage = JSONStorage(path)
    storage.close()
    assert os.path.exists(path)


def test_json_storage_persists_to_disk_as_valid_json(tmp_path):
    path = str(tmp_path / "on_disk.json")
    storage = JSONStorage(path)
    try:
        storage.write({"_default": {"1": {"x": 1}}})
    finally:
        storage.close()

    with open(path) as f:
        raw = json.load(f)
    assert raw == {"_default": {"1": {"x": 1}}}


def test_json_storage_overwrite_with_shorter_data_truncates(tmp_path):
    path = str(tmp_path / "shrink.json")
    storage = JSONStorage(path)
    try:
        storage.write({"_default": {str(i): {"n": i} for i in range(20)}})
        storage.write({"_default": {"1": {"n": 1}}})
        result = storage.read()
    finally:
        storage.close()

    assert result == {"_default": {"1": {"n": 1}}}


def test_json_storage_readonly_mode_raises_on_write(tmp_path):
    path = str(tmp_path / "readonly.json")
    with open(path, "w") as f:
        json.dump({"_default": {}}, f)

    storage = JSONStorage(path, access_mode="r")
    try:
        with pytest.raises(IOError):
            storage.write({"_default": {"1": {"a": 1}}})
    finally:
        storage.close()


def test_json_storage_warns_on_unsafe_access_mode(tmp_path):
    path = str(tmp_path / "warn.json")
    with pytest.warns(UserWarning):
        storage = JSONStorage(path, access_mode="w")
    storage.close()
