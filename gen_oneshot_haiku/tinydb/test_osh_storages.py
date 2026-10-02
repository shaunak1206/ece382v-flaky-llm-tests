"""Tests for TinyDB storage implementations."""

import json
import tempfile
from pathlib import Path
import pytest
from tinydb import TinyDB
from tinydb.storages import JSONStorage, MemoryStorage


class TestMemoryStorage:
    """Test MemoryStorage implementation."""

    def test_memory_storage_basic(self):
        """Test basic MemoryStorage operations."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'name': 'Alice'})
        assert db.get(doc_id=doc_id)['name'] == 'Alice'

    def test_memory_storage_read_empty(self):
        """Test MemoryStorage read on empty database."""
        storage = MemoryStorage()
        assert storage.read() is None

    def test_memory_storage_write_and_read(self):
        """Test MemoryStorage write and read."""
        storage = MemoryStorage()
        data = {'table1': {1: {'name': 'Alice'}}}
        storage.write(data)
        assert storage.read() == data

    def test_memory_storage_is_not_persistent(self):
        """Test that MemoryStorage is not persistent."""
        db1 = TinyDB(storage=MemoryStorage)
        db1.insert({'name': 'Alice'})

        db2 = TinyDB(storage=MemoryStorage)
        assert len(db2) == 0

    def test_memory_storage_multiple_tables(self):
        """Test MemoryStorage with multiple tables."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'x': 1})
        db.table('users').insert({'name': 'Alice'})
        db.table('posts').insert({'title': 'Hello'})

        storage = db.storage
        data = storage.read()

        assert '_default' in data
        assert 'users' in data
        assert 'posts' in data


class TestJSONStorage:
    """Test JSONStorage implementation."""

    def test_json_storage_basic(self):
        """Test basic JSONStorage operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'
            db = TinyDB(str(db_path), storage=JSONStorage)
            doc_id = db.insert({'name': 'Alice'})
            assert db.get(doc_id=doc_id)['name'] == 'Alice'
            db.close()

    def test_json_storage_persistence(self):
        """Test JSONStorage persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'

            # First session
            db = TinyDB(str(db_path), storage=JSONStorage)
            db.insert({'name': 'Alice'})
            db.close()

            # Verify file exists
            assert db_path.exists()

            # Second session
            db = TinyDB(str(db_path), storage=JSONStorage)
            assert len(db) == 1
            assert db.all()[0]['name'] == 'Alice'
            db.close()

    def test_json_storage_empty_file(self):
        """Test JSONStorage with empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'
            db_path.touch()  # Create empty file

            db = TinyDB(str(db_path), storage=JSONStorage)
            assert len(db) == 0
            db.close()

    def test_json_storage_creates_file(self):
        """Test that JSONStorage creates file if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'
            assert not db_path.exists()

            db = TinyDB(str(db_path), storage=JSONStorage)
            db.insert({'name': 'Alice'})
            db.close()

            assert db_path.exists()

    def test_json_storage_creates_dirs(self):
        """Test that JSONStorage creates directories with create_dirs=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'subdir' / 'nested' / 'test.json'
            assert not db_path.parent.exists()

            db = TinyDB(str(db_path), create_dirs=True, storage=JSONStorage)
            db.insert({'name': 'Alice'})
            db.close()

            assert db_path.exists()

    def test_json_storage_read_empty(self):
        """Test JSONStorage read on empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'
            db_path.touch()

            storage = JSONStorage(str(db_path))
            assert storage.read() is None
            storage.close()

    def test_json_storage_multiple_tables(self):
        """Test JSONStorage with multiple tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'

            db = TinyDB(str(db_path), storage=JSONStorage)
            db.insert({'x': 1})
            db.table('users').insert({'name': 'Alice'})
            db.table('posts').insert({'title': 'Hello'})
            db.close()

            # Verify all tables persisted
            db = TinyDB(str(db_path), storage=JSONStorage)
            assert len(db.table('users')) == 1
            assert len(db.table('posts')) == 1
            db.close()

    def test_json_storage_data_integrity(self):
        """Test that JSONStorage preserves data integrity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'

            data = {
                'value': 123,
                'text': 'hello',
                'list': [1, 2, 3],
                'nested': {'key': 'value'}
            }

            db = TinyDB(str(db_path), storage=JSONStorage)
            db.insert(data)
            db.close()

            db = TinyDB(str(db_path), storage=JSONStorage)
            retrieved = db.all()[0]
            assert retrieved == data
            db.close()

    def test_json_storage_handles_special_characters(self):
        """Test JSONStorage with special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'

            db = TinyDB(str(db_path), storage=JSONStorage)
            db.insert({'text': 'hello\nworld', 'emoji': '🎉'})
            db.close()

            db = TinyDB(str(db_path), storage=JSONStorage)
            doc = db.all()[0]
            assert doc['text'] == 'hello\nworld'
            assert doc['emoji'] == '🎉'
            db.close()

    def test_json_storage_access_modes(self):
        """Test JSONStorage with different access modes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'

            # Write with r+
            db = TinyDB(str(db_path), access_mode='r+', storage=JSONStorage)
            db.insert({'name': 'Alice'})
            db.close()

            # Read with r
            storage = JSONStorage(str(db_path), access_mode='r')
            data = storage.read()
            assert data is not None
            storage.close()

    def test_json_storage_json_serialization_kwargs(self):
        """Test JSONStorage with custom JSON serialization kwargs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'

            # Store with indentation
            db = TinyDB(str(db_path), indent=2, storage=JSONStorage)
            db.insert({'name': 'Alice'})
            db.close()

            # Verify file is formatted
            content = db_path.read_text()
            assert '  ' in content or '\n' in content


class TestStorageInterface:
    """Test storage interface compliance."""

    def test_storage_read_write_contract(self):
        """Test that storage read/write contract is maintained."""
        storage = MemoryStorage()

        # Write data
        data = {'table1': {1: {'name': 'Alice'}, 2: {'name': 'Bob'}}}
        storage.write(data)

        # Read data
        read_data = storage.read()
        assert read_data == data

    def test_storage_write_overwrites_previous(self):
        """Test that write overwrites previous data."""
        storage = MemoryStorage()

        storage.write({'table1': {1: {'x': 1}}})
        storage.write({'table2': {1: {'y': 2}}})

        data = storage.read()
        assert 'table1' not in data
        assert 'table2' in data

    def test_json_storage_file_truncation(self):
        """Test that JSONStorage truncates file on write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'

            # Write large data
            db = TinyDB(str(db_path), storage=JSONStorage)
            for i in range(100):
                db.insert({'num': i, 'data': 'x' * 100})
            db.close()

            initial_size = db_path.stat().st_size

            # Write small data
            db = TinyDB(str(db_path), storage=JSONStorage)
            db.truncate()
            db.insert({'x': 1})
            db.close()

            final_size = db_path.stat().st_size
            assert final_size < initial_size


class TestStorageClose:
    """Test storage close behavior."""

    def test_json_storage_close(self):
        """Test JSONStorage close method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'
            storage = JSONStorage(str(db_path))
            storage.write({'table1': {1: {'x': 1}}})
            storage.close()

            # After close, we should not be able to use the storage
            with pytest.raises(ValueError):
                storage.read()

    def test_memory_storage_close(self):
        """Test MemoryStorage close method."""
        storage = MemoryStorage()
        storage.write({'table1': {1: {'x': 1}}})
        storage.close()

        # MemoryStorage.close() is a no-op, so this should still work
        data = storage.read()
        assert data is not None
