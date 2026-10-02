"""Tests for TinyDB operations and middleware."""

import tempfile
from pathlib import Path
import pytest
from tinydb import TinyDB, where, operations
from tinydb.storages import JSONStorage, MemoryStorage
from tinydb.middlewares import CachingMiddleware


class TestOperationsDelete:
    """Test delete operation."""

    def test_delete_field(self):
        """Test deleting a field from document."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'name': 'Alice', 'age': 25, 'email': 'alice@example.com'})

        db.update(operations.delete('email'), doc_ids=[doc_id])
        doc = db.get(doc_id=doc_id)
        assert 'email' not in doc
        assert 'name' in doc
        assert 'age' in doc

    def test_delete_multiple_fields_sequentially(self):
        """Test deleting multiple fields."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'a': 1, 'b': 2, 'c': 3})

        db.update(operations.delete('b'), doc_ids=[doc_id])
        db.update(operations.delete('c'), doc_ids=[doc_id])

        doc = db.get(doc_id=doc_id)
        assert doc == {'a': 1}


class TestOperationsAdd:
    """Test add operation."""

    def test_add_to_field(self):
        """Test adding to numeric field."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'count': 5})

        db.update(operations.add('count', 3), doc_ids=[doc_id])
        assert db.get(doc_id=doc_id)['count'] == 8

    def test_add_to_float(self):
        """Test adding to float field."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'price': 10.5})

        db.update(operations.add('price', 2.5), doc_ids=[doc_id])
        assert db.get(doc_id=doc_id)['price'] == 13.0

    def test_add_negative_value(self):
        """Test adding negative value."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'balance': 100})

        db.update(operations.add('balance', -30), doc_ids=[doc_id])
        assert db.get(doc_id=doc_id)['balance'] == 70


class TestOperationsSubtract:
    """Test subtract operation."""

    def test_subtract_from_field(self):
        """Test subtracting from numeric field."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'count': 10})

        db.update(operations.subtract('count', 3), doc_ids=[doc_id])
        assert db.get(doc_id=doc_id)['count'] == 7

    def test_subtract_from_float(self):
        """Test subtracting from float field."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'temperature': 25.5})

        db.update(operations.subtract('temperature', 5.5), doc_ids=[doc_id])
        assert db.get(doc_id=doc_id)['temperature'] == 20.0


class TestOperationsSet:
    """Test set operation."""

    def test_set_field(self):
        """Test setting field value."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'status': 'pending'})

        db.update(operations.set('status', 'complete'), doc_ids=[doc_id])
        assert db.get(doc_id=doc_id)['status'] == 'complete'

    def test_set_to_complex_value(self):
        """Test setting field to complex value."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'data': 'old'})

        new_data = {'nested': {'value': 123}}
        db.update(operations.set('data', new_data), doc_ids=[doc_id])
        assert db.get(doc_id=doc_id)['data'] == new_data


class TestOperationsIncrement:
    """Test increment operation."""

    def test_increment_field(self):
        """Test incrementing field by 1."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'counter': 5})

        db.update(operations.increment('counter'), doc_ids=[doc_id])
        assert db.get(doc_id=doc_id)['counter'] == 6

    def test_increment_multiple_times(self):
        """Test incrementing multiple times."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'counter': 0})

        for _ in range(5):
            db.update(operations.increment('counter'), doc_ids=[doc_id])

        assert db.get(doc_id=doc_id)['counter'] == 5


class TestOperationsDecrement:
    """Test decrement operation."""

    def test_decrement_field(self):
        """Test decrementing field by 1."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'counter': 5})

        db.update(operations.decrement('counter'), doc_ids=[doc_id])
        assert db.get(doc_id=doc_id)['counter'] == 4

    def test_decrement_multiple_times(self):
        """Test decrementing multiple times."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'counter': 10})

        for _ in range(3):
            db.update(operations.decrement('counter'), doc_ids=[doc_id])

        assert db.get(doc_id=doc_id)['counter'] == 7


class TestOperationsWithConditions:
    """Test operations with conditions."""

    def test_delete_with_condition(self):
        """Test delete operation with condition."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'status': 'active', 'temp': True})
        db.insert({'name': 'Bob', 'status': 'inactive', 'temp': True})

        db.update(operations.delete('temp'), where('status') == 'active')

        alice = db.get(where('name') == 'Alice')
        bob = db.get(where('name') == 'Bob')

        assert 'temp' not in alice
        assert 'temp' in bob

    def test_increment_with_condition(self):
        """Test increment with condition."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'score': 10})
        db.insert({'name': 'Bob', 'score': 5})

        db.update(operations.increment('score'), where('score') > 7)

        assert db.get(where('name') == 'Alice')['score'] == 11
        assert db.get(where('name') == 'Bob')['score'] == 5


class TestCombinedOperations:
    """Test combining multiple operations."""

    def test_multiple_operations_in_update(self):
        """Test using multiple operations in single update call."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'a': 1, 'b': 2, 'c': 3})

        # Update with lambda
        db.update(lambda doc: (
            doc.update({'a': 10}),
            doc.pop('b', None)
        ), doc_ids=[doc_id])

        doc = db.get(doc_id=doc_id)
        assert doc['a'] == 10
        assert 'b' not in doc


class TestCachingMiddleware:
    """Test CachingMiddleware."""

    def test_caching_middleware_with_memory(self):
        """Test CachingMiddleware with MemoryStorage."""
        db = TinyDB(storage=CachingMiddleware(MemoryStorage))

        db.insert({'name': 'Alice'})
        db.insert({'name': 'Bob'})

        assert len(db) == 2

    def test_caching_middleware_with_json(self):
        """Test CachingMiddleware with JSONStorage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'
            db = TinyDB(str(db_path), storage=CachingMiddleware(JSONStorage))

            db.insert({'name': 'Alice'})
            db.insert({'name': 'Bob'})
            db.insert({'name': 'Charlie'})

            assert len(db) == 3
            db.close()

            # Verify persistence
            db = TinyDB(str(db_path), storage=CachingMiddleware(JSONStorage))
            assert len(db) == 3
            db.close()

    def test_caching_middleware_cache_size(self):
        """Test CachingMiddleware with custom cache size."""
        db = TinyDB(storage=CachingMiddleware(MemoryStorage))
        middleware = db.storage

        # Insert documents
        for i in range(10):
            db.insert({'num': i})

        # Cache should be updated
        assert middleware.cache is not None

    def test_caching_middleware_flush(self):
        """Test CachingMiddleware flush() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'
            db = TinyDB(str(db_path), storage=CachingMiddleware(JSONStorage))

            db.insert({'name': 'Alice'})
            middleware = db.storage
            middleware.flush()

            db.close()

    def test_caching_middleware_cache_invalidation(self):
        """Test that cache is invalidated on write."""
        db = TinyDB(storage=CachingMiddleware(MemoryStorage))

        doc_id = db.insert({'value': 1})
        middleware = db.storage

        # Cache should be set
        assert middleware.cache is not None
        initial_count = middleware._cache_modified_count

        # Update should increment cache counter
        db.update({'value': 2}, doc_ids=[doc_id])
        assert middleware._cache_modified_count > initial_count


class TestUpdateMultiple:
    """Test update_multiple operation."""

    def test_update_multiple(self):
        """Test updating multiple documents with different queries."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'status': 'active', 'score': 0})
        db.insert({'status': 'inactive', 'score': 0})
        db.insert({'status': 'active', 'score': 0})

        updates = [
            ({'score': 10}, where('status') == 'active'),
            ({'score': 5}, where('status') == 'inactive'),
        ]

        updated_ids = db.update_multiple(updates)
        assert len(updated_ids) == 3

        active_docs = db.search(where('status') == 'active')
        assert all(d['score'] == 10 for d in active_docs)

        inactive_docs = db.search(where('status') == 'inactive')
        assert all(d['score'] == 5 for d in inactive_docs)


class TestOperationsPersistence:
    """Test that operations persist in file storage."""

    def test_operations_persist(self):
        """Test that operations persist across database reopens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'

            # First session
            db = TinyDB(str(db_path), storage=JSONStorage)
            doc_id = db.insert({'counter': 0})
            db.update(operations.increment('counter'), doc_ids=[doc_id])
            db.update(operations.increment('counter'), doc_ids=[doc_id])
            db.close()

            # Second session
            db = TinyDB(str(db_path), storage=JSONStorage)
            doc = db.get(doc_id=doc_id)
            assert doc['counter'] == 2
            db.close()
