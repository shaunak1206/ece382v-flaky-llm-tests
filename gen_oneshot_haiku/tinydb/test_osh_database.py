"""Tests for TinyDB database and table operations."""

import os
import tempfile
from pathlib import Path
import pytest
from tinydb import TinyDB, Query, where
from tinydb.storages import JSONStorage, MemoryStorage
from tinydb.table import Document


class TestTinyDBBasic:
    """Test basic TinyDB functionality."""

    def test_database_with_memory_storage(self):
        """Test TinyDB with MemoryStorage."""
        db = TinyDB(storage=MemoryStorage)
        assert len(db) == 0
        db.insert({'name': 'John'})
        assert len(db) == 1

    def test_database_with_json_storage(self):
        """Test TinyDB with JSONStorage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'
            db = TinyDB(str(db_path), storage=JSONStorage)
            db.insert({'name': 'Alice'})
            assert len(db) == 1
            db.close()

            # Reopen database
            db = TinyDB(str(db_path), storage=JSONStorage)
            assert len(db) == 1
            assert db.all()[0]['name'] == 'Alice'
            db.close()

    def test_database_context_manager(self):
        """Test TinyDB as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.json'
            with TinyDB(str(db_path), storage=JSONStorage) as db:
                db.insert({'name': 'Bob'})
                assert len(db) == 1


class TestTinyDBTables:
    """Test table management in TinyDB."""

    def test_default_table(self):
        """Test default table access."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'data': 'test'})
        assert doc_id == 1
        assert db.get(doc_id=doc_id)['data'] == 'test'

    def test_named_tables(self):
        """Test accessing named tables."""
        db = TinyDB(storage=MemoryStorage)
        table1 = db.table('users')
        table2 = db.table('posts')

        table1.insert({'name': 'Alice'})
        table2.insert({'title': 'Hello'})

        assert len(table1) == 1
        assert len(table2) == 1
        assert table1.all()[0]['name'] == 'Alice'
        assert table2.all()[0]['title'] == 'Hello'

    def test_table_persistence(self):
        """Test that table instances are cached."""
        db = TinyDB(storage=MemoryStorage)
        table1 = db.table('test_table')
        table2 = db.table('test_table')
        assert table1 is table2

    def test_tables_method(self):
        """Test tables() method."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'x': 1})  # default table
        db.table('users').insert({'name': 'Alice'})
        db.table('posts').insert({'title': 'Hello'})

        tables = db.tables()
        assert '_default' in tables
        assert 'users' in tables
        assert 'posts' in tables
        assert len(tables) == 3

    def test_drop_table(self):
        """Test dropping a specific table."""
        db = TinyDB(storage=MemoryStorage)
        db.table('users').insert({'name': 'Alice'})
        db.table('posts').insert({'title': 'Hello'})

        assert 'users' in db.tables()
        db.drop_table('users')
        assert 'users' not in db.tables()
        assert 'posts' in db.tables()

    def test_drop_tables(self):
        """Test dropping all tables."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'x': 1})
        db.table('users').insert({'name': 'Alice'})
        db.table('posts').insert({'title': 'Hello'})

        assert len(db.tables()) > 0
        db.drop_tables()
        assert len(db.tables()) == 0

    def test_repr(self):
        """Test database representation."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'x': 1})
        db.table('users').insert({'name': 'Alice'})
        repr_str = repr(db)
        assert 'TinyDB' in repr_str
        assert '_default' in repr_str


class TestTableInsert:
    """Test insert operations on tables."""

    def test_insert_single_document(self):
        """Test inserting a single document."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'name': 'John', 'age': 30})
        assert doc_id == 1
        assert len(db) == 1

    def test_insert_multiple_documents(self):
        """Test inserting multiple documents."""
        db = TinyDB(storage=MemoryStorage)
        docs = [
            {'name': 'John', 'age': 30},
            {'name': 'Jane', 'age': 28},
            {'name': 'Bob', 'age': 35}
        ]
        doc_ids = db.insert_multiple(docs)
        assert doc_ids == [1, 2, 3]
        assert len(db) == 3

    def test_insert_with_explicit_doc_id(self):
        """Test inserting a document with explicit ID."""
        db = TinyDB(storage=MemoryStorage)
        doc = Document({'name': 'John'}, doc_id=10)
        doc_id = db.insert(doc)
        assert doc_id == 10
        assert db.get(doc_id=10)['name'] == 'John'

    def test_insert_invalid_document_raises_error(self):
        """Test that inserting non-mapping raises error."""
        db = TinyDB(storage=MemoryStorage)
        with pytest.raises(ValueError):
            db.insert("not a dict")


class TestTableSearch:
    """Test search operations on tables."""

    def test_search_basic(self):
        """Test basic search."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'age': 25})
        db.insert({'name': 'Bob', 'age': 30})
        db.insert({'name': 'Charlie', 'age': 25})

        results = db.search(where('age') == 25)
        assert len(results) == 2
        names = [r['name'] for r in results]
        assert 'Alice' in names
        assert 'Charlie' in names

    def test_search_returns_documents(self):
        """Test that search returns Document objects."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice'})

        results = db.search(where('name') == 'Alice')
        assert isinstance(results[0], Document)
        assert results[0].doc_id == 1

    def test_search_empty_result(self):
        """Test search with no results."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice'})

        results = db.search(where('name') == 'NonExistent')
        assert results == []

    def test_all_documents(self):
        """Test all() method."""
        db = TinyDB(storage=MemoryStorage)
        docs = [
            {'name': 'Alice'},
            {'name': 'Bob'},
            {'name': 'Charlie'}
        ]
        db.insert_multiple(docs)

        all_docs = db.all()
        assert len(all_docs) == 3
        names = [d['name'] for d in all_docs]
        assert names == ['Alice', 'Bob', 'Charlie']


class TestTableGet:
    """Test get operations on tables."""

    def test_get_by_doc_id(self):
        """Test getting document by ID."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'name': 'Alice'})
        doc = db.get(doc_id=doc_id)
        assert doc['name'] == 'Alice'

    def test_get_nonexistent_doc_id(self):
        """Test getting nonexistent document returns None."""
        db = TinyDB(storage=MemoryStorage)
        doc = db.get(doc_id=999)
        assert doc is None

    def test_get_by_condition(self):
        """Test getting document by condition."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'age': 25})
        db.insert({'name': 'Bob', 'age': 30})

        doc = db.get(where('name') == 'Alice')
        assert doc['age'] == 25

    def test_get_multiple_by_doc_ids(self):
        """Test getting multiple documents by IDs."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice'})
        db.insert({'name': 'Bob'})
        db.insert({'name': 'Charlie'})

        docs = db.get(doc_ids=[1, 3])
        assert len(docs) == 2
        names = [d['name'] for d in docs]
        assert 'Alice' in names
        assert 'Charlie' in names


class TestTableContains:
    """Test contains operations on tables."""

    def test_contains_by_doc_id(self):
        """Test checking if document exists by ID."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'name': 'Alice'})
        assert db.contains(doc_id=doc_id)
        assert not db.contains(doc_id=999)

    def test_contains_by_condition(self):
        """Test checking if document exists by condition."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice'})
        assert db.contains(where('name') == 'Alice')
        assert not db.contains(where('name') == 'NonExistent')


class TestTableCount:
    """Test count operations on tables."""

    def test_count_all(self):
        """Test counting all documents."""
        db = TinyDB(storage=MemoryStorage)
        assert len(db) == 0
        db.insert({'x': 1})
        assert len(db) == 1
        db.insert_multiple([{'x': 2}, {'x': 3}])
        assert len(db) == 3

    def test_count_with_condition(self):
        """Test counting with condition."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'age': 25})
        db.insert({'name': 'Bob', 'age': 30})
        db.insert({'name': 'Charlie', 'age': 25})

        count = db.count(where('age') == 25)
        assert count == 2


class TestTableUpdate:
    """Test update operations on tables."""

    def test_update_by_doc_id(self):
        """Test updating document by ID."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'name': 'Alice', 'age': 25})

        updated_ids = db.update({'age': 26}, doc_ids=[doc_id])
        assert updated_ids == [doc_id]
        assert db.get(doc_id=doc_id)['age'] == 26

    def test_update_by_condition(self):
        """Test updating documents matching condition."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'age': 25})
        db.insert({'name': 'Bob', 'age': 30})
        db.insert({'name': 'Charlie', 'age': 25})

        updated_ids = db.update({'age': 26}, where('age') == 25)
        assert len(updated_ids) == 2

        results = db.search(where('age') == 26)
        assert len(results) == 2

    def test_update_all_documents(self):
        """Test updating all documents."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'status': 'pending'})
        db.insert({'status': 'pending'})

        updated_ids = db.update({'status': 'done'})
        assert len(updated_ids) == 2
        assert all(d['status'] == 'done' for d in db.all())

    def test_update_with_callable(self):
        """Test updating with callable."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'count': 5})

        db.update(lambda doc: doc.update({'count': 10}), doc_ids=[doc_id])
        assert db.get(doc_id=doc_id)['count'] == 10

    def test_update_nonexistent_doc_id_is_skipped(self):
        """Test updating nonexistent IDs is skipped."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'x': 1})

        updated_ids = db.update({'x': 2}, doc_ids=[999])
        assert updated_ids == []


class TestTableUpsert:
    """Test upsert operations on tables."""

    def test_upsert_update_existing(self):
        """Test upsert updates existing document."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'name': 'Alice', 'age': 25})

        result_ids = db.upsert({'name': 'Alice', 'age': 26}, where('name') == 'Alice')
        assert result_ids == [doc_id]
        assert db.get(doc_id=doc_id)['age'] == 26

    def test_upsert_insert_new(self):
        """Test upsert inserts new document if not found."""
        db = TinyDB(storage=MemoryStorage)
        result_ids = db.upsert({'name': 'Bob', 'age': 30}, where('name') == 'Bob')

        assert len(result_ids) == 1
        assert db.get(doc_id=result_ids[0])['name'] == 'Bob'

    def test_upsert_with_doc_id(self):
        """Test upsert with explicit doc_id."""
        db = TinyDB(storage=MemoryStorage)
        doc = Document({'name': 'Charlie', 'age': 35}, doc_id=5)

        result_ids = db.upsert(doc)
        assert result_ids == [5]
        assert db.get(doc_id=5)['name'] == 'Charlie'


class TestTableRemove:
    """Test remove operations on tables."""

    def test_remove_by_doc_id(self):
        """Test removing document by ID."""
        db = TinyDB(storage=MemoryStorage)
        doc_id = db.insert({'name': 'Alice'})

        removed_ids = db.remove(doc_ids=[doc_id])
        assert removed_ids == [doc_id]
        assert db.get(doc_id=doc_id) is None

    def test_remove_by_condition(self):
        """Test removing documents by condition."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'status': 'active'})
        db.insert({'name': 'Bob', 'status': 'inactive'})
        db.insert({'name': 'Charlie', 'status': 'active'})

        removed_ids = db.remove(where('status') == 'active')
        assert len(removed_ids) == 2
        assert len(db) == 1

    def test_remove_nonexistent_doc_id_is_skipped(self):
        """Test removing nonexistent IDs is skipped."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'x': 1})

        removed_ids = db.remove(doc_ids=[999])
        assert removed_ids == []
        assert len(db) == 1


class TestTableTruncate:
    """Test truncate operations on tables."""

    def test_truncate_removes_all(self):
        """Test truncate removes all documents."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple([{'x': 1}, {'x': 2}, {'x': 3}])
        assert len(db) == 3

        db.truncate()
        assert len(db) == 0

    def test_truncate_resets_id_counter(self):
        """Test truncate resets ID counter."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'x': 1})
        db.truncate()

        doc_id = db.insert({'x': 2})
        assert doc_id == 1


class TestTableIteration:
    """Test iteration over documents."""

    def test_iterate_all_documents(self):
        """Test iterating over all documents."""
        db = TinyDB(storage=MemoryStorage)
        docs = [{'x': 1}, {'x': 2}, {'x': 3}]
        db.insert_multiple(docs)

        iterated = list(db)
        assert len(iterated) == 3
        assert all(isinstance(d, Document) for d in iterated)

    def test_iterate_empty_table(self):
        """Test iterating over empty table."""
        db = TinyDB(storage=MemoryStorage)
        assert list(db) == []
