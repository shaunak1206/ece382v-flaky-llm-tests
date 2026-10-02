"""Tests for TinyDB Query operations."""

import re
import pytest
from tinydb import TinyDB, Query, where
from tinydb.storages import MemoryStorage
from tinydb.queries import QueryInstance


class TestBasicComparisons:
    """Test basic comparison operators."""

    def test_equality(self):
        """Test equality operator."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'age': 25})
        db.insert({'name': 'Bob', 'age': 25})
        db.insert({'name': 'Charlie', 'age': 30})

        results = db.search(where('age') == 25)
        assert len(results) == 2

    def test_inequality(self):
        """Test inequality operator."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'age': 25})
        db.insert({'name': 'Bob', 'age': 30})

        results = db.search(where('age') != 25)
        assert len(results) == 1
        assert results[0]['name'] == 'Bob'

    def test_less_than(self):
        """Test less than operator."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'value': 10})
        db.insert({'value': 20})
        db.insert({'value': 30})

        results = db.search(where('value') < 25)
        assert len(results) == 2

    def test_less_than_or_equal(self):
        """Test less than or equal operator."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'value': 10})
        db.insert({'value': 20})
        db.insert({'value': 30})

        results = db.search(where('value') <= 20)
        assert len(results) == 2

    def test_greater_than(self):
        """Test greater than operator."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'value': 10})
        db.insert({'value': 20})
        db.insert({'value': 30})

        results = db.search(where('value') > 15)
        assert len(results) == 2

    def test_greater_than_or_equal(self):
        """Test greater than or equal operator."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'value': 10})
        db.insert({'value': 20})
        db.insert({'value': 30})

        results = db.search(where('value') >= 20)
        assert len(results) == 2


class TestExists:
    """Test field existence checking."""

    def test_exists_true(self):
        """Test exists() for existing field."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'age': 25})
        db.insert({'name': 'Bob'})

        results = db.search(where('age').exists())
        assert len(results) == 1
        assert results[0]['name'] == 'Alice'

    def test_exists_false_for_missing_field(self):
        """Test exists() for missing field."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice'})
        db.insert({'name': 'Bob', 'age': 30})

        results = db.search(where('missing_field').exists())
        assert len(results) == 0


class TestRegexMatching:
    """Test regex matching operations."""

    def test_matches_full_string(self):
        """Test matches() for full string."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'email': 'alice@example.com'})
        db.insert({'email': 'bob@test.com'})
        db.insert({'email': 'charlie'})

        results = db.search(where('email').matches(r'^[\w\.-]+@[\w\.-]+\.\w+$'))
        assert len(results) == 2

    def test_search_partial_match(self):
        """Test search() for partial string."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'text': 'hello world'})
        db.insert({'text': 'world hello'})
        db.insert({'text': 'goodbye'})

        results = db.search(where('text').search(r'hello'))
        assert len(results) == 2

    def test_matches_with_flags(self):
        """Test matches() with regex flags."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice'})
        db.insert({'name': 'alice'})
        db.insert({'name': 'BOB'})

        results = db.search(where('name').matches(r'^alice$', re.IGNORECASE))
        assert len(results) == 2


class TestCustomTest:
    """Test custom test functions."""

    def test_test_function(self):
        """Test custom test function."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'value': 10})
        db.insert({'value': 20})
        db.insert({'value': 30})

        def is_even(val):
            return val % 2 == 0

        results = db.search(where('value').test(is_even))
        assert len(results) == 3

    def test_test_with_additional_args(self):
        """Test custom test function with additional arguments."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'value': 10})
        db.insert({'value': 25})
        db.insert({'value': 30})

        def divisible_by(val, n):
            return val % n == 0

        results = db.search(where('value').test(divisible_by, 5))
        assert len(results) == 3


class TestAnyAllOperators:
    """Test any() and all() operators for lists."""

    def test_any_with_query(self):
        """Test any() with query condition."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'tags': [{'name': 'python'}, {'name': 'java'}]})
        db.insert({'tags': [{'name': 'javascript'}]})

        results = db.search(where('tags').any(where('name') == 'python'))
        assert len(results) == 1

    def test_any_with_list(self):
        """Test any() with list membership."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'tags': ['python', 'java']})
        db.insert({'tags': ['javascript']})
        db.insert({'tags': ['python', 'go']})

        results = db.search(where('tags').any(['python', 'java']))
        assert len(results) == 2

    def test_all_with_query(self):
        """Test all() with query condition."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'scores': [{'value': 90}, {'value': 85}]})
        db.insert({'scores': [{'value': 95}, {'value': 60}]})

        results = db.search(where('scores').all(where('value') >= 80))
        assert len(results) == 1

    def test_all_with_list(self):
        """Test all() with list membership."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'items': [1, 2, 3]})
        db.insert({'items': [1, 2]})
        db.insert({'items': [1, 2, 4]})

        results = db.search(where('items').all([1, 2]))
        assert len(results) == 2


class TestOneOf:
    """Test one_of operator."""

    def test_one_of(self):
        """Test one_of() operator."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'status': 'active'})
        db.insert({'status': 'inactive'})
        db.insert({'status': 'pending'})
        db.insert({'status': 'archived'})

        results = db.search(where('status').one_of(['active', 'pending']))
        assert len(results) == 2


class TestFragment:
    """Test fragment operator."""

    def test_fragment(self):
        """Test fragment() operator."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'age': 25, 'city': 'NYC'})
        db.insert({'name': 'Alice', 'age': 25, 'city': 'LA'})
        db.insert({'name': 'Bob', 'age': 25, 'city': 'NYC'})

        fragment = Query().fragment({'name': 'Alice', 'age': 25})
        results = db.search(fragment)
        assert len(results) == 2


class TestLogicalOperators:
    """Test logical AND, OR, and NOT operators."""

    def test_and_operator(self):
        """Test AND operator."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'age': 25})
        db.insert({'name': 'Bob', 'age': 30})
        db.insert({'name': 'Charlie', 'age': 25})

        results = db.search((where('name') == 'Alice') & (where('age') == 25))
        assert len(results) == 1
        assert results[0]['name'] == 'Alice'

    def test_or_operator(self):
        """Test OR operator."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'status': 'active'})
        db.insert({'name': 'Bob', 'status': 'inactive'})
        db.insert({'name': 'Charlie', 'status': 'active'})

        results = db.search((where('name') == 'Alice') | (where('name') == 'Bob'))
        assert len(results) == 2

    def test_not_operator(self):
        """Test NOT operator."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'status': 'active'})
        db.insert({'status': 'inactive'})
        db.insert({'status': 'pending'})

        results = db.search(~(where('status') == 'active'))
        assert len(results) == 2

    def test_complex_logical_expression(self):
        """Test complex logical expression."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice', 'age': 25, 'status': 'active'})
        db.insert({'name': 'Bob', 'age': 30, 'status': 'inactive'})
        db.insert({'name': 'Charlie', 'age': 25, 'status': 'inactive'})
        db.insert({'name': 'Diana', 'age': 35, 'status': 'active'})

        results = db.search(
            ((where('age') < 30) & (where('status') == 'active')) |
            (where('name') == 'Diana')
        )
        assert len(results) == 2


class TestNoop:
    """Test noop query."""

    def test_noop_matches_all(self):
        """Test noop() matches all documents."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'x': 1})
        db.insert({'x': 2})
        db.insert({'y': 3})

        query = Query().noop()
        results = db.search(query)
        assert len(results) == 3


class TestMap:
    """Test map operator for custom path resolution."""

    def test_map_with_function(self):
        """Test map() with custom function."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'data': 'hello'})
        db.insert({'data': 'world'})

        results = db.search(where('data').map(str.upper) == 'HELLO')
        assert len(results) == 1


class TestNestedFields:
    """Test querying nested fields."""

    def test_access_nested_field(self):
        """Test accessing nested field."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'user': {'name': 'Alice', 'age': 25}})
        db.insert({'user': {'name': 'Bob', 'age': 30}})

        results = db.search(where('user')['name'] == 'Alice')
        assert len(results) == 1

    def test_nested_field_with_bracket_notation(self):
        """Test nested field with bracket notation."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'config': {'debug': True, 'port': 8080}})
        db.insert({'config': {'debug': False, 'port': 9000}})

        results = db.search(where('config')['debug'] == True)
        assert len(results) == 1


class TestQueryCaching:
    """Test query caching behavior."""

    def test_repeated_query_uses_cache(self):
        """Test that repeated queries use cache."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'value': 1})
        db.insert({'value': 2})

        query = where('value') == 1
        results1 = db.search(query)
        results2 = db.search(query)

        # Both should return the same results
        assert len(results1) == 1
        assert len(results2) == 1
        assert results1[0]['value'] == results2[0]['value']

    def test_cache_cleared_after_write(self):
        """Test that cache is cleared after write operation."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'value': 1})

        query = where('value') == 1
        results1 = db.search(query)
        assert len(results1) == 1

        # Write invalidates cache
        db.insert({'value': 1})
        results2 = db.search(query)
        assert len(results2) == 2


class TestWhereShorthand:
    """Test where() shorthand function."""

    def test_where_function(self):
        """Test where() shorthand function."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice'})
        db.insert({'name': 'Bob'})

        results = db.search(where('name') == 'Alice')
        assert len(results) == 1

    def test_where_with_complex_paths(self):
        """Test where() with complex field paths."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'profile': {'contact': {'email': 'alice@example.com'}}})
        db.insert({'profile': {'contact': {'email': 'bob@example.com'}}})

        results = db.search(where('profile')['contact']['email'] == 'alice@example.com')
        assert len(results) == 1


class TestQueryInstance:
    """Test QueryInstance behavior."""

    def test_query_instance_is_callable(self):
        """Test that QueryInstance is callable."""
        query = where('value') == 5
        doc = {'value': 5}
        assert query(doc) is True

        doc2 = {'value': 3}
        assert query(doc2) is False

    def test_query_hash_stability(self):
        """Test that query hash is stable."""
        query1 = where('name') == 'Alice'
        query2 = where('name') == 'Alice'

        assert hash(query1) == hash(query2)

    def test_query_equality(self):
        """Test query equality."""
        query1 = where('name') == 'Alice'
        query2 = where('name') == 'Alice'
        query3 = where('name') == 'Bob'

        assert query1 == query2
        assert query1 != query3


class TestEdgeCases:
    """Test edge cases in querying."""

    def test_query_missing_field_returns_false(self):
        """Test querying missing field returns False."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'name': 'Alice'})
        db.insert({'age': 30})

        results = db.search(where('age') == 30)
        assert len(results) == 1

    def test_query_with_none_value(self):
        """Test querying None values."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'value': None})
        db.insert({'value': 5})

        results = db.search(where('value') == None)
        assert len(results) == 1

    def test_query_with_complex_types(self):
        """Test querying with complex types."""
        db = TinyDB(storage=MemoryStorage)
        db.insert({'data': [1, 2, 3]})
        db.insert({'data': [4, 5, 6]})

        results = db.search(where('data') == [1, 2, 3])
        assert len(results) == 1
