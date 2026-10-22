import pytest
from tinydb import Query, TinyDB, where
from tinydb.storages import MemoryStorage


@pytest.fixture
def db():
    database = TinyDB(storage=MemoryStorage)
    database.insert_multiple([
        {'name': 'John', 'age': 22, 'tags': ['a', 'b']},
        {'name': 'John', 'age': 37, 'tags': ['b']},
        {'name': 'Megan', 'age': 27, 'tags': []},
        {'name': 'Bob', 'age': 42},
    ])
    return database


def names(docs):
    return sorted(d['name'] + str(d['age']) for d in docs)


def test_equality(db):
    User = Query()
    assert names(db.search(User.name == 'Megan')) == ['Megan27']


def test_inequality(db):
    User = Query()
    assert names(db.search(User.name != 'John')) == ['Bob42', 'Megan27']


def test_less_than_and_greater_than(db):
    User = Query()
    assert names(db.search(User.age < 27)) == ['John22']
    assert names(db.search(User.age > 37)) == ['Bob42']


def test_less_equal_and_greater_equal(db):
    User = Query()
    assert names(db.search(User.age <= 27)) == ['John22', 'Megan27']
    assert names(db.search(User.age >= 37)) == ['Bob42', 'John37']


def test_and_combination(db):
    User = Query()
    result = db.search((User.name == 'John') & (User.age <= 30))
    assert result == [{'name': 'John', 'age': 22, 'tags': ['a', 'b']}]


def test_or_combination(db):
    User = Query()
    result = db.search((User.name == 'Megan') | (User.name == 'Bob'))
    assert names(result) == ['Bob42', 'Megan27']


def test_not_combination(db):
    User = Query()
    result = db.search(~(User.name == 'John'))
    assert names(result) == ['Bob42', 'Megan27']


def test_missing_field_does_not_match():
    database = TinyDB(storage=MemoryStorage)
    database.insert({'a': 1})
    assert database.search(where('b') == 1) == []
    assert database.search(where('b') > 0) == []


def test_exists(db):
    assert names(db.search(where('tags').exists())) == ['John22', 'John37', 'Megan27']
    assert names(db.search(~where('tags').exists())) == ['Bob42']


def test_nested_field_access():
    database = TinyDB(storage=MemoryStorage)
    database.insert({'user': {'address': {'city': 'Austin'}}})
    database.insert({'user': {'address': {'city': 'Dallas'}}})
    database.insert({'user': {}})
    User = Query()
    assert len(database.search(User.user.address.city == 'Austin')) == 1
    assert len(database.search(User.user.address.exists())) == 2


def test_matches_requires_regex_to_match_value():
    database = TinyDB(storage=MemoryStorage)
    database.insert_multiple([{'w': 'John'}, {'w': 'John Doe'}, {'w': 'x'}])
    result = database.search(where('w').matches(r'^\w+$'))
    assert result == [{'w': 'John'}, {'w': 'x'}]


def test_search_regex_matches_substring():
    database = TinyDB(storage=MemoryStorage)
    database.insert_multiple([{'w': 'abc123def'}, {'w': 'abcdef'}])
    result = database.search(where('w').search(r'\d+'))
    assert result == [{'w': 'abc123def'}]


def test_test_with_custom_function():
    database = TinyDB(storage=MemoryStorage)
    database.insert_multiple([{'v': 41}, {'v': 42}, {'v': 43}])
    assert database.search(where('v').test(lambda val: val == 42)) == [{'v': 42}]


def test_test_with_extra_arguments():
    database = TinyDB(storage=MemoryStorage)
    database.insert_multiple([{'v': 1}, {'v': 5}, {'v': 10}])

    def between(val, low, high):
        return low <= val <= high

    result = database.search(where('v').test(between, 2, 6))
    assert result == [{'v': 5}]


def test_one_of():
    database = TinyDB(storage=MemoryStorage)
    database.insert_multiple([{'c': 'red'}, {'c': 'green'}, {'c': 'blue'}])
    result = database.search(where('c').one_of(['red', 'blue']))
    assert result == [{'c': 'red'}, {'c': 'blue'}]


def test_any_with_subquery():
    database = TinyDB(storage=MemoryStorage)
    database.insert_multiple([
        {'f1': [{'f2': 1}, {'f2': 0}]},
        {'f1': [{'f2': 0}]},
    ])
    result = database.search(Query().f1.any(Query().f2 == 1))
    assert result == [{'f1': [{'f2': 1}, {'f2': 0}]}]


def test_any_with_list():
    database = TinyDB(storage=MemoryStorage)
    database.insert_multiple([{'f1': [1, 2]}, {'f1': [3, 4, 5]}, {'f1': [7, 8]}])
    result = database.search(Query().f1.any([1, 2, 3]))
    assert result == [{'f1': [1, 2]}, {'f1': [3, 4, 5]}]


def test_all_with_subquery():
    database = TinyDB(storage=MemoryStorage)
    database.insert_multiple([
        {'f1': [{'f2': 1}, {'f2': 1}]},
        {'f1': [{'f2': 1}, {'f2': 0}]},
    ])
    result = database.search(Query().f1.all(Query().f2 == 1))
    assert result == [{'f1': [{'f2': 1}, {'f2': 1}]}]


def test_all_with_list():
    database = TinyDB(storage=MemoryStorage)
    database.insert_multiple([{'f1': [1, 2, 3, 4, 5]}, {'f1': [1, 2]}])
    result = database.search(Query().f1.all([1, 2, 3]))
    assert result == [{'f1': [1, 2, 3, 4, 5]}]


def test_noop_matches_everything(db):
    assert len(db.search(Query().noop())) == 4


def test_map_transforms_value_before_comparison(db):
    User = Query()
    result = db.search(User.age.map(lambda x: x + x) == 44)
    assert names(result) == ['John22']


def test_fragment_matches_subset_of_document(db):
    result = db.search(Query().fragment({'name': 'John', 'age': 37}))
    assert names(result) == ['John37']
