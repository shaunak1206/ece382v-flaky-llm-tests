import json
from json import JSONEncoder

import pytest

from PyJsonFriendly.jsonfriendly import JsonFriendly


def test_jsonfriendly_is_abstract_and_cannot_be_instantiated():
    with pytest.raises(TypeError):
        JsonFriendly()


def test_subclass_without_implementation_is_still_abstract():
    class MissingImpl(JsonFriendly):
        pass

    with pytest.raises(TypeError):
        MissingImpl()


def test_jsonencoder_default_is_patched_to_lambda():
    assert JSONEncoder.default.__name__ == "<lambda>"


def test_concrete_subclass_can_be_instantiated():
    class Person(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"name": "Ada"}

    p = Person()
    assert isinstance(p, Person)


def test_json_dumps_uses_custom_json_friendly_object():
    class Person(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"name": "Ada", "age": 37}

    assert json.dumps(Person()) == '{"name": "Ada", "age": 37}'


def test_json_dumps_works_with_nested_json_friendly_object_in_list():
    class Item(JsonFriendly):
        def __init__(self, value):
            self.value = value

        def _as_json_friendly_obj(self):
            return {"value": self.value}

    assert json.dumps([Item(1), Item(2)]) == '[{"value": 1}, {"value": 2}]'


def test_json_dumps_works_with_nested_json_friendly_object_in_dict():
    class Payload(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"ok": True}

    data = {"payload": Payload(), "count": 1}
    assert json.dumps(data) == '{"payload": {"ok": true}, "count": 1}'


def test_json_friendly_method_may_return_string():
    class Greeting(JsonFriendly):
        def _as_json_friendly_obj(self):
            return "hello"

    assert json.dumps(Greeting()) == '"hello"'


def test_json_friendly_method_may_return_integer():
    class Answer(JsonFriendly):
        def _as_json_friendly_obj(self):
            return 42

    assert json.dumps(Answer()) == "42"


def test_json_friendly_method_may_return_list():
    class Numbers(JsonFriendly):
        def _as_json_friendly_obj(self):
            return [1, 2, 3]

    assert json.dumps(Numbers()) == "[1, 2, 3]"


def test_encoding_sets_default_backup_attribute():
    class Thing(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"x": 1}

    assert not hasattr(JSONEncoder, "default_backup")
    assert json.dumps(Thing()) == '{"x": 1}'
    assert hasattr(JSONEncoder, "default_backup")
    assert JSONEncoder.default_backup is JSONEncoder.default


def test_default_backup_updates_to_current_default_on_each_encoding():
    class First(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"first": True}

    class Second(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"second": True}

    json.dumps(First())
    first_backup = JSONEncoder.default_backup

    json.dumps(Second())
    second_backup = JSONEncoder.default_backup

    assert first_backup is JSONEncoder.default
    assert second_backup is JSONEncoder.default


def test_multiple_encodings_remain_consistent():
    class Value(JsonFriendly):
        def __init__(self, n):
            self.n = n

        def _as_json_friendly_obj(self):
            return {"n": self.n}

    assert json.dumps(Value(1)) == '{"n": 1}'
    assert json.dumps(Value(2)) == '{"n": 2}'
    assert json.dumps(Value(3)) == '{"n": 3}'
