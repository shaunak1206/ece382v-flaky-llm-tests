import io
import json
import os
import sys
from json import JSONEncoder

import pytest

# Make the project root importable when pytest is run from the project root.
sys.path.insert(0, os.getcwd())

from PyJsonFriendly.jsonfriendly import JsonFriendly  # noqa: E402


class Point(JsonFriendly):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def _as_json_friendly_obj(self):
        return {"x": self.x, "y": self.y}


class InheritedPoint(Point):
    """Inherits _as_json_friendly_obj without overriding it."""


class Overridden(Point):
    def _as_json_friendly_obj(self):
        return {"overridden": True, "x": self.x}


class Named(JsonFriendly):
    def __init__(self, name):
        self.name = name

    def _as_json_friendly_obj(self):
        return self.name


class Wrapper(JsonFriendly):
    def __init__(self, inner):
        self.inner = inner

    def _as_json_friendly_obj(self):
        return {"inner": self.inner}


class Unimplemented(JsonFriendly):
    """Does not implement the abstract method."""


class NoneObj(JsonFriendly):
    def _as_json_friendly_obj(self):
        return None


class TupleObj(JsonFriendly):
    def _as_json_friendly_obj(self):
        return (1, 2)


class Boom(JsonFriendly):
    def _as_json_friendly_obj(self):
        raise ValueError("boom")


class Duck:
    """Not a JsonFriendly subclass, but provides the same method."""

    def _as_json_friendly_obj(self):
        return {"quack": True}


class Shadow:
    """Has no class-level method; only an instance attribute."""


def test_module_patches_json_encoder_default():
    assert JSONEncoder.default.__name__ == "<lambda>"


def test_jsonfriendly_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        JsonFriendly()


def test_abstract_method_is_declared():
    assert "_as_json_friendly_obj" in JsonFriendly.__abstractmethods__


def test_subclass_missing_abstract_method_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Unimplemented()


def test_subclass_implementing_method_can_be_instantiated():
    p = Point(1, 2)
    assert isinstance(p, JsonFriendly)


def test_dumps_serializes_dict_representation():
    assert json.loads(json.dumps(Point(1, 2))) == {"x": 1, "y": 2}


def test_dumps_serializes_string_representation():
    assert json.dumps(Named("bob")) == '"bob"'


def test_jsonfriendly_inside_list():
    data = [Point(1, 2), Point(3, 4)]
    assert json.loads(json.dumps(data)) == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]


def test_jsonfriendly_inside_dict():
    data = {"a": Point(5, 6), "b": "plain"}
    assert json.loads(json.dumps(data)) == {"a": {"x": 5, "y": 6}, "b": "plain"}


def test_jsonfriendly_nested_inside_jsonfriendly():
    wrapped = Wrapper(Point(3, 4))
    assert json.loads(json.dumps(wrapped)) == {"inner": {"x": 3, "y": 4}}


def test_inherited_method_is_used():
    assert json.loads(json.dumps(InheritedPoint(7, 8))) == {"x": 7, "y": 8}


def test_overridden_method_in_subclass_is_used():
    assert json.loads(json.dumps(Overridden(9, 0))) == {"overridden": True, "x": 9}


def test_duck_typed_class_with_method_serializes():
    assert json.loads(json.dumps(Duck())) == {"quack": True}


def test_instance_attribute_is_not_used_for_serialization():
    s = Shadow()
    s._as_json_friendly_obj = lambda: {"a": 1}
    with pytest.raises(TypeError):
        json.dumps(s)


def test_unsupported_type_raises_type_error():
    with pytest.raises(TypeError):
        json.dumps(object())


def test_method_returning_none_becomes_null():
    assert json.dumps(NoneObj()) == "null"


def test_method_returning_tuple_becomes_list():
    assert json.loads(json.dumps(TupleObj())) == [1, 2]


def test_exception_in_method_propagates():
    with pytest.raises(ValueError, match="boom"):
        json.dumps(Boom())


def test_json_dump_to_file_like_object():
    buf = io.StringIO()
    json.dump(Point(1, 2), buf)
    assert json.loads(buf.getvalue()) == {"x": 1, "y": 2}


def test_indent_argument_is_respected():
    assert json.dumps(Point(1, 2), indent=2) == json.dumps({"x": 1, "y": 2}, indent=2)


def test_encoder_encode_method_handles_jsonfriendly():
    assert json.loads(JSONEncoder().encode(Point(1, 2))) == {"x": 1, "y": 2}


def test_repeated_dumps_are_consistent():
    p = Point(10, 20)
    first = json.dumps(p)
    second = json.dumps(p)
    assert first == second == '{"x": 10, "y": 20}'


def test_plain_types_still_serialize_normally():
    data = {"a": [1, 2, "x"], "b": None, "c": True}
    assert json.loads(json.dumps(data)) == data


def test_default_backup_attribute_is_recorded_after_dumps():
    json.dumps(Point(1, 2))
    assert hasattr(JSONEncoder, "default_backup")
