import json
import os
import sys

import pytest

# Ensure the project root (current working directory when running from root) is importable.
_ROOT = os.getcwd()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyJsonFriendly.jsonfriendly import JsonFriendly


class Inner(JsonFriendly):
    def _as_json_friendly_obj(self):
        return {"v": 1}


class Point(JsonFriendly):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def _as_json_friendly_obj(self):
        return {"x": self.x, "y": self.y}


class Base(JsonFriendly):
    def _as_json_friendly_obj(self):
        return {"kind": "base"}


class Child(Base):
    def _as_json_friendly_obj(self):
        return {"kind": "child"}


class InheritsBase(Base):
    pass


class Outer(JsonFriendly):
    def _as_json_friendly_obj(self):
        return {"inner": Inner()}


class Failing(JsonFriendly):
    def _as_json_friendly_obj(self):
        raise ValueError("boom")


class Incomplete(JsonFriendly):
    pass


def test_jsonfriendly_itself_cannot_be_instantiated():
    with pytest.raises(TypeError):
        JsonFriendly()


def test_subclass_without_implementation_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Incomplete()


def test_subclass_with_implementation_can_be_instantiated():
    obj = Inner()
    assert isinstance(obj, JsonFriendly)


def test_dumps_uses_returned_dict():
    assert json.loads(json.dumps(Point(1, 2))) == {"x": 1, "y": 2}


def test_dumps_produces_expected_text():
    assert json.dumps(Point(1, 2)) == '{"x": 1, "y": 2}'


def test_returned_list_is_serialized_as_array():
    class ListObj(JsonFriendly):
        def _as_json_friendly_obj(self):
            return [1, 2, 3]

    assert json.dumps(ListObj()) == "[1, 2, 3]"


def test_returned_string_is_serialized_as_string():
    class StrObj(JsonFriendly):
        def _as_json_friendly_obj(self):
            return "hello"

    assert json.dumps(StrObj()) == '"hello"'


def test_returned_number_is_serialized_as_number():
    class NumObj(JsonFriendly):
        def _as_json_friendly_obj(self):
            return 42

    assert json.dumps(NumObj()) == "42"


def test_returned_none_is_serialized_as_null():
    class NoneObj(JsonFriendly):
        def _as_json_friendly_obj(self):
            return None

    assert json.dumps(NoneObj()) == "null"


def test_returned_bool_is_serialized_as_boolean():
    class BoolObj(JsonFriendly):
        def _as_json_friendly_obj(self):
            return True

    assert json.dumps(BoolObj()) == "true"


def test_nested_jsonfriendly_inside_returned_dict_is_encoded():
    assert json.loads(json.dumps(Outer())) == {"inner": {"v": 1}}


def test_jsonfriendly_inside_list_is_encoded():
    assert json.dumps([Inner(), 3]) == '[{"v": 1}, 3]'


def test_jsonfriendly_inside_dict_value_is_encoded():
    assert json.dumps({"a": Inner()}) == '{"a": {"v": 1}}'


def test_method_receives_instance_state():
    first = json.loads(json.dumps(Point(1, 2)))
    second = json.loads(json.dumps(Point(10, 20)))
    assert first == {"x": 1, "y": 2}
    assert second == {"x": 10, "y": 20}


def test_subclass_override_takes_precedence():
    assert json.loads(json.dumps(Child())) == {"kind": "child"}
    assert json.loads(json.dumps(Base())) == {"kind": "base"}


def test_inherited_implementation_is_used():
    assert json.loads(json.dumps(InheritsBase())) == {"kind": "base"}


def test_direct_method_call_returns_value():
    assert Point(3, 4)._as_json_friendly_obj() == {"x": 3, "y": 4}


def test_indent_option_still_works_with_jsonfriendly():
    assert json.dumps(Inner(), indent=2) == '{\n  "v": 1\n}'


def test_unsupported_type_still_raises_type_error():
    with pytest.raises(TypeError):
        json.dumps(object())


def test_unsupported_set_raises_type_error():
    with pytest.raises(TypeError):
        json.dumps({"s": set([1, 2])})


def test_unsupported_value_nested_in_returned_output_raises_type_error():
    class BadNested(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"s": set([1])}

    with pytest.raises(TypeError):
        json.dumps(BadNested())


def test_exception_from_method_propagates():
    with pytest.raises(ValueError):
        json.dumps(Failing())


def test_primitives_are_unaffected_by_patch():
    data = {"a": [1, "b", None, True, 2.5]}
    assert json.loads(json.dumps(data)) == data


def test_list_of_many_jsonfriendly_objects():
    items = [Point(i, i * 2) for i in range(3)]
    decoded = json.loads(json.dumps(items))
    assert decoded == [{"x": 0, "y": 0}, {"x": 1, "y": 2}, {"x": 2, "y": 4}]
