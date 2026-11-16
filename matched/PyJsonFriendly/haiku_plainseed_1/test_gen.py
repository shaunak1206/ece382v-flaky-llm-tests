import json

import pytest

from PyJsonFriendly.jsonfriendly import JsonFriendly


class Point(JsonFriendly):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def _as_json_friendly_obj(self):
        return {"x": self.x, "y": self.y}


class Constant(JsonFriendly):
    def __init__(self, value):
        self.value = value

    def _as_json_friendly_obj(self):
        return self.value


class Wrapper(JsonFriendly):
    def __init__(self, inner):
        self.inner = inner

    def _as_json_friendly_obj(self):
        return {"wrapped": self.inner}


class Base(JsonFriendly):
    def _as_json_friendly_obj(self):
        return {"kind": "base"}


class Child(Base):
    def _as_json_friendly_obj(self):
        return {"kind": "child"}


class GrandChild(Child):
    pass


class Duck:
    """Not a JsonFriendly subclass, but provides the same method."""

    def _as_json_friendly_obj(self):
        return {"duck": True}


class Failing(JsonFriendly):
    def _as_json_friendly_obj(self):
        raise ValueError("cannot serialize")


class MyEncoder(json.JSONEncoder):
    pass


def test_jsonfriendly_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        JsonFriendly()


def test_abstract_method_is_declared():
    assert "_as_json_friendly_obj" in JsonFriendly.__abstractmethods__


def test_subclass_without_implementation_cannot_be_instantiated():
    class Incomplete(JsonFriendly):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_subclass_with_implementation_can_be_instantiated():
    p = Point(1, 2)
    assert isinstance(p, JsonFriendly)


def test_dumps_uses_as_json_friendly_obj():
    assert json.loads(json.dumps(Point(1, 2))) == {"x": 1, "y": 2}


def test_dumps_output_matches_returned_dict_serialization():
    expected = json.dumps({"x": 5, "y": 6})
    assert json.dumps(Point(5, 6)) == expected


def test_method_receives_instance_state():
    assert json.loads(json.dumps(Point(1, 2))) != json.loads(json.dumps(Point(3, 4)))


@pytest.mark.parametrize(
    "value",
    [42, "text", None, True, [1, "a", None], {"nested": [1, 2]}, 3.5],
)
def test_returned_value_of_various_types_is_serialized(value):
    assert json.loads(json.dumps(Constant(value))) == value


def test_tuple_return_is_encoded_as_list():
    assert json.dumps(Constant((1, 2))) == "[1, 2]"


def test_friendly_objects_nested_in_list_and_dict():
    data = {"points": [Point(1, 2), Point(3, 4)], "single": Point(0, 0)}
    result = json.loads(json.dumps(data))
    assert result == {
        "points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
        "single": {"x": 0, "y": 0},
    }


def test_returned_friendly_object_is_recursively_encoded():
    wrapped = Wrapper(Point(7, 8))
    assert json.loads(json.dumps(wrapped)) == {"wrapped": {"x": 7, "y": 8}}


def test_subclass_override_takes_precedence():
    assert json.loads(json.dumps(Child())) == {"kind": "child"}


def test_inherited_implementation_is_used_by_grandchild():
    assert json.loads(json.dumps(GrandChild())) == {"kind": "child"}


def test_exception_from_method_propagates():
    with pytest.raises(ValueError, match="cannot serialize"):
        json.dumps(Failing())


def test_unsupported_plain_object_still_raises_type_error():
    with pytest.raises(TypeError):
        json.dumps(object())


def test_plain_json_types_are_unaffected():
    data = {"a": [1, 2, None], "b": "text", "c": True}
    assert json.dumps(data, sort_keys=True) == json.dumps(data, sort_keys=True)
    assert json.dumps({"a": [1, 2, None]}) == '{"a": [1, 2, null]}'


def test_duck_typed_class_with_method_is_serialized():
    assert json.loads(json.dumps(Duck())) == {"duck": True}


def test_custom_encoder_subclass_supports_friendly_objects():
    assert json.loads(json.dumps(Point(3, 4), cls=MyEncoder)) == {"x": 3, "y": 4}


def test_instance_attribute_does_not_shadow_class_method():
    p = Point(1, 2)
    p._as_json_friendly_obj = lambda: {"hacked": True}
    assert json.loads(json.dumps(p)) == {"x": 1, "y": 2}


def test_list_of_friendly_objects_preserves_order():
    items = [Point(i, i * 2) for i in range(5)]
    result = json.loads(json.dumps(items))
    assert result == [{"x": i, "y": i * 2} for i in range(5)]
