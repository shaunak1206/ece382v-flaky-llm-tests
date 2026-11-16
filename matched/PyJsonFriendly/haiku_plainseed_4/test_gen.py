import json

import pytest

try:
    from jsonfriendly import JsonFriendly
except ImportError:
    from PyJsonFriendly.jsonfriendly import JsonFriendly


class Point(JsonFriendly):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def _as_json_friendly_obj(self):
        return {"x": self.x, "y": self.y}


class Labeled(JsonFriendly):
    def __init__(self, label):
        self.label = label

    def _as_json_friendly_obj(self):
        return self.label


class Wrapper(JsonFriendly):
    def __init__(self, inner):
        self.inner = inner

    def _as_json_friendly_obj(self):
        return {"inner": self.inner}


class Counting(JsonFriendly):
    calls = 0

    def _as_json_friendly_obj(self):
        Counting.calls += 1
        return {"n": 1}


class Failing(JsonFriendly):
    def _as_json_friendly_obj(self):
        raise ValueError("boom")


class BasePoint(JsonFriendly):
    def __init__(self, x):
        self.x = x

    def _as_json_friendly_obj(self):
        return {"kind": "base", "x": self.x}


class ChildPoint(BasePoint):
    def _as_json_friendly_obj(self):
        return {"kind": "child", "x": self.x}


class InheritedOnly(BasePoint):
    pass


class Unused(JsonFriendly):
    def _as_json_friendly_obj(self):
        return None


class TupleReturner(JsonFriendly):
    def _as_json_friendly_obj(self):
        return (1, 2, 3)


class NoneReturner(JsonFriendly):
    def _as_json_friendly_obj(self):
        return None


class StrReturner(JsonFriendly):
    def _as_json_friendly_obj(self):
        return "abc"


def test_base_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        JsonFriendly()


def test_abstract_method_is_declared_abstract():
    assert "_as_json_friendly_obj" in JsonFriendly.__abstractmethods__


def test_subclass_without_method_cannot_be_instantiated():
    class Missing(JsonFriendly):
        pass

    with pytest.raises(TypeError):
        Missing()


def test_subclass_with_method_can_be_instantiated():
    p = Point(1, 2)
    assert isinstance(p, JsonFriendly)
    assert p.x == 1 and p.y == 2


def test_dumps_uses_custom_method():
    result = json.loads(json.dumps(Point(1, 2)))
    assert result == {"x": 1, "y": 2}


def test_dumps_object_inside_list():
    result = json.loads(json.dumps([Point(1, 2), Point(3, 4)]))
    assert result == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]


def test_dumps_object_inside_dict():
    result = json.loads(json.dumps({"p": Point(5, 6)}))
    assert result == {"p": {"x": 5, "y": 6}}


def test_method_receives_instance_state():
    assert json.loads(json.dumps(Point(10, 20))) == {"x": 10, "y": 20}
    assert json.loads(json.dumps(Point(-1, 0))) == {"x": -1, "y": 0}


def test_method_returning_string_value():
    assert json.loads(json.dumps(Labeled("hello"))) == "hello"


def test_method_returning_none_encodes_as_null():
    assert json.dumps(NoneReturner()) == "null"


def test_method_returning_plain_string_encodes_as_json_string():
    assert json.dumps(StrReturner()) == '"abc"'


def test_method_returning_tuple_encodes_as_list():
    assert json.loads(json.dumps(TupleReturner())) == [1, 2, 3]


def test_nested_jsonfriendly_objects_are_encoded_recursively():
    obj = Wrapper(Point(7, 8))
    assert json.loads(json.dumps(obj)) == {"inner": {"x": 7, "y": 8}}


def test_method_called_once_per_encoded_object():
    Counting.calls = 0
    json.dumps(Counting())
    assert Counting.calls == 1


def test_subclass_override_takes_precedence():
    result = json.loads(json.dumps(ChildPoint(3)))
    assert result == {"kind": "child", "x": 3}


def test_inherited_method_is_used_when_not_overridden():
    result = json.loads(json.dumps(InheritedOnly(4)))
    assert result == {"kind": "base", "x": 4}


def test_exception_in_method_propagates():
    with pytest.raises(ValueError, match="boom"):
        json.dumps(Failing())


def test_plain_json_types_still_serialize():
    data = {"a": [1, 2, 3], "b": {"c": True, "d": None}, "e": 1.5}
    assert json.loads(json.dumps(data)) == data


def test_default_encoder_instance_uses_custom_method():
    assert json.loads(json.JSONEncoder().encode(Point(9, 9))) == {"x": 9, "y": 9}


def test_indent_option_still_uses_custom_method():
    text = json.dumps(Point(1, 2), indent=2)
    assert json.loads(text) == {"x": 1, "y": 2}
    assert "\n" in text


def test_multiple_classes_coexist():
    payload = [Point(1, 2), Labeled("tag"), Unused(), TupleReturner()]
    result = json.loads(json.dumps(payload))
    assert result == [{"x": 1, "y": 2}, "tag", None, [1, 2, 3]]
