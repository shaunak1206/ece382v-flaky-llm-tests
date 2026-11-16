import json
from json import JSONEncoder

import pytest

from PyJsonFriendly.jsonfriendly import JsonFriendly


class Point(JsonFriendly):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def _as_json_friendly_obj(self):
        return {"x": self.x, "y": self.y}


class Name(JsonFriendly):
    def __init__(self, name):
        self.name = name

    def _as_json_friendly_obj(self):
        return self.name


class Line(JsonFriendly):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def _as_json_friendly_obj(self):
        return {"start": self.a, "end": self.b}


class ListLike(JsonFriendly):
    def __init__(self, *items):
        self.items = items

    def _as_json_friendly_obj(self):
        return list(self.items)


class Plain:
    pass


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        JsonFriendly()


def test_subclass_without_implementation_cannot_be_instantiated():
    class Incomplete(JsonFriendly):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_subclass_with_implementation_is_instantiable():
    p = Point(1, 2)
    assert isinstance(p, JsonFriendly)


def test_dumps_friendly_object_to_dict():
    assert json.loads(json.dumps(Point(1, 2))) == {"x": 1, "y": 2}


def test_dumps_friendly_object_to_string():
    assert json.dumps(Name("alice")) == '"alice"'


def test_dumps_friendly_object_to_list():
    assert json.dumps(ListLike(1, 2, 3)) == "[1, 2, 3]"


def test_friendly_objects_inside_list():
    result = json.loads(json.dumps([Point(1, 2), Point(3, 4)]))
    assert result == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]


def test_friendly_objects_as_dict_values():
    result = json.loads(json.dumps({"a": Name("bob"), "b": Point(0, 0)}))
    assert result == {"a": "bob", "b": {"x": 0, "y": 0}}


def test_nested_friendly_objects_are_encoded_recursively():
    line = Line(Point(0, 1), Point(2, 3))
    result = json.loads(json.dumps(line))
    assert result == {"start": {"x": 0, "y": 1}, "end": {"x": 2, "y": 3}}


def test_explicit_encoder_class_uses_patched_default():
    out = json.dumps(Point(5, 6), cls=JSONEncoder)
    assert json.loads(out) == {"x": 5, "y": 6}


def test_encoder_instance_encode_and_default():
    enc = JSONEncoder()
    assert json.loads(enc.encode(Point(7, 8))) == {"x": 7, "y": 8}
    assert enc.default(Point(7, 8)) == {"x": 7, "y": 8}


def test_unsupported_object_still_raises_type_error():
    with pytest.raises(TypeError):
        json.dumps(Plain())


def test_unsupported_object_inside_container_raises_type_error():
    with pytest.raises(TypeError):
        json.dumps({"ok": Point(1, 1), "bad": Plain()})


def test_builtin_types_unaffected():
    data = {"a": [1, 2.5, None, True], "b": "text"}
    assert json.loads(json.dumps(data)) == data


def test_default_backup_attribute_set_after_encoding_friendly_object():
    json.dumps(Point(1, 2))
    assert hasattr(JSONEncoder, "default_backup")


def test_subclass_override_changes_representation():
    class Point3(Point):
        def _as_json_friendly_obj(self):
            d = super()._as_json_friendly_obj()
            d["z"] = 0
            return d

    assert json.loads(json.dumps(Point3(1, 2))) == {"x": 1, "y": 2, "z": 0}
