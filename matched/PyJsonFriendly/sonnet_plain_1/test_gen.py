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


class Named(JsonFriendly):
    def __init__(self, name, child=None):
        self.name = name
        self.child = child

    def _as_json_friendly_obj(self):
        return {"name": self.name, "child": self.child}


class ListLike(JsonFriendly):
    def __init__(self, items):
        self.items = items

    def _as_json_friendly_obj(self):
        return list(self.items)


class StrLike(JsonFriendly):
    def _as_json_friendly_obj(self):
        return "hello"


class DuckTyped:
    """Not a JsonFriendly subclass, but exposes the hook method."""

    def _as_json_friendly_obj(self):
        return {"duck": True}


class Plain:
    pass


class SubPoint(Point):
    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z = z

    def _as_json_friendly_obj(self):
        d = super()._as_json_friendly_obj()
        d["z"] = self.z
        return d


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


def test_dumps_simple_object():
    assert json.loads(json.dumps(Point(1, 2))) == {"x": 1, "y": 2}


def test_dumps_exact_string():
    assert json.dumps(Point(3, 4), sort_keys=True) == '{"x": 3, "y": 4}'


def test_dumps_list_returning_object():
    assert json.loads(json.dumps(ListLike([1, 2, 3]))) == [1, 2, 3]


def test_dumps_string_returning_object():
    assert json.dumps(StrLike()) == '"hello"'


def test_dumps_nested_friendly_objects():
    obj = Named("parent", Named("child", Point(5, 6)))
    assert json.loads(json.dumps(obj)) == {
        "name": "parent",
        "child": {"name": "child", "child": {"x": 5, "y": 6}},
    }


def test_dumps_friendly_objects_inside_containers():
    data = {"points": [Point(1, 1), Point(2, 2)], "n": 2}
    assert json.loads(json.dumps(data)) == {
        "points": [{"x": 1, "y": 1}, {"x": 2, "y": 2}],
        "n": 2,
    }


def test_subclass_override_with_super_call():
    assert json.loads(json.dumps(SubPoint(1, 2, 3))) == {"x": 1, "y": 2, "z": 3}


def test_duck_typed_object_is_serialized():
    assert json.loads(json.dumps(DuckTyped())) == {"duck": True}


def test_json_encoder_encode_and_dump_use_hook():
    assert json.loads(JSONEncoder().encode(Point(7, 8))) == {"x": 7, "y": 8}
    import io

    buf = io.StringIO()
    json.dump(Point(9, 10), buf)
    assert json.loads(buf.getvalue()) == {"x": 9, "y": 10}


def test_unserializable_object_still_raises_type_error():
    with pytest.raises(TypeError):
        json.dumps(Plain())


def test_unserializable_builtin_still_raises_type_error():
    with pytest.raises(TypeError):
        json.dumps({1, 2, 3})


def test_explicit_default_argument_takes_precedence():
    result = json.dumps(Plain(), default=lambda o: "custom")
    assert result == '"custom"'


def test_encoder_subclass_overriding_default_is_respected():
    class MyEncoder(JSONEncoder):
        def default(self, o):
            return "mine"

    assert json.dumps(Point(1, 2), cls=MyEncoder) == '"mine"'


def test_default_backup_is_set_after_serialization():
    json.dumps(Point(1, 2))
    assert hasattr(JSONEncoder, "default_backup")


def test_regular_types_unaffected():
    data = {"a": [1, 2.5, "s", None, True]}
    assert json.loads(json.dumps(data)) == data
