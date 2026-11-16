import io
import json
from abc import ABCMeta
from json import JSONEncoder

import pytest

from PyJsonFriendly.jsonfriendly import JsonFriendly


class Point(JsonFriendly):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def _as_json_friendly_obj(self):
        return {"x": self.x, "y": self.y}


class Line(JsonFriendly):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def _as_json_friendly_obj(self):
        return {"start": self.start, "end": self.end}


class Wrapper(JsonFriendly):
    """Returns another JsonFriendly object from its conversion method."""

    def __init__(self, inner):
        self.inner = inner

    def _as_json_friendly_obj(self):
        return self.inner


class ListLike(JsonFriendly):
    def __init__(self, *items):
        self.items = items

    def _as_json_friendly_obj(self):
        return list(self.items)


class PlainObject:
    pass


class DuckTyped:
    """Does not subclass JsonFriendly but has the hook method."""

    def _as_json_friendly_obj(self):
        return {"duck": True}


class SubPoint(Point):
    pass


class SubPointOverride(Point):
    def _as_json_friendly_obj(self):
        return {"kind": "sub", "x": self.x}


def test_metaclass_is_abcmeta():
    assert type(JsonFriendly) is ABCMeta


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        JsonFriendly()


def test_subclass_without_implementation_cannot_be_instantiated():
    class Incomplete(JsonFriendly):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_subclass_can_be_instantiated():
    p = Point(1, 2)
    assert isinstance(p, JsonFriendly)
    assert p._as_json_friendly_obj() == {"x": 1, "y": 2}


def test_dumps_simple_object():
    assert json.loads(json.dumps(Point(1, 2))) == {"x": 1, "y": 2}


def test_dumps_exact_string():
    assert json.dumps(Point(3, 4), sort_keys=True) == '{"x": 3, "y": 4}'


def test_dumps_nested_objects():
    line = Line(Point(0, 0), Point(5, 6))
    result = json.loads(json.dumps(line))
    assert result == {"start": {"x": 0, "y": 0}, "end": {"x": 5, "y": 6}}


def test_dumps_list_of_objects():
    data = [Point(1, 1), Point(2, 2), "text", 3]
    result = json.loads(json.dumps(data))
    assert result == [{"x": 1, "y": 1}, {"x": 2, "y": 2}, "text", 3]


def test_dumps_object_as_dict_value():
    result = json.loads(json.dumps({"p": Point(7, 8)}))
    assert result == {"p": {"x": 7, "y": 8}}


def test_conversion_returning_another_friendly_object():
    result = json.loads(json.dumps(Wrapper(Point(9, 10))))
    assert result == {"x": 9, "y": 10}


def test_conversion_returning_list():
    assert json.loads(json.dumps(ListLike(1, 2, 3))) == [1, 2, 3]


def test_inherited_subclass_uses_parent_conversion():
    assert json.loads(json.dumps(SubPoint(1, 2))) == {"x": 1, "y": 2}


def test_subclass_override_is_used():
    assert json.loads(json.dumps(SubPointOverride(1, 2))) == {"kind": "sub", "x": 1}


def test_json_dump_to_file_like():
    buf = io.StringIO()
    json.dump(Point(4, 5), buf)
    assert json.loads(buf.getvalue()) == {"x": 4, "y": 5}


def test_explicit_jsonencoder_instance():
    encoded = JSONEncoder().encode(Point(1, 2))
    assert json.loads(encoded) == {"x": 1, "y": 2}


def test_non_friendly_object_raises_type_error():
    with pytest.raises(TypeError):
        json.dumps(PlainObject())


def test_conversion_returning_unserializable_raises_type_error():
    class Bad(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {1, 2, 3}  # set is not serializable

    with pytest.raises(TypeError):
        json.dumps(Bad())


def test_duck_typed_object_is_serialized():
    assert json.loads(json.dumps(DuckTyped())) == {"duck": True}


def test_builtin_types_still_serialize():
    data = {"a": [1, 2.5, None, True], "b": "str"}
    assert json.loads(json.dumps(data)) == data


def test_default_backup_is_set_after_encoding():
    json.dumps(Point(1, 2))
    assert hasattr(JSONEncoder, "default_backup")


def test_custom_encoder_default_override_takes_precedence():
    class MyEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Point):
                return {"custom": True}
            return super().default(obj)

    assert json.loads(json.dumps(Point(1, 2), cls=MyEncoder)) == {"custom": True}
