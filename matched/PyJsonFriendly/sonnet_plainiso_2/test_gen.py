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


class Wrapper(JsonFriendly):
    def __init__(self, inner):
        self.inner = inner

    def _as_json_friendly_obj(self):
        return {"inner": self.inner}


class ListLike(JsonFriendly):
    def __init__(self, *items):
        self.items = items

    def _as_json_friendly_obj(self):
        return list(self.items)


class Scalar(JsonFriendly):
    def _as_json_friendly_obj(self):
        return 42


class DuckTyped:
    """Does not inherit JsonFriendly but provides the hook method."""

    def _as_json_friendly_obj(self):
        return {"duck": True}


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


def test_abstract_method_is_registered():
    assert "_as_json_friendly_obj" in JsonFriendly.__abstractmethods__


def test_subclass_with_implementation_can_be_instantiated():
    p = Point(1, 2)
    assert isinstance(p, JsonFriendly)


def test_dumps_simple_object_to_dict():
    assert json.loads(json.dumps(Point(1, 2))) == {"x": 1, "y": 2}


def test_dumps_object_returning_list():
    assert json.dumps(ListLike(1, "a", None)) == '[1, "a", null]'


def test_dumps_object_returning_scalar():
    assert json.dumps(Scalar()) == "42"


def test_objects_nested_in_containers():
    data = {"points": [Point(0, 0), Point(1, 1)], "single": Point(5, 6)}
    result = json.loads(json.dumps(data))
    assert result == {
        "points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}],
        "single": {"x": 5, "y": 6},
    }


def test_friendly_object_nested_in_friendly_object():
    obj = Wrapper(Wrapper(Point(3, 4)))
    assert json.loads(json.dumps(obj)) == {"inner": {"inner": {"x": 3, "y": 4}}}


def test_dumps_with_indent_and_sort_keys():
    out = json.dumps(Point(2, 1), indent=2, sort_keys=True)
    assert json.loads(out) == {"x": 2, "y": 1}
    assert out.index('"x"') < out.index('"y"')
    assert "\n" in out


def test_inherited_hook_is_used_by_child_class():
    class Point3D(Point):
        def __init__(self, x, y, z):
            super().__init__(x, y)
            self.z = z

    assert json.loads(json.dumps(Point3D(1, 2, 3))) == {"x": 1, "y": 2}


def test_child_class_can_override_hook():
    class Point3D(Point):
        def __init__(self, x, y, z):
            super().__init__(x, y)
            self.z = z

        def _as_json_friendly_obj(self):
            return {"x": self.x, "y": self.y, "z": self.z}

    assert json.loads(json.dumps(Point3D(1, 2, 3))) == {"x": 1, "y": 2, "z": 3}


def test_duck_typed_object_is_serialized():
    assert json.loads(json.dumps(DuckTyped())) == {"duck": True}


def test_unsupported_object_still_raises_type_error():
    with pytest.raises(TypeError):
        json.dumps(Plain())


def test_unsupported_set_raises_type_error():
    with pytest.raises(TypeError):
        json.dumps({1, 2, 3})


def test_encoder_default_called_directly():
    enc = JSONEncoder()
    assert enc.default(Point(7, 8)) == {"x": 7, "y": 8}
    with pytest.raises(TypeError):
        enc.default(Plain())


def test_default_backup_attribute_set_after_use():
    json.dumps(Point(1, 1))
    assert hasattr(JSONEncoder, "default_backup")
    assert callable(JSONEncoder.default_backup)


def test_custom_encoder_subclass_inherits_patched_default():
    class MyEncoder(JSONEncoder):
        pass

    assert json.loads(MyEncoder().encode([Point(1, 2)])) == [{"x": 1, "y": 2}]
