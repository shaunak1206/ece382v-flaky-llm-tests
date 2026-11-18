import json
from json import JSONEncoder

import pytest

from PyJsonFriendly.jsonfriendly import JsonFriendly


@pytest.fixture(autouse=True)
def restore_jsonencoder_default():
    original_default = JSONEncoder.default
    original_backup = getattr(JSONEncoder, "default_backup", None)
    had_backup = hasattr(JSONEncoder, "default_backup")
    yield
    JSONEncoder.default = original_default
    if had_backup:
        JSONEncoder.default_backup = original_backup
    elif hasattr(JSONEncoder, "default_backup"):
        delattr(JSONEncoder, "default_backup")


def test_jsonfriendly_is_abstract():
    with pytest.raises(TypeError):
        JsonFriendly()


def test_jsonencoder_default_is_patched():
    assert JSONEncoder.default is not None
    assert JSONEncoder.default != object.__dict__.get("default")
    assert callable(JSONEncoder.default)


def test_default_backup_attribute_is_created():
    assert hasattr(JSONEncoder, "default_backup")
    assert callable(JSONEncoder.default_backup)


def test_jsonfriendly_subclass_encodes_to_dict():
    class Foo(JsonFriendly):
        def __init__(self, value):
            self.value = value

        def _as_json_friendly_obj(self):
            return {"value": self.value}

    assert json.dumps(Foo(123)) == '{"value": 123}'


def test_jsonfriendly_subclass_encodes_to_list():
    class Bar(JsonFriendly):
        def __init__(self, items):
            self.items = items

        def _as_json_friendly_obj(self):
            return self.items

    assert json.dumps(Bar([1, 2, 3])) == "[1, 2, 3]"


def test_jsonfriendly_subclass_encodes_nested_objects():
    class Child(JsonFriendly):
        def __init__(self, name):
            self.name = name

        def _as_json_friendly_obj(self):
            return {"name": self.name}

    class Parent(JsonFriendly):
        def __init__(self, child):
            self.child = child

        def _as_json_friendly_obj(self):
            return {"child": self.child}

    assert json.dumps(Parent(Child("x"))) == '{"child": {"name": "x"}}'


def test_jsonfriendly_subclass_method_can_return_builtin_scalar():
    class Num(JsonFriendly):
        def _as_json_friendly_obj(self):
            return 42

    assert json.dumps(Num()) == "42"


def test_jsonfriendly_subclass_with_inherited_method_works():
    class Base(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"kind": "base"}

    class Derived(Base):
        pass

    assert json.dumps(Derived()) == '{"kind": "base"}'


def test_jsonencoder_default_sets_backup_when_called():
    class Foo(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"ok": True}

    encoder = JSONEncoder()
    assert "default_backup" in JSONEncoder.__dict__
    json.dumps(Foo(), cls=JSONEncoder)
    assert JSONEncoder.default_backup is not None
    assert callable(JSONEncoder.default_backup)
    assert encoder.default is not None


def test_non_jsonfriendly_object_raises_typeerror():
    class NotSerializable:
        pass

    with pytest.raises(TypeError):
        json.dumps(NotSerializable())


def test_jsonfriendly_object_with_missing_method_falls_back_and_raises():
    class Broken(JsonFriendly):
        pass

    with pytest.raises(TypeError):
        json.dumps(Broken())


def test_custom_encoder_still_handles_jsonfriendly_objects():
    class Foo(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"foo": "bar"}

    class CustomEncoder(JSONEncoder):
        pass

    assert json.dumps(Foo(), cls=CustomEncoder) == '{"foo": "bar"}'
