import json
from json import JSONEncoder

import pytest

from PyJsonFriendly.jsonfriendly import JsonFriendly


@pytest.fixture(autouse=True)
def restore_jsonencoder_default():
    original_default = JSONEncoder.default
    had_backup = hasattr(JSONEncoder, "default_backup")
    original_backup = getattr(JSONEncoder, "default_backup", None)
    yield
    JSONEncoder.default = original_default
    if had_backup:
        JSONEncoder.default_backup = original_backup
    elif hasattr(JSONEncoder, "default_backup"):
        delattr(JSONEncoder, "default_backup")


def test_jsonencoder_default_is_patched_to_callable():
    assert callable(JSONEncoder.default)
    assert JSONEncoder.default is not JSONEncoder.__dict__.get("default")


def test_jsonencoder_default_backup_not_present_before_use():
    assert not hasattr(JSONEncoder, "default_backup")


def test_abstract_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        JsonFriendly()


def test_subclass_without_implementation_is_still_abstract():
    class Incomplete(JsonFriendly):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_subclass_can_implement_method_and_return_dict():
    class Simple(JsonFriendly):
        def __init__(self, value):
            self.value = value

        def _as_json_friendly_obj(self):
            return {"value": self.value}

    obj = Simple(10)
    assert obj._as_json_friendly_obj() == {"value": 10}


def test_jsonencoder_default_serializes_custom_jsonfriendly_object():
    class Simple(JsonFriendly):
        def __init__(self, value):
            self.value = value

        def _as_json_friendly_obj(self):
            return {"value": self.value}

    encoded = json.dumps({"obj": Simple(7)}, default=JSONEncoder.default)
    assert encoded == '{"obj": {"value": 7}}'


def test_jsonencoder_default_calls_instance_method_with_self():
    calls = []

    class Tracker(JsonFriendly):
        def __init__(self, value):
            self.value = value

        def _as_json_friendly_obj(self):
            calls.append(self.value)
            return {"value": self.value}

    result = JSONEncoder().default(Tracker("abc"))
    assert result == {"value": "abc"}
    assert calls == ["abc"]


def test_jsonencoder_default_works_with_inherited_method():
    class Base(JsonFriendly):
        def __init__(self, value):
            self.value = value

        def _as_json_friendly_obj(self):
            return {"value": self.value}

    class Child(Base):
        pass

    assert json.dumps(Child(3), default=JSONEncoder.default) == '{"value": 3}'


def test_jsonencoder_default_backup_is_set_after_encoding():
    class Simple(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"ok": True}

    json.dumps(Simple(), default=JSONEncoder.default)
    assert hasattr(JSONEncoder, "default_backup")
    assert JSONEncoder.default_backup is JSONEncoder.default


def test_jsonencoder_default_backup_updates_to_current_default_on_each_call():
    class Simple(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"ok": True}

    json.dumps(Simple(), default=JSONEncoder.default)
    first_backup = JSONEncoder.default_backup
    json.dumps(Simple(), default=JSONEncoder.default)
    second_backup = JSONEncoder.default_backup

    assert first_backup is JSONEncoder.default
    assert second_backup is JSONEncoder.default


def test_jsonencoder_default_raises_recursionerror_for_unsupported_object():
    class Unsupported:
        pass

    with pytest.raises(RecursionError):
        json.dumps(Unsupported(), default=JSONEncoder.default)


def test_jsonencoder_default_preserves_nested_jsonfriendly_objects():
    class Leaf(JsonFriendly):
        def __init__(self, name):
            self.name = name

        def _as_json_friendly_obj(self):
            return {"name": self.name}

    class Parent(JsonFriendly):
        def __init__(self, leaf):
            self.leaf = leaf

        def _as_json_friendly_obj(self):
            return {"leaf": self.leaf}

    encoded = json.dumps(Parent(Leaf("x")), default=JSONEncoder.default)
    assert encoded == '{"leaf": {"name": "x"}}'


def test_jsonencoder_default_can_encode_multiple_distinct_instances():
    class Simple(JsonFriendly):
        def __init__(self, value):
            self.value = value

        def _as_json_friendly_obj(self):
            return {"value": self.value}

    payload = [Simple(1), Simple(2), Simple(3)]
    encoded = json.dumps(payload, default=JSONEncoder.default)
    assert encoded == '[{"value": 1}, {"value": 2}, {"value": 3}]'
