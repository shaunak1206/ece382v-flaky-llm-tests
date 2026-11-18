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


def test_jsonfriendly_is_abstract_and_cannot_instantiate():
    with pytest.raises(TypeError):
        JsonFriendly()


def test_subclass_without_override_is_still_abstract():
    class Incomplete(JsonFriendly):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_subclass_with_override_can_be_instantiated():
    class FriendlyThing(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"value": 1}

    obj = FriendlyThing()
    assert isinstance(obj, JsonFriendly)


def test_jsonencoder_default_is_patched_by_jsonfriendly():
    assert JSONEncoder.default is not None
    assert callable(JSONEncoder.default)
    assert JSONEncoder.default is not JSONEncoder.__dict__.get("default")


def test_jsonencoder_default_sets_default_backup_on_first_use():
    class FriendlyThing(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"value": 1}

    assert not hasattr(JSONEncoder, "default_backup")
    json.dumps(FriendlyThing())
    assert hasattr(JSONEncoder, "default_backup")
    assert callable(JSONEncoder.default_backup)


def test_jsonencoder_default_backup_refers_to_original_method():
    original_default = JSONEncoder.default_backup

    class FriendlyThing(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"value": 1}

    json.dumps(FriendlyThing())
    assert JSONEncoder.default_backup is original_default


def test_json_dumps_uses_as_json_friendly_obj_result():
    class FriendlyThing(JsonFriendly):
        def __init__(self, name):
            self.name = name

        def _as_json_friendly_obj(self):
            return {"name": self.name, "kind": "friendly"}

    result = json.dumps(FriendlyThing("alpha"), sort_keys=True)
    assert result == '{"kind": "friendly", "name": "alpha"}'


def test_json_dumps_encodes_nested_friendly_objects():
    class FriendlyThing(JsonFriendly):
        def __init__(self, value):
            self.value = value

        def _as_json_friendly_obj(self):
            return {"value": self.value}

    payload = {"items": [FriendlyThing(1), FriendlyThing(2)]}
    result = json.dumps(payload, sort_keys=True)
    assert result == '{"items": [{"value": 1}, {"value": 2}]}'


def test_json_dumps_uses_inherited_as_json_friendly_obj():
    class BaseFriendly(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"base": True}

    class ChildFriendly(BaseFriendly):
        pass

    result = json.dumps(ChildFriendly(), sort_keys=True)
    assert result == '{"base": true}'


def test_json_dumps_calls_instance_method_each_time():
    class CountingFriendly(JsonFriendly):
        def __init__(self):
            self.calls = 0

        def _as_json_friendly_obj(self):
            self.calls += 1
            return {"calls": self.calls}

    obj = CountingFriendly()
    first = json.dumps(obj, sort_keys=True)
    second = json.dumps(obj, sort_keys=True)

    assert first == '{"calls": 1}'
    assert second == '{"calls": 2}'
    assert obj.calls == 2


def test_json_dumps_plain_object_raises_type_error():
    class PlainObject:
        pass

    with pytest.raises(TypeError):
        json.dumps(PlainObject())


def test_json_dumps_works_with_custom_jsonencoder_subclass():
    class FriendlyThing(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"value": 42}

    class MyEncoder(JSONEncoder):
        pass

    result = json.dumps(FriendlyThing(), cls=MyEncoder, sort_keys=True)
    assert result == '{"value": 42}'


def test_jsonencoder_default_backup_is_reusable_original_function():
    class FriendlyThing(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"value": 99}

    json.dumps(FriendlyThing())
    backup = JSONEncoder.default_backup
    assert backup is not None

    with pytest.raises(TypeError):
        backup(object(), object())
