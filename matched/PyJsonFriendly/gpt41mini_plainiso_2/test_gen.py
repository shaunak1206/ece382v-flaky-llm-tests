import pytest
from json import dumps, JSONEncoder
from PyJsonFriendly.jsonfriendly import JsonFriendly
from abc import ABCMeta, abstractmethod


def test_jsonencoder_default_backup_is_callable():
    # Confirm that the JSONEncoder.default_backup exists and is callable after import
    assert hasattr(JSONEncoder, "default_backup")
    assert callable(JSONEncoder.default_backup)


def test_jsonencoder_default_backup_behavior_with_standard_obj():
    obj = 123
    # JSONEncoder.default_backup should behave like the original default for unsupported types
    # Here int is supported so base default raises TypeError.
    with pytest.raises(TypeError):
        JSONEncoder.default_backup(obj)


def test_jsonencoder_default_calls_as_json_friendly_obj():
    class MyJsonFriendly(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"foo": "bar"}

    my_obj = MyJsonFriendly()

    # The default encoder should use _as_json_friendly_obj when called by JSONEncoder.default
    result = JSONEncoder.default(JSONEncoder(), my_obj)
    assert result == {"foo": "bar"}


def test_jsonencoder_default_with_non_jsonfriendly_object_raises_typeerror():
    class Unrelated:
        pass

    obj = Unrelated()

    with pytest.raises(TypeError):
        JSONEncoder.default(JSONEncoder(), obj)


def test_subclass_must_implement_as_json_friendly_obj():
    # A subclass that does not implement _as_json_friendly_obj should raise TypeError on instantiation
    with pytest.raises(TypeError):
        class BadSub(JsonFriendly):
            pass


def test_subclass_implements_abstract_method_correctly():
    class GoodSub(JsonFriendly):
        def _as_json_friendly_obj(self):
            return "ok"

    instance = GoodSub()
    assert instance._as_json_friendly_obj() == "ok"


def test_jsonfriendly_is_abstract_meta():
    # JsonFriendly should use ABCMeta as metaclass
    assert isinstance(JsonFriendly, ABCMeta)


def test_jsonfriendly_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        JsonFriendly()


def test_jsonfriendly_as_json_friendly_obj_is_abstract():
    # The abstract method should raise error if called directly from base class
    class Dummy(JsonFriendly):
        def _as_json_friendly_obj(self):
            return super()._as_json_friendly_obj()

    inst = Dummy()
    with pytest.raises(TypeError):
        inst._as_json_friendly_obj()


def test_jsonencoder_default_preserves_existing_functionality_for_dict():
    data = {"a": 1, "b": 2}
    # dict is JSON serializable, default should not interfere and return the dict itself
    result = JSONEncoder.default(JSONEncoder(), data)
    assert result == data


def test_jsonencoder_default_serializing_object_returns_serializable_obj():
    class CFriendly(JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"key": 123}

    obj = CFriendly()
    encoded = dumps(obj)
    assert encoded == '{"key": 123}'


def test_jsonencoder_default_serializing_nested_jsonfriendly_objects():
    class Nested(JsonFriendly):
        def __init__(self, val):
            self.val = val

        def _as_json_friendly_obj(self):
            return {"val": self.val._as_json_friendly_obj()}

    class Inner(JsonFriendly):
        def _as_json_friendly_obj(self):
            return "inner"

    inner = Inner()
    nested = Nested(inner)
    encoded = dumps(nested)
    assert encoded == '{"val": "inner"}'


def test_jsonencoder_default_should_restore_original_behavior_after_assignment():
    # Save current backup
    original_backup = JSONEncoder.default_backup

    class Dummy(JsonFriendly):
        def _as_json_friendly_obj(self):
            return "dummy"

    dummy = Dummy()
    # Call default to confirm function output
    result = JSONEncoder.default(JSONEncoder(), dummy)
    assert result == "dummy"

    # Check backup is still same function reference
    assert JSONEncoder.default_backup == original_backup


def test_jsonencoder_default_calls_default_backup_if_no_as_json_friendly_obj(monkeypatch):
    class Dummy:
        pass

    dummy = Dummy()
    # forcibly remove _as_json_friendly_obj if present
    if hasattr(dummy.__class__, "_as_json_friendly_obj"):
        delattr(dummy.__class__, "_as_json_friendly_obj")

    # Backup the original default_backup function
    orig_backup = JSONEncoder.default_backup

    # Monkeypatch default_backup to a function that returns a known value
    monkeypatch.setattr(JSONEncoder, "default_backup", lambda self, o: "backup_return")

    result = JSONEncoder.default(JSONEncoder(), dummy)
    assert result == "backup_return"

    # Restore default_backup to original function after test
    monkeypatch.setattr(JSONEncoder, "default_backup", orig_backup)


def test_jsonencoder_default_with_builtin_type_set():
    # JSONEncoder default does not support set directly
    my_set = {1, 2, 3}
    with pytest.raises(TypeError):
        JSONEncoder().encode(my_set)


def test_jsonfriendly_subclass_with_non_dict_return_type():
    class ReturnList(JsonFriendly):
        def _as_json_friendly_obj(self):
            return [1, 2, 3]

    rl = ReturnList()
    encoded = dumps(rl)
    assert encoded == "[1, 2, 3]"


def test_jsonfriendly_subclass_with_str_return_type():
    class ReturnStr(JsonFriendly):
        def _as_json_friendly_obj(self):
            return "string"

    rs = ReturnStr()
    encoded = dumps(rs)
    assert encoded == '"string"'


def test_jsonencoder_default_behavior_for_object_with_as_json_friendly_obj_attribute_but_not_callable():
    class Bad:
        _as_json_friendly_obj = "not callable"

    obj = Bad()
    # Since _as_json_friendly_obj is not callable, default_backup should be called, resulting in TypeError
    with pytest.raises(TypeError):
        JSONEncoder.default(JSONEncoder(), obj)
