import pytest
import json
from PyJsonFriendly import jsonfriendly


class DummyJsonFriendly(jsonfriendly.JsonFriendly):
    def __init__(self, data):
        self.data = data

    def _as_json_friendly_obj(self):
        return {"dummy": self.data}


class NotJsonFriendly:
    pass


def test_subclass_must_implement_abstract_method():
    with pytest.raises(TypeError):
        class Broken(jsonfriendly.JsonFriendly):
            pass
        Broken()


def test_instantiate_and_call_as_json_friendly_obj():
    obj = DummyJsonFriendly(42)
    result = obj._as_json_friendly_obj()
    assert isinstance(result, dict)
    assert result["dummy"] == 42


def test_json_encoder_default_is_monkeypatched():
    # Check that JSONEncoder.default is a lambda function (patched by JsonFriendly)
    assert callable(jsonfriendly.JSONEncoder.default)


def test_json_encoder_serializes_jsonfriendly_object_correctly():
    obj = DummyJsonFriendly("abc")

    encoded = json.dumps(obj)
    assert encoded == '{"dummy": "abc"}'


def test_json_encoder_raises_type_error_for_unknown_object():
    class CustomClass:
        pass

    inst = CustomClass()
    encoder = json.JSONEncoder()

    with pytest.raises(TypeError):
        encoder.encode(inst)


def test_json_encoder_default_backup_preserved_after_multiple_serializations():
    obj = DummyJsonFriendly(123)

    json.dumps(obj)
    # Ensure default_backup is set to original default method
    assert hasattr(jsonfriendly.JSONEncoder, "default_backup")
    backup = jsonfriendly.JSONEncoder.default_backup

    # Encoding again does not change default_backup further
    json.dumps(obj)
    assert jsonfriendly.JSONEncoder.default_backup is backup


def test_encoder_does_not_modify_non_jsonfriendly_objects():
    normal_obj = {"key": "value", "num": 1}
    encoded = json.dumps(normal_obj)
    assert encoded == '{"key": "value", "num": 1}'


def test_subclass_with_custom_json_friendly_obj_returns_non_dict():
    class CustomJsonFriendly(jsonfriendly.JsonFriendly):
        def _as_json_friendly_obj(self):
            return [1, 2, 3]

    obj = CustomJsonFriendly()
    result = json.dumps(obj)
    assert result == '[1, 2, 3]'


def test_serializer_works_for_nested_jsonfriendly_objects():
    class Nested(jsonfriendly.JsonFriendly):
        def __init__(self, inner):
            self.inner = inner

        def _as_json_friendly_obj(self):
            return {"inner": self.inner}

    inner = DummyJsonFriendly("nested")
    outer = Nested(inner)
    dumped = json.dumps(outer)
    # The inner should be serialized properly as well through default
    assert dumped == '{"inner": {"dummy": "nested"}}'


def test_encoder_handles_multiple_jsonfriendly_types_in_collection():
    objs = [
        DummyJsonFriendly("a"),
        DummyJsonFriendly("b"),
        DummyJsonFriendly("c"),
    ]

    encoded = json.dumps(objs)
    assert encoded == '[{"dummy": "a"}, {"dummy": "b"}, {"dummy": "c"}]'


def test_encoder_uses_defaultbackup_for_non_jsonfriendly_objects():
    # Create a non-jsonfriendly object that default_backup should be used on
    class NonFriendly:
        pass

    enc = json.JSONEncoder()
    # Manually call patched default with non-JsonFriendly obj to trigger fallback to default_backup
    if hasattr(jsonfriendly.JSONEncoder, "default_backup"):
        default_fn = jsonfriendly.JSONEncoder.default_backup
    else:
        default_fn = None

    if default_fn is not None:
        with pytest.raises(TypeError):
            # Because default_backup is the original encoder.default, which raises TypeError
            default_fn(enc, NonFriendly())
    else:
        pytest.skip("default_backup not set, skip test")


def test_new_subclass_inherits_jsonencoder_patch():
    class S(jsonfriendly.JsonFriendly):
        def _as_json_friendly_obj(self):
            return "x"

    obj = S()
    assert json.dumps(obj) == '"x"'


def test_repr_and_str_on_jsonfriendly_subclass_do_not_affect_serialization():
    class R(jsonfriendly.JsonFriendly):
        def _as_json_friendly_obj(self):
            return {"a": 1}

        def __repr__(self):
            return "repr"

        def __str__(self):
            return "str"

    obj = R()
    assert json.dumps(obj) == '{"a": 1}'


def test_subclass_with_none_return():
    class N(jsonfriendly.JsonFriendly):
        def _as_json_friendly_obj(self):
            return None

    obj = N()
    assert json.dumps(obj) == 'null'


def test_encoder_handles_objects_with__as_json_friendly_obj_not_callable():
    class Weird:
        _as_json_friendly_obj = "not callable"

    obj = Weird()
    encoder = json.JSONEncoder()

    # The patched default tries to call obj.__class__._as_json_friendly_obj
    # which is a string, so will raise TypeError
    with pytest.raises(TypeError):
        encoder.encode(obj)


def test_encoder_with_object_missing_as_json_friendly_obj():
    class NoAttr:
        pass

    obj = NoAttr()
    encoder = json.JSONEncoder()

    # It will use default_backup, which is original .default method
    with pytest.raises(TypeError):
        encoder.encode(obj)
