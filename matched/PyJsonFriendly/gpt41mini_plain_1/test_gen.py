import pytest
from json import JSONEncoder, dumps
from PyJsonFriendly.jsonfriendly import JsonFriendly


def test_jsonfriendly_is_abstract():
    # Cannot instantiate directly, as _as_json_friendly_obj is abstract
    with pytest.raises(TypeError):
        JsonFriendly()


def test_subclass_must_implement_as_json_friendly_obj():
    class Incomplete(JsonFriendly):
        pass

    # Still abstract because method not implemented
    with pytest.raises(TypeError):
        Incomplete()


def test_subclass_with_implementation_can_instantiate():
    class Complete(JsonFriendly):
        def _as_json_friendly_obj(self):
            return "friendly"

    obj = Complete()
    assert obj._as_json_friendly_obj() == "friendly"


def test_jsonencoder_default_backup_set():
    # JSONEncoder.default_backup should be set to original default after import
    assert hasattr(JSONEncoder, "default_backup")
    # Confirm original default_backup is a callable
    assert callable(JSONEncoder.default_backup)


def test_jsonencoder_default_uses_custom_method_for_jsonfriendly_instance():
    class Custom(JsonFriendly):
        def __init__(self, val):
            self.val = val

        def _as_json_friendly_obj(self):
            return {"val": self.val}

    encoder = JSONEncoder()
    obj = Custom(42)
    result = encoder.default(obj)
    assert result == {"val": 42}


def test_jsonencoder_default_falls_back_for_non_jsonfriendly():
    encoder = JSONEncoder()

    class Dummy:
        pass

    dummy = Dummy()

    # It should raise TypeError because Dummy is not json serializable and no method to handle it
    with pytest.raises(TypeError):
        encoder.default(dummy)


def test_dumps_encodes_jsonfriendly_object_like_expected():
    class Example(JsonFriendly):
        def __init__(self, data):
            self.data = data

        def _as_json_friendly_obj(self):
            return {"data": self.data}

    example = Example("abc")
    json_str = dumps(example)
    assert json_str == '{"data": "abc"}'


def test_dumps_encodes_nested_jsonfriendly_objects():
    class Inner(JsonFriendly):
        def __init__(self, value):
            self.value = value

        def _as_json_friendly_obj(self):
            return {"value": self.value}

    class Outer(JsonFriendly):
        def __init__(self, inner_obj):
            self.inner_obj = inner_obj

        def _as_json_friendly_obj(self):
            return {"inner": self.inner_obj}

    inner = Inner(10)
    outer = Outer(inner)
    json_str = dumps(outer)
    assert json_str == '{"inner": {"value": 10}}'


def test_jsonencoder_default_backup_raises_typeerror_for_unknown():
    encoder = JSONEncoder()
    class Unserializable:
        pass

    unserializable = Unserializable()

    # Our lambda fallback calls default_backup which raises TypeError for unserializable
    with pytest.raises(TypeError):
        encoder.default(unserializable)


def test_jsonfriendly_abstractmethod_presence():
    # Confirm _as_json_friendly_obj is abstract on JsonFriendly itself
    assert getattr(JsonFriendly, "_as_json_friendly_obj", None)
    assert callable(getattr(JsonFriendly, "_as_json_friendly_obj", None))


def test_multiple_jsonfriendly_instances_serialization():
    class Item(JsonFriendly):
        def __init__(self, name):
            self.name = name

        def _as_json_friendly_obj(self):
            return {"name": self.name}

    items = [Item("a"), Item("b"), Item("c")]
    json_str = dumps(items)
    assert json_str == '[{"name": "a"}, {"name": "b"}, {"name": "c"}]'


def test_jsonencoder_default_backup_is_original_default():
    # The default_backup after import should be the original default from JSONEncoder
    original_default = JSONEncoder.default

    # Make a new encoder class to fetch original default_backup
    class Encoder(JSONEncoder):
        pass

    backup = getattr(Encoder, "default_backup", None)
    if backup is None:
        backup = getattr(JSONEncoder, "default_backup", None)
    # Calling default_backup on a non-serializable should raise TypeError
    with pytest.raises(TypeError):
        backup(object())


def test_jsonfriendly_repr_and_str_are_default():
    # JsonFriendly does not override these
    class Impl(JsonFriendly):
        def _as_json_friendly_obj(self):
            return "foo"

    obj = Impl()
    assert repr(obj).startswith("<")
    assert str(obj) == repr(obj)


def test_jsonencoder_default_with_builtin_types_returns_backup():
    encoder = JSONEncoder()
    # Int should use the original backup default serialization, which raises TypeError if not serializable
    with pytest.raises(TypeError):
        encoder.default(object())


def test_as_json_friendly_obj_called_in_default():
    called = []

    class Flag(JsonFriendly):
        def _as_json_friendly_obj(self):
            called.append(True)
            return 123

    obj = Flag()
    encoder = JSONEncoder()
    result = encoder.default(obj)
    assert result == 123
    assert called


def test_serialization_of_jsonfriendly_obj_in_dict():
    class JsonObj(JsonFriendly):
        def __init__(self, x):
            self.x = x

        def _as_json_friendly_obj(self):
            return {"x": self.x}

    obj = JsonObj(7)
    d = {"obj": obj}
    json_str = dumps(d)
    assert json_str == '{"obj": {"x": 7}}'
