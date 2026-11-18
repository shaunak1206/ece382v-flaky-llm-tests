import pytest
from json import JSONEncoder
from PyJsonFriendly.jsonfriendly import JsonFriendly


def test_jsonfriendly_is_abstract():
    # You cannot instantiate JsonFriendly because it has an abstract method
    with pytest.raises(TypeError):
        JsonFriendly()


class DummyJsonFriendly(JsonFriendly):
    def _as_json_friendly_obj(self):
        return {"dummy": True}


def test_subclass_can_instantiate():
    obj = DummyJsonFriendly()
    assert isinstance(obj, DummyJsonFriendly)


def test_as_json_friendly_obj_method():
    obj = DummyJsonFriendly()
    result = obj._as_json_friendly_obj()
    assert isinstance(result, dict)
    assert result.get("dummy") is True


def test_jsonencoder_default_backup_set_after_first_call():
    # Before serialization, default_backup attribute is not set
    if hasattr(JSONEncoder, "default_backup"):
        delattr(JSONEncoder, "default_backup")

    obj = DummyJsonFriendly()
    encoder = JSONEncoder()
    # Trigger JSONEncoder.default with a DummyJsonFriendly instance
    try:
        encoder.default(obj)
    except TypeError:
        # In case default fails downstream, ignore error here, as we only test attribute setting
        pass
    # After call, default_backup should be set and be a callable
    assert hasattr(JSONEncoder, "default_backup")
    assert callable(JSONEncoder.default_backup)


def test_jsonencoder_default_calls_as_json_friendly_obj(monkeypatch):
    obj = DummyJsonFriendly()
    called = {}

    def fake_as_json_friendly(self):
        called["yes"] = True
        return {"fake": 123}

    # Patch the class method _as_json_friendly_obj
    setattr(DummyJsonFriendly, "_as_json_friendly_obj", fake_as_json_friendly)
    encoder = JSONEncoder()
    result = encoder.default(obj)
    assert called.get("yes") is True
    assert result == {"fake": 123}


def test_jsonencoder_default_falls_back_to_default_backup(monkeypatch):
    class Unrelated:
        pass

    # Remove default_backup if exists
    if hasattr(JSONEncoder, "default_backup"):
        delattr(JSONEncoder, "default_backup")

    # Save original default to restore later
    orig_default = JSONEncoder.default

    # Setup a dummy default_backup that returns a special string
    def dummy_default_backup(self, obj):
        return "backup"

    JSONEncoder.default_backup = dummy_default_backup

    # Patch JSONEncoder.default to simulate fallback call when obj has no _as_json_friendly_obj
    obj = Unrelated()
    encoder = JSONEncoder()

    # Because the default method tries getattr(obj.__class__, "_as_json_friendly_obj", fallback)(obj)
    # and Unrelated has no _as_json_friendly_obj, it calls default_backup which is dummy_default_backup.
    result = encoder.default(obj)
    assert result == "backup"

    # Clean up
    delattr(JSONEncoder, "default_backup")
    JSONEncoder.default = orig_default


def test_dummyjsonfriendly_str_repr():
    obj = DummyJsonFriendly()
    s = str(obj)
    r = repr(obj)
    # Just check the methods work and return strings
    assert isinstance(s, str)
    assert isinstance(r, str)


def test_jsonencoder_default_with_builtin_type():
    encoder = JSONEncoder()
    # built-in int has no _as_json_friendly_obj so default_backup should be used
    # Set default_backup for the test
    JSONEncoder.default_backup = JSONEncoder.default

    value = 123
    # This should raise TypeError, because JSONEncoder does not support ints in default method
    # Actually, int serialization uses a different code path, so default() usually not called for int,
    # but let's call default explicitly to test fallback.
    with pytest.raises(TypeError):
        encoder.default(value)

    # clean up the attribute
    delattr(JSONEncoder, "default_backup")


def test_jsonfriendly_abstract_method_signature():
    # The abstract method _as_json_friendly_obj has no arguments except self
    assert "self" in JsonFriendly._as_json_friendly_obj.__code__.co_varnames


def test_jsonencoder_default_does_not_set_default_backup_again(monkeypatch):
    # Setup a dummy DummyJsonFriendly with tracker
    calls = {"count": 0}

    class MyJsonFriendly(JsonFriendly):
        def _as_json_friendly_obj(self_inner):
            calls["count"] += 1
            return {"called": calls["count"]}

    obj = MyJsonFriendly()

    # Remove attribute if exists
    if hasattr(JSONEncoder, "default_backup"):
        delattr(JSONEncoder, "default_backup")

    encoder = JSONEncoder()
    # First call sets default_backup
    encoder.default(obj)
    assert hasattr(JSONEncoder, "default_backup")
    count_after_first = calls["count"]

    # Save current default_backup
    default_backup_attr = JSONEncoder.default_backup

    # Call again, it should NOT reset default_backup but call _as_json_friendly_obj
    encoder.default(obj)
    # default_backup attribute must still be the same (not replaced again)
    assert JSONEncoder.default_backup is default_backup_attr
    # _as_json_friendly_obj count will have increased by one more call
    assert calls["count"] > count_after_first


def test_dummyjsonfriendly_multiple_times_in_jsonencoder(monkeypatch):
    class MyJsonFriendly(JsonFriendly):
        def __init__(self, val):
            self.val = val

        def _as_json_friendly_obj(self):
            return {"val": self.val}

    objs = [MyJsonFriendly(i) for i in range(3)]
    encoder = JSONEncoder()

    results = [encoder.default(obj) for obj in objs]

    expected = [{"val": 0}, {"val": 1}, {"val": 2}]
    assert results == expected


def test_jsonfriendly_inheritance_chain():
    # Create a subclass of DummyJsonFriendly and override _as_json_friendly_obj
    class SubDummy(DummyJsonFriendly):
        def _as_json_friendly_obj(self):
            return {"subdummy": True}

    obj = SubDummy()
    encoder = JSONEncoder()
    result = encoder.default(obj)
    assert result == {"subdummy": True}


def test_abstractmethod_presence():
    # Check that _as_json_friendly_obj is an abstract method of JsonFriendly
    am = getattr(JsonFriendly, "_as_json_friendly_obj")
    assert hasattr(am, "__isabstractmethod__")
    assert am.__isabstractmethod__ is True
