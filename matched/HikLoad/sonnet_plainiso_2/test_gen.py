import copy
import importlib

import pytest

import hikload.config as config_module
from hikload.config import CONFIG


EXPECTED_KEYS = {"server", "cameras", "user", "password", "downloadPath", "debug"}


def test_config_is_dict():
    assert isinstance(CONFIG, dict)


def test_config_has_expected_keys():
    assert set(CONFIG.keys()) == EXPECTED_KEYS


def test_config_not_empty():
    assert len(CONFIG) > 0


def test_server_is_string():
    assert isinstance(CONFIG["server"], str)
    assert CONFIG["server"] != ""


def test_server_value():
    assert CONFIG["server"] == "192.168.1.239"


def test_server_looks_like_ipv4():
    parts = CONFIG["server"].split(".")
    assert len(parts) == 4
    for part in parts:
        assert part.isdigit()
        assert 0 <= int(part) <= 255


def test_cameras_is_list_of_strings():
    cameras = CONFIG["cameras"]
    assert isinstance(cameras, list)
    assert len(cameras) > 0
    assert all(isinstance(c, str) for c in cameras)


def test_cameras_default_values():
    assert CONFIG["cameras"] == ["101", "201"]


def test_cameras_are_numeric_and_unique():
    cameras = CONFIG["cameras"]
    assert all(c.isdigit() for c in cameras)
    assert len(set(cameras)) == len(cameras)


def test_user_value():
    assert isinstance(CONFIG["user"], str)
    assert CONFIG["user"] == "admin"


def test_password_is_string_and_default_empty():
    assert isinstance(CONFIG["password"], str)
    assert CONFIG["password"] == ""


def test_download_path_value_and_type():
    assert isinstance(CONFIG["downloadPath"], str)
    assert CONFIG["downloadPath"] == "./Downloads/"


def test_download_path_ends_with_slash():
    assert CONFIG["downloadPath"].endswith("/")


def test_debug_is_bool_true():
    assert isinstance(CONFIG["debug"], bool)
    assert CONFIG["debug"] is True


def test_config_same_object_after_reimport():
    again = importlib.import_module("hikload.config")
    assert again is config_module
    assert again.CONFIG is CONFIG


def test_config_reload_restores_defaults():
    original = copy.deepcopy(CONFIG)
    try:
        CONFIG["server"] = "10.0.0.1"
        CONFIG["cameras"].append("301")
        reloaded = importlib.reload(config_module)
        assert reloaded.CONFIG["server"] == original["server"]
        assert reloaded.CONFIG["cameras"] == original["cameras"]
    finally:
        # restore the original module-level object state for other tests
        CONFIG.clear()
        CONFIG.update(original)
        importlib.reload(config_module)


def test_config_is_mutable_dict_copy_independent():
    snapshot = copy.deepcopy(CONFIG)
    snapshot["cameras"].append("999")
    snapshot["user"] = "other"
    assert CONFIG["cameras"] == ["101", "201"]
    assert CONFIG["user"] == "admin"


def test_missing_key_raises_keyerror():
    with pytest.raises(KeyError):
        CONFIG["nonexistent"]


def test_config_values_have_expected_types():
    expected_types = {
        "server": str,
        "cameras": list,
        "user": str,
        "password": str,
        "downloadPath": str,
        "debug": bool,
    }
    for key, typ in expected_types.items():
        assert isinstance(CONFIG[key], typ), key
