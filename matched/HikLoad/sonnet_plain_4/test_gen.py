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


def test_server_is_non_empty_string():
    assert isinstance(CONFIG["server"], str)
    assert CONFIG["server"] != ""


def test_server_is_valid_ipv4_address():
    parts = CONFIG["server"].split(".")
    assert len(parts) == 4
    for part in parts:
        assert part.isdigit()
        assert 0 <= int(part) <= 255


def test_server_default_value():
    assert CONFIG["server"] == "192.168.1.239"


def test_cameras_is_list_of_strings():
    assert isinstance(CONFIG["cameras"], list)
    assert all(isinstance(c, str) for c in CONFIG["cameras"])


def test_cameras_default_values():
    assert CONFIG["cameras"] == ["101", "201"]


def test_cameras_are_numeric_and_unique():
    cams = CONFIG["cameras"]
    assert len(cams) == len(set(cams))
    assert all(c.isdigit() for c in cams)


def test_user_is_string_and_default():
    assert isinstance(CONFIG["user"], str)
    assert CONFIG["user"] == "admin"


def test_password_is_string_default_empty():
    assert isinstance(CONFIG["password"], str)
    assert CONFIG["password"] == ""


def test_download_path_is_string_with_default():
    assert isinstance(CONFIG["downloadPath"], str)
    assert CONFIG["downloadPath"] == "./Downloads/"


def test_download_path_ends_with_slash():
    assert CONFIG["downloadPath"].endswith("/")


def test_debug_is_bool_and_true():
    assert isinstance(CONFIG["debug"], bool)
    assert CONFIG["debug"] is True


def test_missing_key_raises_keyerror():
    with pytest.raises(KeyError):
        CONFIG["nonexistent"]


def test_module_exposes_same_object():
    assert config_module.CONFIG is CONFIG


def test_reload_restores_defaults_after_mutation():
    original = copy.deepcopy(CONFIG)
    try:
        CONFIG["user"] = "other"
        CONFIG["cameras"].append("301")
        assert CONFIG["user"] == "other"
        reloaded = importlib.reload(config_module)
        assert reloaded.CONFIG == original
        assert reloaded.CONFIG["user"] == "admin"
        assert reloaded.CONFIG["cameras"] == ["101", "201"]
    finally:
        # make sure the original module-level state is consistent for other tests
        importlib.reload(config_module)
        CONFIG.clear()
        CONFIG.update(copy.deepcopy(original))


def test_deepcopy_is_independent():
    clone = copy.deepcopy(CONFIG)
    clone["cameras"].append("999")
    clone["user"] = "someone"
    assert "999" not in CONFIG["cameras"]
    assert CONFIG["user"] == "admin"
