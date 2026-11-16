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
    assert CONFIG["server"].strip() != ""


def test_server_is_valid_ipv4_address():
    parts = CONFIG["server"].split(".")
    assert len(parts) == 4
    for part in parts:
        assert part.isdigit()
        assert 0 <= int(part) <= 255


def test_server_default_value():
    assert CONFIG["server"] == "192.168.1.239"


def test_cameras_is_list_of_strings():
    cameras = CONFIG["cameras"]
    assert isinstance(cameras, list)
    assert all(isinstance(c, str) for c in cameras)


def test_cameras_default_values():
    assert CONFIG["cameras"] == ["101", "201"]


def test_cameras_are_numeric_and_unique():
    cameras = CONFIG["cameras"]
    assert len(cameras) > 0
    assert all(c.isdigit() for c in cameras)
    assert len(set(cameras)) == len(cameras)


def test_user_default():
    assert isinstance(CONFIG["user"], str)
    assert CONFIG["user"] == "admin"


def test_password_is_string_and_empty_by_default():
    assert isinstance(CONFIG["password"], str)
    assert CONFIG["password"] == ""


def test_download_path_is_string_with_trailing_slash():
    path = CONFIG["downloadPath"]
    assert isinstance(path, str)
    assert path == "./Downloads/"
    assert path.endswith("/")


def test_debug_is_bool():
    assert isinstance(CONFIG["debug"], bool)
    assert CONFIG["debug"] is True


def test_config_is_same_object_on_reimport():
    module = importlib.import_module("hikload.config")
    assert module.CONFIG is CONFIG
    assert config_module.CONFIG is CONFIG


def test_config_mutation_is_reflected_and_can_be_restored():
    original = copy.deepcopy(CONFIG)
    try:
        CONFIG["user"] = "other"
        CONFIG["cameras"].append("301")
        assert config_module.CONFIG["user"] == "other"
        assert "301" in config_module.CONFIG["cameras"]
    finally:
        CONFIG.clear()
        CONFIG.update(original)
    assert CONFIG == original
    assert CONFIG["user"] == "admin"
    assert CONFIG["cameras"] == ["101", "201"]


def test_config_deepcopy_is_independent():
    clone = copy.deepcopy(CONFIG)
    clone["cameras"].append("999")
    clone["server"] = "10.0.0.1"
    assert CONFIG["cameras"] == ["101", "201"]
    assert CONFIG["server"] == "192.168.1.239"


def test_missing_key_raises_keyerror():
    with pytest.raises(KeyError):
        CONFIG["nonexistent"]
