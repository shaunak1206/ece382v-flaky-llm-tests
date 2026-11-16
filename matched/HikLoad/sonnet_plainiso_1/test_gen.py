import copy
import importlib

import pytest

import hikload.config as config_module
from hikload.config import CONFIG


EXPECTED_KEYS = {"server", "cameras", "user", "password", "downloadPath", "debug"}


def test_config_is_a_dict():
    assert isinstance(CONFIG, dict)


def test_config_is_not_empty():
    assert len(CONFIG) > 0


def test_config_has_exactly_expected_keys():
    assert set(CONFIG.keys()) == EXPECTED_KEYS


def test_server_is_string():
    assert isinstance(CONFIG["server"], str)


def test_server_value():
    assert CONFIG["server"] == "192.168.1.239"


def test_server_is_valid_ipv4_address():
    parts = CONFIG["server"].split(".")
    assert len(parts) == 4
    for part in parts:
        assert part.isdigit()
        assert 0 <= int(part) <= 255


def test_cameras_is_list_of_strings():
    assert isinstance(CONFIG["cameras"], list)
    assert all(isinstance(c, str) for c in CONFIG["cameras"])


def test_cameras_values():
    assert CONFIG["cameras"] == ["101", "201"]


def test_cameras_are_numeric_and_unique():
    cameras = CONFIG["cameras"]
    assert all(c.isdigit() for c in cameras)
    assert len(cameras) == len(set(cameras))


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


def test_debug_is_bool():
    assert isinstance(CONFIG["debug"], bool)
    assert CONFIG["debug"] is True


def test_config_is_same_object_on_import():
    from hikload import config as cfg
    assert cfg.CONFIG is CONFIG


def test_reload_restores_defaults_after_mutation():
    original = copy.deepcopy(CONFIG)
    try:
        CONFIG["user"] = "someone"
        CONFIG["cameras"].append("301")
        assert CONFIG["user"] == "someone"
        reloaded = importlib.reload(config_module)
        assert reloaded.CONFIG["user"] == original["user"]
        assert reloaded.CONFIG["cameras"] == original["cameras"]
        assert reloaded.CONFIG == original
    finally:
        CONFIG.clear()
        CONFIG.update(original)
        importlib.reload(config_module)


def test_config_mutation_is_visible_through_module():
    original_user = CONFIG["user"]
    try:
        CONFIG["user"] = "tester"
        assert config_module.CONFIG["user"] == "tester"
    finally:
        CONFIG["user"] = original_user


def test_deepcopy_is_independent_of_original():
    clone = copy.deepcopy(CONFIG)
    clone["cameras"].append("999")
    clone["server"] = "10.0.0.1"
    assert "999" not in CONFIG["cameras"]
    assert CONFIG["server"] == "192.168.1.239"
