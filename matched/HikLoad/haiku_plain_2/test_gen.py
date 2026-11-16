import ipaddress
import json

import pytest

from hikload.config import CONFIG


EXPECTED_KEYS = {"server", "cameras", "user", "password", "downloadPath", "debug"}


def test_config_is_dict():
    assert isinstance(CONFIG, dict)


def test_config_has_exactly_expected_keys():
    assert set(CONFIG.keys()) == EXPECTED_KEYS


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_each_expected_key_is_present(key):
    assert key in CONFIG


def test_server_is_string():
    assert isinstance(CONFIG["server"], str)


def test_server_is_valid_ipv4_address():
    # Raises ValueError if the address is malformed.
    address = ipaddress.ip_address(CONFIG["server"])
    assert address.version == 4


def test_server_is_not_empty():
    assert CONFIG["server"].strip() != ""


def test_cameras_is_list():
    assert isinstance(CONFIG["cameras"], list)


def test_cameras_is_not_empty():
    assert len(CONFIG["cameras"]) > 0


def test_cameras_entries_are_numeric_strings():
    for camera in CONFIG["cameras"]:
        assert isinstance(camera, str)
        assert camera.isdigit()


def test_cameras_have_no_duplicates():
    cameras = CONFIG["cameras"]
    assert len(cameras) == len(set(cameras))


def test_user_is_non_empty_string():
    assert isinstance(CONFIG["user"], str)
    assert CONFIG["user"] != ""


def test_password_is_string():
    # An empty password is allowed by the default configuration.
    assert isinstance(CONFIG["password"], str)


def test_download_path_is_string_ending_with_separator():
    path = CONFIG["downloadPath"]
    assert isinstance(path, str)
    assert path.endswith("/")


def test_debug_is_boolean():
    assert isinstance(CONFIG["debug"], bool)


def test_config_is_json_serializable():
    serialized = json.dumps(CONFIG)
    assert json.loads(serialized) == CONFIG


def test_config_values_have_expected_defaults():
    assert CONFIG["server"] == "192.168.1.239"
    assert CONFIG["cameras"] == ["101", "201"]
    assert CONFIG["user"] == "admin"
    assert CONFIG["password"] == ""
    assert CONFIG["downloadPath"] == "./Downloads/"
    assert CONFIG["debug"] is True


def test_reimporting_config_returns_same_object():
    from hikload import config as config_module

    assert config_module.CONFIG is CONFIG
