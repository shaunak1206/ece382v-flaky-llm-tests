import ipaddress
import json

import pytest

from hikload.config import CONFIG


EXPECTED_KEYS = {"server", "cameras", "user", "password", "downloadPath", "debug"}


def test_config_is_dict():
    assert isinstance(CONFIG, dict)


def test_config_has_expected_keys():
    assert set(CONFIG.keys()) == EXPECTED_KEYS


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_each_expected_key_is_present(key):
    assert key in CONFIG


def test_no_value_is_none():
    assert all(value is not None for value in CONFIG.values())


def test_server_is_string():
    assert isinstance(CONFIG["server"], str)


def test_server_is_valid_ipv4_address():
    # Raises ValueError if the address is malformed.
    address = ipaddress.IPv4Address(CONFIG["server"])
    assert isinstance(address, ipaddress.IPv4Address)


def test_server_is_private_network_address():
    assert ipaddress.IPv4Address(CONFIG["server"]).is_private


def test_user_is_nonempty_string():
    assert isinstance(CONFIG["user"], str)
    assert CONFIG["user"].strip() != ""


def test_password_is_string():
    assert isinstance(CONFIG["password"], str)


def test_cameras_is_list():
    assert isinstance(CONFIG["cameras"], list)


def test_cameras_list_is_nonempty():
    assert len(CONFIG["cameras"]) > 0


def test_camera_ids_are_strings():
    assert all(isinstance(camera_id, str) for camera_id in CONFIG["cameras"])


def test_camera_ids_are_numeric():
    assert all(camera_id.isdigit() for camera_id in CONFIG["cameras"])


def test_camera_ids_are_unique():
    cameras = CONFIG["cameras"]
    assert len(cameras) == len(set(cameras))


def test_camera_ids_match_expected_values():
    assert CONFIG["cameras"] == ["101", "201"]


def test_download_path_is_string():
    assert isinstance(CONFIG["downloadPath"], str)


def test_download_path_ends_with_separator():
    assert CONFIG["downloadPath"].endswith("/")


def test_debug_is_bool():
    assert isinstance(CONFIG["debug"], bool)


def test_config_is_json_serializable():
    serialized = json.dumps(CONFIG)
    assert json.loads(serialized) == CONFIG
