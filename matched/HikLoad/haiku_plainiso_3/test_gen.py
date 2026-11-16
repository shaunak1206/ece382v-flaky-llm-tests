import ipaddress
import os

import pytest

from hikload.config import CONFIG


REQUIRED_KEYS = {"server", "cameras", "user", "password", "downloadPath", "debug"}


def test_config_is_dict():
    assert isinstance(CONFIG, dict)


def test_config_has_exactly_required_keys():
    assert set(CONFIG.keys()) == REQUIRED_KEYS


@pytest.mark.parametrize("key", sorted(REQUIRED_KEYS))
def test_required_key_present(key):
    assert key in CONFIG


def test_server_is_string():
    assert isinstance(CONFIG["server"], str)


def test_server_is_valid_ipv4_address():
    # Raises ValueError if the address is malformed.
    address = ipaddress.IPv4Address(CONFIG["server"])
    assert address.version == 4


def test_server_is_not_empty():
    assert CONFIG["server"].strip() != ""


def test_cameras_is_list():
    assert isinstance(CONFIG["cameras"], list)


def test_cameras_is_not_empty():
    assert len(CONFIG["cameras"]) > 0


def test_camera_entries_are_strings():
    for camera in CONFIG["cameras"]:
        assert isinstance(camera, str), f"camera {camera!r} is not a string"


def test_camera_entries_are_numeric_identifiers():
    for camera in CONFIG["cameras"]:
        assert camera.isdigit(), f"camera {camera!r} is not numeric"


def test_camera_entries_are_unique():
    cameras = CONFIG["cameras"]
    assert len(cameras) == len(set(cameras))


def test_user_is_nonempty_string():
    assert isinstance(CONFIG["user"], str)
    assert CONFIG["user"].strip() != ""


def test_password_is_string():
    assert isinstance(CONFIG["password"], str)


def test_download_path_is_relative_string_ending_with_separator():
    path = CONFIG["downloadPath"]
    assert isinstance(path, str)
    assert not os.path.isabs(path)
    assert path.endswith("/")


def test_debug_is_bool():
    assert isinstance(CONFIG["debug"], bool)
