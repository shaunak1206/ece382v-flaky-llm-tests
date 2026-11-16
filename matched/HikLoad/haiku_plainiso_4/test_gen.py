import copy
import ipaddress

import pytest

from hikload.config import CONFIG


EXPECTED_KEYS = {"server", "cameras", "user", "password", "downloadPath", "debug"}


def test_config_is_dict():
    assert isinstance(CONFIG, dict)


def test_config_has_exactly_expected_keys():
    assert set(CONFIG.keys()) == EXPECTED_KEYS


def test_config_has_no_extra_keys():
    assert len(CONFIG) == len(EXPECTED_KEYS)


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_each_expected_key_present(key):
    assert key in CONFIG


def test_server_is_string():
    assert isinstance(CONFIG["server"], str)


def test_server_is_valid_ipv4_address():
    # Raises ValueError if the address is malformed.
    ipaddress.IPv4Address(CONFIG["server"])


def test_server_value():
    assert CONFIG["server"] == "192.168.1.239"


def test_cameras_is_list():
    assert isinstance(CONFIG["cameras"], list)


def test_cameras_not_empty():
    assert len(CONFIG["cameras"]) > 0


def test_cameras_elements_are_strings():
    assert all(isinstance(cam, str) for cam in CONFIG["cameras"])


def test_cameras_elements_are_numeric_ids():
    assert all(cam.isdigit() for cam in CONFIG["cameras"])


def test_cameras_are_unique():
    cameras = CONFIG["cameras"]
    assert len(cameras) == len(set(cameras))


def test_cameras_value():
    assert CONFIG["cameras"] == ["101", "201"]


def test_user_is_nonempty_string():
    assert isinstance(CONFIG["user"], str)
    assert CONFIG["user"].strip() != ""


def test_password_is_string():
    assert isinstance(CONFIG["password"], str)


def test_download_path_is_string_ending_with_separator():
    path = CONFIG["downloadPath"]
    assert isinstance(path, str)
    assert path.endswith("/")


def test_debug_is_bool():
    assert isinstance(CONFIG["debug"], bool)


def test_debug_value_is_true():
    assert CONFIG["debug"] is True


def test_config_can_be_deep_copied_without_loss():
    copied = copy.deepcopy(CONFIG)
    assert copied == CONFIG
    assert copied is not CONFIG
    assert copied["cameras"] is not CONFIG["cameras"]
