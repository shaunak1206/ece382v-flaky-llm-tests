import pytest

from hikload.config import CONFIG


def test_config_is_a_dictionary():
    assert isinstance(CONFIG, dict)


def test_config_has_expected_keys():
    assert set(CONFIG.keys()) == {
        "server",
        "cameras",
        "user",
        "password",
        "downloadPath",
        "debug",
    }


def test_server_value():
    assert CONFIG["server"] == "192.168.1.239"


def test_server_is_string():
    assert isinstance(CONFIG["server"], str)


def test_cameras_value():
    assert CONFIG["cameras"] == ["101", "201"]


def test_cameras_is_list():
    assert isinstance(CONFIG["cameras"], list)


def test_camera_entries_are_strings():
    assert all(isinstance(camera, str) for camera in CONFIG["cameras"])


def test_user_value():
    assert CONFIG["user"] == "admin"


def test_user_is_string():
    assert isinstance(CONFIG["user"], str)


def test_password_value():
    assert CONFIG["password"] == ""


def test_download_path_value():
    assert CONFIG["downloadPath"] == "./Downloads/"


def test_debug_value_and_type():
    assert CONFIG["debug"] is True
    assert isinstance(CONFIG["debug"], bool)
