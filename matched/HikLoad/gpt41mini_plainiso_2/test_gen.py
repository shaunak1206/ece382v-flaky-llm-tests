import pytest
from hikload.config import CONFIG

def test_config_is_dict():
    assert isinstance(CONFIG, dict)

def test_config_contains_expected_keys():
    expected_keys = {"server", "cameras", "user", "password", "downloadPath", "debug"}
    assert set(CONFIG.keys()) == expected_keys

def test_server_value_is_correct():
    assert CONFIG["server"] == "192.168.1.239"

def test_server_value_is_string():
    assert isinstance(CONFIG["server"], str)

def test_cameras_is_list():
    assert isinstance(CONFIG["cameras"], list)

def test_cameras_contains_expected_ids():
    assert "101" in CONFIG["cameras"]
    assert "201" in CONFIG["cameras"]

def test_all_cameras_are_strings():
    assert all(isinstance(cam, str) for cam in CONFIG["cameras"])

def test_user_is_admin():
    assert CONFIG["user"] == "admin"

def test_user_is_string():
    assert isinstance(CONFIG["user"], str)

def test_password_is_string():
    assert isinstance(CONFIG["password"], str)

def test_password_is_empty_string():
    assert CONFIG["password"] == ""

def test_download_path_is_string():
    assert isinstance(CONFIG["downloadPath"], str)

def test_download_path_ends_with_slash():
    assert CONFIG["downloadPath"].endswith("/") or CONFIG["downloadPath"].endswith("\\")

def test_debug_is_boolean():
    assert isinstance(CONFIG["debug"], bool)

def test_debug_is_true():
    assert CONFIG["debug"] is True

def test_cameras_list_length():
    assert len(CONFIG["cameras"]) == 2

def test_no_extra_keys_in_config():
    # Confirm no keys outside expected keys
    extra_keys = set(CONFIG.keys()) - {"server", "cameras", "user", "password", "downloadPath", "debug"}
    assert len(extra_keys) == 0

def test_server_ip_format():
    parts = CONFIG["server"].split(".")
    assert len(parts) == 4
    assert all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)

def test_download_path_relative():
    # The path should be relative and start with ./
    assert CONFIG["downloadPath"].startswith("./")

def test_cameras_all_non_empty_strings():
    assert all(isinstance(cam, str) and len(cam) > 0 for cam in CONFIG["cameras"])

def test_user_non_empty_string():
    assert len(CONFIG["user"]) > 0

def test_password_can_be_empty_string():
    # Password string is allowed empty
    assert CONFIG["password"] == "" or isinstance(CONFIG["password"], str)
