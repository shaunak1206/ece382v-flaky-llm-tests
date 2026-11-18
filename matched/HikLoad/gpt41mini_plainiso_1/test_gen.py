import pytest
from hikload.config import CONFIG

def test_config_is_dict():
    assert isinstance(CONFIG, dict), "CONFIG should be a dictionary"

def test_config_has_server_key():
    assert "server" in CONFIG, "CONFIG missing 'server' key"

def test_server_value_is_string():
    assert isinstance(CONFIG.get("server"), str), "'server' should be a string"

def test_server_value_not_empty():
    assert CONFIG.get("server"), "'server' should not be empty"

def test_config_has_cameras_key():
    assert "cameras" in CONFIG, "CONFIG missing 'cameras' key"

def test_cameras_is_list():
    assert isinstance(CONFIG.get("cameras"), list), "'cameras' should be a list"

def test_cameras_list_not_empty():
    cameras = CONFIG.get("cameras")
    assert len(cameras) > 0, "'cameras' list should not be empty"

def test_each_camera_is_string():
    cameras = CONFIG.get("cameras")
    for camera in cameras:
        assert isinstance(camera, str), "Each camera in 'cameras' should be a string"

def test_config_has_user_key():
    assert "user" in CONFIG, "CONFIG missing 'user' key"

def test_user_is_string():
    assert isinstance(CONFIG.get("user"), str), "'user' should be a string"

def test_user_value_not_empty():
    user = CONFIG.get("user")
    assert user != "", "'user' should not be empty"

def test_config_has_password_key():
    assert "password" in CONFIG, "CONFIG missing 'password' key"

def test_password_is_string():
    assert isinstance(CONFIG.get("password"), str), "'password' should be a string"

def test_config_has_download_path_key():
    assert "downloadPath" in CONFIG, "CONFIG missing 'downloadPath' key"

def test_download_path_is_string():
    assert isinstance(CONFIG.get("downloadPath"), str), "'downloadPath' should be a string"

def test_download_path_is_relative_or_absolute():
    path = CONFIG.get("downloadPath")
    assert path.startswith("./") or path.startswith("/") or path == "", "'downloadPath' should be relative or absolute path"

def test_config_has_debug_key():
    assert "debug" in CONFIG, "CONFIG missing 'debug' key"

def test_debug_is_boolean():
    assert isinstance(CONFIG.get("debug"), bool), "'debug' should be a boolean"

def test_debug_value_true():
    # According to current CONFIG, debug is True. This test checks that expectation.
    assert CONFIG.get("debug") is True

def test_no_extra_keys_in_config():
    allowed_keys = {"server", "cameras", "user", "password", "downloadPath", "debug"}
    config_keys = set(CONFIG.keys())
    assert config_keys <= allowed_keys, f"CONFIG has unexpected keys: {config_keys - allowed_keys}"

def test_cameras_values_are_nonempty_strings():
    cameras = CONFIG.get("cameras")
    for cam in cameras:
        assert cam.strip() != "", "Camera id strings should not be empty or whitespace only"

def test_password_can_be_empty_string():
    # According to current config, password can be empty string
    password = CONFIG.get("password")
    assert isinstance(password, str)
    # Empty string allowed, so check no error for empty string
    assert password == "" or len(password) > 0

def test_download_path_ends_with_slash():
    path = CONFIG.get("downloadPath")
    # Expect downloadPath ends with slash for directory
    assert path.endswith("/"), "'downloadPath' should end with a slash"

def test_server_address_format():
    server = CONFIG.get("server")
    # simple validation for IPv4 format: four segments of digits separated by '.'
    parts = server.split('.')
    assert len(parts) == 4, "'server' should look like an IPv4 address"
    for p in parts:
        assert p.isdigit() and 0 <= int(p) <= 255, f"Segment '{p}' of server IP is invalid"

