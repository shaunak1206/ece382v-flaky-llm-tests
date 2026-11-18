import pytest
from hikload import config


def test_config_is_dict():
    assert isinstance(config.CONFIG, dict), "CONFIG should be a dictionary"


def test_server_exists_and_is_str():
    assert "server" in config.CONFIG, "CONFIG should have a 'server' key"
    assert isinstance(config.CONFIG["server"], str), "'server' should be a string"


def test_server_value_is_as_expected():
    assert config.CONFIG["server"] == "192.168.1.239"


def test_cameras_exists_and_is_list():
    assert "cameras" in config.CONFIG, "CONFIG should have a 'cameras' key"
    assert isinstance(config.CONFIG["cameras"], list), "'cameras' should be a list"


def test_cameras_elements_are_strings():
    cameras = config.CONFIG.get("cameras", [])
    assert all(isinstance(cam, str) for cam in cameras), "All cameras should be strings"


def test_cameras_has_expected_values():
    assert config.CONFIG["cameras"] == ["101", "201"]


def test_user_exists_and_is_str():
    assert "user" in config.CONFIG, "CONFIG should have a 'user' key"
    assert isinstance(config.CONFIG["user"], str), "'user' should be a string"


def test_user_value_is_as_expected():
    assert config.CONFIG["user"] == "admin"


def test_password_exists_and_is_str():
    assert "password" in config.CONFIG, "CONFIG should have a 'password' key"
    assert isinstance(config.CONFIG["password"], str), "'password' should be a string"


def test_password_value_is_empty_string():
    assert config.CONFIG["password"] == ""


def test_download_path_exists_and_is_str():
    assert "downloadPath" in config.CONFIG, "CONFIG should have a 'downloadPath' key"
    assert isinstance(config.CONFIG["downloadPath"], str), "'downloadPath' should be a string"


def test_download_path_value_is_as_expected():
    assert config.CONFIG["downloadPath"] == "./Downloads/"


def test_debug_exists_and_is_bool():
    assert "debug" in config.CONFIG, "CONFIG should have a 'debug' key"
    assert isinstance(config.CONFIG["debug"], bool), "'debug' should be a boolean"


def test_debug_value_is_true():
    assert config.CONFIG["debug"] is True


def test_no_extra_keys_in_config():
    allowed_keys = {"server", "cameras", "user", "password", "downloadPath", "debug"}
    assert set(config.CONFIG.keys()) == allowed_keys, "CONFIG contains unexpected keys"


def test_cameras_list_not_empty():
    assert len(config.CONFIG["cameras"]) > 0, "cameras list should not be empty"


def test_server_ip_address_format():
    server = config.CONFIG["server"]
    parts = server.split(".")
    assert len(parts) == 4, "Server IP should have four parts separated by dots"
    for part in parts:
        assert part.isdigit(), f"Each part of IP '{part}' should be numeric"
        val = int(part)
        assert 0 <= val <= 255, f"IP segment '{part}' should be between 0 and 255"


def test_download_path_ends_with_slash():
    path = config.CONFIG["downloadPath"]
    assert path.endswith("/") or path.endswith("\\"), "downloadPath should end with a slash or backslash"


def test_modify_config_and_cleanup():
    original_user = config.CONFIG["user"]
    config.CONFIG["user"] = "testuser"
    assert config.CONFIG["user"] == "testuser"
    # revert changes
    config.CONFIG["user"] = original_user


def test_modify_cameras_and_cleanup():
    original_cameras = config.CONFIG["cameras"][:]
    config.CONFIG["cameras"].append("301")
    assert "301" in config.CONFIG["cameras"]
    # revert changes
    config.CONFIG["cameras"] = original_cameras


def test_modify_debug_and_cleanup():
    original_debug = config.CONFIG["debug"]
    config.CONFIG["debug"] = False
    assert config.CONFIG["debug"] is False
    # revert changes
    config.CONFIG["debug"] = original_debug
