import pytest
from hikload import config


def test_config_is_dict():
    assert isinstance(config.CONFIG, dict)


def test_config_server_key_exists():
    assert "server" in config.CONFIG


def test_config_server_value():
    assert config.CONFIG["server"] == "192.168.1.239"


def test_config_cameras_key_exists():
    assert "cameras" in config.CONFIG


def test_config_cameras_is_list():
    assert isinstance(config.CONFIG["cameras"], list)


def test_config_cameras_contents():
    cameras = config.CONFIG["cameras"]
    assert cameras == ["101", "201"]
    for cam in cameras:
        assert isinstance(cam, str)


def test_config_user_key_exists():
    assert "user" in config.CONFIG


def test_config_user_value():
    assert config.CONFIG["user"] == "admin"


def test_config_password_key_exists():
    assert "password" in config.CONFIG


def test_config_password_is_string():
    assert isinstance(config.CONFIG["password"], str)


def test_config_password_value_empty():
    # It is expected to be the empty string
    assert config.CONFIG["password"] == ""


def test_config_download_path_key_exists():
    assert "downloadPath" in config.CONFIG


def test_config_download_path_value():
    assert config.CONFIG["downloadPath"] == "./Downloads/"


def test_config_debug_key_exists():
    assert "debug" in config.CONFIG


def test_config_debug_value_is_bool():
    assert isinstance(config.CONFIG["debug"], bool)


def test_config_debug_value_true():
    assert config.CONFIG["debug"] is True


def test_no_extra_keys_in_config():
    expected_keys = {"server", "cameras", "user", "password", "downloadPath", "debug"}
    assert set(config.CONFIG.keys()) == expected_keys


def test_cameras_not_empty():
    assert len(config.CONFIG["cameras"]) > 0


def test_cameras_all_strings():
    assert all(isinstance(cam, str) for cam in config.CONFIG["cameras"])


def test_server_is_ip_address_format():
    server = config.CONFIG["server"]
    parts = server.split(".")
    assert len(parts) == 4
    for part in parts:
        assert part.isdigit()
        num = int(part)
        assert 0 <= num <= 255
