import importlib

import hikload.config as config_module


def test_config_is_a_dict():
    assert isinstance(config_module.CONFIG, dict)


def test_config_has_expected_keys():
    assert set(config_module.CONFIG.keys()) == {
        "server",
        "cameras",
        "user",
        "password",
        "downloadPath",
        "debug",
    }


def test_server_default_value():
    assert config_module.CONFIG["server"] == "192.168.1.239"


def test_user_default_value():
    assert config_module.CONFIG["user"] == "admin"


def test_password_default_value_is_empty_string():
    assert config_module.CONFIG["password"] == ""


def test_download_path_default_value():
    assert config_module.CONFIG["downloadPath"] == "./Downloads/"


def test_debug_default_value_is_true():
    assert config_module.CONFIG["debug"] is True


def test_cameras_is_a_list():
    assert isinstance(config_module.CONFIG["cameras"], list)


def test_cameras_default_values():
    assert config_module.CONFIG["cameras"] == ["101", "201"]


def test_cameras_are_strings():
    assert all(isinstance(camera, str) for camera in config_module.CONFIG["cameras"])


def test_config_contains_no_extra_keys():
    expected_keys = {
        "server",
        "cameras",
        "user",
        "password",
        "downloadPath",
        "debug",
    }
    assert len(config_module.CONFIG) == len(expected_keys)
    assert set(config_module.CONFIG) == expected_keys


def test_module_reimport_preserves_config_contents():
    reloaded = importlib.reload(config_module)
    assert reloaded.CONFIG == {
        "server": "192.168.1.239",
        "cameras": ["101", "201"],
        "user": "admin",
        "password": "",
        "downloadPath": "./Downloads/",
        "debug": True,
    }


def test_cameras_list_order_is_preserved():
    cameras = config_module.CONFIG["cameras"]
    assert cameras[0] == "101"
    assert cameras[1] == "201"


def test_download_path_looks_like_relative_directory():
    path = config_module.CONFIG["downloadPath"]
    assert path.startswith("./")
    assert path.endswith("/")


def test_server_looks_like_ipv4_address():
    parts = config_module.CONFIG["server"].split(".")
    assert len(parts) == 4
    assert all(part.isdigit() for part in parts)
