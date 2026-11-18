import copy
import importlib

from hikload.config import CONFIG


def test_config_is_a_dict():
    assert isinstance(CONFIG, dict)


def test_config_has_expected_top_level_keys():
    assert set(CONFIG.keys()) == {
        "server",
        "cameras",
        "user",
        "password",
        "downloadPath",
        "debug",
    }


def test_server_value_is_expected():
    assert CONFIG["server"] == "192.168.1.239"


def test_server_value_is_string():
    assert isinstance(CONFIG["server"], str)


def test_cameras_value_is_list():
    assert isinstance(CONFIG["cameras"], list)


def test_cameras_value_is_expected():
    assert CONFIG["cameras"] == ["101", "201"]


def test_each_camera_id_is_string():
    assert all(isinstance(camera_id, str) for camera_id in CONFIG["cameras"])


def test_user_value_is_expected():
    assert CONFIG["user"] == "admin"


def test_user_value_is_string():
    assert isinstance(CONFIG["user"], str)


def test_password_is_empty_string():
    assert CONFIG["password"] == ""


def test_download_path_is_expected():
    assert CONFIG["downloadPath"] == "./Downloads/"


def test_debug_flag_is_true():
    assert CONFIG["debug"] is True


def test_debug_flag_is_boolean():
    assert isinstance(CONFIG["debug"], bool)


def test_config_can_be_copied_without_modifying_original():
    original = copy.deepcopy(CONFIG)
    copied = copy.deepcopy(CONFIG)

    copied["server"] = "10.0.0.1"
    copied["cameras"].append("301")
    copied["debug"] = False

    assert CONFIG == original
    assert copied != original


def test_reimporting_config_keeps_same_expected_values():
    module = importlib.import_module("hikload.config")
    reloaded = importlib.reload(module)

    assert reloaded.CONFIG["server"] == "192.168.1.239"
    assert reloaded.CONFIG["cameras"] == ["101", "201"]
    assert reloaded.CONFIG["user"] == "admin"
    assert reloaded.CONFIG["password"] == ""
    assert reloaded.CONFIG["downloadPath"] == "./Downloads/"
    assert reloaded.CONFIG["debug"] is True
