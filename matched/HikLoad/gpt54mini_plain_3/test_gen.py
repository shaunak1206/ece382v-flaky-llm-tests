import copy

from hikload.config import CONFIG


def test_config_is_a_dict():
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


def test_cameras_contains_two_items():
    assert len(CONFIG["cameras"]) == 2


def test_each_camera_id_is_string():
    assert all(isinstance(camera_id, str) for camera_id in CONFIG["cameras"])


def test_user_value():
    assert CONFIG["user"] == "admin"


def test_user_is_string():
    assert isinstance(CONFIG["user"], str)


def test_password_value_is_empty_string():
    assert CONFIG["password"] == ""


def test_download_path_value():
    assert CONFIG["downloadPath"] == "./Downloads/"


def test_debug_value_is_true():
    assert CONFIG["debug"] is True


def test_debug_is_bool():
    assert isinstance(CONFIG["debug"], bool)


def test_config_copy_can_be_modified_without_affecting_original():
    config_copy = copy.deepcopy(CONFIG)
    config_copy["server"] = "10.0.0.1"
    config_copy["cameras"].append("301")

    assert CONFIG["server"] == "192.168.1.239"
    assert CONFIG["cameras"] == ["101", "201"]


def test_config_values_match_expected_types():
    assert isinstance(CONFIG["server"], str)
    assert isinstance(CONFIG["cameras"], list)
    assert isinstance(CONFIG["user"], str)
    assert isinstance(CONFIG["password"], str)
    assert isinstance(CONFIG["downloadPath"], str)
    assert isinstance(CONFIG["debug"], bool)
