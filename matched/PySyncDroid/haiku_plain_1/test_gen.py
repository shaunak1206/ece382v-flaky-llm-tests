import os

import pytest

from pysyncdroid import exceptions
from pysyncdroid import sync as sync_module
from pysyncdroid.sync import (
    IGNORE,
    REMOVE,
    SYNCHRONIZE,
    Sync,
    readlink,
)

MTP_DETAILS = ("mtp://device/", "/run/user/1000/gvfs/mtp:host=device")
GVFS_PATH = MTP_DETAILS[1]


@pytest.fixture(autouse=True)
def no_bash(monkeypatch):
    """Make run_bash_cmd return an empty string so readlink falls back to its input."""
    monkeypatch.setattr(sync_module, "run_bash_cmd", lambda cmd: "")


@pytest.fixture
def gvfs_calls(monkeypatch):
    """Replace gvfs operations with recorders. Returns the list of recorded calls."""
    calls = []

    def make_recorder(name):
        def recorder(*args):
            calls.append((name,) + args)

        return recorder

    for name in ("mkdir", "cp", "rm", "mount"):
        monkeypatch.setattr(sync_module.gvfs, name, make_recorder(name))
    return calls


# ---------------------------------------------------------------- readlink


def test_readlink_empty_path_returns_input_without_running_command(monkeypatch):
    def fail(cmd):
        raise AssertionError("run_bash_cmd should not be called")

    monkeypatch.setattr(sync_module, "run_bash_cmd", fail)
    assert readlink("") == ""
    assert readlink(None) is None


def test_readlink_returns_command_output(monkeypatch):
    seen = []

    def fake_cmd(cmd):
        seen.append(cmd)
        return "/resolved/path"

    monkeypatch.setattr(sync_module, "run_bash_cmd", fake_cmd)
    assert readlink("some/../path") == "/resolved/path"
    assert seen == [["readlink", "-f", "some/../path"]]


def test_readlink_falls_back_to_input_when_command_returns_nothing(monkeypatch):
    monkeypatch.setattr(sync_module, "run_bash_cmd", lambda cmd: None)
    assert readlink("/some/path") == "/some/path"


def test_readlink_expands_tilde_before_resolving(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    seen = []
    monkeypatch.setattr(
        sync_module, "run_bash_cmd", lambda cmd: seen.append(cmd) or ""
    )
    result = readlink("~/music")
    expected = os.path.join(str(tmp_path), "music")
    assert result == expected
    assert seen == [["readlink", "-f", expected]]


# ---------------------------------------------------------------- Sync.__init__


def test_init_unpacks_mtp_details_and_sets_attributes():
    s = Sync(MTP_DETAILS, "/src", "/dst", verbose=True, overwrite_existing=True)
    assert s.mtp_url == MTP_DETAILS[0]
    assert s.mtp_gvfs_path == MTP_DETAILS[1]
    assert s.source == "/src"
    assert s.destination == "/dst"
    assert s.verbose is True
    assert s.overwrite_existing is True
    assert s.unmatched == IGNORE
    assert s.ignore_file_types is None


def test_init_lowercases_ignore_file_types():
    s = Sync(MTP_DETAILS, "/src", "/dst", ignore_file_types=["MP3", "Tmp"])
    assert s.ignore_file_types == ["mp3", "tmp"]


# ---------------------------------------------------------------- _verbose


def test_verbose_prints_only_when_enabled(capsys):
    Sync(MTP_DETAILS, "/src", "/dst", verbose=True)._verbose("hello")
    assert capsys.readouterr().out == "hello\n"

    Sync(MTP_DETAILS, "/src", "/dst", verbose=False)._verbose("hidden")
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------- gvfs_wrapper


def test_gvfs_wrapper_calls_function_with_args():
    received = []
    s = Sync(MTP_DETAILS, "/src", "/dst")
    s.gvfs_wrapper(lambda *args: received.append(args), "a", "b")
    assert received == [("a", "b")]


def test_gvfs_wrapper_remounts_and_retries_on_connection_reset(gvfs_calls):
    attempts = []

    def flaky(path):
        attempts.append(path)
        if len(attempts) == 1:
            raise exceptions.BashException("Error: Connection reset by peer")

    s = Sync(MTP_DETAILS, "/src", "/dst")
    s.gvfs_wrapper(flaky, "/x")

    assert attempts == ["/x", "/x"]
    assert ("mount", MTP_DETAILS[0]) in gvfs_calls


def test_gvfs_wrapper_reraises_other_bash_errors():
    def failing(path):
        raise exceptions.BashException("Permission denied")

    s = Sync(MTP_DETAILS, "/src", "/dst")
    with pytest.raises(exceptions.BashException):
        s.gvfs_wrapper(failing, "/x")


# ---------------------------------------------------------------- set_source_abs


def test_set_source_abs_keeps_existing_absolute_directory(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    s = Sync(MTP_DETAILS, str(src), "/dst")
    s.set_source_abs()
    assert s.source == str(src)


def test_set_source_abs_resolves_relative_path_against_device(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    gvfs_root = tmp_path / "gvfs"
    (gvfs_root / "Music").mkdir(parents=True)
    monkeypatch.chdir(cwd)

    s = Sync(("mtp://device/", str(gvfs_root)), "Music", "/dst")
    s.set_source_abs()
    assert s.source == os.path.join(str(gvfs_root), "Music")


def test_set_source_abs_raises_when_source_missing(tmp_path):
    s = Sync(MTP_DETAILS, str(tmp_path / "missing"), "/dst")
    with pytest.raises(OSError, match="does not exist"):
        s.set_source_abs()


def test_set_source_abs_raises_when_source_is_file(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("data")
    s = Sync(MTP_DETAILS, str(f), "/dst")
    with pytest.raises(OSError, match="is not a directory"):
        s.set_source_abs()


# ---------------------------------------------------------------- set_destination_abs


def test_set_destination_abs_joins_device_path_when_source_is_computer():
    s = Sync(MTP_DETAILS, "/home/user/pics", "Pictures")
    s.set_destination_abs()
    assert s.destination == os.path.join(GVFS_PATH, "Pictures")


def test_set_destination_abs_keeps_path_when_device_is_source():
    s = Sync(MTP_DETAILS, "mtp:host=device/Music", "/home/user/out")
    s.set_destination_abs()
    assert s.destination == "/home/user/out"


# ---------------------------------------------------------------- subdir helpers


def test_set_destination_subdir_abs_maps_nested_source_dir():
    s = Sync(MTP_DETAILS, "/src", "/dst")
    assert s.set_destination_subdir_abs("/src/a/b") == os.path.join(
        "/dst", "a", "b"
    )


def test_sync_data_template_strips_trailing_slashes():
    s = Sync(MTP_DETAILS, "/src", "/dst")
    data = s.sync_data_template("/src/a/", "/dst/a/")
    assert data == {
        "src_dir_abs": "/src/a",
        "dst_dir_abs": "/dst/a",
        "src_dir_fls": [],
        "dst_dir_fls": [],
    }


# ---------------------------------------------------------------- ignored types


def test_handle_ignored_file_type_is_case_insensitive_and_optional():
    s = Sync(MTP_DETAILS, "/src", "/dst", ignore_file_types=["mp3", "tmp"])
    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("song.MP3")
    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("/a/b/scratch.tmp")
    s.handle_ignored_file_type("notes.txt")
    s.handle_ignored_file_type("README")

    no_filter = Sync(MTP_DETAILS, "/src", "/dst")
    no_filter.handle_ignored_file_type("song.mp3")


def test_get_source_subdir_data_skips_ignored_files():
    s = Sync(MTP_DETAILS, "/src", "/dst", ignore_file_types=["mp3"])
    data = s.sync_data_template("/src", "/dst")
    s.get_source_subdir_data(["a.txt", "b.mp3"], data)
    assert data["src_dir_fls"] == [os.path.join("/src", "a.txt")]


# ---------------------------------------------------------------- destination data


def test_get_destination_subdir_data_creates_missing_directory(tmp_path, gvfs_calls):
    dst = tmp_path / "missing"
    s = Sync(MTP_DETAILS, str(tmp_path), "/dst")
    data = s.sync_data_template(str(tmp_path), str(dst))
    s.get_destination_subdir_data(data)
    assert ("mkdir", str(dst)) in gvfs_calls
    assert data["dst_dir_fls"] == []


def test_get_destination_subdir_data_lists_existing_non_ignored_files(tmp_path):
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "keep.txt").write_text("x")
    (dst / "skip.mp3").write_text("y")

    s = Sync(MTP_DETAILS, str(tmp_path), str(dst), ignore_file_types=["mp3"])
    data = s.sync_data_template(str(tmp_path), str(dst))
    s.get_destination_subdir_data(data)
    assert data["dst_dir_fls"] == [os.path.join(str(dst), "keep.txt")]


# ---------------------------------------------------------------- do_sync


def test_do_sync_respects_overwrite_flag(gvfs_calls):
    def make_data():
        return {
            "src_dir_abs": "/s",
            "dst_dir_abs": "/d",
            "src_dir_fls": ["/s/a"],
            "dst_dir_fls": ["/d/a"],
        }

    skip = make_data()
    Sync(MTP_DETAILS, "/s", "/d", overwrite_existing=False).do_sync(skip)
    assert gvfs_calls == []
    assert skip["dst_dir_fls"] == []

    overwrite = make_data()
    Sync(MTP_DETAILS, "/s", "/d", overwrite_existing=True).do_sync(overwrite)
    assert gvfs_calls == [("cp", "/s/a", "/d/a")]


# ---------------------------------------------------------------- end-to-end sync


def _prepare_pair(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "new.txt").write_text("new")
    (dst / "old.txt").write_text("old")
    return src, dst


def test_sync_ignore_copies_new_files_and_keeps_unmatched(tmp_path, gvfs_calls):
    src, dst = _prepare_pair(tmp_path)
    s = Sync(MTP_DETAILS, str(src), str(dst), unmatched=IGNORE)
    s.set_source_abs()
    s.set_destination_abs()
    s.sync()

    assert gvfs_calls == [("cp", str(src / "new.txt"), str(dst / "new.txt"))]


def test_sync_remove_deletes_unmatched_destination_files(tmp_path, gvfs_calls):
    src, dst = _prepare_pair(tmp_path)
    s = Sync(MTP_DETAILS, str(src), str(dst), unmatched=REMOVE)
    s.set_source_abs()
    s.set_destination_abs()
    s.sync()

    assert ("cp", str(src / "new.txt"), str(dst / "new.txt")) in gvfs_calls
    assert ("rm", str(dst / "old.txt")) in gvfs_calls
