# -*- coding: utf-8 -*-

"""Unit tests for pysyncdroid.sync and pysyncdroid package initialisation."""

import os

import pytest

from pysyncdroid import exceptions
from pysyncdroid import sync as sync_mod
from pysyncdroid.sync import IGNORE, REMOVE, SYNCHRONIZE, Sync, readlink


MTP = ("mtp://device/", "/gvfs/mtp")


@pytest.fixture(autouse=True)
def no_bash(monkeypatch):
    """Make readlink fall back to its input path by default (no real shell)."""
    monkeypatch.setattr(sync_mod, "run_bash_cmd", lambda cmd: "")


@pytest.fixture
def calls(monkeypatch):
    """Replace gvfs operations with recorders and expose the recorded calls."""
    recorded = {"mkdir": [], "cp": [], "rm": [], "mount": []}

    def make_recorder(name):
        def fake(*args):
            recorded[name].append(args)

        return fake

    for name in recorded:
        monkeypatch.setattr(sync_mod.gvfs, name, make_recorder(name))
    return recorded


def make_sync(source="/src", destination="/dst", **kwargs):
    return Sync(MTP, source, destination, **kwargs)


# ---------------------------------------------------------------- readlink


def test_readlink_returns_input_for_empty_path():
    assert readlink("") == ""
    assert readlink(None) is None


def test_readlink_uses_readlink_command_output(monkeypatch):
    seen = []
    monkeypatch.setattr(
        sync_mod, "run_bash_cmd", lambda cmd: seen.append(cmd) or "/resolved"
    )

    assert readlink("link") == "/resolved"
    assert seen == [["readlink", "-f", "link"]]


def test_readlink_expands_home_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    seen = []
    monkeypatch.setattr(
        sync_mod, "run_bash_cmd", lambda cmd: seen.append(cmd) or ""
    )

    expected = os.path.join(str(tmp_path), "Music")
    assert readlink("~/Music") == expected
    assert seen == [["readlink", "-f", expected]]


# ------------------------------------------------------------ Sync init/ignore


def test_init_stores_values_and_lowercases_ignored_types():
    s = Sync(
        MTP,
        "/src",
        "/dst",
        unmatched=REMOVE,
        overwrite_existing=True,
        ignore_file_types=["MP3", "Txt"],
        verbose=True,
    )
    assert s.mtp_url == "mtp://device/"
    assert s.mtp_gvfs_path == "/gvfs/mtp"
    assert s.source == "/src"
    assert s.destination == "/dst"
    assert s.unmatched == REMOVE
    assert s.overwrite_existing is True
    assert s.ignore_file_types == ["mp3", "txt"]
    assert s.verbose is True


def test_handle_ignored_file_type_is_case_insensitive():
    s = make_sync(ignore_file_types=["mp3"])
    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("/x/Song.Mp3")
    s.handle_ignored_file_type("/x/photo.jpg")
    s.handle_ignored_file_type("/x/README")


# ------------------------------------------------------------ gvfs_wrapper


def test_gvfs_wrapper_calls_function_with_arguments():
    received = []
    make_sync().gvfs_wrapper(lambda *args: received.append(args), "x", "y")
    assert received == [("x", "y")]


def test_gvfs_wrapper_remounts_and_retries_on_connection_reset(monkeypatch):
    mounts = []
    monkeypatch.setattr(sync_mod.gvfs, "mount", lambda url: mounts.append(url))

    attempts = []

    def flaky(*args):
        attempts.append(args)
        if len(attempts) == 1:
            raise exceptions.BashException("cp: Connection reset by peer")

    make_sync().gvfs_wrapper(flaky, "a", "b")

    assert mounts == ["mtp://device/"]
    assert attempts == [("a", "b"), ("a", "b")]


def test_gvfs_wrapper_reraises_other_bash_errors(monkeypatch):
    mounts = []
    monkeypatch.setattr(sync_mod.gvfs, "mount", lambda url: mounts.append(url))

    def failing(*args):
        raise exceptions.BashException("permission denied")

    with pytest.raises(exceptions.BashException):
        make_sync().gvfs_wrapper(failing)
    assert mounts == []


# ------------------------------------------------------- source / destination


def test_set_source_abs_resolves_relative_source_against_cwd(tmp_path, monkeypatch):
    (tmp_path / "music").mkdir()
    monkeypatch.chdir(tmp_path)

    s = make_sync(source="music")
    s.set_source_abs()

    assert s.source == os.path.join(os.getcwd(), "music")


def test_set_source_abs_falls_back_to_device_path(tmp_path, monkeypatch):
    device_root = tmp_path / "device"
    (device_root / "Pictures").mkdir(parents=True)
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    s = Sync(("mtp://device/", str(device_root)), "Pictures", "/dst")
    s.set_source_abs()

    assert s.source == os.path.join(str(device_root), "Pictures")


def test_set_source_abs_raises_when_source_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Sync(("mtp://device/", str(tmp_path / "nodevice")), "nope", "/dst")

    with pytest.raises(OSError, match="does not exist"):
        s.set_source_abs()


def test_set_source_abs_raises_when_source_is_file(tmp_path, monkeypatch):
    (tmp_path / "file.txt").write_text("content")
    monkeypatch.chdir(tmp_path)

    s = make_sync(source="file.txt")

    with pytest.raises(OSError, match="is not a directory"):
        s.set_source_abs()


def test_set_destination_abs_depends_on_where_source_lives():
    s = make_sync(destination="/home/user/out")
    s.source = "mtp:host=device/Music"
    s.set_destination_abs()
    assert s.destination == "/home/user/out"

    s2 = make_sync(destination="Backup")
    s2.source = "/home/user/Music"
    s2.set_destination_abs()
    assert s2.destination == os.path.join("/gvfs/mtp", "Backup")


def test_set_destination_subdir_abs_mirrors_relative_path():
    s = make_sync(source="/a/src", destination="/b/dst")

    assert s.set_destination_subdir_abs("/a/src/sub/deep") == "/b/dst/sub/deep"
    assert s.set_destination_subdir_abs("/a/src") == os.path.join("/b/dst", "")


# ------------------------------------------------------- subdir data collection


def test_get_source_subdir_data_skips_ignored_types():
    s = make_sync(ignore_file_types=["mp3"])
    sync_data = {"src_dir_abs": "/src/album", "src_dir_fls": []}

    s.get_source_subdir_data(["track.mp3", "cover.jpg"], sync_data)

    assert sync_data["src_dir_fls"] == ["/src/album/cover.jpg"]


def test_get_destination_subdir_data_creates_missing_dir(tmp_path, calls):
    dst = tmp_path / "new"
    sync_data = {"dst_dir_abs": str(dst), "dst_dir_fls": []}

    make_sync().get_destination_subdir_data(sync_data)

    assert calls["mkdir"] == [(str(dst),)]
    assert sync_data["dst_dir_fls"] == []


def test_get_destination_subdir_data_lists_existing_files(tmp_path):
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "old.txt").write_text("x")
    (dst / "old.mp3").write_text("y")

    s = make_sync(ignore_file_types=["mp3"])
    sync_data = {"dst_dir_abs": str(dst), "dst_dir_fls": []}

    s.get_destination_subdir_data(sync_data)

    assert sync_data["dst_dir_fls"] == [os.path.join(str(dst), "old.txt")]


def test_get_sync_data_walks_tree_and_skips_empty_dirs(tmp_path, calls):
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "empty").mkdir()
    (src / "top.txt").write_text("t")
    (src / "a" / "one.txt").write_text("1")
    dst = tmp_path / "dst"

    s = make_sync(source=str(src), destination=str(dst))
    result = s.get_sync_data()

    by_src = {d["src_dir_abs"]: d for d in result}
    assert set(by_src) == {str(src), str(src / "a")}
    assert by_src[str(src)]["src_dir_fls"] == [os.path.join(str(src), "top.txt")]
    assert by_src[str(src / "a")]["dst_dir_abs"] == str(dst / "a")
    assert len(calls["mkdir"]) == 2


# ------------------------------------------------------- do_sync / unmatched


def test_do_sync_skips_existing_files_without_overwrite(calls):
    s = make_sync(overwrite_existing=False)
    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": ["/src/a.txt", "/src/b.txt"],
        "dst_dir_fls": ["/dst/a.txt"],
    }

    s.do_sync(sync_data)

    assert calls["cp"] == [("/src/b.txt", "/dst/b.txt")]
    assert sync_data["dst_dir_fls"] == []


def test_do_sync_overwrites_existing_files_when_enabled(calls):
    s = make_sync(overwrite_existing=True)
    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": ["/src/a.txt", "/src/b.txt"],
        "dst_dir_fls": ["/dst/a.txt"],
    }

    s.do_sync(sync_data)

    assert calls["cp"] == [
        ("/src/a.txt", "/dst/a.txt"),
        ("/src/b.txt", "/dst/b.txt"),
    ]


def test_handle_destination_dir_data_removes_unmatched(calls):
    s = make_sync(unmatched=REMOVE)
    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "dst_dir_fls": ["/dst/old.txt"],
    }

    s.handle_destination_dir_data(sync_data)

    assert calls["rm"] == [("/dst/old.txt",)]
    assert calls["cp"] == []


def test_handle_destination_dir_data_synchronizes_unmatched_back(calls):
    s = make_sync(unmatched=SYNCHRONIZE)
    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "dst_dir_fls": ["/dst/old.txt"],
    }

    s.handle_destination_dir_data(sync_data)

    assert calls["cp"] == [("/dst/old.txt", "/src/old.txt")]
    assert calls["rm"] == []


# ------------------------------------------------------------ sync() end-to-end


def test_sync_ignore_copies_without_touching_unmatched(tmp_path, calls):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("a")
    (dst / "old.txt").write_text("o")

    s = make_sync(source=str(src), destination=str(dst), unmatched=IGNORE)
    s.sync()

    assert calls["cp"] == [
        (os.path.join(str(src), "a.txt"), os.path.join(str(dst), "a.txt"))
    ]
    assert calls["rm"] == []


def test_sync_remove_deletes_unmatched_destination_files(tmp_path, calls):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("a")
    (dst / "old.txt").write_text("o")

    s = make_sync(source=str(src), destination=str(dst), unmatched=REMOVE)
    s.sync()

    assert calls["rm"] == [(os.path.join(str(dst), "old.txt"),)]


def test_sync_stops_when_no_files_remain_after_filtering(tmp_path, capsys, calls):
    src = tmp_path / "src"
    src.mkdir()
    (src / "clip.mp3").write_text("m")

    s = make_sync(
        source=str(src),
        destination=str(tmp_path / "dst"),
        ignore_file_types=["mp3"],
        verbose=True,
    )
    s.sync()

    assert "No files to sync" in capsys.readouterr().out
    assert calls["cp"] == []
