# -*- coding: utf-8 -*-

"""Unit tests for pysyncdroid.sync and the pysyncdroid package."""

import os
import shutil

import pytest

import pysyncdroid
from pysyncdroid import exceptions
from pysyncdroid import sync as sync_mod
from pysyncdroid.sync import (
    IGNORE,
    REMOVE,
    SYNCHRONIZE,
    Sync,
    readlink,
)

MTP_DETAILS = ("mtp://[usb:001,002]/", "/run/user/1000/gvfs/mtp:host=DEVICE")


@pytest.fixture
def fake_run_bash(monkeypatch):
    """Replace run_bash_cmd used by sync.py. Returns empty output by default."""
    state = {"calls": [], "result": ""}

    def _fake(cmd):
        state["calls"].append(cmd)
        return state["result"]

    monkeypatch.setattr(sync_mod, "run_bash_cmd", _fake)
    return state


@pytest.fixture
def fake_gvfs(monkeypatch):
    """Replace gvfs operations with recording fakes that act on the local FS."""
    calls = {"mount": [], "mkdir": [], "cp": [], "rm": []}

    def _mount(url):
        calls["mount"].append(url)

    def _mkdir(path):
        calls["mkdir"].append(path)
        os.makedirs(path, exist_ok=True)

    def _cp(src, dst):
        calls["cp"].append((src, dst))
        shutil.copyfile(src, dst)

    def _rm(path):
        calls["rm"].append(path)
        os.remove(path)

    monkeypatch.setattr(sync_mod.gvfs, "mount", _mount)
    monkeypatch.setattr(sync_mod.gvfs, "mkdir", _mkdir)
    monkeypatch.setattr(sync_mod.gvfs, "cp", _cp)
    monkeypatch.setattr(sync_mod.gvfs, "rm", _rm)
    return calls


@pytest.fixture
def make_sync(fake_run_bash):
    """Factory creating Sync instances with sensible defaults."""

    def _make(**kwargs):
        kwargs.setdefault("mtp_details", MTP_DETAILS)
        kwargs.setdefault("source", "/unused/source")
        kwargs.setdefault("destination", "/unused/destination")
        return Sync(**kwargs)

    return _make


def test_readlink_empty_path_returned_unchanged(fake_run_bash):
    assert readlink("") == ""
    assert readlink(None) is None
    assert fake_run_bash["calls"] == []


def test_readlink_expands_tilde_before_resolving(fake_run_bash, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    fake_run_bash["result"] = "/resolved/path"

    result = readlink("~/music")

    assert result == "/resolved/path"
    assert fake_run_bash["calls"] == [["readlink", "-f", str(tmp_path / "music")]]


def test_readlink_returns_resolved_output(fake_run_bash):
    fake_run_bash["result"] = "/real/location"

    assert readlink("/link") == "/real/location"
    assert fake_run_bash["calls"] == [["readlink", "-f", "/link"]]


def test_readlink_falls_back_to_input_when_resolution_empty(fake_run_bash):
    fake_run_bash["result"] = ""

    assert readlink("relative/dir") == "relative/dir"


def test_init_maps_details_defaults_and_lowercases_extensions(make_sync):
    sync = make_sync(
        source="/src",
        destination="/dst",
        ignore_file_types=["MP3", "Tmp"],
    )

    assert sync.mtp_url == MTP_DETAILS[0]
    assert sync.mtp_gvfs_path == MTP_DETAILS[1]
    assert sync.source == "/src"
    assert sync.destination == "/dst"
    assert sync.unmatched == IGNORE
    assert sync.overwrite_existing is False
    assert sync.verbose is False
    assert sync.ignore_file_types == ["mp3", "tmp"]


def test_init_keeps_none_ignore_file_types(make_sync):
    sync = make_sync(ignore_file_types=None)

    assert sync.ignore_file_types is None


def test_verbose_prints_only_when_enabled(make_sync, capsys):
    make_sync(verbose=False)._verbose("hidden")
    assert capsys.readouterr().out == ""

    make_sync(verbose=True)._verbose("shown")
    assert capsys.readouterr().out == "shown\n"


def test_gvfs_wrapper_calls_function_once_on_success(make_sync, fake_gvfs):
    sync = make_sync()
    received = []

    sync.gvfs_wrapper(lambda *args: received.append(args), "a", "b")

    assert received == [("a", "b")]
    assert fake_gvfs["mount"] == []


def test_gvfs_wrapper_reraises_unhandled_bash_error(make_sync, fake_gvfs):
    sync = make_sync()

    def failing(*args):
        raise exceptions.BashException("some other failure")

    with pytest.raises(exceptions.BashException):
        sync.gvfs_wrapper(failing, "x")

    assert fake_gvfs["mount"] == []


def test_gvfs_wrapper_remounts_and_retries_on_connection_reset(make_sync, fake_gvfs):
    sync = make_sync()
    attempts = []

    def flaky(*args):
        attempts.append(args)
        if len(attempts) == 1:
            raise exceptions.BashException("Connection reset by peer")

    sync.gvfs_wrapper(flaky, "file")

    assert len(attempts) == 2
    assert fake_gvfs["mount"] == [MTP_DETAILS[0]]


def test_set_source_abs_accepts_existing_absolute_dir(make_sync, fake_run_bash, tmp_path):
    src_dir = tmp_path / "photos"
    src_dir.mkdir()
    sync = make_sync(source=str(src_dir))

    sync.set_source_abs()

    assert sync.source == str(src_dir)


def test_set_source_abs_falls_back_to_device_prefix(make_sync, fake_run_bash, tmp_path, monkeypatch):
    device = tmp_path / "device"
    (device / "photos").mkdir(parents=True)
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)

    sync = make_sync(
        mtp_details=(MTP_DETAILS[0], str(device)),
        source="photos",
    )
    sync.set_source_abs()

    assert sync.source == str(device / "photos")


def test_set_source_abs_raises_when_missing(make_sync, fake_run_bash, tmp_path):
    sync = make_sync(
        mtp_details=(MTP_DETAILS[0], str(tmp_path / "device")),
        source=str(tmp_path / "does_not_exist"),
    )

    with pytest.raises(OSError, match="does not exist"):
        sync.set_source_abs()


def test_set_source_abs_raises_when_not_directory(make_sync, fake_run_bash, tmp_path):
    a_file = tmp_path / "file.txt"
    a_file.write_text("data")
    sync = make_sync(source=str(a_file))

    with pytest.raises(OSError, match="is not a directory"):
        sync.set_source_abs()


def test_set_destination_abs_computer_is_destination(make_sync, fake_run_bash):
    sync = make_sync(
        source="/run/user/1000/gvfs/mtp:host=DEVICE/Internal/DCIM",
        destination="/home/user/backup",
    )

    sync.set_destination_abs()

    assert sync.destination == "/home/user/backup"


def test_set_destination_abs_device_is_destination(make_sync, fake_run_bash):
    sync = make_sync(
        mtp_details=(MTP_DETAILS[0], "/gvfs/dev"),
        source="/home/user/pics",
        destination="Backup",
    )

    sync.set_destination_abs()

    assert sync.destination == "/gvfs/dev/Backup"


def test_set_destination_subdir_abs_maps_relative_subdir(make_sync):
    sync = make_sync(source="/src/root", destination="/dst/root")

    result = sync.set_destination_subdir_abs("/src/root/a/b")

    assert result == os.path.join("/dst/root", "a", "b")


def test_set_destination_subdir_abs_for_source_root(make_sync):
    sync = make_sync(source="/src/root", destination="/dst/root")

    result = sync.set_destination_subdir_abs("/src/root")

    assert result == "/dst/root/"


def test_sync_data_template_strips_trailing_slashes_and_initialises_lists(make_sync):
    sync = make_sync()

    data = sync.sync_data_template("/src/dir/", "/dst/dir//")

    assert data == {
        "src_dir_abs": "/src/dir",
        "dst_dir_abs": "/dst/dir",
        "src_dir_fls": [],
        "dst_dir_fls": [],
    }


def test_handle_ignored_file_type_is_case_insensitive(make_sync):
    sync = make_sync(ignore_file_types=["mp3"])

    with pytest.raises(exceptions.IgnoredTypeException):
        sync.handle_ignored_file_type("SONG.MP3")

    sync.handle_ignored_file_type("notes.txt")  # must not raise
    sync.handle_ignored_file_type("no_extension")  # must not raise


def test_get_source_subdir_data_filters_ignored_files(make_sync):
    sync = make_sync(ignore_file_types=["mp3"])
    sync_data = sync.sync_data_template("/src/dir", "/dst/dir")

    sync.get_source_subdir_data(["a.txt", "b.mp3"], sync_data)

    assert sync_data["src_dir_fls"] == [os.path.join("/src/dir", "a.txt")]


def test_get_destination_subdir_data_creates_missing_dir(make_sync, fake_gvfs, tmp_path):
    missing = tmp_path / "new_dst"
    sync = make_sync()
    sync_data = sync.sync_data_template("/src/dir", str(missing))

    sync.get_destination_subdir_data(sync_data)

    assert fake_gvfs["mkdir"] == [str(missing)]
    assert os.path.isdir(str(missing))
    assert sync_data["dst_dir_fls"] == []


def test_do_sync_skips_existing_files_by_default(make_sync, fake_gvfs, tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("new")
    (src / "b.txt").write_text("new")
    (dst / "a.txt").write_text("old")

    sync = make_sync()
    sync_data = sync.sync_data_template(str(src), str(dst))
    sync_data["src_dir_fls"] = [str(src / "a.txt"), str(src / "b.txt")]
    sync_data["dst_dir_fls"] = [str(dst / "a.txt")]

    sync.do_sync(sync_data)

    assert fake_gvfs["cp"] == [(str(src / "b.txt"), str(dst / "b.txt"))]
    assert sync_data["dst_dir_fls"] == []
    assert (dst / "a.txt").read_text() == "old"


def test_do_sync_overwrites_existing_files_when_enabled(make_sync, fake_gvfs, tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("new")
    (dst / "a.txt").write_text("old")

    sync = make_sync(overwrite_existing=True)
    sync_data = sync.sync_data_template(str(src), str(dst))
    sync_data["src_dir_fls"] = [str(src / "a.txt")]
    sync_data["dst_dir_fls"] = [str(dst / "a.txt")]

    sync.do_sync(sync_data)

    assert fake_gvfs["cp"] == [(str(src / "a.txt"), str(dst / "a.txt"))]
    assert (dst / "a.txt").read_text() == "new"


def test_handle_destination_dir_data_removes_unmatched_files(make_sync, fake_gvfs, tmp_path):
    dst = tmp_path / "dst"
    dst.mkdir()
    stale = dst / "stale.txt"
    stale.write_text("x")

    sync = make_sync(unmatched=REMOVE)
    sync_data = sync.sync_data_template(str(tmp_path / "src"), str(dst))
    sync_data["dst_dir_fls"] = [str(stale)]

    sync.handle_destination_dir_data(sync_data)

    assert fake_gvfs["rm"] == [str(stale)]
    assert not stale.exists()


def test_handle_destination_dir_data_copies_unmatched_back_when_synchronizing(
    make_sync, fake_gvfs, tmp_path
):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    device_only = dst / "device_only.txt"
    device_only.write_text("from device")

    sync = make_sync(unmatched=SYNCHRONIZE)
    sync_data = sync.sync_data_template(str(src), str(dst))
    sync_data["dst_dir_fls"] = [str(device_only)]

    sync.handle_destination_dir_data(sync_data)

    expected_dst = str(src / "device_only.txt")
    assert fake_gvfs["cp"] == [(str(device_only), expected_dst)]
    assert (src / "device_only.txt").read_text() == "from device"


def test_sync_end_to_end_with_local_directories(make_sync, fake_gvfs, tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    dst.mkdir()
    (src / "a.txt").write_text("a")
    (src / "b.mp3").write_text("music")
    (src / "sub" / "c.txt").write_text("c")
    (dst / "a.txt").write_text("old a")
    (dst / "stale.txt").write_text("stale")

    sync = make_sync(
        source=str(src),
        destination=str(dst),
        unmatched=REMOVE,
        ignore_file_types=["mp3"],
    )

    sync.sync()

    assert (dst / "a.txt").read_text() == "old a"
    assert (dst / "sub" / "c.txt").read_text() == "c"
    assert not (dst / "b.mp3").exists()
    assert not (dst / "stale.txt").exists()
    assert fake_gvfs["mkdir"] == [str(dst / "sub")]


def test_package_module_imports_cleanly():
    assert pysyncdroid is not None
    assert hasattr(sync_mod, "Sync")
