import os
from pathlib import Path

import pytest

import pysyncdroid
import pysyncdroid.sync as syncmod
from pysyncdroid import exceptions


def test_package_init_is_importable_and_empty():
    assert hasattr(pysyncdroid, "__name__")
    assert pysyncdroid.__name__ == "pysyncdroid"
    assert not hasattr(pysyncdroid, "Sync")


def test_readlink_returns_empty_like_input_unchanged(monkeypatch):
    called = []

    def fake_run(cmd):
        called.append(cmd)
        return "ignored"

    monkeypatch.setattr(syncmod, "run_bash_cmd", fake_run)
    assert syncmod.readlink("") == ""
    assert called == []


def test_readlink_expands_tilde_and_uses_run_bash_cmd(monkeypatch):
    captured = {}

    def fake_expanduser(path):
        captured["expanduser"] = path
        return "/home/test/file"

    def fake_run(cmd):
        captured["cmd"] = cmd
        return "/resolved/path"

    monkeypatch.setattr(syncmod.os.path, "expanduser", fake_expanduser)
    monkeypatch.setattr(syncmod, "run_bash_cmd", fake_run)

    assert syncmod.readlink("~/file") == "/resolved/path"
    assert captured["expanduser"] == "~/file"
    assert captured["cmd"] == ["readlink", "-f", "/home/test/file"]


def test_readlink_falls_back_to_original_path_when_command_returns_falsey(monkeypatch):
    monkeypatch.setattr(syncmod, "run_bash_cmd", lambda cmd: "")
    assert syncmod.readlink("/some/path") == "/some/path"


def test_sync_init_lowercases_ignore_file_types():
    s = syncmod.Sync(
        mtp_details=("mtp://x", "/mnt/device"),
        source="src",
        destination="dst",
        ignore_file_types=["JPG", "Png", "Txt"],
    )
    assert s.ignore_file_types == ["jpg", "png", "txt"]
    assert s.mtp_url == "mtp://x"
    assert s.mtp_gvfs_path == "/mnt/device"


def test_handle_ignored_file_type_raises_for_matching_extension():
    s = syncmod.Sync(
        mtp_details=("mtp://x", "/mnt/device"),
        source="src",
        destination="dst",
        ignore_file_types=["mp3", "jpg"],
    )
    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("photo.JPG")


def test_handle_ignored_file_type_allows_non_matching_or_no_extension():
    s = syncmod.Sync(
        mtp_details=("mtp://x", "/mnt/device"),
        source="src",
        destination="dst",
        ignore_file_types=["mp3"],
    )
    s.handle_ignored_file_type("document.txt")
    s.handle_ignored_file_type("no_extension")


def test_set_source_abs_uses_cwd_for_relative_computer_source(tmp_path, monkeypatch):
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(syncmod, "readlink", lambda p: "source")

    s = syncmod.Sync(("mtp://x", "/device"), "source", "dst")
    s.set_source_abs()

    assert s.source == str(src_dir)


def test_set_source_abs_uses_device_path_when_not_on_computer(tmp_path, monkeypatch):
    device_root = tmp_path / "device"
    src_dir = device_root / "camera"
    src_dir.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(syncmod, "readlink", lambda p: "camera")

    s = syncmod.Sync(("mtp://x", str(device_root)), "camera", "dst")
    s.set_source_abs()

    assert s.source == str(src_dir)


def test_set_source_abs_raises_when_source_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(syncmod, "readlink", lambda p: "missing")

    s = syncmod.Sync(("mtp://x", "/device"), "missing", "dst")
    with pytest.raises(OSError, match='"missing" does not exist on computer or on device'):
        s.set_source_abs()


def test_set_destination_abs_computer_destination_when_source_on_host(monkeypatch):
    monkeypatch.setattr(syncmod, "readlink", lambda p: "dst")
    s = syncmod.Sync(("mtp://x", "/device"), "mtp:host/source", "dst")
    s.set_destination_abs()
    assert s.destination == "dst"


def test_set_destination_abs_device_destination_when_source_on_device(monkeypatch):
    monkeypatch.setattr(syncmod, "readlink", lambda p: "dst")
    s = syncmod.Sync(("mtp://x", "/device"), "/device/source", "dst")
    s.set_destination_abs()
    assert s.destination == os.path.join("/device", "dst")


def test_set_destination_subdir_abs_strips_source_prefix_and_slashes():
    s = syncmod.Sync(("mtp://x", "/device"), "/root/source", "/dest")
    result = s.set_destination_subdir_abs("/root/source/sub/inner")
    assert result == os.path.join("/dest", "sub", "inner")


def test_sync_data_template_creates_expected_structure():
    s = syncmod.Sync(("mtp://x", "/device"), "src", "dst")
    data = s.sync_data_template("/a/b/", "/c/d/")
    assert data == {
        "src_dir_abs": "/a/b",
        "dst_dir_abs": "/c/d",
        "src_dir_fls": [],
        "dst_dir_fls": [],
    }


def test_get_source_subdir_data_skips_ignored_types():
    s = syncmod.Sync(
        ("mtp://x", "/device"),
        "src",
        "dst",
        ignore_file_types=["jpg"],
    )
    sync_data = {"src_dir_abs": "/src", "dst_dir_abs": "/dst", "src_dir_fls": [], "dst_dir_fls": []}
    s.get_source_subdir_data(["a.jpg", "b.txt", "c.JPG", "d"], sync_data)
    assert sync_data["src_dir_fls"] == ["/src/b.txt", "/src/d"]


def test_get_destination_subdir_data_creates_missing_dir_and_calls_mkdir(monkeypatch):
    calls = []

    def fake_exists(path):
        return False

    def fake_mkdir(path):
        calls.append(("mkdir", path))

    monkeypatch.setattr(syncmod.os.path, "exists", fake_exists)
    monkeypatch.setattr(syncmod.gvfs, "mkdir", fake_mkdir)

    s = syncmod.Sync(("mtp://x", "/device"), "src", "dst", verbose=True)
    messages = []
    monkeypatch.setattr(s, "_verbose", lambda msg: messages.append(msg))

    sync_data = {"src_dir_abs": "/src", "dst_dir_abs": "/dst", "src_dir_fls": [], "dst_dir_fls": []}
    s.get_destination_subdir_data(sync_data)

    assert calls == [("mkdir", "/dst")]
    assert messages == ["Creating directory /dst"]
    assert sync_data["dst_dir_fls"] == []


def test_get_destination_subdir_data_collects_existing_files_and_ignores_types(monkeypatch):
    monkeypatch.setattr(syncmod.os.path, "exists", lambda path: True)
    monkeypatch.setattr(syncmod.os, "listdir", lambda path: ["a.jpg", "b.txt", "c"])

    s = syncmod.Sync(("mtp://x", "/device"), "src", "dst", ignore_file_types=["jpg"])
    sync_data = {"src_dir_abs": "/src", "dst_dir_abs": "/dst", "src_dir_fls": [], "dst_dir_fls": []}
    s.get_destination_subdir_data(sync_data)

    assert sync_data["dst_dir_fls"] == ["/dst/b.txt", "/dst/c"]


def test_do_sync_skips_existing_when_overwrite_disabled(monkeypatch):
    copied = []
    s = syncmod.Sync(("mtp://x", "/device"), "src", "dst", overwrite_existing=False)
    s.copy_file = lambda src, dst: copied.append((src, dst))

    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": ["/src/a.txt", "/src/b.txt"],
        "dst_dir_fls": ["/dst/a.txt"],
    }
    s.do_sync(sync_data)

    assert copied == [("/src/b.txt", "/dst/b.txt")]
    assert sync_data["dst_dir_fls"] == []


def test_do_sync_overwrites_existing_when_enabled():
    copied = []
    s = syncmod.Sync(("mtp://x", "/device"), "src", "dst", overwrite_existing=True)
    s.copy_file = lambda src, dst: copied.append((src, dst))

    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": ["/src/a.txt"],
        "dst_dir_fls": ["/dst/a.txt"],
    }
    s.do_sync(sync_data)

    assert copied == [("/src/a.txt", "/dst/a.txt")]
    assert sync_data["dst_dir_fls"] == []


def test_handle_destination_dir_data_remove_calls_rm(monkeypatch):
    removed = []
    s = syncmod.Sync(("mtp://x", "/device"), "src", "dst", unmatched=syncmod.REMOVE, verbose=True)

    monkeypatch.setattr(s, "_verbose", lambda msg: removed.append(("verbose", msg)))
    monkeypatch.setattr(syncmod.gvfs, "rm", lambda path: removed.append(("rm", path)))

    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": [],
        "dst_dir_fls": ["/dst/orphan.txt"],
    }
    s.handle_destination_dir_data(sync_data)

    assert removed == [("verbose", "Removing /dst/orphan.txt"), ("rm", "/dst/orphan.txt")]


def test_handle_destination_dir_data_synchronize_copies_back_to_source():
    copied = []
    s = syncmod.Sync(("mtp://x", "/device"), "src", "dst", unmatched=syncmod.SYNCHRONIZE)
    s.copy_file = lambda src_file, dst_file: copied.append((src_file, dst_file))

    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": [],
        "dst_dir_fls": ["/dst/orphan.txt"],
    }
    s.handle_destination_dir_data(sync_data)

    assert copied == [("/dst/orphan.txt", "/src/orphan.txt")]


def test_gvfs_wrapper_remounts_and_retries_on_connection_reset(monkeypatch):
    calls = []

    class FakeBashException(Exception):
        pass

    def fake_func(arg):
        calls.append(arg)
        if len(calls) == 1:
            raise FakeBashException("something Connection reset by peer")
        return None

    monkeypatch.setattr(syncmod.exceptions, "BashException", FakeBashException)
    monkeypatch.setattr(syncmod.gvfs, "mount", lambda url: calls.append(("mount", url)))

    s = syncmod.Sync(("mtp://x", "/device"), "src", "dst")
    s.gvfs_wrapper(fake_func, "payload")

    assert calls == ["payload", ("mount", "mtp://x"), "payload"]


def test_gvfs_wrapper_reraises_unhandled_bash_exception(monkeypatch):
    class FakeBashException(Exception):
        pass

    def fake_func(*args):
        raise FakeBashException("some other error")

    monkeypatch.setattr(syncmod.exceptions, "BashException", FakeBashException)

    s = syncmod.Sync(("mtp://x", "/device"), "src", "dst")
    with pytest.raises(FakeBashException, match="some other error"):
        s.gvfs_wrapper(fake_func, "payload")


def test_sync_returns_early_when_no_source_files(monkeypatch):
    s = syncmod.Sync(("mtp://x", "/device"), "src", "dst", unmatched=syncmod.REMOVE)
    monkeypatch.setattr(s, "get_sync_data", lambda: [{"src_dir_fls": [], "dst_dir_fls": ["/dst/x"], "src_dir_abs": "/src", "dst_dir_abs": "/dst"}])

    called = []
    monkeypatch.setattr(s, "do_sync", lambda data: called.append("do_sync"))
    monkeypatch.setattr(s, "handle_destination_dir_data", lambda data: called.append("handle_dst"))

    s.sync()
    assert called == []


def test_sync_ignores_unmatched_files_when_configured(monkeypatch):
    s = syncmod.Sync(("mtp://x", "/device"), "src", "dst", unmatched=syncmod.IGNORE)
    sync_data = {"src_dir_fls": ["/src/a.txt"], "dst_dir_fls": ["/dst/orphan.txt"], "src_dir_abs": "/src", "dst_dir_abs": "/dst"}

    monkeypatch.setattr(s, "get_sync_data", lambda: [sync_data])

    called = []
    monkeypatch.setattr(s, "do_sync", lambda data: called.append("do_sync"))
    monkeypatch.setattr(s, "handle_destination_dir_data", lambda data: called.append("handle_dst"))

    s.sync()
    assert called == ["do_sync"]


def test_sync_processes_unmatched_files_when_configured(monkeypatch):
    s = syncmod.Sync(("mtp://x", "/device"), "src", "dst", unmatched=syncmod.REMOVE)
    sync_data = {"src_dir_fls": ["/src/a.txt"], "dst_dir_fls": ["/dst/orphan.txt"], "src_dir_abs": "/src", "dst_dir_abs": "/dst"}

    monkeypatch.setattr(s, "get_sync_data", lambda: [sync_data])

    called = []
    monkeypatch.setattr(s, "do_sync", lambda data: called.append("do_sync"))
    monkeypatch.setattr(s, "handle_destination_dir_data", lambda data: called.append("handle_dst"))

    s.sync()
    assert called == ["do_sync", "handle_dst"]
