import os
import pytest

import pysyncdroid.sync as syncmod
from pysyncdroid.sync import IGNORE, REMOVE, SYNCHRONIZE, Sync, readlink


class DummyBashException(Exception):
    pass


class DummyIgnoredTypeException(Exception):
    pass


def test_readlink_returns_empty_path_unchanged():
    assert readlink("") == ""
    assert readlink(None) is None


def test_readlink_expands_tilde_and_uses_run_bash_cmd(monkeypatch):
    calls = []

    def fake_expanduser(path):
        calls.append(("expanduser", path))
        return "/home/test/file"

    def fake_run(cmd):
        calls.append(("run", cmd))
        return "/resolved/path"

    monkeypatch.setattr(syncmod.os.path, "expanduser", fake_expanduser)
    monkeypatch.setattr(syncmod, "run_bash_cmd", fake_run)

    assert readlink("~/file") == "/resolved/path"
    assert calls == [
        ("expanduser", "~/file"),
        ("run", ["readlink", "-f", "/home/test/file"]),
    ]


def test_readlink_falls_back_to_original_path_if_command_returns_falsey(monkeypatch):
    monkeypatch.setattr(syncmod, "run_bash_cmd", lambda cmd: "")
    assert readlink("/some/path") == "/some/path"


def test_sync_init_lowercases_ignore_file_types():
    s = Sync(("mtp://x", "/gvfs"), "src", "dst", ignore_file_types=["JPG", "Pdf"])
    assert s.ignore_file_types == ["jpg", "pdf"]
    assert s.mtp_url == "mtp://x"
    assert s.mtp_gvfs_path == "/gvfs"
    assert s.source == "src"
    assert s.destination == "dst"


def test_handle_ignored_file_type_raises_for_matching_extension():
    s = Sync(("u", "/g"), "src", "dst", ignore_file_types=["jpg", "png"])
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(syncmod.exceptions, "IgnoredTypeException", DummyIgnoredTypeException, raising=False)
    try:
        with pytest.raises(DummyIgnoredTypeException):
            s.handle_ignored_file_type("photo.JPG")
    finally:
        monkeypatch.undo()


def test_handle_ignored_file_type_allows_non_matching_extension_and_no_extension():
    s = Sync(("u", "/g"), "src", "dst", ignore_file_types=["jpg"])
    s.handle_ignored_file_type("document.txt")
    s.handle_ignored_file_type("README")


def test_set_source_abs_uses_cwd_for_relative_source(monkeypatch):
    s = Sync(("u", "/gvfs"), "srcdir", "dst")
    monkeypatch.setattr(syncmod, "readlink", lambda p: p)
    monkeypatch.setattr(syncmod.os, "getcwd", lambda: "/cwd")
    monkeypatch.setattr(syncmod.os.path, "isabs", lambda p: False)
    monkeypatch.setattr(syncmod.os.path, "exists", lambda p: p == "/cwd/srcdir")
    monkeypatch.setattr(syncmod.os.path, "isdir", lambda p: True)

    s.set_source_abs()
    assert s.source == "/cwd/srcdir"


def test_set_source_abs_checks_device_prefix_if_not_found_on_computer(monkeypatch):
    s = Sync(("u", "/gvfs"), "srcdir", "dst")
    monkeypatch.setattr(syncmod, "readlink", lambda p: p)
    monkeypatch.setattr(syncmod.os, "getcwd", lambda: "/cwd")
    monkeypatch.setattr(syncmod.os.path, "isabs", lambda p: False)

    def fake_exists(path):
        return path == "/gvfs/srcdir"

    monkeypatch.setattr(syncmod.os.path, "exists", fake_exists)
    monkeypatch.setattr(syncmod.os.path, "isdir", lambda p: True)

    s.set_source_abs()
    assert s.source == "/gvfs/srcdir"


def test_set_source_abs_raises_if_source_missing(monkeypatch):
    s = Sync(("u", "/gvfs"), "missing", "dst")
    monkeypatch.setattr(syncmod, "readlink", lambda p: p)
    monkeypatch.setattr(syncmod.os, "getcwd", lambda: "/cwd")
    monkeypatch.setattr(syncmod.os.path, "isabs", lambda p: False)
    monkeypatch.setattr(syncmod.os.path, "exists", lambda p: False)

    with pytest.raises(OSError) as exc:
        s.set_source_abs()
    assert '"missing" does not exist on computer or on device.' in str(exc.value)


def test_set_destination_abs_uses_destination_directly_for_computer_source(monkeypatch):
    s = Sync(("u", "/gvfs"), "mtp:host:/source", "dstdir")
    monkeypatch.setattr(syncmod, "readlink", lambda p: p)

    s.set_destination_abs()
    assert s.destination == "dstdir"


def test_set_destination_abs_joins_gvfs_path_for_device_destination(monkeypatch):
    s = Sync(("u", "/gvfs"), "/computer/source", "dstdir")
    monkeypatch.setattr(syncmod, "readlink", lambda p: p)

    s.set_destination_abs()
    assert s.destination == os.path.join("/gvfs", "dstdir")


def test_set_destination_subdir_abs_builds_relative_path_correctly():
    s = Sync(("u", "/gvfs"), "/src", "/dst")
    assert s.set_destination_subdir_abs("/src") == "/dst"
    assert s.set_destination_subdir_abs("/src/sub/a") == os.path.join("/dst", "sub", "a")


def test_sync_data_template_strips_trailing_slashes():
    s = Sync(("u", "/gvfs"), "src", "dst")
    data = s.sync_data_template("/tmp/source/", "/tmp/dest///")
    assert data["src_dir_abs"] == "/tmp/source"
    assert data["dst_dir_abs"] == "/tmp/dest"
    assert data["src_dir_fls"] == []
    assert data["dst_dir_fls"] == []


def test_get_source_subdir_data_ignores_matching_types(monkeypatch):
    s = Sync(("u", "/gvfs"), "src", "dst", ignore_file_types=["tmp"])
    monkeypatch.setattr(syncmod.exceptions, "IgnoredTypeException", DummyIgnoredTypeException, raising=False)

    sync_data = s.sync_data_template("/src", "/dst")
    s.get_source_subdir_data(["a.txt", "b.TMP", "c.md"], sync_data)

    assert sync_data["src_dir_fls"] == ["/src/a.txt", "/src/c.md"]


def test_get_destination_subdir_data_creates_missing_directory(monkeypatch):
    s = Sync(("u", "/gvfs"), "src", "dst", verbose=True)
    calls = []

    monkeypatch.setattr(syncmod.os.path, "exists", lambda p: False)
    monkeypatch.setattr(syncmod.gvfs, "mkdir", lambda p: calls.append(("mkdir", p)))

    sync_data = s.sync_data_template("/src", "/dst")
    s.get_destination_subdir_data(sync_data)

    assert calls == [("mkdir", "/dst")]
    assert sync_data["dst_dir_fls"] == []


def test_do_sync_skips_existing_files_when_not_overwriting(monkeypatch):
    s = Sync(("u", "/gvfs"), "src", "dst", overwrite_existing=False)
    copied = []
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


def test_handle_destination_dir_data_remove_and_synchronize(monkeypatch):
    s_remove = Sync(("u", "/gvfs"), "src", "dst", unmatched=REMOVE, verbose=True)
    removed = []
    monkeypatch.setattr(syncmod.gvfs, "rm", lambda p: removed.append(p))

    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": [],
        "dst_dir_fls": ["/dst/x.txt", "/dst/y.txt"],
    }
    s_remove.handle_destination_dir_data(sync_data)
    assert removed == ["/dst/x.txt", "/dst/y.txt"]

    s_sync = Sync(("u", "/gvfs"), "src", "dst", unmatched=SYNCHRONIZE)
    copied = []
    s_sync.copy_file = lambda src_file, dst_file: copied.append((src_file, dst_file))
    sync_data2 = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": [],
        "dst_dir_fls": ["/dst/x.txt"],
    }
    s_sync.handle_destination_dir_data(sync_data2)
    assert copied == [("/dst/x.txt", "/src/x.txt")]


def test_sync_returns_early_when_no_source_files(monkeypatch):
    s = Sync(("u", "/gvfs"), "src", "dst", unmatched=REMOVE)
    monkeypatch.setattr(s, "get_sync_data", lambda: [{
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": [],
        "dst_dir_fls": ["/dst/leftover.txt"],
    }])

    called = []
    monkeypatch.setattr(s, "do_sync", lambda data: called.append("do_sync"))
    monkeypatch.setattr(s, "handle_destination_dir_data", lambda data: called.append("handle"))

    s.sync()
    assert called == []
