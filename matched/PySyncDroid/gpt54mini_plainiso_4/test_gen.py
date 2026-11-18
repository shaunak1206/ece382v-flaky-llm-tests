import os
from types import SimpleNamespace

import pytest

import pysyncdroid.sync as sync_mod
from pysyncdroid import exceptions


def test_readlink_returns_empty_like_input_when_path_is_falsy():
    assert sync_mod.readlink("") == ""
    assert sync_mod.readlink(None) is None


def test_readlink_expands_tilde_and_uses_run_bash_cmd(monkeypatch):
    captured = {}

    def fake_expanduser(path):
        captured["expanded"] = path
        return "/home/test/file"

    def fake_run(cmd):
        captured["cmd"] = cmd
        return "/resolved/file"

    monkeypatch.setattr(sync_mod.os.path, "expanduser", fake_expanduser)
    monkeypatch.setattr(sync_mod, "run_bash_cmd", fake_run)

    assert sync_mod.readlink("~/file") == "/resolved/file"
    assert captured["expanded"] == "~/file"
    assert captured["cmd"] == ["readlink", "-f", "/home/test/file"]


def test_readlink_falls_back_to_original_path_when_command_returns_none(monkeypatch):
    monkeypatch.setattr(sync_mod, "run_bash_cmd", lambda cmd: None)
    assert sync_mod.readlink("/tmp/x") == "/tmp/x"


def test_sync_init_normalizes_ignore_file_types_to_lowercase():
    s = sync_mod.Sync(("url", "/mnt/device"), "src", "dst", ignore_file_types=["JPG", "Pdf"])
    assert s.ignore_file_types == ["jpg", "pdf"]
    assert s.mtp_url == "url"
    assert s.mtp_gvfs_path == "/mnt/device"
    assert s.source == "src"
    assert s.destination == "dst"


def test_verbose_prints_only_when_enabled(capsys):
    s = sync_mod.Sync(("url", "/mnt/device"), "src", "dst", verbose=True)
    s._verbose("hello")
    out = capsys.readouterr().out
    assert "hello" in out


def test_gvfs_wrapper_retries_on_connection_reset(monkeypatch):
    calls = {"func": 0, "mount": 0}

    def failing_then_working(arg):
        calls["func"] += 1
        if calls["func"] == 1:
            raise exceptions.BashException("something Connection reset by peer")
        return None

    def fake_mount(url):
        calls["mount"] += 1
        assert url == "mtp://device"

    monkeypatch.setattr(sync_mod.gvfs, "mount", fake_mount)

    s = sync_mod.Sync(("mtp://device", "/mnt/device"), "src", "dst")
    s.gvfs_wrapper(failing_then_working, "arg")

    assert calls["func"] == 2
    assert calls["mount"] == 1


def test_gvfs_wrapper_reraises_other_bash_exceptions():
    s = sync_mod.Sync(("url", "/mnt/device"), "src", "dst")

    def boom(arg):
        raise exceptions.BashException("different error")

    with pytest.raises(exceptions.BashException, match="different error"):
        s.gvfs_wrapper(boom, "arg")


def test_set_source_abs_prefers_computer_path_when_exists(monkeypatch):
    s = sync_mod.Sync(("url", "/mnt/device"), "relsrc", "dst")
    monkeypatch.setattr(sync_mod, "readlink", lambda p: p)
    monkeypatch.setattr(sync_mod.os, "getcwd", lambda: "/cwd")
    monkeypatch.setattr(sync_mod.os.path, "isabs", lambda p: False)
    monkeypatch.setattr(sync_mod.os.path, "exists", lambda p: p == "/cwd/relsrc")
    monkeypatch.setattr(sync_mod.os.path, "isdir", lambda p: True)

    s.set_source_abs()
    assert s.source == "/cwd/relsrc"


def test_set_source_abs_finds_source_on_device_and_decodes_bytes(monkeypatch):
    s = sync_mod.Sync(("url", "/mnt/device"), "relsrc", "dst")
    monkeypatch.setattr(sync_mod, "readlink", lambda p: p)
    monkeypatch.setattr(sync_mod.os, "getcwd", lambda: "/cwd")
    monkeypatch.setattr(sync_mod.os.path, "isabs", lambda p: False)

    def fake_exists(path):
        return path == "/mnt/device/relsrc"

    monkeypatch.setattr(sync_mod.os.path, "exists", fake_exists)
    monkeypatch.setattr(sync_mod.os.path, "isdir", lambda p: True)

    s.set_source_abs()
    assert s.source == "/mnt/device/relsrc"


def test_set_source_abs_raises_when_source_missing(monkeypatch):
    s = sync_mod.Sync(("url", "/mnt/device"), "missing", "dst")
    monkeypatch.setattr(sync_mod, "readlink", lambda p: p)
    monkeypatch.setattr(sync_mod.os, "getcwd", lambda: "/cwd")
    monkeypatch.setattr(sync_mod.os.path, "isabs", lambda p: False)
    monkeypatch.setattr(sync_mod.os.path, "exists", lambda p: False)

    with pytest.raises(OSError, match='"missing" does not exist on computer or on device.'):
        s.set_source_abs()


def test_set_destination_abs_uses_host_destination_when_source_is_on_computer(monkeypatch):
    s = sync_mod.Sync(("url", "/mnt/device"), "mtp:host:/src", "dest")
    monkeypatch.setattr(sync_mod, "readlink", lambda p: p)

    s.set_destination_abs()
    assert s.destination == "dest"


def test_set_destination_abs_joins_device_path_when_source_is_on_device(monkeypatch):
    s = sync_mod.Sync(("url", "/mnt/device"), "/mnt/device/src", "dest")
    monkeypatch.setattr(sync_mod, "readlink", lambda p: p)

    s.set_destination_abs()
    assert s.destination == "/mnt/device/dest"


def test_handle_ignored_file_type_raises_for_matching_extension():
    s = sync_mod.Sync(("url", "/mnt/device"), "src", "dst", ignore_file_types=["jpg", "tmp"])
    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("photo.JPG")


def test_get_source_subdir_data_skips_ignored_types_and_collects_others():
    s = sync_mod.Sync(("url", "/mnt/device"), "src", "dst", ignore_file_types=["tmp"])
    sync_data = {"src_dir_abs": "/src", "src_dir_fls": []}

    s.get_source_subdir_data(["a.txt", "b.TMP", "c.md"], sync_data)
    assert sync_data["src_dir_fls"] == ["/src/a.txt", "/src/c.md"]


def test_get_destination_subdir_data_creates_missing_directory(monkeypatch):
    s = sync_mod.Sync(("url", "/mnt/device"), "src", "dst", verbose=True)
    created = []

    monkeypatch.setattr(sync_mod.os.path, "exists", lambda p: False)
    monkeypatch.setattr(sync_mod.gvfs, "mkdir", lambda p: created.append(p))

    sync_data = {"dst_dir_abs": "/dst/sub", "dst_dir_fls": []}
    s.get_destination_subdir_data(sync_data)

    assert created == ["/dst/sub"]
    assert sync_data["dst_dir_fls"] == []


def test_get_destination_subdir_data_collects_existing_files_and_skips_ignored(monkeypatch):
    s = sync_mod.Sync(("url", "/mnt/device"), "src", "dst", ignore_file_types=["log"])
    monkeypatch.setattr(sync_mod.os.path, "exists", lambda p: True)
    monkeypatch.setattr(sync_mod.os, "listdir", lambda p: ["a.txt", "b.LOG", "c.md"])

    sync_data = {"dst_dir_abs": "/dst/sub", "dst_dir_fls": []}
    s.get_destination_subdir_data(sync_data)

    assert sync_data["dst_dir_fls"] == ["/dst/sub/a.txt", "/dst/sub/c.md"]


def test_get_sync_data_builds_entries_for_directories_with_files_and_skips_empty(monkeypatch):
    s = sync_mod.Sync(("url", "/mnt/device"), "/src", "/dst")
    monkeypatch.setattr(sync_mod, "readlink", lambda p: p)
    monkeypatch.setattr(sync_mod.os.path, "exists", lambda p: True)
    monkeypatch.setattr(sync_mod.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(sync_mod.os, "walk", lambda root: iter([("/src", ["sub"], ["a.txt"]), ("/src/sub", [], [])]))

    monkeypatch.setattr(s, "set_destination_subdir_abs", lambda src: src.replace("/src", "/dst"))
    monkeypatch.setattr(s, "get_source_subdir_data", lambda files, sd: sd["src_dir_fls"].extend([os.path.join(sd["src_dir_abs"], f) for f in files]))
    monkeypatch.setattr(s, "get_destination_subdir_data", lambda sd: sd["dst_dir_fls"].extend([]))

    data = s.get_sync_data()
    assert len(data) == 1
    assert data[0]["src_dir_abs"] == "/src"
    assert data[0]["dst_dir_abs"] == "/dst"
    assert data[0]["src_dir_fls"] == ["/src/a.txt"]


def test_do_sync_respects_overwrite_existing_flag(monkeypatch):
    s = sync_mod.Sync(("url", "/mnt/device"), "src", "dst", overwrite_existing=False)
    copied = []
    monkeypatch.setattr(s, "copy_file", lambda src, dst: copied.append((src, dst)))

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
    s = sync_mod.Sync(("url", "/mnt/device"), "/src", "/dst", unmatched=sync_mod.REMOVE)
    removed = []
    copied = []

    monkeypatch.setattr(s, "gvfs_wrapper", lambda func, *args: removed.append(args[0]) if func is sync_mod.gvfs.rm else None)
    monkeypatch.setattr(s, "copy_file", lambda src_file, dst_file: copied.append((src_file, dst_file)))

    sync_data = {"src_dir_abs": "/src", "dst_dir_abs": "/dst", "dst_dir_fls": ["/dst/x.txt", "/dst/y.txt"]}
    s.handle_destination_dir_data(sync_data)

    assert removed == ["/dst/x.txt", "/dst/y.txt"]
    assert copied == []

    s2 = sync_mod.Sync(("url", "/mnt/device"), "/src", "/dst", unmatched=sync_mod.SYNCHRONIZE)
    copied2 = []
    monkeypatch.setattr(s2, "copy_file", lambda src_file, dst_file: copied2.append((src_file, dst_file)))

    s2.handle_destination_dir_data(sync_data)
    assert copied2 == [("/dst/x.txt", "/src/x.txt"), ("/dst/y.txt", "/src/y.txt")]


def test_sync_stops_early_when_source_has_no_files(monkeypatch):
    s = sync_mod.Sync(("url", "/mnt/device"), "/src", "/dst")
    monkeypatch.setattr(s, "get_sync_data", lambda: [{"src_dir_fls": [], "dst_dir_fls": ["x"]}])
    called = []
    monkeypatch.setattr(s, "do_sync", lambda sd: called.append(True))

    s.sync()
    assert called == []
