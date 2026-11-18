import os
from types import SimpleNamespace

import pytest

from pysyncdroid import exceptions
from pysyncdroid import sync as syncmod


def test_readlink_returns_empty_input_unchanged():
    assert syncmod.readlink("") == ""


def test_readlink_expands_tilde_and_returns_command_output(monkeypatch):
    captured = {}

    def fake_expanduser(path):
        captured["expanded"] = path
        return "/home/test/user"

    def fake_run_bash_cmd(cmd):
        captured["cmd"] = cmd
        return "/resolved/path"

    monkeypatch.setattr(syncmod.os.path, "expanduser", fake_expanduser)
    monkeypatch.setattr(syncmod, "run_bash_cmd", fake_run_bash_cmd)

    assert syncmod.readlink("~/user") == "/resolved/path"
    assert captured["expanded"] == "~/user"
    assert captured["cmd"] == ["readlink", "-f", "/home/test/user"]


def test_readlink_falls_back_to_original_path_when_command_returns_empty(monkeypatch):
    monkeypatch.setattr(syncmod, "run_bash_cmd", lambda cmd: "")
    assert syncmod.readlink("/some/path") == "/some/path"


def test_sync_init_lowercases_ignore_file_types():
    s = syncmod.Sync(
        ("mtp:url", "/mnt/device"),
        "src",
        "dst",
        ignore_file_types=["JPG", "Mp3"],
    )
    assert s.ignore_file_types == ["jpg", "mp3"]


def test_verbose_prints_only_when_enabled(monkeypatch):
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "src", "dst", verbose=True)
    printed = []
    monkeypatch.setattr(syncmod, "print", lambda msg: printed.append(msg))
    s._verbose("hello")
    assert printed == ["hello"]


def test_gvfs_wrapper_retries_on_connection_reset(monkeypatch):
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "src", "dst")
    calls = {"func": 0, "mount": 0}

    def fake_func(*args):
        calls["func"] += 1
        if calls["func"] == 1:
            raise exceptions.BashException("Connection reset by peer")
        return "ok"

    monkeypatch.setattr(syncmod.gvfs, "mount", lambda url: calls.__setitem__("mount", calls["mount"] + 1))

    s.gvfs_wrapper(fake_func, "arg1")
    assert calls["func"] == 2
    assert calls["mount"] == 1


def test_gvfs_wrapper_reraises_other_bash_exception():
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "src", "dst")

    def fake_func(*args):
        raise exceptions.BashException("some other error")

    with pytest.raises(exceptions.BashException, match="some other error"):
        s.gvfs_wrapper(fake_func, "arg1")


def test_set_source_abs_uses_current_working_directory_for_relative_source(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    source_dir = cwd / "source"
    source_dir.mkdir()

    monkeypatch.setattr(syncmod.os, "getcwd", lambda: str(cwd))
    monkeypatch.setattr(syncmod.os.path, "isabs", lambda p: False)
    monkeypatch.setattr(syncmod.os.path, "exists", lambda p: p == str(source_dir))
    monkeypatch.setattr(syncmod.os.path, "isdir", lambda p: p == str(source_dir))
    monkeypatch.setattr(syncmod, "readlink", lambda p: "source")

    s = syncmod.Sync(("mtp:url", "/mnt/device"), "source", "dst")
    s.set_source_abs()

    assert s.source == str(source_dir)


def test_set_source_abs_raises_when_source_missing(monkeypatch):
    monkeypatch.setattr(syncmod, "readlink", lambda p: p)
    monkeypatch.setattr(syncmod.os.path, "isabs", lambda p: True)
    monkeypatch.setattr(syncmod.os.path, "exists", lambda p: False)

    s = syncmod.Sync(("mtp:url", "/mnt/device"), "/missing", "dst")
    with pytest.raises(OSError, match='"\/missing" does not exist on computer or on device\.'):
        s.set_source_abs()


def test_set_source_abs_raises_when_source_is_not_directory(monkeypatch):
    monkeypatch.setattr(syncmod, "readlink", lambda p: p)
    monkeypatch.setattr(syncmod.os.path, "isabs", lambda p: True)
    monkeypatch.setattr(syncmod.os.path, "exists", lambda p: True)
    monkeypatch.setattr(syncmod.os.path, "isdir", lambda p: False)

    s = syncmod.Sync(("mtp:url", "/mnt/device"), "/file", "dst")
    with pytest.raises(OSError, match='"\/file" is not a directory\.'):
        s.set_source_abs()


def test_set_destination_abs_uses_host_path_when_source_is_on_computer(monkeypatch):
    monkeypatch.setattr(syncmod, "readlink", lambda p: "dest")
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "mtp:host/source", "dest")
    s.set_destination_abs()
    assert s.destination == "dest"


def test_set_destination_subdir_abs_builds_relative_destination_path():
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "/src", "/dst")
    s.source = "/src"
    s.destination = "/dst"
    assert s.set_destination_subdir_abs("/src/sub/dir") == os.path.join("/dst", "sub", "dir")


def test_sync_data_template_strips_trailing_slashes():
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "/src", "/dst")
    data = s.sync_data_template("/src/sub/", "/dst/sub/")
    assert data["src_dir_abs"] == "/src/sub"
    assert data["dst_dir_abs"] == "/dst/sub"
    assert data["src_dir_fls"] == []
    assert data["dst_dir_fls"] == []


def test_handle_ignored_file_type_skips_matching_extension():
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "/src", "/dst", ignore_file_types=["mp3", "jpg"])
    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("song.MP3")
    s.handle_ignored_file_type("document.txt")  # should not raise


def test_get_source_subdir_data_ignores_configured_file_types():
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "/src", "/dst", ignore_file_types=["tmp"])
    sync_data = s.sync_data_template("/src", "/dst")
    s.get_source_subdir_data(["a.txt", "b.TMP", "c.jpg"], sync_data)
    assert sync_data["src_dir_fls"] == ["/src/a.txt", "/src/c.jpg"]


def test_get_destination_subdir_data_creates_missing_directory(monkeypatch):
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "/src", "/dst")
    sync_data = s.sync_data_template("/src", "/dst/newdir")

    called = {"mkdir": 0, "verbose": []}
    monkeypatch.setattr(syncmod.os.path, "exists", lambda p: False)
    monkeypatch.setattr(syncmod.gvfs, "mkdir", lambda p: called.__setitem__("mkdir", called["mkdir"] + 1))
    monkeypatch.setattr(s, "_verbose", lambda msg: called["verbose"].append(msg))

    s.get_destination_subdir_data(sync_data)

    assert called["mkdir"] == 1
    assert called["verbose"] == ["Creating directory /dst/newdir"]


def test_get_destination_subdir_data_collects_existing_files_and_ignores_types(monkeypatch):
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "/src", "/dst", ignore_file_types=["tmp"])
    sync_data = s.sync_data_template("/src", "/dst")
    monkeypatch.setattr(syncmod.os.path, "exists", lambda p: True)
    monkeypatch.setattr(syncmod.os, "listdir", lambda p: ["keep.txt", "skip.TMP"])

    s.get_destination_subdir_data(sync_data)
    assert sync_data["dst_dir_fls"] == ["/dst/keep.txt"]


def test_get_sync_data_skips_empty_directories_and_builds_data(monkeypatch):
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "/src", "/dst")
    s.source = "/src"
    s.destination = "/dst"

    walk_result = [
        ("/src", ["sub"], ["a.txt"]),
        ("/src/sub", [], []),
        ("/src/sub2", [], ["b.txt"]),
    ]
    monkeypatch.setattr(syncmod.os, "walk", lambda src: iter(walk_result))
    monkeypatch.setattr(s, "set_destination_subdir_abs", lambda root: root.replace("/src", "/dst", 1))
    monkeypatch.setattr(s, "get_source_subdir_data", lambda files, data: data["src_dir_fls"].extend([os.path.join(data["src_dir_abs"], f) for f in files]))
    monkeypatch.setattr(s, "get_destination_subdir_data", lambda data: data["dst_dir_fls"].append(data["dst_dir_abs"] + "/existing"))

    data = s.get_sync_data()
    assert len(data) == 2
    assert data[0]["src_dir_abs"] == "/src"
    assert data[1]["src_dir_abs"] == "/src/sub2"


def test_do_sync_skips_overwrite_when_disabled():
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "/src", "/dst", overwrite_existing=False)
    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": ["/src/a.txt"],
        "dst_dir_fls": ["/dst/a.txt"],
    }
    copied = []
    s.copy_file = lambda src_file, dst_file: copied.append((src_file, dst_file))

    s.do_sync(sync_data)

    assert copied == []
    assert sync_data["dst_dir_fls"] == []


def test_do_sync_overwrites_existing_when_enabled():
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "/src", "/dst", overwrite_existing=True)
    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": ["/src/a.txt"],
        "dst_dir_fls": ["/dst/a.txt"],
    }
    copied = []
    s.copy_file = lambda src_file, dst_file: copied.append((src_file, dst_file))

    s.do_sync(sync_data)

    assert copied == [("/src/a.txt", "/dst/a.txt")]
    assert sync_data["dst_dir_fls"] == []


def test_handle_destination_dir_data_remove_and_synchronize(monkeypatch):
    s_remove = syncmod.Sync(("mtp:url", "/mnt/device"), "/src", "/dst", unmatched=syncmod.REMOVE)
    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": [],
        "dst_dir_fls": ["/dst/orphan.txt"],
    }
    removed = []
    s_remove.gvfs_wrapper = lambda func, *args: removed.append((func.__name__, args))
    s_remove.handle_destination_dir_data(sync_data)
    assert removed == [("rm", ("/dst/orphan.txt",))]

    s_sync = syncmod.Sync(("mtp:url", "/mnt/device"), "/src", "/dst", unmatched=syncmod.SYNCHRONIZE)
    copied = []
    s_sync.copy_file = lambda src_file, dst_file: copied.append((src_file, dst_file))
    s_sync.handle_destination_dir_data(sync_data)
    assert copied == [("/dst/orphan.txt", "/src/orphan.txt")]


def test_sync_returns_early_when_no_source_files(monkeypatch):
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "/src", "/dst")
    monkeypatch.setattr(s, "get_sync_data", lambda: [{"src_dir_fls": [], "dst_dir_fls": ["x"], "src_dir_abs": "/src", "dst_dir_abs": "/dst"}])
    called = {"do_sync": 0}
    monkeypatch.setattr(s, "do_sync", lambda data: called.__setitem__("do_sync", called["do_sync"] + 1))

    s.sync()
    assert called["do_sync"] == 0


def test_sync_ignores_unmatched_files_when_configured(monkeypatch):
    s = syncmod.Sync(("mtp:url", "/mnt/device"), "/src", "/dst", unmatched=syncmod.IGNORE)
    sync_data = [{
        "src_dir_fls": ["/src/a.txt"],
        "dst_dir_fls": ["/dst/orphan.txt"],
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
    }]
    calls = {"do_sync": 0, "handle": 0}
    monkeypatch.setattr(s, "get_sync_data", lambda: sync_data)
    monkeypatch.setattr(s, "do_sync", lambda data: calls.__setitem__("do_sync", calls["do_sync"] + 1))
    monkeypatch.setattr(s, "handle_destination_dir_data", lambda data: calls.__setitem__("handle", calls["handle"] + 1))

    s.sync()

    assert calls["do_sync"] == 1
    assert calls["handle"] == 0
