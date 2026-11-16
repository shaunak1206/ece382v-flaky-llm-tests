# -*- coding: utf-8 -*-
import os

import pytest

import pysyncdroid
from pysyncdroid import exceptions
from pysyncdroid import sync as sync_module
from pysyncdroid.sync import (
    IGNORE,
    REMOVE,
    SYNCHRONIZE,
    Sync,
    readlink,
)


def make_sync(source="/src", destination="/dst", **kwargs):
    return Sync(
        ("mtp://host/", "/run/user/1000/gvfs/mtp:host=x"),
        source,
        destination,
        **kwargs
    )


def test_package_importable():
    assert pysyncdroid.__name__ == "pysyncdroid"


def test_constants():
    assert IGNORE == "ignore"
    assert REMOVE == "remove"
    assert SYNCHRONIZE == "synchronize"


def test_readlink_empty_returns_input():
    assert readlink("") == ""
    assert readlink(None) is None


def test_readlink_expands_user_and_uses_result(monkeypatch):
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        return "/resolved"

    monkeypatch.setattr(sync_module, "run_bash_cmd", fake_run)
    assert readlink("~/foo") == "/resolved"
    assert calls[0][:2] == ["readlink", "-f"]
    assert not calls[0][2].startswith("~")


def test_readlink_falls_back_to_path(monkeypatch):
    monkeypatch.setattr(sync_module, "run_bash_cmd", lambda cmd: "")
    assert readlink("/some/path") == "/some/path"


def test_init_defaults_and_lowercase_ignore_types():
    s = make_sync(ignore_file_types=["JPG", "Mp3"])
    assert s.mtp_url == "mtp://host/"
    assert s.mtp_gvfs_path == "/run/user/1000/gvfs/mtp:host=x"
    assert s.ignore_file_types == ["jpg", "mp3"]
    assert s.unmatched == IGNORE
    assert s.overwrite_existing is False
    assert make_sync().ignore_file_types is None


def test_verbose_prints_only_when_enabled(capsys):
    make_sync(verbose=True)._verbose("hello")
    assert "hello" in capsys.readouterr().out
    make_sync(verbose=False)._verbose("hello")
    assert capsys.readouterr().out == ""


def test_gvfs_wrapper_success():
    s = make_sync()
    calls = []
    s.gvfs_wrapper(lambda *a: calls.append(a), 1, 2)
    assert calls == [(1, 2)]


def test_gvfs_wrapper_remounts_on_connection_reset(monkeypatch):
    s = make_sync()
    mounted = []
    monkeypatch.setattr(sync_module.gvfs, "mount", mounted.append)
    state = {"n": 0}

    def func(*args):
        state["n"] += 1
        if state["n"] == 1:
            raise exceptions.BashException("error: Connection reset by peer\n")

    s.gvfs_wrapper(func, "a")
    assert state["n"] == 2
    assert mounted == ["mtp://host/"]


def test_gvfs_wrapper_reraises_other_errors():
    s = make_sync()

    def func(*args):
        raise exceptions.BashException("something else")

    with pytest.raises(exceptions.BashException):
        s.gvfs_wrapper(func)


def test_handle_ignored_file_type():
    s = make_sync(ignore_file_types=["JPG"])
    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("/a/b/pic.jpg")
    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("PIC.JPG")
    s.handle_ignored_file_type("doc.txt")
    s.handle_ignored_file_type("noextension")
    make_sync().handle_ignored_file_type("pic.jpg")


def test_set_destination_subdir_abs():
    s = make_sync(source="/src", destination="/dst")
    assert s.set_destination_subdir_abs("/src/a/b") == "/dst/a/b"
    assert s.set_destination_subdir_abs("/src") == "/dst/"


def test_sync_data_template_strips_trailing_slash():
    d = make_sync().sync_data_template("/src/a/", "/dst/a/")
    assert d == {
        "src_dir_abs": "/src/a",
        "dst_dir_abs": "/dst/a",
        "src_dir_fls": [],
        "dst_dir_fls": [],
    }


def test_get_source_subdir_data_skips_ignored():
    s = make_sync(ignore_file_types=["tmp"])
    data = s.sync_data_template("/src", "/dst")
    s.get_source_subdir_data(["a.txt", "b.tmp", "c.md"], data)
    assert data["src_dir_fls"] == ["/src/a.txt", "/src/c.md"]


def test_set_source_abs_existing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_module, "readlink", lambda p: p)
    s = make_sync(source=str(tmp_path))
    s.set_source_abs()
    assert s.source == str(tmp_path)


def test_set_source_abs_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_module, "readlink", lambda p: p)
    s = make_sync(source=str(tmp_path / "missing"))
    s.mtp_gvfs_path = str(tmp_path)
    with pytest.raises(OSError, match="does not exist"):
        s.set_source_abs()

    f = tmp_path / "file.txt"
    f.write_text("x")
    s = make_sync(source=str(f))
    with pytest.raises(OSError, match="not a directory"):
        s.set_source_abs()


def test_set_destination_abs_device_and_computer(monkeypatch):
    monkeypatch.setattr(sync_module, "readlink", lambda p: p)
    s = make_sync(source="/local/src", destination="Music")
    s.set_destination_abs()
    assert s.destination == os.path.join(s.mtp_gvfs_path, "Music")

    s = make_sync(source="/run/user/1000/gvfs/mtp:host=x/DCIM",
                  destination="/home/me/pics")
    s.set_destination_abs()
    assert s.destination == "/home/me/pics"


def test_do_sync_copies_new_and_respects_overwrite(monkeypatch):
    copied = []
    monkeypatch.setattr(sync_module.gvfs, "cp",
                        lambda s, d: copied.append((s, d)))

    def data():
        return {
            "src_dir_abs": "/src",
            "dst_dir_abs": "/dst",
            "src_dir_fls": ["/src/a", "/src/b"],
            "dst_dir_fls": ["/dst/a", "/dst/extra"],
        }

    d = data()
    make_sync().do_sync(d)
    assert copied == [("/src/b", "/dst/b")]
    assert d["dst_dir_fls"] == ["/dst/extra"]

    copied.clear()
    make_sync(overwrite_existing=True).do_sync(data())
    assert copied == [("/src/a", "/dst/a"), ("/src/b", "/dst/b")]


def test_handle_destination_dir_data_remove_and_synchronize(monkeypatch):
    removed, copied = [], []
    monkeypatch.setattr(sync_module.gvfs, "rm", removed.append)
    monkeypatch.setattr(sync_module.gvfs, "cp",
                        lambda s, d: copied.append((s, d)))
    data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": [],
        "dst_dir_fls": ["/dst/x"],
    }
    make_sync(unmatched=REMOVE).handle_destination_dir_data(data)
    assert removed == ["/dst/x"] and copied == []

    removed.clear()
    make_sync(unmatched=SYNCHRONIZE).handle_destination_dir_data(data)
    assert copied == [("/dst/x", "/src/x")] and removed == []

    make_sync(unmatched=IGNORE).handle_destination_dir_data(data)
    assert removed == [] and len(copied) == 1


def test_get_sync_data_and_full_sync(tmp_path, monkeypatch):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("a")
    (src / "sub" / "b.txt").write_text("b")
    dst.mkdir()
    (dst / "old.txt").write_text("old")

    copied, removed, made = [], [], []
    monkeypatch.setattr(sync_module.gvfs, "cp",
                        lambda s, d: copied.append((s, d)))
    monkeypatch.setattr(sync_module.gvfs, "rm", removed.append)
    monkeypatch.setattr(sync_module.gvfs, "mkdir", made.append)

    s = make_sync(source=str(src), destination=str(dst), unmatched=REMOVE)
    s.sync()

    assert (str(src / "a.txt"), str(dst / "a.txt")) in copied
    assert (str(src / "sub" / "b.txt"), str(dst / "sub" / "b.txt")) in copied
    assert removed == [str(dst / "old.txt")]
    assert made == [str(dst / "sub")]
