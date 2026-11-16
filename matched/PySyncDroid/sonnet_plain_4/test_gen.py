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


MTP_DETAILS = ("mtp://host/", "/run/user/1000/gvfs/mtp:host=device")


def make_sync(**kwargs):
    params = dict(
        mtp_details=MTP_DETAILS,
        source="/src",
        destination="/dst",
    )
    params.update(kwargs)
    return Sync(**params)


def test_init_basic_attributes():
    s = make_sync(unmatched=REMOVE, overwrite_existing=True, verbose=True)
    assert s.mtp_url == MTP_DETAILS[0]
    assert s.mtp_gvfs_path == MTP_DETAILS[1]
    assert s.source == "/src"
    assert s.destination == "/dst"
    assert s.unmatched == REMOVE
    assert s.overwrite_existing is True
    assert s.verbose is True
    assert s.ignore_file_types is None


def test_init_ignore_file_types_lowercased():
    s = make_sync(ignore_file_types=["JPG", "Mp3"])
    assert s.ignore_file_types == ["jpg", "mp3"]


def test_readlink_empty_returns_input():
    assert readlink("") == ""
    assert readlink(None) is None


def test_readlink_expands_user_and_calls_cmd(monkeypatch):
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        return "/resolved"

    monkeypatch.setattr(sync_module, "run_bash_cmd", fake_run)
    assert readlink("~/foo") == "/resolved"
    assert calls[0][:2] == ["readlink", "-f"]
    assert calls[0][2] == os.path.expanduser("~/foo")


def test_readlink_falls_back_to_path(monkeypatch):
    monkeypatch.setattr(sync_module, "run_bash_cmd", lambda cmd: "")
    assert readlink("some/path") == "some/path"


def test_verbose_prints_only_when_enabled(capsys):
    make_sync(verbose=True)._verbose("hello")
    assert "hello" in capsys.readouterr().out
    make_sync(verbose=False)._verbose("hello")
    assert capsys.readouterr().out == ""


def test_gvfs_wrapper_success_calls_once():
    calls = []
    s = make_sync()
    s.gvfs_wrapper(lambda *a: calls.append(a), 1, 2)
    assert calls == [(1, 2)]


def test_gvfs_wrapper_remounts_on_connection_reset(monkeypatch):
    mounted = []
    monkeypatch.setattr(sync_module.gvfs, "mount", mounted.append)
    state = {"n": 0}

    def func(*args):
        state["n"] += 1
        if state["n"] == 1:
            raise exceptions.BashException("error: Connection reset by peer\n")

    s = make_sync()
    s.gvfs_wrapper(func, "x")
    assert state["n"] == 2
    assert mounted == [MTP_DETAILS[0]]


def test_gvfs_wrapper_reraises_other_errors(monkeypatch):
    def func():
        raise exceptions.BashException("something else")

    with pytest.raises(exceptions.BashException):
        make_sync().gvfs_wrapper(func)


def test_set_source_abs_existing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_module, "readlink", lambda p: p)
    s = make_sync(source=str(tmp_path))
    s.set_source_abs()
    assert s.source == str(tmp_path)


def test_set_source_abs_relative_uses_cwd(tmp_path, monkeypatch):
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sync_module, "readlink", lambda p: p)
    s = make_sync(source="sub")
    s.set_source_abs()
    assert s.source == os.path.join(os.getcwd(), "sub")


def test_set_source_abs_nonexistent_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_module, "readlink", lambda p: p)
    s = make_sync(
        source=str(tmp_path / "missing"),
        mtp_details=("mtp://x/", str(tmp_path)),
    )
    with pytest.raises(OSError, match="does not exist"):
        s.set_source_abs()


def test_set_source_abs_file_raises(tmp_path, monkeypatch):
    f = tmp_path / "file.txt"
    f.write_text("x")
    monkeypatch.setattr(sync_module, "readlink", lambda p: p)
    s = make_sync(source=str(f))
    with pytest.raises(OSError, match="not a directory"):
        s.set_source_abs()


def test_set_destination_abs_device_and_computer(monkeypatch):
    monkeypatch.setattr(sync_module, "readlink", lambda p: p)
    s = make_sync(source="/home/user/music", destination="Music")
    s.set_destination_abs()
    assert s.destination == os.path.join(MTP_DETAILS[1], "Music")

    s = make_sync(
        source=MTP_DETAILS[1] + "/Music", destination="/home/user/out"
    )
    s.set_destination_abs()
    assert s.destination == "/home/user/out"


def test_set_destination_subdir_abs():
    s = make_sync(source="/src", destination="/dst")
    assert s.set_destination_subdir_abs("/src/a/b") == "/dst/a/b"
    assert s.set_destination_subdir_abs("/src") == "/dst/"


def test_sync_data_template_strips_trailing_slashes():
    data = make_sync().sync_data_template("/src/a/", "/dst/a/")
    assert data == {
        "src_dir_abs": "/src/a",
        "dst_dir_abs": "/dst/a",
        "src_dir_fls": [],
        "dst_dir_fls": [],
    }


def test_handle_ignored_file_type():
    s = make_sync(ignore_file_types=["jpg"])
    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("photo.JPG")
    s.handle_ignored_file_type("song.mp3")
    s.handle_ignored_file_type("noextension")
    make_sync().handle_ignored_file_type("photo.jpg")


def test_get_source_subdir_data_filters_ignored():
    s = make_sync(ignore_file_types=["tmp"])
    data = s.sync_data_template("/src", "/dst")
    s.get_source_subdir_data(["a.txt", "b.tmp", "c.mp3"], data)
    assert data["src_dir_fls"] == ["/src/a.txt", "/src/c.mp3"]


def test_get_destination_subdir_data_existing_and_missing(
    tmp_path, monkeypatch
):
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "a.txt").write_text("1")
    (dst / "b.tmp").write_text("2")
    s = make_sync(ignore_file_types=["tmp"])
    data = s.sync_data_template("/src", str(dst))
    s.get_destination_subdir_data(data)
    assert data["dst_dir_fls"] == [str(dst / "a.txt")]

    made = []
    monkeypatch.setattr(sync_module.gvfs, "mkdir", made.append)
    missing = str(tmp_path / "missing")
    data = s.sync_data_template("/src", missing)
    s.get_destination_subdir_data(data)
    assert made == [missing]


def test_get_sync_data_walks_and_skips_empty_dirs(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "empty" / "inner").mkdir(parents=True)
    (src / "empty" / "inner" / "f.txt").write_text("x")
    (src / "root.txt").write_text("y")
    dst.mkdir()
    (dst / "old.txt").write_text("z")

    s = make_sync(source=str(src), destination=str(dst))
    s.gvfs_wrapper = lambda func, *a: None  # avoid real mkdir
    data = s.get_sync_data()

    by_src = {d["src_dir_abs"]: d for d in data}
    assert set(by_src) == {str(src), str(src / "empty" / "inner")}
    assert by_src[str(src)]["src_dir_fls"] == [str(src / "root.txt")]
    assert by_src[str(src)]["dst_dir_fls"] == [str(dst / "old.txt")]


def test_do_sync_skips_existing_unless_overwrite(monkeypatch):
    copied = []
    monkeypatch.setattr(
        sync_module.gvfs, "cp", lambda s, d: copied.append((s, d))
    )

    def make_data():
        return {
            "src_dir_abs": "/src",
            "dst_dir_abs": "/dst",
            "src_dir_fls": ["/src/a", "/src/b"],
            "dst_dir_fls": ["/dst/a", "/dst/c"],
        }

    data = make_data()
    make_sync().do_sync(data)
    assert copied == [("/src/b", "/dst/b")]
    assert data["dst_dir_fls"] == ["/dst/c"]

    copied.clear()
    make_sync(overwrite_existing=True).do_sync(make_data())
    assert copied == [("/src/a", "/dst/a"), ("/src/b", "/dst/b")]


def test_handle_destination_dir_data_remove_and_synchronize(monkeypatch):
    removed, copied = [], []
    monkeypatch.setattr(sync_module.gvfs, "rm", removed.append)
    monkeypatch.setattr(
        sync_module.gvfs, "cp", lambda s, d: copied.append((s, d))
    )

    def make_data():
        return {
            "src_dir_abs": "/src",
            "dst_dir_abs": "/dst",
            "src_dir_fls": [],
            "dst_dir_fls": ["/dst/x"],
        }

    make_sync(unmatched=REMOVE).handle_destination_dir_data(make_data())
    assert removed == ["/dst/x"]
    assert copied == []

    make_sync(unmatched=SYNCHRONIZE).handle_destination_dir_data(make_data())
    assert copied == [("/dst/x", "/src/x")]

    removed.clear()
    copied.clear()
    make_sync(unmatched=IGNORE).handle_destination_dir_data(make_data())
    assert removed == [] and copied == []


def test_sync_end_to_end(tmp_path, monkeypatch):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("a")
    (dst / "stale.txt").write_text("s")

    copied, removed = [], []
    monkeypatch.setattr(
        sync_module.gvfs, "cp", lambda s, d: copied.append((s, d))
    )
    monkeypatch.setattr(sync_module.gvfs, "rm", removed.append)

    s = make_sync(source=str(src), destination=str(dst), unmatched=REMOVE)
    s.sync()
    assert copied == [(str(src / "a.txt"), str(dst / "a.txt"))]
    assert removed == [str(dst / "stale.txt")]


def test_sync_no_files_does_nothing(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        sync_module.gvfs, "cp", lambda s, d: called.append((s, d))
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.tmp").write_text("a")
    dst = tmp_path / "dst"
    dst.mkdir()
    s = make_sync(
        source=str(src), destination=str(dst), ignore_file_types=["tmp"]
    )
    s.sync()
    assert called == []
