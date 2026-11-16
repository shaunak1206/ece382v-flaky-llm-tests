import os
import shutil

import pytest

import pysyncdroid
from pysyncdroid import exceptions
from pysyncdroid import sync as sync_module
from pysyncdroid.sync import Sync, IGNORE, REMOVE, SYNCHRONIZE


MTP = ("mtp://host/", "/nonexistent/gvfs/mtp:host=dev")


def make_sync(**kwargs):
    params = dict(mtp_details=MTP, source="/src", destination="/dst")
    params.update(kwargs)
    return Sync(**params)


@pytest.fixture
def calls(monkeypatch):
    """Patch gvfs operations to record calls and act on the local fs."""
    rec = {"cp": [], "rm": [], "mkdir": [], "mount": []}

    def cp(src, dst):
        rec["cp"].append((src, dst))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    def rm(path):
        rec["rm"].append(path)
        os.remove(path)

    def mkdir(path):
        rec["mkdir"].append(path)
        os.makedirs(path, exist_ok=True)

    def mount(url):
        rec["mount"].append(url)

    monkeypatch.setattr(sync_module.gvfs, "cp", cp)
    monkeypatch.setattr(sync_module.gvfs, "rm", rm)
    monkeypatch.setattr(sync_module.gvfs, "mkdir", mkdir)
    monkeypatch.setattr(sync_module.gvfs, "mount", mount)
    return rec


def test_package_importable():
    assert pysyncdroid.__name__ == "pysyncdroid"


def test_constants():
    assert IGNORE == "ignore"
    assert REMOVE == "remove"
    assert SYNCHRONIZE == "synchronize"


def test_readlink_empty_path():
    assert sync_module.readlink("") == ""
    assert sync_module.readlink(None) is None


def test_readlink_uses_command_and_falls_back(monkeypatch):
    seen = []

    def fake(cmd):
        seen.append(cmd)
        return "/resolved"

    monkeypatch.setattr(sync_module, "run_bash_cmd", fake)
    assert sync_module.readlink("/some/path") == "/resolved"
    assert seen[0] == ["readlink", "-f", "/some/path"]

    monkeypatch.setattr(sync_module, "run_bash_cmd", lambda cmd: "")
    assert sync_module.readlink("/some/path") == "/some/path"


def test_readlink_expands_tilde(monkeypatch):
    seen = []
    monkeypatch.setattr(
        sync_module, "run_bash_cmd", lambda cmd: seen.append(cmd) or ""
    )
    result = sync_module.readlink("~/foo")
    assert seen[0][-1] == os.path.expanduser("~/foo")
    assert result == os.path.expanduser("~/foo")


def test_init_stores_attributes_and_lowercases_types():
    s = make_sync(ignore_file_types=["MP3", "Txt"], unmatched=REMOVE)
    assert s.mtp_url == "mtp://host/"
    assert s.mtp_gvfs_path == "/nonexistent/gvfs/mtp:host=dev"
    assert s.source == "/src"
    assert s.destination == "/dst"
    assert s.unmatched == REMOVE
    assert s.overwrite_existing is False
    assert s.ignore_file_types == ["mp3", "txt"]
    assert make_sync().ignore_file_types is None


def test_verbose_output(capsys):
    make_sync(verbose=True)._verbose("hello")
    assert "hello" in capsys.readouterr().out
    make_sync(verbose=False)._verbose("hello")
    assert capsys.readouterr().out == ""


def test_gvfs_wrapper_remounts_on_connection_reset(calls):
    s = make_sync()
    attempts = []

    def flaky(arg):
        attempts.append(arg)
        if len(attempts) == 1:
            raise exceptions.BashException("error: Connection reset by peer\n")

    s.gvfs_wrapper(flaky, "x")
    assert attempts == ["x", "x"]
    assert calls["mount"] == ["mtp://host/"]


def test_gvfs_wrapper_reraises_other_errors(calls):
    s = make_sync()

    def broken():
        raise exceptions.BashException("something else")

    with pytest.raises(exceptions.BashException):
        s.gvfs_wrapper(broken)
    assert calls["mount"] == []


def test_set_source_abs_existing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_module, "readlink", lambda p: p)
    src = tmp_path / "source"
    src.mkdir()
    s = make_sync(source=str(src))
    s.set_source_abs()
    assert s.source == str(src)


def test_set_source_abs_relative_on_device(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_module, "readlink", lambda p: p)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    device = tmp_path / "device"
    (device / "Music").mkdir(parents=True)
    monkeypatch.chdir(cwd)
    s = make_sync(
        mtp_details=("mtp://host/", str(device)), source="Music"
    )
    s.set_source_abs()
    assert s.source == os.path.join(str(device), "Music")


def test_set_source_abs_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_module, "readlink", lambda p: p)
    missing = make_sync(source=str(tmp_path / "nope"))
    with pytest.raises(OSError, match="does not exist"):
        missing.set_source_abs()

    f = tmp_path / "file.txt"
    f.write_text("x")
    notdir = make_sync(source=str(f))
    with pytest.raises(OSError, match="not a directory"):
        notdir.set_source_abs()


def test_set_destination_abs_device_and_computer(monkeypatch):
    monkeypatch.setattr(sync_module, "readlink", lambda p: p)
    to_device = make_sync(source="/home/me/music", destination="Music")
    to_device.set_destination_abs()
    assert to_device.destination == os.path.join(MTP[1], "Music")

    to_computer = make_sync(
        source="/run/user/1000/gvfs/mtp:host=dev/Music",
        destination="/home/me/backup",
    )
    to_computer.set_destination_abs()
    assert to_computer.destination == "/home/me/backup"


def test_set_destination_subdir_abs():
    s = make_sync(source="/a/b", destination="/d")
    assert s.set_destination_subdir_abs("/a/b/c/e") == "/d/c/e"
    assert s.set_destination_subdir_abs("/a/b") == os.path.join("/d", "")


def test_sync_data_template_strips_trailing_slash():
    data = make_sync().sync_data_template("/a/b/", "/c/d/")
    assert data == {
        "src_dir_abs": "/a/b",
        "dst_dir_abs": "/c/d",
        "src_dir_fls": [],
        "dst_dir_fls": [],
    }


def test_handle_ignored_file_type():
    s = make_sync(ignore_file_types=["mp3"])
    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("/x/song.MP3")
    s.handle_ignored_file_type("/x/song.flac")
    s.handle_ignored_file_type("/x/noextension")
    make_sync().handle_ignored_file_type("/x/song.mp3")


def test_get_source_subdir_data_skips_ignored():
    s = make_sync(ignore_file_types=["tmp"])
    data = s.sync_data_template("/src", "/dst")
    s.get_source_subdir_data(["a.txt", "b.tmp", "c.jpg"], data)
    assert data["src_dir_fls"] == ["/src/a.txt", "/src/c.jpg"]


def test_get_destination_subdir_data_creates_missing(tmp_path, calls):
    dst = tmp_path / "new" / "dir"
    s = make_sync()
    data = s.sync_data_template("/src", str(dst))
    s.get_destination_subdir_data(data)
    assert calls["mkdir"] == [str(dst)]
    assert dst.is_dir()
    assert data["dst_dir_fls"] == []


def test_get_destination_subdir_data_lists_existing(tmp_path, calls):
    (tmp_path / "keep.txt").write_text("1")
    (tmp_path / "skip.tmp").write_text("2")
    s = make_sync(ignore_file_types=["tmp"])
    data = s.sync_data_template("/src", str(tmp_path))
    s.get_destination_subdir_data(data)
    assert data["dst_dir_fls"] == [str(tmp_path / "keep.txt")]
    assert calls["mkdir"] == []


def test_do_sync_skips_existing_unless_overwrite(calls):
    def data():
        return {
            "src_dir_abs": "/src",
            "dst_dir_abs": "/dst",
            "src_dir_fls": ["/src/a", "/src/b"],
            "dst_dir_fls": ["/dst/a", "/dst/z"],
        }

    copied = []
    s = make_sync()
    s.copy_file = lambda src_file, dst_file: copied.append((src_file, dst_file))
    d = data()
    s.do_sync(d)
    assert copied == [("/src/b", "/dst/b")]
    assert d["dst_dir_fls"] == ["/dst/z"]

    copied.clear()
    s = make_sync(overwrite_existing=True)
    s.copy_file = lambda src_file, dst_file: copied.append((src_file, dst_file))
    s.do_sync(data())
    assert copied == [("/src/a", "/dst/a"), ("/src/b", "/dst/b")]


def test_handle_destination_dir_data_remove_and_synchronize(calls):
    sd = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": [],
        "dst_dir_fls": ["/dst/x"],
    }
    removed = []
    copied = []

    s = make_sync(unmatched=REMOVE)
    s.gvfs_wrapper = lambda func, *a: removed.append((func, a))
    s.handle_destination_dir_data(sd)
    assert removed == [(sync_module.gvfs.rm, ("/dst/x",))]

    s = make_sync(unmatched=SYNCHRONIZE)
    s.copy_file = lambda src_file, dst_file: copied.append((src_file, dst_file))
    s.handle_destination_dir_data(sd)
    assert copied == [("/dst/x", "/src/x")]


def test_full_sync_copies_and_removes_unmatched(tmp_path, calls):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    dst.mkdir()
    (src / "a.txt").write_text("a")
    (src / "ignored.tmp").write_text("t")
    (src / "sub" / "b.txt").write_text("b")
    (dst / "extra.txt").write_text("e")

    s = make_sync(
        source=str(src),
        destination=str(dst),
        unmatched=REMOVE,
        ignore_file_types=["tmp"],
    )
    s.sync()

    assert (dst / "a.txt").read_text() == "a"
    assert (dst / "sub" / "b.txt").read_text() == "b"
    assert not (dst / "ignored.tmp").exists()
    assert not (dst / "extra.txt").exists()
    assert str(dst / "extra.txt") in calls["rm"]


def test_full_sync_ignore_keeps_unmatched(tmp_path, calls):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("a")
    (dst / "extra.txt").write_text("e")

    s = make_sync(source=str(src), destination=str(dst), unmatched=IGNORE)
    s.sync()

    assert (dst / "a.txt").exists()
    assert (dst / "extra.txt").exists()
    assert calls["rm"] == []
