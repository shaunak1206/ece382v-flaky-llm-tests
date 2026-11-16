import os

import pytest

from pysyncdroid import exceptions
from pysyncdroid import sync as sync_module
from pysyncdroid.sync import IGNORE, REMOVE, SYNCHRONIZE, Sync, readlink


class FakeGvfs(object):
    def __init__(self):
        self.calls = []

    def cp(self, *args):
        self.calls.append(("cp",) + args)

    def rm(self, *args):
        self.calls.append(("rm",) + args)

    def mkdir(self, *args):
        self.calls.append(("mkdir",) + args)

    def mount(self, *args):
        self.calls.append(("mount",) + args)


@pytest.fixture
def fake_gvfs(monkeypatch):
    fake = FakeGvfs()
    monkeypatch.setattr(sync_module.gvfs, "cp", fake.cp)
    monkeypatch.setattr(sync_module.gvfs, "rm", fake.rm)
    monkeypatch.setattr(sync_module.gvfs, "mkdir", fake.mkdir)
    monkeypatch.setattr(sync_module.gvfs, "mount", fake.mount)
    return fake


@pytest.fixture
def identity_readlink(monkeypatch):
    monkeypatch.setattr(sync_module, "readlink", lambda p: p)


def make_sync(source="/src", destination="/dst", gvfs_path="/gvfs", **kwargs):
    return Sync(("mtp://device/", gvfs_path), source, destination, **kwargs)


def test_readlink_empty_path_returned_as_is():
    assert readlink("") == ""
    assert readlink(None) is None


def test_readlink_uses_bash_cmd_and_expands_home(monkeypatch):
    seen = []

    def fake_run(cmd):
        seen.append(cmd)
        return "/resolved"

    monkeypatch.setattr(sync_module, "run_bash_cmd", fake_run)
    assert readlink("~/foo") == "/resolved"
    assert seen[0][:2] == ["readlink", "-f"]
    assert seen[0][2] == os.path.expanduser("~/foo")
    assert not seen[0][2].startswith("~")


def test_readlink_falls_back_to_original_path(monkeypatch):
    monkeypatch.setattr(sync_module, "run_bash_cmd", lambda cmd: "")
    assert readlink("some/path") == "some/path"


def test_init_stores_attributes_and_lowercases_ignored_types():
    s = make_sync(ignore_file_types=["JPG", "Mp3"], unmatched=REMOVE,
                  overwrite_existing=True, verbose=True)
    assert s.mtp_url == "mtp://device/"
    assert s.mtp_gvfs_path == "/gvfs"
    assert s.source == "/src"
    assert s.destination == "/dst"
    assert s.ignore_file_types == ["jpg", "mp3"]
    assert s.unmatched == REMOVE
    assert s.overwrite_existing is True
    assert s.verbose is True


def test_init_defaults():
    s = make_sync()
    assert s.ignore_file_types is None
    assert s.unmatched == IGNORE
    assert s.overwrite_existing is False
    assert s.verbose is False


def test_verbose_prints_only_when_enabled(capsys):
    make_sync(verbose=False)._verbose("quiet")
    assert capsys.readouterr().out == ""
    make_sync(verbose=True)._verbose("loud")
    assert capsys.readouterr().out == "loud\n"


def test_gvfs_wrapper_remounts_on_connection_reset(fake_gvfs):
    s = make_sync()
    attempts = []

    def flaky(*args):
        attempts.append(args)
        if len(attempts) == 1:
            raise exceptions.BashException("oops: Connection reset by peer\n")

    s.gvfs_wrapper(flaky, "a", "b")
    assert attempts == [("a", "b"), ("a", "b")]
    assert fake_gvfs.calls == [("mount", "mtp://device/")]


def test_gvfs_wrapper_reraises_other_errors(fake_gvfs):
    s = make_sync()

    def broken(*args):
        raise exceptions.BashException("something else")

    with pytest.raises(exceptions.BashException):
        s.gvfs_wrapper(broken, "x")
    assert fake_gvfs.calls == []


def test_set_source_abs_existing_directory(tmp_path, identity_readlink):
    src = tmp_path / "source"
    src.mkdir()
    s = make_sync(source=str(src), gvfs_path=str(tmp_path / "nodevice"))
    s.set_source_abs()
    assert s.source == str(src)


def test_set_source_abs_relative_path_on_device(tmp_path, monkeypatch,
                                                identity_readlink):
    device = tmp_path / "device"
    (device / "Download").mkdir(parents=True)
    other = tmp_path / "cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    s = make_sync(source="Download", gvfs_path=str(device))
    s.set_source_abs()
    assert s.source == os.path.join(str(device), "Download")


def test_set_source_abs_missing_raises(tmp_path, identity_readlink):
    s = make_sync(source=str(tmp_path / "missing"),
                  gvfs_path=str(tmp_path / "device"))
    with pytest.raises(OSError, match="does not exist"):
        s.set_source_abs()


def test_set_source_abs_file_not_directory(tmp_path, identity_readlink):
    f = tmp_path / "file.txt"
    f.write_text("x")
    s = make_sync(source=str(f), gvfs_path=str(tmp_path / "device"))
    with pytest.raises(OSError, match="is not a directory"):
        s.set_source_abs()


def test_set_destination_abs_device_and_computer(identity_readlink):
    s = make_sync(source="/home/user/music", destination="Music",
                  gvfs_path="/run/gvfs/mtp:host=x")
    s.set_destination_abs()
    assert s.destination == os.path.join("/run/gvfs/mtp:host=x", "Music")

    s = make_sync(source="/run/gvfs/mtp:host=x/Music",
                  destination="/home/user/backup",
                  gvfs_path="/run/gvfs/mtp:host=x")
    s.set_destination_abs()
    assert s.destination == "/home/user/backup"


def test_set_destination_subdir_abs():
    s = make_sync(source="/src", destination="/dst")
    assert s.set_destination_subdir_abs("/src") == "/dst/"
    assert s.set_destination_subdir_abs("/src/a/b") == "/dst/a/b"


def test_sync_data_template_strips_trailing_slashes():
    data = make_sync().sync_data_template("/src/a/", "/dst/a/")
    assert data == {
        "src_dir_abs": "/src/a",
        "dst_dir_abs": "/dst/a",
        "src_dir_fls": [],
        "dst_dir_fls": [],
    }


def test_handle_ignored_file_type():
    s = make_sync(ignore_file_types=["TMP"])
    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("/x/file.Tmp")
    s.handle_ignored_file_type("/x/file.txt")
    s.handle_ignored_file_type("/x/noextension")
    make_sync().handle_ignored_file_type("/x/file.tmp")


def test_get_source_subdir_data_skips_ignored():
    s = make_sync(ignore_file_types=["tmp"])
    data = s.sync_data_template("/src", "/dst")
    s.get_source_subdir_data(["a.txt", "b.tmp", "c.md"], data)
    assert data["src_dir_fls"] == ["/src/a.txt", "/src/c.md"]


def test_get_destination_subdir_data_creates_missing_dir(tmp_path, fake_gvfs):
    dst = str(tmp_path / "newdir")
    s = make_sync()
    data = s.sync_data_template("/src", dst)
    s.get_destination_subdir_data(data)
    assert fake_gvfs.calls == [("mkdir", dst)]
    assert data["dst_dir_fls"] == []


def test_get_destination_subdir_data_lists_existing(tmp_path, fake_gvfs):
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.tmp").write_text("2")
    s = make_sync(ignore_file_types=["tmp"])
    data = s.sync_data_template("/src", str(tmp_path))
    s.get_destination_subdir_data(data)
    assert data["dst_dir_fls"] == [str(tmp_path / "a.txt")]
    assert fake_gvfs.calls == []


def test_get_sync_data_skips_dirs_without_files(tmp_path, fake_gvfs):
    src = tmp_path / "src"
    (src / "sub" / "deep").mkdir(parents=True)
    (src / "sub" / "f.txt").write_text("x")
    dst = tmp_path / "dst"
    s = make_sync(source=str(src), destination=str(dst))
    data = s.get_sync_data()
    assert len(data) == 1
    assert data[0]["src_dir_abs"] == str(src / "sub")
    assert data[0]["dst_dir_abs"] == str(dst / "sub")
    assert data[0]["src_dir_fls"] == [str(src / "sub" / "f.txt")]


def test_do_sync_skips_existing_files_by_default(fake_gvfs):
    s = make_sync()
    data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": ["/src/a.txt", "/src/b.txt"],
        "dst_dir_fls": ["/dst/a.txt", "/dst/c.txt"],
    }
    s.do_sync(data)
    assert fake_gvfs.calls == [("cp", "/src/b.txt", "/dst/b.txt")]
    assert data["dst_dir_fls"] == ["/dst/c.txt"]


def test_do_sync_overwrites_when_requested(fake_gvfs):
    s = make_sync(overwrite_existing=True)
    data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": ["/src/a.txt", "/src/b.txt"],
        "dst_dir_fls": ["/dst/a.txt"],
    }
    s.do_sync(data)
    assert fake_gvfs.calls == [
        ("cp", "/src/a.txt", "/dst/a.txt"),
        ("cp", "/src/b.txt", "/dst/b.txt"),
    ]


def test_handle_destination_dir_data_remove_and_synchronize(fake_gvfs):
    data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": [],
        "dst_dir_fls": ["/dst/c.txt"],
    }
    make_sync(unmatched=REMOVE).handle_destination_dir_data(dict(data))
    assert fake_gvfs.calls == [("rm", "/dst/c.txt")]

    fake_gvfs.calls[:] = []
    make_sync(unmatched=SYNCHRONIZE).handle_destination_dir_data(dict(data))
    assert fake_gvfs.calls == [("cp", "/dst/c.txt", "/src/c.txt")]

    fake_gvfs.calls[:] = []
    make_sync(unmatched=REMOVE).handle_destination_dir_data(
        {"src_dir_abs": "/src", "dst_dir_abs": "/dst",
         "src_dir_fls": [], "dst_dir_fls": []}
    )
    assert fake_gvfs.calls == []


def test_sync_full_flow_with_remove(tmp_path, fake_gvfs):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("a")
    (src / "skip.tmp").write_text("t")
    (dst / "old.txt").write_text("old")
    s = make_sync(source=str(src), destination=str(dst), unmatched=REMOVE,
                  ignore_file_types=["tmp"])
    s.sync()
    assert ("cp", str(src / "a.txt"), str(dst / "a.txt")) in fake_gvfs.calls
    assert ("rm", str(dst / "old.txt")) in fake_gvfs.calls
    assert not any(c[0] == "cp" and c[1].endswith("skip.tmp")
                   for c in fake_gvfs.calls)


def test_sync_ignore_leaves_unmatched_files(tmp_path, fake_gvfs):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("a")
    (dst / "old.txt").write_text("old")
    s = make_sync(source=str(src), destination=str(dst), unmatched=IGNORE)
    s.sync()
    assert fake_gvfs.calls == [("cp", str(src / "a.txt"), str(dst / "a.txt"))]
