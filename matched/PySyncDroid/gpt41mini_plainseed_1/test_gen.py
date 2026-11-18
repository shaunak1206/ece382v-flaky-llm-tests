import os
import builtins
import types
import pytest

import pysyncdroid.sync as sync
from pysyncdroid import exceptions


@pytest.fixture
def dummy_mtp_details(tmp_path):
    # dummy MTP url and gvfs path
    return ("mtp://dummy", tmp_path.as_posix())


def test_readlink_none_and_empty():
    # empty string returns empty string
    assert sync.readlink("") == ""
    # None returns None
    assert sync.readlink(None) is None


def test_readlink_expanduser(monkeypatch):
    called = {}
    def fake_run(cmd):
        called["cmd"] = cmd
        return "/expanded/path"
    monkeypatch.setattr(sync, "run_bash_cmd", fake_run)
    # path starting with ~ should be expanded before run_bash_cmd
    path = "~/some/path"
    resolved = sync.readlink(path)
    assert called["cmd"] == ["readlink", "-f", os.path.expanduser(path)]
    assert resolved == "/expanded/path"


def test_Readlink_no_result(monkeypatch):
    # Simulate run_bash_cmd returning None, so original path returned
    monkeypatch.setattr(sync, "run_bash_cmd", lambda cmd: None)
    path = "/some/path"
    # Should return path itself since run_bash_cmd returns None
    assert sync.readlink(path) == path


def test_sync_init_lowercases_ignore_file_types():
    obj = sync.Sync(
        mtp_details=("url", "gvfspath"),
        source="/src",
        destination="/dst",
        ignore_file_types=["TXT", "Mp3"]
    )
    assert obj.ignore_file_types == ["txt", "mp3"]


def test_sync_init_ignore_file_types_none():
    obj = sync.Sync(
        mtp_details=("url", "gvfspath"),
        source="/src",
        destination="/dst",
        ignore_file_types=None
    )
    assert obj.ignore_file_types is None


def test_verbose_prints(capsys):
    s = sync.Sync(("url", "gvfspath"), "/s", "/d", verbose=True)
    s._verbose("hello")
    captured = capsys.readouterr()
    assert "hello" in captured.out

    s2 = sync.Sync(("url", "gvfspath"), "/s", "/d", verbose=False)
    s2._verbose("hello")
    captured = capsys.readouterr()
    # No output expected
    assert captured.out == ""


def test_gvfs_wrapper_calls_func(monkeypatch):
    called = {}

    def fake_func(arg1, arg2):
        called["args"] = (arg1, arg2)
    s = sync.Sync(("url", "gvfspath"), "/s", "/d")

    s.gvfs_wrapper(fake_func, 1, 2)
    assert called["args"] == (1, 2)


def test_gvfs_wrapper_retries_on_connection_reset(monkeypatch):
    calls = {"count": 0}

    def faulty_func(arg):
        calls["count"] += 1
        if calls["count"] == 1:
            raise exceptions.BashException("Error: Connection reset by peer")
        calls["called_args"] = arg

    mounted = {"called": False}

    def fake_mount(mtp_url):
        mounted["called"] = True
        mounted["url"] = mtp_url

    s = sync.Sync(("mtp://url", "/gvfs/path"), "/src", "/dst")
    monkeypatch.setattr(sync.gvfs, "mount", fake_mount)

    s.gvfs_wrapper(faulty_func, 42)

    assert calls["count"] == 2
    assert mounted["called"] is True
    assert mounted["url"] == "mtp://url"
    assert calls["called_args"] == 42


def test_gvfs_wrapper_propagates_other_errors():
    def raise_other():
        raise exceptions.BashException("Some other error")
    s = sync.Sync(("url", "gvfspath"), "/src", "/dst")

    with pytest.raises(exceptions.BashException):
        s.gvfs_wrapper(raise_other)


def test_set_source_abs_existing_dir(tmp_path, monkeypatch):
    # Create directory in current working directory to simulate source existing there
    src_dir = tmp_path / "source_dir"
    src_dir.mkdir()
    src_rel = "source_dir"

    s = sync.Sync(("url", str(tmp_path)), src_rel, "/dst")

    monkeypatch.setattr(sync, "readlink", lambda p: p)
    monkeypatch.setattr(os.path, "isabs", os.path.isabs)
    monkeypatch.setattr(os.path, "getcwd", lambda: tmp_path.as_posix())
    monkeypatch.setattr(os.path, "exists", lambda p: p == src_dir.as_posix())
    monkeypatch.setattr(os.path, "isdir", lambda p: p == src_dir.as_posix())

    s.set_source_abs()
    assert s.source == src_dir.as_posix()


def test_set_source_abs_raises_if_not_exist(monkeypatch):
    monkeypatch.setattr(sync, "readlink", lambda p: p)
    monkeypatch.setattr(os.path, "isabs", lambda p: os.path.isabs(p))
    monkeypatch.setattr(os.path, "getcwd", lambda: "/cwd")
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    s = sync.Sync(("url", "/gvfs"), "/nonexistent", "/dst")

    with pytest.raises(OSError, match="does not exist on computer or on device"):
        s.set_source_abs()


def test_set_source_abs_raises_if_not_dir(tmp_path, monkeypatch):
    file_p = tmp_path / "file.txt"
    file_p.write_text("data")
    monkeypatch.setattr(sync, "readlink", lambda p: p)
    monkeypatch.setattr(os.path, "isabs", os.path.isabs)
    monkeypatch.setattr(os.path, "getcwd", lambda: tmp_path.as_posix())
    monkeypatch.setattr(os.path, "exists", lambda p: p == file_p.as_posix())
    monkeypatch.setattr(os.path, "isdir", lambda p: False if p == file_p.as_posix() else True)

    s = sync.Sync(("url", str(tmp_path)), file_p.name, "/dst")
    with pytest.raises(OSError, match="is not a directory"):
        s.set_source_abs()


def test_set_destination_abs_source_contains_mtp(monkeypatch):
    monkeypatch.setattr(sync, "readlink", lambda p: p)
    s = sync.Sync(("url", "/gvfs_path"), "mtp:host", "destdir")
    s.set_destination_abs()
    # if source contains "mtp:host", destination is absolute (no joining with gvfs_path)
    assert s.destination == "destdir"


def test_set_destination_abs_source_not_mtp(monkeypatch):
    monkeypatch.setattr(sync, "readlink", lambda p: p)
    s = sync.Sync(("url", "/gvfs_path"), "/some/source", "destdir")
    s.set_destination_abs()
    # destination joined to gvfs_path
    assert s.destination == os.path.join("/gvfs_path", "destdir")


def test_set_destination_subdir_abs_removes_leading_slash():
    s = sync.Sync(("url", "/gvfs_path"), "/base/source", "/dest")
    s.source = "/base/source"
    dest_dir = "/dest"
    # Case where src_subdir_abs is source itself
    start = s.set_destination_subdir_abs(s.source)
    assert start == dest_dir
    # Case where src_subdir_abs includes a subdir, with leading slash stripped in return
    sub = os.path.join(s.source, "subdir")
    expected = os.path.join(dest_dir, "subdir")
    assert s.set_destination_subdir_abs(sub) == expected


def test_sync_data_template_basic():
    s = sync.Sync(("url", "/gvfs"), "/src", "/dst")
    d = s.sync_data_template("/src/subdir", "/dst/subdir")
    assert d["src_dir_abs"] == "/src/subdir"
    assert d["dst_dir_abs"] == "/dst/subdir"
    assert d["src_dir_fls"] == []
    assert d["dst_dir_fls"] == []


def test_handle_ignored_file_type_raises(monkeypatch):
    s = sync.Sync(("url", "/gvfs"), "/src", "/dst", ignore_file_types=["txt"])
    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("file.TXT")
    # not ignored extension raises nothing
    s.handle_ignored_file_type("file.mp3")


def test_handle_ignored_file_type_none_ignore():
    s = sync.Sync(("url", "/gvfs"), "/src", "/dst", ignore_file_types=None)
    # should not raise anything
    s.handle_ignored_file_type("file.anyext")


def test_get_source_subdir_data_filters(monkeypatch):
    s = sync.Sync(("url","/gvfs"), "/src", "/dst", ignore_file_types=["ignore"])
    sync_data = {"src_dir_abs": "/src/subdir", "src_dir_fls": []}
    files = ["good.mp3", "bad.ignore", "other.txt"]

    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("bad.ignore")  # confirm exception type

    s.get_source_subdir_data(files, sync_data)
    # "bad.ignore" should be skipped and not in src_dir_fls
    expected = [os.path.join(sync_data["src_dir_abs"], f) for f in ["good.mp3", "other.txt"]]
    assert sync_data["src_dir_fls"] == expected


def test_get_destination_subdir_data_creates_dir(monkeypatch, capsys):
    # Setup to simulate non-existing destination directory
    s = sync.Sync(("url", "/gvfs"), "/src", "/dst", verbose=True)

    # sync_data with dst_dir_abs that does not exist
    sync_data = {"dst_dir_abs": "/dst/subdir", "dst_dir_fls": []}
    exists_calls = {"called": False}

    def fake_exists(path):
        return False

    monkeypatch.setattr(os.path, "exists", fake_exists)

    called_args = []
    def fake_mkdir(path):
        called_args.append(path)

    monkeypatch.setattr(sync.gvfs, "mkdir", fake_mkdir)
    s.gvfs_wrapper = lambda func, *a: func(*a)

    s.get_destination_subdir_data(sync_data)
    captured = capsys.readouterr()
    assert "Creating directory" in captured.out
    assert called_args == [sync_data["dst_dir_abs"]]
    assert sync_data["dst_dir_fls"] == []


def test_get_destination_subdir_data_existing(monkeypatch):
    # Setup to simulate existing directory with files
    s = sync.Sync(("url", "/gvfs"), "/src", "/dst", ignore_file_types=["ignore"])

    files_in_dir = ["file1.txt", "file2.ignore", "file3.mp3"]
    sync_data = {"dst_dir_abs": "/dest/dir", "dst_dir_fls": []}

    def fake_exists(path):
        return True

    def fake_listdir(path):
        return files_in_dir

    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(os, "listdir", fake_listdir)
    # override handle_ignored_file_type to raise for "ignore"
    s.handle_ignored_file_type = lambda x: (_ for _ in ()).throw(exceptions.IgnoredTypeException) if x.endswith(".ignore") else None

    s.get_destination_subdir_data(sync_data)

    # Only files without ignored extensions should be added
    expected = [os.path.join(sync_data["dst_dir_abs"], "file1.txt"), os.path.join(sync_data["dst_dir_abs"], "file3.mp3")]
    assert sorted(sync_data["dst_dir_fls"]) == sorted(expected)


def test_copy_file_calls_gvfs_cp(monkeypatch, capsys):
    called = {}
    s = sync.Sync(("url", "/gvfs"), "/src", "/dst", verbose=True)

    def fake_cp(src, dst):
        called["src"] = src
        called["dst"] = dst

    monkeypatch.setattr(sync.gvfs, "cp", fake_cp)

    s.gvfs_wrapper = lambda func, *a: func(*a)
    src_file = "/src/file.txt"
    dst_file = "/dst/file.txt"
    s.copy_file(src_file, dst_file)
    captured = capsys.readouterr()
    assert "Copying" in captured.out
    assert called["src"] == src_file
    assert called["dst"] == dst_file


def test_do_sync_skips_existing_when_not_overwrite(monkeypatch):
    s = sync.Sync(("url", "/gvfs"), "/src", "/dst", overwrite_existing=False)
    copy_calls = []

    def fake_copy(src, dst):
        copy_calls.append((src, dst))

    s.copy_file = fake_copy

    sync_data = {
        "src_dir_fls": ["/src/file1", "/src/file2"],
        "dst_dir_abs": "/dst",
        "dst_dir_fls": ["/dst/file1"]
    }

    s.do_sync(sync_data)

    # file1 skip copy; file2 copied
    assert ("/src/file1", "/dst/file1") not in copy_calls
    assert ("/src/file2", "/dst/file2") in copy_calls
    # dst_dir_fls updated (file1 removed)
    assert "/dst/file1" not in sync_data["dst_dir_fls"]


def test_do_sync_overwrites_existing(monkeypatch):
    s = sync.Sync(("url", "/gvfs"), "/src", "/dst", overwrite_existing=True)
    copy_calls = []

    def fake_copy(src, dst):
        copy_calls.append((src, dst))

    s.copy_file = fake_copy

    sync_data = {
        "src_dir_fls": ["/src/file1", "/src/file2"],
        "dst_dir_abs": "/dst",
        "dst_dir_fls": ["/dst/file1"]
    }

    s.do_sync(sync_data)

    # Both files copied
    assert ("/src/file1", "/dst/file1") in copy_calls
    assert ("/src/file2", "/dst/file2") in copy_calls
    # dst_dir_fls updated (file1 removed)
    assert "/dst/file1" not in sync_data["dst_dir_fls"]


def test_handle_destination_dir_data_remove(monkeypatch, capsys):
    s = sync.Sync(("url", "/gvfs"), "/src", "/dst", unmatched=sync.REMOVE)

    removed = []

    def fake_rm(path):
        removed.append(path)

    s.gvfs_wrapper = lambda func, *a: func(*a)
    monkeypatch.setattr(sync.gvfs, "rm", fake_rm)

    sync_data = {
        "dst_dir_fls": ["/dst/unmatched_file1", "/dst/unmatched_file2"],
        "dst_dir_abs": "/dst",
        "src_dir_abs": "/src"
    }

    s.handle_destination_dir_data(sync_data)
    captured = capsys.readouterr()

    # Both files removed
    assert set(removed) == {"/dst/unmatched_file1", "/dst/unmatched_file2"}
    # Verbose output mentions removals
    assert "Removing" in captured.out


def test_handle_destination_dir_data_synchronize(monkeypatch):
    s = sync.Sync(("url", "/gvfs"), "/src", "/dst", unmatched=sync.SYNCHRONIZE)

    copied = []

    def fake_copy(src, dst):
        copied.append((src, dst))

    s.copy_file = fake_copy

    sync_data = {
        "dst_dir_fls": ["/dst/unmatched_file1"],
        "dst_dir_abs": "/dst",
        "src_dir_abs": "/src"
    }

    s.handle_destination_dir_data(sync_data)

    # Should copy unmatched from dst to src (reverse direction)
    assert copied == [("/dst/unmatched_file1", "/src/unmatched_file1")]


def test_handle_destination_dir_data_ignore_does_nothing():
    s = sync.Sync(("url", "/gvfs"), "/src", "/dst", unmatched=sync.IGNORE)
    called = {"copied": False, "removed": False}
    s.copy_file = lambda *a, **k: called.update(copied=True)
    s.gvfs_wrapper = lambda *a, **k: called.update(removed=True)
    sync_data = {
        "dst_dir_fls": ["/dst/file1"],
        "dst_dir_abs": "/dst",
        "src_dir_abs": "/src",
    }
    s.handle_destination_dir_data(sync_data)
    # Nothing should be done
    assert called == {"copied": False, "removed": False}


def test_sync_runs_all(monkeypatch):
    s = sync.Sync(("url", "/gvfs"), "/src", "/dst", unmatched=sync.IGNORE)
    sync_data_calls = []

    # Setup fake get_sync_data to return one dict with files
    fake_sync_data = {
        "src_dir_fls": ["/src/file1"],
        "dst_dir_fls": [],
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst"
    }

    def fake_get_sync_data():
        return [fake_sync_data]

    s.get_sync_data = fake_get_sync_data

    def fake_do_sync(sd):
        sync_data_calls.append(sd)

    s.do_sync = fake_do_sync
    s.handle_destination_dir_data = lambda sd: (_ for _ in ()).throw(Exception("Should not be called"))

    # Since unmatched is ignore, handle_destination_dir_data not called

    s.sync()
    assert sync_data_calls == [fake_sync_data]

def test_sync_prints_no_files_to_sync(monkeypatch, capsys):
    s = sync.Sync(("url", "/gvfs"), "/src", "/dst", unmatched=sync.IGNORE, verbose=True)

    # Provide sync data with no source files
    fake_sync_data = {
        "src_dir_fls": [],
        "dst_dir_fls": [],
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst"
    }
    s.get_sync_data = lambda: [fake_sync_data]
    # Override do_sync to raise if called unexpectedly
    s.do_sync = lambda sd: pytest.fail("do_sync should not be called")

    s.sync()
    captured = capsys.readouterr()
    assert "No files to sync" in captured.out
