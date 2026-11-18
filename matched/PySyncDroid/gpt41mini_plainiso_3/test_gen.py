import os
import builtins
import types
import pytest

from pysyncdroid import sync
from pysyncdroid.sync import Sync, IGNORE, REMOVE, SYNCHRONIZE, readlink
from pysyncdroid import exceptions


def test_readlink_none_or_empty(monkeypatch):
    # When path is empty string or None, returns as is
    assert readlink("") == ""
    assert readlink(None) is None


def test_readlink_expanduser(monkeypatch):
    # confirm that ~ gets expanded and run_bash_cmd is called
    calls = {}

    def fake_expanduser(path):
        calls["called"] = True
        return "/home/user"

    def fake_run_bash_cmd(args):
        calls["bash"] = args
        return "/resolved/path"

    monkeypatch.setattr(os.path, "expanduser", fake_expanduser)
    monkeypatch.setattr(sync, "run_bash_cmd", fake_run_bash_cmd)

    result = readlink("~")
    assert calls.get("called")
    assert calls.get("bash") == ["readlink", "-f", "/home/user"]
    assert result == "/resolved/path"


def test_readlink_run_bash_cmd_returns_none(monkeypatch):
    # Simulate run_bash_cmd returning None, readlink should return original path
    def fake_run_bash_cmd(args):
        return None

    monkeypatch.setattr(sync, "run_bash_cmd", fake_run_bash_cmd)
    path = "/some/path"
    result = readlink(path)
    assert result == path


def test_sync_init_with_ignore_file_types():
    s = Sync(
        mtp_details=("mtp_url_fake", "/gvfs/path_fake"),
        source="src",
        destination="dst",
        unmatched=IGNORE,
        overwrite_existing=True,
        ignore_file_types=["JPG", "Txt"],
        verbose=True,
    )
    assert s.mtp_url == "mtp_url_fake"
    assert s.mtp_gvfs_path == "/gvfs/path_fake"
    assert s.source == "src"
    assert s.destination == "dst"
    assert s.unmatched == IGNORE
    assert s.overwrite_existing is True
    assert s.ignore_file_types == ["jpg", "txt"]
    assert s.verbose is True


def test_sync_verbose_prints(capsys):
    s = Sync(("url", "/gvfs"), "src", "dst", verbose=True)
    s._verbose("message here")
    captured = capsys.readouterr()
    assert "message here" in captured.out

    s = Sync(("url", "/gvfs"), "src", "dst", verbose=False)
    s._verbose("message here2")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_gvfs_wrapper_handles_connection_reset(monkeypatch):
    calls = {"mounted": False, "func_calls": 0}

    def fake_mount(mtp_url):
        calls["mounted"] = True
        assert mtp_url == "mtp_url"

    def fail_once(arg):
        # On first call, raise, on second just pass
        if calls["func_calls"] == 0:
            calls["func_calls"] += 1
            raise exceptions.BashException("Something Connection reset by peer")
        calls["func_calls"] += 1

    s = Sync(("mtp_url", "/gvfs"), "src", "dst")

    # Patch gvfs.mount and func
    monkeypatch.setattr(sync.gvfs, "mount", fake_mount)

    s.gvfs_wrapper(fail_once, "arg")
    assert calls["mounted"] is True
    assert calls["func_calls"] == 2


def test_gvfs_wrapper_raises_other_bash_exceptions(monkeypatch):
    def fail_func(*args):
        raise exceptions.BashException("some other error")

    s = Sync(("url", "/gvfs"), "src", "dst")
    with pytest.raises(exceptions.BashException):
        s.gvfs_wrapper(fail_func)


def test_set_source_abs_relative_and_absolute(monkeypatch, tmp_path):
    # Create a dummy directory for test
    test_dir = tmp_path / "testsource"
    test_dir.mkdir()

    s = Sync(("url", str(tmp_path)), "testsource", "dst")

    # Patch readlink to return input, os.getcwd to tmp_path
    monkeypatch.setattr(sync, "readlink", lambda p: p)
    monkeypatch.setattr(sync.os.path, "exists", lambda p: p == str(test_dir))
    monkeypatch.setattr(sync.os.path, "isabs", os.path.isabs)
    monkeypatch.setattr(sync.os.path, "isdir", lambda p: True if p == str(test_dir) else False)
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

    s.set_source_abs()
    assert s.source == str(test_dir)

    # Test absolute path provided as source
    s = Sync(("url", str(tmp_path)), str(test_dir), "dst")

    monkeypatch.setattr(sync, "readlink", lambda p: p)
    s.set_source_abs()
    assert s.source == str(test_dir)


def test_set_source_abs_nonexistent_raises(monkeypatch):
    s = Sync(("url", "/gvfs"), "nonexistent_dir", "dst")

    monkeypatch.setattr(sync, "readlink", lambda p: p)
    monkeypatch.setattr(sync.os.path, "exists", lambda p: False)
    monkeypatch.setattr(sync.os.path, "isabs", os.path.isabs)

    with pytest.raises(OSError) as excinfo:
        s.set_source_abs()
    assert "does not exist" in str(excinfo.value)


def test_set_source_abs_not_directory_raises(monkeypatch, tmp_path):
    f = tmp_path / "notadir"
    f.write_text("content")
    s = Sync(("url", str(tmp_path)), str(f), "dst")

    monkeypatch.setattr(sync, "readlink", lambda p: p)
    # exists returns True only for the file path
    monkeypatch.setattr(sync.os.path, "exists", lambda p: p == str(f))
    monkeypatch.setattr(sync.os.path, "isabs", os.path.isabs)
    monkeypatch.setattr(sync.os.path, "isdir", lambda p: False)

    with pytest.raises(OSError) as excinfo:
        s.set_source_abs()
    assert "not a directory" in str(excinfo.value)


def test_set_destination_abs(monkeypatch):
    s = Sync(("mtp_url", "/gvfs/path"), "src", "dst")

    # Case when source contains "mtp:host" -> destination is as given from readlink
    s.source = "mtp:host/something"
    monkeypatch.setattr(sync, "readlink", lambda p: "final_dest")
    s.set_destination_abs()
    assert s.destination == "final_dest"

    # Case when source does not contain mtp:host -> destination joined with mtp_gvfs_path
    s.source = "/some/normal/path"
    monkeypatch.setattr(sync, "readlink", lambda p: "final_dest2")
    s.set_destination_abs()
    assert s.destination == os.path.join("/gvfs/path", "final_dest2")


def test_set_destination_subdir_abs_simple_and_lstrip():
    s = Sync(("mtp_url", "/gvfs/path"), "/source", "/destination")

    s.source = "/source"

    # Case where subdir is directly under source
    subdir = "/source/subfolder"
    expected = os.path.join(s.destination, "subfolder")
    result = s.set_destination_subdir_abs(subdir)
    assert result == expected

    # Case where subdir has multiple slashes and starting slashes
    subdir = "/source///subfolder2"
    expected = os.path.join(s.destination, "subfolder2")
    result = s.set_destination_subdir_abs(subdir)
    assert result == expected


def test_sync_data_template_and_get_source_subdir_data(monkeypatch):
    s = Sync(("url", "gvfs"), "src", "dst")
    src_subdir_abs = "/src/dir"
    dst_subdir_abs = "/dst/dir"
    sync_data = s.sync_data_template(src_subdir_abs, dst_subdir_abs)
    assert sync_data["src_dir_abs"] == src_subdir_abs.rstrip("/")
    assert sync_data["dst_dir_abs"] == dst_subdir_abs.rstrip("/")
    assert sync_data["src_dir_fls"] == []
    assert sync_data["dst_dir_fls"] == []

    # patch handle_ignored_file_type to raise for one file
    def raising_handle(file):
        if file == "ignore.me":
            raise exceptions.IgnoredTypeException

    s.handle_ignored_file_type = raising_handle

    src_files = ["a.txt", "ignore.me", "b.txt"]
    s.get_source_subdir_data(src_files, sync_data)
    # It should ignore "ignore.me"
    assert any("a.txt" in f for f in sync_data["src_dir_fls"])
    assert any("b.txt" in f for f in sync_data["src_dir_fls"])
    assert not any("ignore.me" in f for f in sync_data["src_dir_fls"])


def test_handle_ignored_file_type_raises(monkeypatch):
    s = Sync(("url", "gvfs"), "src", "dst", ignore_file_types=["jpg", "png"])

    # file with extension in ignore_file_types raises exception
    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("file.JPG")

    # file with extension not in ignore_file_types does not raise
    s.handle_ignored_file_type("file.txt")


def test_get_destination_subdir_data_creates_dir(monkeypatch):
    s = Sync(("url", "gvfs"), "src", "dst")
    sync_data = {
        "dst_dir_abs": "/path/to/dst",
        "dst_dir_fls": [],
        "src_dir_abs": "/path/to/src",
        "src_dir_fls": ["file1.txt"],
    }
    # Directory does not exist
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    mkdir_calls = []

    def fake_mkdir(path):
        mkdir_calls.append(path)

    s.gvfs_wrapper = lambda func, *args: func(*args)
    monkeypatch.setattr(sync.gvfs, "mkdir", fake_mkdir)

    s._verbose = lambda msg: None  # avoid printing

    s.get_destination_subdir_data(sync_data)
    assert mkdir_calls == ["/path/to/dst"]
    # dst_dir_fls remains empty
    assert sync_data["dst_dir_fls"] == []

def test_get_destination_subdir_data_existing_dir(monkeypatch, tmp_path):
    s = Sync(("url", str(tmp_path)), "src", "dst")
    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()
    (dst_dir / "file1.txt").write_text("foo")
    (dst_dir / "ignore.jpg").write_text("bar")

    ignore_extensions = ["jpg"]
    s.ignore_file_types = ignore_extensions

    sync_data = {
        "dst_dir_abs": str(dst_dir),
        "dst_dir_fls": [],
        "src_dir_abs": "/src",
    }

    # Patch handle_ignored_file_type to raise for .jpg files
    orig_handle = s.handle_ignored_file_type
    def patched_handle(path):
        _, ext = os.path.splitext(path)
        if ext.lower().replace(".", "") in ignore_extensions:
            raise exceptions.IgnoredTypeException
    s.handle_ignored_file_type = patched_handle

    s.gvfs_wrapper = lambda func, *args: func(*args)

    s._verbose = lambda msg: None
    s.get_destination_subdir_data(sync_data)

    # It should have file1.txt added but ignore.jpg skipped.
    files = [os.path.basename(f) for f in sync_data["dst_dir_fls"]]
    assert "file1.txt" in files
    assert "ignore.jpg" not in files


def test_get_sync_data(monkeypatch):
    s = Sync(("url", "/gvfs"), "/src", "/dst")
    s.source = "/src"
    s.destination = "/dst"

    # Setup os.walk to return:
    # 3 directories, but one with no files (should be skipped)
    walk_data = [
        ("/src", ["subdir"], ["file1", "file2"]),
        ("/src/subdir", [], []),  # no files, skipped
        ("/src2", [], ["filex"]),
    ]

    def fake_os_walk(path):
        for data in walk_data:
            yield data

    monkeypatch.setattr(os, "walk", fake_os_walk)

    # patch dependent methods to track calls
    calls = {"get_src": 0, "get_dst": 0, "sync_data_template": sync.Sync.sync_data_template}

    def fake_get_source_subdir_data(files, sync_data):
        calls["get_src"] += 1
        for f in files:
            sync_data["src_dir_fls"].append(f)

    def fake_get_destination_subdir_data(sync_data):
        calls["get_dst"] += 1
        # Pretend no destination files
        sync_data["dst_dir_fls"].clear()

    monkeypatch.setattr(s, "get_source_subdir_data", fake_get_source_subdir_data)
    monkeypatch.setattr(s, "get_destination_subdir_data", fake_get_destination_subdir_data)

    # Patch sync_data_template to actual method
    monkeypatch.setattr(s, "sync_data_template", s.sync_data_template)

    result = s.get_sync_data()
    # Only the directory with files should be included
    assert isinstance(result, list)
    assert len(result) == 2  # /src and /src2
    assert calls["get_src"] == 2
    assert calls["get_dst"] == 2


def test_copy_file(monkeypatch, capsys):
    s = Sync(("url", "gvfs"), "src", "dst", verbose=True)
    called = {}

    def fake_cp(src, dst):
        called["src"] = src
        called["dst"] = dst

    monkeypatch.setattr(sync.gvfs, "cp", fake_cp)
    s.gvfs_wrapper = lambda func, *args: func(*args)

    s.copy_file("srcfile", "dstfile")
    captured = capsys.readouterr()
    assert "Copying srcfile to dstfile" in captured.out
    assert called["src"] == "srcfile"
    assert called["dst"] == "dstfile"


def test_do_sync_copies_and_removes(monkeypatch):
    s = Sync(("url", "gvfs"), "src", "dst", overwrite_existing=False)

    # Setup sync_data with src files and dst files where one file matches
    src_file_existing = "/src/file1.txt"
    src_file_new = "/src/file2.txt"
    dst_file_existing = "/dst/file1.txt"

    sync_data = {
        "src_dir_fls": [src_file_existing, src_file_new],
        "dst_dir_fls": [dst_file_existing],
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
    }

    copied_files = []

    def fake_copy_file(src, dst):
        copied_files.append((src, dst))

    s.copy_file = fake_copy_file

    s.do_sync(sync_data)

    # Since overwrite_existing=False, existing dst file should not be copied again
    # So only src_file_new is copied
    expected_dst_new = src_file_new.replace(sync_data["src_dir_abs"], sync_data["dst_dir_abs"])
    assert copied_files == [(src_file_new, expected_dst_new)]
    assert dst_file_existing not in sync_data["dst_dir_fls"]  # removed from unmatched


def test_do_sync_with_overwrite(monkeypatch):
    s = Sync(("url", "gvfs"), "src", "dst", overwrite_existing=True)

    src_file_existing = "/src/file1.txt"
    dst_file_existing = "/dst/file1.txt"

    sync_data = {
        "src_dir_fls": [src_file_existing],
        "dst_dir_fls": [dst_file_existing],
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
    }
    copied = []

    def fake_copy_file(src, dst):
        copied.append((src, dst))

    s.copy_file = fake_copy_file
    s.do_sync(sync_data)
    expected_dst = src_file_existing.replace(sync_data["src_dir_abs"], sync_data["dst_dir_abs"])
    assert copied == [(src_file_existing, expected_dst)]
    assert dst_file_existing not in sync_data["dst_dir_fls"]


def test_handle_destination_dir_data_remove(monkeypatch, capsys):
    s = Sync(("url", "gvfs"), "src", "dst", unmatched=REMOVE)
    removed = []

    sync_data = {
        "dst_dir_fls": ["/dst/file1.txt", "/dst/file2.txt"],
        "dst_dir_abs": "/dst",
        "src_dir_abs": "/src",
    }

    def fake_rm(path):
        removed.append(path)

    s.gvfs_wrapper = lambda f, *a: f(*a)
    monkeypatch.setattr(sync.gvfs, "rm", fake_rm)
    s._verbose = lambda msg: None

    s.handle_destination_dir_data(sync_data)
    assert set(removed) == {"/dst/file1.txt", "/dst/file2.txt"}


def test_handle_destination_dir_data_synchronize(monkeypatch):
    s = Sync(("url", "gvfs"), "src", "dst", unmatched=SYNCHRONIZE)
    copied = []

    sync_data = {
        "dst_dir_fls": ["/dst/file1.txt"],
        "dst_dir_abs": "/dst",
        "src_dir_abs": "/src",
    }

    def fake_copy_file(src_file, dst_file):
        copied.append((src_file, dst_file))

    s.copy_file = fake_copy_file
    s.gvfs_wrapper = lambda f, *a: f(*a)

    s.handle_destination_dir_data(sync_data)
    expected_src = "/src/file1.txt"
    assert copied == [("/dst/file1.txt", expected_src)]


def test_handle_destination_dir_data_ignore(monkeypatch):
    s = Sync(("url", "gvfs"), "src", "dst", unmatched=IGNORE)
    # dst_dir_fls non-empty but unmatched is IGNORE: nothing happens
    sync_data = {
        "dst_dir_fls": ["/dst/file1.txt"],
        "dst_dir_abs": "/dst",
        "src_dir_abs": "/src",
    }
    s.gvfs_wrapper = lambda f, *a: pytest.fail("gvfs_wrapper shouldn't be called")
    s.copy_file = pytest.fail
    s._verbose = lambda msg: None

    s.handle_destination_dir_data(sync_data)  # should not raise


def test_sync_returns_early_no_files(capsys, monkeypatch):
    s = Sync(("url", "gvfs"), "src", "dst")
    # get_sync_data returns a list with one empty src_dir_fls dict to trigger early return
    monkeypatch.setattr(s, "get_sync_data", lambda: [{"src_dir_fls": []}])
    s._verbose = lambda msg: print(msg)

    s.sync()
    captured = capsys.readouterr()
    assert "No files to sync" in captured.out


def test_sync_full_flow(monkeypatch):
    # Test syncing with files to copy and unmatched files removed
    s = Sync(("url", "gvfs"), "src", "dst", unmatched=REMOVE, overwrite_existing=False)

    # Setup fake sync data returned by get_sync_data
    file_src = "/src/file1.txt"
    file_dst = "/dst/file1.txt"
    sync_data = {
        "src_dir_fls": [file_src],
        "dst_dir_fls": [file_dst, "/dst/unmatched.txt"],
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
    }

    # Patch methods
    called = {"copy_file": [], "rm": []}
    s.get_sync_data = lambda: [sync_data]
    s.copy_file = lambda src, dst: called["copy_file"].append((src, dst))
    s.gvfs_wrapper = lambda func, *args: func(*args)
    monkeypatch.setattr(sync.gvfs, "rm", lambda path: called["rm"].append(path))
    s._verbose = lambda msg: None

    s.sync()
    # copy_file called for file_src -> file_dst
    assert (file_src, file_dst) in called["copy_file"]
    # 'unmatched.txt' is removed
    assert "/dst/unmatched.txt" in called["rm"]


@pytest.mark.parametrize("ignore_exts,filename,raises", [
    (["mp3"], "song.mp3", True),
    (["Mp3"], "song.MP3", True),
    (["mp3"], "picture.jpg", False),
    ([], "song.mp3", False),
    (None, "song.mp3", False),
])
def test_handle_ignored_file_type_various(ignore_exts, filename, raises):
    s = Sync(("url", "gvfs"), "src", "dst", ignore_file_types=ignore_exts)
    if raises:
        with pytest.raises(exceptions.IgnoredTypeException):
            s.handle_ignored_file_type(filename)
    else:
        s.handle_ignored_file_type(filename)
