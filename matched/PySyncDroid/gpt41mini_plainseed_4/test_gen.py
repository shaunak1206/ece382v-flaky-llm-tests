import os
import builtins
import types
import pytest
from unittest.mock import MagicMock, patch

from pysyncdroid.sync import (
    Sync,
    readlink,
    IGNORE,
    REMOVE,
    SYNCHRONIZE,
)
from pysyncdroid import exceptions


def test_readlink_empty_path_returns_empty():
    assert readlink("") == ""
    assert readlink(None) is None


def test_readlink_expanduser_and_run_bash_cmd(monkeypatch):
    called = {}

    def fake_run_bash_cmd(cmd):
        called["cmd"] = cmd
        return "/resolved/path"

    monkeypatch.setattr("pysyncdroid.sync.run_bash_cmd", fake_run_bash_cmd)
    result = readlink("~/somepath")

    assert result == "/resolved/path"
    assert called["cmd"][0] == "readlink"
    assert called["cmd"][1] == "-f"
    assert os.path.expanduser("~/somepath") in called["cmd"]


def test_readlink_no_expansion_calls_run_bash_cmd(monkeypatch):
    monkeypatch.setattr("pysyncdroid.sync.run_bash_cmd", lambda cmd: "/abs/path")
    assert readlink("/abs/path") == "/abs/path"


def make_sync_instance(
    unmatched=IGNORE,
    overwrite_existing=False,
    ignore_file_types=None,
    verbose=False,
    mtp_details=("mtp:test", "/fake/mtp_gvfs"),
    source="/source",
    destination="/destination",
):
    return Sync(
        mtp_details=mtp_details,
        source=source,
        destination=destination,
        unmatched=unmatched,
        overwrite_existing=overwrite_existing,
        ignore_file_types=ignore_file_types,
        verbose=verbose,
    )


def test_sync_init_ignore_file_types_lowercased():
    s = make_sync_instance(ignore_file_types=["TXT", "Mp3"])
    assert s.ignore_file_types == ["txt", "mp3"]


def test_verbose_prints_when_verbose_true(capsys):
    s = make_sync_instance(verbose=True)
    s._verbose("hello world")
    captured = capsys.readouterr()
    assert "hello world" in captured.out


def test_verbose_does_not_print_when_verbose_false(capsys):
    s = make_sync_instance(verbose=False)
    s._verbose("hello world")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_gvfs_wrapper_retries_on_connection_reset(monkeypatch):
    s = make_sync_instance()

    call_count = {"calls": 0}

    def flaky_func(*args):
        call_count["calls"] += 1
        if call_count["calls"] == 1:
            raise exceptions.BashException("Connection reset by peer")
        call_count["calls"] += 1
        return "ok"

    mounted = {"called": False}

    def fake_mount(url):
        mounted["called"] = True
        assert url == s.mtp_url

    monkeypatch.setattr("pysyncdroid.sync.gvfs.mount", fake_mount)

    monkeypatch.setattr("pysyncdroid.sync.gvfs_wrapper", s.gvfs_wrapper)

    s.gvfs_wrapper(flaky_func, "arg1")

    assert mounted["called"]
    assert call_count["calls"] == 3


def test_gvfs_wrapper_raises_on_other_bash_exception():
    s = make_sync_instance()

    def raise_other(*args):
        raise exceptions.BashException("Some other error")

    with pytest.raises(exceptions.BashException):
        s.gvfs_wrapper(raise_other)


@patch("os.getcwd", return_value="/cwd")
@patch("os.path.exists")
@patch("os.path.isdir")
@patch("pysyncdroid.sync.readlink", side_effect=lambda x: x)
def test_set_source_abs_exists_in_cwd(mock_readlink, mock_isdir, mock_exists, mock_getcwd):
    s = make_sync_instance(source="relpath")

    mock_exists.side_effect = lambda p: p == "/cwd/relpath"
    mock_isdir.return_value = True

    s.set_source_abs()

    assert s.source == "/cwd/relpath"


@patch("os.getcwd", return_value="/cwd")
@patch("os.path.exists")
@patch("os.path.isdir")
@patch("pysyncdroid.sync.readlink", side_effect=lambda x: x)
def test_set_source_abs_exists_in_mtp_gvfs(mock_readlink, mock_isdir, mock_exists, mock_getcwd):
    s = make_sync_instance(source="relpath")

    def exists(path):
        return path == "/fake/mtp_gvfs/relpath"

    mock_exists.side_effect = exists
    mock_isdir.return_value = True

    s.set_source_abs()

    assert s.source == "/fake/mtp_gvfs/relpath"


@patch("os.getcwd", return_value="/cwd")
@patch("os.path.exists", return_value=False)
@patch("pysyncdroid.sync.readlink", side_effect=lambda x: x)
def test_set_source_abs_raises_when_not_exist(mock_readlink, mock_exists, mock_getcwd):
    s = make_sync_instance(source="relpath")
    with pytest.raises(OSError) as err:
        s.set_source_abs()
    assert "does not exist" in str(err.value)


@patch("os.getcwd", return_value="/cwd")
@patch("os.path.exists", return_value=True)
@patch("os.path.isdir", return_value=False)
@patch("pysyncdroid.sync.readlink", side_effect=lambda x: x)
def test_set_source_abs_raises_when_not_dir(mock_readlink, mock_isdir, mock_exists, mock_getcwd):
    s = make_sync_instance(source="relpath")
    with pytest.raises(OSError) as err:
        s.set_source_abs()
    assert "is not a directory" in str(err.value)


@patch("pysyncdroid.sync.readlink", side_effect=lambda x: x)
def test_set_destination_abs_for_computer_destination(monkeypatch):
    # when source is on device (contains mtp:host)
    s = make_sync_instance(source="mtp:host/something", destination="destpath")
    s.mtp_gvfs_path = "/fake/mtp_gvfs"

    s.set_destination_abs()

    assert s.destination == "destpath"


@patch("pysyncdroid.sync.readlink", side_effect=lambda x: x)
def test_set_destination_abs_for_device_destination(monkeypatch):
    s = make_sync_instance(source="/some/source", destination="destpath")
    s.mtp_gvfs_path = "/fake/mtp_gvfs"

    s.set_destination_abs()

    assert s.destination == os.path.join("/fake/mtp_gvfs", "destpath")


def test_set_destination_subdir_abs_no_relative_remains():
    s = make_sync_instance()
    s.source = "/base/source"
    s.destination = "/base/destination"

    # same path (subdir == source)
    dst = s.set_destination_subdir_abs("/base/source")
    assert dst == "/base/destination"

    # relative subdir with leading slash
    dst = s.set_destination_subdir_abs("/base/source/subdir")
    assert dst == os.path.join("/base/destination", "subdir")

    # relative subdir with multiple leading slashes removed
    dst = s.set_destination_subdir_abs("/base/source///subdir2")
    assert dst == os.path.join("/base/destination", "subdir2")


def test_sync_data_template_fields_correct():
    s = make_sync_instance()
    sd = s.sync_data_template("/src/dir/", "/dst/dir/")
    assert sd["src_dir_abs"] == "/src/dir"
    assert sd["dst_dir_abs"] == "/dst/dir"
    assert isinstance(sd["src_dir_fls"], list)
    assert isinstance(sd["dst_dir_fls"], list)
    assert sd["src_dir_fls"] == []
    assert sd["dst_dir_fls"] == []


def test_handle_ignored_file_type_raises_for_ignored_extension():
    s = make_sync_instance(ignore_file_types=["txt"])

    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("file.TXT")

    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("file.txt")

    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("file.TxT")


def test_handle_ignored_file_type_passes_for_nonignored_extension():
    s = make_sync_instance(ignore_file_types=["txt"])
    # no exception raised
    s.handle_ignored_file_type("file.mp3")
    s.handle_ignored_file_type("noextensionfile")


def test_get_source_subdir_data_skips_ignored_files():
    s = make_sync_instance(ignore_file_types=["txt"])
    sync_data = s.sync_data_template("/src", "/dst")

    files = ["good.mp3", "skip.TXT", "another.Gif"]

    s.get_source_subdir_data(files, sync_data)

    # "skip.TXT" should be skipped, others added with full source paths
    expected = {os.path.join("/src", "good.mp3"), os.path.join("/src", "another.Gif")}
    assert set(sync_data["src_dir_fls"]) == expected


@patch("os.path.exists")
@patch("os.listdir")
def test_get_destination_subdir_data_creates_dir_if_not_exist(mock_listdir, mock_exists):
    s = make_sync_instance(verbose=True)
    sync_data = s.sync_data_template("/src", "/dst")

    mock_exists.return_value = False
    mock_listdir.return_value = []

    called = {}

    def fake_mkdir(path):
        called["path"] = path

    patch_gvfs_mkdir = patch("pysyncdroid.sync.gvfs.mkdir", fake_mkdir)

    with patch_gvfs_mkdir:
        s.get_destination_subdir_data(sync_data)

    assert called["path"] == sync_data["dst_dir_abs"]


@patch("os.path.exists")
@patch("os.listdir")
def test_get_destination_subdir_data_lists_files_skips_ignored(mock_listdir, mock_exists):
    s = make_sync_instance(ignore_file_types=["txt"])

    sync_data = s.sync_data_template("/src", "/dst")
    sync_data["dst_dir_abs"] = "/dst"

    mock_exists.return_value = True
    mock_listdir.return_value = ["keep.mp3", "skip.TXT", "another.gif"]

    s.get_destination_subdir_data(sync_data)

    dst_files_expected = {os.path.join("/dst", "keep.mp3"), os.path.join("/dst", "another.gif")}
    assert set(sync_data["dst_dir_fls"]) == dst_files_expected


@patch("pysyncdroid.sync.SYNC", IGNORE)
def test_get_sync_data_empty_directory(monkeypatch):
    s = make_sync_instance()
    s.source = "/source"

    monkeypatch.setattr("os.walk", lambda src: iter([("/source", [], [])]))
    result = s.get_sync_data()
    # No files present, so no sync data returned
    assert result == []


@patch("os.walk")
def test_get_sync_data_produces_correct_data(monkeypatch):
    s = make_sync_instance()
    s.source = "/source"
    s.destination = "/destination"

    monkeypatch.setattr("os.walk", lambda src: iter([("/source", ["subdir"], ["file1.jpg", "file2.txt"])]))
    s.set_destination_subdir_abs = lambda src_subdir_abs: "/dest"  # stub
    s.sync_data_template = lambda a, b: {"src_dir_abs": a, "dst_dir_abs": b, "src_dir_fls": [], "dst_dir_fls": []}
    s.get_source_subdir_data = MagicMock()
    s.get_destination_subdir_data = MagicMock()

    result = s.get_sync_data()

    assert isinstance(result, list)
    assert len(result) == 1
    data = result[0]
    assert data["src_dir_abs"] == "/source"
    assert data["dst_dir_abs"] == "/dest"
    s.get_source_subdir_data.assert_called_once()
    s.get_destination_subdir_data.assert_called_once()


@patch("pysyncdroid.sync.gvfs.cp")
def test_copy_file_invokes_gvfs_cp(monkeypatch):
    s = make_sync_instance(verbose=True)
    called = {}

    def fake_gvfs_cp(src, dst):
        called["src"] = src
        called["dst"] = dst

    monkeypatch.setattr("pysyncdroid.sync.gvfs.cp", fake_gvfs_cp)

    s.copy_file("/src/f", "/dst/f")

    assert called["src"] == "/src/f"
    assert called["dst"] == "/dst/f"


@patch("pysyncdroid.sync.Sync.copy_file")
def test_do_sync_skips_existing_files_if_no_overwrite(mock_copy_file):
    s = make_sync_instance(overwrite_existing=False)
    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": ["/src/a.txt", "/src/b.txt"],
        "dst_dir_fls": ["/dst/a.txt"],
    }

    s.do_sync(sync_data)

    # It should copy only /src/b.txt because /dst/a.txt exists and overwrite is False
    mock_copy_file.assert_called_once_with("/src/b.txt", "/dst/b.txt")
    assert "/dst/a.txt" not in sync_data["dst_dir_fls"]


@patch("pysyncdroid.sync.Sync.copy_file")
def test_do_sync_overwrites_existing_files(mock_copy_file):
    s = make_sync_instance(overwrite_existing=True)
    sync_data = {
        "src_dir_abs": "/src",
        "dst_dir_abs": "/dst",
        "src_dir_fls": ["/src/a.txt", "/src/b.txt"],
        "dst_dir_fls": ["/dst/a.txt"],
    }

    s.do_sync(sync_data)

    # Both files copied, because overwrite is True
    calls = [((src, src.replace("/src", "/dst")),) for src in sync_data["src_dir_fls"]]

    expected_calls = set((src, dst) for (src, dst) in calls)
    actual_calls = set(call.args for call in mock_copy_file.mock_calls)
    assert expected_calls == actual_calls


@patch("pysyncdroid.sync.Sync.gvfs_wrapper")
@patch("pysyncdroid.sync.Sync.copy_file")
def test_handle_destination_dir_data_remove_and_synchronize(mock_copy_file, mock_gvfs_wrapper):
    s = make_sync_instance(unmatched=REMOVE)
    sync_data = {
        "dst_dir_fls": ["/dst/old.txt"],
        "dst_dir_abs": "/dst",
        "src_dir_abs": "/src",
    }

    # REMOVE should call gvfs_wrapper to remove the file
    s.handle_destination_dir_data(sync_data)
    assert mock_gvfs_wrapper.call_count == 1

    # Now test SYNCHRONIZE mode copies unmatched files back to source
    s.unmatched = SYNCHRONIZE
    mock_gvfs_wrapper.reset_mock()
    mock_copy_file.reset_mock()

    s.handle_destination_dir_data(sync_data)
    mock_copy_file.assert_called_once()
    mock_gvfs_wrapper.assert_not_called()


@patch("pysyncdroid.sync.Sync.get_sync_data")
@patch("pysyncdroid.sync.Sync.do_sync")
@patch("pysyncdroid.sync.Sync.handle_destination_dir_data")
def test_sync_skip_when_no_files_to_sync(mock_handle, mock_do_sync, mock_get_sync_data):
    s = make_sync_instance(unmatched=IGNORE, verbose=True)

    # sync_data with empty src_dir_fls triggers "No files to sync" and return early
    sync_data = {"src_dir_fls": [], "dst_dir_fls": [], "src_dir_abs": "/src", "dst_dir_abs": "/dst"}
    mock_get_sync_data.return_value = [sync_data]

    s.sync()

    mock_do_sync.assert_not_called()
    mock_handle.assert_not_called()


@patch("pysyncdroid.sync.Sync.get_sync_data")
@patch("pysyncdroid.sync.Sync.do_sync")
@patch("pysyncdroid.sync.Sync.handle_destination_dir_data")
def test_sync_runs_all_steps_for_nonempty_files(mock_handle, mock_do_sync, mock_get_sync_data):
    s = make_sync_instance(unmatched=REMOVE)

    sync_data = {"src_dir_fls": ["/src/file"], "dst_dir_fls": ["/dst/file"], "src_dir_abs": "/src", "dst_dir_abs": "/dst"}
    mock_get_sync_data.return_value = [sync_data]

    s.sync()

    mock_do_sync.assert_called_once_with(sync_data)
    mock_handle.assert_called_once_with(sync_data)


@patch("pysyncdroid.sync.Sync.get_sync_data")
@patch("pysyncdroid.sync.Sync.do_sync")
@patch("pysyncdroid.sync.Sync.handle_destination_dir_data")
def test_sync_skips_handle_destination_if_unmatched_ignore(mock_handle, mock_do_sync, mock_get_sync_data):
    s = make_sync_instance(unmatched=IGNORE)

    sync_data = {"src_dir_fls": ["/src/file"], "dst_dir_fls": ["/dst/file"], "src_dir_abs": "/src", "dst_dir_abs": "/dst"}
    mock_get_sync_data.return_value = [sync_data]

    s.sync()

    mock_do_sync.assert_called_once_with(sync_data)
    mock_handle.assert_not_called()
