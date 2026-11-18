import os
import builtins
import pytest
from unittest.mock import MagicMock, patch, call
from pysyncdroid.sync import Sync, readlink, IGNORE, REMOVE, SYNCHRONIZE
from pysyncdroid import exceptions


def test_readlink_empty_path():
    assert readlink("") == ""
    assert readlink(None) is None


@patch("pysyncdroid.sync.run_bash_cmd", return_value="/resolved/path")
def test_readlink_expands_and_resolves(mock_run_bash):
    # ~ expands to home, run_bash_cmd result is used
    home = os.path.expanduser("~")
    path = "~/somepath"
    result = readlink(path)
    assert result == "/resolved/path"
    mock_run_bash.assert_called_once_with(["readlink", "-f", os.path.join(home, "somepath")])


@patch("pysyncdroid.sync.run_bash_cmd", return_value="")
def test_readlink_returns_path_when_run_bash_returns_empty(mock_run_bash):
    path = "/some/path"
    result = readlink(path)
    assert result == path
    mock_run_bash.assert_called_once_with(["readlink", "-f", path])


def test_sync_init_basic_properties():
    sync = Sync(
        mtp_details=("mtp://some_url", "/gvfs/path"),
        source="source",
        destination="dest",
        unmatched=REMOVE,
        overwrite_existing=True,
        ignore_file_types=["MP3", "txt"],
        verbose=True,
    )
    assert sync.mtp_url == "mtp://some_url"
    assert sync.mtp_gvfs_path == "/gvfs/path"
    assert sync.source == "source"
    assert sync.destination == "dest"
    assert sync.unmatched == REMOVE
    assert sync.overwrite_existing is True
    # ignore_file_types lowercased
    assert sync.ignore_file_types == ["mp3", "txt"]
    assert sync.verbose is True


def test_sync_init_ignore_file_types_none():
    sync = Sync(
        ("url", "gvfs"),
        "src",
        "dst",
        ignore_file_types=None,
    )
    assert sync.ignore_file_types is None


def test_verbose_prints_only_when_verbose_true(capfd):
    sync = Sync(("url", "gvfs"), "src", "dst", verbose=True)
    sync._verbose("message")
    out, _ = capfd.readouterr()
    assert "message" in out

    sync2 = Sync(("url", "gvfs"), "src", "dst", verbose=False)
    sync2._verbose("another message")
    out, _ = capfd.readouterr()
    assert out == ""


@pytest.mark.parametrize("exc_msg", ["some error", "Connection reset by peer"])
def test_gvfs_wrapper_retries_on_connection_reset(exc_msg):
    sync = Sync(("url", "gvfs"), "src", "dst")

    func = MagicMock()
    # First call raises BashException, second successful

    def side_effect(*args, **kwargs):
        if side_effect.call_count == 0:
            side_effect.call_count += 1
            raise exceptions.BashException(exc_msg)
        return None

    side_effect.call_count = 0
    func.side_effect = side_effect

    if exc_msg == "Connection reset by peer":
        with patch("pysyncdroid.sync.gvfs.mount") as mock_mount:
            sync.gvfs_wrapper(func, "argument")
            assert mock_mount.called
            assert func.call_count == 2
    else:
        with pytest.raises(exceptions.BashException):
            sync.gvfs_wrapper(func, "argument")
        assert func.call_count == 1


@patch("pysyncdroid.sync.readlink")
@patch("os.getcwd")
@patch("os.path.isabs")
@patch("os.path.exists")
@patch("os.path.isdir")
def test_set_source_abs_success(
    mock_isdir, mock_exists, mock_isabs, mock_getcwd, mock_readlink
):
    mock_readlink.return_value = "relative/path"
    mock_getcwd.return_value = "/cwd"
    mock_isabs.side_effect = lambda x: x.startswith("/")
    mock_exists.side_effect = lambda x: x in ["/cwd/relative/path", "/gvfs/source"]
    mock_isdir.side_effect = lambda x: x == "/cwd/relative/path"

    sync = Sync(("url", "/gvfs"))
    sync.source = "relative/path"
    # It should detect source exists on computer prefix (/cwd)
    sync.set_source_abs()
    assert sync.source == "/cwd/relative/path"


@patch("pysyncdroid.sync.readlink")
@patch("os.getcwd")
@patch("os.path.isabs")
@patch("os.path.exists")
@patch("os.path.isdir")
def test_set_source_abs_on_device(
    mock_isdir, mock_exists, mock_isabs, mock_getcwd, mock_readlink
):
    mock_readlink.return_value = "source"
    mock_getcwd.return_value = "/cwd"
    mock_isabs.return_value = False
    mock_exists.side_effect = lambda x: x == "/gvfs/source"
    mock_isdir.side_effect = lambda x: x == "/gvfs/source"

    sync = Sync(("url", "/gvfs"))
    sync.source = "source"
    sync.set_source_abs()
    assert sync.source == "/gvfs/source"


@patch("pysyncdroid.sync.readlink", return_value="missing")
@patch("os.getcwd", return_value="/cwd")
@patch("os.path.isabs", return_value=False)
@patch("os.path.exists", return_value=False)
def test_set_source_abs_raises_oserror_on_missing(
    mock_exists, mock_isabs, mock_getcwd, mock_readlink
):
    sync = Sync(("url", "/gvfs"))
    sync.source = "missing"
    with pytest.raises(OSError) as excinfo:
        sync.set_source_abs()
    assert "does not exist" in str(excinfo.value)


@patch("pysyncdroid.sync.readlink", return_value="file.txt")
@patch("os.getcwd", return_value="/cwd")
@patch("os.path.isabs", return_value=False)
@patch("os.path.exists", return_value=True)
@patch("os.path.isdir", return_value=False)
def test_set_source_abs_raises_oserror_if_not_dir(
    mock_isdir, mock_exists, mock_isabs, mock_getcwd, mock_readlink
):
    sync = Sync(("url", "/gvfs"))
    sync.source = "file.txt"
    with pytest.raises(OSError) as excinfo:
        sync.set_source_abs()
    assert "not a directory" in str(excinfo.value)


@patch("pysyncdroid.sync.readlink", return_value="/dest")
def test_set_destination_abs_on_computer_source():
    sync = Sync(("url", "/gvfs"))
    sync.source = "mtp:host:/something"  # mimic device source
    sync.destination = "/dest"
    sync.set_destination_abs()
    # Source indicates device, so destination on computer
    assert sync.destination == "/dest"


@patch("pysyncdroid.sync.readlink", return_value="dest")
def test_set_destination_abs_on_device_source():
    sync = Sync(("url", "/gvfs"))
    sync.source = "/some/source"  # computer source
    sync.destination = "dest"
    sync.set_destination_abs()
    assert sync.destination == os.path.join("/gvfs", "dest")


def test_set_destination_subdir_abs_basic():
    sync = Sync(("url", "/gvfs"))
    sync.source = "/source"
    sync.destination = "/dest"
    src_subdir = "/source/subdir1/subdir2"
    expected_rel = os.path.join("subdir1", "subdir2")
    result = sync.set_destination_subdir_abs(src_subdir)
    expected = os.path.join(sync.destination, expected_rel)
    assert result == expected


def test_set_destination_subdir_abs_empty_when_root_source():
    sync = Sync(("url", "/gvfs"))
    sync.source = "/source"
    sync.destination = "/dest"
    # src_subdir is exactly source directory -> rel_src_subdir_pth is empty
    result = sync.set_destination_subdir_abs(sync.source)
    assert result == sync.destination


def test_sync_data_template_has_expected_keys():
    sync = Sync(("url", "/gvfs"))
    src = "/src"
    dst = "/dst"
    data = sync.sync_data_template(src, dst)
    assert data["src_dir_abs"] == src.rstrip("/")
    assert data["dst_dir_abs"] == dst.rstrip("/")
    assert data["src_dir_fls"] == []
    assert data["dst_dir_fls"] == []


def test_handle_ignored_file_type_raises():
    sync = Sync(("url", "/gvfs"))
    sync.ignore_file_types = ["mp3", "txt"]
    with pytest.raises(exceptions.IgnoredTypeException):
        sync.handle_ignored_file_type("file.MP3")
    with pytest.raises(exceptions.IgnoredTypeException):
        sync.handle_ignored_file_type("file.txt")

    # Does not raise for other extension
    sync.handle_ignored_file_type("other.doc")


def test_handle_ignored_file_type_no_ignore_list():
    sync = Sync(("url", "/gvfs"))
    sync.ignore_file_types = None
    # Should not raise
    sync.handle_ignored_file_type("file.mp3")


def test_get_source_subdir_data_adds_files_skipping_ignored():
    sync = Sync(("url", "/gvfs"))
    sync.ignore_file_types = ["ignore"]

    files = ["a.txt", "b.ignore", "c.doc"]
    sync_data = {"src_dir_abs": "/src", "src_dir_fls": []}

    sync.get_source_subdir_data(files, sync_data)
    # ignored files are not added
    assert "/src/a.txt" in sync_data["src_dir_fls"]
    assert "/src/c.doc" in sync_data["src_dir_fls"]
    assert all("b.ignore" not in f for f in sync_data["src_dir_fls"])


def test_get_destination_subdir_data_creates_dir_and_skips_ignored():
    sync = Sync(("url", "/gvfs"), verbose=True)
    sync.ignore_file_types = ["skip"]

    sync_data = {"dst_dir_abs": "/dest", "dst_dir_fls": []}

    with patch("os.path.exists", return_value=False), patch.object(sync, "_verbose") as mock_verbose, patch("pysyncdroid.sync.gvfs.mkdir") as mock_gvfs_mkdir:
        sync.get_destination_subdir_data(sync_data)
        mock_verbose.assert_called_once()
        mock_gvfs_mkdir.assert_called_once_with("/dest")

    # Now test when directory exists and files present
    files = ["a.txt", "b.skip", "c.doc"]

    with patch("os.path.exists", return_value=True), patch("os.listdir", return_value=files):
        sync_data["dst_dir_fls"] = []
        sync.get_destination_subdir_data(sync_data)
        # b.skip should be skipped, others included
        included = [os.path.join("/dest", f) for f in ["a.txt", "c.doc"]]
        assert all(f in sync_data["dst_dir_fls"] for f in included)
        assert all(os.path.join("/dest", "b.skip") not in sync_data["dst_dir_fls"] for _ in range(1))


def test_get_sync_data_skips_empty_dirs(tmp_path):
    # Build a nested directory under tmp_path
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    # subdir1 without files
    subdir1 = source_dir / "subdir1"
    subdir1.mkdir()
    # subdir2 with a file
    subdir2 = source_dir / "subdir2"
    subdir2.mkdir()
    file_in_subdir2 = subdir2 / "file.txt"
    file_in_subdir2.write_text("test")

    sync = Sync(("url", "/gvfs"))
    sync.source = str(source_dir)
    sync.destination = "/dest"

    with patch.object(sync, "set_destination_subdir_abs", side_effect=lambda src: os.path.join(sync.destination, os.path.relpath(src, sync.source))):
        with patch.object(sync, "sync_data_template") as mock_template, patch.object(sync, "get_source_subdir_data") as mock_get_source, patch.object(sync, "get_destination_subdir_data") as mock_get_dest:
            mock_template.side_effect = lambda s, d: {
                "src_dir_abs": s,
                "dst_dir_abs": d,
                "src_dir_fls": [],
                "dst_dir_fls": [],
            }
            result = sync.get_sync_data()
            # Only subdir2 should be included since it has files
            for d in result:
                assert "subdir1" not in d["src_dir_abs"]
                assert "subdir2" in d["src_dir_abs"]

@pytest.mark.parametrize("overwrite_existing", [True, False])
def test_do_sync_copies_and_skips_existing(tmp_path, overwrite_existing):
    sync = Sync(("url", "/gvfs"))
    sync.overwrite_existing = overwrite_existing
    sync.source = str(tmp_path / "src")
    sync.destination = str(tmp_path / "dst")

    # Setup sync_data with a single source file
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()

    src_file = src_dir / "file.txt"
    src_file.write_text("test")

    dst_file = dst_dir / "file.txt"

    sync_data = {
        "src_dir_abs": str(src_dir),
        "dst_dir_abs": str(dst_dir),
        "src_dir_fls": [str(src_file)],
        "dst_dir_fls": [str(dst_file)],
    }

    with patch.object(sync, "copy_file") as mock_copy_file:
        sync.do_sync(sync_data)
        if overwrite_existing:
            mock_copy_file.assert_called_once_with(str(src_file), str(dst_file))
        else:
            mock_copy_file.assert_not_called()


def test_copy_file_prints_verbose_and_calls_gvfs_cp():
    sync = Sync(("mtp_url", "mtp_gvfs"), verbose=True)
    with patch("pysyncdroid.sync.gvfs.cp") as mock_cp, patch.object(sync, "_verbose") as mock_verbose:
        sync.copy_file("/src/file", "/dst/file")
        mock_verbose.assert_called_once_with("Copying /src/file to /dst/file")
        mock_cp.assert_called_once_with("/src/file", "/dst/file")


def test_handle_destination_dir_data_remove_and_synchronize(tmp_path):
    sync_remove = Sync(("url", "/gvfs"))
    sync_remove.unmatched = REMOVE

    sync_sync = Sync(("url", "/gvfs"))
    sync_sync.unmatched = SYNCHRONIZE

    sync_ignore = Sync(("url", "/gvfs"))
    sync_ignore.unmatched = IGNORE

    sync_remove_copy_mock = MagicMock()
    sync_sync_copy_mock = MagicMock()

    unmatched_file = "/dst/unmatched.file"
    src_dir = "/src"

    sync_data = {
        "dst_dir_fls": [unmatched_file],
        "src_dir_abs": src_dir,
        "dst_dir_abs": "/dst",
    }

    with patch("pysyncdroid.sync.gvfs.rm") as mock_rm:
        # Test REMOVE: calls rm and verbose
        with patch.object(sync_remove, "_verbose") as mock_verbose:
            sync_remove.gvfs_wrapper = MagicMock()
            sync_remove.handle_destination_dir_data(sync_data)
            sync_remove.gvfs_wrapper.assert_called_once_with(mock_rm, unmatched_file)
            mock_verbose.assert_called_once()

    # Test SYNCHRONIZE copies the unmatched file back to source
    with patch.object(sync_sync, "copy_file") as mock_copy, patch.object(sync_sync, "_verbose") as mock_verbose:
        sync_sync.gvfs_wrapper = MagicMock()
        sync_sync.handle_destination_dir_data(sync_data)
        expected_dst_file = unmatched_file.replace(sync_data["dst_dir_abs"], sync_data["src_dir_abs"])
        mock_copy.assert_called_once_with(src_file=unmatched_file, dst_file=expected_dst_file)

    # Test IGNORE: does nothing
    sync_ignore.handle_destination_dir_data({"dst_dir_fls": [], "src_dir_abs": src_dir, "dst_dir_abs": "/dst"})


def test_sync_method_runs_entire_flow(tmp_path):
    # Prepare directory structure and files
    src = tmp_path / "src"
    src.mkdir()
    (src / "file1.txt").write_text("abc")
    (src / "subdir").mkdir()
    (src / "subdir" / "file2.txt").write_text("def")

    sync = Sync(("url", str(tmp_path)))
    sync.source = str(src)
    sync.destination = "/destination"
    sync.unmatched = IGNORE

    # Patch underlying methods to monitor calls
    with patch.object(sync, "get_sync_data") as mock_get_sync_data, \
         patch.object(sync, "do_sync") as mock_do_sync, \
         patch.object(sync, "handle_destination_dir_data") as mock_handle_dest, \
         patch.object(sync, "_verbose") as mock_verbose:

        # Prepare data returned by get_sync_data
        sync_data = {
            "src_dir_abs": str(src),
            "dst_dir_abs": "/destination",
            "src_dir_fls": [str(src / "file1.txt"), str(src / "subdir" / "file2.txt")],
            "dst_dir_fls": [],
        }
        mock_get_sync_data.return_value = [sync_data]

        sync.sync()
        mock_verbose.assert_any_call(f'Gathering list of files to synchronize in "{sync.source}", this may take a while ...')
        mock_do_sync.assert_called_once_with(sync_data)
        # Since unmatched == IGNORE, handle_destination_dir_data is not called
        mock_handle_dest.assert_not_called()


def test_sync_method_prints_no_files_to_sync(tmp_path, capfd):
    src = tmp_path / "src"
    src.mkdir()
    sync = Sync(("url", str(tmp_path)))
    sync.source = str(src)
    sync.destination = "/destination"
    sync.unmatched = IGNORE

    # Return one sync_data dict with empty src_dir_fls to trigger "No files to sync" message
    sync_data = {
        "src_dir_abs": str(src),
        "dst_dir_abs": "/destination",
        "src_dir_fls": [],
        "dst_dir_fls": [],
    }

    with patch.object(sync, "get_sync_data", return_value=[sync_data]):
        sync.sync()
        out, _ = capfd.readouterr()
        assert "No files to sync" in out
