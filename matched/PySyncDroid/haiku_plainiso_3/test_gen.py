import os
import shutil

import pytest

from pysyncdroid import exceptions
from pysyncdroid import sync as sync_module
from pysyncdroid.sync import IGNORE, REMOVE, SYNCHRONIZE, Sync


MTP_DETAILS = (
    "mtp://device/",
    "/run/user/1000/gvfs/mtp:host=device",
)


def make_sync(**kwargs):
    kwargs.setdefault("source", "/src")
    kwargs.setdefault("destination", "/dst")
    return Sync(MTP_DETAILS, **kwargs)


def make_sync_data(src_dir_abs, dst_dir_abs, src_files=(), dst_files=()):
    return {
        "src_dir_abs": src_dir_abs,
        "dst_dir_abs": dst_dir_abs,
        "src_dir_fls": list(src_files),
        "dst_dir_fls": list(dst_files),
    }


@pytest.fixture
def fake_gvfs(monkeypatch):
    """Replace gvfs operations with recorders that do not touch the disk."""
    calls = {"mount": [], "mkdir": [], "cp": [], "rm": []}

    def make_recorder(name):
        def _recorder(*args):
            calls[name].append(args)

        return _recorder

    for name in calls:
        monkeypatch.setattr(sync_module.gvfs, name, make_recorder(name))
    return calls


@pytest.fixture
def local_gvfs(monkeypatch):
    """Replace gvfs operations with local filesystem equivalents."""

    def cp(src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    def mkdir(path):
        os.makedirs(path, exist_ok=True)

    def rm(path):
        os.remove(path)

    monkeypatch.setattr(sync_module.gvfs, "cp", cp)
    monkeypatch.setattr(sync_module.gvfs, "mkdir", mkdir)
    monkeypatch.setattr(sync_module.gvfs, "rm", rm)


@pytest.fixture
def identity_readlink(monkeypatch):
    monkeypatch.setattr(sync_module, "readlink", lambda path: path)


def test_readlink_empty_path_returned_without_running_command(monkeypatch):
    def fail(cmd):
        raise AssertionError("run_bash_cmd should not be called")

    monkeypatch.setattr(sync_module, "run_bash_cmd", fail)
    assert sync_module.readlink("") == ""


def test_readlink_returns_command_output_and_passes_expected_args(monkeypatch):
    captured = {}

    def fake_run(cmd):
        captured["cmd"] = cmd
        return "/resolved/path"

    monkeypatch.setattr(sync_module, "run_bash_cmd", fake_run)

    assert sync_module.readlink("rel") == "/resolved/path"
    assert captured["cmd"] == ["readlink", "-f", "rel"]


def test_readlink_expands_tilde_before_resolving(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    captured = {}

    def fake_run(cmd):
        captured["cmd"] = cmd
        return cmd[-1]

    monkeypatch.setattr(sync_module, "run_bash_cmd", fake_run)

    result = sync_module.readlink("~/music")

    expected = os.path.join(str(tmp_path), "music")
    assert result == expected
    assert captured["cmd"][-1] == expected


def test_init_stores_mtp_details_and_defaults():
    s = make_sync(source="/a", destination="/b")

    assert s.mtp_url == MTP_DETAILS[0]
    assert s.mtp_gvfs_path == MTP_DETAILS[1]
    assert s.source == "/a"
    assert s.destination == "/b"
    assert s.unmatched == IGNORE
    assert s.overwrite_existing is False
    assert s.verbose is False
    assert s.ignore_file_types is None


def test_init_lowercases_ignore_file_types():
    s = make_sync(ignore_file_types=["JPG", "Tmp"])
    assert s.ignore_file_types == ["jpg", "tmp"]


def test_verbose_prints_only_when_enabled(capsys):
    make_sync(verbose=True)._verbose("hello")
    assert capsys.readouterr().out == "hello\n"

    make_sync(verbose=False)._verbose("silent")
    assert capsys.readouterr().out == ""


def test_gvfs_wrapper_calls_function_once_on_success(fake_gvfs):
    received = []

    def func(*args):
        received.append(args)

    make_sync().gvfs_wrapper(func, "x", "y")

    assert received == [("x", "y")]
    assert fake_gvfs["mount"] == []


def test_gvfs_wrapper_remounts_and_retries_on_connection_reset(fake_gvfs):
    attempts = []

    def flaky(*args):
        attempts.append(args)
        if len(attempts) == 1:
            raise exceptions.BashException("Error: Connection reset by peer")

    make_sync().gvfs_wrapper(flaky, "arg")

    assert len(attempts) == 2
    assert fake_gvfs["mount"] == [(MTP_DETAILS[0],)]


def test_gvfs_wrapper_reraises_unhandled_errors(fake_gvfs):
    def failing(*args):
        raise exceptions.BashException("Permission denied")

    with pytest.raises(exceptions.BashException):
        make_sync().gvfs_wrapper(failing)

    assert fake_gvfs["mount"] == []


def test_set_source_abs_resolves_relative_source_against_cwd(
    tmp_path, monkeypatch, identity_readlink
):
    (tmp_path / "docs").mkdir()
    monkeypatch.chdir(tmp_path)

    s = make_sync(source="docs")
    s.set_source_abs()

    assert s.source == os.path.join(os.getcwd(), "docs")


def test_set_source_abs_falls_back_to_device_prefix(
    tmp_path, monkeypatch, identity_readlink
):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    device = tmp_path / "device"
    (device / "photos").mkdir(parents=True)
    monkeypatch.chdir(cwd)

    s = Sync((MTP_DETAILS[0], str(device)), source="photos", destination="/dst")
    s.set_source_abs()

    assert s.source == os.path.join(str(device), "photos")


def test_set_source_abs_raises_when_source_missing(
    tmp_path, monkeypatch, identity_readlink
):
    monkeypatch.chdir(tmp_path)

    s = make_sync(source="nope")
    with pytest.raises(OSError, match="does not exist"):
        s.set_source_abs()


def test_set_source_abs_raises_when_source_is_file(
    tmp_path, identity_readlink
):
    f = tmp_path / "file.txt"
    f.write_text("data")

    s = make_sync(source=str(f))
    with pytest.raises(OSError, match="is not a directory"):
        s.set_source_abs()


def test_set_destination_abs_keeps_destination_when_computer_is_target(
    identity_readlink,
):
    s = make_sync(source="/run/gvfs/mtp:host=device/Music", destination="/home/u/out")
    s.set_destination_abs()

    assert s.destination == "/home/u/out"


def test_set_destination_abs_joins_device_path_when_device_is_target(
    identity_readlink,
):
    s = make_sync(source="/home/u/src", destination="Music")
    s.set_destination_abs()

    assert s.destination == os.path.join(MTP_DETAILS[1], "Music")


def test_set_destination_subdir_abs_maps_nested_directory():
    s = make_sync(source="/a/b", destination="/dst")

    result = s.set_destination_subdir_abs("/a/b/c/d")

    assert result == os.path.join("/dst", "c", "d")


def test_set_destination_subdir_abs_for_root_source_returns_destination():
    s = make_sync(source="/a/b", destination="/dst")

    result = s.set_destination_subdir_abs("/a/b")

    assert result == os.path.join("/dst", "")


def test_sync_data_template_strips_trailing_slashes_and_initializes_lists():
    s = make_sync()

    data = s.sync_data_template("/a/b/", "/d/e/")

    assert data == {
        "src_dir_abs": "/a/b",
        "dst_dir_abs": "/d/e",
        "src_dir_fls": [],
        "dst_dir_fls": [],
    }


def test_handle_ignored_file_type_is_case_insensitive():
    s = make_sync(ignore_file_types=["jpg"])

    with pytest.raises(exceptions.IgnoredTypeException):
        s.handle_ignored_file_type("IMG_001.JPG")

    # allowed file and extensionless file should not raise
    s.handle_ignored_file_type("notes.txt")
    s.handle_ignored_file_type("README")


def test_get_source_subdir_data_filters_ignored_files():
    s = make_sync(ignore_file_types=["tmp"])
    data = make_sync_data("/src/x", "/dst/x")

    s.get_source_subdir_data(["a.txt", "b.tmp"], data)

    assert data["src_dir_fls"] == [os.path.join("/src/x", "a.txt")]


def test_get_destination_subdir_data_creates_missing_directory(
    tmp_path, fake_gvfs
):
    missing = str(tmp_path / "new")
    data = make_sync_data("/src/x", missing)

    make_sync()._verbose  # ensure instance creation is valid
    make_sync(verbose=True).get_destination_subdir_data(data)

    assert fake_gvfs["mkdir"] == [(missing,)]
    assert data["dst_dir_fls"] == []


def test_do_sync_skips_existing_files_when_not_overwriting(fake_gvfs):
    s = make_sync(overwrite_existing=False)
    data = make_sync_data(
        "/src/x", "/dst/x",
        src_files=["/src/x/a.txt"],
        dst_files=["/dst/x/a.txt"],
    )

    s.do_sync(data)

    assert fake_gvfs["cp"] == []
    assert data["dst_dir_fls"] == []


def test_do_sync_overwrites_existing_files_when_enabled(fake_gvfs):
    s = make_sync(overwrite_existing=True)
    data = make_sync_data(
        "/src/x", "/dst/x",
        src_files=["/src/x/a.txt"],
        dst_files=["/dst/x/a.txt"],
    )

    s.do_sync(data)

    assert fake_gvfs["cp"] == [("/src/x/a.txt", "/dst/x/a.txt")]
    assert data["dst_dir_fls"] == []


def test_handle_destination_dir_data_removes_unmatched_files(fake_gvfs):
    s = make_sync(unmatched=REMOVE)
    data = make_sync_data(
        "/src/x", "/dst/x", dst_files=["/dst/x/old.txt"]
    )

    s.handle_destination_dir_data(data)

    assert fake_gvfs["rm"] == [("/dst/x/old.txt",)]


def test_handle_destination_dir_data_synchronizes_unmatched_files_back(
    fake_gvfs,
):
    s = make_sync(unmatched=SYNCHRONIZE)
    data = make_sync_data(
        "/src/x", "/dst/x", dst_files=["/dst/x/old.txt"]
    )

    s.handle_destination_dir_data(data)

    assert fake_gvfs["cp"] == [("/dst/x/old.txt", "/src/x/old.txt")]
    assert fake_gvfs["rm"] == []


def test_sync_end_to_end_copies_files_and_respects_ignore_list(
    tmp_path, local_gvfs
):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("alpha")
    (src / "sub" / "b.txt").write_text("beta")
    (src / "sub" / "skip.tmp").write_text("ignored")

    s = Sync(
        MTP_DETAILS,
        source=str(src),
        destination=str(dst),
        ignore_file_types=["tmp"],
    )
    s.destination = str(dst)
    s.sync()

    assert (dst / "a.txt").read_text() == "alpha"
    assert (dst / "sub" / "b.txt").read_text() == "beta"
    assert not (dst / "sub" / "skip.tmp").exists()
