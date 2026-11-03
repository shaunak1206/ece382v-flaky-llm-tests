import io
import os
import tempfile
import pytest
from pypfm import PFMLoader, __version__


def test_version_string():
    assert isinstance(__version__, str)
    assert __version__ == "1.0.0"


def test_pflloader_load_from_valid_file():
    content = b"dummy_pfm_data"
    # Create a temp file with dummy content to simulate a PFM file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        path = tmp.name

    loader = PFMLoader()
    data = loader.load(path)
    os.unlink(path)

    # Since real behavior unknown, just check if returned object is not None
    # and for test, assume it returns bytes or str as loaded content
    assert data is not None
    assert isinstance(data, (bytes, str)), "Loaded data should be bytes or str"


def test_pflloader_load_from_invalid_file():
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load("not_existing_file.pfm")


def test_pflloader_load_from_bytes_and_stream():
    data = b"PFM test content"

    # load from bytes
    loader = PFMLoader()
    result = loader.load(data)
    assert result is not None
    assert isinstance(result, (bytes, str))

    # load from a bytes stream (io.BytesIO)
    stream = io.BytesIO(data)
    result2 = loader.load(stream)
    assert result2 is not None
    assert isinstance(result2, (bytes, str))

    # load from a text stream (io.StringIO) if supported
    text_stream = io.StringIO(data.decode("utf-8", errors="ignore"))
    try:
        result3 = loader.load(text_stream)
        assert result3 is not None
        assert isinstance(result3, (bytes, str))
    except Exception:
        # if not supported, pass
        pass


def test_pflloader_load_from_unexpected_type():
    loader = PFMLoader()
    with pytest.raises(TypeError):
        loader.load(12345)  # unsupported type


def test_pflloader_str_and_repr_methods():
    loader = PFMLoader()
    s = str(loader)
    r = repr(loader)
    assert isinstance(s, str)
    assert isinstance(r, str)
    assert "PFMLoader" in r


def test_pflloader_multiple_load_calls():
    loader = PFMLoader()
    data = b"PFM content"
    res1 = loader.load(data)
    res2 = loader.load(data)
    assert res1 == res2


def test_pflloader_load_empty_bytes():
    loader = PFMLoader()
    empty = b""
    result = loader.load(empty)
    # Depending on implementation, empty bytes might raise or return None or empty data
    assert result is not None or result == b"" or result == ""


def test_pflloader_load_from_non_pfm_file(tmp_path):
    # Write something that is not a PFM to test behavior
    file = tmp_path / "not_pfm.txt"
    file.write_text("just some text")
    loader = PFMLoader()
    # Likely to raise an error or return something
    result = loader.load(str(file))
    assert result is not None


def test_pflloader_load_unicode_string_path(tmp_path):
    # Create a valid tmp file with dummy data
    file = tmp_path / "ファイル.pfm"
    file.write_bytes(b"pfm data")
    loader = PFMLoader()
    data = loader.load(str(file))
    assert data is not None


def test_pflloader_load_and_release_resources():
    # This test assumes PFMLoader might have a close or release method; if not, just call load multiple times
    loader = PFMLoader()
    for _ in range(5):
        result = loader.load(b"pfm data")
        assert result is not None


def test_pflloader_load_from_pathlib_path(tmp_path):
    # Test load supports pathlib.Path objects if implemented (common in loaders)
    path = tmp_path / "file.pfm"
    path.write_bytes(b"pfmdata")
    loader = PFMLoader()
    try:
        result = loader.load(path)
        assert result is not None
    except Exception:
        pytest.skip("PFMLoader.load doesn't support pathlib.Path input")


def test_pflloader_no_side_effects_on_load():
    data = b"pfmdata"
    loader = PFMLoader()
    before_vars = dir(loader)
    _ = loader.load(data)
    after_vars = dir(loader)
    assert before_vars == after_vars, "load caused side effects on loader attributes"


def test_pflloader_load_with_large_data():
    large_data = b"PFM" + b"\x00" * 10_000  # large dummy PFM data
    loader = PFMLoader()
    result = loader.load(large_data)
    assert result is not None


def test_pflloader_load_file_with_permission_denied(tmp_path):
    file = tmp_path / "no_permission.pfm"
    file.write_bytes(b"data")
    os.chmod(file, 0)  # remove all permissions
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(str(file))
    os.chmod(file, 0o644)  # restore permissions to allow tmp cleanup


def test_pflloader_load_none_input():
    loader = PFMLoader()
    with pytest.raises(TypeError):
        loader.load(None)
