import random
import numpy as np
import pytest

from pypfm import PFMLoader

def test_version_string():
    # __version__ is a string and matches semantic versioning pattern
    from pypfm import __version__
    assert isinstance(__version__, str)
    parts = __version__.split('.')
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)

def test_pfploader_init_with_path(tmp_path):
    random.seed(1234)
    np.random.seed(1234)
    test_file = tmp_path / "dummy.pfm"
    test_file.write_text("dummy content")
    loader = PFMLoader(str(test_file))
    # loader path equals input path
    assert loader.path == str(test_file)

def test_pfploader_init_with_nonexistent_path(tmp_path):
    random.seed(1234)
    np.random.seed(1234)
    fake_path = tmp_path / "no_such_file.pfm"
    # The code might not raise at init, just store the path
    loader = PFMLoader(str(fake_path))
    assert loader.path == str(fake_path)

def test_pfploader_load_returns_expected_array(tmp_path):
    random.seed(42)
    np.random.seed(42)
    # Create a .pfm file with a known content according to PFMLoader expectations
    # Since we do not have the pypfm/pypfm.py source code, we assume load() returns np.ndarray.
    # We create a minimal PFM file representing a 2x2 image with float32 rgb values.
    pfm_content = (
        b"PFM\n-1.0\n2 2\n" +  # header: PFM, scale negative means little endian, size 2x2
        np.array([
            [[0.1, 0.2, 0.3],
             [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9],
             [1.0, 1.1, 1.2]],
        ], dtype=np.float32).tobytes()
    )
    test_file = tmp_path / "test.pfm"
    test_file.write_bytes(pfm_content)

    loader = PFMLoader(str(test_file))
    data = loader.load()

    np.testing.assert_allclose(data, np.array([
        [[0.1, 0.2, 0.3],
         [0.4, 0.5, 0.6]],
        [[0.7, 0.8, 0.9],
         [1.0, 1.1, 1.2]],
    ], dtype=np.float32), rtol=1e-6, atol=1e-7)

def test_load_invalid_file_raises(tmp_path):
    random.seed(7)
    np.random.seed(7)
    test_file = tmp_path / "invalid.pfm"
    test_file.write_text("this is not a valid pfm file")
    loader = PFMLoader(str(test_file))
    with pytest.raises(Exception):
        loader.load()

def test_load_empty_file_raises(tmp_path):
    random.seed(7)
    np.random.seed(7)
    test_file = tmp_path / "empty.pfm"
    test_file.write_bytes(b"")
    loader = PFMLoader(str(test_file))
    with pytest.raises(Exception):
        loader.load()

def test_load_with_negative_scale_and_endian_little(tmp_path):
    random.seed(11)
    np.random.seed(11)
    # Create a PFM with negative scale (little endian)
    scale = -1.0
    width = 1
    height = 1
    header = f"PFM\n{scale}\n{width} {height}\n".encode()
    # one pixel RGB float32 values little endian
    vals = np.array([[[1.5, 2.5, 3.5]]], dtype='<f4')
    pfm_bytes = header + vals.tobytes()
    test_file = tmp_path / "little.pfm"
    test_file.write_bytes(pfm_bytes)

    loader = PFMLoader(str(test_file))
    data = loader.load()
    assert data.shape == (height, width, 3)
    np.testing.assert_allclose(data, np.array([[[1.5, 2.5, 3.5]]], dtype=np.float32), rtol=1e-6, atol=1e-7)

def test_load_with_positive_scale_and_endian_big(tmp_path):
    random.seed(12)
    np.random.seed(12)
    # Create a PFM with positive scale (big endian)
    scale = 1.0
    width = 1
    height = 1
    header = f"PFM\n{scale}\n{width} {height}\n".encode()
    # one pixel RGB float32 values big endian
    vals = np.array([[[4.5, 5.5, 6.5]]], dtype='>f4')
    pfm_bytes = header + vals.tobytes()
    test_file = tmp_path / "big.pfm"
    test_file.write_bytes(pfm_bytes)

    loader = PFMLoader(str(test_file))
    data = loader.load()
    assert data.shape == (height, width, 3)
    np.testing.assert_allclose(data, np.array([[[4.5, 5.5, 6.5]]], dtype=np.float32), rtol=1e-6, atol=1e-7)

def test_load_strips_excess_whitespace(tmp_path):
    random.seed(99)
    np.random.seed(99)
    # Write a PFM with extra spaces and blank lines in header
    pfm_header = b"PFM  \n\n  -1.0  \n  1  1  \n"
    vals = np.array([[[7.0, 8.0, 9.0]]], dtype=np.float32).tobytes()
    test_file = tmp_path / "extra_ws.pfm"
    test_file.write_bytes(pfm_header + vals)

    loader = PFMLoader(str(test_file))
    data = loader.load()
    np.testing.assert_allclose(data, np.array([[[7.0, 8.0, 9.0]]], dtype=np.float32), rtol=1e-6, atol=1e-7)

def test_load_multiline_file_raises_if_wrong_header(tmp_path):
    random.seed(15)
    np.random.seed(15)
    # Header does not start with PFM
    bad_header = b"PFX\n-1.0\n1 1\n"
    vals = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32).tobytes()
    test_file = tmp_path / "bad_header.pfm"
    test_file.write_bytes(bad_header + vals)

    loader = PFMLoader(str(test_file))
    with pytest.raises(Exception):
        loader.load()

def test_load_produces_float32_array(tmp_path):
    random.seed(17)
    np.random.seed(17)
    pfm_header = b"PFM\n-1.0\n1 1\n"
    vals = np.array([[[12.3, 45.6, 78.9]]], dtype=np.float32).tobytes()
    test_file = tmp_path / "typecheck.pfm"
    test_file.write_bytes(pfm_header + vals)
    loader = PFMLoader(str(test_file))
    data = loader.load()
    assert data.dtype == np.float32

def test_multiple_load_calls_return_consistent_data(tmp_path):
    random.seed(21)
    np.random.seed(21)
    pfm_content = (
        b"PFM\n-1.0\n2 1\n" +
        np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32).tobytes()
    )
    test_file = tmp_path / "repeat.pfm"
    test_file.write_bytes(pfm_content)
    loader = PFMLoader(str(test_file))
    data1 = loader.load()
    data2 = loader.load()
    np.testing.assert_allclose(data1, data2, rtol=1e-6, atol=1e-7)

def test_str_path_input_and_pathlib_path_equiv(tmp_path):
    random.seed(27)
    np.random.seed(27)
    pfm_content = (
        b"PFM\n-1.0\n1 1\n" +
        np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32).tobytes()
    )
    test_file = tmp_path / "str_path.pfm"
    test_file.write_bytes(pfm_content)

    loader_str = PFMLoader(str(test_file))
    data_str = loader_str.load()

    loader_pathlib = PFMLoader(test_file)
    data_pathlib = loader_pathlib.load()
    np.testing.assert_allclose(data_str, data_pathlib, rtol=1e-6, atol=1e-7)

def test_pfploader_repr_contains_path(tmp_path):
    random.seed(31)
    np.random.seed(31)
    test_file = tmp_path / "repr.pfm"
    test_file.write_bytes(b"PFM\n-1.0\n1 1\n" + np.zeros((1,1,3), dtype=np.float32).tobytes())
    loader = PFMLoader(str(test_file))
    r = repr(loader)
    assert str(test_file) in r


# No direct testing needed for pypfm.__init__ beyond version string shown above
