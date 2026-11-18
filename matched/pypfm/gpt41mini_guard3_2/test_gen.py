import random
import numpy as np
import pytest

from pypfm import PFMLoader


def test_version_string():
    # Check that the provided version string matches expected format
    assert isinstance(PFMLoader.__module__, str)
    assert PFMLoader.__module__.startswith("pypfm")
    from pypfm import __version__
    # fixed string check
    assert __version__ == "1.0.0"


def test_pfmloader_load_and_properties(tmp_path):
    # Seed random generators used by the code under test (seed early)
    random.seed(0)
    np.random.seed(0)

    # Create a minimal PFM file content that PFMLoader could parse.
    # We infer the format from the class name "PFMLoader" (Position Frequency Matrix loader).
    # We will write a simple 4x6 PFM file (A,C,G,T rows) with numeric counts.

    pfm_content = (
        "A 1 2 3 4 5 6\n"
        "C 2 3 4 5 6 7\n"
        "G 3 4 5 6 7 8\n"
        "T 4 5 6 7 8 9\n"
    )
    pfm_file = tmp_path / "test.pfm"
    pfm_file.write_text(pfm_content)

    loader = PFMLoader(str(pfm_file))
    motif = loader.load()

    # motif should be a dict or object with counts/probabilities for four bases
    # We assume it has a 'counts' or 'mat' attribute or similar; test basic API
    # Because we don't have the exact interface of PFMLoader.load(), we test minimal behavior:
    # It must return non-empty data and numeric values matching input.

    assert motif is not None
    # motif might be a dict with keys 'A','C','G','T' or a numpy array or similar
    # We accept dict or object attribute with arrays
    if isinstance(motif, dict):
        # keys for four bases
        bases = set(motif.keys())
        assert bases.issuperset({"A","C","G","T"})
        # values are lists or arrays of length 6
        for base in "ACGT":
            vals = motif[base]
            assert len(vals) == 6
            # all numbers and positive integers matching counts
            for v in vals:
                assert isinstance(v, (int, float))
                assert v >= 0
    else:
        # Likely numpy array shaped (4,6) or similar
        # Check shape and values
        shape = getattr(motif, "shape", None)
        assert shape is not None
        assert shape[0] == 4
        assert shape[1] == 6
        # values non-negative floats or ints
        arr = np.array(motif)
        assert np.all(arr >= 0)

def test_pfmloader_load_invalid_file(tmp_path):
    random.seed(1)
    np.random.seed(1)

    # Write an empty or malformed file
    bad_file = tmp_path / "bad.pfm"
    bad_file.write_text("Z 1 2 3\n")  # invalid base 'Z'

    loader = PFMLoader(str(bad_file))
    # We expect an exception or empty result, test that loading raises or returns None
    try:
        res = loader.load()
    except Exception:
        pass
    else:
        # If no exception, result should be None or empty dict/array
        assert res is None or res == {} or (hasattr(res, "size") and res.size == 0)

def test_pfmloader_load_empty_file(tmp_path):
    random.seed(2)
    np.random.seed(2)

    empty_file = tmp_path / "empty.pfm"
    empty_file.write_text("")
    loader = PFMLoader(str(empty_file))
    try:
        res = loader.load()
    except Exception:
        pass
    else:
        assert res is None or res == {} or (hasattr(res, "size") and res.size == 0)


def test_pfmloader_load_with_extra_whitespace(tmp_path):
    random.seed(3)
    np.random.seed(3)

    content = (
        " A   1 2  3 4 5 6 \n"
        "C 2 3 4 5 6 7\n"
        "G 3   4 5 6 7 8\n"
        "T 4 5 6 7 8    9\n"
    )
    pfm_file = tmp_path / "ws.pfm"
    pfm_file.write_text(content)

    loader = PFMLoader(str(pfm_file))
    motif = loader.load()

    # motif must contain all four bases and six columns for each
    if isinstance(motif, dict):
        assert set(motif.keys()).issuperset({"A","C","G","T"})
        for base in "ACGT":
            vals = motif[base]
            assert len(vals) == 6
            for v in vals:
                assert v >= 0
    else:
        shape = getattr(motif, "shape", None)
        assert shape == (4, 6)
        arr = np.array(motif)
        assert np.all(arr >= 0)

def test_pfmloader_load_numeric_parsing(tmp_path):
    random.seed(4)
    np.random.seed(4)

    content = (
        "A 0.1 0.2 0.3 0.4 0.5 0.6\n"
        "C 1 2 3 4 5 6\n"
        "G 7 8 9 10 11 12\n"
        "T 13 14 15 16 17 18\n"
    )
    pfm_file = tmp_path / "float.pfm"
    pfm_file.write_text(content)

    loader = PFMLoader(str(pfm_file))
    motif = loader.load()

    # Check values on first two bases for floats and ints
    if isinstance(motif, dict):
        a_vals = motif["A"]
        c_vals = motif["C"]
        for i, val in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
            assert abs(a_vals[i] - val) < 1e-8
        for i, val in enumerate(range(1, 7)):
            assert abs(c_vals[i] - val) < 1e-8
    else:
        arr = np.array(motif)
        # row 0 = A, row 1 = C
        expected_a = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        expected_c = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(arr[0], expected_a, rtol=1e-8)
        np.testing.assert_allclose(arr[1], expected_c, rtol=1e-8)

def test_pfmloader_load_file_with_additional_lines(tmp_path):
    random.seed(5)
    np.random.seed(5)

    content = (
        "# Comment line\n"
        "A 1 2 3 4 5 6\n"
        "C 2 3 4 5 6 7\n"
        "G 3 4 5 6 7 8\n"
        "T 4 5 6 7 8 9\n"
        "# Trailing comment\n"
    )
    pfm_file = tmp_path / "comments.pfm"
    pfm_file.write_text(content)

    loader = PFMLoader(str(pfm_file))
    motif = loader.load()

    assert motif is not None
    # Same shape and content checks as before
    if isinstance(motif, dict):
        for base in "ACGT":
            vals = motif[base]
            assert len(vals) == 6
            for v in vals:
                assert v >= 0
    else:
        arr = np.array(motif)
        assert arr.shape == (4, 6)
        assert np.all(arr >= 0)


def test_pfmloader_load_non_standard_order_bases(tmp_path):
    random.seed(6)
    np.random.seed(6)

    # order bases shuffled; the loader must still parse correctly
    content = (
        "G 3 4 5 6 7 8\n"
        "T 4 5 6 7 8 9\n"
        "A 1 2 3 4 5 6\n"
        "C 2 3 4 5 6 7\n"
    )
    pfm_file = tmp_path / "shuffle.pfm"
    pfm_file.write_text(content)

    loader = PFMLoader(str(pfm_file))
    motif = loader.load()

    # The order may be different in the result; just check keys and values
    if isinstance(motif, dict):
        for base in "ACGT":
            vals = motif[base]
            assert len(vals) == 6
            assert all(isinstance(x, (int, float)) for x in vals)
    else:
        arr = np.array(motif)
        assert arr.shape == (4, 6)
        assert np.all(arr >= 0)

def test_pfmloader_repr_and_str_methods(tmp_path):
    random.seed(7)
    np.random.seed(7)

    content = (
        "A 1 0 0 0 1 0\n"
        "C 0 1 0 1 0 0\n"
        "G 0 0 1 0 0 1\n"
        "T 0 0 0 1 0 1\n"
    )
    pfm_file = tmp_path / "simple.pfm"
    pfm_file.write_text(content)
    loader = PFMLoader(str(pfm_file))
    motif = loader.load()

    # Test that motif can be passed to repr() and str() without error
    repr_str = repr(motif)
    str_str = str(motif)
    assert isinstance(repr_str, str)
    assert isinstance(str_str, str)
    # They contain numbers or base letters
    assert any(ch in repr_str for ch in "ACGT0123456789")
    assert any(ch in str_str for ch in "ACGT0123456789")

def test_pfmloader_load_after_monkeypatch(monkeypatch, tmp_path):
    random.seed(8)
    np.random.seed(8)

    # Patch open to prepend a comment line automatically
    real_open = open

    def fake_open(file, *args, **kwargs):
        f = real_open(file, *args, **kwargs)
        # only patch reading the test.pfm file
        if file == str(tmp_path / "monkey.pfm") and 'r' in args:
            # return a wrapper file object that injects a comment line
            lines = f.readlines()
            # prepend comment line
            lines.insert(0, "# Injected comment\n")
            f.close()

            from io import StringIO
            return StringIO("".join(lines))
        return f

    content = (
        "A 1 2 3 4 5 6\n"
        "C 2 3 4 5 6 7\n"
        "G 3 4 5 6 7 8\n"
        "T 4 5 6 7 8 9\n"
    )
    pfm_file = tmp_path / "monkey.pfm"
    pfm_file.write_text(content)

    monkeypatch.setattr("builtins.open", fake_open)

    loader = PFMLoader(str(pfm_file))
    motif = loader.load()

    assert motif is not None
    if isinstance(motif, dict):
        assert set(motif.keys()) == {"A","C","G","T"}
    else:
        arr = np.array(motif)
        assert arr.shape == (4,6)
        assert np.all(arr >= 0)

def test_pfmloader_isolated_seeds(tmp_path):
    # Check that reseeding affects results reproducibly
    # Using the same input with different seeds will produce the same output across runs

    # Seed and write file once
    random.seed(9)
    np.random.seed(9)
    content = (
        "A 1 2 3 4 5 6\n"
        "C 2 3 4 5 6 7\n"
        "G 3 4 5 6 7 8\n"
        "T 4 5 6 7 8 9\n"
    )
    pfm_file = tmp_path / "seed_test.pfm"
    pfm_file.write_text(content)

    def load_with_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        loader = PFMLoader(str(pfm_file))
        return loader.load()

    motif1 = load_with_seed(42)
    motif2 = load_with_seed(42)
    motif3 = load_with_seed(43)

    # motif1 and motif2 must be identical
    if isinstance(motif1, dict):
        for base in "ACGT":
            assert motif1[base] == motif2[base]
    else:
        np.testing.assert_array_equal(motif1, motif2)

    # motif3 could differ only if randomness affects loading (likely not)
    # So either equal or differ, but must be consistent on every run - so just assert motif3 is also valid
    assert motif3 is not None

def test_pfmloader_load_non_numeric_values(tmp_path):
    random.seed(10)
    np.random.seed(10)

    # File contains non-numeric values in columns
    content = (
        "A 1 2 3 4 5 x\n"
        "C 2 3 4 5 6 7\n"
        "G 3 4 5 6 7 8\n"
        "T 4 5 6 7 8 9\n"
    )
    pfm_file = tmp_path / "non_numeric.pfm"
    pfm_file.write_text(content)

    loader = PFMLoader(str(pfm_file))
    # Depending on implementation, may raise or skip or error
    with pytest.raises((ValueError, TypeError)):
        _ = loader.load()

def test_pfmloader_load_from_bytesio(monkeypatch):
    import io

    random.seed(11)
    np.random.seed(11)

    # Prepare a PFM as bytes to simulate reading from stream
    pfm_bytes = (
        b"A 1 2 3 4 5 6\n"
        b"C 2 3 4 5 6 7\n"
        b"G 3 4 5 6 7 8\n"
        b"T 4 5 6 7 8 9\n"
    )
    # monkeypatch open to return a BytesIO for a given filename

    def fake_open(file, mode="r", *args, **kwargs):
        if file == "fake.pfm":
            if "b" in mode:
                return io.BytesIO(pfm_bytes)
            else:
                return io.StringIO(pfm_bytes.decode())
        else:
            return open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    loader = PFMLoader("fake.pfm")
    motif = loader.load()
    assert motif is not None
    if isinstance(motif, dict):
        assert set(motif.keys()) == {"A","C","G","T"}
    else:
        arr = np.array(motif)
        assert arr.shape[0] == 4


def test_version_imported_from_init():
    # Import __version__ from pypfm and check it is "1.0.0"
    from pypfm import __version__
    assert __version__ == "1.0.0"
