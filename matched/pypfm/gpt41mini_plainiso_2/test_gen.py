import pytest
from pypfm import PFMLoader, __version__

def test_version_string():
    assert isinstance(__version__, str)
    assert __version__ == "1.0.0"

def test_pfmloader_init_empty_path():
    loader = PFMLoader("")
    # The internal .path attribute should reflect what was passed
    assert loader.path == ""
    assert loader._loaded is False

def test_pfmloader_init_none_path():
    loader = PFMLoader(None)
    assert loader.path is None
    assert loader._loaded is False

def test_pfmloader_properties_before_load():
    loader = PFMLoader("dummy.pfm")
    with pytest.raises(AttributeError):
        _ = loader.width
    with pytest.raises(AttributeError):
        _ = loader.height
    with pytest.raises(AttributeError):
        _ = loader.frame_count

def test_pfmloader_load_invalid_file(tmp_path):
    # Create a dummy empty file that is not a valid PFM file
    f = tmp_path / "invalid.pfm"
    f.write_text("This is not a PFM file")
    loader = PFMLoader(str(f))
    with pytest.raises(Exception):
        loader.load()

def test_pfmloader_load_and_properties(monkeypatch, tmp_path):
    # We'll patch the internal _parse_pfm method to simulate a load
    class DummyLoader(PFMLoader):
        def _parse_pfm(self):
            self.width = 2
            self.height = 3
            self.frame_count = 1
            self.data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    loader = DummyLoader("dummy.pfm")
    loader.load()
    assert loader._loaded is True
    assert loader.width == 2
    assert loader.height == 3
    assert loader.frame_count == 1
    assert isinstance(loader.data, list)
    assert len(loader.data) == 3
    assert all(len(row) == 2 for row in loader.data)

def test_pfmloader_repr_before_and_after_load():
    loader = PFMLoader("dummy.pfm")
    r1 = repr(loader)
    assert "PFMLoader" in r1
    # Patch _parse_pfm to prevent loading error and simulate load
    def dummy_parse(self):
        self.width = 1
        self.height = 1
        self.frame_count = 1
        self.data = [[0]]
    loader._parse_pfm = dummy_parse.__get__(loader)
    loader.load()
    r2 = repr(loader)
    assert "PFMLoader" in r2
    assert "width=1" in r2 or "1" in r2

def test_pfmloader_load_called_twice(monkeypatch):
    class DummyLoader(PFMLoader):
        def _parse_pfm(self):
            self.width = 1
            self.height = 1
            self.frame_count = 1
            self.data = [[0]]

    loader = DummyLoader("dummy.pfm")
    loader.load()
    with pytest.raises(RuntimeError):
        loader.load()

def test_pfmloader_data_before_load_attribute_error():
    loader = PFMLoader("dummy.pfm")
    with pytest.raises(AttributeError):
        _ = loader.data

def test_pfmloader_load_file_not_found():
    loader = PFMLoader("nonexistent.pfm")
    with pytest.raises(FileNotFoundError):
        loader.load()

def test_pfmloader_load_empty_file(tmp_path):
    f = tmp_path / "empty.pfm"
    f.write_text("")
    loader = PFMLoader(str(f))
    with pytest.raises(Exception):
        loader.load()

def test_pfmloader_load_valid_file(tmp_path):
    # Write a minimal valid PFM file if possible - PFM is binary, so simulate a minimal valid header
    # We'll patch _parse_pfm to bypass file parsing and just set attributes
    f = tmp_path / "valid.pfm"
    f.write_bytes(b'PFM content')
    class DummyLoader(PFMLoader):
        def _parse_pfm(self):
            self.width = 4
            self.height = 4
            self.frame_count = 2
            self.data = [[0]*4 for _ in range(4)]
    loader = DummyLoader(str(f))
    loader.load()
    assert loader.width == 4
    assert loader.height == 4
    assert loader.frame_count == 2

def test_pfmloader_load_raises_if_no_path():
    loader = PFMLoader(None)
    with pytest.raises(ValueError):
        loader.load()
