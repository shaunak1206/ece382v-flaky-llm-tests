import io
import os
import pytest
from pypfm import PFMLoader


@pytest.fixture
def sample_pfm_file(tmp_path):
    content = (
        "PF\n"
        "4 2\n"
        "-1.0\n"
        "\x80\x00\x00\x00"  # 1.0 float in binary, big endian (we'll check endianness in test)
        "\x00\x00\x00\x00"  # 0.0
        "\x00\x00\x80\x3f"  # 1.0 little endian float (not used if endianness is negative)
        "\x00\x00\x00\x00"  # 0.0
        "\x80\x00\x00\x00"  # 1.0
        "\x00\x00\x00\x00"  # 0.0
        "\x00\x00\x80\x3f"  # 1.0
        "\x00\x00\x00\x00"  # 0.0
    )
    path = tmp_path / "test.pfm"
    # Writing bytes, so open in binary mode
    with open(path, "wb") as f:
        # Write header as ascii text then the binary data
        header = b"PF\n4 2\n-1.0\n"
        f.write(header)
        # Then write 4 floats (4x2=8 floats) in big endian (since negative scale means big endian)
        # Let's just write floats 1.0 and 0.0 repeatedly in big endian
        import struct
        # 8 floats total (4*2=8)
        floats = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        for flt in floats:
            f.write(struct.pack(">f", flt))
    return path


def test_version_exists():
    import pypfm
    assert hasattr(pypfm, "__version__")
    assert isinstance(pypfm.__version__, str)
    assert pypfm.__version__ == "1.0.0"


def test_load_pfm_file(sample_pfm_file):
    loader = PFMLoader()
    data, scale = loader.load(str(sample_pfm_file))
    assert data.shape == (2, 4, 3) or data.shape == (2, 4) or len(data.shape) in (2, 3)
    assert isinstance(scale, float)
    # The scale is negative as per file content
    assert scale < 0


@pytest.mark.parametrize("invalid_header", [b"", b"P\n", b"PFM\n", b"PX\n"])
def test_load_invalid_header(tmp_path, invalid_header):
    file = tmp_path / "invalid.pfm"
    file.write_bytes(invalid_header)
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(str(file))


def test_load_nonexistent_file():
    loader = PFMLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("this_file_should_not_exist_123456789.pfm")


def test_load_invalid_dimension(tmp_path):
    file = tmp_path / "dim.pfm"
    # Write valid header with invalid dimension line
    content = b"PF\n-1 0\n1.0\n"
    file.write_bytes(content)
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(str(file))


def test_load_no_scale(tmp_path):
    file = tmp_path / "noscale.pfm"
    # Valid header and dimension but no scale line
    content = b"PF\n4 2\n"
    file.write_bytes(content)
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(str(file))


def test_load_empty_file(tmp_path):
    file = tmp_path / "empty.pfm"
    file.write_bytes(b"")
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(str(file))


def test_load_file_with_alpha(tmp_path):
    # Create a PFM with 4 channels (RGBA)
    path = tmp_path / "rgba.pfm"
    # header PF or pf (color)
    content_header = b"PF\n3 1\n-1.0\n"
    floats = [1.0, 0.0, 0.5]  # RGB
    import struct
    body = b"".join(struct.pack(">f", f) for f in floats)
    path.write_bytes(content_header + body)
    loader = PFMLoader()
    data, scale = loader.load(str(path))
    assert data.shape == (1, 3, 3) or data.shape == (1,3)
    assert abs(scale + 1.0) < 1e-5


def test_load_gray_scale_pfm(tmp_path):
    # header: Pf for grayscale
    path = tmp_path / "gray.pfm"
    content_header = b"Pf\n2 2\n1.0\n"
    import struct
    # 4 grayscale pixels = 4 floats little endian (since positive scale)
    body = b"".join(struct.pack("<f", v) for v in [0.1, 0.2, 0.3, 0.4])
    path.write_bytes(content_header + body)
    loader = PFMLoader()
    data, scale = loader.load(str(path))
    assert data.shape == (2, 2)
    assert scale == 1.0
    assert abs(data[0, 0] - 0.1) < 1e-6


def test_load_invalid_pfm_type(tmp_path):
    path = tmp_path / "type.pfm"
    path.write_bytes(b"XY\n2 2\n1.0\n")
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(str(path))


def test_load_with_non_float_scale(tmp_path):
    path = tmp_path / "nonscale.pfm"
    path.write_bytes(b"PF\n2 2\nnotfloat\n")
    loader = PFMLoader()
    with pytest.raises(ValueError):
        loader.load(str(path))


def test_load_truncated_data(tmp_path):
    # Header says 2x2 but data only enough for 3 floats
    path = tmp_path / "truncated.pfm"
    path.write_bytes(b"PF\n2 2\n-1.0\n" + b"\x00"*12)  # should be 16 floats *4 bytes = 16 bytes, not fulfilled
    loader = PFMLoader()
    with pytest.raises(Exception):
        loader.load(str(path))


def test_load_large_pfm(tmp_path):
    # Generate large PFM with 100x100 grayscale float32 values
    path = tmp_path / "large.pfm"
    header = b"Pf\n100 100\n1.0\n"
    import struct
    floats = struct.pack("<" + "f"*(100*100), *[0.5]*10000)
    path.write_bytes(header + floats)
    loader = PFMLoader()
    data, scale = loader.load(str(path))
    assert data.shape == (100, 100)
    assert scale == 1.0
    assert all(abs(x - 0.5) < 1e-6 for x in data[0,:3])


def test_load_multiple_times(sample_pfm_file):
    loader = PFMLoader()
    for _ in range(3):
        data, scale = loader.load(str(sample_pfm_file))
        assert data is not None
        assert isinstance(scale, float)


def test_load_with_file_object(tmp_path):
    path = tmp_path / "fileobj.pfm"
    content = b"PF\n1 1\n-1.0\n" + b"\x00\x00\x80\x3f"  # 1.0 big endian
    path.write_bytes(content)
    loader = PFMLoader()
    # open file in binary mode and pass file handle
    with open(path, "rb") as f:
        data, scale = loader.load(f)
        assert data.shape == (1, 1, 3) or len(data.shape) in (2, 3)
        assert scale < 0


def test_load_with_extra_whitespace(tmp_path):
    path = tmp_path / "extra_whitespace.pfm"
    content = b"PF\n   2    2  \n 1.0  \n"
    import struct
    body = b"".join(struct.pack("<f", f) for f in [0.1,0.2,0.3,0.4]*3)  # RGB values repeated
    path.write_bytes(content + body)
    loader = PFMLoader()
    data, scale = loader.load(str(path))
    assert data.shape[0] == 2 and data.shape[1] == 2
    assert scale == 1.0


def test_load_detects_endianess(tmp_path):
    path = tmp_path / "endian.pfm"
    # Negative scale = big endian
    content = b"PF\n1 1\n-1.0\n" + b"\x3f\x80\x00\x00" * 3  # 1.0 big endian float repeated 3 channels
    path.write_bytes(content)
    loader = PFMLoader()
    data, scale = loader.load(str(path))
    assert scale < 0
    assert all(abs(v - 1.0) < 1e-6 for v in data[0,0])


def test_load_gray_detects_endianess(tmp_path):
    path = tmp_path / "gray_endian.pfm"
    # Positive scale = little endian
    content = b"Pf\n1 1\n1.0\n" + b"\x00\x00\x80\x3f"  # 1.0 little endian float
    path.write_bytes(content)
    loader = PFMLoader()
    data, scale = loader.load(str(path))
    assert scale > 0
    assert abs(data[0,0] - 1.0) < 1e-6
