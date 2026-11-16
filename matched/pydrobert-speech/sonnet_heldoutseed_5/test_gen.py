import numpy as np
import pytest

from pydrobert.speech.filters import (
    ComplexGammatoneFilterBank,
    Fbank,
    GaborFilterBank,
    LinearFilterBank,
    TriangularOverlappingFilterBank,
)

RATE = 16000


def _half_len(width):
    return (width + 1) // 2 if width % 2 else width // 2 + 1


def _rebuild_from_truncated(bank, filt_idx, width):
    """Return (rebuilt, expected) following the documented recipe"""
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    if bank.is_real:
        half = np.zeros(_half_len(width), dtype=trnc.dtype)
        half[bin_idx:bin_idx + len(trnc)] = trnc
        expected = bank.get_frequency_response(filt_idx, width, half=True)
        return half, expected
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    expected = bank.get_frequency_response(filt_idx, width)
    return full, expected


@pytest.fixture(scope="module")
def tri_bank():
    return TriangularOverlappingFilterBank('mel', num_filts=40)


@pytest.fixture(scope="module")
def gabor_bank():
    return GaborFilterBank('mel', num_filts=40)


@pytest.fixture(scope="module")
def gammatone_bank():
    return ComplexGammatoneFilterBank('mel', num_filts=40)


# ---------------------------------------------------------------------------
# TriangularOverlappingFilterBank
# ---------------------------------------------------------------------------


def test_triangular_properties(tri_bank):
    assert isinstance(tri_bank, LinearFilterBank)
    assert tri_bank.num_filts == 40
    assert tri_bank.sampling_rate == RATE
    assert tri_bank.is_real
    assert not tri_bank.is_analytic
    assert tri_bank.is_zero_phase
    centers = tri_bank.centers_hz
    assert len(centers) == 40
    assert all(a < b for a, b in zip(centers[:-1], centers[1:]))
    assert len(tri_bank.supports_hz) == 40
    for c, (lo, hi) in zip(centers, tri_bank.supports_hz):
        assert lo < c < hi
    assert tri_bank.supports_hz[0][0] == pytest.approx(20., rel=1e-6)
    assert tri_bank.supports_hz[-1][1] == pytest.approx(RATE // 2, rel=1e-6)


def test_triangular_analytic_flags():
    bank = TriangularOverlappingFilterBank('mel', num_filts=10, analytic=True)
    assert bank.is_analytic
    assert not bank.is_real
    assert bank.is_zero_phase


@pytest.mark.parametrize('kwargs', [
    {'low_hz': -1.},
    {'low_hz': 100., 'high_hz': 50.},
    {'low_hz': 100., 'high_hz': 100.},
    {'high_hz': RATE // 2 + 1000},
])
def test_triangular_invalid_frequency_range(kwargs):
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank('mel', num_filts=10, **kwargs)


def test_triangular_supports_and_ms(tri_bank):
    supports = tri_bank.supports
    supports_ms = tri_bank.supports_ms
    assert len(supports) == len(supports_ms) == tri_bank.num_filts
    for (lo, hi), (lo_ms, hi_ms) in zip(supports, supports_ms):
        assert lo < 0 < hi
        assert lo_ms == pytest.approx(lo * 1000 / RATE)
        assert hi_ms == pytest.approx(hi * 1000 / RATE)


def test_triangular_frequency_response_shape(tri_bank):
    width = 4096
    idx = 20
    resp = tri_bank.get_frequency_response(idx, width)
    assert resp.shape == (width,)
    assert resp.dtype == np.float64
    assert resp.min() >= 0
    assert resp.max() <= 1. + 1e-9
    assert resp.max() > 0.9
    # peak is at the center frequency
    peak_hz = np.argmax(resp[:width // 2 + 1]) * RATE / width
    assert abs(peak_hz - tri_bank.centers_hz[idx]) <= RATE / width
    # nonzero region lies inside the support (positive frequencies)
    lo, hi = tri_bank.supports_hz[idx]
    nz = np.nonzero(resp[:width // 2 + 1])[0] * RATE / width
    assert nz.min() >= lo
    assert nz.max() <= hi


def test_triangular_real_response_is_symmetric(tri_bank):
    width = 512
    resp = tri_bank.get_frequency_response(10, width)
    assert np.allclose(resp[1:], resp[1:][::-1])


def test_triangular_analytic_suppresses_negative_frequencies():
    bank = TriangularOverlappingFilterBank('mel', num_filts=10, analytic=True)
    width = 512
    for idx in (0, 5, 9):
        resp = bank.get_frequency_response(idx, width)
        assert np.any(resp[:width // 2 + 1] > 0)
        assert np.all(resp[width // 2 + 1:] == 0)


@pytest.mark.parametrize('width', [512, 511])
def test_triangular_half_response(tri_bank, width):
    for idx in (0, 20, 39):
        full = tri_bank.get_frequency_response(idx, width)
        half = tri_bank.get_frequency_response(idx, width, half=True)
        assert len(half) == _half_len(width)
        assert np.allclose(half, full[:len(half)])


@pytest.mark.parametrize('analytic', [False, True])
@pytest.mark.parametrize('width', [512, 511])
def test_triangular_truncated_response_matches_full(analytic, width):
    bank = TriangularOverlappingFilterBank(
        'mel', num_filts=20, analytic=analytic)
    for idx in (0, 10, 19):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert bin_idx >= 0
        assert len(trnc) > 0
        rebuilt, expected = _rebuild_from_truncated(bank, idx, width)
        assert rebuilt.shape == expected.shape
        assert np.allclose(rebuilt, expected)


def test_triangular_impulse_response(tri_bank):
    width = 256
    ir = tri_bank.get_impulse_response(10, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.float64
    assert np.all(np.isfinite(ir))
    analytic = TriangularOverlappingFilterBank(
        'mel', num_filts=40, analytic=True)
    ir = analytic.get_impulse_response(10, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.complex128


# ---------------------------------------------------------------------------
# Fbank
# ---------------------------------------------------------------------------


def test_fbank_basic():
    bank = Fbank(num_filts=20)
    assert bank.num_filts == 20
    assert bank.is_real
    assert bank.is_zero_phase
    assert not bank.is_analytic
    assert len(bank.centers_hz) == 20
    width = 512
    for idx in (0, 10):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert full.shape == (width,)
        assert len(half) == _half_len(width)
        assert np.all(np.isfinite(full))
        assert full.min() >= 0
        assert full.max() <= 1. + 1e-9
        assert np.allclose(half, full[:len(half)])
        rebuilt, expected = _rebuild_from_truncated(bank, idx, width)
        assert np.allclose(rebuilt, expected)


# ---------------------------------------------------------------------------
# GaborFilterBank
# ---------------------------------------------------------------------------


def test_gabor_properties(gabor_bank):
    assert isinstance(gabor_bank, LinearFilterBank)
    assert gabor_bank.num_filts == 40
    assert gabor_bank.sampling_rate == RATE
    assert not gabor_bank.is_real
    assert gabor_bank.is_zero_phase
    assert isinstance(gabor_bank.is_analytic, bool)
    assert not gabor_bank.scaled_l2_norm
    assert not gabor_bank.erb
    centers = gabor_bank.centers_hz
    assert len(centers) == 40
    assert all(a < b for a, b in zip(centers[:-1], centers[1:]))
    for c, (lo, hi) in zip(centers, gabor_bank.supports_hz):
        assert lo < c < hi
    for lo, hi in gabor_bank.supports:
        assert lo < 0 < hi
        assert lo == -hi


def test_gabor_invalid_frequency_range():
    with pytest.raises(ValueError):
        GaborFilterBank('mel', num_filts=10, low_hz=-5.)
    with pytest.raises(ValueError):
        GaborFilterBank('mel', num_filts=10, low_hz=500., high_hz=400.)
    with pytest.raises(ValueError):
        GaborFilterBank('mel', num_filts=10, high_hz=RATE)


def test_gabor_frequency_response_peak_and_half(gabor_bank):
    width = 4096
    idx = 20
    resp = gabor_bank.get_frequency_response(idx, width)
    assert resp.shape == (width,)
    assert resp.min() >= 0
    assert resp.max() == pytest.approx(1., abs=0.05)
    peak_hz = np.argmax(resp[:width // 2 + 1]) * RATE / width
    assert abs(peak_hz - gabor_bank.centers_hz[idx]) <= RATE / width
    for w in (512, 511):
        full = gabor_bank.get_frequency_response(idx, w)
        half = gabor_bank.get_frequency_response(idx, w, half=True)
        assert len(half) == _half_len(w)
        assert np.allclose(half, full[:len(half)])


def test_gabor_l2_scaling_increases_peak():
    plain = GaborFilterBank('mel', num_filts=40)
    scaled = GaborFilterBank('mel', num_filts=40, scale_l2_norm=True)
    assert scaled.scaled_l2_norm
    width = 4096
    a = plain.get_frequency_response(20, width).max()
    b = scaled.get_frequency_response(20, width).max()
    assert b > a


def test_gabor_impulse_response_is_inverse_of_frequency_response(gabor_bank):
    width = 2048
    idx = 20
    ir = gabor_bank.get_impulse_response(idx, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.complex128
    freq = gabor_bank.get_frequency_response(idx, width)
    assert np.allclose(np.fft.fft(ir), freq, atol=1e-5)


def test_gabor_truncated_response_matches_full(gabor_bank):
    for width in (512, 1000):
        for idx in (0, 20, 39):
            bin_idx, trnc = gabor_bank.get_truncated_response(idx, width)
            assert 0 <= bin_idx < width
            assert len(trnc) <= width
            rebuilt, expected = _rebuild_from_truncated(gabor_bank, idx, width)
            assert np.allclose(rebuilt, expected, atol=2e-3)


# ---------------------------------------------------------------------------
# ComplexGammatoneFilterBank
# ---------------------------------------------------------------------------


def test_gammatone_properties(gammatone_bank):
    assert isinstance(gammatone_bank, LinearFilterBank)
    assert gammatone_bank.num_filts == 40
    assert gammatone_bank.order == 4
    assert gammatone_bank.sampling_rate == RATE
    assert not gammatone_bank.is_real
    assert not gammatone_bank.is_zero_phase
    assert not gammatone_bank.scaled_l2_norm
    assert not gammatone_bank.erb
    assert len(gammatone_bank.supports) == 40
    assert len(gammatone_bank.supports_hz) == 40
    centers = gammatone_bank.centers_hz
    assert all(a < b for a, b in zip(centers[:-1], centers[1:]))
    for c, (lo, hi) in zip(centers, gammatone_bank.supports_hz):
        assert lo < c < hi


@pytest.mark.parametrize('order', [0, -1, 1.5])
def test_gammatone_invalid_order(order):
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank('mel', num_filts=10, order=order)


def test_gammatone_frequency_response(gammatone_bank):
    width = 4096
    idx = 20
    resp = gammatone_bank.get_frequency_response(idx, width)
    assert resp.shape == (width,)
    assert resp.dtype == np.complex128
    mag = np.abs(resp)
    assert mag.max() == pytest.approx(1., abs=0.02)
    peak_hz = np.argmax(mag[:width // 2 + 1]) * RATE / width
    assert abs(peak_hz - gammatone_bank.centers_hz[idx]) <= RATE / width
    for w in (512, 511):
        full = gammatone_bank.get_frequency_response(idx, w)
        half = gammatone_bank.get_frequency_response(idx, w, half=True)
        assert len(half) == _half_len(w)
        assert np.allclose(half, full[:len(half)])


def test_gammatone_impulse_response_is_causal(gammatone_bank):
    width = 2048
    ir = gammatone_bank.get_impulse_response(20, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.complex128
    assert ir[0] == 0
    assert np.abs(ir).max() > 0
    assert np.all(np.isfinite(ir))


def test_gammatone_truncated_response_matches_full(gammatone_bank):
    width = 1024
    for idx in (0, 20, 39):
        bin_idx, trnc = gammatone_bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        assert len(trnc) <= width
        assert trnc.dtype == np.complex128
        rebuilt, expected = _rebuild_from_truncated(
            gammatone_bank, idx, width)
        assert np.allclose(rebuilt, expected, atol=5e-3)
