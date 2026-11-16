import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.filters import ComplexGammatoneFilterBank
from pydrobert.speech.filters import Fbank
from pydrobert.speech.filters import GaborFilterBank
from pydrobert.speech.filters import LinearFilterBank
from pydrobert.speech.filters import TriangularOverlappingFilterBank
from pydrobert.speech.scales import MelScaling

EPS = config.EFFECTIVE_SUPPORT_THRESHOLD


def _half_width(width):
    return width // 2 + 1 if width % 2 == 0 else (width + 1) // 2


def _expand_complex(bin_idx, trnc, width):
    """Rebuild full response from a truncated one, with wrapping"""
    full = np.zeros(width, dtype=trnc.dtype)
    for i, v in enumerate(trnc):
        full[(bin_idx + i) % width] = v
    return full


# ---------------------------------------------------------------------------
# TriangularOverlappingFilterBank
# ---------------------------------------------------------------------------


def test_triangular_basic_properties():
    bank = TriangularOverlappingFilterBank(
        'linear', num_filts=10, low_hz=20., high_hz=8000, sampling_rate=16000)
    assert isinstance(bank, LinearFilterBank)
    assert bank.num_filts == 10
    assert bank.sampling_rate == 16000
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    centers = bank.centers_hz
    assert len(centers) == 10
    delta = (8000 - 20.) / 11
    expected = [20. + delta * (i + 1) for i in range(10)]
    assert centers == pytest.approx(expected, rel=1e-6)
    supports_hz = bank.supports_hz
    assert len(supports_hz) == 10
    for center, (low, high) in zip(centers, supports_hz):
        assert low < center < high
    assert supports_hz[0][0] == pytest.approx(20.)
    assert supports_hz[-1][1] == pytest.approx(8000.)


def test_triangular_analytic_flags():
    bank = TriangularOverlappingFilterBank('mel', num_filts=5, analytic=True)
    assert bank.is_analytic
    assert not bank.is_real
    assert bank.is_zero_phase


def test_triangular_scaling_instance_equals_string():
    bank_a = TriangularOverlappingFilterBank('mel', num_filts=8)
    bank_b = TriangularOverlappingFilterBank(MelScaling(), num_filts=8)
    assert bank_a.centers_hz == pytest.approx(bank_b.centers_hz)
    assert bank_a.supports_hz == pytest.approx(bank_b.supports_hz)


@pytest.mark.parametrize('cls', [
    TriangularOverlappingFilterBank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
])
@pytest.mark.parametrize('kwargs', [
    dict(low_hz=-1., high_hz=8000),
    dict(low_hz=20., high_hz=10.),
    dict(low_hz=20., high_hz=20.),
    dict(low_hz=20., high_hz=9000),
])
def test_invalid_frequency_ranges_raise(cls, kwargs):
    with pytest.raises(ValueError):
        cls('mel', num_filts=5, sampling_rate=16000, **kwargs)


def test_triangular_supports_and_supports_ms():
    bank = TriangularOverlappingFilterBank('mel', num_filts=10)
    supports = bank.supports
    supports_ms = bank.supports_ms
    assert len(supports) == 10
    assert len(supports_ms) == 10
    for (lo, hi), (lo_ms, hi_ms) in zip(supports, supports_ms):
        assert lo < 0 < hi
        assert lo_ms == pytest.approx(lo * 1000 / 16000)
        assert hi_ms == pytest.approx(hi * 1000 / 16000)


def test_triangular_frequency_response_shape_and_peak():
    rate = 16000
    width = 16000  # 1 Hz per bin
    bank = TriangularOverlappingFilterBank(
        'linear', num_filts=10, low_hz=20., high_hz=8000, sampling_rate=rate)
    idx = 3
    resp = bank.get_frequency_response(idx, width)
    assert resp.shape == (width,)
    assert resp.dtype == np.float64
    assert resp.min() >= -1e-12
    assert resp.max() <= 1 + 1e-9
    assert resp.max() > 0.99
    center = bank.centers_hz[idx]
    assert abs(int(np.argmax(resp)) - center * width / rate) <= 1
    # zero outside of support; real bank is conjugate (here even) symmetric
    low, high = bank.supports_hz[idx]
    for k in range(width // 2 + 1):
        hz = k * rate / width
        if hz < low or hz > high:
            assert resp[k] == 0
    assert np.allclose(resp[1:], resp[1:][::-1])


@pytest.mark.parametrize('width', [256, 255])
def test_triangular_half_response_matches_full(width):
    bank = TriangularOverlappingFilterBank('mel', num_filts=10)
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert len(half) == _half_width(width) - (1 if width % 2 else 0) \
            or len(half) == _half_width(width)
        if width % 2:
            assert len(half) == (width + 1) // 2
        else:
            assert len(half) == width // 2 + 1
        assert np.allclose(half, full[:len(half)])


def test_triangular_analytic_frequency_response_positive_only():
    width = 512
    real = TriangularOverlappingFilterBank('mel', num_filts=10)
    analytic = TriangularOverlappingFilterBank(
        'mel', num_filts=10, analytic=True)
    for idx in range(10):
        r = real.get_frequency_response(idx, width)
        a = analytic.get_frequency_response(idx, width)
        assert a.shape == (width,)
        assert np.all(a[width // 2 + 1:] == 0)
        assert np.allclose(a[:width // 2 + 1], r[:width // 2 + 1])


@pytest.mark.parametrize('width', [512, 511])
def test_triangular_truncated_response_real(width):
    bank = TriangularOverlappingFilterBank('mel', num_filts=10)
    half_width = (width + width % 2) // 2 + 1 - width % 2
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert bin_idx >= 0
        assert len(trnc) > 0
        half = np.zeros(half_width, dtype=trnc.dtype)
        half[bin_idx:bin_idx + len(trnc)] = trnc
        expected = bank.get_frequency_response(idx, width, half=True)
        assert np.allclose(half, expected)


def test_triangular_truncated_response_analytic():
    width = 512
    bank = TriangularOverlappingFilterBank(
        'linear', num_filts=10, analytic=True)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        full = np.zeros(width, dtype=trnc.dtype)
        full[bin_idx:bin_idx + len(trnc)] = trnc
        assert np.allclose(full, bank.get_frequency_response(idx, width))


def test_triangular_impulse_response_real_and_analytic():
    width = 64
    real = TriangularOverlappingFilterBank('mel', num_filts=5)
    for idx in range(real.num_filts):
        ir = real.get_impulse_response(idx, width)
        assert ir.shape == (width,)
        assert ir.dtype == np.float64
        # real, even filter
        assert np.allclose(ir[1:], ir[:0:-1])
    analytic = TriangularOverlappingFilterBank(
        'mel', num_filts=5, analytic=True)
    ir = analytic.get_impulse_response(2, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.complex128


# ---------------------------------------------------------------------------
# GaborFilterBank
# ---------------------------------------------------------------------------


def test_gabor_basic_properties():
    bank = GaborFilterBank(
        'linear', num_filts=10, scale_l2_norm=True, erb=True)
    assert isinstance(bank, LinearFilterBank)
    assert bank.num_filts == 10
    assert not bank.is_real
    assert bank.is_zero_phase
    assert bank.scaled_l2_norm
    assert bank.erb
    assert bank.sampling_rate == 16000
    centers = bank.centers_hz
    assert len(centers) == 10
    assert all(b > a for a, b in zip(centers[:-1], centers[1:]))
    assert 20. < centers[0] and centers[-1] < 8000.
    for center, (low, high) in zip(centers, bank.supports_hz):
        assert low < center < high
    for lo, hi in bank.supports:
        assert lo < 0 < hi
        assert lo == -hi
    assert len(bank.supports_ms) == 10


@pytest.mark.parametrize('erb', [False, True])
def test_gabor_frequency_response_peak(erb):
    rate = 16000
    width = 4096
    bank = GaborFilterBank(
        'linear', num_filts=10, sampling_rate=rate, erb=erb)
    for idx in (2, 5, 8):
        resp = bank.get_frequency_response(idx, width)
        assert resp.shape == (width,)
        assert resp.dtype == np.float64
        assert resp.max() == pytest.approx(1., abs=1e-3)
        center = bank.centers_hz[idx]
        assert abs(int(np.argmax(resp)) - center * width / rate) <= 1


@pytest.mark.parametrize('width', [256, 255])
def test_gabor_half_response_matches_full(width):
    bank = GaborFilterBank('mel', num_filts=6)
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        if width % 2:
            assert len(half) == (width + 1) // 2
        else:
            assert len(half) == width // 2 + 1
        assert np.allclose(half, full[:len(half)])


@pytest.mark.parametrize('scale_l2_norm', [False, True])
def test_gabor_impulse_response_is_inverse_of_frequency_response(
        scale_l2_norm):
    width = 512
    bank = GaborFilterBank(
        'linear', num_filts=10, scale_l2_norm=scale_l2_norm)
    for idx in (2, 6):
        ir = bank.get_impulse_response(idx, width)
        assert ir.shape == (width,)
        assert ir.dtype == np.complex128
        resp = bank.get_frequency_response(idx, width)
        assert np.allclose(np.fft.fft(ir), resp, atol=1e-5)


@pytest.mark.parametrize('scale_l2_norm', [False, True])
def test_gabor_truncated_response_reconstructs_full(scale_l2_norm):
    width = 256
    bank = GaborFilterBank(
        'mel', num_filts=20, scale_l2_norm=scale_l2_norm)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        assert len(trnc) <= width
        full = _expand_complex(bin_idx, trnc, width)
        expected = bank.get_frequency_response(idx, width)
        assert np.allclose(full, expected, atol=5 * EPS)


# ---------------------------------------------------------------------------
# ComplexGammatoneFilterBank
# ---------------------------------------------------------------------------


def test_gammatone_basic_properties():
    bank = ComplexGammatoneFilterBank('linear', num_filts=10, order=3)
    assert isinstance(bank, LinearFilterBank)
    assert bank.num_filts == 10
    assert bank.order == 3
    assert not bank.is_real
    assert not bank.is_zero_phase
    assert not bank.scaled_l2_norm
    assert not bank.erb
    centers = bank.centers_hz
    assert len(centers) == 10
    assert all(b > a for a, b in zip(centers[:-1], centers[1:]))
    assert len(bank.supports_hz) == 10
    for lo, hi in bank.supports:
        assert lo == 0  # causal, not max centered
        assert hi > 0


@pytest.mark.parametrize('order', [0, -1, 2.5])
def test_gammatone_invalid_order_raises(order):
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank('linear', num_filts=5, order=order)


def test_gammatone_frequency_response_peak_and_dtype():
    rate = 16000
    width = 4096
    bank = ComplexGammatoneFilterBank('linear', num_filts=10)
    for idx in (2, 5, 8):
        resp = bank.get_frequency_response(idx, width)
        assert resp.shape == (width,)
        assert resp.dtype == np.complex128
        mag = np.abs(resp)
        assert mag.max() == pytest.approx(1., abs=0.02)
        center = bank.centers_hz[idx]
        assert abs(int(np.argmax(mag)) - center * width / rate) <= 1


@pytest.mark.parametrize('width', [128, 127])
def test_gammatone_half_response_matches_full(width):
    bank = ComplexGammatoneFilterBank('linear', num_filts=5)
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        if width % 2:
            assert len(half) == (width + 1) // 2
        else:
            assert len(half) == width // 2 + 1
        assert np.allclose(half, full[:len(half)])


def test_gammatone_truncated_response_reconstructs_full():
    width = 256
    bank = ComplexGammatoneFilterBank('mel', num_filts=40)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        assert len(trnc) <= width
        assert trnc.dtype == np.complex128
        full = _expand_complex(bin_idx, trnc, width)
        expected = bank.get_frequency_response(idx, width)
        assert np.allclose(full, expected, atol=5 * EPS)


def test_gammatone_impulse_response_causality_and_max_centering():
    width = 1024
    causal = ComplexGammatoneFilterBank('linear', num_filts=10)
    ir = causal.get_impulse_response(3, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.complex128
    assert abs(ir[0]) < 1e-12
    assert abs(ir[1]) > 1e-6
    assert int(np.argmax(np.abs(ir))) > 0

    centered = ComplexGammatoneFilterBank(
        'linear', num_filts=10, max_centered=True)
    ir = centered.get_impulse_response(3, width)
    assert int(np.argmax(np.abs(ir))) == 0
    assert centered.supports[3][0] < 0


# ---------------------------------------------------------------------------
# Fbank
# ---------------------------------------------------------------------------


def test_fbank_properties_and_frequency_response():
    rate = 16000
    width = 16000
    bank = Fbank(num_filts=10, sampling_rate=rate)
    assert isinstance(bank, LinearFilterBank)
    assert bank.num_filts == 10
    assert bank.is_real
    assert bank.is_zero_phase
    assert len(bank.centers_hz) == 10
    idx = 4
    resp = bank.get_frequency_response(idx, width)
    assert resp.shape == (width,)
    assert resp.min() >= 0
    assert resp.max() <= 1 + 1e-9
    assert resp.max() > 0.98
    center = bank.centers_hz[idx]
    assert abs(int(np.argmax(resp)) - center * width / rate) <= 2


@pytest.mark.parametrize('width', [512, 511])
def test_fbank_truncated_response_matches_half(width):
    bank = Fbank(num_filts=10)
    half_width = (width + width % 2) // 2 + 1 - width % 2
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        half = np.zeros(half_width, dtype=trnc.dtype)
        half[bin_idx:bin_idx + len(trnc)] = trnc
        expected = bank.get_frequency_response(idx, width, half=True)
        assert np.allclose(half, expected)
