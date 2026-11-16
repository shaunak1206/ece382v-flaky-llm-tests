import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.filters import (
    ComplexGammatoneFilterBank,
    Fbank,
    GaborFilterBank,
    LinearFilterBank,
    TriangularOverlappingFilterBank,
)
from pydrobert.speech.scales import MelScaling

EPS = config.EFFECTIVE_SUPPORT_THRESHOLD


def _tri(**kwargs):
    kwargs.setdefault("num_filts", 10)
    kwargs.setdefault("low_hz", 100.0)
    kwargs.setdefault("high_hz", 7000.0)
    return TriangularOverlappingFilterBank(MelScaling(), **kwargs)


def _fbank(**kwargs):
    kwargs.setdefault("num_filts", 10)
    kwargs.setdefault("low_hz", 100.0)
    kwargs.setdefault("high_hz", 7000.0)
    return Fbank(**kwargs)


def _gabor(**kwargs):
    kwargs.setdefault("num_filts", 12)
    kwargs.setdefault("low_hz", 100.0)
    kwargs.setdefault("high_hz", 7000.0)
    return GaborFilterBank(MelScaling(), **kwargs)


def _gammatone(**kwargs):
    kwargs.setdefault("num_filts", 12)
    kwargs.setdefault("low_hz", 100.0)
    kwargs.setdefault("high_hz", 7000.0)
    return ComplexGammatoneFilterBank(MelScaling(), **kwargs)


def _half_size(width):
    return (width + 1) // 2 if width % 2 else width // 2 + 1


def _reconstruct_wrapped(bin_idx, trnc, width):
    """Rebuild full spectrum from a (possibly wrapping) truncated response"""
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    return full


def test_triangular_properties():
    bank = _tri(num_filts=10)
    assert bank.num_filts == 10
    assert bank.sampling_rate == 16000
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert len(bank.centers_hz) == 10
    assert len(bank.supports_hz) == 10
    assert len(bank.supports) == 10
    assert list(bank.centers_hz) == sorted(bank.centers_hz)
    for center, (low, high) in zip(bank.centers_hz, bank.supports_hz):
        assert low < center < high
    for (start, end), (ms_start, ms_end) in zip(
            bank.supports, bank.supports_ms):
        assert start < 0 < end
        assert ms_start == pytest.approx(start * 1000 / 16000)
        assert ms_end == pytest.approx(end * 1000 / 16000)
    analytic = _tri(analytic=True)
    assert analytic.is_analytic
    assert not analytic.is_real


@pytest.mark.parametrize("factory", [_tri, _fbank, _gabor, _gammatone])
@pytest.mark.parametrize(
    "low_hz,high_hz", [(-1.0, 4000.0), (4000.0, 4000.0), (5000.0, 3000.0),
                       (20.0, 9000.0)])
def test_invalid_frequency_ranges(factory, low_hz, high_hz):
    with pytest.raises(ValueError):
        factory(low_hz=low_hz, high_hz=high_hz)


def test_triangular_frequency_response_shape_and_symmetry():
    width = 128
    bank = _tri(num_filts=8)
    for idx in range(bank.num_filts):
        resp = bank.get_frequency_response(idx, width)
        assert resp.shape == (width,)
        assert resp.dtype == np.float64
        assert resp.min() >= 0
        assert resp.max() <= 1 + 1e-12
        assert resp.max() > 0
        assert resp[0] == 0
        # real filter: Hermitian (here even) symmetry
        assert np.allclose(resp[1:], resp[:0:-1])


def test_triangular_peak_near_center():
    width = 8000
    bank = _tri(num_filts=10)
    resp = bank.get_frequency_response(0, width, half=True)
    center_bin = bank.centers_hz[0] * width / bank.sampling_rate
    assert abs(int(np.argmax(resp)) - center_bin) <= 1
    assert 0.95 < resp.max() <= 1 + 1e-12


@pytest.mark.parametrize("width", [100, 101, 128, 257])
def test_triangular_half_response_is_prefix_of_full(width):
    bank = _tri(num_filts=6)
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert len(half) == _half_size(width)
        assert np.allclose(half, full[:len(half)])


def test_triangular_matches_triangle_formula():
    width = 1000
    bank = _tri(num_filts=7)
    rate = bank.sampling_rate
    for idx in range(bank.num_filts):
        left, right = bank.supports_hz[idx]
        mid = bank.centers_hz[idx]
        half = bank.get_frequency_response(idx, width, half=True)
        for k in range(len(half)):
            hz = rate * k / width
            expected = max(
                0.0, min((hz - left) / (mid - left),
                         (right - hz) / (right - mid)))
            assert half[k] == pytest.approx(expected, abs=1e-9)


def test_triangular_analytic_suppresses_negative_frequencies():
    width = 128
    bank = _tri(num_filts=6, analytic=True)
    for idx in range(bank.num_filts):
        resp = bank.get_frequency_response(idx, width)
        assert resp.shape == (width,)
        assert np.all(resp[width // 2 + 1:] == 0)
        assert resp[:width // 2 + 1].max() > 0


@pytest.mark.parametrize("width", [127, 128, 400])
def test_triangular_truncated_response_matches_half(width):
    bank = _tri(num_filts=6)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        half_ref = bank.get_frequency_response(idx, width, half=True)
        assert bin_idx >= 0
        assert bin_idx + len(trnc) <= len(half_ref)
        half = np.zeros(len(half_ref), dtype=trnc.dtype)
        half[bin_idx:bin_idx + len(trnc)] = trnc
        assert np.allclose(half, half_ref)


def test_fbank_properties():
    bank = _fbank(num_filts=9)
    assert bank.num_filts == 9
    assert bank.is_real
    assert bank.is_zero_phase
    assert not bank.is_analytic
    assert bank.sampling_rate == 16000
    assert len(bank.centers_hz) == 9
    assert len(bank.supports) == 9
    for center, (low, high) in zip(bank.centers_hz, bank.supports_hz):
        assert low < center < high
    assert _fbank(analytic=True).is_analytic


def test_fbank_squared_response_is_triangular_in_mel():
    width = 1000
    scaling = MelScaling()
    bank = _fbank(num_filts=7)
    rate = bank.sampling_rate
    for idx in range(bank.num_filts):
        left, right = bank.supports_hz[idx]
        mid = bank.centers_hz[idx]
        left_mel = scaling.hertz_to_scale(left)
        mid_mel = scaling.hertz_to_scale(mid)
        right_mel = scaling.hertz_to_scale(right)
        half = bank.get_frequency_response(idx, width, half=True)
        assert half.min() >= 0
        for k in range(len(half)):
            hz = rate * k / width
            if hz <= left or hz >= right:
                assert half[k] ** 2 == pytest.approx(0.0, abs=1e-6)
                continue
            mel = scaling.hertz_to_scale(hz)
            expected = max(0.0, min((mel - left_mel) / (mid_mel - left_mel),
                                    (right_mel - mel) / (right_mel - mid_mel)))
            assert half[k] ** 2 == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("width", [255, 256, 512])
def test_fbank_truncated_response_matches_half(width):
    bank = _fbank(num_filts=6)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        half_ref = bank.get_frequency_response(idx, width, half=True)
        half = np.zeros(len(half_ref), dtype=trnc.dtype)
        half[bin_idx:bin_idx + len(trnc)] = trnc
        assert np.allclose(half, half_ref)


def test_fbank_full_response_symmetric_and_analytic():
    width = 256
    real_bank = _fbank(num_filts=5)
    analytic_bank = _fbank(num_filts=5, analytic=True)
    for idx in range(5):
        real = real_bank.get_frequency_response(idx, width)
        assert real.shape == (width,)
        assert np.allclose(real[1:], real[:0:-1])
        ana = analytic_bank.get_frequency_response(idx, width)
        assert np.all(ana[width // 2 + 1:] == 0)
        assert np.allclose(ana[:width // 2 + 1], real[:width // 2 + 1])


def test_gabor_properties():
    bank = _gabor(num_filts=12, scale_l2_norm=True, erb=True)
    assert bank.num_filts == 12
    assert not bank.is_real
    assert bank.is_zero_phase
    assert bank.scaled_l2_norm
    assert bank.erb
    assert isinstance(bank.is_analytic, bool)
    assert len(bank.centers_hz) == 12
    assert list(bank.centers_hz) == sorted(bank.centers_hz)
    for center, (low, high) in zip(bank.centers_hz, bank.supports_hz):
        assert low < center < high
    for start, end in bank.supports:
        assert start < 0 < end
    default = _gabor()
    assert not default.scaled_l2_norm
    assert not default.erb


def test_gabor_frequency_response_peak_at_center():
    width = 8000
    bank = _gabor(num_filts=10)
    for idx in range(bank.num_filts):
        resp = bank.get_frequency_response(idx, width)
        assert resp.shape == (width,)
        assert resp.dtype == np.float64
        assert resp.min() >= 0
        center_bin = bank.centers_hz[idx] * width / bank.sampling_rate
        assert abs(int(np.argmax(resp)) - center_bin) <= 1
        assert 0.9 < resp.max() <= 1 + 1e-6


@pytest.mark.parametrize("width", [64, 99])
def test_gabor_half_response_is_prefix_of_full(width):
    bank = _gabor(num_filts=5)
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert len(half) == _half_size(width)
        assert np.allclose(half, full[:len(half)])


def test_gabor_truncated_response_matches_full():
    width = 512
    bank = _gabor(num_filts=12)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        full_ref = bank.get_frequency_response(idx, width)
        full = _reconstruct_wrapped(bin_idx, trnc, width)
        assert np.allclose(full, full_ref, atol=2 * EPS + 1e-9)


def test_gammatone_properties_and_invalid_order():
    bank = _gammatone(num_filts=8, order=3, scale_l2_norm=True, erb=True)
    assert bank.num_filts == 8
    assert bank.order == 3
    assert not bank.is_real
    assert not bank.is_zero_phase
    assert bank.scaled_l2_norm
    assert bank.erb
    assert len(bank.centers_hz) == 8
    assert len(bank.supports) == 8
    for center, (low, high) in zip(bank.centers_hz, bank.supports_hz):
        assert low < center < high
    for order in (0, -2, 2.5):
        with pytest.raises(ValueError):
            _gammatone(order=order)


def test_gammatone_frequency_response_peak():
    width = 8000
    bank = _gammatone(num_filts=40, low_hz=20.0, high_hz=None)
    resp = bank.get_frequency_response(0, width)
    assert resp.shape == (width,)
    assert resp.dtype == np.complex128
    mag = np.abs(resp)
    center_bin = bank.centers_hz[0] * width / bank.sampling_rate
    assert abs(int(np.argmax(mag)) - center_bin) <= 1.5
    assert 0.95 < mag.max() < 1.05


@pytest.mark.parametrize("width", [64, 101])
def test_gammatone_half_response_is_prefix_of_full(width):
    bank = _gammatone(num_filts=5)
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert half.dtype == np.complex128
        assert len(half) == _half_size(width)
        assert np.allclose(half, full[:len(half)])


def test_gammatone_max_centered_changes_phase_only():
    width = 128
    normal = _gammatone(num_filts=5, max_centered=False)
    centered = _gammatone(num_filts=5, max_centered=True)
    for idx in range(5):
        a = normal.get_frequency_response(idx, width)
        b = centered.get_frequency_response(idx, width)
        assert np.allclose(np.abs(a), np.abs(b), atol=1e-9)


def test_gammatone_truncated_response_matches_full():
    width = 512
    bank = _gammatone(num_filts=12)
    saw_truncation = False
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        if len(trnc) < width:
            saw_truncation = True
        full_ref = bank.get_frequency_response(idx, width)
        full = _reconstruct_wrapped(bin_idx, trnc, width)
        assert np.allclose(full, full_ref, atol=5 * EPS + 1e-9)
    assert saw_truncation


def test_filter_banks_are_linear_filter_banks_and_have_aliases():
    for bank in (_tri(), _fbank(), _gabor(), _gammatone()):
        assert isinstance(bank, LinearFilterBank)
    bank = LinearFilterBank.from_alias(
        "tri", scaling_function="mel", num_filts=5)
    assert isinstance(bank, TriangularOverlappingFilterBank)
    assert bank.num_filts == 5
    bank = LinearFilterBank.from_alias(
        "gabor", scaling_function="mel", num_filts=4)
    assert isinstance(bank, GaborFilterBank)
    assert bank.num_filts == 4
