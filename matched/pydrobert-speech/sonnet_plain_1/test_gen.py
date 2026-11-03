"""Unit tests for the linear filter banks in pydrobert.speech.filters"""

import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.filters import ComplexGammatoneFilterBank
from pydrobert.speech.filters import Fbank
from pydrobert.speech.filters import GaborFilterBank
from pydrobert.speech.filters import LinearFilterBank
from pydrobert.speech.filters import TriangularOverlappingFilterBank
from pydrobert.speech.scales import MelScaling

RATE = 8000


def _tri(**kwargs):
    kwargs.setdefault('num_filts', 10)
    kwargs.setdefault('sampling_rate', RATE)
    return TriangularOverlappingFilterBank(MelScaling(), **kwargs)


def _fbank(**kwargs):
    kwargs.setdefault('num_filts', 10)
    kwargs.setdefault('sampling_rate', RATE)
    return Fbank(**kwargs)


def _gabor(**kwargs):
    kwargs.setdefault('num_filts', 6)
    kwargs.setdefault('sampling_rate', RATE)
    return GaborFilterBank(MelScaling(), **kwargs)


def _gammatone(**kwargs):
    kwargs.setdefault('num_filts', 5)
    kwargs.setdefault('sampling_rate', RATE)
    return ComplexGammatoneFilterBank(MelScaling(), **kwargs)


def _half_len(width):
    return width // 2 + 1 if width % 2 == 0 else (width + 1) // 2


# ---------------------------------------------------------------------------
# TriangularOverlappingFilterBank
# ---------------------------------------------------------------------------


def test_triangular_basic_properties():
    bank = _tri()
    assert bank.num_filts == 10
    assert bank.sampling_rate == RATE
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    centers = bank.centers_hz
    assert len(centers) == 10
    assert all(a < b for a, b in zip(centers, centers[1:]))
    assert len(bank.supports_hz) == 10
    for (low, high), center in zip(bank.supports_hz, centers):
        assert low < center < high
    assert len(bank.supports) == 10
    for low, high in bank.supports:
        assert low < 0 < high
    for (s0, s1), (m0, m1) in zip(bank.supports, bank.supports_ms):
        assert m0 == pytest.approx(s0 * 1000 / RATE)
        assert m1 == pytest.approx(s1 * 1000 / RATE)


def test_triangular_invalid_frequency_range_raises():
    with pytest.raises(ValueError):
        _tri(low_hz=-1., high_hz=4000.)
    with pytest.raises(ValueError):
        _tri(low_hz=100., high_hz=100.)
    with pytest.raises(ValueError):
        _tri(low_hz=200., high_hz=100.)
    with pytest.raises(ValueError):
        _tri(low_hz=20., high_hz=RATE // 2 + 1)


def test_triangular_full_frequency_response_is_symmetric_and_bounded():
    bank = _tri()
    width = 128
    for idx in range(bank.num_filts):
        resp = bank.get_frequency_response(idx, width)
        assert resp.shape == (width,)
        assert resp.dtype == np.float64
        assert resp.min() >= 0
        assert resp.max() <= 1 + 1e-9
        assert (resp > 0).any()
        # conjugate symmetry of a real zero-phase filter
        assert np.allclose(resp[1:], resp[:0:-1])


def test_triangular_half_frequency_response_matches_full():
    bank = _tri()
    for width in (64, 65, 100):
        for idx in range(bank.num_filts):
            full = bank.get_frequency_response(idx, width)
            half = bank.get_frequency_response(idx, width, half=True)
            assert len(half) == _half_len(width)
            assert np.allclose(half, full[:len(half)])


def test_triangular_frequency_response_peaks_at_center():
    bank = _tri()
    width = 1600  # 5Hz bins
    bin_hz = RATE / width
    for idx in range(bank.num_filts):
        resp = bank.get_frequency_response(idx, width)
        peak_hz = np.argmax(resp) * bin_hz
        assert abs(peak_hz - bank.centers_hz[idx]) <= bin_hz + 1e-6
        assert 0.9 < resp.max() <= 1 + 1e-9


def test_triangular_truncated_response_matches_half_spectrum():
    bank = _tri()
    for width in (64, 65, 200, 257):
        for idx in range(bank.num_filts):
            bin_idx, trunc = bank.get_truncated_response(idx, width)
            half = bank.get_frequency_response(idx, width, half=True)
            assert isinstance(bin_idx, int)
            assert trunc.dtype == np.float64
            assert bin_idx >= 0
            assert bin_idx + len(trunc) <= len(half)
            rebuilt = np.zeros(len(half), dtype=trunc.dtype)
            rebuilt[bin_idx:bin_idx + len(trunc)] = trunc
            assert np.allclose(rebuilt, half)


def test_triangular_analytic_frequency_response_has_no_negative_freqs():
    bank = _tri(analytic=True)
    assert bank.is_analytic
    assert not bank.is_real
    width = 128
    real_bank = _tri()
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert len(half) == width // 2 + 1
        assert np.all(full[width // 2 + 1:] == 0)
        assert np.allclose(half, full[:len(half)])
        real_full = real_bank.get_frequency_response(idx, width)
        assert np.allclose(full[:len(half)], real_full[:len(half)])


def test_triangular_impulse_response_dtype_and_symmetry():
    width = 64
    real_bank = _tri()
    ir = real_bank.get_impulse_response(3, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.float64
    assert np.allclose(ir[1:], ir[:0:-1])
    analytic_bank = _tri(analytic=True)
    air = analytic_bank.get_impulse_response(3, width)
    assert air.shape == (width,)
    assert air.dtype == np.complex128


def test_triangular_scaling_function_arguments_and_aliases():
    from_alias = LinearFilterBank.from_alias(
        'tri', 'mel', num_filts=10, sampling_rate=RATE)
    assert isinstance(from_alias, TriangularOverlappingFilterBank)
    assert np.allclose(from_alias.centers_hz, _tri().centers_hz)
    # default high_hz is the Nyquist frequency
    assert from_alias.supports_hz[-1][1] == pytest.approx(RATE // 2)
    linear = TriangularOverlappingFilterBank(
        {'alias': 'linear', 'low_hz': 0}, num_filts=5, sampling_rate=RATE,
        low_hz=100., high_hz=3900.)
    diffs = np.diff(np.array(linear.centers_hz))
    assert np.allclose(diffs, diffs[0])
    assert linear.centers_hz[0] == pytest.approx(100. + 3800. / 6)


# ---------------------------------------------------------------------------
# Fbank
# ---------------------------------------------------------------------------


def test_fbank_properties_and_validation():
    bank = LinearFilterBank.from_alias(
        'fbank', num_filts=10, sampling_rate=RATE)
    assert isinstance(bank, Fbank)
    assert bank.num_filts == 10
    assert bank.sampling_rate == RATE
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert len(bank.centers_hz) == 10
    for (low, high), center in zip(bank.supports_hz, bank.centers_hz):
        assert low < center < high
    for low, high in bank.supports:
        assert low < 0 < high
    analytic = _fbank(analytic=True)
    assert analytic.is_analytic
    assert not analytic.is_real
    with pytest.raises(ValueError):
        _fbank(low_hz=-1., high_hz=4000.)
    with pytest.raises(ValueError):
        _fbank(low_hz=500., high_hz=100.)
    with pytest.raises(ValueError):
        _fbank(high_hz=RATE // 2 + 10)


def test_fbank_frequency_response_shape_symmetry_and_peak():
    bank = _fbank()
    width = 1600
    bin_hz = RATE / width
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        assert full.shape == (width,)
        assert full.dtype == np.float64
        assert np.nanmin(full) >= 0
        assert np.nanmax(full) <= 1 + 1e-9
        assert np.allclose(full[1:], full[:0:-1], equal_nan=True)
        peak_hz = np.argmax(full) * bin_hz
        assert abs(peak_hz - bank.centers_hz[idx]) <= bin_hz + 1e-6
        assert full.max() > 0.9
    for width in (64, 65):
        full = bank.get_frequency_response(2, width)
        half = bank.get_frequency_response(2, width, half=True)
        assert len(half) == _half_len(width)
        assert np.allclose(half, full[:len(half)], equal_nan=True)


def test_fbank_truncated_response_matches_half_spectrum():
    bank = _fbank()
    for width in (64, 65, 200, 257):
        for idx in range(bank.num_filts):
            bin_idx, trunc = bank.get_truncated_response(idx, width)
            half = bank.get_frequency_response(idx, width, half=True)
            assert bin_idx >= 0
            assert bin_idx + len(trunc) <= len(half)
            rebuilt = np.zeros(len(half), dtype=trunc.dtype)
            rebuilt[bin_idx:bin_idx + len(trunc)] = trunc
            assert np.allclose(rebuilt, half, equal_nan=True)


def test_fbank_impulse_response_is_inverse_of_frequency_response():
    width = 128
    real_bank = _fbank()
    ir = real_bank.get_impulse_response(4, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.float64
    half = real_bank.get_frequency_response(4, width, half=True)
    assert np.allclose(np.fft.rfft(ir), half, atol=1e-9)
    analytic_bank = _fbank(analytic=True)
    air = analytic_bank.get_impulse_response(4, width)
    assert air.dtype == np.complex128
    full = analytic_bank.get_frequency_response(4, width)
    assert np.allclose(np.fft.fft(air), full, atol=1e-9)


# ---------------------------------------------------------------------------
# GaborFilterBank
# ---------------------------------------------------------------------------


def test_gabor_properties_and_validation():
    bank = _gabor()
    assert bank.num_filts == 6
    assert bank.sampling_rate == RATE
    assert not bank.is_real
    assert bank.is_zero_phase
    assert not bank.scaled_l2_norm
    assert not bank.erb
    centers = bank.centers_hz
    assert len(centers) == 6
    assert all(a < b for a, b in zip(centers, centers[1:]))
    for (low, high), center in zip(bank.supports_hz, centers):
        assert low < center < high
    for low, high in bank.supports:
        assert high > 0
        assert low == -high
    assert bank.is_analytic == all(low >= 0 for low, _ in bank.supports_hz)
    flagged = _gabor(scale_l2_norm=True, erb=True)
    assert flagged.scaled_l2_norm
    assert flagged.erb
    with pytest.raises(ValueError):
        _gabor(low_hz=-5., high_hz=4000.)
    with pytest.raises(ValueError):
        _gabor(low_hz=300., high_hz=200.)
    with pytest.raises(ValueError):
        _gabor(high_hz=RATE)


def test_gabor_frequency_response_shape_and_half_consistency():
    bank = _gabor()
    for width in (64, 65, 128):
        for idx in range(bank.num_filts):
            full = bank.get_frequency_response(idx, width)
            half = bank.get_frequency_response(idx, width, half=True)
            assert full.shape == (width,)
            assert full.dtype == np.float64
            assert full.min() >= 0
            assert len(half) == _half_len(width)
            assert np.allclose(half, full[:len(half)])


def test_gabor_frequency_response_peaks_at_center_with_unit_gain():
    bank = _gabor()
    width = 1024
    bin_hz = RATE / width
    for idx in range(bank.num_filts):
        resp = bank.get_frequency_response(idx, width)
        peak_hz = np.argmax(resp) * bin_hz
        assert abs(peak_hz - bank.centers_hz[idx]) <= bin_hz + 1e-6
        assert 0.9 < resp.max() <= 1 + 1e-6


def test_gabor_truncated_response_agrees_with_full_response():
    bank = _gabor()
    width = 512
    eps = config.EFFECTIVE_SUPPORT_THRESHOLD
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        bin_idx, trunc = bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        idxs = (bin_idx + np.arange(len(trunc))) % width
        assert np.allclose(full[idxs], trunc, atol=2e-3)
        mask = np.ones(width, dtype=bool)
        mask[idxs] = False
        if mask.any():
            assert full[mask].max() <= 4 * eps


def test_gabor_impulse_response_transforms_to_frequency_response():
    bank = _gabor()
    width = 1024
    idx = 3
    ir = bank.get_impulse_response(idx, width)
    assert ir.shape == (width,)
    assert ir.dtype == np.complex128
    spectrum = np.fft.fft(ir)
    resp = bank.get_frequency_response(idx, width)
    assert np.allclose(spectrum.real, resp, atol=1e-6)
    assert np.allclose(spectrum.imag, 0, atol=1e-6)


# ---------------------------------------------------------------------------
# ComplexGammatoneFilterBank
# ---------------------------------------------------------------------------


def test_gammatone_properties_and_validation():
    bank = LinearFilterBank.from_alias(
        'gammatone', 'mel', num_filts=5, sampling_rate=RATE)
    assert isinstance(bank, ComplexGammatoneFilterBank)
    assert bank.num_filts == 5
    assert bank.order == 4
    assert bank.sampling_rate == RATE
    assert not bank.is_real
    assert not bank.is_zero_phase
    assert not bank.scaled_l2_norm
    assert not bank.erb
    centers = bank.centers_hz
    assert len(centers) == 5
    assert all(a < b for a, b in zip(centers, centers[1:]))
    for (low, high), center in zip(bank.supports_hz, centers):
        assert low < center < high
    assert bank.is_analytic == all(low >= 0 for low, _ in bank.supports_hz)
    with pytest.raises(ValueError):
        _gammatone(order=0)
    with pytest.raises(ValueError):
        _gammatone(order=-2)
    with pytest.raises(ValueError):
        _gammatone(order=2.5)
    with pytest.raises(ValueError):
        _gammatone(low_hz=-1., high_hz=4000.)
    with pytest.raises(ValueError):
        _gammatone(high_hz=RATE)


def test_gammatone_max_centered_shifts_support_to_negative_time():
    causal = _gammatone()
    centered = _gammatone(max_centered=True)
    for (c_low, c_high), (m_low, m_high) in zip(
            causal.supports, centered.supports):
        assert c_low == 0
        assert c_high > 0
        assert m_low < 0
        assert m_low < m_high


def test_gammatone_frequency_response_dtype_peak_and_half():
    bank = _gammatone()
    width = 512
    bin_hz = RATE / width
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        assert full.shape == (width,)
        assert full.dtype == np.complex128
        mag = np.abs(full)
        peak_hz = np.argmax(mag) * bin_hz
        assert abs(peak_hz - bank.centers_hz[idx]) <= 2 * bin_hz + 1e-6
        assert 0.95 < mag.max() < 1.05
    for width in (64, 65):
        full = bank.get_frequency_response(1, width)
        half = bank.get_frequency_response(1, width, half=True)
        assert len(half) == _half_len(width)
        assert np.allclose(half, full[:len(half)])


def test_gammatone_truncated_response_agrees_with_full_response():
    bank = _gammatone()
    width = 256
    eps = config.EFFECTIVE_SUPPORT_THRESHOLD
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        bin_idx, trunc = bank.get_truncated_response(idx, width)
        assert trunc.dtype == np.complex128
        assert 0 <= bin_idx < width
        idxs = (bin_idx + np.arange(len(trunc))) % width
        assert np.allclose(full[idxs], trunc, atol=5e-3)
        mask = np.ones(width, dtype=bool)
        mask[idxs] = False
        if mask.any():
            assert np.abs(full[mask]).max() <= 10 * eps
