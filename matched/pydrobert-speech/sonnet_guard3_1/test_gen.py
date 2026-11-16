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

RATE = 16000


def _rebuild_wrapped(bin_idx, trnc, width):
    """Rebuild a full-width response from a (possibly wrapping) truncation"""
    assert len(trnc) <= width
    full = np.zeros(width, dtype=trnc.dtype)
    full[(bin_idx + np.arange(len(trnc))) % width] = trnc
    return full


def _half_len(width):
    return (width + 1) // 2 if width % 2 else width // 2 + 1


def _tri(**kwargs):
    params = dict(num_filts=3, low_hz=1000., high_hz=5000.)
    params.update(kwargs)
    return TriangularOverlappingFilterBank('linear', **params)


def _gabor(**kwargs):
    params = dict(num_filts=5, low_hz=100., high_hz=8000.)
    params.update(kwargs)
    return GaborFilterBank('linear', **params)


def _gammatone(**kwargs):
    params = dict(num_filts=6)
    params.update(kwargs)
    return ComplexGammatoneFilterBank('mel', **params)


# ---------------------------------------------------------------- triangular


def test_triangular_basic_properties():
    bank = _tri()
    assert bank.num_filts == 3
    assert bank.sampling_rate == RATE
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert np.allclose(bank.centers_hz, [2000., 3000., 4000.])
    expected = [(1000., 3000.), (2000., 4000.), (3000., 5000.)]
    assert len(bank.supports_hz) == 3
    assert np.allclose(np.array(bank.supports_hz), np.array(expected))
    assert isinstance(bank, LinearFilterBank)


def test_triangular_analytic_flags():
    bank = _tri(analytic=True)
    assert bank.is_analytic
    assert not bank.is_real
    assert bank.is_zero_phase


def test_triangular_invalid_frequency_ranges():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            'linear', num_filts=3, low_hz=-1., high_hz=5000.)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            'linear', num_filts=3, low_hz=1000., high_hz=500.)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            'linear', num_filts=3, low_hz=1000., high_hz=9000.)


def test_triangular_supports_and_supports_ms():
    bank = TriangularOverlappingFilterBank('mel', num_filts=8)
    supports = bank.supports
    supports_ms = bank.supports_ms
    assert len(supports) == len(supports_ms) == 8
    for (start, end), (start_ms, end_ms) in zip(supports, supports_ms):
        assert start < 0 < end
        assert start_ms == pytest.approx(start * 1000 / RATE)
        assert end_ms == pytest.approx(end * 1000 / RATE)


def test_triangular_frequency_response_values():
    bank = _tri()
    # 100Hz per bin with width 160
    res = bank.get_frequency_response(0, 160)
    assert res.shape == (160,)
    assert res.dtype == np.float64
    assert res[15] == pytest.approx(.5, abs=1e-6)
    assert res[20] == pytest.approx(1., abs=1e-6)
    assert res[25] == pytest.approx(.5, abs=1e-6)
    assert res[10] == pytest.approx(0., abs=1e-6)
    assert res[30] == pytest.approx(0., abs=1e-6)
    assert res[5] == 0.
    assert res[40] == 0.
    assert res.min() >= 0.
    assert res.max() <= 1. + 1e-9
    # real filter: Hermitian (even) symmetry
    assert np.allclose(res[1:], res[1:][::-1])


@pytest.mark.parametrize('width', [160, 161])
def test_triangular_frequency_response_half(width):
    bank = _tri()
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert full.shape == (width,)
        assert half.shape == (_half_len(width),)
        assert np.allclose(full[:len(half)], half)
        assert np.allclose(full[1:], full[1:][::-1])


def test_triangular_analytic_frequency_response():
    width = 160
    real_bank = _tri()
    analytic_bank = _tri(analytic=True)
    for idx in range(3):
        real_half = real_bank.get_frequency_response(idx, width, half=True)
        full = analytic_bank.get_frequency_response(idx, width)
        assert full.shape == (width,)
        assert np.allclose(full[:len(real_half)], real_half)
        # negative frequencies are suppressed
        assert np.all(full[width // 2 + 1:] == 0)
        assert full.max() > 0.9


@pytest.mark.parametrize('width', [160, 161, 400])
def test_triangular_truncated_response_matches_half(width):
    bank = _tri()
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        rebuilt = np.zeros(len(half), dtype=trnc.dtype)
        rebuilt[bin_idx:bin_idx + len(trnc)] = trnc
        assert np.allclose(rebuilt, half)
        assert bin_idx >= 0
        assert trnc.max() > 0.


def test_triangular_truncated_response_matches_full_real():
    # check the documented full-spectrum recovery for a real filter
    width = 160
    bank = _tri()
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        full = np.zeros(width, dtype=trnc.dtype)
        full[bin_idx:bin_idx + len(trnc)] = trnc
        full[width - bin_idx - len(trnc) + 1:width - bin_idx + 1] = \
            trnc[:None if bin_idx else 0:-1].conj()
        expected = bank.get_frequency_response(idx, width)
        assert np.allclose(full, expected)


def test_triangular_truncated_response_analytic():
    width = 200
    bank = _tri(analytic=True)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        full = bank.get_frequency_response(idx, width)
        assert np.allclose(_rebuild_wrapped(bin_idx, trnc, width), full)


def test_triangular_impulse_response_shape_dtype_symmetry():
    real_bank = _tri()
    analytic_bank = _tri(analytic=True)
    res = real_bank.get_impulse_response(1, 64)
    assert res.shape == (64,)
    assert res.dtype == np.float64
    # real, zero-phase filter => even impulse response
    assert np.allclose(res[1:], res[1:][::-1])
    ares = analytic_bank.get_impulse_response(1, 64)
    assert ares.shape == (64,)
    assert ares.dtype == np.complex128


# -------------------------------------------------------------------- fbank


def test_fbank_properties_and_responses():
    bank = Fbank(num_filts=6, high_hz=6000.)
    assert bank.num_filts == 6
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert len(bank.centers_hz) == 6
    assert all(
        lo < c < hi
        for c, (lo, hi) in zip(bank.centers_hz, bank.supports_hz))
    width = 256
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert full.shape == (width,)
        assert half.shape == (_half_len(width),)
        assert np.allclose(full[:len(half)], half)
        assert np.allclose(full[1:], full[1:][::-1])
        assert full.min() >= 0.
        assert full.max() <= 1. + 1e-9
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        rebuilt = np.zeros(len(half), dtype=trnc.dtype)
        rebuilt[bin_idx:bin_idx + len(trnc)] = trnc
        assert np.allclose(rebuilt, half)


# -------------------------------------------------------------------- gabor


def test_gabor_properties():
    bank = _gabor()
    assert bank.num_filts == 5
    assert not bank.is_real
    assert bank.is_zero_phase
    assert bank.sampling_rate == RATE
    assert not bank.erb
    assert not bank.scaled_l2_norm
    centers = bank.centers_hz
    assert len(centers) == 5
    assert all(a < b for a, b in zip(centers[:-1], centers[1:]))
    for center, (low, high), (start, end), (ms_a, ms_b) in zip(
            centers, bank.supports_hz, bank.supports, bank.supports_ms):
        assert low < center < high
        assert start == -end
        assert end > 0
        assert ms_a == pytest.approx(start * 1000 / RATE)
        assert ms_b == pytest.approx(end * 1000 / RATE)
    expected_analytic = all(low >= 0 for low, _ in bank.supports_hz)
    assert bank.is_analytic == expected_analytic


def test_gabor_flags_and_invalid_range():
    bank = _gabor(erb=True, scale_l2_norm=True)
    assert bank.erb
    assert bank.scaled_l2_norm
    plain = _gabor()
    # an ERB-based bandwidth gives different support than 3dB
    assert bank.supports != plain.supports
    with pytest.raises(ValueError):
        GaborFilterBank('linear', num_filts=3, low_hz=-5., high_hz=5000.)
    with pytest.raises(ValueError):
        GaborFilterBank('linear', num_filts=3, low_hz=3000., high_hz=2000.)
    with pytest.raises(ValueError):
        GaborFilterBank('linear', num_filts=3, low_hz=100., high_hz=9000.)


def test_gabor_frequency_response_peak_and_half():
    bank = _gabor()
    width = 512
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert full.shape == (width,)
        assert half.shape == (width // 2 + 1,)
        assert np.allclose(full[:len(half)], half)
        assert full.min() >= 0.
        assert full.max() <= 1. + 1e-3
        assert full.max() > 0.95
        # maximum is at the bin closest to the center frequency
        center_bin = bank.centers_hz[idx] / RATE * width
        assert abs(int(np.argmax(full)) - center_bin) <= 1.
    odd_half = bank.get_frequency_response(2, 513, half=True)
    assert odd_half.shape == (257,)


@pytest.mark.parametrize('width', [512, 511])
def test_gabor_truncated_response_matches_full(width):
    bank = _gabor()
    atol = 5 * config.EFFECTIVE_SUPPORT_THRESHOLD
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        full = bank.get_frequency_response(idx, width)
        assert np.allclose(
            _rebuild_wrapped(bin_idx, trnc, width), full, atol=atol)


@pytest.mark.parametrize('scale_l2_norm', [False, True])
def test_gabor_impulse_response_matches_inverse_dft(scale_l2_norm):
    bank = _gabor(scale_l2_norm=scale_l2_norm)
    width = 128
    for idx in range(bank.num_filts):
        imp = bank.get_impulse_response(idx, width)
        freq = bank.get_frequency_response(idx, width)
        assert imp.shape == (width,)
        assert imp.dtype == np.complex128
        assert np.allclose(imp, np.fft.ifft(freq), atol=1e-8)


def test_gabor_l2_normalized_impulse_response_has_unit_norm():
    bank = _gabor(scale_l2_norm=True)
    for idx in range(bank.num_filts):
        imp = bank.get_impulse_response(idx, 256)
        assert np.sum(np.abs(imp) ** 2) == pytest.approx(1., abs=1e-6)


# --------------------------------------------------------------- gammatone


def test_gammatone_properties_and_errors():
    bank = _gammatone()
    assert bank.num_filts == 6
    assert bank.order == 4
    assert not bank.is_real
    assert not bank.is_zero_phase
    assert not bank.erb
    assert not bank.scaled_l2_norm
    assert bank.sampling_rate == RATE
    centers = bank.centers_hz
    assert all(a < b for a, b in zip(centers[:-1], centers[1:]))
    assert len(bank.supports) == len(bank.supports_hz) == 6
    for center, (low, high) in zip(centers, bank.supports_hz):
        assert low < center < high
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank('mel', num_filts=3, order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank('mel', num_filts=3, order=2.5)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank('mel', num_filts=3, low_hz=-1., high_hz=4000.)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank('mel', num_filts=3, high_hz=9000.)


@pytest.mark.parametrize('erb', [False, True])
def test_gammatone_frequency_response_peak_and_half(erb):
    bank = _gammatone(erb=erb)
    width = 2048
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert full.shape == (width,)
        assert full.dtype == np.complex128
        assert half.shape == (width // 2 + 1,)
        assert np.allclose(full[:len(half)], half)
        # unnormalized gammatone has unit gain at the carrier frequency
        assert np.abs(full).max() == pytest.approx(1., abs=0.02)
        center_bin = bank.centers_hz[idx] / RATE * width
        assert abs(int(np.argmax(np.abs(full))) - center_bin) <= 2.
    assert bank.get_frequency_response(0, 1025, half=True).shape == (513,)


@pytest.mark.parametrize('width', [512, 511])
def test_gammatone_truncated_response_matches_full(width):
    bank = _gammatone()
    atol = 5 * config.EFFECTIVE_SUPPORT_THRESHOLD
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        assert trnc.dtype == np.complex128
        full = bank.get_frequency_response(idx, width)
        assert np.allclose(
            _rebuild_wrapped(bin_idx, trnc, width), full, atol=atol)


def test_gammatone_impulse_response_is_causal():
    bank = _gammatone()
    width = 4096
    for idx in range(bank.num_filts):
        res = bank.get_impulse_response(idx, width)
        assert res.shape == (width,)
        assert res.dtype == np.complex128
        assert abs(res[0]) < 1e-8
        peak = int(np.argmax(np.abs(res)))
        assert 0 < peak <= bank.supports[idx][1]
        assert bank.supports[idx][0] == 0


def test_gammatone_max_centered_impulse_response_peaks_at_zero():
    bank = _gammatone(max_centered=True)
    width = 4096
    for idx in range(bank.num_filts):
        assert bank.supports[idx][0] < 0
        res = bank.get_impulse_response(idx, width)
        assert int(np.argmax(np.abs(res))) == 0


# ----------------------------------------------------------------- factory


def test_linear_filter_bank_from_alias():
    gabor = LinearFilterBank.from_alias('gabor', 'mel', num_filts=3)
    assert isinstance(gabor, GaborFilterBank)
    assert gabor.num_filts == 3
    tri = LinearFilterBank.from_alias('tri', 'mel', num_filts=4)
    assert isinstance(tri, TriangularOverlappingFilterBank)
    assert tri.num_filts == 4
    tone = LinearFilterBank.from_alias('gammatone', 'mel', num_filts=2)
    assert isinstance(tone, ComplexGammatoneFilterBank)
    assert tone.num_filts == 2
