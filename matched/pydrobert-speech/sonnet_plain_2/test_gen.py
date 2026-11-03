"""Unit tests for linear filter banks in pydrobert.speech.filters"""

import numpy as np
import pytest

from pydrobert.speech.filters import ComplexGammatoneFilterBank
from pydrobert.speech.filters import Fbank
from pydrobert.speech.filters import GaborFilterBank
from pydrobert.speech.filters import LinearFilterBank
from pydrobert.speech.filters import TriangularOverlappingFilterBank
from pydrobert.speech.scales import LinearScaling
from pydrobert.speech.scales import MelScaling


RATE = 8000


def _half_width(width):
    return (width + width % 2) // 2 + 1 - width % 2


def _rebuild_half(bin_idx, trnc, width):
    half = np.zeros(_half_width(width), dtype=trnc.dtype)
    half[bin_idx:bin_idx + len(trnc)] = trnc
    return half


def _rebuild_full(bin_idx, trnc, width):
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    return full


def _simple_tri(analytic=False):
    # vertices land on 0, 1000, 2000, 3000, 4000 Hz
    return TriangularOverlappingFilterBank(
        LinearScaling(0.), num_filts=3, high_hz=4000, low_hz=0,
        sampling_rate=RATE, analytic=analytic)


def _mel_tri():
    return TriangularOverlappingFilterBank(
        MelScaling(), num_filts=10, high_hz=3500, low_hz=100,
        sampling_rate=RATE)


def _mel_fbank(analytic=False):
    return Fbank(
        num_filts=10, high_hz=3500, low_hz=100, sampling_rate=RATE,
        analytic=analytic)


def _gabor(**kwargs):
    return GaborFilterBank(
        MelScaling(), num_filts=8, high_hz=3800, low_hz=100,
        sampling_rate=RATE, **kwargs)


def _gammatone(**kwargs):
    return ComplexGammatoneFilterBank(
        MelScaling(), num_filts=6, high_hz=3500, low_hz=300,
        sampling_rate=RATE, **kwargs)


def test_linear_filter_bank_is_abstract():
    with pytest.raises(TypeError):
        LinearFilterBank()


def test_triangular_basic_properties():
    bank = _simple_tri()
    assert bank.num_filts == 3
    assert bank.sampling_rate == RATE
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert bank.centers_hz == pytest.approx((1000., 2000., 3000.))
    analytic = _simple_tri(analytic=True)
    assert analytic.is_analytic
    assert not analytic.is_real


@pytest.mark.parametrize(
    'low_hz,high_hz',
    [(-1., 4000), (2000., 1000.), (1000., 1000.), (100., 5000.)])
def test_triangular_invalid_frequency_range(low_hz, high_hz):
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            LinearScaling(0.), num_filts=3, high_hz=high_hz, low_hz=low_hz,
            sampling_rate=RATE)


def test_triangular_supports_and_centers():
    bank = _mel_tri()
    centers = bank.centers_hz
    supports_hz = bank.supports_hz
    supports = bank.supports
    supports_ms = bank.supports_ms
    assert len(centers) == len(supports_hz) == len(supports) == 10
    assert len(supports_ms) == 10
    assert all(a < b for a, b in zip(centers[:-1], centers[1:]))
    for center, (low, high) in zip(centers, supports_hz):
        assert low < center < high
    assert supports_hz[0][0] == pytest.approx(100., rel=1e-6)
    assert supports_hz[-1][1] == pytest.approx(3500., rel=1e-6)
    for (first, last), (first_ms, last_ms) in zip(supports, supports_ms):
        assert first < 0 < last
        assert first_ms == pytest.approx(first * 1000. / RATE)
        assert last_ms == pytest.approx(last * 1000. / RATE)


def test_triangular_frequency_response_known_values():
    bank = _simple_tri()
    res = bank.get_frequency_response(1, 16)
    assert res.dtype == np.float64
    assert res.shape == (16,)
    expected = np.zeros(16)
    expected[[3, 5]] = .5
    expected[4] = 1.
    expected[[13, 11]] = .5
    expected[12] = 1.
    np.testing.assert_allclose(res, expected, atol=1e-9)


@pytest.mark.parametrize('width', [16, 15])
def test_triangular_frequency_response_half(width):
    bank = _simple_tri()
    full = bank.get_frequency_response(1, width)
    half = bank.get_frequency_response(1, width, half=True)
    assert half.shape == (_half_width(width),)
    np.testing.assert_allclose(half, full[:len(half)], atol=1e-9)
    if width == 16:
        assert len(half) == 9
        np.testing.assert_allclose(
            half, [0, 0, 0, .5, 1, .5, 0, 0, 0], atol=1e-9)
    else:
        assert len(half) == 8


def test_triangular_frequency_response_symmetric_and_bounded():
    bank = _mel_tri()
    width = 512
    for filt_idx in range(bank.num_filts):
        res = bank.get_frequency_response(filt_idx, width)
        assert res.shape == (width,)
        assert np.all(res >= 0)
        assert np.all(res <= 1. + 1e-9)
        # real filter => hermitian symmetric
        np.testing.assert_allclose(res[1:], res[1:][::-1], atol=1e-9)
        # sampled finely enough that the peak is near unity
        assert res.max() > .9


def test_triangular_analytic_frequency_response():
    bank = _simple_tri(analytic=True)
    res = bank.get_frequency_response(1, 16)
    assert res.shape == (16,)
    np.testing.assert_allclose(res[:9], [0, 0, 0, .5, 1, .5, 0, 0, 0], atol=1e-9)
    # no negative frequencies
    assert np.all(res[9:] == 0)


def test_triangular_truncated_response_known_values():
    bank = _simple_tri()
    bin_idx, trnc = bank.get_truncated_response(1, 16)
    assert bin_idx == 2
    assert trnc.dtype == np.float64
    np.testing.assert_allclose(trnc, [0, .5, 1, .5, 0], atol=1e-9)


@pytest.mark.parametrize('width', [64, 257])
def test_triangular_truncated_matches_half_response(width):
    bank = _mel_tri()
    for filt_idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
        half = bank.get_frequency_response(filt_idx, width, half=True)
        assert 0 <= bin_idx
        assert bin_idx + len(trnc) <= len(half)
        np.testing.assert_allclose(
            _rebuild_half(bin_idx, trnc, width), half, atol=1e-9)


def test_fbank_properties_and_invalid_range():
    bank = _mel_fbank()
    assert bank.num_filts == 10
    assert bank.sampling_rate == RATE
    assert bank.is_real
    assert bank.is_zero_phase
    assert not bank.is_analytic
    assert _mel_fbank(analytic=True).is_analytic
    centers = bank.centers_hz
    assert len(centers) == 10
    assert all(a < b for a, b in zip(centers[:-1], centers[1:]))
    for center, (low, high) in zip(centers, bank.supports_hz):
        assert low < center < high
    with pytest.raises(ValueError):
        Fbank(num_filts=5, high_hz=100., low_hz=200., sampling_rate=RATE)
    with pytest.raises(ValueError):
        Fbank(num_filts=5, high_hz=5000., low_hz=20., sampling_rate=RATE)


@pytest.mark.parametrize('width', [128, 257])
def test_fbank_frequency_response_half_and_bounds(width):
    bank = _mel_fbank()
    for filt_idx in range(bank.num_filts):
        full = bank.get_frequency_response(filt_idx, width)
        half = bank.get_frequency_response(filt_idx, width, half=True)
        assert full.shape == (width,)
        assert half.shape == (_half_width(width),)
        assert np.all(full >= 0)
        assert np.all(full <= 1. + 1e-9)
        np.testing.assert_allclose(half, full[:len(half)], atol=1e-9)
        assert full.max() > .5


@pytest.mark.parametrize('width', [128, 257])
def test_fbank_truncated_matches_half_response(width):
    bank = _mel_fbank()
    for filt_idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
        half = bank.get_frequency_response(filt_idx, width, half=True)
        assert bin_idx + len(trnc) <= len(half)
        np.testing.assert_allclose(
            _rebuild_half(bin_idx, trnc, width), half, atol=1e-9)


def test_fbank_impulse_response_inverts_frequency_response():
    width = 128
    real = _mel_fbank()
    analytic = _mel_fbank(analytic=True)
    for filt_idx in (0, 4, 9):
        ir = real.get_impulse_response(filt_idx, width)
        assert ir.shape == (width,)
        assert not np.iscomplexobj(ir)
        np.testing.assert_allclose(
            np.fft.fft(ir), real.get_frequency_response(filt_idx, width),
            atol=1e-9)
        air = analytic.get_impulse_response(filt_idx, width)
        assert np.iscomplexobj(air)
        np.testing.assert_allclose(
            np.fft.fft(air), analytic.get_frequency_response(filt_idx, width),
            atol=1e-9)


def test_gabor_properties():
    bank = _gabor()
    assert bank.num_filts == 8
    assert bank.sampling_rate == RATE
    assert not bank.is_real
    assert bank.is_zero_phase
    assert not bank.scaled_l2_norm
    assert not bank.erb
    centers = bank.centers_hz
    assert len(centers) == 8
    assert all(a < b for a, b in zip(centers[:-1], centers[1:]))
    for center, (low, high) in zip(centers, bank.supports_hz):
        assert low < center < high
    for first, last in bank.supports:
        assert first < 0 < last
        assert first == -last
    erb_bank = _gabor(scale_l2_norm=True, erb=True)
    assert erb_bank.scaled_l2_norm
    assert erb_bank.erb
    with pytest.raises(ValueError):
        GaborFilterBank(
            MelScaling(), num_filts=3, high_hz=100., low_hz=200.,
            sampling_rate=RATE)


def test_gabor_frequency_response_shape_and_peak():
    bank = _gabor()
    width = 1024
    for filt_idx in range(bank.num_filts):
        full = bank.get_frequency_response(filt_idx, width)
        half = bank.get_frequency_response(filt_idx, width, half=True)
        assert full.dtype == np.float64
        assert full.shape == (width,)
        assert half.shape == (width // 2 + 1,)
        np.testing.assert_allclose(half, full[:len(half)], atol=1e-9)
        assert np.all(full >= 0)
        assert .95 < full.max() <= 1.01
    odd = bank.get_frequency_response(2, 101, half=True)
    assert odd.shape == (51,)


def test_gabor_l2_norm_scaling():
    width = 512
    plain = _gabor()
    normed = _gabor(scale_l2_norm=True)
    plain_peak = plain.get_frequency_response(0, width).max()
    normed_peak = normed.get_frequency_response(0, width).max()
    assert normed_peak > plain_peak
    for filt_idx in (0, 3, 7):
        ir = normed.get_impulse_response(filt_idx, width)
        assert ir.shape == (width,)
        assert np.sum(np.abs(ir) ** 2) == pytest.approx(1., abs=1e-3)


@pytest.mark.parametrize('width', [256, 513])
def test_gabor_truncated_matches_frequency_response(width):
    bank = _gabor()
    for filt_idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
        full = bank.get_frequency_response(filt_idx, width)
        assert 0 <= bin_idx < width
        assert len(trnc) <= width
        np.testing.assert_allclose(
            _rebuild_full(bin_idx, trnc, width), full, atol=2e-3)


def test_gabor_impulse_response_matches_frequency_response():
    bank = _gabor()
    width = 512
    for filt_idx in range(bank.num_filts):
        ir = bank.get_impulse_response(filt_idx, width)
        assert ir.dtype == np.complex128
        assert ir.shape == (width,)
        np.testing.assert_allclose(
            np.fft.fft(ir), bank.get_frequency_response(filt_idx, width),
            atol=1e-5)


def test_gammatone_properties_and_invalid_args():
    bank = _gammatone()
    assert bank.num_filts == 6
    assert bank.order == 4
    assert bank.sampling_rate == RATE
    assert not bank.is_real
    assert not bank.is_zero_phase
    assert not bank.scaled_l2_norm
    assert not bank.erb
    centers = bank.centers_hz
    assert len(centers) == 6
    assert all(a < b for a, b in zip(centers[:-1], centers[1:]))
    for first, last in bank.supports:
        assert first <= 0 < last
    assert _gammatone(order=2).order == 2
    for bad_order in (0, -1, 2.5):
        with pytest.raises(ValueError):
            _gammatone(order=bad_order)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(
            MelScaling(), num_filts=3, high_hz=100., low_hz=200.,
            sampling_rate=RATE)


@pytest.mark.parametrize('width', [512, 1023])
def test_gammatone_frequency_response_shape_and_peak(width):
    bank = _gammatone()
    for filt_idx in range(bank.num_filts):
        full = bank.get_frequency_response(filt_idx, width)
        half = bank.get_frequency_response(filt_idx, width, half=True)
        assert full.dtype == np.complex128
        assert full.shape == (width,)
        assert half.shape == (_half_width(width),)
        np.testing.assert_allclose(half, full[:len(half)], atol=1e-9)
        assert .9 < np.abs(full).max() < 1.05


@pytest.mark.parametrize('width', [256, 511])
def test_gammatone_truncated_matches_frequency_response(width):
    bank = _gammatone()
    for filt_idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
        full = bank.get_frequency_response(filt_idx, width)
        assert trnc.dtype == np.complex128
        assert 0 <= bin_idx < width
        assert len(trnc) <= width
        np.testing.assert_allclose(
            _rebuild_full(bin_idx, trnc, width), full, atol=2e-3)


def test_gammatone_impulse_response_causal_and_max_centered():
    width = 256
    causal = _gammatone()
    ir = causal.get_impulse_response(2, width)
    assert ir.dtype == np.complex128
    assert ir.shape == (width,)
    # causal filters are zero at t = 0 and nonzero shortly after
    assert abs(ir[0]) < 1e-8
    assert np.abs(ir[1:]).max() > 1e-4
    centered = _gammatone(max_centered=True)
    for first, _ in centered.supports:
        assert first < 0
    cir = centered.get_impulse_response(2, width)
    assert cir.shape == (width,)
    assert int(np.argmax(np.abs(cir))) == 0


def test_linear_filter_bank_from_alias():
    tri = LinearFilterBank.from_alias('tri', 'mel', num_filts=4)
    assert isinstance(tri, TriangularOverlappingFilterBank)
    assert tri.num_filts == 4
    tri2 = LinearFilterBank.from_alias(
        'triangular', {'alias': 'linear', 'low_hz': 0.}, num_filts=3)
    assert isinstance(tri2, TriangularOverlappingFilterBank)
    assert tri2.num_filts == 3
    fbank = LinearFilterBank.from_alias('fbank', num_filts=5)
    assert isinstance(fbank, Fbank)
    assert fbank.num_filts == 5
    gabor = LinearFilterBank.from_alias('gabor', 'mel', num_filts=4)
    assert isinstance(gabor, GaborFilterBank)
    assert gabor.num_filts == 4
    gamma = LinearFilterBank.from_alias('gammatone', 'mel', num_filts=4)
    assert isinstance(gamma, ComplexGammatoneFilterBank)
    assert gamma.num_filts == 4
    with pytest.raises(ValueError):
        LinearFilterBank.from_alias('not-a-real-bank')
