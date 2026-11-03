"""Unit tests for the linear filter banks in pydrobert.speech.filters.

Covers LinearFilterBank and its concrete subclasses (triangular, Fbank,
Gabor, complex gammatone), focusing on get_frequency_response and
get_truncated_response. Everything is deterministic and runs offline.
"""

import numpy as np
import pytest

from pydrobert.speech.filters import ComplexGammatoneFilterBank
from pydrobert.speech.filters import Fbank
from pydrobert.speech.filters import GaborFilterBank
from pydrobert.speech.filters import LinearFilterBank
from pydrobert.speech.filters import TriangularOverlappingFilterBank
from pydrobert.speech.scales import MelScaling

RATE = 16000
WIDTH = 512
ODD_WIDTH = 511
HALF_EVEN = WIDTH // 2 + 1
HALF_ODD = (ODD_WIDTH + 1) // 2


def make_tri(num_filts=20, low_hz=100., high_hz=7000., analytic=False):
    return TriangularOverlappingFilterBank(
        'mel', num_filts=num_filts, low_hz=low_hz, high_hz=high_hz,
        sampling_rate=RATE, analytic=analytic)


def make_fbank(num_filts=24, low_hz=50., high_hz=7500.):
    return Fbank(
        num_filts=num_filts, low_hz=low_hz, high_hz=high_hz,
        sampling_rate=RATE)


def make_gabor(num_filts=16, low_hz=50., high_hz=7000.):
    return GaborFilterBank(
        'mel', num_filts=num_filts, low_hz=low_hz, high_hz=high_hz,
        sampling_rate=RATE, scale_l2_norm=True, erb=True)


def make_gammatone(num_filts=12, low_hz=50., high_hz=7000., order=4):
    return ComplexGammatoneFilterBank(
        'mel', num_filts=num_filts, low_hz=low_hz, high_hz=high_hz,
        sampling_rate=RATE, order=order)


def test_linear_filter_bank_is_abstract():
    with pytest.raises(TypeError):
        LinearFilterBank()


def test_concrete_banks_implement_abstract_interface():
    for cls in (TriangularOverlappingFilterBank, Fbank,
                GaborFilterBank, ComplexGammatoneFilterBank):
        assert issubclass(cls, LinearFilterBank)
        assert not cls.__abstractmethods__


def test_from_alias_resolves_each_bank_and_rejects_unknown():
    assert isinstance(
        LinearFilterBank.from_alias('fbank', num_filts=8), Fbank)
    assert isinstance(
        LinearFilterBank.from_alias('tri', 'mel', num_filts=8),
        TriangularOverlappingFilterBank)
    assert isinstance(
        LinearFilterBank.from_alias('gabor', 'mel', num_filts=8),
        GaborFilterBank)
    assert isinstance(
        LinearFilterBank.from_alias('gammatone', 'mel', num_filts=8),
        ComplexGammatoneFilterBank)
    with pytest.raises(ValueError):
        LinearFilterBank.from_alias('not-a-bank')


def test_triangular_counts_and_flags():
    bank = make_tri(num_filts=20)
    assert bank.num_filts == 20
    assert len(bank.centers_hz) == 20
    assert len(bank.supports_hz) == 20
    assert len(bank.supports) == 20
    assert bank.sampling_rate == RATE
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase


def test_triangular_centers_uniform_on_mel():
    bank = make_tri(num_filts=20)
    mel = MelScaling().hertz_to_scale(
        np.asarray(bank.centers_hz, dtype=np.float64))
    steps = np.diff(mel)
    assert np.all(steps > 0)
    assert np.allclose(steps, steps[0], rtol=1e-6)


def test_triangular_supports_span_requested_band():
    bank = make_tri(num_filts=20, low_hz=100., high_hz=7000.)
    low, _ = bank.supports_hz[0]
    _, high = bank.supports_hz[-1]
    assert low == pytest.approx(100., rel=1e-6)
    assert high == pytest.approx(7000., rel=1e-6)
    assert low < bank.centers_hz[0] < bank.centers_hz[-1] < high


def test_triangular_default_high_hz_is_nyquist():
    bank = TriangularOverlappingFilterBank(
        'mel', num_filts=10, low_hz=20., sampling_rate=RATE)
    _, high = bank.supports_hz[-1]
    assert high == pytest.approx(RATE // 2, rel=1e-6)


def test_triangular_invalid_frequency_range_raises():
    # high_hz is always passed explicitly: the error message formats it
    # with '.2f', which fails on None
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            'mel', num_filts=10, low_hz=-10., high_hz=4000.,
            sampling_rate=RATE)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            'mel', num_filts=10, low_hz=2000., high_hz=1000.,
            sampling_rate=RATE)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            'mel', num_filts=10, low_hz=100., high_hz=9000.,
            sampling_rate=RATE)


def test_triangular_response_real_symmetric_and_bounded():
    bank = make_tri()
    res = bank.get_frequency_response(3, WIDTH)
    assert res.shape == (WIDTH,)
    assert not np.iscomplexobj(res)
    # real filter: X[k] == X[N - k]
    assert np.allclose(res[1:], res[1:][::-1])
    assert res.min() >= -1e-12
    assert res.max() <= 1 + 1e-12
    assert res.max() > 0


def test_triangular_half_response_is_prefix_of_full():
    bank = make_tri()
    full = bank.get_frequency_response(2, WIDTH)
    half = bank.get_frequency_response(2, WIDTH, half=True)
    assert half.shape == (HALF_EVEN,)
    assert np.allclose(half, full[:HALF_EVEN])

    full_odd = bank.get_frequency_response(2, ODD_WIDTH)
    half_odd = bank.get_frequency_response(2, ODD_WIDTH, half=True)
    assert half_odd.shape == (HALF_ODD,)
    assert np.allclose(half_odd, full_odd[:HALF_ODD])


def test_triangular_truncated_response_matches_full_slice():
    bank = make_tri()
    for filt in range(bank.num_filts):
        left_idx, trunc = bank.get_truncated_response(filt, WIDTH)
        low_hz = bank.supports_hz[filt][0]
        assert left_idx == int(np.ceil(WIDTH * low_hz / RATE))
        full = bank.get_frequency_response(filt, WIDTH)
        assert np.allclose(full[left_idx:left_idx + len(trunc)], trunc)


def test_triangular_truncated_response_is_zero_outside_support():
    bank = make_tri()
    for filt in range(bank.num_filts):
        left_idx, trunc = bank.get_truncated_response(filt, WIDTH)
        half = bank.get_frequency_response(filt, WIDTH, half=True)
        nonzero = np.flatnonzero(half)
        assert nonzero.size > 0
        assert nonzero[0] >= left_idx
        assert nonzero[-1] < left_idx + len(trunc)


def test_triangular_analytic_suppresses_negative_frequencies():
    bank = make_tri(analytic=True)
    assert not bank.is_real
    assert bank.is_analytic
    res = bank.get_frequency_response(5, WIDTH)
    assert np.allclose(res[WIDTH // 2 + 1:], 0)
    assert np.any(res[:WIDTH // 2 + 1] > 0)


def test_triangular_supports_ms_and_centring():
    bank = make_tri()
    supports = np.asarray(bank.supports, dtype=np.float64)
    assert np.allclose(
        np.asarray(bank.supports_ms), supports * 1000. / RATE)
    assert all(lo < 0 < hi for lo, hi in bank.supports)


def test_fbank_counts_and_centers():
    bank = make_fbank(num_filts=24)
    assert bank.num_filts == 24
    centers = np.asarray(bank.centers_hz)
    assert centers.shape == (24,)
    assert np.all(np.diff(centers) > 0)
    for filt, (low, high) in enumerate(bank.supports_hz):
        assert low < centers[filt] < high
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase


def test_fbank_truncated_response_matches_full_slice():
    bank = make_fbank()
    for filt in range(bank.num_filts):
        left_idx, trunc = bank.get_truncated_response(filt, WIDTH)
        low_hz = bank.supports_hz[filt][0]
        assert left_idx == int(np.ceil(WIDTH * low_hz / RATE))
        full = bank.get_frequency_response(filt, WIDTH)
        assert np.allclose(full[left_idx:left_idx + len(trunc)], trunc)
        assert np.all(trunc >= 0)
        assert np.all(trunc <= 1 + 1e-12)


def test_fbank_response_bounded_and_symmetric():
    bank = make_fbank()
    res = bank.get_frequency_response(4, WIDTH)
    assert np.allclose(res[1:], res[1:][::-1])
    assert res.min() >= 0
    assert res.max() <= 1 + 1e-12
    assert res.max() > 0.5


def test_gabor_counts_flags_and_centers():
    bank = make_gabor(num_filts=16)
    assert bank.num_filts == 16
    centers = np.asarray(bank.centers_hz)
    assert centers.shape == (16,)
    assert np.all(np.diff(centers) > 0)
    assert all(lo == -hi and hi >= 1 for lo, hi in bank.supports)
    assert not bank.is_real
    assert bank.is_zero_phase
    assert bank.erb is True
    assert bank.scaled_l2_norm is True


def test_gabor_response_nonnegative_and_peaks_near_centers():
    bank = make_gabor(num_filts=16)
    centers = np.asarray(bank.centers_hz)
    for filt in range(bank.num_filts):
        res = bank.get_frequency_response(filt, WIDTH)
        assert res.shape == (WIDTH,)
        assert np.all(np.isfinite(res))
        assert np.all(res >= 0)
        # the response is a function of (omega - center) only, so its
        # maximum sits on the bin nearest the center frequency
        center_bin = centers[filt] * WIDTH / RATE
        assert abs(int(np.argmax(res)) - center_bin) <= 1.0


def test_gabor_truncated_response_reconstructs_full():
    bank = make_gabor(num_filts=16)
    for filt in range(bank.num_filts):
        left_idx, trunc = bank.get_truncated_response(filt, WIDTH)
        full = bank.get_frequency_response(filt, WIDTH)
        # the truncated region may wrap around the end of the buffer
        recon = np.zeros(WIDTH)
        for j, value in enumerate(trunc):
            recon[(left_idx + j) % WIDTH] += value
        # outside the truncated region the full response is only
        # bounded by the effective support threshold (5e-4)
        assert np.max(np.abs(recon - full)) < 5e-3


def test_gammatone_order_validation():
    for bad_order in (0, -1, 2.5):
        with pytest.raises(ValueError):
            ComplexGammatoneFilterBank(
                'mel', num_filts=8, low_hz=50., high_hz=7000.,
                sampling_rate=RATE, order=bad_order)
    assert make_gammatone().order == 4


def test_gammatone_flags_and_response_shapes():
    bank = make_gammatone(num_filts=12)
    assert bank.num_filts == 12
    assert len(bank.centers_hz) == 12
    assert not bank.is_zero_phase
    assert not bank.is_real
    assert bank.sampling_rate == RATE
    full = bank.get_frequency_response(0, WIDTH)
    assert full.shape == (WIDTH,)
    assert np.iscomplexobj(full)
    half = bank.get_frequency_response(0, WIDTH, half=True)
    assert half.shape == (HALF_EVEN,)
    assert np.iscomplexobj(half)


def test_gammatone_response_peaks_near_centers():
    bank = make_gammatone()
    centers = np.asarray(bank.centers_hz)
    for filt in range(bank.num_filts):
        mag = np.abs(bank.get_frequency_response(filt, WIDTH))
        # |H| is a decreasing function of |omega - xi|, so the largest
        # magnitude sits on the bin nearest the carrier frequency
        center_bin = centers[filt] * WIDTH / RATE
        assert abs(int(np.argmax(mag)) - center_bin) <= 1.0


def test_gammatone_truncated_matches_full_in_support():
    bank = make_gammatone()
    for filt in range(bank.num_filts):
        left_idx, trunc = bank.get_truncated_response(filt, WIDTH)
        full = bank.get_frequency_response(filt, WIDTH)
        bins = (left_idx + np.arange(len(trunc))) % WIDTH
        assert np.allclose(full[bins], trunc, atol=1e-6)
