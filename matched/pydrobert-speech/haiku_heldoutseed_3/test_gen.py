import numpy as np
import pytest

from pydrobert.speech.filters import (
    LinearFilterBank,
    TriangularOverlappingFilterBank,
    Fbank,
    GaborFilterBank,
    ComplexGammatoneFilterBank,
)
from pydrobert.speech.scales import MelScaling

RATE = 16000
NUM_FILTS = 10
LOW_HZ = 50.
HIGH_HZ = 7000.


def _tri(**kwargs):
    kwargs.setdefault('num_filts', NUM_FILTS)
    kwargs.setdefault('low_hz', LOW_HZ)
    kwargs.setdefault('high_hz', HIGH_HZ)
    kwargs.setdefault('sampling_rate', RATE)
    return TriangularOverlappingFilterBank(MelScaling(), **kwargs)


def _fbank(**kwargs):
    kwargs.setdefault('num_filts', NUM_FILTS)
    kwargs.setdefault('low_hz', LOW_HZ)
    kwargs.setdefault('high_hz', HIGH_HZ)
    kwargs.setdefault('sampling_rate', RATE)
    return Fbank(**kwargs)


def _gabor(**kwargs):
    kwargs.setdefault('num_filts', NUM_FILTS)
    kwargs.setdefault('low_hz', LOW_HZ)
    kwargs.setdefault('high_hz', HIGH_HZ)
    kwargs.setdefault('sampling_rate', RATE)
    return GaborFilterBank(MelScaling(), **kwargs)


def _gammatone(**kwargs):
    kwargs.setdefault('num_filts', NUM_FILTS)
    kwargs.setdefault('low_hz', LOW_HZ)
    kwargs.setdefault('high_hz', HIGH_HZ)
    kwargs.setdefault('sampling_rate', RATE)
    return ComplexGammatoneFilterBank(MelScaling(), **kwargs)


def _banks():
    return [
        _tri(),
        _tri(analytic=True),
        _fbank(),
        _fbank(analytic=True),
        _gabor(),
        _gammatone(),
    ]


def test_triangular_properties():
    bank = _tri()
    assert isinstance(bank, LinearFilterBank)
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert bank.num_filts == NUM_FILTS
    assert bank.sampling_rate == RATE
    centers = np.array(bank.centers_hz)
    assert len(centers) == NUM_FILTS
    assert LOW_HZ < centers[0]
    assert centers[-1] < HIGH_HZ


def test_triangular_analytic_flags():
    bank = _tri(analytic=True)
    assert not bank.is_real
    assert bank.is_analytic
    assert bank.is_zero_phase


def test_fbank_and_gabor_flags():
    fb = _fbank()
    assert fb.is_real and not fb.is_analytic and fb.is_zero_phase
    assert fb.num_filts == NUM_FILTS
    gb = _gabor()
    assert not gb.is_real
    assert gb.is_zero_phase
    assert gb.num_filts == NUM_FILTS
    assert gb.scaled_l2_norm is False
    assert gb.erb is False
    assert _gabor(erb=True, scale_l2_norm=True).erb is True
    assert _gabor(scale_l2_norm=True).scaled_l2_norm is True


def test_gammatone_flags():
    bank = _gammatone(order=3)
    assert not bank.is_real
    assert not bank.is_zero_phase
    assert bank.order == 3
    assert bank.num_filts == NUM_FILTS
    assert bank.sampling_rate == RATE


def test_invalid_frequency_range_raises():
    ctors = [
        lambda **kw: TriangularOverlappingFilterBank(MelScaling(), **kw),
        lambda **kw: Fbank(**kw),
        lambda **kw: GaborFilterBank(MelScaling(), **kw),
        lambda **kw: ComplexGammatoneFilterBank(MelScaling(), **kw),
    ]
    for ctor in ctors:
        with pytest.raises(ValueError):
            ctor(low_hz=-1., high_hz=HIGH_HZ, sampling_rate=RATE)
        with pytest.raises(ValueError):
            ctor(low_hz=1000., high_hz=500., sampling_rate=RATE)
        with pytest.raises(ValueError):
            ctor(low_hz=LOW_HZ, high_hz=9000., sampling_rate=RATE)


def test_gammatone_invalid_order_raises():
    with pytest.raises(ValueError):
        _gammatone(order=0)
    with pytest.raises(ValueError):
        _gammatone(order=2.5)


def test_centers_increasing_and_within_supports_hz():
    for bank in _banks():
        centers = np.array(bank.centers_hz)
        assert len(centers) == NUM_FILTS
        assert np.all(np.diff(centers) > 0)
        supports = bank.supports_hz
        assert len(supports) == NUM_FILTS
        for c, (lo, hi) in zip(centers, supports):
            assert lo < c < hi


def test_triangular_and_fbank_full_response_shape_and_bounds():
    width = 512
    for bank in (_tri(), _fbank()):
        for f in range(bank.num_filts):
            resp = bank.get_frequency_response(f, width)
            assert resp.shape == (width,)
            assert np.all(resp >= 0)
            assert np.all(resp <= 1 + 1e-9)
            assert resp.max() > 0


def test_half_response_length():
    for bank in _banks():
        for width in (512, 511):
            half = bank.get_frequency_response(0, width, half=True)
            if width % 2:
                expected = (width + 1) // 2
            else:
                expected = width // 2 + 1
            assert len(half) == expected


def test_half_is_prefix_of_full():
    width = 512
    for bank in _banks():
        for f in range(bank.num_filts):
            full = bank.get_frequency_response(f, width)
            half = bank.get_frequency_response(f, width, half=True)
            np.testing.assert_allclose(half, full[:len(half)], rtol=1e-12,
                                       atol=1e-12)


def test_dtype_matches_zero_phase():
    for bank in _banks():
        resp = bank.get_frequency_response(0, 256)
        if bank.is_zero_phase:
            assert resp.dtype == np.float64
        else:
            assert resp.dtype == np.complex128


def test_real_response_is_hermitian_symmetric():
    width = 512
    for bank in (_tri(), _fbank()):
        assert bank.is_real
        for f in range(bank.num_filts):
            full = bank.get_frequency_response(f, width)
            np.testing.assert_allclose(full[1:], full[1:][::-1], rtol=1e-12,
                                       atol=1e-12)


def test_analytic_response_zero_on_negative_freqs():
    width = 512
    for bank in (_tri(analytic=True), _fbank(analytic=True)):
        assert bank.is_analytic
        for f in range(bank.num_filts):
            full = bank.get_frequency_response(f, width)
            np.testing.assert_array_equal(full[width // 2 + 1:], 0)


def test_triangular_truncated_matches_full():
    width = 512
    for bank in (_tri(), _tri(analytic=True)):
        for f in range(bank.num_filts):
            full = bank.get_frequency_response(f, width)
            bin_idx, buf = bank.get_truncated_response(f, width)
            assert isinstance(bin_idx, (int, np.integer))
            assert len(buf) >= 1
            np.testing.assert_allclose(full[bin_idx:bin_idx + len(buf)], buf,
                                       rtol=1e-12, atol=1e-12)


def test_triangular_truncated_zero_outside_support():
    bank = _tri()
    width = 512
    for f in range(bank.num_filts):
        full = bank.get_frequency_response(f, width)
        bin_idx, buf = bank.get_truncated_response(f, width)
        support = set()
        for j in range(len(buf)):
            k = (bin_idx + j) % width
            support.add(k)
            support.add((-k) % width)
        outside = [k for k in range(width) if k not in support]
        np.testing.assert_array_equal(full[outside], 0)


def test_fbank_and_gabor_truncated_matches_full():
    width = 1024
    for bank in (_fbank(), _fbank(analytic=True), _gabor()):
        for f in range(bank.num_filts):
            full = bank.get_frequency_response(f, width)
            bin_idx, buf = bank.get_truncated_response(f, width)
            assert len(buf) >= 1
            idx = (bin_idx + np.arange(len(buf))) % width
            np.testing.assert_allclose(buf, full[idx], rtol=1e-6, atol=1e-6)


def test_gammatone_truncated_matches_full():
    bank = _gammatone()
    width = 1024
    for f in range(bank.num_filts):
        full = bank.get_frequency_response(f, width)
        bin_idx, buf = bank.get_truncated_response(f, width)
        assert np.iscomplexobj(buf)
        assert len(buf) >= 1
        idx = (bin_idx + np.arange(len(buf))) % width
        scale = np.abs(full).max()
        assert np.max(np.abs(buf - full[idx])) <= 1e-2 * scale


def test_truncated_bin_index_in_range():
    width = 1024
    for bank in _banks():
        for f in range(bank.num_filts):
            bin_idx, buf = bank.get_truncated_response(f, width)
            assert 0 <= int(bin_idx) < width
            assert len(buf) >= 1


def test_gabor_peak_near_center():
    bank = _gabor()
    width = 2048
    for f, center in enumerate(bank.centers_hz):
        full = bank.get_frequency_response(f, width)
        k = int(np.argmax(full))
        hz = k * RATE / width
        if k > width // 2:
            hz -= RATE
        assert abs(hz - center) <= 1.5 * RATE / width
        assert full[k] > 0.95


def test_gabor_l2_norm_is_constant_scale():
    default = _gabor()
    l2 = _gabor(scale_l2_norm=True)
    width = 1024
    for f in range(default.num_filts):
        a = default.get_frequency_response(f, width)
        b = l2.get_frequency_response(f, width)
        mask = a > 1e-6
        assert mask.any()
        ratio = b[mask] / a[mask]
        assert ratio[0] > 0
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-6)
