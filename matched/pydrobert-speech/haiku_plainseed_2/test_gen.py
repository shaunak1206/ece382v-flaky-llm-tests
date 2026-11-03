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
from pydrobert.speech.scales import LinearScaling, MelScaling

RATE = 16000
# 1600 bins over 16 kHz gives exactly 10 Hz per bin, so the linear-scale
# banks below (vertices every 1000 Hz) land exactly on bin boundaries.
WIDTH = 1600


def _linear_scale():
    return LinearScaling(low_hz=0.)


def _tri(**kwargs):
    kw = dict(num_filts=7, high_hz=8000., low_hz=0., sampling_rate=RATE)
    kw.update(kwargs)
    return TriangularOverlappingFilterBank(_linear_scale(), **kw)


def test_linear_filter_bank_is_abstract():
    with pytest.raises(TypeError):
        LinearFilterBank()


def test_incomplete_subclass_cannot_be_instantiated():
    class Partial(LinearFilterBank):
        is_real = True

    with pytest.raises(TypeError):
        Partial()


def test_tri_centers_follow_linear_grid():
    bank = _tri()
    assert bank.num_filts == 7
    assert len(bank.centers_hz) == 7
    assert list(bank.centers_hz) == pytest.approx(
        [1000. * k for k in range(1, 8)])
    assert bank.sampling_rate == RATE


def test_tri_supports_hz_span_neighbouring_centers():
    bank = _tri()
    for k, (lo, hi) in enumerate(bank.supports_hz):
        assert lo == pytest.approx(1000. * k)
        assert hi == pytest.approx(1000. * (k + 2))


def test_tri_flags_follow_analytic_argument():
    real = _tri()
    assert real.is_real and not real.is_analytic and real.is_zero_phase
    anal = _tri(analytic=True)
    assert not anal.is_real and anal.is_analytic and anal.is_zero_phase


def test_tri_rejects_invalid_frequency_ranges():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            MelScaling(), num_filts=5, high_hz=9000., low_hz=20.,
            sampling_rate=RATE)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            MelScaling(), num_filts=5, high_hz=500., low_hz=500.,
            sampling_rate=RATE)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(
            MelScaling(), num_filts=5, high_hz=1000., low_hz=-5.,
            sampling_rate=RATE)
    with pytest.raises(ValueError):
        Fbank(num_filts=5, high_hz=9000., low_hz=20., sampling_rate=RATE)
    with pytest.raises(ValueError):
        GaborFilterBank(
            MelScaling(), num_filts=5, high_hz=9000., low_hz=20.,
            sampling_rate=RATE)


def test_scaling_aliases_and_dicts_resolve():
    kw = dict(num_filts=7, high_hz=8000., low_hz=0., sampling_rate=RATE)
    by_name = TriangularOverlappingFilterBank('mel', **kw)
    by_obj = TriangularOverlappingFilterBank(MelScaling(), **kw)
    assert list(by_name.centers_hz) == pytest.approx(list(by_obj.centers_hz))
    by_dict = TriangularOverlappingFilterBank(
        {'alias': 'linear', 'low_hz': 0.}, **kw)
    assert list(by_dict.centers_hz) == pytest.approx(
        [1000. * k for k in range(1, 8)])
    via_alias = LinearFilterBank.from_alias(
        'fbank', num_filts=7, sampling_rate=RATE)
    assert isinstance(via_alias, Fbank)
    assert via_alias.num_filts == 7


def test_tri_frequency_response_is_triangular():
    bank = _tri()
    res = bank.get_frequency_response(0, WIDTH)
    assert res.dtype == np.float64
    assert res.shape == (WIDTH,)
    # filter 0 spans 0 to 2000 Hz with its peak at 1000 Hz (bin 100)
    assert res[0] == pytest.approx(0.)
    assert res[50] == pytest.approx(0.5)
    assert res[100] == pytest.approx(1.)
    assert res[150] == pytest.approx(0.5)
    assert res[200] == pytest.approx(0.)
    assert np.all(res[201:1400] == 0)


def test_tri_frequency_response_is_symmetric_for_real_bank():
    bank = _tri()
    for k in range(bank.num_filts):
        res = bank.get_frequency_response(k, WIDTH)
        assert np.allclose(res[1:800], res[WIDTH - 1:800:-1])


def test_tri_truncated_response_matches_full_response():
    bank = _tri()
    # filter 0 touches DC; its nonzero region is bins [0, 200]
    bin_idx, trnc = bank.get_truncated_response(0, WIDTH)
    full = bank.get_frequency_response(0, WIDTH)
    assert bin_idx == 0
    assert len(trnc) == 201
    assert np.allclose(trnc, full[:201])
    assert np.all(full[201:1400] == 0)
    # filter 3 sits away from DC; its nonzero region is [300, 500] plus
    # the mirror image at [1100, 1300]
    bin_idx, trnc = bank.get_truncated_response(3, WIDTH)
    full = bank.get_frequency_response(3, WIDTH)
    assert bin_idx == 300
    assert len(trnc) == 201
    assert np.allclose(trnc, full[300:501])
    assert np.all(full[:300] == 0)
    assert np.all(full[501:1100] == 0)
    assert np.all(full[1301:] == 0)
    assert np.allclose(full[1100:1301], trnc[::-1])


def test_tri_half_spectrum_lengths_and_values():
    bank = _tri()
    full = bank.get_frequency_response(0, WIDTH)
    half = bank.get_frequency_response(0, WIDTH, half=True)
    assert half.shape == (WIDTH // 2 + 1,)
    assert np.allclose(half, full[:WIDTH // 2 + 1])
    odd = bank.get_frequency_response(0, WIDTH + 1, half=True)
    assert odd.shape == ((WIDTH + 2) // 2,)
    assert odd.max() > 0.99


def test_tri_analytic_response_has_no_negative_frequencies():
    bank = _tri(analytic=True)
    res = bank.get_frequency_response(3, WIDTH)
    assert res[400] == pytest.approx(1.)
    assert np.all(res[:300] == 0)
    assert np.all(res[501:] == 0)


def test_tri_supports_bracket_zero_and_supports_ms():
    bank = _tri()
    supports = bank.supports
    assert len(supports) == 7
    for a, b in supports:
        assert isinstance(a, int) and isinstance(b, int)
        assert a < 0 < b
        assert a + b in (-1, 0)
    ms = bank.supports_ms
    assert len(ms) == len(supports)
    for (a, b), (a_ms, b_ms) in zip(supports, ms):
        assert a_ms == pytest.approx(a * 1000. / RATE)
        assert b_ms == pytest.approx(b * 1000. / RATE)


def test_fbank_centers_are_uniform_on_mel_scale():
    bank = Fbank(num_filts=10, sampling_rate=RATE)
    assert bank.num_filts == 10
    assert bank.is_real and not bank.is_analytic and bank.is_zero_phase
    mel = MelScaling()
    scales = np.array([mel.hertz_to_scale(c) for c in bank.centers_hz])
    diffs = np.diff(scales)
    assert np.allclose(diffs, diffs[0], rtol=1e-6)
    assert bank.supports_hz[0][0] == pytest.approx(20., rel=1e-6)
    assert bank.supports_hz[-1][1] == pytest.approx(8000., rel=1e-6)


def test_fbank_response_bounded_and_symmetric():
    bank = Fbank(num_filts=10, sampling_rate=RATE)
    res = bank.get_frequency_response(4, WIDTH)
    assert res.dtype == np.float64
    assert res.shape == (WIDTH,)
    assert np.all(res >= 0)
    assert np.all(res <= 1 + 1e-12)
    assert res.max() > 0.9
    assert np.allclose(res[1:800], res[WIDTH - 1:800:-1])


def test_fbank_truncated_response_matches_full_response():
    bank = Fbank(num_filts=10, sampling_rate=RATE)
    left, trnc = bank.get_truncated_response(4, WIDTH)
    full = bank.get_frequency_response(4, WIDTH)
    n = len(trnc)
    assert n > 0
    assert 0 <= left and left + n <= WIDTH
    assert np.allclose(trnc, full[left:left + n])
    outside = np.ones(WIDTH, dtype=bool)
    outside[left:left + n] = False
    # mirror image of the nonzero region
    outside[WIDTH - left - n + 1:WIDTH - left + 1] = False
    assert np.all(full[outside] == 0)


def test_fbank_analytic_and_half_spectrum():
    real = Fbank(num_filts=10, sampling_rate=RATE)
    anal = Fbank(num_filts=10, sampling_rate=RATE, analytic=True)
    assert anal.is_analytic and not anal.is_real
    full_r = real.get_frequency_response(4, WIDTH)
    full_a = anal.get_frequency_response(4, WIDTH)
    assert np.allclose(full_a[:800], full_r[:800])
    assert np.all(full_a[800:] == 0)
    half = real.get_frequency_response(4, WIDTH, half=True)
    assert half.shape == (WIDTH // 2 + 1,)
    assert np.allclose(half, full_r[:WIDTH // 2 + 1])


def test_gabor_centers_and_symmetric_supports():
    bank = GaborFilterBank(
        _linear_scale(), num_filts=7, high_hz=8000., low_hz=0.,
        sampling_rate=RATE)
    assert bank.num_filts == 7
    assert list(bank.centers_hz) == pytest.approx(
        [1000. * k for k in range(1, 8)])
    assert not bank.is_real and bank.is_zero_phase
    assert not bank.scaled_l2_norm and not bank.erb
    for center, (lo, hi), (d_lo, d_hi) in zip(
            bank.centers_hz, bank.supports_hz, bank.supports):
        assert (lo + hi) / 2 == pytest.approx(center)
        assert d_lo == -d_hi and d_hi > 0


def test_gabor_analyticity_depends_on_lowest_support():
    # the first filter's support dips below 0 rad, so it wraps
    low = GaborFilterBank(
        _linear_scale(), num_filts=7, high_hz=8000., low_hz=0.,
        sampling_rate=RATE)
    assert not low.is_analytic
    # every support stays above 0 rad when the bank is high enough
    high = GaborFilterBank(
        _linear_scale(), num_filts=2, high_hz=8000., low_hz=4000.,
        sampling_rate=RATE)
    assert high.is_analytic


def test_gabor_peak_scales_with_l2_norm():
    kw = dict(num_filts=7, high_hz=8000., low_hz=0., sampling_rate=RATE)
    plain = GaborFilterBank(_linear_scale(), **kw)
    l2 = GaborFilterBank(_linear_scale(), scale_l2_norm=True, **kw)
    assert l2.scaled_l2_norm
    res_plain = plain.get_frequency_response(0, WIDTH)
    res_l2 = l2.get_frequency_response(0, WIDTH)
    # the std of filter 0 is bandwidth_const / angular(500 Hz)
    std = np.sqrt(0.3 * np.log(10)) / (2 * np.pi * 500. / RATE)
    assert res_plain[100] == pytest.approx(1., abs=1e-6)
    assert res_l2[100] / res_plain[100] == pytest.approx(
        np.sqrt(2 * std) * np.pi ** .25, rel=1e-6)


def test_gabor_truncated_response_matches_full_response():
    bank = GaborFilterBank(
        _linear_scale(), num_filts=7, high_hz=8000., low_hz=0.,
        sampling_rate=RATE)
    left, trnc = bank.get_truncated_response(3, WIDTH)
    full = bank.get_frequency_response(3, WIDTH)
    assert left == 166
    assert len(trnc) == 469
    assert np.allclose(trnc, full[left:left + len(trnc)])
    # outside the truncated region the response is below the threshold
    eps = config.EFFECTIVE_SUPPORT_THRESHOLD * (1 + 1e-6)
    assert np.all(np.abs(full[:left]) <= eps)
    assert np.all(np.abs(full[left + len(trnc):]) <= eps)


def test_gabor_truncated_early_return_for_wide_filter():
    # a single filter spanning 2 kHz to 6 kHz has support wider than 2 pi
    bank = GaborFilterBank(
        _linear_scale(), num_filts=1, high_hz=8000., low_hz=0.,
        sampling_rate=RATE)
    left, trnc = bank.get_truncated_response(0, WIDTH)
    assert left == 0
    assert len(trnc) == WIDTH
    assert np.allclose(trnc, bank.get_frequency_response(0, WIDTH))


def test_gammatone_flags_and_order_validation():
    bank = ComplexGammatoneFilterBank(
        _linear_scale(), num_filts=7, high_hz=8000., low_hz=0.,
        sampling_rate=RATE, order=3)
    assert bank.order == 3
    assert bank.num_filts == 7
    assert not bank.is_real and not bank.is_zero_phase
    assert list(bank.centers_hz) == pytest.approx(
        [1000. * k for k in range(1, 8)])
    for bad in (0, -2, 1.5):
        with pytest.raises(ValueError):
            ComplexGammatoneFilterBank(
                _linear_scale(), num_filts=7, high_hz=8000., low_hz=0.,
                sampling_rate=RATE, order=bad)


def test_gammatone_peak_is_unity_and_wide_filter_uses_full_response():
    bank = ComplexGammatoneFilterBank(
        _linear_scale(), num_filts=7, high_hz=8000., low_hz=0.,
        sampling_rate=RATE)
    full = bank.get_frequency_response(0, WIDTH)
    assert full.dtype == np.complex128
    # the gain is normalized to 1 at the carrier (bin 100, 1000 Hz)
    assert abs(full[100] - 1.) < 1e-3
    # filter 0's support is wider than 2 pi, so the full response is returned
    left, trnc = bank.get_truncated_response(0, WIDTH)
    assert left == 0
    assert np.allclose(trnc, full)
