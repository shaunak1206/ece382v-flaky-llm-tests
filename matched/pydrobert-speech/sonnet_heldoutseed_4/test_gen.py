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


def _half_width(width):
    return width // 2 + 1 if width % 2 == 0 else (width + 1) // 2


def _wrap_full(bin_idx, trnc, width):
    full = np.zeros(width, dtype=trnc.dtype)
    idxs = (bin_idx + np.arange(len(trnc))) % width
    full[idxs] = trnc
    return full


# ---------------------------------------------------------------------------
# TriangularOverlappingFilterBank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", ["mel", "linear"])
def test_triangular_basic_properties(scale):
    bank = TriangularOverlappingFilterBank(scale, num_filts=10)
    assert bank.num_filts == 10
    assert bank.sampling_rate == RATE
    assert bank.is_real
    assert not bank.is_analytic
    assert bank.is_zero_phase
    assert len(bank.centers_hz) == 10
    assert len(bank.supports_hz) == 10
    assert len(bank.supports) == 10
    assert list(bank.centers_hz) == sorted(bank.centers_hz)
    for center, (low, high) in zip(bank.centers_hz, bank.supports_hz):
        assert low < center < high
    assert bank.supports_hz[0][0] == pytest.approx(20.0, rel=1e-6)
    assert bank.supports_hz[-1][1] == pytest.approx(RATE // 2, rel=1e-6)


def test_triangular_analytic_flags():
    bank = TriangularOverlappingFilterBank("mel", num_filts=5, analytic=True)
    assert bank.is_analytic
    assert not bank.is_real


def test_triangular_linear_scale_uniform_centers():
    bank = TriangularOverlappingFilterBank(
        "linear", num_filts=9, low_hz=100.0, high_hz=5100.0
    )
    centers = np.array(bank.centers_hz)
    assert np.allclose(centers, np.linspace(600.0, 4600.0, 9))
    diffs = np.diff(centers)
    assert np.allclose(diffs, diffs[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(low_hz=-1.0, high_hz=4000.0),
        dict(low_hz=500.0, high_hz=400.0),
        dict(low_hz=500.0, high_hz=500.0),
        dict(low_hz=20.0, high_hz=9000.0),
    ],
)
def test_triangular_invalid_frequency_range(kwargs):
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank("mel", num_filts=5, **kwargs)


def test_triangular_supports_ms_matches_supports():
    bank = TriangularOverlappingFilterBank("mel", num_filts=6)
    for (s_lo, s_hi), (ms_lo, ms_hi) in zip(bank.supports, bank.supports_ms):
        assert ms_lo == pytest.approx(s_lo * 1000.0 / RATE)
        assert ms_hi == pytest.approx(s_hi * 1000.0 / RATE)
        assert s_lo < 0 < s_hi


@pytest.mark.parametrize("width", [100, 101, 512])
def test_triangular_frequency_response_shape_and_values(width):
    bank = TriangularOverlappingFilterBank("linear", num_filts=8)
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert full.shape == (width,)
        assert half.shape == (_half_width(width),)
        assert full.dtype == np.float64
        assert np.all(full >= 0) and np.all(full <= 1.0 + 1e-12)
        assert np.allclose(full[: len(half)], half)
        # conjugate (here real) symmetry for the real bank
        assert np.allclose(full[1:], full[1:][::-1])


def test_triangular_frequency_response_peaks_at_center():
    bank = TriangularOverlappingFilterBank(
        "linear", num_filts=5, low_hz=100.0, high_hz=7000.0
    )
    width = 1600  # 10 Hz per bin
    for idx, center in enumerate(bank.centers_hz):
        half = bank.get_frequency_response(idx, width, half=True)
        peak_hz = np.argmax(half) * RATE / width
        assert abs(peak_hz - center) <= RATE / width
        assert half.max() > 0.95
        # no response outside the support
        low, high = bank.supports_hz[idx]
        bins_hz = np.arange(len(half)) * RATE / width
        outside = (bins_hz < low) | (bins_hz > high)
        assert np.all(half[outside] == 0)


def test_triangular_analytic_frequency_response_has_no_negative_freqs():
    bank = TriangularOverlappingFilterBank("mel", num_filts=6, analytic=True)
    width = 256
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        assert np.all(full[width // 2 + 1:] == 0)
        assert full.max() > 0


@pytest.mark.parametrize("analytic", [False, True])
@pytest.mark.parametrize("width", [512, 513])
def test_triangular_truncated_response_matches_frequency_response(
    analytic, width
):
    bank = TriangularOverlappingFilterBank(
        "mel", num_filts=10, analytic=analytic
    )
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        assert len(trnc) < width
        if analytic:
            full = np.zeros(width)
            full[bin_idx:bin_idx + len(trnc)] = trnc
            assert np.allclose(full, bank.get_frequency_response(idx, width))
        else:
            half = np.zeros(_half_width(width))
            half[bin_idx:bin_idx + len(trnc)] = trnc
            assert np.allclose(
                half, bank.get_frequency_response(idx, width, half=True)
            )


def test_triangular_impulse_response_real_and_symmetric():
    bank = TriangularOverlappingFilterBank("mel", num_filts=5)
    for idx in range(bank.num_filts):
        res = bank.get_impulse_response(idx, 128)
        assert res.shape == (128,)
        assert res.dtype == np.float64
        assert np.allclose(res[1:], res[1:][::-1])
    abank = TriangularOverlappingFilterBank("mel", num_filts=5, analytic=True)
    assert abank.get_impulse_response(0, 128).dtype == np.complex128


# ---------------------------------------------------------------------------
# Fbank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [512, 513])
def test_fbank_frequency_and_truncated_response(width):
    bank = Fbank(num_filts=8)
    assert bank.num_filts == 8
    assert bank.is_real and bank.is_zero_phase and not bank.is_analytic
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert np.all(full >= 0) and np.all(full <= 1.0 + 1e-12)
        assert half.shape == (_half_width(width),)
        assert np.allclose(full[: len(half)], half)
        assert np.allclose(full[1:], full[1:][::-1])
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        rebuilt = np.zeros(_half_width(width))
        rebuilt[bin_idx:bin_idx + len(trnc)] = trnc
        assert np.allclose(rebuilt, half)


# ---------------------------------------------------------------------------
# GaborFilterBank
# ---------------------------------------------------------------------------


def test_gabor_basic_properties():
    bank = GaborFilterBank("mel", num_filts=6, erb=True, scale_l2_norm=True)
    assert bank.num_filts == 6
    assert bank.sampling_rate == RATE
    assert not bank.is_real
    assert bank.is_zero_phase
    assert bank.erb is True
    assert bank.scaled_l2_norm is True
    assert len(bank.centers_hz) == 6
    assert list(bank.centers_hz) == sorted(bank.centers_hz)
    for center, (low, high) in zip(bank.centers_hz, bank.supports_hz):
        assert low < center < high
    for lo, hi in bank.supports:
        assert lo < 0 < hi
        assert lo == -hi
    with pytest.raises(ValueError):
        GaborFilterBank("mel", num_filts=3, low_hz=100.0, high_hz=50.0)
    with pytest.raises(ValueError):
        GaborFilterBank("mel", num_filts=3, low_hz=-5.0, high_hz=4000.0)


@pytest.mark.parametrize("width", [200, 201])
def test_gabor_frequency_response_half_and_full(width):
    bank = GaborFilterBank("mel", num_filts=4)
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert full.shape == (width,)
        assert full.dtype == np.float64
        assert half.shape == (_half_width(width),)
        assert np.allclose(full[: len(half)], half)
        assert np.all(full >= 0)


def test_gabor_frequency_response_peak_values():
    width = 2048
    plain = GaborFilterBank("mel", num_filts=4)
    normed = GaborFilterBank("mel", num_filts=4, scale_l2_norm=True)
    for idx in range(4):
        resp = plain.get_frequency_response(idx, width)
        assert 0.9 < resp.max() <= 1.0 + 1e-3
        center_bin = int(round(plain.centers_hz[idx] * width / RATE))
        assert abs(int(np.argmax(resp)) - center_bin) <= 1
        resp_n = normed.get_frequency_response(idx, width)
        expected = np.sqrt(2 * normed._stds[idx]) * np.pi ** 0.25
        assert resp_n.max() == pytest.approx(expected, rel=5e-2)


@pytest.mark.parametrize("scale_l2_norm", [False, True])
def test_gabor_truncated_response_matches_frequency_response(scale_l2_norm):
    width = 512
    bank = GaborFilterBank("mel", num_filts=5, scale_l2_norm=scale_l2_norm)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        assert len(trnc) <= width
        rebuilt = _wrap_full(bin_idx, trnc, width)
        full = bank.get_frequency_response(idx, width)
        assert np.allclose(rebuilt, full, atol=2e-3)


def test_gabor_impulse_response_conjugate_symmetry():
    bank = GaborFilterBank("mel", num_filts=4)
    width = 512
    for idx in range(bank.num_filts):
        res = bank.get_impulse_response(idx, width)
        assert res.shape == (width,)
        assert res.dtype == np.complex128
        assert abs(res[0].imag) < 1e-12
        assert res[0].real > 0
        assert np.allclose(res[1:][::-1], res[1:].conj())


# ---------------------------------------------------------------------------
# ComplexGammatoneFilterBank
# ---------------------------------------------------------------------------


def test_gammatone_basic_properties():
    bank = ComplexGammatoneFilterBank("mel", num_filts=5, order=3, erb=True)
    assert bank.num_filts == 5
    assert bank.order == 3
    assert bank.sampling_rate == RATE
    assert not bank.is_real
    assert not bank.is_zero_phase
    assert bank.erb is True
    assert bank.scaled_l2_norm is False
    assert len(bank.centers_hz) == 5
    assert len(bank.supports_hz) == 5
    assert len(bank.supports) == 5
    for center, (low, high) in zip(bank.centers_hz, bank.supports_hz):
        assert low < center < high
    for lo, hi in bank.supports:
        assert lo <= 0 < hi


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(order=0),
        dict(order=-2),
        dict(order=2.5),
        dict(low_hz=300.0, high_hz=200.0),
        dict(high_hz=RATE),
    ],
)
def test_gammatone_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank("mel", num_filts=3, **kwargs)


@pytest.mark.parametrize("width", [300, 301])
def test_gammatone_frequency_response_shape_and_peak(width):
    bank = ComplexGammatoneFilterBank("mel", num_filts=4)
    for idx in range(bank.num_filts):
        full = bank.get_frequency_response(idx, width)
        half = bank.get_frequency_response(idx, width, half=True)
        assert full.dtype == np.complex128
        assert full.shape == (width,)
        assert half.shape == (_half_width(width),)
        assert np.allclose(full[: len(half)], half)
    big = bank.get_frequency_response(2, 4096)
    assert 0.9 < np.abs(big).max() < 1.1


@pytest.mark.parametrize("order", [1, 4])
def test_gammatone_truncated_response_matches_frequency_response(order):
    width = 512
    bank = ComplexGammatoneFilterBank("mel", num_filts=5, order=order)
    for idx in range(bank.num_filts):
        bin_idx, trnc = bank.get_truncated_response(idx, width)
        assert 0 <= bin_idx < width
        assert trnc.dtype == np.complex128
        rebuilt = _wrap_full(bin_idx, trnc, width)
        full = bank.get_frequency_response(idx, width)
        assert np.allclose(rebuilt, full, atol=2e-3)


def test_gammatone_impulse_response_is_causal():
    bank = ComplexGammatoneFilterBank("mel", num_filts=4)
    width = 2048
    for idx in range(bank.num_filts):
        res = bank.get_impulse_response(idx, width)
        assert res.shape == (width,)
        assert res.dtype == np.complex128
        assert np.abs(res).max() > 0
        # causal, so the zeroth sample is (nearly) zero
        assert abs(res[0]) < config.EFFECTIVE_SUPPORT_THRESHOLD


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_linear_filter_bank_from_alias():
    tri = LinearFilterBank.from_alias("tri", "mel", num_filts=3)
    assert isinstance(tri, TriangularOverlappingFilterBank)
    assert tri.num_filts == 3
    gabor = LinearFilterBank.from_alias("gabor", "mel", num_filts=3)
    assert isinstance(gabor, GaborFilterBank)
    tone = LinearFilterBank.from_alias("gammatone", "mel", num_filts=3)
    assert isinstance(tone, ComplexGammatoneFilterBank)
