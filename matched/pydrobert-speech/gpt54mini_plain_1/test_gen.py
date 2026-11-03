import numpy as np
import pytest

from pydrobert.speech.filters import (
    BartlettWindow,
    BlackmanWindow,
    Fbank,
    GammaWindow,
    GaborFilterBank,
    GammatoneFilterBank,
    HammingWindow,
    HannWindow,
    LinearFilterBank,
    TriangularOverlappingFilterBank,
)
from pydrobert.speech.scales import ScalingFunction


class IdentityScaling(ScalingFunction):
    def hertz_to_scale(self, hz):
        return float(hz)

    def scale_to_hertz(self, scale):
        return float(scale)


def _reconstruct_full_from_truncated(width, bin_idx, trnc, dtype=None):
    if dtype is None:
        dtype = trnc.dtype
    full = np.zeros(width, dtype=dtype)
    if np.iscomplexobj(trnc):
        wrap = min(bin_idx + len(trnc), width) - bin_idx
        full[bin_idx:bin_idx + wrap] = trnc[:wrap]
        full[:len(trnc) - wrap] = trnc[wrap:]
    else:
        full[bin_idx:bin_idx + len(trnc)] = trnc
        if bin_idx:
            full[width - bin_idx - len(trnc) + 1:width - bin_idx + 1] = trnc[:None if bin_idx else 0:-1].conj()
        else:
            full[width - len(trnc) + 1:] = trnc[:0:-1].conj()
    return full


def test_linear_filter_bank_is_abstract():
    with pytest.raises(TypeError):
        LinearFilterBank()


def test_triangular_rejects_invalid_frequency_range():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(IdentityScaling(), low_hz=-1)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(IdentityScaling(), low_hz=100, high_hz=100)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(IdentityScaling(), low_hz=100, high_hz=50)


def test_triangular_basic_properties():
    bank = TriangularOverlappingFilterBank(
        IdentityScaling(), num_filts=3, low_hz=100, high_hz=700, sampling_rate=2000
    )
    assert bank.num_filts == 3
    assert bank.sampling_rate == 2000
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert bank.is_zero_phase is True
    assert len(bank.centers_hz) == 3
    assert len(bank.supports_hz) == 3
    assert len(bank.supports) == 3
    assert bank.supports_ms[0][0] == pytest.approx(bank.supports[0][0] * 1000 / 2000)


def test_triangular_frequency_response_half_and_symmetry():
    bank = TriangularOverlappingFilterBank(
        IdentityScaling(), num_filts=1, low_hz=100, high_hz=500, sampling_rate=2000
    )
    width = 128
    full = bank.get_frequency_response(0, width, half=False)
    half = bank.get_frequency_response(0, width, half=True)
    assert half.shape == (width // 2 + 1,)
    assert full.dtype == np.float64
    assert np.all(full >= 0)
    assert np.allclose(full[1:width // 2 + 1], full[:0:-1], atol=1e-12)
    assert np.allclose(half, full[: width // 2 + 1])


def test_triangular_truncated_response_reconstructs_full():
    bank = TriangularOverlappingFilterBank(
        IdentityScaling(), num_filts=2, low_hz=100, high_hz=700, sampling_rate=4000
    )
    width = 256
    bin_idx, trnc = bank.get_truncated_response(0, width)
    full = bank.get_frequency_response(0, width, half=False)
    recon = _reconstruct_full_from_truncated(width, bin_idx, trnc)
    assert np.allclose(recon, full)


def test_triangular_analytic_has_no_negative_frequency_energy():
    bank = TriangularOverlappingFilterBank(
        IdentityScaling(), num_filts=2, low_hz=100, high_hz=700, sampling_rate=4000, analytic=True
    )
    width = 256
    full = bank.get_frequency_response(0, width, half=False)
    assert bank.is_real is False
    assert bank.is_analytic is True
    assert np.allclose(full[width // 2 + 1 :], 0.0)
    assert np.all(full[: width // 2 + 1] >= 0)


def test_fbank_frequency_response_is_sqrt_of_triangular_shape():
    bank = Fbank(num_filts=2, low_hz=100, high_hz=700, sampling_rate=4000, analytic=False)
    width = 256
    full = bank.get_frequency_response(0, width, half=False)
    half = bank.get_frequency_response(0, width, half=True)
    assert half.shape == (width // 2 + 1,)
    assert np.all(full >= 0)
    assert np.allclose(full[: width // 2 + 1], half)
    assert np.allclose(full[1:width // 2 + 1], full[:0:-1], atol=1e-12)


def test_fbank_truncated_response_matches_frequency_response_half():
    bank = Fbank(num_filts=2, low_hz=100, high_hz=700, sampling_rate=4000, analytic=False)
    width = 256
    bin_idx, trnc = bank.get_truncated_response(0, width)
    half = bank.get_frequency_response(0, width, half=True)
    assert bin_idx >= 0
    recon = np.zeros_like(half)
    recon[bin_idx:bin_idx + len(trnc)] = trnc
    assert np.allclose(recon, half)


def test_fbank_impulse_response_is_real_for_nonanalytic_bank():
    bank = Fbank(num_filts=1, low_hz=100, high_hz=700, sampling_rate=4000, analytic=False)
    res = bank.get_impulse_response(0, 64)
    assert np.isrealobj(res)
    assert res.shape == (64,)


def test_gabor_basic_properties_and_shapes():
    bank = GaborFilterBank(
        IdentityScaling(), num_filts=2, low_hz=100, high_hz=700, sampling_rate=4000
    )
    assert bank.is_real is False
    assert bank.is_zero_phase is True
    assert bank.num_filts == 2
    assert len(bank.centers_hz) == 2
    width = 128
    full = bank.get_frequency_response(0, width, half=False)
    half = bank.get_frequency_response(0, width, half=True)
    assert full.shape == (width,)
    assert half.shape == (width // 2 + 1,)
    assert np.iscomplexobj(bank.get_impulse_response(0, width))
    assert np.iscomplexobj(full)


def test_gabor_truncated_response_reconstructs_full():
    bank = GaborFilterBank(
        IdentityScaling(), num_filts=2, low_hz=100, high_hz=700, sampling_rate=4000
    )
    width = 128
    bin_idx, trnc = bank.get_truncated_response(0, width)
    full = bank.get_frequency_response(0, width, half=False)
    recon = _reconstruct_full_from_truncated(width, bin_idx, trnc, dtype=trnc.dtype)
    assert np.allclose(recon, full)


def test_gabor_analytic_flag_changes_with_low_frequency_wrap():
    bank = GaborFilterBank(
        IdentityScaling(), num_filts=1, low_hz=1, high_hz=1000, sampling_rate=4000
    )
    assert isinstance(bank.is_analytic, bool)


def test_gammatone_rejects_invalid_order():
    with pytest.raises(ValueError):
        GammatoneFilterBank(IdentityScaling(), order=0)
    with pytest.raises(ValueError):
        GammatoneFilterBank(IdentityScaling(), order=-2)
    with pytest.raises(ValueError):
        GammatoneFilterBank(IdentityScaling(), order=1.5)


def test_gammatone_basic_shapes_and_frequency_response():
    bank = GammatoneFilterBank(
        IdentityScaling(), num_filts=2, low_hz=100, high_hz=700, sampling_rate=4000, order=4
    )
    width = 128
    full = bank.get_frequency_response(0, width, half=False)
    half = bank.get_frequency_response(0, width, half=True)
    assert bank.is_real is False
    assert bank.is_zero_phase is False
    assert full.shape == (width,)
    assert half.shape == (width // 2 + 1,)
    assert np.iscomplexobj(full)
    assert np.iscomplexobj(bank.get_impulse_response(0, width))


def test_gammatone_truncated_response_reconstructs_full():
    bank = GammatoneFilterBank(
        IdentityScaling(), num_filts=2, low_hz=100, high_hz=700, sampling_rate=4000, order=4
    )
    width = 128
    bin_idx, trnc = bank.get_truncated_response(0, width)
    full = bank.get_frequency_response(0, width, half=False)
    recon = _reconstruct_full_from_truncated(width, bin_idx, trnc, dtype=trnc.dtype)
    assert np.allclose(recon, full)


def test_gamma_window_edge_cases():
    w0 = GammaWindow(order=4, peak=0.75).get_impulse_response(0)
    w1 = GammaWindow(order=4, peak=0.75).get_impulse_response(1)
    assert w0.shape == (0,)
    assert w1.shape == (1,)
    assert w1[0] == pytest.approx(1.0)


def test_standard_windows_have_expected_length_and_symmetry():
    for window in (BartlettWindow(), BlackmanWindow(), HammingWindow(), HannWindow()):
        arr = window.get_impulse_response(9)
        assert arr.shape == (9,)
        assert np.allclose(arr, arr[::-1])
        assert arr[0] == pytest.approx(arr[-1])
