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


def _reconstruct_full_from_truncated(bank, filt_idx, width):
    """Reconstruct a full response from get_truncated_response output."""
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = np.zeros(width, dtype=trnc.dtype)
    if np.issubdtype(trnc.dtype, np.complexfloating):
        wrap = min(bin_idx + len(trnc), width) - bin_idx
        full[bin_idx:bin_idx + wrap] = trnc[:wrap]
        full[:len(trnc) - wrap] = trnc[wrap:]
    else:
        full[bin_idx:bin_idx + len(trnc)] = trnc
    return bin_idx, trnc, full


def test_linear_filter_bank_is_abstract():
    with pytest.raises(TypeError):
        LinearFilterBank()


def test_triangular_bank_basic_properties():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, sampling_rate=16000)
    assert bank.num_filts == 3
    assert bank.sampling_rate == 16000
    assert bank.is_zero_phase is True
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert len(bank.centers_hz) == 3
    assert len(bank.supports_hz) == 3
    assert len(bank.supports) == 3


def test_triangular_bank_supports_ms_matches_supports():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=2, sampling_rate=8000)
    supports_ms = bank.supports_ms
    for (s0, s1), (m0, m1) in zip(bank.supports, supports_ms):
        assert m0 == pytest.approx(s0 * 1000 / 8000)
        assert m1 == pytest.approx(s1 * 1000 / 8000)


def test_triangular_frequency_response_shape_and_symmetry():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, sampling_rate=16000)
    width = 16
    resp = bank.get_frequency_response(1, width, half=False)
    assert resp.shape == (width,)
    assert resp.dtype == np.float64
    # real filter bank should be Hermitian-symmetric in the frequency domain
    for k in range(1, width):
        assert resp[k] == pytest.approx(resp[-k])


def test_triangular_frequency_response_half_matches_full_prefix():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, sampling_rate=16000)
    width = 15
    half = bank.get_frequency_response(1, width, half=True)
    full = bank.get_frequency_response(1, width, half=False)
    assert half.shape == ((width + 1) // 2,)
    np.testing.assert_allclose(half, full[: len(half)])


def test_triangular_truncated_response_reconstructs_full():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, sampling_rate=16000)
    width = 32
    bin_idx, trnc, full = _reconstruct_full_from_truncated(bank, 1, width)
    resp = bank.get_frequency_response(1, width, half=False)
    assert 0 <= bin_idx < width
    np.testing.assert_allclose(full, resp)


def test_triangular_analytic_bank_has_no_negative_frequency_mirror():
    bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=3, sampling_rate=16000, analytic=True
    )
    width = 32
    resp = bank.get_frequency_response(1, width, half=False)
    # Analytic response should not mirror into negative frequencies
    assert resp.shape == (width,)
    assert np.count_nonzero(resp[1:]) > 0
    assert np.all(resp[width // 2 + 1 :] == 0)


def test_fbank_matches_triangular_basic_structure():
    bank = Fbank(num_filts=4, sampling_rate=16000)
    assert bank.num_filts == 4
    assert bank.sampling_rate == 16000
    assert bank.is_zero_phase is True
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert len(bank.centers_hz) == 4
    assert len(bank.supports_hz) == 4


def test_fbank_frequency_response_half_is_nonnegative():
    bank = Fbank(num_filts=4, sampling_rate=16000)
    width = 24
    half = bank.get_frequency_response(2, width, half=True)
    assert half.shape == (width // 2 + 1,)
    assert np.all(half >= 0)
    assert np.max(half) <= 1.0 + 1e-12


def test_fbank_truncated_response_reconstructs_half_spectrum():
    bank = Fbank(num_filts=4, sampling_rate=16000)
    width = 32
    bin_idx, trnc = bank.get_truncated_response(1, width)
    half_width = width // 2 + 1
    half = np.zeros(half_width, dtype=trnc.dtype)
    half[bin_idx:bin_idx + len(trnc)] = trnc
    resp_half = bank.get_frequency_response(1, width, half=True)
    np.testing.assert_allclose(half, resp_half)


def test_gabor_bank_properties_and_frequency_response_dtype():
    bank = GaborFilterBank(MelScaling(), num_filts=3, sampling_rate=16000)
    assert bank.num_filts == 3
    assert bank.sampling_rate == 16000
    assert bank.is_real is False
    assert bank.is_zero_phase is True
    resp = bank.get_frequency_response(0, 16, half=False)
    assert resp.shape == (16,)
    assert resp.dtype == np.complex128
    assert np.all(np.isfinite(resp))


def test_gabor_bank_truncated_response_reconstructs_full():
    bank = GaborFilterBank(MelScaling(), num_filts=2, sampling_rate=16000)
    width = 32
    bin_idx, trnc = bank.get_truncated_response(0, width)
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    resp = bank.get_frequency_response(0, width, half=False)
    np.testing.assert_allclose(full, resp)


def test_gabor_bank_analytic_flag_depends_on_wrap_below():
    # Use a sufficiently high low_hz to avoid negative-frequency wrap.
    bank = GaborFilterBank(MelScaling(), num_filts=2, low_hz=1000, high_hz=4000, sampling_rate=16000)
    assert bank.is_analytic is True


def test_complex_gammatone_invalid_order_raises():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), order=0)


def test_complex_gammatone_bank_basic_shapes():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=3, sampling_rate=16000, order=4)
    assert bank.num_filts == 3
    assert bank.sampling_rate == 16000
    assert bank.is_real is False
    assert bank.is_zero_phase is False
    assert bank.order == 4
    resp = bank.get_frequency_response(1, 16, half=False)
    imp = bank.get_impulse_response(1, 16)
    assert resp.shape == (16,)
    assert imp.shape == (16,)
    assert resp.dtype == np.complex128
    assert imp.dtype == np.complex128
    assert np.all(np.isfinite(resp))
    assert np.all(np.isfinite(imp))


def test_complex_gammatone_truncated_response_reconstructs_full():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=2, sampling_rate=16000, order=4)
    width = 24
    bin_idx, trnc = bank.get_truncated_response(0, width)
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    resp = bank.get_frequency_response(0, width, half=False)
    np.testing.assert_allclose(full, resp)
