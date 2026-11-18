import numpy as np
import pytest

from pydrobert.speech.filters import (
    BartlettWindow,
    BlackmanWindow,
    ComplexGammatoneFilterBank,
    Fbank,
    GammaWindow,
    GaborFilterBank,
    HammingWindow,
    HannWindow,
    LinearFilterBank,
    TriangularOverlappingFilterBank,
)
from pydrobert.speech.scales import MelScaling


def _reconstruct_from_truncated_real(width, bin_idx, trnc, dtype=float):
    full = np.zeros(width, dtype=dtype)
    full[bin_idx:bin_idx + len(trnc)] = trnc
    if bin_idx:
        full[width - bin_idx - len(trnc) + 1: width - bin_idx + 1] = trnc[:None if bin_idx else 0:-1].conj()
    return full


def _reconstruct_from_truncated_complex(width, bin_idx, trnc):
    full = np.zeros(width, dtype=trnc.dtype)
    wrap = min(bin_idx + len(trnc), width) - bin_idx
    full[bin_idx:bin_idx + wrap] = trnc[:wrap]
    full[:len(trnc) - wrap] = trnc[wrap:]
    return full


def test_linearfilterbank_supports_ms_computes_from_supports():
    class DummyBank(LinearFilterBank):
        @property
        def is_real(self):
            return True

        @property
        def is_analytic(self):
            return False

        @property
        def is_zero_phase(self):
            return True

        @property
        def num_filts(self):
            return 2

        @property
        def sampling_rate(self):
            return 1000

        @property
        def supports_hz(self):
            return ((0.0, 1.0), (2.0, 3.0))

        @property
        def supports(self):
            return ((10, 20), (-5, 5))

        def get_impulse_response(self, filt_idx, width):
            raise NotImplementedError

        def get_frequency_response(self, filt_idx, width, half=False):
            raise NotImplementedError

        def get_truncated_response(self, filt_idx, width):
            raise NotImplementedError

    bank = DummyBank()
    assert bank.supports_ms == ((10.0, 20.0), (-5.0, 5.0))


def test_triangular_bank_basic_properties_and_supports():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=20, high_hz=8000, sampling_rate=16000)
    assert bank.num_filts == 3
    assert bank.is_real is True
    assert bank.is_analytic is False
    assert bank.is_zero_phase is True
    assert len(bank.centers_hz) == 3
    assert len(bank.supports_hz) == 3
    assert len(bank.supports) == 3
    assert bank.supports_hz[0][0] < bank.centers_hz[0] < bank.supports_hz[0][1]


def test_triangular_bank_invalid_frequency_range_raises():
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=-1, high_hz=8000, sampling_rate=16000)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=1000, high_hz=999, sampling_rate=16000)
    with pytest.raises(ValueError):
        TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=20, high_hz=9000, sampling_rate=16000)


def test_triangular_frequency_response_full_is_symmetric_for_real_bank():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=20, high_hz=4000, sampling_rate=16000)
    width = 128
    filt_idx = 1
    full = bank.get_frequency_response(filt_idx, width, half=False)
    assert full.shape == (width,)
    np.testing.assert_allclose(full, full[::-1], atol=1e-12)
    assert np.all(full >= 0)


def test_triangular_frequency_response_half_shape_even_and_odd():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=2, low_hz=50, high_hz=4000, sampling_rate=16000)
    for width, expected in [(128, 65), (127, 64)]:
        half = bank.get_frequency_response(0, width, half=True)
        assert half.shape == (expected,)
        assert half.dtype == np.float64


def test_triangular_truncated_response_reconstructs_half_spectrum():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=3, low_hz=20, high_hz=4000, sampling_rate=16000)
    width = 128
    filt_idx = 1
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    half_width = (width + width % 2) // 2 + 1 - width % 2
    half = np.zeros(half_width, dtype=trnc.dtype)
    half[bin_idx:bin_idx + len(trnc)] = trnc
    direct_half = bank.get_frequency_response(filt_idx, width, half=True)
    np.testing.assert_allclose(half, direct_half, atol=1e-12)


def test_triangular_analytic_bank_truncated_matches_full_region():
    bank = TriangularOverlappingFilterBank(
        MelScaling(), num_filts=3, low_hz=20, high_hz=4000, sampling_rate=16000, analytic=True
    )
    width = 128
    filt_idx = 1
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = bank.get_frequency_response(filt_idx, width, half=False)
    recon = np.zeros(width, dtype=trnc.dtype)
    recon[bin_idx:bin_idx + len(trnc)] = trnc
    np.testing.assert_allclose(recon, full, atol=1e-12)


def test_triangular_impulse_response_is_zero_phase_and_real_when_not_analytic():
    bank = TriangularOverlappingFilterBank(MelScaling(), num_filts=2, low_hz=50, high_hz=3000, sampling_rate=16000)
    ir = bank.get_impulse_response(0, 32)
    assert ir.shape == (32,)
    assert np.isrealobj(ir)
    np.testing.assert_allclose(ir.imag if np.iscomplexobj(ir) else 0.0, 0.0, atol=1e-12)


def test_fbank_frequency_response_and_impulse_response_shapes():
    bank = Fbank(num_filts=3, low_hz=20, high_hz=4000, sampling_rate=16000, analytic=False)
    width = 128
    fr_half = bank.get_frequency_response(0, width, half=True)
    ir = bank.get_impulse_response(0, width)
    assert fr_half.shape == (65,)
    assert ir.shape == (width,)
    assert np.isrealobj(fr_half)
    assert np.isrealobj(ir)


def test_fbank_get_truncated_response_reconstructs_half_spectrum():
    bank = Fbank(num_filts=3, low_hz=20, high_hz=4000, sampling_rate=16000, analytic=False)
    width = 128
    filt_idx = 1
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    half_width = width // 2 + 1
    half = np.zeros(half_width, dtype=trnc.dtype)
    half[bin_idx:bin_idx + len(trnc)] = trnc
    direct_half = bank.get_frequency_response(filt_idx, width, half=True)
    np.testing.assert_allclose(half, direct_half, atol=1e-12)


def test_fbank_analytic_impulse_response_uses_complex_ifft():
    bank = Fbank(num_filts=2, low_hz=20, high_hz=3000, sampling_rate=16000, analytic=True)
    ir = bank.get_impulse_response(0, 64)
    assert ir.shape == (64,)
    assert np.iscomplexobj(ir)
    full = bank.get_frequency_response(0, 64, half=False)
    assert full.shape == (64,)


def test_gabor_bank_basic_properties_and_analytic_flag():
    bank = GaborFilterBank(MelScaling(), num_filts=3, low_hz=20, high_hz=4000, sampling_rate=16000)
    assert bank.is_real is False
    assert bank.is_zero_phase is True
    assert bank.num_filts == 3
    assert bank.sampling_rate == 16000
    assert len(bank.centers_hz) == 3
    assert bank.is_analytic in (True, False)


def test_gabor_frequency_response_shape_and_nonnegativity():
    bank = GaborFilterBank(MelScaling(), num_filts=3, low_hz=20, high_hz=4000, sampling_rate=16000)
    width = 128
    fr = bank.get_frequency_response(0, width, half=False)
    fr_half = bank.get_frequency_response(0, width, half=True)
    assert fr.shape == (width,)
    assert fr_half.shape == (65,)
    assert np.all(np.isfinite(fr))
    assert np.all(fr >= 0)


def test_gabor_truncated_response_reconstructs_full_spectrum():
    bank = GaborFilterBank(MelScaling(), num_filts=3, low_hz=20, high_hz=4000, sampling_rate=16000)
    width = 128
    filt_idx = 1
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = bank.get_frequency_response(filt_idx, width, half=False)
    recon = _reconstruct_from_truncated_complex(width, bin_idx, trnc)
    np.testing.assert_allclose(recon, full, atol=1e-10, rtol=1e-10)


def test_complex_gammatone_invalid_order_raises():
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), num_filts=2, low_hz=20, high_hz=4000, sampling_rate=16000, order=0)
    with pytest.raises(ValueError):
        ComplexGammatoneFilterBank(MelScaling(), num_filts=2, low_hz=20, high_hz=4000, sampling_rate=16000, order=1.5)


def test_complex_gammatone_properties_and_shapes():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=2, low_hz=20, high_hz=4000, sampling_rate=16000, order=4)
    assert bank.is_real is False
    assert bank.is_zero_phase is False
    assert bank.order == 4
    assert bank.num_filts == 2
    assert np.iscomplexobj(bank.get_impulse_response(0, 64))
    assert np.iscomplexobj(bank.get_frequency_response(0, 64, half=False))


def test_complex_gammatone_truncated_matches_frequency_response_when_not_wrapping():
    bank = ComplexGammatoneFilterBank(MelScaling(), num_filts=2, low_hz=20, high_hz=4000, sampling_rate=16000, order=4)
    width = 128
    filt_idx = 0
    bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
    full = bank.get_frequency_response(filt_idx, width, half=False)
    recon = _reconstruct_from_truncated_complex(width, bin_idx, trnc)
    np.testing.assert_allclose(recon, full, atol=1e-10, rtol=1e-10)


def test_window_functions_are_symmetric_and_correct_length():
    windows = [BartlettWindow(), BlackmanWindow(), HammingWindow(), HannWindow()]
    for win in windows:
        arr = win.get_impulse_response(9)
        assert arr.shape == (9,)
        np.testing.assert_allclose(arr, arr[::-1], atol=1e-12)
        assert np.isfinite(arr).all()


def test_gamma_window_edge_cases_and_peak_shape():
    gw = GammaWindow(order=4, peak=0.75)
    assert gw.get_impulse_response(0).shape == (0,)
    np.testing.assert_allclose(gw.get_impulse_response(1), np.array([1.0]))
    arr = gw.get_impulse_response(12)
    assert arr.shape == (12,)
    assert np.argmax(arr) <= 11
    assert np.all(arr >= 0)
