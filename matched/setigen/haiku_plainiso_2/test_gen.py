"""Unit tests for Frame.add_noise and Frame.get_data in setigen/frame.py."""
import math

import numpy as np
import pytest
from astropy import units as u

from setigen.frame import Frame


# With df = 1 Hz and dt = 1 s, chi2_df = 4 * round(1 * 1) = 4.
CHI2_DF = 4


def make_frame(fchans=64, tchans=32, data=None):
    return Frame(fchans=fchans,
                 tchans=tchans,
                 df=1.0 * u.Hz,
                 dt=1.0 * u.s,
                 fch1=8 * u.GHz,
                 data=data)


def expected_chi2(x_mean, shape):
    return np.random.chisquare(df=CHI2_DF, size=shape) * x_mean / CHI2_DF


def test_add_noise_returns_array_with_frame_shape():
    f = make_frame(fchans=64, tchans=32)
    noise = f.add_noise(x_mean=5.0)
    assert noise.shape == (32, 64)
    assert f.get_data().shape == (32, 64)


def test_add_noise_default_type_is_chi2():
    f = make_frame()
    np.random.seed(5)
    noise = f.add_noise(x_mean=5.0)
    np.random.seed(5)
    np.testing.assert_allclose(noise, expected_chi2(5.0, (32, 64)))


def test_add_noise_chi2_sample_mean_near_x_mean():
    f = make_frame(fchans=100, tchans=100)
    np.random.seed(0)
    noise = f.add_noise(x_mean=10.0, noise_type='chi2')
    assert abs(np.mean(noise) - 10.0) < 0.5


def test_add_noise_chi2_sample_std_matches_theory():
    f = make_frame(fchans=100, tchans=100)
    np.random.seed(1)
    noise = f.add_noise(x_mean=10.0)
    theory = math.sqrt(2 * CHI2_DF) * 10.0 / CHI2_DF
    assert np.std(noise) == pytest.approx(theory, rel=0.05)


def test_add_noise_chi2_values_are_nonnegative():
    f = make_frame(fchans=100, tchans=100)
    np.random.seed(2)
    noise = f.add_noise(x_mean=3.0)
    assert np.all(noise >= 0)


def test_add_noise_chi2_scales_linearly_with_x_mean():
    f1 = make_frame()
    f2 = make_frame()
    np.random.seed(3)
    n1 = f1.add_noise(x_mean=3.0)
    np.random.seed(3)
    n2 = f2.add_noise(x_mean=6.0)
    np.testing.assert_allclose(n2, 2 * n1)


def test_add_noise_chi2_ignores_x_std_and_x_min():
    f = make_frame()
    np.random.seed(4)
    noise = f.add_noise(x_mean=4.0, x_std=999.0, x_min=1e6, noise_type='chi2')
    # x_min must not truncate chi2 noise, so every value stays far below 1e6.
    assert noise.max() < 1e6
    # Stored std is the chi2-theoretical one, not the supplied x_std.
    _, stored_std = f.get_noise_stats()
    assert stored_std == pytest.approx(math.sqrt(2 * CHI2_DF) * 4.0 / CHI2_DF)


def test_add_noise_chi2_sets_noise_stats_on_first_call():
    f = make_frame()
    f.add_noise(x_mean=4.0)
    mean, std = f.get_noise_stats()
    assert mean == pytest.approx(4.0)
    assert std == pytest.approx(math.sqrt(2 * CHI2_DF) * 4.0 / CHI2_DF)


def test_add_noise_chi2_zero_mean_gives_zero_noise():
    f = make_frame()
    noise = f.add_noise(x_mean=0.0)
    assert np.all(noise == 0)
    assert np.all(f.get_data() == 0)


def test_add_noise_normal_matches_numpy_draws():
    f = make_frame()
    np.random.seed(6)
    noise = f.add_noise(x_mean=5.0, x_std=2.0, noise_type='normal')
    np.random.seed(6)
    np.testing.assert_allclose(noise, np.random.normal(5.0, 2.0, (32, 64)))


def test_add_noise_gaussian_alias_matches_normal():
    f1 = make_frame()
    f2 = make_frame()
    np.random.seed(8)
    n1 = f1.add_noise(x_mean=1.0, x_std=0.5, noise_type='normal')
    np.random.seed(8)
    n2 = f2.add_noise(x_mean=1.0, x_std=0.5, noise_type='gaussian')
    np.testing.assert_array_equal(n1, n2)


def test_add_noise_normal_without_x_std_exits():
    f = make_frame()
    with pytest.raises(SystemExit):
        f.add_noise(x_mean=1.0, noise_type='normal')


def test_add_noise_invalid_type_exits_and_leaves_data_untouched():
    f = make_frame()
    with pytest.raises(SystemExit):
        f.add_noise(x_mean=1.0, noise_type='poisson')
    assert np.all(f.get_data() == 0)


def test_add_noise_normal_with_x_min_is_truncated():
    f = make_frame(fchans=64, tchans=32)
    np.random.seed(9)
    noise = f.add_noise(x_mean=0.0, x_std=1.0, x_min=0.0, noise_type='normal')
    assert np.all(noise >= 0.0)
    assert np.any(noise == 0.0)


def test_add_noise_normal_without_x_min_allows_negative_values():
    f = make_frame()
    np.random.seed(10)
    noise = f.add_noise(x_mean=0.0, x_std=1.0, noise_type='normal')
    assert np.any(noise < 0.0)


def test_add_noise_normal_sets_noise_stats_on_first_call():
    f = make_frame()
    f.add_noise(x_mean=5.0, x_std=2.0, x_min=1.0, noise_type='normal')
    mean, std = f.get_noise_stats()
    assert mean == pytest.approx(5.0)
    assert std == pytest.approx(2.0)


def test_add_noise_second_call_recomputes_noise_stats_from_data():
    f = make_frame(fchans=100, tchans=100)
    np.random.seed(11)
    f.add_noise(x_mean=0.0, x_std=1.0, noise_type='normal')
    first_mean, first_std = f.get_noise_stats()
    assert first_mean == pytest.approx(0.0)
    assert first_std == pytest.approx(1.0)

    # Second call: stats come from the sample of the accumulated data,
    # whose std is sqrt(2) for two independent unit normals.
    np.random.seed(12)
    f.add_noise(x_mean=0.0, x_std=1.0, noise_type='normal')
    mean, std = f.get_noise_stats()
    assert abs(mean) < 0.1
    assert abs(std - math.sqrt(2)) < 0.1


def test_add_noise_accumulates_onto_preloaded_data():
    base = np.full((32, 64), 2.0)
    f = make_frame(data=base.copy())
    np.random.seed(13)
    noise = f.add_noise(x_mean=3.0)
    np.testing.assert_allclose(f.get_data(), base + noise)


def test_get_data_default_returns_stored_array():
    arr = np.array([[1.0, 10.0], [100.0, 1000.0]])
    f = make_frame(fchans=2, tchans=2, data=arr)
    assert f.get_data() is f.data
    np.testing.assert_array_equal(f.get_data(), arr)


def test_get_data_use_db_is_ten_log10_of_data():
    arr = np.array([[1.0, 10.0], [100.0, 1000.0]])
    f = make_frame(fchans=2, tchans=2, data=arr)
    np.testing.assert_allclose(f.get_data(use_db=True),
                               [[0.0, 10.0], [20.0, 30.0]])
    # Requesting dB must not modify the stored linear data.
    np.testing.assert_array_equal(f.get_data(), arr)


def test_get_data_use_db_random_positive_data():
    rng = np.random.RandomState(13)
    arr = rng.uniform(0.1, 50.0, size=(16, 8))
    f = make_frame(fchans=8, tchans=16, data=arr)
    np.testing.assert_allclose(f.get_data(use_db=True), 10 * np.log10(arr))


def test_get_data_use_db_zero_is_negative_infinity():
    f = make_frame(fchans=4, tchans=4)
    with np.errstate(divide='ignore'):
        db = f.get_data(use_db=True)
    assert np.all(np.isneginf(db))


def test_get_data_reflects_add_noise_result():
    f = make_frame()
    np.random.seed(14)
    noise = f.add_noise(x_mean=5.0)
    np.testing.assert_array_equal(f.get_data(), noise)
