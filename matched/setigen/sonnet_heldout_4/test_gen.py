import numpy as np
import pytest
from astropy import units as u

import setigen as stg


def make_frame(fchans=64, tchans=16, df=1.0, dt=10.0, data=None):
    return stg.Frame(fchans=fchans,
                     tchans=tchans,
                     df=df * u.Hz,
                     dt=dt * u.s,
                     fch1=6000 * u.MHz,
                     data=data)


# ---------------------------- add_noise: chi2 ----------------------------

def test_add_noise_chi2_returns_noise_with_frame_shape():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == frame.shape


def test_add_noise_chi2_data_equals_noise_on_zero_frame():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.data, noise)


def test_add_noise_chi2_values_nonnegative():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.all(noise >= 0)


def test_add_noise_chi2_sample_mean_close_to_x_mean():
    np.random.seed(1234)
    frame = make_frame(fchans=128, tchans=32)
    noise = frame.add_noise(x_mean=10)
    assert abs(np.mean(noise) - 10) < 0.5


def test_add_noise_chi2_sets_noise_stats_from_parameters_on_empty_frame():
    frame = make_frame()
    frame.add_noise(x_mean=10)
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert frame.noise_mean == 10
    assert frame.noise_std == pytest.approx(expected_std)


def test_add_noise_chi2_ignores_x_std_argument():
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=1000.0, noise_type='chi2')
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std)


def test_add_noise_twice_accumulates_into_data():
    np.random.seed(0)
    frame = make_frame()
    n1 = frame.add_noise(x_mean=10)
    n2 = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.data, n1 + n2)


def test_add_noise_second_call_updates_stats_from_data():
    np.random.seed(5)
    frame = make_frame(fchans=128, tchans=32)
    frame.add_noise(x_mean=10)
    frame.add_noise(x_mean=10)
    # Stats now come from sigma-clipped data (~ mean 20), not the parameters
    assert abs(frame.noise_mean - 20) < 1.5
    assert frame.noise_std > 0


def test_add_noise_after_zero_data_sets_parameters_again():
    frame = make_frame()
    frame.add_noise(x_mean=10)
    frame.zero_data()
    assert frame.get_noise_stats() == (0, 0)
    frame.add_noise(x_mean=3)
    assert frame.noise_mean == 3


# -------------------------- add_noise: gaussian --------------------------

@pytest.mark.parametrize('noise_type', ['normal', 'gaussian'])
def test_add_noise_gaussian_types_statistics(noise_type):
    np.random.seed(42)
    frame = make_frame(fchans=128, tchans=32)
    noise = frame.add_noise(x_mean=100, x_std=5, noise_type=noise_type)
    assert noise.shape == frame.shape
    assert abs(np.mean(noise) - 100) < 1
    assert abs(np.std(noise) - 5) < 1
    assert frame.noise_mean == 100
    assert frame.noise_std == 5


def test_add_noise_gaussian_with_min_respects_lower_bound():
    np.random.seed(7)
    frame = make_frame()
    noise = frame.add_noise(x_mean=5, x_std=3, x_min=2, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.all(noise >= 2 - 1e-9)
    np.testing.assert_allclose(frame.data, noise)


def test_add_noise_gaussian_without_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='gaussian')


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, x_std=1, noise_type='uniform')


# ------------------------------- get_data --------------------------------

def test_get_data_returns_underlying_data_array():
    data = np.arange(12, dtype=float).reshape(3, 4) + 1
    frame = make_frame(fchans=4, tchans=3, data=data)
    out = frame.get_data()
    assert out is frame.data
    np.testing.assert_array_equal(out, data)


def test_get_data_use_db_converts_to_decibels_without_mutating():
    data = np.full((3, 4), 100.0)
    data[0, 0] = 10.0
    frame = make_frame(fchans=4, tchans=3, data=data.copy())
    out = frame.get_data(use_db=True)
    expected = 10 * np.log10(data)
    np.testing.assert_allclose(out, expected)
    assert out[0, 0] == pytest.approx(10.0)
    assert out[1, 1] == pytest.approx(20.0)
    # Original data remains linear
    np.testing.assert_array_equal(frame.data, data)


def test_get_data_reflects_added_noise():
    frame = make_frame()
    assert np.all(frame.get_data() == 0)
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data(), noise)
    db = frame.get_data(use_db=True)
    assert db.shape == frame.shape
    assert np.all(np.isfinite(db[noise > 0]))
