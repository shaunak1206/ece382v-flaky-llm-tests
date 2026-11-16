import numpy as np
import pytest
from astropy import units as u

import setigen as stg


FCHANS = 256
TCHANS = 64


def make_frame(fchans=FCHANS, tchans=TCHANS, df=10 * u.Hz, dt=10 * u.s, data=None):
    # df * dt = 100 -> chi2_df = 400
    return stg.Frame(fchans=fchans,
                     tchans=tchans,
                     df=df,
                     dt=dt,
                     fch1=6 * u.GHz,
                     data=data)


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(12345)


# ---------------------- add_noise ----------------------

def test_add_noise_chi2_returns_noise_of_frame_shape():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == (TCHANS, FCHANS)


def test_add_noise_chi2_on_zero_frame_data_equals_noise():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.data, noise)


def test_add_noise_chi2_mean_close_to_x_mean_and_positive():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.all(noise >= 0)
    assert abs(np.mean(noise) - 10) < 0.2


def test_add_noise_chi2_sets_noise_stats_to_theoretical_values():
    frame = make_frame()
    assert frame.chi2_df == 400
    frame.add_noise(x_mean=10)
    expected_std = np.sqrt(2 * 400) * 10 / 400
    mean, std = frame.get_noise_stats()
    assert mean == pytest.approx(10)
    assert std == pytest.approx(expected_std)


def test_add_noise_chi2_ignores_x_std_argument():
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=1000)
    assert frame.noise_std == pytest.approx(np.sqrt(2 * 400) * 10 / 400)


def test_add_noise_gaussian_stats_and_distribution():
    frame = make_frame()
    noise = frame.add_noise(x_mean=5, x_std=2, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert frame.noise_mean == 5
    assert frame.noise_std == 2
    assert abs(np.mean(noise) - 5) < 0.1
    assert abs(np.std(noise) - 2) < 0.1


def test_add_noise_normal_alias_works():
    frame = make_frame()
    noise = frame.add_noise(x_mean=3, x_std=1, noise_type='normal')
    assert noise.shape == frame.shape
    assert abs(np.mean(noise) - 3) < 0.1
    assert frame.get_noise_stats() == (3, 1)


def test_add_noise_gaussian_without_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='gaussian')
    # Data should be untouched
    assert np.all(frame.data == 0)


def test_add_noise_invalid_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, x_std=1, noise_type='uniform')
    assert np.all(frame.data == 0)


def test_add_noise_truncated_gaussian_respects_minimum():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10, x_std=5, x_min=5, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.min(noise) >= 5 - 1e-9
    assert np.min(frame.data) >= 5 - 1e-9


def test_add_noise_accumulates_in_data():
    frame = make_frame()
    n1 = frame.add_noise(x_mean=10)
    n2 = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.data, n1 + n2)


def test_add_noise_second_call_updates_stats_from_data():
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=1, noise_type='gaussian')
    frame.add_noise(x_mean=10, x_std=1, noise_type='gaussian')
    # Stats are now estimated from clipped data (~mean 20, std ~sqrt(2))
    assert abs(frame.noise_mean - 20) < 0.2
    assert abs(frame.noise_std - np.sqrt(2)) < 0.2


def test_add_noise_adds_to_existing_data():
    base = np.full((TCHANS, FCHANS), 100.0)
    frame = make_frame(data=base.copy())
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.data, 100.0 + noise)


# ---------------------- get_data ----------------------

def test_get_data_returns_underlying_array():
    frame = make_frame()
    assert frame.get_data() is frame.data
    assert frame.get_data().shape == (TCHANS, FCHANS)


def test_get_data_default_is_not_db():
    data = np.full((4, 8), 100.0)
    frame = make_frame(fchans=8, tchans=4, data=data)
    np.testing.assert_array_equal(frame.get_data(), data)
    np.testing.assert_array_equal(frame.get_data(use_db=False), data)


def test_get_data_use_db_converts_to_decibels():
    data = np.tile(np.array([1.0, 10.0, 100.0, 1000.0]), (2, 1))
    frame = make_frame(fchans=4, tchans=2, data=data)
    expected = np.tile(np.array([0.0, 10.0, 20.0, 30.0]), (2, 1))
    np.testing.assert_allclose(frame.get_data(use_db=True), expected)


def test_get_data_use_db_does_not_modify_frame_data():
    data = np.full((4, 8), 10.0)
    frame = make_frame(fchans=8, tchans=4, data=data)
    db = frame.get_data(use_db=True)
    assert db is not frame.data
    np.testing.assert_array_equal(frame.data, np.full((4, 8), 10.0))


def test_get_data_use_db_zero_gives_negative_infinity():
    frame = make_frame(fchans=8, tchans=4)
    with np.errstate(divide='ignore'):
        db = frame.get_data(use_db=True)
    assert db.shape == (4, 8)
    assert np.all(np.isneginf(db))


def test_get_data_use_db_negative_values_give_nan():
    data = np.full((2, 4), -1.0)
    frame = make_frame(fchans=4, tchans=2, data=data)
    with np.errstate(invalid='ignore'):
        db = frame.get_data(use_db=True)
    assert np.all(np.isnan(db))


def test_get_data_reflects_added_noise():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data(), noise)
    np.testing.assert_allclose(frame.get_data(use_db=True), 10 * np.log10(noise))
