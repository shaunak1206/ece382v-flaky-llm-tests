import numpy as np
import pytest
from astropy import units as u

import setigen as stg


def make_frame(fchans=64, tchans=32, df=10 * u.Hz, dt=1 * u.s, data=None):
    return stg.Frame(fchans=fchans,
                     tchans=tchans,
                     df=df,
                     dt=dt,
                     fch1=6000 * u.MHz,
                     data=data)


def test_new_frame_data_is_zeros_with_correct_shape():
    frame = make_frame(fchans=64, tchans=32)
    data = frame.get_data()
    assert data.shape == (32, 64)
    assert np.all(data == 0)


def test_get_data_returns_underlying_array_without_copy():
    frame = make_frame()
    assert frame.get_data() is frame.data
    assert frame.get_data(use_db=False) is frame.data


def test_get_data_default_matches_use_db_false():
    rng = np.random.RandomState(0)
    data = rng.uniform(1, 10, size=(8, 16))
    frame = make_frame(fchans=16, tchans=8, data=data.copy())
    np.testing.assert_array_equal(frame.get_data(), frame.get_data(use_db=False))


def test_get_data_use_db_converts_to_decibels():
    data = np.full((4, 8), 100.0)
    frame = make_frame(fchans=8, tchans=4, data=data)
    db = frame.get_data(use_db=True)
    assert db.shape == (4, 8)
    np.testing.assert_allclose(db, 20.0)


def test_get_data_use_db_does_not_modify_frame_data():
    data = np.full((4, 8), 10.0)
    frame = make_frame(fchans=8, tchans=4, data=data)
    _ = frame.get_data(use_db=True)
    np.testing.assert_array_equal(frame.data, np.full((4, 8), 10.0))


def test_get_data_use_db_returns_new_array():
    data = np.full((4, 8), 10.0)
    frame = make_frame(fchans=8, tchans=4, data=data)
    db = frame.get_data(use_db=True)
    assert db is not frame.data


def test_get_data_use_db_with_zero_gives_negative_infinity():
    frame = make_frame(fchans=8, tchans=4)
    with np.errstate(divide='ignore'):
        db = frame.get_data(use_db=True)
    assert np.all(np.isneginf(db))


def test_add_noise_chi2_returns_noise_of_frame_shape():
    np.random.seed(1)
    frame = make_frame(fchans=64, tchans=32)
    noise = frame.add_noise(x_mean=5)
    assert noise.shape == frame.shape


def test_add_noise_chi2_adds_noise_to_data():
    np.random.seed(2)
    frame = make_frame()
    noise = frame.add_noise(x_mean=5)
    np.testing.assert_allclose(frame.get_data(), noise)


def test_add_noise_chi2_is_nonnegative():
    np.random.seed(3)
    frame = make_frame()
    frame.add_noise(x_mean=5)
    assert np.all(frame.get_data() >= 0)


def test_add_noise_chi2_mean_close_to_requested_mean():
    np.random.seed(4)
    frame = make_frame(fchans=256, tchans=64)
    frame.add_noise(x_mean=10)
    assert abs(np.mean(frame.get_data()) - 10) < 0.5


def test_add_noise_chi2_sets_noise_stats_from_theory_on_empty_frame():
    np.random.seed(5)
    frame = make_frame()
    frame.add_noise(x_mean=10)
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    mean, std = frame.get_noise_stats()
    assert mean == 10
    assert std == pytest.approx(expected_std)


def test_add_noise_chi2_ignores_x_std_argument():
    np.random.seed(6)
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=1000)
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std)


def test_add_noise_gaussian_mean_and_std_close_to_requested():
    np.random.seed(7)
    frame = make_frame(fchans=256, tchans=64)
    frame.add_noise(x_mean=50, x_std=3, noise_type='gaussian')
    data = frame.get_data()
    assert abs(np.mean(data) - 50) < 0.3
    assert abs(np.std(data) - 3) < 0.3


def test_add_noise_normal_alias_sets_noise_stats_to_parameters():
    np.random.seed(8)
    frame = make_frame()
    frame.add_noise(x_mean=20, x_std=2, noise_type='normal')
    assert frame.get_noise_stats() == (20, 2)


def test_add_noise_gaussian_respects_x_min():
    np.random.seed(9)
    frame = make_frame(fchans=128, tchans=32)
    frame.add_noise(x_mean=0, x_std=5, x_min=0, noise_type='gaussian')
    data = frame.get_data()
    assert data.min() >= 0
    assert np.any(data == 0)


def test_add_noise_gaussian_without_x_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='gaussian')


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='poisson')


def test_add_noise_invalid_type_leaves_data_unchanged():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='bogus')
    assert np.all(frame.get_data() == 0)


def test_add_noise_twice_accumulates_into_data():
    np.random.seed(10)
    frame = make_frame()
    first = frame.add_noise(x_mean=5)
    second = frame.add_noise(x_mean=5)
    np.testing.assert_allclose(frame.get_data(), first + second)


def test_add_noise_second_call_updates_stats_empirically():
    np.random.seed(11)
    frame = make_frame(fchans=256, tchans=64)
    frame.add_noise(x_mean=5)
    frame.add_noise(x_mean=5)
    # Two independent noise realizations of mean 5 sum to a mean near 10
    assert abs(frame.noise_mean - 10) < 1.0
    assert frame.noise_std > 0


def test_add_noise_is_reproducible_with_same_seed():
    frame_a = make_frame()
    frame_b = make_frame()
    np.random.seed(12)
    frame_a.add_noise(x_mean=5)
    np.random.seed(12)
    frame_b.add_noise(x_mean=5)
    np.testing.assert_array_equal(frame_a.get_data(), frame_b.get_data())
