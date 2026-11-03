import numpy as np
import pytest
from astropy import units as u

import setigen as stg


def make_frame(fchans=64, tchans=32, df=10 * u.Hz, dt=10 * u.s):
    # df * dt = 100, so chi2_df = 400
    return stg.Frame(fchans=fchans, tchans=tchans, df=df, dt=dt)


def test_add_noise_chi2_returns_array_of_frame_shape():
    np.random.seed(0)
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == frame.shape


def test_add_noise_chi2_on_empty_frame_data_equals_noise():
    np.random.seed(1)
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.data, noise)


def test_add_noise_chi2_is_nonnegative():
    np.random.seed(2)
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.all(noise >= 0)


def test_add_noise_chi2_mean_close_to_x_mean():
    np.random.seed(3)
    frame = make_frame(fchans=128, tchans=64)
    noise = frame.add_noise(x_mean=10)
    assert np.mean(noise) == pytest.approx(10, rel=0.1)


def test_add_noise_chi2_sets_noise_stats_from_params_on_empty_frame():
    np.random.seed(4)
    frame = make_frame()
    assert frame.chi2_df == 400
    frame.add_noise(x_mean=10)
    expected_std = np.sqrt(2 * 400) * 10 / 400
    mean, std = frame.get_noise_stats()
    assert mean == 10
    assert std == pytest.approx(expected_std)


def test_add_noise_chi2_ignores_x_std_argument():
    np.random.seed(5)
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=1000)
    _, std = frame.get_noise_stats()
    assert std == pytest.approx(np.sqrt(2 * 400) * 10 / 400)


def test_add_noise_gaussian_mean_and_std_close_to_params():
    np.random.seed(6)
    frame = make_frame(fchans=128, tchans=64)
    noise = frame.add_noise(x_mean=5, x_std=2, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.mean(noise) == pytest.approx(5, abs=0.3)
    assert np.std(noise) == pytest.approx(2, rel=0.2)


def test_add_noise_normal_alias_matches_gaussian_with_same_seed():
    frame_a = make_frame()
    frame_b = make_frame()
    np.random.seed(7)
    noise_a = frame_a.add_noise(x_mean=5, x_std=2, noise_type='normal')
    np.random.seed(7)
    noise_b = frame_b.add_noise(x_mean=5, x_std=2, noise_type='gaussian')
    np.testing.assert_array_equal(noise_a, noise_b)
    np.testing.assert_array_equal(frame_a.data, frame_b.data)


def test_add_noise_gaussian_sets_noise_stats_from_params_on_empty_frame():
    np.random.seed(8)
    frame = make_frame()
    frame.add_noise(x_mean=5, x_std=2, noise_type='gaussian')
    assert frame.get_noise_stats() == (5, 2)


def test_add_noise_gaussian_with_min_enforces_lower_bound():
    np.random.seed(9)
    frame = make_frame(fchans=128, tchans=64)
    noise = frame.add_noise(x_mean=0, x_std=5, x_min=1, noise_type='gaussian')
    assert np.min(noise) >= 1
    assert np.min(frame.data) >= 1
    # With mean 0 and std 5 many samples should have been clipped to the minimum
    assert np.any(noise == 1)


def test_add_noise_gaussian_without_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='gaussian')


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, x_std=1, noise_type='bogus')


def test_add_noise_accumulates_in_frame_data():
    np.random.seed(10)
    frame = make_frame()
    noise1 = frame.add_noise(x_mean=10)
    noise2 = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.data, noise1 + noise2)


def test_add_noise_second_call_updates_noise_stats_from_data():
    np.random.seed(11)
    frame = make_frame(fchans=128, tchans=64)
    frame.add_noise(x_mean=10)
    first_mean, _ = frame.get_noise_stats()
    frame.add_noise(x_mean=10)
    second_mean, second_std = frame.get_noise_stats()
    # Stats are now estimated from the data, which is roughly twice as large
    assert first_mean == 10
    assert second_mean == pytest.approx(20, rel=0.1)
    assert second_std > 0


def test_add_noise_after_zero_data_resets_stats_to_params():
    np.random.seed(12)
    frame = make_frame()
    frame.add_noise(x_mean=10)
    frame.zero_data()
    assert frame.get_noise_stats() == (0, 0)
    frame.add_noise(x_mean=3, x_std=1, noise_type='gaussian')
    assert frame.get_noise_stats() == (3, 1)


def test_get_data_returns_underlying_data_array():
    frame = make_frame()
    assert frame.get_data() is frame.data
    assert frame.get_data().shape == frame.shape


def test_get_data_default_is_zeros_for_new_frame():
    frame = make_frame()
    np.testing.assert_array_equal(frame.get_data(), np.zeros((32, 64)))


def test_get_data_reflects_added_noise():
    np.random.seed(13)
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data(), noise)


def test_get_data_use_db_converts_to_decibels():
    data = np.full((4, 8), 100.0)
    frame = stg.Frame.from_data(df=1 * u.Hz, dt=1 * u.s, fch1=1 * u.GHz, data=data)
    db = frame.get_data(use_db=True)
    np.testing.assert_allclose(db, np.full((4, 8), 20.0))


def test_get_data_use_db_does_not_modify_stored_data():
    data = np.full((4, 8), 10.0)
    frame = stg.Frame.from_data(df=1 * u.Hz, dt=1 * u.s, fch1=1 * u.GHz, data=data)
    db = frame.get_data(use_db=True)
    assert db is not frame.data
    np.testing.assert_allclose(frame.get_data(), np.full((4, 8), 10.0))
    np.testing.assert_allclose(db, np.full((4, 8), 10.0))


def test_get_data_use_db_of_zeros_is_negative_infinity():
    frame = make_frame()
    with np.errstate(divide='ignore'):
        db = frame.get_data(use_db=True)
    assert db.shape == frame.shape
    assert np.all(np.isneginf(db))
