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


def test_add_noise_returns_array_with_frame_shape():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert isinstance(noise, np.ndarray)
    assert noise.shape == frame.shape == (16, 64)


def test_add_noise_to_zero_frame_data_equals_noise():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data(), noise)


def test_add_noise_chi2_mean_close_to_requested_mean():
    frame = make_frame(fchans=256, tchans=256)
    noise = frame.add_noise(x_mean=10, noise_type='chi2')
    assert np.mean(noise) == pytest.approx(10, rel=0.02)


def test_add_noise_chi2_values_are_nonnegative():
    frame = make_frame()
    noise = frame.add_noise(x_mean=5)
    assert np.all(noise >= 0)


def test_add_noise_chi2_sets_noise_stats_from_ideal_distribution():
    frame = make_frame()
    assert frame.get_noise_stats() == (0, 0)
    frame.add_noise(x_mean=10)
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    mean, std = frame.get_noise_stats()
    assert mean == 10
    assert std == pytest.approx(expected_std)


def test_add_noise_chi2_ignores_x_std_argument():
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=1000, noise_type='chi2')
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std)


def test_add_noise_gaussian_mean_and_std():
    frame = make_frame(fchans=256, tchans=256)
    noise = frame.add_noise(x_mean=20, x_std=3, noise_type='gaussian')
    assert np.mean(noise) == pytest.approx(20, rel=0.02)
    assert np.std(noise) == pytest.approx(3, rel=0.05)


def test_add_noise_normal_alias_sets_noise_stats_to_params():
    frame = make_frame()
    noise = frame.add_noise(x_mean=20, x_std=3, noise_type='normal')
    assert noise.shape == frame.shape
    assert frame.get_noise_stats() == (20, 3)


def test_add_noise_gaussian_without_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='gaussian')


def test_add_noise_invalid_noise_type_exits_and_leaves_data_unchanged():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='bogus')
    assert np.all(frame.get_data() == 0)


def test_add_noise_truncated_gaussian_respects_minimum():
    frame = make_frame(fchans=128, tchans=64)
    noise = frame.add_noise(x_mean=5, x_std=3, x_min=2, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.all(noise >= 2)


def test_add_noise_accumulates_onto_existing_data():
    initial = np.ones((16, 64)) * 100
    frame = make_frame(data=initial.copy())
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data(), initial + noise)


def test_add_noise_twice_sums_both_noise_arrays():
    frame = make_frame()
    n1 = frame.add_noise(x_mean=10)
    n2 = frame.add_noise(x_mean=5)
    np.testing.assert_allclose(frame.get_data(), n1 + n2)


def test_add_noise_second_call_updates_noise_stats_from_data():
    frame = make_frame(fchans=128, tchans=128)
    frame.add_noise(x_mean=10)
    frame.add_noise(x_mean=10)
    mean, std = frame.get_noise_stats()
    # Mean of the combined noise is ~20, estimated via sigma clipping
    assert mean == pytest.approx(20, rel=0.05)
    assert std > 0


def test_get_data_returns_underlying_array_by_default():
    frame = make_frame()
    assert frame.get_data() is frame.data
    assert frame.get_data(use_db=False) is frame.data


def test_get_data_shape_and_initial_zeros():
    frame = make_frame(fchans=32, tchans=8)
    data = frame.get_data()
    assert data.shape == (8, 32)
    assert np.all(data == 0)


def test_get_data_use_db_converts_to_decibels():
    data = np.full((4, 8), 100.0)
    data[0, 0] = 10.0
    frame = make_frame(fchans=8, tchans=4, data=data.copy())
    db = frame.get_data(use_db=True)
    np.testing.assert_allclose(db, 10 * np.log10(data))
    assert db[0, 0] == pytest.approx(10.0)
    assert db[1, 1] == pytest.approx(20.0)


def test_get_data_use_db_does_not_modify_frame_data():
    data = np.full((4, 8), 100.0)
    frame = make_frame(fchans=8, tchans=4, data=data.copy())
    _ = frame.get_data(use_db=True)
    np.testing.assert_allclose(frame.get_data(), data)


def test_get_data_use_db_of_zeros_is_negative_infinity():
    frame = make_frame()
    with np.errstate(divide='ignore'):
        db = frame.get_data(use_db=True)
    assert np.all(np.isneginf(db))


def test_get_data_reflects_added_noise_in_db():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data(use_db=True),
                               10 * np.log10(noise))


def test_get_data_from_data_constructor_returns_given_values():
    data = np.arange(1, 13, dtype=float).reshape(3, 4)
    frame = stg.Frame.from_data(df=1 * u.Hz, dt=10 * u.s,
                                fch1=6000 * u.MHz, data=data)
    np.testing.assert_array_equal(frame.get_data(), data)
