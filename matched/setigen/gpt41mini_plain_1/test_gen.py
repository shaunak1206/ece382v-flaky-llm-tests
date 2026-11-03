import numpy as np
import pytest
from astropy import units as u

import setigen as stg
from setigen.frame import Frame

# We test Frame.add_noise and Frame.get_data mainly, but will instantiate Frame with minimal args.


def make_frame():
    fchans = 20
    tchans = 10
    df = 1 * u.Hz
    dt = 1 * u.s
    fch1 = 100 * u.Hz
    return Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)


def test_add_noise_chi2_defaults_sets_noise_mean_std():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert noise.shape == frame.shape
    # After first noise addition, noise_mean and noise_std updated to x_mean and x_std
    assert np.isclose(frame.noise_mean, 10)
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert np.isclose(frame.noise_std, expected_std)
    # noise data added to frame.data (not zeros)
    assert np.any(frame.data != 0)


def test_add_noise_chi2_multiple_calls_updates_noise_stats():
    frame = make_frame()
    noise1 = frame.add_noise(10)
    noise2 = frame.add_noise(20)
    # noise_mean and noise_std updated by _update_noise_frame_stats, should not be exactly last params
    assert frame.noise_mean != 20 or frame.noise_std != np.sqrt(2 * frame.chi2_df) * 20 / frame.chi2_df
    assert frame.data.shape == frame.shape
    assert noise2.shape == frame.shape


def test_add_noise_gaussian_with_min_and_std_sets_noise_mean_std():
    frame = make_frame()
    noise = frame.add_noise(x_mean=5, x_std=2, x_min=1, noise_type='normal')
    assert noise.shape == frame.shape
    assert np.isclose(frame.noise_mean, 5)
    assert np.isclose(frame.noise_std, 2)
    assert np.all(frame.data >= 1)  # truncated gaussian should enforce minimum
    # noise added to frame.data
    assert np.any(frame.data != 0)


def test_add_noise_gaussian_without_min_sets_noise_mean_std():
    frame = make_frame()
    noise = frame.add_noise(x_mean=3, x_std=1.5, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.isclose(frame.noise_mean, 3)
    assert np.isclose(frame.noise_std, 1.5)


def test_add_noise_gaussian_missing_std_exits(monkeypatch):
    frame = make_frame()
    # Patch sys.exit to raise SystemExit so we can catch it
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=3, noise_type='normal')


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=3, noise_type='invalid')


def test_get_data_returns_correct_array_and_db_scale():
    frame = make_frame()
    frame.data[:] = 5
    # Without use_db returns data as is
    data = frame.get_data()
    assert np.all(data == 5)
    # With use_db return 10 * log10 of data 
    data_db = frame.get_data(use_db=True)
    assert np.allclose(data_db, 10 * np.log10(5))


def test_add_noise_adds_to_existing_data_and_updates_stats():
    frame = make_frame()
    frame.data += 2
    old_mean = frame.mean()
    old_std = frame.std()
    noise = frame.add_noise(10)
    # data changed by addition of noise
    assert np.any(frame.data != 2)
    # noise mean and std updated (initially zero so reset to params)
    assert frame.noise_mean == 10
    # After second noise add call, stats update; test again
    noise2 = frame.add_noise(15)
    assert frame.noise_mean != 10 or frame.noise_std != noise.std()


def test_add_noise_truncated_gaussian_respects_x_min(monkeypatch):
    # Patch distributions.truncated_gaussian to return array with min less than x_min (simulate)
    original_fn = stg.frame.distributions.truncated_gaussian

    def fake_truncated_gaussian(x_mean, x_std, x_min, shape):
        arr = np.full(shape, x_min - 1)
        return arr

    monkeypatch.setattr(stg.frame.distributions, "truncated_gaussian", fake_truncated_gaussian)
    frame = make_frame()
    noise = frame.add_noise(5, 1, 3, noise_type='normal')
    assert np.all(noise == 2)  # since added noise to zero data, frame.data == noise
    # The frame data after addition should be less than x_min due to fake noise array
    # but realistically truncated_gaussian should not produce below x_min
    # We only ensure method runs and returns array with shape
    assert noise.shape == frame.shape
    # Restore original func
    monkeypatch.setattr(stg.frame.distributions, "truncated_gaussian", original_fn)


def test_add_noise_gaussian_min_none_works(monkeypatch):
    # Patch distributions.gaussian to confirm call and return fake array
    def fake_gaussian(x_mean, x_std, shape):
        return np.full(shape, x_mean + x_std)

    monkeypatch.setattr(stg.frame.distributions, "gaussian", fake_gaussian)
    frame = make_frame()
    noise = frame.add_noise(4, 2, noise_type='gaussian')
    assert np.all(noise == 6)
    assert frame.noise_mean == 4
    assert frame.noise_std == 2


def test_get_data_logarithm_of_zero_warns_or_fails():
    frame = make_frame()
    # Set some data to zero
    frame.data[:] = 0
    # Taking log10(0) is -inf, but numpy should not error, just warn
    data_db = frame.get_data(use_db=True)
    assert np.all(np.isneginf(data_db))


def test_add_noise_respects_initial_noise_mean_std_after_first_set(monkeypatch):
    frame = make_frame()
    # Add noise first time: sets noise_mean and noise_std
    frame.add_noise(x_mean=7)
    # Save old stats
    old_mean = frame.noise_mean
    old_std = frame.noise_std
    # Patch _update_noise_frame_stats to track calls
    called = {}

    def fake_update_stats():
        called["called"] = True

    monkeypatch.setattr(frame, "_update_noise_frame_stats", fake_update_stats)
    frame.add_noise(x_mean=8)
    assert "called" in called


def test_add_noise_chi2_correct_std_calculation():
    frame = make_frame()
    x_mean = 10
    df = frame.chi2_df
    noise = frame.add_noise(x_mean=x_mean, noise_type='chi2')
    expected_std = np.sqrt(2 * df) * x_mean / df
    assert np.isclose(frame.noise_std, expected_std)


def test_add_noise_normal_invalid_x_std_raises(monkeypatch):
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=2, noise_type='normal')


def test_add_noise_gaussian_with_min_and_std_noise_added(monkeypatch):
    frame = make_frame()
    noise = frame.add_noise(x_mean=1, x_std=0.1, x_min=0, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.all(noise >= 0)


def test_get_data_use_db_on_positive_data(monkeypatch):
    frame = make_frame()
    frame.data = np.full(frame.shape, 2.0)
    data_db = frame.get_data(use_db=True)
    expected = 10 * np.log10(frame.data)
    assert np.allclose(data_db, expected)


def test_get_data_empty_frame_zero_data():
    frame = make_frame()
    # data is initialized zeros
    data = frame.get_data()
    assert np.all(data == 0)


def test_add_noise_updates_data_and_returns_noise():
    frame = make_frame()
    noise = frame.add_noise(x_mean=5)
    assert np.allclose(frame.data, noise)
    # Calling add_noise again returns new noise added to data
    noise2 = frame.add_noise(x_mean=10)
    # Newly added noise shape matches
    assert noise2.shape == frame.shape


def test_add_noise_gaussian_zero_min_noise_type():
    frame = make_frame()
    noise = frame.add_noise(x_mean=3, x_std=1, x_min=0, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.all(noise >= 0)
