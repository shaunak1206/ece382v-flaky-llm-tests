import numpy as np
import pytest
import random

import setigen.frame as frame_mod
from setigen.frame import Frame


# Helper: create a Frame with zeros of given shape and resolutions
def make_frame(tchans=10, fchans=5, df=1.0, dt=1.0, fch1=10.0):
    # Use astropy.Quantity with dimensionless units for simplicity, since Frame accepts raw float as well
    from astropy import units as u
    return Frame(fchans=fchans,
                 tchans=tchans,
                 df=df * u.Hz,
                 dt=dt * u.s,
                 fch1=fch1 * u.Hz)


def seed_all_random_generators(seed=12345):
    # Seed all RNGs that Frame.add_noise uses internally
    random.seed(seed)
    np.random.seed(seed)


def isclose_array(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol, rtol=0)


def test_add_noise_chi2_sets_noise_mean_std_on_clean_frame():
    seed_all_random_generators()
    f = make_frame()
    assert f.noise_mean == 0
    assert f.noise_std == 0
    noise = f.add_noise(x_mean=5.0, noise_type='chi2')
    # On a clean frame, noise_mean and noise_std are set to x_mean and computed std
    assert np.all(noise >= 0)  # chi2 noise non-negative
    assert abs(f.noise_mean - 5.0) < 1e-8
    expected_std = np.sqrt(2 * f.chi2_df) * 5.0 / f.chi2_df
    assert abs(f.noise_std - expected_std) < 1e-8
    # Noise shape is frame shape
    assert noise.shape == f.shape


def test_add_noise_chi2_updates_noise_stats_on_nonclean_frame():
    seed_all_random_generators()
    f = make_frame()
    f.add_noise(5.0, noise_type='chi2')  # first noise addition sets noise_mean/std
    old_mean, old_std = f.noise_mean, f.noise_std
    # Add more noise again
    noise = f.add_noise(10.0, noise_type='chi2')
    # Statistics are updated and noise_mean/std should differ from old ones
    assert noise.shape == f.shape
    m, s = f.noise_mean, f.noise_std
    assert m != old_mean
    assert s != old_std


def test_add_noise_gaussian_with_x_std_and_x_min_adds_truncated():
    seed_all_random_generators()
    f = make_frame()
    x_mean = 10.0
    x_std = 2.0
    x_min = 5.0
    noise = f.add_noise(x_mean, x_std=x_std, x_min=x_min, noise_type='gaussian')
    # Noise shape correct
    assert noise.shape == f.shape
    # All noise + data >= x_min (data+=noise, data started zero)
    assert np.all(f.data >= x_min)
    # On clean frame, noise_mean and noise_std are set accordingly
    assert abs(f.noise_mean - x_mean) < 1e-8
    assert abs(f.noise_std - x_std) < 1e-8


def test_add_noise_gaussian_with_x_std_without_x_min_adds_gaussian():
    seed_all_random_generators()
    f = make_frame()
    x_mean = 0.0
    x_std = 1.5
    noise = f.add_noise(x_mean, x_std=x_std, noise_type='normal')
    assert noise.shape == f.shape
    # Noise mean approximately x_mean (within 5 std errors due to randomness)
    assert abs(np.mean(noise) - x_mean) < 0.5
    # Frame noise_mean set to x_mean and noise_std set to x_std on clean frame
    assert abs(f.noise_mean - x_mean) < 1e-8
    assert abs(f.noise_std - x_std) < 1e-8


def test_add_noise_gaussian_without_x_std_exits(monkeypatch):
    # monkeypatch sys.exit to raise RuntimeError for test
    def fake_exit(msg=None):
        raise RuntimeError("sys.exit called")

    monkeypatch.setattr(frame_mod.sys, 'exit', fake_exit)

    f = make_frame()
    with pytest.raises(RuntimeError):
        f.add_noise(x_mean=1.0, noise_type='normal')  # no x_std provided


def test_add_noise_invalid_noise_type_exits(monkeypatch):
    def fake_exit(msg=None):
        raise RuntimeError("sys.exit called")

    monkeypatch.setattr(frame_mod.sys, 'exit', fake_exit)

    f = make_frame()
    with pytest.raises(RuntimeError):
        f.add_noise(x_mean=1.0, noise_type='invalid_type')


def test_get_data_returns_same_data_and_db_conversion():
    f = make_frame()
    # Initialize data with known values
    f.data[:] = 10.0
    # Without db returns data unchanged
    out = f.get_data(use_db=False)
    assert np.shares_memory(out, f.data) or np.array_equal(out, f.data)
    # With db returns 10*log10 of data
    expected_db = 10 * np.log10(f.data)
    actual_db = f.get_data(use_db=True)
    assert isclose_array(actual_db, expected_db)


def test_add_noise_accumulates_noise_and_updates_stats():
    seed_all_random_generators()
    f = make_frame()
    noise1 = f.add_noise(5.0, noise_type='chi2')
    # Add second batch of noise
    noise2 = f.add_noise(5.0, noise_type='chi2')
    # Data now roughly equals noise1 + noise2
    data_sum = noise1 + noise2
    # Because noise is added to data, data close to sum of noises
    assert isclose_array(f.data, data_sum, tol=0.01)


def test_add_noise_from_obs_chi2_selects_and_adds_noise(monkeypatch):
    seed_all_random_generators()
    f = make_frame()
    # Prepare arrays with dummy deterministic data
    x_means = np.array([1.0, 2.0, 3.0])
    # chi2 uses only x_mean_array, ignore x_std_array and x_min_array
    # Monkeypatch np.random.choice to always pick index 1 (value 2.0)
    monkeypatch.setattr(np.random, 'choice', lambda array: 2.0)

    noise = f.add_noise_from_obs(x_mean_array=x_means, noise_type='chi2')
    # Noise shape correct
    assert noise.shape == f.shape
    # noise mean updated
    assert abs(f.noise_mean - 2.0) < 1e-8


def test_add_noise_from_obs_gaussian_share_index(monkeypatch):
    seed_all_random_generators()
    f = make_frame()
    x_means = np.array([1.0, 2.0, 3.0])
    x_stds = np.array([0.5, 1.0, 1.5])
    x_mins = np.array([-1.0, 0.0, 0.5])
    # Monkeypatch np.random.randint to always return 2
    monkeypatch.setattr(np.random, 'randint', lambda a: 2)
    noise = f.add_noise_from_obs(x_mean_array=x_means,
                                x_std_array=x_stds,
                                x_min_array=x_mins,
                                share_index=True,
                                noise_type='normal')
    assert noise.shape == f.shape
    mean_expected = x_means[2]
    std_expected = x_stds[2]
    min_expected = x_mins[2]
    # Frame noise_mean and noise_std set to expected
    assert abs(f.noise_mean - mean_expected) < 1e-8
    assert abs(f.noise_std - std_expected) < 1e-8
    # All data >= min_expected (because truncated_gaussian)
    assert np.all(f.data >= min_expected)


def test_add_noise_from_obs_gaussian_no_share_index(monkeypatch):
    # This test exercises code path with share_index = False, where sample_from_obs.sample_gaussian_params is called.
    seed_all_random_generators()
    f = make_frame()

    x_means = np.array([1.0, 2.0])
    x_stds = np.array([0.5, 1.0])
    x_mins = np.array([0.0, -0.5])

    called = {}

    def fake_sample_gaussian_params(x_mean_array, x_std_array, x_min_array=None):
        called['called'] = True
        return 1.5, 0.75, 0.0  # fixed values

    monkeypatch.setattr(frame_mod.sample_from_obs,
                        'sample_gaussian_params',
                        fake_sample_gaussian_params)
    noise = f.add_noise_from_obs(x_mean_array=x_means,
                                x_std_array=x_stds,
                                x_min_array=x_mins,
                                share_index=False,
                                noise_type='normal')
    assert noise.shape == f.shape
    assert called.get('called') is True
    # Frame noise mean and std set to those returned by fake_sample_gaussian_params
    assert abs(f.noise_mean - 1.5) < 1e-8
    assert abs(f.noise_std - 0.75) < 1e-8
    # All noise >= 0 because x_min=0 passed to truncated_gaussian internally
    assert np.all(f.data >= 0.0)


def test_add_noise_from_obs_invalid_noise_type_exits(monkeypatch):
    def fake_exit(msg=None):
        raise RuntimeError("sys.exit called")

    monkeypatch.setattr(frame_mod.sys, 'exit', fake_exit)

    f = make_frame()
    with pytest.raises(RuntimeError):
        f.add_noise_from_obs(noise_type='invalid_type')


def test_get_data_db_with_zero_and_positive_values():
    f = make_frame()
    # Set some zero and positive values
    f.data[:] = 1.0
    f.data[0, 0] = 0.0
    # log10(0) = -inf, test that get_data does not throw, returns -inf where data=0
    result = f.get_data(use_db=True)
    assert result.shape == f.data.shape
    assert np.isneginf(result[0, 0])
    # Other values ~ 10*log10(1) = 0
    assert abs(result[0, 1]) < 1e-8


def test_add_noise_on_nonzero_data_updates_stats_correctly():
    seed_all_random_generators()
    f = make_frame()
    # Set data nonzero initially
    f.data[:] = 3.0
    f.noise_mean = 3.0
    f.noise_std = 1.0
    # Add gaussian noise
    noise = f.add_noise(5.0, x_std=1.2, noise_type='normal')
    # noise shape correct
    assert noise.shape == f.shape
    # noise_mean and noise_std updated (not equal original)
    assert f.noise_mean != 3.0 or f.noise_std != 1.0


@pytest.mark.parametrize("noise_type", ["chi2", "normal", "gaussian"])
def test_add_noise_supported_noise_types_do_not_exit(noise_type):
    seed_all_random_generators()
    f = make_frame()
    if noise_type == 'chi2':
        noise = f.add_noise(x_mean=10.0, noise_type=noise_type)
        assert noise.shape == f.shape
    else:
        noise = f.add_noise(x_mean=10.0, x_std=3.3, noise_type=noise_type)
        assert noise.shape == f.shape
