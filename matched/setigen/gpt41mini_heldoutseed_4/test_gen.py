import pytest
import numpy as np
from astropy import units as u
from setigen.frame import Frame
from setigen import distributions

@pytest.fixture
def basic_frame():
    fchans = 10
    tchans = 5
    df = 1 * u.Hz
    dt = 1 * u.s
    fch1 = 20 * u.Hz
    return Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)

@pytest.fixture
def frame_with_noise(basic_frame):
    basic_frame.add_noise(x_mean=5)
    return basic_frame

def test_add_noise_chi2_noise_added_and_stats_updated(basic_frame):
    noise = basic_frame.add_noise(x_mean=10)
    # Noise array shape matches frame shape
    assert noise.shape == basic_frame.shape
    # Data should be all zeros plus noise => data equals noise
    np.testing.assert_allclose(basic_frame.data, noise)
    # noise_mean and noise_std set to parameters on first call
    assert basic_frame.noise_mean == 10
    assert basic_frame.noise_std > 0

def test_add_noise_chi2_multiple_calls_accumulate_noise_and_stats_update(basic_frame):
    noise1 = basic_frame.add_noise(8)
    old_data = basic_frame.data.copy()
    # On second noise add, noise stats update with sigma clipping
    noise2 = basic_frame.add_noise(10)
    assert noise2.shape == basic_frame.shape
    # Data should have grown larger than before
    assert np.any(basic_frame.data != old_data)
    # noise_mean and noise_std should not be exactly equal to inputs after second call
    assert not (basic_frame.noise_mean == 10 and basic_frame.noise_std == 10)

def test_add_noise_normal_requires_x_std_and_adds_noise(basic_frame):
    noise = basic_frame.add_noise(x_mean=3, x_std=1.5, noise_type='normal')
    assert noise.shape == basic_frame.shape
    # Data has noise added
    np.testing.assert_allclose(basic_frame.data, noise)
    assert basic_frame.noise_mean == 3
    assert np.isclose(basic_frame.noise_std, 1.5)

def test_add_noise_normal_with_x_min_truncates(basic_frame):
    noise = basic_frame.add_noise(x_mean=2, x_std=0.5, x_min=1.0, noise_type='normal')
    assert noise.shape == basic_frame.shape
    # All data >= x_min (because truncated)
    assert np.all(noise >= 1.0)
    assert basic_frame.noise_mean == 2

def test_add_noise_normal_without_x_std_fails(basic_frame):
    with pytest.raises(SystemExit):
        basic_frame.add_noise(x_mean=5, noise_type='normal')

def test_add_noise_invalid_noise_type_fails(basic_frame):
    with pytest.raises(SystemExit):
        basic_frame.add_noise(x_mean=5, noise_type='invalid')

def test_get_data_returns_correct_array_and_with_dB(basic_frame):
    # Initially data zeros
    data = basic_frame.get_data()
    assert np.all(data == 0.0)
    # Add some noise first
    basic_frame.add_noise(5)
    dB = basic_frame.get_data(use_db=True)
    # dB is 10*log10 of data
    expected_db = 10 * np.log10(basic_frame.data)
    np.testing.assert_allclose(dB, expected_db)

def test_get_data_with_zero_or_negative_data_use_db_handles_warnings(basic_frame):
    basic_frame.data[:] = -1.0  # negative data; log10 will warn but not fail
    # Should not raise, but produce -inf or nan values
    result = basic_frame.get_data(use_db=True)
    assert result.shape == basic_frame.shape

def test_add_noise_from_obs_defaults_loads_and_adds_noise(monkeypatch, basic_frame):
    # Patch np.load to return dummy noise param array
    dummy_params = np.array([[2.0, 0.5, 1.0], [3.0, 1.0, 0.5]])
    monkeypatch.setattr(np, "load", lambda path: dummy_params)
    noise = basic_frame.add_noise_from_obs()
    assert noise.shape == basic_frame.shape
    # Noise added to frame data
    assert np.any(basic_frame.data != 0)

def test_add_noise_from_obs_chi2_with_arrays(monkeypatch, basic_frame):
    arr = np.array([1.0, 2.0, 3.0])
    noise = basic_frame.add_noise_from_obs(x_mean_array=arr, noise_type='chi2')
    assert noise.shape == basic_frame.shape

def test_add_noise_from_obs_normal_share_index_equal_length(monkeypatch, basic_frame):
    arr_mean = np.array([1.0, 2.0])
    arr_std = np.array([0.1, 0.2])
    arr_min = np.array([0.0, 0.0])
    noise = basic_frame.add_noise_from_obs(x_mean_array=arr_mean,
                                          x_std_array=arr_std,
                                          x_min_array=arr_min,
                                          share_index=True,
                                          noise_type='normal')
    assert noise.shape == basic_frame.shape

def test_add_noise_from_obs_normal_share_index_mismatch_length_raises(monkeypatch, basic_frame):
    arr_mean = np.array([1.0, 2.0])
    arr_std = np.array([0.1])
    arr_min = np.array([0.0, 0.0])
    with pytest.raises(IndexError):
        basic_frame.add_noise_from_obs(x_mean_array=arr_mean,
                                      x_std_array=arr_std,
                                      x_min_array=arr_min,
                                      share_index=True,
                                      noise_type='normal')

def test_add_noise_from_obs_normal_no_x_min_share_index_mismatch_raises(monkeypatch, basic_frame):
    arr_mean = np.array([1.0, 2.0])
    arr_std = np.array([0.1])
    with pytest.raises(IndexError):
        basic_frame.add_noise_from_obs(x_mean_array=arr_mean,
                                      x_std_array=arr_std,
                                      share_index=True,
                                      noise_type='normal')

def test_add_noise_from_obs_invalid_noise_type_fails(monkeypatch, basic_frame):
    with pytest.raises(SystemExit):
        basic_frame.add_noise_from_obs(x_mean_array=np.array([1.0]),
                                      noise_type='invalid')

def test_add_noise_and_update_noise_stats(basic_frame):
    # Initially noise stats are 0
    assert basic_frame.noise_mean == 0
    assert basic_frame.noise_std == 0
    basic_frame.add_noise(x_mean=5)
    old_mean = basic_frame.noise_mean
    old_std = basic_frame.noise_std
    # Add additional noise changes noise stats using sigma clipping
    basic_frame.add_noise(x_mean=5)
    assert basic_frame.noise_mean != old_mean or basic_frame.noise_std != old_std

def test_add_noise_retains_noise_param_when_first_call(basic_frame):
    assert basic_frame.noise_mean == 0 and basic_frame.noise_std == 0
    noise = basic_frame.add_noise(x_mean=7)
    assert basic_frame.noise_mean == 7
    assert noise.shape == basic_frame.shape
