import numpy as np
import pytest
from astropy import units as u
from setigen.frame import Frame
from setigen import distributions

# Helper function to create a simple Frame with zeros
def make_simple_frame():
    fchans = 64
    tchans = 32
    df = 1 * u.Hz
    dt = 1 * u.s
    fch1 = 100 * u.Hz
    return Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)


def test_add_noise_chi2_defaults():
    frame = make_simple_frame()
    x_mean = 10.0
    noise = frame.add_noise(x_mean=x_mean, noise_type='chi2')
    assert noise.shape == frame.shape
    # Check data increased by noise
    assert np.all(frame.data >= 0)
    # Noise mean and std set
    assert frame.noise_mean == pytest.approx(x_mean)
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std)


def test_add_noise_gaussian_with_x_min():
    frame = make_simple_frame()
    x_mean = 5.0
    x_std = 2.0
    x_min = 1.0
    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, x_min=x_min, noise_type='normal')
    assert noise.shape == frame.shape
    assert np.all(frame.data >= x_min)
    assert frame.noise_mean == pytest.approx(x_mean)
    assert frame.noise_std == pytest.approx(x_std)


def test_add_noise_gaussian_without_x_min():
    frame = make_simple_frame()
    x_mean = 7.0
    x_std = 3.0
    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert frame.noise_mean == pytest.approx(x_mean)
    assert frame.noise_std == pytest.approx(x_std)


def test_add_noise_gaussian_missing_x_std_exits(monkeypatch):
    frame = make_simple_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type='normal')


def test_add_noise_invalid_noise_type_exits(monkeypatch):
    frame = make_simple_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type='invalid_type')


def test_get_data_default_returns_data_array():
    frame = make_simple_frame()
    frame.data = np.ones(frame.shape) * 42
    data = frame.get_data()
    assert isinstance(data, np.ndarray)
    assert np.all(data == 42)


def test_get_data_use_db_returns_log10_scaled(monkeypatch):
    frame = make_simple_frame()
    frame.data = np.full(frame.shape, 100.0)
    data_db = frame.get_data(use_db=True)
    expected_db = 10 * np.log10(frame.data)
    np.testing.assert_allclose(data_db, expected_db)


def test_add_noise_updates_noise_stats_after_existing_noise():
    frame = make_simple_frame()
    frame.add_noise(x_mean=1, noise_type='chi2')
    old_mean, old_std = frame.noise_mean, frame.noise_std
    noise = frame.add_noise(x_mean=3, noise_type='chi2')
    # Should not be equal to previous mean/std because stats got updated
    assert not (frame.noise_mean == old_mean and frame.noise_std == old_std)
    assert noise.shape == frame.shape


def test_add_noise_returns_array_consistent_with_frame_shape():
    frame = make_simple_frame()
    noise = frame.add_noise(x_mean=2.5, noise_type='chi2')
    assert noise.shape == frame.shape


def test_add_noise_with_set_noise_mean_std_not_zero_calls_update(mocker):
    frame = make_simple_frame()
    # Set noise_mean and noise_std nonzero manually
    frame.noise_mean = 1
    frame.noise_std = 1
    spy = mocker.spy(frame, '_update_noise_frame_stats')
    _ = frame.add_noise(x_mean=1.5, noise_type='chi2')
    assert spy.call_count == 1


def test_get_data_with_zero_and_negative_values_and_use_db():
    frame = make_simple_frame()
    frame.data = np.array([[0, 1], [10, 100]])
    with pytest.raises(RuntimeWarning):
        # log10(0) is -inf, log10(negative) is NaN, so test to ensure no crash but warning
        # We purposely add a negative value to test behavior
        frame.data[0, 0] = 0
        frame.data[0, 1] = -1
        _ = frame.get_data(use_db=True)


@pytest.mark.parametrize("noise_type", ['chi2', 'normal', 'gaussian'])
def test_add_noise_from_obs_behavior(noise_type):
    frame = make_simple_frame()
    length = 100
    means = np.linspace(1, 10, length)
    stds = np.linspace(0.1, 5, length)
    mins = np.zeros(length)
    noise = frame.add_noise_from_obs(x_mean_array=means,
                                    x_std_array=stds,
                                    x_min_array=mins,
                                    share_index=True,
                                    noise_type=noise_type)
    assert noise.shape == frame.shape
    assert frame.noise_mean >= 0
    assert frame.noise_std >= 0


def test_add_noise_from_obs_incompatible_array_lengths_raises():
    frame = make_simple_frame()
    x_mean_array = np.array([1, 2, 3])
    x_std_array = np.array([1, 2])
    x_min_array = np.array([0, 1, 2])
    with pytest.raises(IndexError):
        frame.add_noise_from_obs(x_mean_array=x_mean_array,
                                x_std_array=x_std_array,
                                x_min_array=x_min_array,
                                share_index=True,
                                noise_type='normal')


def test_add_noise_from_obs_no_arrays_load_defaults(monkeypatch):
    frame = make_simple_frame()
    # Patch np.load to a dummy array
    dummy_params = np.ones((10, 3))
    monkeypatch.setattr(np, 'load', lambda path: dummy_params)
    noise = frame.add_noise_from_obs()
    assert noise.shape == frame.shape


def test_add_noise_from_obs_invalid_noise_type_exits(monkeypatch):
    frame = make_simple_frame()
    with pytest.raises(SystemExit):
        frame.add_noise_from_obs(noise_type='invalid')


def test_add_noise_and_get_data_interaction():
    frame = make_simple_frame()
    frame.add_noise(x_mean=5, noise_type='chi2')
    data1 = frame.get_data()
    frame.add_noise(x_mean=5, noise_type='chi2')
    data2 = frame.get_data()
    assert not np.array_equal(data1, data2), "Data should have changed after adding more noise"


def test_add_noise_minimum_clip_behavior():
    frame = make_simple_frame()
    x_mean = 3.0
    x_std = 1.0
    x_min = 1.0
    noise = frame.add_noise(x_mean, x_std, x_min, noise_type='normal')
    # Because truncated_gaussian is called, values should be >= x_min
    assert np.all(frame.data >= x_min)


def test_add_noise_gaussian_without_x_min_and_x_std_none_exits():
    frame = make_simple_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type='normal')


def test_add_noise_gaussian_with_x_min_none_and_x_std_none_exits():
    frame = make_simple_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type='gaussian')


def test_add_noise_chi2_ignores_x_std_and_x_min_params():
    frame = make_simple_frame()
    # Provide x_std and x_min but they should be ignored for chi2 noise
    noise = frame.add_noise(x_mean=4.0, x_std=2.0, x_min=1.0, noise_type='chi2')
    assert noise.shape == frame.shape


# If available, use monkeypatch to mock distributions functions to verify calls happen as expected.
def test_add_noise_chi2_calls_distributions_chi2(monkeypatch):
    frame = make_simple_frame()
    called = {}

    def fake_chi2(x_mean, df, shape):
        called['called'] = True
        assert x_mean == 7.0
        assert df == frame.chi2_df
        assert shape == frame.shape
        return np.ones(shape) * 3  # dummy noise
    monkeypatch.setattr(distributions, "chi2", fake_chi2)
    noise = frame.add_noise(x_mean=7.0, noise_type='chi2')
    assert called.get('called', False)
    assert np.all(noise == 3)
    assert np.all(frame.data == 3)


def test_add_noise_gaussian_calls_distribution(monkeypatch):
    frame = make_simple_frame()
    called = {}

    def fake_gaussian(x_mean, x_std, shape):
        called['called'] = True
        assert x_mean == 4.0
        assert x_std == 1.0
        assert shape == frame.shape
        return np.ones(shape) * 5
    monkeypatch.setattr(distributions, "gaussian", fake_gaussian)

    noise = frame.add_noise(x_mean=4.0, x_std=1.0, noise_type='normal')
    assert called.get('called', False)
    assert np.all(noise == 5)
    assert np.all(frame.data == 5)


def test_add_noise_truncated_gaussian_calls_distribution(monkeypatch):
    frame = make_simple_frame()
    called = {}

    def fake_truncated_gaussian(x_mean, x_std, x_min, shape):
        called['called'] = True
        assert x_mean == 2.0
        assert x_std == 0.5
        assert x_min == 1.0
        assert shape == frame.shape
        return np.ones(shape) * 6
    monkeypatch.setattr(distributions, "truncated_gaussian", fake_truncated_gaussian)

    noise = frame.add_noise(x_mean=2.0, x_std=0.5, x_min=1.0, noise_type='normal')
    assert called.get('called', False)
    assert np.all(noise == 6)
    assert np.all(frame.data == 6)
