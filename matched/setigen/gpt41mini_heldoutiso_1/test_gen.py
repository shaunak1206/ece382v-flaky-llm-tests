import pytest
import numpy as np
from astropy import units as u
from setigen.frame import Frame
from setigen import distributions

# Helper to create a minimal Frame
def create_minimal_frame():
    fchans = 10
    tchans = 5
    df = 1 * u.Hz
    dt = 1 * u.s
    fch1 = 10 * u.Hz
    return Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)

# ---------- Tests for Frame.add_noise ----------

def test_add_noise_chi2_changes_data_and_returns_noise():
    frame = create_minimal_frame()
    x_mean = 10
    noise = frame.add_noise(x_mean=x_mean, noise_type='chi2')
    # Added noise shape matches frame.data shape
    assert noise.shape == frame.data.shape
    # Data is no longer all zeros
    assert np.any(frame.data != 0)
    # Returned noise matches what was added to data (frame.data - zeros)
    # Rough check: noise is close to frame.data because initial data was zeros
    np.testing.assert_allclose(frame.data, noise, rtol=0.1)

def test_add_noise_chi2_sets_noise_mean_std_correctly_when_zero():
    frame = create_minimal_frame()
    # noise_mean and noise_std initially zero after init
    assert frame.noise_mean == 0 and frame.noise_std == 0
    x_mean = 5
    noise = frame.add_noise(x_mean=x_mean, noise_type='chi2')
    assert frame.noise_mean == x_mean
    # Chi2 std expected by formula in code
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert pytest.approx(frame.noise_std, rel=1e-6) == expected_std

def test_add_noise_normal_without_x_std_exits(monkeypatch):
    frame = create_minimal_frame()
    called = {}
    def fake_exit(msg):
        called['msg'] = msg
        raise SystemExit(msg)
    monkeypatch.setattr("sys.exit", fake_exit)
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, noise_type='normal')  # no x_std

def test_add_noise_normal_with_x_std_and_x_min_calls_truncated(monkeypatch):
    frame = create_minimal_frame()
    calls = {}
    # patch truncated_gaussian to record inputs and return fixed noise
    def fake_truncated_gaussian(mean, std, shape):
        calls['mean'] = mean
        calls['std'] = std
        calls['shape'] = shape
        return np.ones(shape) * mean
    monkeypatch.setattr(distributions, "truncated_gaussian", fake_truncated_gaussian)
    x_mean = 3
    x_std = 1
    x_min = 0
    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, x_min=x_min, noise_type='normal')
    # Check call args
    assert calls['mean'] == x_mean
    assert calls['std'] == x_std
    assert calls['shape'] == frame.shape
    # Data incremented by noise
    np.testing.assert_allclose(frame.data, noise)

def test_add_noise_normal_with_x_std_without_x_min_calls_gaussian(monkeypatch):
    frame = create_minimal_frame()
    calls = {}
    def fake_gaussian(mean, std, shape):
        calls['mean'] = mean
        calls['std'] = std
        calls['shape'] = shape
        return np.ones(shape) * mean * 2
    monkeypatch.setattr(distributions, "gaussian", fake_gaussian)
    x_mean = 4
    x_std = 1.5
    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, noise_type='normal')
    # Check noise returned matches fake_gaussian result
    assert calls['mean'] == x_mean
    assert calls['std'] == x_std
    np.testing.assert_allclose(noise, frame.data)
    np.testing.assert_allclose(noise, np.ones(frame.shape) * (x_mean*2))

def test_add_noise_invalid_noise_type_exits(monkeypatch):
    frame = create_minimal_frame()
    def fake_exit(msg):
        raise SystemExit(msg)
    monkeypatch.setattr("sys.exit", fake_exit)
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, noise_type='invalid_type')

def test_add_noise_accumulates_noise_and_updates_stats(monkeypatch):
    frame = create_minimal_frame()
    # Add first noise: should set noise_mean, noise_std to x_mean, x_std
    x_mean1 = 5
    noise1 = frame.add_noise(x_mean=x_mean1, noise_type='chi2')
    noise_mean_1 = frame.noise_mean
    noise_std_1 = frame.noise_std
    # Add second noise, should accumulate and update noise stats
    noise2 = frame.add_noise(x_mean=3, noise_type='chi2')
    # Noise mean and std should be updated and different than first
    assert frame.noise_mean != noise_mean_1 or frame.noise_std != noise_std_1

# ---------- Tests for Frame.get_data ----------

def test_get_data_without_db_returns_data_array():
    frame = create_minimal_frame()
    frame.data[:] = np.arange(frame.tchans * frame.fchans).reshape(frame.shape)
    data = frame.get_data()
    assert np.array_equal(data, frame.data)

def test_get_data_with_db_returns_log10_scaled_data():
    frame = create_minimal_frame()
    arr = np.linspace(1, 100, num=frame.tchans * frame.fchans).reshape(frame.shape)
    frame.data[:] = arr
    data_db = frame.get_data(use_db=True)
    expected = 10 * np.log10(arr)
    np.testing.assert_allclose(data_db, expected)

def test_get_data_with_db_handles_zero_or_negative_values(monkeypatch):
    frame = create_minimal_frame()
    # Set data to contain zeros and negatives
    frame.data[:] = -1
    # Log10 of negative is nan or -inf, numpy will warn (valid behavior)
    result = frame.get_data(use_db=True)
    assert result.shape == frame.shape
    # Values should be -inf or nan, test at least shape returned and type
    assert np.isnan(result).any() or np.isneginf(result).any()

def test_get_data_does_not_modify_internal_data_array():
    frame = create_minimal_frame()
    original_copy = frame.data.copy()
    _ = frame.get_data()
    np.testing.assert_array_equal(frame.data, original_copy)
    _ = frame.get_data(use_db=True)
    # data remains unchanged after get_data with db
    np.testing.assert_array_equal(frame.data, original_copy)

def test_add_noise_returns_array_of_correct_shape_and_dtype():
    frame = create_minimal_frame()
    noise = frame.add_noise(x_mean=2, noise_type='chi2')
    assert noise.shape == frame.shape
    assert isinstance(noise, np.ndarray)
    assert np.issubdtype(noise.dtype, np.floating)

def test_add_noise_with_gaussian_updates_noise_stats(monkeypatch):
    frame = create_minimal_frame()
    x_mean = 1.0
    x_std = 0.5
    # Patch distributions.gaussian to return fixed noise
    def fake_gaussian(mean, std, shape):
        return np.full(shape, mean)
    monkeypatch.setattr(distributions, 'gaussian', fake_gaussian)
    frame.add_noise(x_mean=x_mean, x_std=x_std, noise_type='gaussian')
    assert frame.noise_mean == x_mean
    assert frame.noise_std == x_std

def test_add_noise_with_truncated_gaussian_updates_noise_stats(monkeypatch):
    frame = create_minimal_frame()
    x_mean = 1.0
    x_std = 0.5
    x_min = 0
    def fake_truncated_gaussian(mean, std, minimum, shape):
        return np.full(shape, mean)
    monkeypatch.setattr(distributions, 'truncated_gaussian', fake_truncated_gaussian)
    frame.add_noise(x_mean=x_mean, x_std=x_std, x_min=x_min, noise_type='truncated_gaussian')
    assert frame.noise_mean == x_mean
    assert frame.noise_std == x_std

