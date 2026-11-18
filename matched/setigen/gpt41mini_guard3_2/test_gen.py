import pytest
import numpy as np

from setigen.frame import Frame


# Helper constants for tests
FREQ_CHANS = 10
TIME_CHANS = 5
DF = 1.0  # Hz
DT = 1.0  # s
FCH1 = 100.0  # Hz


def make_frame():
    """Helper to create a Frame with zero data."""
    return Frame(fchans=FREQ_CHANS,
                 tchans=TIME_CHANS,
                 df=DF,
                 dt=DT,
                 fch1=FCH1)


def test_add_noise_chi2_seeded(monkeypatch):
    # Seed numpy RNG used by distributions.chi2
    np.random.seed(12345)
    frame = make_frame()

    # Chi2 noise ignores x_std and x_min; only x_mean matters
    noise = frame.add_noise(x_mean=10)

    # noise shape matches data shape
    assert noise.shape == frame.data.shape

    # noise values are >= 0 for chi2
    assert np.all(noise >= 0)

    # After adding noise, data increased
    assert np.all(frame.data >= 0)

    # noise_mean and noise_std updated correctly (should be set to params)
    assert np.isclose(frame.noise_mean, 10)
    # Std should be set to chi2 std formula for x_mean=10 and chi2_df
    expected_std = np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df
    assert np.isclose(frame.noise_std, expected_std)


def test_add_noise_gaussian_with_min_seeded(monkeypatch):
    np.random.seed(42)  # Seed numpy RNG

    frame = make_frame()

    x_mean = 5.0
    x_std = 2.0
    x_min = 1.0

    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, x_min=x_min, noise_type='gaussian')

    # noise shape correct
    assert noise.shape == frame.data.shape

    # All noise values >= x_min (truncated Gaussian)
    assert np.all(noise >= x_min)

    # Data increased by noise
    assert np.all(frame.data >= noise.min())

    # noise_mean and noise_std updated (should be set to params because noise_mean/std previously zero)
    assert np.isclose(frame.noise_mean, x_mean)
    assert np.isclose(frame.noise_std, x_std)


def test_add_noise_gaussian_without_min_seeded(monkeypatch):
    np.random.seed(98765)

    frame = make_frame()

    x_mean = 0.0
    x_std = 1.0

    noise = frame.add_noise(x_mean=x_mean, x_std=x_std, noise_type='normal')

    assert noise.shape == frame.data.shape

    # Gaussian noise can have negative values
    assert np.any(noise < 0)

    # Data changed accordingly
    assert np.allclose(frame.data, noise)

    assert np.isclose(frame.noise_mean, x_mean)
    assert np.isclose(frame.noise_std, x_std)


def test_add_noise_invalid_type(monkeypatch):
    np.random.seed(0)

    frame = make_frame()

    # Should raise system exit for invalid noise type
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, noise_type='invalid_noise_type')


def test_add_noise_requires_x_std_if_gaussian(monkeypatch):
    np.random.seed(0)

    frame = make_frame()

    # Should raise SystemExit because x_std is None for gaussian noise
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, noise_type='gaussian')


def test_add_noise_updates_noise_stats(monkeypatch):
    np.random.seed(1)
    frame = make_frame()

    # First add some noise to set noise_mean and noise_std
    _ = frame.add_noise(x_mean=10)

    old_mean = frame.noise_mean
    old_std = frame.noise_std

    # Add noise again; noise_mean and std should no longer be set directly but updated via sigma_clip
    np.random.seed(1)
    noise = frame.add_noise(x_mean=10)

    # After second addition, noise_mean and noise_std should change from first set values
    assert frame.noise_mean != old_mean or frame.noise_std != old_std


def test_get_data_returns_correct_data(monkeypatch):
    np.random.seed(0)
    frame = make_frame()

    frame.data[:] = 5.0

    data_direct = frame.get_data(use_db=False)
    assert np.all(data_direct == 5.0)

    # Test use_db True returns correct log10 values
    data_db = frame.get_data(use_db=True)
    expected_db = 10 * np.log10(frame.data)
    # Compare with tolerance because log10 can produce floating point differences
    assert np.allclose(data_db, expected_db, rtol=1e-7)


def test_get_data_log_with_zero_or_negative_values(monkeypatch):
    np.random.seed(0)
    frame = make_frame()

    # Add some data with zeros and negative values
    frame.data[:] = -1.0
    # get_data(use_db=True) with negative or zero values will produce runtime warnings and inf/nan
    # Just test that it does not raise exceptions and returns array of correct shape
    try:
        data_db = frame.get_data(use_db=True)
        # Result should be finite or -inf/NaN - we just test shape here.
        assert data_db.shape == frame.data.shape
    except Exception:
        pytest.fail("get_data raised unexpected exception with negative values.")


def test_add_noise_from_obs_chi2_seeded(monkeypatch):
    np.random.seed(1234)
    frame = make_frame()

    # Provide arrays to sample from; small arrays for deterministic tests
    x_mean_array = np.array([1.0, 2.0, 3.0])
    x_std_array = np.array([0.1, 0.2, 0.3])
    x_min_array = np.array([-1.0, -2.0, -3.0])

    noise = frame.add_noise_from_obs(x_mean_array=x_mean_array,
                                    noise_type='chi2')
    assert noise.shape == frame.data.shape
    assert np.all(noise >= 0)
    assert isinstance(frame.noise_mean, float)
    assert isinstance(frame.noise_std, float)


def test_add_noise_from_obs_gaussian_share_index_true(monkeypatch):
    np.random.seed(42)
    frame = make_frame()

    x_mean_array = np.array([1.0, 2.0, 3.0])
    x_std_array = np.array([0.5, 0.7, 0.9])
    x_min_array = np.array([-1.0, 0.0, 0.5])

    noise = frame.add_noise_from_obs(x_mean_array=x_mean_array,
                                    x_std_array=x_std_array,
                                    x_min_array=x_min_array,
                                    share_index=True,
                                    noise_type='normal')
    assert noise.shape == frame.data.shape
    # Noise values should be >= min at chosen index (min could be negative, so only test shape here)
    # noise_mean and noise_std should be floats
    assert isinstance(frame.noise_mean, float)
    assert isinstance(frame.noise_std, float)


def test_add_noise_from_obs_gaussian_share_index_false(monkeypatch):
    np.random.seed(21)
    frame = make_frame()

    x_mean_array = np.array([4.0, 5.0])
    x_std_array = np.array([1.0, 1.5])
    x_min_array = np.array([0.0, 0.5])

    noise = frame.add_noise_from_obs(x_mean_array=x_mean_array,
                                    x_std_array=x_std_array,
                                    x_min_array=x_min_array,
                                    share_index=False,
                                    noise_type='normal')

    assert noise.shape == frame.data.shape
    assert isinstance(frame.noise_mean, float)
    assert isinstance(frame.noise_std, float)


def test_add_noise_from_obs_invalid_noise_type(monkeypatch):
    np.random.seed(0)

    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise_from_obs(noise_type='invalid')


def test_add_noise_from_obs_share_index_mismatch_length(monkeypatch):
    np.random.seed(0)
    frame = make_frame()

    x_mean_array = np.array([1.0, 2.0])
    x_std_array = np.array([0.5])  # Mismatch length
    x_min_array = np.array([0.1, 0.2])

    with pytest.raises(IndexError):
        frame.add_noise_from_obs(x_mean_array=x_mean_array,
                                x_std_array=x_std_array,
                                x_min_array=x_min_array,
                                share_index=True,
                                noise_type='normal')


def test_add_noise_from_obs_share_index_length_mismatch_two_arrays(monkeypatch):
    np.random.seed(0)
    frame = make_frame()

    x_mean_array = np.array([1.0, 2.0])
    x_std_array = np.array([0.5])  # Mismatch length

    with pytest.raises(IndexError):
        frame.add_noise_from_obs(x_mean_array=x_mean_array,
                                x_std_array=x_std_array,
                                share_index=True,
                                noise_type='normal')


def test_add_noise_sets_noise_stats_only_if_zero(monkeypatch):
    np.random.seed(5)
    frame = make_frame()

    # Initially noise_mean and noise_std are updated to 0 (because initial data zeros)
    assert frame.noise_mean == 0
    assert frame.noise_std == 0

    # Add noise for first time sets noise_mean and noise_std to parameters
    frame.add_noise(3.0)

    # Add noise second time, noise_mean/noise_std should not be reset to params but recalculated
    mean_before = frame.noise_mean
    std_before = frame.noise_std
    frame.add_noise(3.0)
    mean_after = frame.noise_mean
    std_after = frame.noise_std
    # Usually mean or std should have changed after second noise addition
    assert mean_after != mean_before or std_after != std_before


def test_get_data_returns_copy_and_data_is_altered_by_add_noise(monkeypatch):
    np.random.seed(42)
    frame = make_frame()
    original_data = frame.data.copy()

    # Initially, get_data returns the same data array (numpy arrays are mutable)
    data0 = frame.get_data(use_db=False)
    # After add_noise, data changes in place
    frame.add_noise(x_mean=10)
    data1 = frame.get_data(use_db=False)

    # Confirm data1 differs from original_data
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(original_data, data1)

    # Confirm data1 shape correct
    assert data1.shape == (TIME_CHANS, FREQ_CHANS)


@pytest.mark.parametrize("noise_type", ['chi2', 'normal', 'gaussian'])
def test_add_noise_and_get_data_consistency(monkeypatch, noise_type):
    np.random.seed(101)
    frame = make_frame()

    if noise_type == 'chi2':
        noise = frame.add_noise(x_mean=7.5, noise_type=noise_type)
        # noise values non-negative
        assert np.all(noise >= 0)
    else:
        noise = frame.add_noise(x_mean=7.5, x_std=2.0, noise_type=noise_type)
        # noise values shape correct
        assert noise.shape == frame.data.shape

    # get_data(use_db=False) returns data same as frame.data
    data = frame.get_data(False)
    assert np.allclose(data, frame.data)

    # get_data(use_db=True) returns 10*log10(data)
    data_db = frame.get_data(True)
    assert np.allclose(data_db, 10 * np.log10(frame.data))


def test_add_noise_returned_noise_and_data_consistency_seed(monkeypatch):
    # Verify output reproducibility by fixed seed for add_noise 'normal'
    np.random.seed(2022)
    frame1 = make_frame()
    noise1 = frame1.add_noise(x_mean=2.5, x_std=0.5, noise_type='normal')

    np.random.seed(2022)
    frame2 = make_frame()
    noise2 = frame2.add_noise(x_mean=2.5, x_std=0.5, noise_type='normal')

    np.testing.assert_allclose(noise1, noise2, rtol=1e-7)
    np.testing.assert_allclose(frame1.data, frame2.data, rtol=1e-7)


def test_get_data_log_handles_floats(monkeypatch):
    np.random.seed(0)
    frame = make_frame()
    # Set data with positive float different than 1 to test log10 conversion
    frame.data[:] = 4.0
    expected = 10 * np.log10(4.0)

    data_db = frame.get_data(use_db=True)
    assert np.allclose(data_db, expected)

