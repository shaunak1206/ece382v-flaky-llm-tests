import numpy as np
import pytest
from astropy import units as u

from setigen.frame import Frame


def make_frame(fchans=8, tchans=4, data=None):
    """Build a small Frame with df=1 Hz and dt=1 s (chi2_df = 4)."""
    return Frame(fchans=fchans,
                 tchans=tchans,
                 df=1 * u.Hz,
                 dt=1 * u.s,
                 data=data)


# ---------------------------------------------------------------------------
# Frame.add_noise
# ---------------------------------------------------------------------------

def test_add_noise_chi2_returns_array_with_frame_shape():
    frame = make_frame(fchans=16, tchans=8)
    np.random.seed(1)
    noise = frame.add_noise(x_mean=10)
    assert noise.shape == frame.shape == (8, 16)


def test_add_noise_chi2_adds_noise_into_existing_data():
    frame = make_frame(fchans=8, tchans=4, data=np.ones((4, 8)))
    before = frame.get_data().copy()
    np.random.seed(2)
    noise = frame.add_noise(x_mean=10)
    assert np.allclose(frame.get_data(), before + noise)


def test_add_noise_chi2_accumulates_across_calls():
    frame = make_frame(fchans=8, tchans=4)
    np.random.seed(3)
    n1 = frame.add_noise(x_mean=5)
    n2 = frame.add_noise(x_mean=5)
    assert np.allclose(frame.get_data(), n1 + n2)


def test_add_noise_chi2_mean_close_to_x_mean():
    frame = make_frame(fchans=200, tchans=200)
    np.random.seed(4)
    noise = frame.add_noise(x_mean=10)
    assert noise.mean() == pytest.approx(10, abs=0.5)


def test_add_noise_chi2_is_nonnegative():
    frame = make_frame(fchans=64, tchans=64)
    np.random.seed(5)
    noise = frame.add_noise(x_mean=3)
    assert np.all(noise >= 0)


def test_add_noise_chi2_first_call_sets_noise_stats_from_parameters():
    frame = make_frame(fchans=8, tchans=4)
    np.random.seed(6)
    frame.add_noise(x_mean=10)
    noise_mean, noise_std = frame.get_noise_stats()
    assert noise_mean == pytest.approx(10)
    assert noise_std == pytest.approx(np.sqrt(2 * 4) * 10 / 4)


def test_add_noise_chi2_ignores_x_std_argument():
    frame = make_frame(fchans=8, tchans=4)
    np.random.seed(7)
    frame.add_noise(x_mean=10, x_std=3)
    _, noise_std = frame.get_noise_stats()
    assert noise_std == pytest.approx(np.sqrt(2 * 4) * 10 / 4)


def test_add_noise_second_call_recomputes_stats_from_data():
    frame = make_frame(fchans=200, tchans=200)
    np.random.seed(8)
    frame.add_noise(x_mean=10)
    frame.add_noise(x_mean=10)
    noise_mean, _ = frame.get_noise_stats()
    # Stats now come from the summed data (mean ~ 20), not the parameters
    assert noise_mean == pytest.approx(20, rel=0.05)


def test_add_noise_chi2_reproducible_with_seed():
    frame1 = make_frame(fchans=8, tchans=4)
    frame2 = make_frame(fchans=8, tchans=4)
    np.random.seed(9)
    noise1 = frame1.add_noise(x_mean=10)
    np.random.seed(9)
    noise2 = frame2.add_noise(x_mean=10)
    assert np.array_equal(noise1, noise2)


def test_add_noise_normal_mean_and_std_match_parameters():
    frame = make_frame(fchans=200, tchans=200)
    np.random.seed(10)
    noise = frame.add_noise(x_mean=5, x_std=2, noise_type='normal')
    assert noise.mean() == pytest.approx(5, abs=0.2)
    assert noise.std() == pytest.approx(2, abs=0.1)


def test_add_noise_gaussian_alias_works():
    frame = make_frame(fchans=8, tchans=4)
    np.random.seed(11)
    noise = frame.add_noise(x_mean=5, x_std=2, noise_type='gaussian')
    assert noise.shape == frame.shape


def test_add_noise_normal_first_call_sets_noise_stats_from_parameters():
    frame = make_frame(fchans=8, tchans=4)
    np.random.seed(12)
    frame.add_noise(x_mean=5, x_std=2, noise_type='normal')
    assert frame.get_noise_stats() == (5, 2)


def test_add_noise_normal_without_x_std_exits():
    frame = make_frame(fchans=8, tchans=4)
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='normal')


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame(fchans=8, tchans=4)
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, x_std=1, noise_type='poisson')


def test_add_noise_truncated_gaussian_respects_x_min():
    frame = make_frame(fchans=100, tchans=100)
    np.random.seed(13)
    noise = frame.add_noise(x_mean=0, x_std=1, x_min=0.0,
                            noise_type='normal')
    assert noise.shape == frame.shape
    assert np.all(noise >= 0.0)


# ---------------------------------------------------------------------------
# Frame.get_data
# ---------------------------------------------------------------------------

def test_get_data_default_returns_underlying_data_array():
    frame = make_frame(fchans=8, tchans=4, data=np.ones((4, 8)))
    assert frame.get_data() is frame.data


def test_get_data_use_db_false_returns_linear_data():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    frame = make_frame(fchans=2, tchans=2, data=arr)
    assert np.array_equal(frame.get_data(use_db=False), arr)


def test_get_data_use_db_converts_to_decibels():
    arr = np.array([[1.0, 10.0], [100.0, 1000.0]])
    frame = make_frame(fchans=2, tchans=2, data=arr)
    expected = np.array([[0.0, 10.0], [20.0, 30.0]])
    assert np.allclose(frame.get_data(use_db=True), expected)


def test_get_data_use_db_does_not_modify_stored_data():
    arr = np.array([[1.0, 10.0], [100.0, 1000.0]])
    frame = make_frame(fchans=2, tchans=2, data=arr.copy())
    frame.get_data(use_db=True)
    assert np.array_equal(frame.data, arr)


def test_get_data_use_db_zero_values_become_negative_infinity():
    frame = make_frame(fchans=2, tchans=2, data=np.zeros((2, 2)))
    with np.errstate(divide='ignore'):
        db = frame.get_data(use_db=True)
    assert np.all(np.isneginf(db))


def test_get_data_use_db_negative_values_become_nan():
    arr = np.array([[-1.0, 1.0], [2.0, -3.0]])
    frame = make_frame(fchans=2, tchans=2, data=arr)
    with np.errstate(invalid='ignore'):
        db = frame.get_data(use_db=True)
    assert np.isnan(db[0, 0])
    assert np.isnan(db[1, 1])
    assert db[0, 1] == pytest.approx(0.0)


def test_get_data_reflects_subsequent_add_noise():
    frame = make_frame(fchans=8, tchans=4)
    np.random.seed(14)
    frame.add_noise(x_mean=10)
    assert np.array_equal(frame.get_data(), frame.data)
    assert np.any(frame.get_data() != 0)


def test_get_data_use_db_matches_current_data_after_add_noise():
    frame = make_frame(fchans=8, tchans=4, data=np.ones((4, 8)))
    np.random.seed(15)
    frame.add_noise(x_mean=10)
    with np.errstate(divide='ignore'):
        expected = 10 * np.log10(frame.data)
        actual = frame.get_data(use_db=True)
    assert np.allclose(actual, expected, equal_nan=True)


def test_get_data_shape_matches_frame_shape():
    frame = make_frame(fchans=16, tchans=6)
    assert frame.get_data().shape == frame.shape == (6, 16)
    assert frame.get_data(use_db=True).shape == frame.shape
