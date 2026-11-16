import numpy as np
import pytest
from astropy import units as u
from astropy.stats import sigma_clip

from setigen.frame import Frame


SMALL_FCHANS = 16
SMALL_TCHANS = 8
LARGE_FCHANS = 256
LARGE_TCHANS = 64


def make_frame(fchans=SMALL_FCHANS, tchans=SMALL_TCHANS, df=2 * u.Hz,
               dt=1 * u.s, data=None):
    """Create a synthetic Frame with the given resolution and optional data."""
    return Frame(fchans=fchans,
                 tchans=tchans,
                 df=df,
                 dt=dt,
                 data=data)


# ---------------------------------------------------------------------------
# Frame.add_noise: chi-squared noise
# ---------------------------------------------------------------------------

def test_add_noise_chi2_returns_frame_shaped_noise():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert noise.shape == (SMALL_TCHANS, SMALL_FCHANS)
    assert noise.shape == frame.shape


def test_add_noise_chi2_adds_noise_to_existing_data():
    original = np.ones((SMALL_TCHANS, SMALL_FCHANS))
    frame = make_frame(data=original.copy())
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.data, original + noise)


def test_add_noise_chi2_mean_close_to_x_mean():
    np.random.seed(0)
    frame = make_frame(fchans=LARGE_FCHANS, tchans=LARGE_TCHANS)
    frame.add_noise(x_mean=10)
    assert np.mean(frame.data) == pytest.approx(10, rel=0.05)


def test_add_noise_chi2_sets_noise_stats_from_parameters():
    # df * dt = 2 -> chi2_df = 8
    # x_std = sqrt(2 * 8) * 10 / 8 = 5.0
    frame = make_frame(df=2 * u.Hz, dt=1 * u.s)
    frame.add_noise(x_mean=10)
    noise_mean, noise_std = frame.get_noise_stats()
    assert noise_mean == 10
    assert noise_std == pytest.approx(5.0)


def test_add_noise_chi2_std_formula_depends_on_chi2_df():
    # df * dt = 1 -> chi2_df = 4
    # x_std = sqrt(2 * 4) * 10 / 4 = sqrt(8) * 2.5
    frame = make_frame(df=1 * u.Hz, dt=1 * u.s)
    frame.add_noise(x_mean=10)
    _, noise_std = frame.get_noise_stats()
    assert noise_std == pytest.approx(np.sqrt(8) * 10 / 4)


def test_add_noise_chi2_ignores_x_std_and_x_min():
    np.random.seed(1)
    frame = make_frame(fchans=LARGE_FCHANS, tchans=LARGE_TCHANS)
    frame.add_noise(x_mean=10, x_std=100, x_min=50, noise_type='chi2')
    noise_mean, noise_std = frame.get_noise_stats()
    # Stats come from the chi2 formula, not the ignored x_std
    assert noise_std == pytest.approx(5.0)
    assert np.mean(frame.data) == pytest.approx(10, rel=0.05)


def test_add_noise_chi2_second_call_updates_stats_from_data():
    frame = make_frame()
    frame.add_noise(x_mean=10)
    frame.add_noise(x_mean=10)
    expected_mean = np.mean(sigma_clip(frame.data, sigma=3, maxiters=5,
                                       masked=False))
    noise_mean, _ = frame.get_noise_stats()
    assert noise_mean == pytest.approx(expected_mean)


def test_add_noise_accumulates_returned_noise():
    frame = make_frame()
    n1 = frame.add_noise(x_mean=10)
    n2 = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.data, n1 + n2)


def test_add_noise_is_reproducible_with_seed():
    np.random.seed(42)
    frame1 = make_frame()
    n1 = frame1.add_noise(x_mean=10)

    np.random.seed(42)
    frame2 = make_frame()
    n2 = frame2.add_noise(x_mean=10)

    np.testing.assert_array_equal(n1, n2)


# ---------------------------------------------------------------------------
# Frame.add_noise: Gaussian noise
# ---------------------------------------------------------------------------

def test_add_noise_normal_returns_frame_shaped_noise():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10, x_std=2, noise_type='normal')
    assert noise.shape == (SMALL_TCHANS, SMALL_FCHANS)


def test_add_noise_normal_mean_and_std_close_to_parameters():
    np.random.seed(2)
    frame = make_frame(fchans=LARGE_FCHANS, tchans=LARGE_TCHANS)
    noise = frame.add_noise(x_mean=10, x_std=3, noise_type='normal')
    assert np.mean(noise) == pytest.approx(10, abs=0.2)
    assert np.std(noise) == pytest.approx(3, rel=0.05)


def test_add_noise_normal_sets_noise_stats_from_parameters():
    frame = make_frame()
    frame.add_noise(x_mean=10, x_std=2, noise_type='normal')
    assert frame.get_noise_stats() == (10, 2)


def test_add_noise_gaussian_alias_matches_normal():
    np.random.seed(7)
    frame1 = make_frame()
    n1 = frame1.add_noise(x_mean=5, x_std=1, noise_type='normal')

    np.random.seed(7)
    frame2 = make_frame()
    n2 = frame2.add_noise(x_mean=5, x_std=1, noise_type='gaussian')

    np.testing.assert_array_equal(n1, n2)


def test_add_noise_normal_with_x_min_respects_lower_bound():
    np.random.seed(3)
    frame = make_frame(fchans=LARGE_FCHANS, tchans=LARGE_TCHANS)
    noise = frame.add_noise(x_mean=0, x_std=5, x_min=0, noise_type='normal')
    assert np.all(noise >= 0)
    assert np.all(frame.data >= 0)


def test_add_noise_normal_without_x_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='normal')


def test_add_noise_normal_without_x_std_leaves_data_unchanged():
    original = np.full((SMALL_TCHANS, SMALL_FCHANS), 3.0)
    frame = make_frame(data=original.copy())
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='normal')
    np.testing.assert_array_equal(frame.data, original)


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='poisson')


# ---------------------------------------------------------------------------
# Frame.get_data
# ---------------------------------------------------------------------------

def test_get_data_default_returns_raw_data():
    data = np.arange(SMALL_TCHANS * SMALL_FCHANS, dtype=float).reshape(
        SMALL_TCHANS, SMALL_FCHANS) + 1
    frame = make_frame(data=data)
    np.testing.assert_array_equal(frame.get_data(), data)
    assert frame.get_data() is frame.data


def test_get_data_use_false_explicit_returns_raw_data():
    data = np.full((SMALL_TCHANS, SMALL_FCHANS), 10.0)
    frame = make_frame(data=data)
    np.testing.assert_array_equal(frame.get_data(use_db=False), data)


def test_get_data_use_db_converts_to_decibels():
    data = np.array([[1.0, 10.0, 100.0, 1000.0]] * 8)
    frame = make_frame(fchans=4, tchans=8, data=data)
    result = frame.get_data(use_db=True)
    expected = np.array([[0.0, 10.0, 20.0, 30.0]] * 8)
    np.testing.assert_allclose(result, expected)


def test_get_data_use_db_does_not_modify_stored_data():
    data = np.full((SMALL_TCHANS, SMALL_FCHANS), 100.0)
    frame = make_frame(data=data.copy())
    frame.get_data(use_db=True)
    np.testing.assert_array_equal(frame.data, data)


def test_get_data_reflects_added_noise():
    original = np.full((SMALL_TCHANS, SMALL_FCHANS), 5.0)
    frame = make_frame(data=original.copy())
    frame.add_noise(x_mean=10, x_std=1, noise_type='normal')
    result = frame.get_data()
    assert not np.array_equal(result, original)
    np.testing.assert_array_equal(result, frame.data)


def test_get_data_use_db_reflects_added_noise():
    frame = make_frame(data=np.ones((SMALL_TCHANS, SMALL_FCHANS)))
    frame.add_noise(x_mean=10, x_std=1, noise_type='normal')
    np.testing.assert_allclose(frame.get_data(use_db=True),
                               10 * np.log10(frame.data))
