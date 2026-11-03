import numpy as np
import pytest
from astropy import units as u

from setigen.frame import Frame


@pytest.fixture(autouse=True)
def _seed_rng():
    # Fix the global RNG so every statistical check is reproducible.
    np.random.seed(20241)
    yield


def _make_frame(fchans=64, tchans=32, df=2.0, dt=1.0, data=None):
    return Frame(fchans=fchans,
                 tchans=tchans,
                 df=df * u.Hz,
                 dt=dt * u.s,
                 fch1=8 * u.GHz,
                 data=data)


def _big_frame(df=2.0, dt=1.0):
    # 100k samples, enough for tight statistical tolerances
    return _make_frame(fchans=500, tchans=200, df=df, dt=dt)


# ---------------------------------------------------------------------------
# Frame.add_noise, chi-squared radiometer noise
# ---------------------------------------------------------------------------

def test_chi2_noise_shape_matches_frame():
    frame = _make_frame(fchans=40, tchans=12)
    noise = frame.add_noise(x_mean=10)
    assert noise.shape == (12, 40)
    assert np.issubdtype(noise.dtype, np.floating)


def test_chi2_noise_is_added_into_data():
    frame = _make_frame()
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_array_equal(frame.get_data(), noise)


def test_chi2_noise_mean_tracks_x_mean():
    frame = _big_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.mean(noise) == pytest.approx(10, rel=0.02)


def test_chi2_noise_std_matches_chi2_variance():
    frame = _big_frame()
    noise = frame.add_noise(x_mean=10)
    # chi2 with k=8 dof scaled by x_mean/k has std sqrt(2k) * x_mean / k = 5
    assert np.std(noise) == pytest.approx(5.0, rel=0.03)


def test_chi2_noise_is_nonnegative():
    frame = _big_frame()
    noise = frame.add_noise(x_mean=3)
    assert np.all(noise >= 0)


def test_chi2_dof_comes_from_df_times_dt():
    frame = _make_frame(df=4.0, dt=1.0)
    assert frame.chi2_df == 16
    frame.add_noise(x_mean=10)
    # std = sqrt(2 * 16) * 10 / 16
    assert frame.get_noise_stats()[1] == pytest.approx(np.sqrt(32) * 10 / 16)


def test_chi2_sets_noise_stats_from_parameters_on_first_call():
    frame = _make_frame()
    frame.add_noise(x_mean=10)
    noise_mean, noise_std = frame.get_noise_stats()
    assert noise_mean == pytest.approx(10)
    assert noise_std == pytest.approx(np.sqrt(16) * 10 / 8)


def test_chi2_ignores_x_std_and_x_min():
    frame = _make_frame()
    frame.add_noise(x_mean=10, x_std=999, x_min=-5, noise_type='chi2')
    assert frame.get_noise_stats()[1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Frame.add_noise, Gaussian noise ('normal' / 'gaussian')
# ---------------------------------------------------------------------------

def test_gaussian_noise_mean_and_std_match_parameters():
    frame = _big_frame()
    noise = frame.add_noise(x_mean=5, x_std=2, noise_type='normal')
    assert np.mean(noise) == pytest.approx(5, abs=0.05)
    assert np.std(noise) == pytest.approx(2, rel=0.03)


def test_gaussian_alias_matches_normal():
    a_frame = _make_frame()
    b_frame = _make_frame()
    np.random.seed(7)
    a = a_frame.add_noise(x_mean=1, x_std=3, noise_type='normal')
    np.random.seed(7)
    b = b_frame.add_noise(x_mean=1, x_std=3, noise_type='gaussian')
    np.testing.assert_array_equal(a, b)


def test_gaussian_sets_noise_stats_to_parameters_on_first_call():
    frame = _make_frame()
    frame.add_noise(x_mean=4, x_std=1.5, noise_type='gaussian')
    noise_mean, noise_std = frame.get_noise_stats()
    assert noise_mean == pytest.approx(4)
    assert noise_std == pytest.approx(1.5)


def test_gaussian_without_x_std_exits_and_leaves_data_alone():
    frame = _make_frame()
    with pytest.raises(SystemExit, match='x_std'):
        frame.add_noise(x_mean=4, noise_type='normal')
    np.testing.assert_array_equal(frame.get_data(), np.zeros(frame.shape))


def test_unknown_noise_type_exits_and_leaves_data_alone():
    frame = _make_frame()
    with pytest.raises(SystemExit, match='not a valid noise type'):
        frame.add_noise(x_mean=4, x_std=1, noise_type='uniform')
    np.testing.assert_array_equal(frame.get_data(), np.zeros(frame.shape))


def test_truncated_gaussian_clips_at_x_min():
    frame = _big_frame()
    noise = frame.add_noise(x_mean=0, x_std=1, x_min=0, noise_type='normal')
    assert np.all(noise >= 0)
    assert np.any(noise == 0)


def test_truncation_with_distant_x_min_matches_plain_gaussian():
    plain = _make_frame()
    truncated = _make_frame()
    np.random.seed(99)
    a = plain.add_noise(x_mean=5, x_std=1, noise_type='normal')
    np.random.seed(99)
    b = truncated.add_noise(x_mean=5, x_std=1, x_min=-1e6, noise_type='normal')
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Frame.add_noise, accumulation and bookkeeping of noise statistics
# ---------------------------------------------------------------------------

def test_repeated_calls_accumulate_into_data():
    frame = _make_frame()
    n1 = frame.add_noise(x_mean=10)
    n2 = frame.add_noise(x_mean=3, x_std=1, noise_type='normal')
    np.testing.assert_array_equal(frame.get_data(), n1 + n2)


def test_second_call_recomputes_noise_stats_from_data():
    frame = _big_frame()
    frame.add_noise(x_mean=10)
    frame.add_noise(x_mean=10)
    noise_mean, _ = frame.get_noise_stats()
    assert noise_mean == pytest.approx(20, rel=0.05)


def test_zero_data_resets_noise_stats_so_next_call_sets_them_again():
    frame = _make_frame()
    frame.add_noise(x_mean=10)
    frame.zero_data()
    frame.add_noise(x_mean=3, x_std=1, noise_type='normal')
    assert frame.get_noise_stats() == pytest.approx((3, 1))


# ---------------------------------------------------------------------------
# Frame.get_data
# ---------------------------------------------------------------------------

def test_get_data_returns_frame_data_array_itself():
    frame = _make_frame()
    frame.add_noise(x_mean=10)
    assert frame.get_data() is frame.data


def test_get_data_db_is_ten_log10_of_linear_values():
    data = np.array([[1.0, 10.0], [100.0, 1000.0]])
    frame = Frame(fchans=2, tchans=2, df=2 * u.Hz, dt=1 * u.s,
                  fch1=8 * u.GHz, data=data)
    np.testing.assert_allclose(frame.get_data(use_db=True),
                               [[0.0, 10.0], [20.0, 30.0]])


def test_get_data_db_does_not_modify_stored_data():
    data = np.array([[1.0, 10.0], [100.0, 1000.0]])
    frame = Frame(fchans=2, tchans=2, df=2 * u.Hz, dt=1 * u.s,
                  fch1=8 * u.GHz, data=data.copy())
    frame.get_data(use_db=True)
    np.testing.assert_array_equal(frame.data, data)


def test_get_data_shape_is_tchans_by_fchans():
    frame = _make_frame(fchans=50, tchans=20)
    frame.add_noise(x_mean=10)
    assert frame.get_data().shape == (20, 50)
    assert frame.get_data(use_db=True).shape == (20, 50)


def test_get_data_after_zero_data_is_all_zeros():
    frame = _make_frame()
    frame.add_noise(x_mean=10)
    frame.zero_data()
    np.testing.assert_array_equal(frame.get_data(), np.zeros((32, 64)))
