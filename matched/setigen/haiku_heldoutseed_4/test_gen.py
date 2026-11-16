import numpy as np
import pytest
from astropy import units as u
from astropy.stats import sigma_clip

from setigen.frame import Frame


@pytest.fixture(autouse=True)
def seed_global_rng():
    """Make every test deterministic."""
    np.random.seed(12345)
    yield


def make_frame(fchans=16, tchans=8, df=1.0, dt=1.0, data=None):
    """Build a small Frame with physically sensible resolutions."""
    return Frame(fchans=fchans,
                 tchans=tchans,
                 df=df * u.Hz,
                 dt=dt * u.s,
                 fch1=8 * u.GHz,
                 data=data)


# ---------------------------------------------------------------------------
# Frame.add_noise
# ---------------------------------------------------------------------------

def test_add_noise_chi2_returns_frame_shape():
    frame = make_frame(fchans=16, tchans=8)
    noise = frame.add_noise(x_mean=10)
    assert noise.shape == (8, 16)
    assert noise.shape == frame.shape


def test_add_noise_chi2_is_default_noise_type():
    frame_default = make_frame()
    frame_explicit = make_frame()

    np.random.seed(99)
    noise_default = frame_default.add_noise(x_mean=10)
    np.random.seed(99)
    noise_explicit = frame_explicit.add_noise(x_mean=10, noise_type='chi2')

    assert np.array_equal(noise_default, noise_explicit)


def test_add_noise_chi2_is_added_to_zero_data():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    # Frame starts as zeros, so data must equal exactly the returned noise
    assert np.array_equal(frame.get_data(), noise)


def test_add_noise_chi2_mean_matches_x_mean():
    frame = make_frame(fchans=200, tchans=200)
    noise = frame.add_noise(x_mean=10)
    assert np.mean(noise) == pytest.approx(10, abs=0.3)


def test_add_noise_chi2_std_matches_theory():
    # For chi2 with chi2_df=4, std = sqrt(2*4) * x_mean / 4 ~= 0.7071 * x_mean
    frame = make_frame(fchans=200, tchans=200, df=1.0, dt=1.0)
    x_mean = 10
    noise = frame.add_noise(x_mean=x_mean)
    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    assert np.std(noise) == pytest.approx(expected_std, rel=0.05)


def test_add_noise_chi2_ignores_x_std_and_x_min():
    frame_a = make_frame()
    frame_b = make_frame()

    np.random.seed(5)
    noise_a = frame_a.add_noise(x_mean=10)
    np.random.seed(5)
    noise_b = frame_b.add_noise(x_mean=10, x_std=1000, x_min=500)

    assert np.array_equal(noise_a, noise_b)


def test_add_noise_chi2_sets_noise_stats_on_empty_frame():
    frame = make_frame(df=2.5, dt=2.0)  # chi2_df = 4 * round(5) = 20
    x_mean = 7.0
    frame.add_noise(x_mean=x_mean)

    expected_std = np.sqrt(2 * frame.chi2_df) * x_mean / frame.chi2_df
    noise_mean, noise_std = frame.get_noise_stats()
    assert noise_mean == pytest.approx(x_mean)
    assert noise_std == pytest.approx(expected_std)


def test_add_noise_chi2_std_depends_on_chi2_df():
    # df * dt = 2 -> chi2_df = 8 -> std = sqrt(16) * x_mean / 8 = x_mean / 2
    frame = make_frame(df=2.0, dt=1.0)
    assert frame.chi2_df == 8
    frame.add_noise(x_mean=10)
    _, noise_std = frame.get_noise_stats()
    assert noise_std == pytest.approx(5.0)


def test_add_noise_accumulates_on_existing_data():
    preloaded = np.ones((8, 16))
    frame = make_frame(data=preloaded.copy())
    noise = frame.add_noise(x_mean=10)
    assert np.allclose(frame.get_data(), preloaded + noise)


def test_add_noise_second_call_recomputes_noise_stats():
    frame = make_frame(fchans=64, tchans=64)
    frame.add_noise(x_mean=10)
    # After the second call the frame holds roughly twice the intensity, so
    # stats must be recomputed from the data rather than the parameters.
    frame.add_noise(x_mean=10)

    expected_mean = np.mean(sigma_clip(frame.data, sigma=3, maxiters=5,
                                       masked=False))
    noise_mean, _ = frame.get_noise_stats()
    assert noise_mean == pytest.approx(expected_mean)
    assert noise_mean == pytest.approx(20, abs=1.0)


def test_add_noise_gaussian_returns_shape_and_matches_params():
    frame = make_frame(fchans=200, tchans=200)
    noise = frame.add_noise(x_mean=5, x_std=2, noise_type='gaussian')
    assert noise.shape == frame.shape
    assert np.mean(noise) == pytest.approx(5, abs=0.1)
    assert np.std(noise) == pytest.approx(2, abs=0.1)


def test_add_noise_normal_and_gaussian_are_equivalent():
    frame_a = make_frame()
    frame_b = make_frame()

    np.random.seed(21)
    noise_a = frame_a.add_noise(x_mean=5, x_std=2, noise_type='normal')
    np.random.seed(21)
    noise_b = frame_b.add_noise(x_mean=5, x_std=2, noise_type='gaussian')

    assert np.array_equal(noise_a, noise_b)


def test_add_noise_gaussian_without_x_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='gaussian')


def test_add_noise_gaussian_with_x_min_respects_lower_bound():
    frame = make_frame(fchans=100, tchans=100)
    noise = frame.add_noise(x_mean=0, x_std=5, x_min=0,
                            noise_type='gaussian')
    assert noise.min() >= 0
    assert noise.max() > 0


def test_add_noise_gaussian_sets_noise_stats_on_empty_frame():
    frame = make_frame()
    frame.add_noise(x_mean=5, x_std=2, noise_type='gaussian')
    assert frame.get_noise_stats() == (5, 2)


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=5, noise_type='poisson')


def test_add_noise_same_seed_is_reproducible():
    frame_a = make_frame()
    frame_b = make_frame()

    np.random.seed(2024)
    noise_a = frame_a.add_noise(x_mean=10)
    np.random.seed(2024)
    noise_b = frame_b.add_noise(x_mean=10)

    assert np.array_equal(noise_a, noise_b)
    assert np.array_equal(frame_a.get_data(), frame_b.get_data())


# ---------------------------------------------------------------------------
# Frame.get_data
# ---------------------------------------------------------------------------

def test_get_data_default_returns_underlying_data():
    frame = make_frame()
    data = frame.get_data()
    assert data is frame.data
    assert data.shape == frame.shape
    assert np.array_equal(data, np.zeros(frame.shape))


def test_get_data_returns_preloaded_data():
    preloaded = np.arange(8 * 16, dtype=float).reshape(8, 16) + 1.0
    frame = make_frame(data=preloaded.copy())
    assert np.array_equal(frame.get_data(), preloaded)


def test_get_data_use_db_converts_to_decibels():
    values = np.array([[1.0, 10.0], [100.0, 1000.0]])
    frame = make_frame(fchans=2, tchans=2, data=values.copy())
    expected = np.array([[0.0, 10.0], [20.0, 30.0]])
    assert np.allclose(frame.get_data(use_db=True), expected)


def test_get_data_use_db_false_matches_default():
    values = np.random.rand(8, 16) + 0.5
    frame = make_frame(data=values.copy())
    assert np.array_equal(frame.get_data(use_db=False), frame.get_data())


def test_get_data_use_db_on_zero_frame_is_neg_inf():
    frame = make_frame()
    with np.errstate(divide='ignore'):
        db = frame.get_data(use_db=True)
    assert np.all(np.isneginf(db))


def test_get_data_use_db_does_not_modify_stored_data():
    values = np.full((8, 16), 10.0)
    frame = make_frame(data=values.copy())
    frame.get_data(use_db=True)
    assert np.array_equal(frame.data, values)


def test_get_data_reflects_added_noise():
    frame = make_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.array_equal(frame.get_data(), noise)
    assert np.allclose(frame.get_data(use_db=True), 10 * np.log10(noise))
