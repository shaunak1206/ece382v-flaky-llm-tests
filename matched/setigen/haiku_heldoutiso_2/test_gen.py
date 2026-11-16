import numpy as np
import pytest
from astropy import units as u

from setigen.frame import Frame


LARGE_FCHANS = 200
LARGE_TCHANS = 200


def make_frame(fchans=8, tchans=4, df=1.0, dt=1.0, data=None):
    """Build a small Frame with df in Hz and dt in seconds."""
    return Frame(fchans=fchans,
                 tchans=tchans,
                 df=df * u.Hz,
                 dt=dt * u.s,
                 fch1=8 * u.GHz,
                 data=data)


def make_large_frame(data=None):
    """Build a frame with enough samples for stable statistical checks."""
    return make_frame(fchans=LARGE_FCHANS,
                      tchans=LARGE_TCHANS,
                      df=1.0,
                      dt=1.0,
                      data=data)


# ---------------------------------------------------------------------------
# Frame.add_noise
# ---------------------------------------------------------------------------

def test_add_noise_chi2_returns_array_matching_frame_shape():
    frame = make_frame(fchans=8, tchans=4)
    noise = frame.add_noise(x_mean=10)
    assert noise.shape == (4, 8)
    assert frame.data.shape == (4, 8)


def test_add_noise_chi2_adds_returned_noise_to_zero_frame():
    frame = make_frame(fchans=8, tchans=4)
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.data, noise)


def test_add_noise_chi2_samples_are_nonnegative():
    frame = make_large_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.all(noise >= 0)


def test_add_noise_chi2_sample_mean_close_to_x_mean():
    frame = make_large_frame()
    noise = frame.add_noise(x_mean=10)
    assert np.mean(noise) == pytest.approx(10, rel=0.05)


def test_add_noise_chi2_sets_noise_stats_from_ideal_formula():
    # df=1 Hz, dt=1 s -> chi2_df = 4 * round(1) = 4
    frame = make_frame(df=1.0, dt=1.0)
    frame.add_noise(x_mean=10)
    noise_mean, noise_std = frame.get_noise_stats()
    expected_std = np.sqrt(2 * 4) * 10 / 4
    assert noise_mean == pytest.approx(10)
    assert noise_std == pytest.approx(expected_std)


def test_add_noise_chi2_std_formula_depends_on_chi2_df():
    # df=2 Hz, dt=1 s -> chi2_df = 4 * round(2) = 8
    frame = make_frame(df=2.0, dt=1.0)
    assert frame.chi2_df == 8
    frame.add_noise(x_mean=10)
    _, noise_std = frame.get_noise_stats()
    assert noise_std == pytest.approx(np.sqrt(2 * 8) * 10 / 8)


def test_add_noise_chi2_ignores_x_std_argument():
    frame = make_frame(df=1.0, dt=1.0)
    frame.add_noise(x_mean=10, x_std=123, noise_type='chi2')
    _, noise_std = frame.get_noise_stats()
    assert noise_std == pytest.approx(np.sqrt(2 * 4) * 10 / 4)


def test_add_noise_second_call_accumulates_into_data():
    frame = make_large_frame()
    frame.add_noise(x_mean=10)
    frame.add_noise(x_mean=10)
    assert np.mean(frame.data) == pytest.approx(20, rel=0.05)


def test_add_noise_second_call_updates_noise_stats_from_data():
    frame = make_large_frame()
    frame.add_noise(x_mean=10)
    frame.add_noise(x_mean=10)
    noise_mean, _ = frame.get_noise_stats()
    assert noise_mean == pytest.approx(20, rel=0.05)


def test_add_noise_preserves_existing_data_and_returns_only_noise():
    base = np.ones((4, 8))
    frame = make_frame(fchans=8, tchans=4, data=base.copy())
    noise = frame.add_noise(x_mean=5)
    np.testing.assert_allclose(frame.data - noise, base)


def test_add_noise_gaussian_sample_mean_and_std_match_parameters():
    frame = make_large_frame()
    noise = frame.add_noise(x_mean=10, x_std=2, noise_type='gaussian')
    assert np.mean(noise) == pytest.approx(10, abs=0.1)
    assert np.std(noise) == pytest.approx(2, rel=0.05)


def test_add_noise_normal_is_accepted_as_alias_for_gaussian():
    frame = make_large_frame()
    noise = frame.add_noise(x_mean=10, x_std=2, noise_type='normal')
    assert noise.shape == (LARGE_TCHANS, LARGE_FCHANS)
    assert np.mean(noise) == pytest.approx(10, abs=0.1)
    assert np.std(noise) == pytest.approx(2, rel=0.05)


def test_add_noise_gaussian_sets_noise_stats_from_parameters_on_zero_frame():
    frame = make_frame(fchans=8, tchans=4)
    frame.add_noise(x_mean=3, x_std=0.5, noise_type='gaussian')
    assert frame.get_noise_stats() == (3, 0.5)


def test_add_noise_gaussian_without_x_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, noise_type='gaussian')


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=10, x_std=1, noise_type='poisson')


def test_add_noise_truncated_gaussian_respects_x_min():
    frame = make_large_frame()
    noise = frame.add_noise(x_mean=0, x_std=1, x_min=0, noise_type='gaussian')
    assert np.all(noise >= 0)
    assert np.any(noise > 0)


def test_add_noise_gaussian_without_x_min_can_produce_negative_values():
    frame = make_large_frame()
    noise = frame.add_noise(x_mean=0, x_std=1, noise_type='gaussian')
    assert np.min(noise) < 0


# ---------------------------------------------------------------------------
# Frame.get_data
# ---------------------------------------------------------------------------

def test_get_data_default_returns_stored_data_object():
    frame = make_frame()
    assert frame.get_data() is frame.data


def test_get_data_use_db_false_returns_linear_data():
    values = np.array([[1.0, 10.0, 100.0, 1000.0],
                       [2.0, 20.0, 200.0, 2000.0]])
    frame = make_frame(fchans=4, tchans=2, data=values.copy())
    np.testing.assert_allclose(frame.get_data(use_db=False), values)


def test_get_data_use_db_converts_to_decibels():
    values = np.array([[1.0, 10.0],
                       [100.0, 1000.0]])
    frame = make_frame(fchans=2, tchans=2, data=values.copy())
    expected = np.array([[0.0, 10.0],
                         [20.0, 30.0]])
    np.testing.assert_allclose(frame.get_data(use_db=True), expected)


def test_get_data_use_db_does_not_modify_stored_data():
    values = np.array([[1.0, 10.0],
                       [100.0, 1000.0]])
    frame = make_frame(fchans=2, tchans=2, data=values.copy())
    frame.get_data(use_db=True)
    np.testing.assert_allclose(frame.data, values)


def test_get_data_use_db_returns_nan_or_neginf_for_nonpositive_values():
    values = np.array([[0.0, -1.0]])
    frame = make_frame(fchans=2, tchans=1, data=values.copy())
    with np.errstate(divide='ignore', invalid='ignore'):
        result = frame.get_data(use_db=True)
    assert np.isneginf(result[0, 0])
    assert np.isnan(result[0, 1])


def test_get_data_use_db_preserves_shape():
    frame = make_frame(fchans=8, tchans=4, data=np.full((4, 8), 2.0))
    result = frame.get_data(use_db=True)
    assert result.shape == (4, 8)
    np.testing.assert_allclose(result, 10 * np.log10(2.0))


def test_get_data_reflects_noise_added_by_add_noise():
    frame = make_frame(fchans=8, tchans=4)
    noise = frame.add_noise(x_mean=10)
    np.testing.assert_allclose(frame.get_data(), noise)
    assert frame.get_data().shape == (4, 8)


def test_get_data_returns_zeros_after_zero_data():
    frame = make_frame(fchans=8, tchans=4, data=np.ones((4, 8)))
    frame.zero_data()
    np.testing.assert_array_equal(frame.get_data(), np.zeros((4, 8)))
