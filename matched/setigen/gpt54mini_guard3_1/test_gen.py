import numpy as np
import pytest
from astropy import units as u

from setigen.frame import Frame


def make_frame(data=None):
    if data is None:
        data = np.zeros((4, 6), dtype=float)
    return Frame(
        fchans=data.shape[1],
        tchans=data.shape[0],
        df=1 * u.Hz,
        dt=2 * u.s,
        fch1=10 * u.Hz,
        data=np.array(data, dtype=float),
    )


def test_get_data_returns_internal_array_when_use_db_false():
    data = np.arange(24, dtype=float).reshape(4, 6)
    frame = make_frame(data=data)

    returned = frame.get_data(use_db=False)

    assert returned is frame.data
    np.testing.assert_array_equal(returned, data)


def test_get_data_default_is_same_as_use_db_false():
    data = np.arange(24, dtype=float).reshape(4, 6)
    frame = make_frame(data=data)

    returned = frame.get_data()

    assert returned is frame.data
    np.testing.assert_array_equal(returned, data)


def test_get_data_use_db_applies_10_log10():
    data = np.array([[1.0, 10.0, 100.0], [0.1, 0.5, 5.0]], dtype=float)
    frame = Frame(fchans=3, tchans=2, df=1 * u.Hz, dt=1 * u.s, fch1=5 * u.Hz, data=data)

    db_data = frame.get_data(use_db=True)

    expected = 10 * np.log10(data)
    np.testing.assert_allclose(db_data, expected)


def test_get_data_use_db_does_not_modify_frame_data():
    data = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=float)
    frame = Frame(fchans=2, tchans=2, df=1 * u.Hz, dt=1 * u.s, fch1=5 * u.Hz, data=data)

    _ = frame.get_data(use_db=True)

    np.testing.assert_array_equal(frame.data, data)


def test_add_noise_chi2_updates_data_and_returns_noise(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 3.5, dtype=float)

    called = {}

    def fake_chi2(x_mean, chi2_df, shape):
        called["args"] = (x_mean, chi2_df, shape)
        return noise

    monkeypatch.setattr("setigen.frame.distributions.chi2", fake_chi2)

    returned = frame.add_noise(x_mean=7.0, noise_type="chi2")

    assert called["args"] == (7.0, frame.chi2_df, frame.shape)
    np.testing.assert_array_equal(returned, noise)
    np.testing.assert_array_equal(frame.data, noise)
    assert frame.noise_mean == 7.0
    expected_std = np.sqrt(2 * frame.chi2_df) * 7.0 / frame.chi2_df
    assert frame.noise_std == pytest.approx(expected_std)


def test_add_noise_chi2_accumulates_onto_existing_data(monkeypatch):
    frame = make_frame(data=np.ones((4, 6), dtype=float))
    noise = np.full(frame.shape, 2.0, dtype=float)

    monkeypatch.setattr("setigen.frame.distributions.chi2", lambda x_mean, chi2_df, shape: noise)

    frame.add_noise(x_mean=1.5, noise_type="chi2")

    np.testing.assert_array_equal(frame.data, np.ones((4, 6), dtype=float) + noise)


def test_add_noise_normal_with_x_std_uses_gaussian(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 4.0, dtype=float)

    called = {}

    def fake_gaussian(x_mean, x_std, shape):
        called["args"] = (x_mean, x_std, shape)
        return noise

    monkeypatch.setattr("setigen.frame.distributions.gaussian", fake_gaussian)

    returned = frame.add_noise(x_mean=2.5, x_std=0.75, noise_type="normal")

    assert called["args"] == (2.5, 0.75, frame.shape)
    np.testing.assert_array_equal(returned, noise)
    assert frame.noise_mean == 2.5
    assert frame.noise_std == 0.75


def test_add_noise_gaussian_alias_uses_gaussian(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 1.25, dtype=float)

    monkeypatch.setattr("setigen.frame.distributions.gaussian", lambda x_mean, x_std, shape: noise)

    returned = frame.add_noise(x_mean=3.0, x_std=0.5, noise_type="gaussian")

    np.testing.assert_array_equal(returned, noise)
    np.testing.assert_array_equal(frame.data, noise)


def test_add_noise_truncated_gaussian_uses_truncated_gaussian(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 6.0, dtype=float)

    called = {}

    def fake_truncated_gaussian(x_mean, x_std, x_min, shape):
        called["args"] = (x_mean, x_std, x_min, shape)
        return noise

    monkeypatch.setattr("setigen.frame.distributions.truncated_gaussian", fake_truncated_gaussian)

    returned = frame.add_noise(x_mean=2.0, x_std=1.0, x_min=0.0, noise_type="gaussian")

    assert called["args"] == (2.0, 1.0, 0.0, frame.shape)
    np.testing.assert_array_equal(returned, noise)


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type="not-a-real-noise-type")


def test_add_noise_normal_requires_x_std():
    frame = make_frame()

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type="normal")


def test_add_noise_updates_stats_by_sigma_clipped_data_when_existing_noise_present(
    monkeypatch,
):
    frame = make_frame()
    frame.noise_mean = 1.0
    frame.noise_std = 2.0

    noise = np.full(frame.shape, 3.0, dtype=float)
    monkeypatch.setattr("setigen.frame.distributions.gaussian", lambda x_mean, x_std, shape: noise)

    frame.add_noise(x_mean=5.0, x_std=7.0, noise_type="normal")

    # Data is uniform, so sigma clipping should not change the statistics.
    np.testing.assert_array_equal(frame.data, noise)
    assert frame.noise_mean == pytest.approx(3.0)
    assert frame.noise_std == pytest.approx(0.0)


def test_add_noise_preserves_zero_noise_stats_on_first_addition_when_data_was_zero(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 9.0, dtype=float)
    monkeypatch.setattr("setigen.frame.distributions.gaussian", lambda x_mean, x_std, shape: noise)

    frame.add_noise(x_mean=4.0, x_std=2.0, noise_type="normal")

    assert frame.noise_mean == 4.0
    assert frame.noise_std == 2.0
    np.testing.assert_array_equal(frame.data, noise)


def test_get_data_use_db_with_integer_like_positive_values():
    data = np.array([[1, 2], [3, 4]], dtype=float)
    frame = make_frame(data=data)

    db_data = frame.get_data(use_db=True)

    np.testing.assert_allclose(db_data, 10 * np.log10(data))
