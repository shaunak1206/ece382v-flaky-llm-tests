import numpy as np
import pytest
from astropy import units as u

from setigen.frame import Frame


def make_frame(fchans=4, tchans=3, df=1 * u.Hz, dt=2 * u.s, fch1=10 * u.Hz):
    return Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)


def test_get_data_returns_raw_array():
    frame = make_frame()
    frame.data[:] = np.arange(frame.data.size).reshape(frame.shape)

    data = frame.get_data()

    assert data is frame.data
    np.testing.assert_array_equal(data, frame.data)


def test_get_data_use_db_applies_log10():
    frame = make_frame()
    frame.data[:] = np.array(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [2.0, 20.0, 200.0, 2000.0],
            [3.0, 30.0, 300.0, 3000.0],
        ]
    )

    db = frame.get_data(use_db=True)

    np.testing.assert_allclose(db, 10 * np.log10(frame.data))


def test_get_data_use_db_preserves_shape():
    frame = make_frame(fchans=5, tchans=2)
    frame.data[:] = np.arange(10, dtype=float).reshape(frame.shape) + 1

    db = frame.get_data(use_db=True)

    assert db.shape == frame.shape


def test_add_noise_chi2_updates_data_and_returns_noise(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 2.5)

    monkeypatch.setattr(
        "setigen.frame.distributions.chi2",
        lambda x_mean, df, shape: noise.copy(),
    )

    returned = frame.add_noise(x_mean=5.0, noise_type="chi2")

    np.testing.assert_array_equal(returned, noise)
    np.testing.assert_array_equal(frame.data, noise)
    assert frame.noise_mean == 5.0
    assert frame.noise_std == pytest.approx(
        np.sqrt(2 * frame.chi2_df) * 5.0 / frame.chi2_df
    )


def test_add_noise_normal_uses_gaussian_when_no_min(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 1.25)

    monkeypatch.setattr(
        "setigen.frame.distributions.gaussian",
        lambda x_mean, x_std, shape: noise.copy(),
    )
    monkeypatch.setattr(
        "setigen.frame.distributions.truncated_gaussian",
        lambda *args, **kwargs: pytest.fail("truncated_gaussian should not be used"),
    )

    returned = frame.add_noise(x_mean=1.0, x_std=0.5, noise_type="normal")

    np.testing.assert_array_equal(returned, noise)
    np.testing.assert_array_equal(frame.data, noise)
    assert frame.noise_mean == 1.0
    assert frame.noise_std == 0.5


def test_add_noise_gaussian_with_min_uses_truncated_gaussian(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 3.0)

    monkeypatch.setattr(
        "setigen.frame.distributions.truncated_gaussian",
        lambda x_mean, x_std, x_min, shape: noise.copy(),
    )
    monkeypatch.setattr(
        "setigen.frame.distributions.gaussian",
        lambda *args, **kwargs: pytest.fail("gaussian should not be used"),
    )

    returned = frame.add_noise(x_mean=2.0, x_std=1.0, x_min=0.0, noise_type="gaussian")

    np.testing.assert_array_equal(returned, noise)
    np.testing.assert_array_equal(frame.data, noise)
    assert frame.noise_mean == 2.0
    assert frame.noise_std == 1.0


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type="invalid")


def test_add_noise_normal_without_x_std_exits():
    frame = make_frame()

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type="normal")


def test_add_noise_updates_stats_on_second_call(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 2.0)

    monkeypatch.setattr(
        "setigen.frame.distributions.chi2",
        lambda x_mean, df, shape: noise.copy(),
    )

    frame.add_noise(x_mean=2.0, noise_type="chi2")
    assert frame.noise_mean == 2.0
    assert frame.noise_std > 0

    frame.add_noise(x_mean=2.0, noise_type="chi2")

    np.testing.assert_array_equal(frame.data, np.full(frame.shape, 4.0))
    assert frame.noise_mean == 4.0
    assert frame.noise_std == 0


def test_add_noise_adds_to_existing_data(monkeypatch):
    frame = make_frame()
    frame.data[:] = 10.0
    noise = np.full(frame.shape, 1.5)

    monkeypatch.setattr(
        "setigen.frame.distributions.gaussian",
        lambda x_mean, x_std, shape: noise.copy(),
    )

    frame.add_noise(x_mean=1.0, x_std=0.2, noise_type="normal")

    np.testing.assert_array_equal(frame.data, np.full(frame.shape, 11.5))


def test_add_noise_preserves_noise_return_value_independent_of_frame(monkeypatch):
    frame = make_frame()
    noise = np.arange(frame.data.size, dtype=float).reshape(frame.shape)

    monkeypatch.setattr(
        "setigen.frame.distributions.chi2",
        lambda x_mean, df, shape: noise.copy(),
    )

    returned = frame.add_noise(x_mean=7.0, noise_type="chi2")
    returned[0, 0] = -999.0

    assert frame.data[0, 0] != -999.0
    assert returned[0, 0] == -999.0


def test_get_data_use_db_after_noise_returns_expected_values(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 4.0)

    monkeypatch.setattr(
        "setigen.frame.distributions.chi2",
        lambda x_mean, df, shape: noise.copy(),
    )

    frame.add_noise(x_mean=4.0, noise_type="chi2")
    db = frame.get_data(use_db=True)

    np.testing.assert_allclose(db, 10 * np.log10(np.full(frame.shape, 4.0)))


def test_get_data_does_not_modify_frame_data():
    frame = make_frame()
    frame.data[:] = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )

    original = frame.data.copy()
    _ = frame.get_data(use_db=True)

    np.testing.assert_array_equal(frame.data, original)
