import numpy as np
import pytest
from astropy import units as u

from setigen.frame import Frame


def make_frame(fchans=4, tchans=3, df=1 * u.Hz, dt=1 * u.s, fch1=10 * u.Hz):
    return Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)


def test_get_data_returns_underlying_array_reference():
    frame = make_frame()
    data = frame.get_data()
    assert data is frame.data
    data[0, 0] = 123.0
    assert frame.data[0, 0] == 123.0


def test_get_data_without_db_returns_unchanged_data():
    frame = make_frame()
    frame.data[:] = np.arange(frame.data.size).reshape(frame.shape)
    out = frame.get_data()
    assert np.array_equal(out, frame.data)


def test_get_data_with_db_applies_log10_scaling():
    frame = make_frame()
    frame.data[:] = 10.0
    out = frame.get_data(use_db=True)
    assert np.allclose(out, 10.0)


def test_get_data_with_db_on_zeros_produces_negative_infinity():
    frame = make_frame()
    out = frame.get_data(use_db=True)
    assert np.isneginf(out).all()


def test_add_noise_chi2_updates_data_and_noise_stats(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 2.5)

    def fake_chi2(x_mean, chi2_df, shape):
        assert x_mean == 7.0
        assert chi2_df == frame.chi2_df
        assert shape == frame.shape
        return noise

    monkeypatch.setattr("setigen.distributions.chi2", fake_chi2)

    out = frame.add_noise(x_mean=7.0, noise_type="chi2")
    assert np.array_equal(out, noise)
    assert np.array_equal(frame.data, noise)
    assert frame.noise_mean == 7.0
    assert frame.noise_std == pytest.approx(np.sqrt(2 * frame.chi2_df) * 7.0 / frame.chi2_df)


def test_add_noise_chi2_keeps_existing_noise_stats_updated(monkeypatch):
    frame = make_frame()
    frame.noise_mean = 1.0
    frame.noise_std = 2.0
    noise = np.ones(frame.shape)

    monkeypatch.setattr("setigen.distributions.chi2", lambda *args, **kwargs: noise)

    called = {"count": 0}

    def fake_update():
        called["count"] += 1

    monkeypatch.setattr(frame, "_update_noise_frame_stats", fake_update)

    frame.add_noise(x_mean=3.0, noise_type="chi2")
    assert called["count"] == 1


def test_add_noise_normal_uses_gaussian_when_no_min(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 4.0)

    def fake_gaussian(x_mean, x_std, shape):
        assert x_mean == 2.0
        assert x_std == 0.5
        assert shape == frame.shape
        return noise

    monkeypatch.setattr("setigen.distributions.gaussian", fake_gaussian)

    out = frame.add_noise(x_mean=2.0, x_std=0.5, noise_type="normal")
    assert np.array_equal(out, noise)
    assert np.array_equal(frame.data, noise)
    assert frame.noise_mean == 2.0
    assert frame.noise_std == 0.5


def test_add_noise_gaussian_alias_works(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 1.25)

    monkeypatch.setattr(
        "setigen.distributions.gaussian",
        lambda x_mean, x_std, shape: noise,
    )

    out = frame.add_noise(x_mean=1.0, x_std=0.25, noise_type="gaussian")
    assert np.array_equal(out, noise)
    assert np.array_equal(frame.data, noise)


def test_add_noise_normal_with_x_min_uses_truncated_gaussian(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 3.0)

    def fake_trunc(x_mean, x_std, x_min, shape):
        assert x_mean == 2.0
        assert x_std == 0.5
        assert x_min == 0.0
        assert shape == frame.shape
        return noise

    monkeypatch.setattr("setigen.distributions.truncated_gaussian", fake_trunc)

    out = frame.add_noise(x_mean=2.0, x_std=0.5, x_min=0.0, noise_type="normal")
    assert np.array_equal(out, noise)
    assert np.array_equal(frame.data, noise)


def test_add_noise_normal_requires_x_std():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type="normal")


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, x_std=0.1, noise_type="not-a-type")


def test_add_noise_accumulates_into_existing_data(monkeypatch):
    frame = make_frame()
    frame.data[:] = 1.0
    noise = np.full(frame.shape, 2.0)

    monkeypatch.setattr("setigen.distributions.chi2", lambda *args, **kwargs: noise)

    frame.add_noise(x_mean=5.0, noise_type="chi2")
    assert np.allclose(frame.data, 3.0)


def test_add_noise_updates_stats_after_existing_noise(monkeypatch):
    frame = make_frame()
    frame.noise_mean = 1.0
    frame.noise_std = 1.0
    noise = np.full(frame.shape, 2.0)

    monkeypatch.setattr("setigen.distributions.chi2", lambda *args, **kwargs: noise)

    def fake_update():
        frame.noise_mean = 9.0
        frame.noise_std = 8.0

    monkeypatch.setattr(frame, "_update_noise_frame_stats", fake_update)

    frame.add_noise(x_mean=5.0, noise_type="chi2")
    assert frame.noise_mean == 9.0
    assert frame.noise_std == 8.0


def test_get_data_reflects_noise_added_to_frame(monkeypatch):
    frame = make_frame()
    noise = np.arange(frame.data.size, dtype=float).reshape(frame.shape)

    monkeypatch.setattr("setigen.distributions.chi2", lambda *args, **kwargs: noise)

    frame.add_noise(x_mean=4.0, noise_type="chi2")
    assert np.array_equal(frame.get_data(), noise)
    assert np.array_equal(frame.get_data(use_db=False), noise)
