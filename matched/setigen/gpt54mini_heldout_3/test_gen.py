import numpy as np
import pytest

from setigen.frame import Frame


def make_frame():
    return Frame(fchans=4, tchans=3, df=1.0, dt=2.0, fch1=10.0)


def test_get_data_returns_internal_array_by_default():
    frame = make_frame()
    frame.data[:] = np.arange(frame.data.size).reshape(frame.shape)
    out = frame.get_data()
    assert out is frame.data
    np.testing.assert_array_equal(out, frame.data)


def test_get_data_db_converts_to_decibels():
    frame = make_frame()
    frame.data[:] = 10.0
    out = frame.get_data(use_db=True)
    expected = 10 * np.log10(frame.data)
    np.testing.assert_allclose(out, expected)


def test_get_data_db_with_zeros_returns_negative_infinity():
    frame = make_frame()
    out = frame.get_data(use_db=True)
    assert out.shape == frame.shape
    assert np.isneginf(out).all()


def test_add_noise_chi2_calls_distribution_with_expected_arguments(monkeypatch):
    frame = make_frame()
    captured = {}

    def fake_chi2(x_mean, df, shape):
        captured["args"] = (x_mean, df, shape)
        return np.full(shape, 2.5)

    monkeypatch.setattr("setigen.frame.distributions.chi2", fake_chi2)

    noise = frame.add_noise(x_mean=7.0, noise_type="chi2")

    assert captured["args"] == (7.0, frame.chi2_df, frame.shape)
    np.testing.assert_allclose(noise, np.full(frame.shape, 2.5))
    np.testing.assert_allclose(frame.data, np.full(frame.shape, 2.5))


def test_add_noise_chi2_sets_noise_stats_on_empty_frame(monkeypatch):
    frame = make_frame()

    monkeypatch.setattr(
        "setigen.frame.distributions.chi2",
        lambda x_mean, df, shape: np.ones(shape),
    )

    frame.add_noise(x_mean=3.0, noise_type="chi2")

    assert frame.noise_mean == 3.0
    expected_std = np.sqrt(2 * frame.chi2_df) * 3.0 / frame.chi2_df
    assert frame.noise_std == expected_std


def test_add_noise_normal_uses_gaussian_when_no_min(monkeypatch):
    frame = make_frame()
    captured = {}

    def fake_gaussian(x_mean, x_std, shape):
        captured["args"] = (x_mean, x_std, shape)
        return np.full(shape, x_mean + x_std)

    monkeypatch.setattr("setigen.frame.distributions.gaussian", fake_gaussian)

    noise = frame.add_noise(x_mean=2.0, x_std=5.0, noise_type="normal")

    assert captured["args"] == (2.0, 5.0, frame.shape)
    np.testing.assert_allclose(noise, np.full(frame.shape, 7.0))


def test_add_noise_gaussian_alias_uses_gaussian(monkeypatch):
    frame = make_frame()
    monkeypatch.setattr(
        "setigen.frame.distributions.gaussian",
        lambda x_mean, x_std, shape: np.full(shape, x_mean - x_std),
    )

    noise = frame.add_noise(x_mean=4.0, x_std=1.5, noise_type="gaussian")

    np.testing.assert_allclose(noise, np.full(frame.shape, 2.5))


def test_add_noise_normal_with_x_min_uses_truncated_gaussian(monkeypatch):
    frame = make_frame()
    captured = {}

    def fake_truncated_gaussian(x_mean, x_std, x_min, shape):
        captured["args"] = (x_mean, x_std, x_min, shape)
        return np.full(shape, x_min)

    monkeypatch.setattr(
        "setigen.frame.distributions.truncated_gaussian",
        fake_truncated_gaussian,
    )

    noise = frame.add_noise(
        x_mean=5.0, x_std=2.0, x_min=1.25, noise_type="normal"
    )

    assert captured["args"] == (5.0, 2.0, 1.25, frame.shape)
    np.testing.assert_allclose(noise, np.full(frame.shape, 1.25))


def test_add_noise_normal_without_x_std_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, noise_type="normal")


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()
    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, x_std=2.0, noise_type="not-a-real-type")


def test_add_noise_adds_noise_in_place(monkeypatch):
    frame = make_frame()
    frame.data[:] = 10.0

    monkeypatch.setattr(
        "setigen.frame.distributions.chi2",
        lambda x_mean, df, shape: np.full(shape, 0.5),
    )

    frame.add_noise(x_mean=1.0, noise_type="chi2")
    np.testing.assert_allclose(frame.data, np.full(frame.shape, 10.5))


def test_add_noise_returns_noise_array_without_aliasing_frame_data(monkeypatch):
    frame = make_frame()

    def fake_chi2(x_mean, df, shape):
        return np.arange(np.prod(shape), dtype=float).reshape(shape)

    monkeypatch.setattr("setigen.frame.distributions.chi2", fake_chi2)

    noise = frame.add_noise(x_mean=1.0, noise_type="chi2")
    assert noise is not frame.data
    np.testing.assert_allclose(frame.data, noise)


def test_add_noise_updates_stats_when_frame_already_has_noise(monkeypatch):
    frame = make_frame()
    frame.noise_mean = 1.0
    frame.noise_std = 2.0

    monkeypatch.setattr(
        "setigen.frame.distributions.chi2",
        lambda x_mean, df, shape: np.ones(shape),
    )

    called = {}

    def fake_update_noise_frame_stats():
        called["yes"] = True
        frame.noise_mean = 9.0
        frame.noise_std = 8.0

    monkeypatch.setattr(frame, "_update_noise_frame_stats", fake_update_noise_frame_stats)

    frame.add_noise(x_mean=3.0, noise_type="chi2")

    assert called["yes"] is True
    assert frame.noise_mean == 9.0
    assert frame.noise_std == 8.0


def test_add_noise_normal_share_minimum_and_stats_when_frame_already_has_noise(monkeypatch):
    frame = make_frame()
    frame.noise_mean = 1.0
    frame.noise_std = 2.0

    monkeypatch.setattr(
        "setigen.frame.distributions.truncated_gaussian",
        lambda x_mean, x_std, x_min, shape: np.full(shape, x_min),
    )

    called = {}

    def fake_update_noise_frame_stats():
        called["yes"] = True
        frame.noise_mean = 6.0
        frame.noise_std = 7.0

    monkeypatch.setattr(frame, "_update_noise_frame_stats", fake_update_noise_frame_stats)

    frame.add_noise(x_mean=5.0, x_std=1.0, x_min=0.0, noise_type="gaussian")

    assert called["yes"] is True
    assert frame.noise_mean == 6.0
    assert frame.noise_std == 7.0


def test_add_noise_chi2_does_not_call_gaussian_functions(monkeypatch):
    frame = make_frame()

    def fail(*args, **kwargs):
        raise AssertionError("This function should not have been called")

    monkeypatch.setattr("setigen.frame.distributions.gaussian", fail)
    monkeypatch.setattr("setigen.frame.distributions.truncated_gaussian", fail)
    monkeypatch.setattr(
        "setigen.frame.distributions.chi2",
        lambda x_mean, df, shape: np.zeros(shape),
    )

    noise = frame.add_noise(x_mean=2.0, noise_type="chi2")
    np.testing.assert_allclose(noise, np.zeros(frame.shape))
