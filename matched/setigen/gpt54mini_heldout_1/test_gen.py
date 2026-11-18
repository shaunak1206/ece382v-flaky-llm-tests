import numpy as np
import pytest

from setigen.frame import Frame
import setigen.frame as frame_module


def make_frame():
    return Frame(fchans=4, tchans=3, df=1, dt=1, fch1=10)


def test_get_data_returns_internal_array_by_default():
    frame = make_frame()
    frame.data[0, 0] = 7.5

    out = frame.get_data()

    assert out is frame.data
    assert out.shape == (3, 4)
    assert out[0, 0] == 7.5


def test_get_data_db_converts_to_decibels():
    frame = make_frame()
    frame.data[:] = 10.0

    out = frame.get_data(use_db=True)

    assert np.allclose(out, 10 * np.log10(10.0))
    assert out.shape == frame.data.shape


def test_get_data_db_on_zeros_creates_negative_infinity():
    frame = make_frame()

    out = frame.get_data(use_db=True)

    assert out.shape == frame.data.shape
    assert np.isneginf(out).all()


def test_add_noise_chi2_adds_returned_noise_and_updates_data_and_stats(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 2.5)

    def fake_chi2(x_mean, df, shape):
        assert x_mean == 5
        assert df == frame.chi2_df
        assert shape == frame.shape
        return noise

    monkeypatch.setattr(frame_module.distributions, "chi2", fake_chi2)

    returned = frame.add_noise(x_mean=5, noise_type="chi2")

    assert np.array_equal(returned, noise)
    assert np.array_equal(frame.data, noise)
    assert frame.noise_mean == 5
    assert frame.noise_std == np.sqrt(2 * frame.chi2_df) * 5 / frame.chi2_df


def test_add_noise_chi2_on_existing_noise_recomputes_stats(monkeypatch):
    frame = make_frame()
    frame.noise_mean = 1.0
    frame.noise_std = 2.0
    noise = np.ones(frame.shape)

    def fake_chi2(x_mean, df, shape):
        return noise

    monkeypatch.setattr(frame_module.distributions, "chi2", fake_chi2)

    called = {"count": 0}

    def fake_update():
        called["count"] += 1
        frame.noise_mean = 123
        frame.noise_std = 456

    monkeypatch.setattr(frame, "_update_noise_frame_stats", fake_update)

    frame.add_noise(x_mean=3, noise_type="chi2")

    assert called["count"] == 1
    assert frame.noise_mean == 123
    assert frame.noise_std == 456


def test_add_noise_normal_uses_gaussian_when_no_min(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 1.25)

    def fake_gaussian(x_mean, x_std, shape):
        assert x_mean == 4
        assert x_std == 2
        assert shape == frame.shape
        return noise

    monkeypatch.setattr(frame_module.distributions, "gaussian", fake_gaussian)

    returned = frame.add_noise(x_mean=4, x_std=2, noise_type="normal")

    assert np.array_equal(returned, noise)
    assert np.array_equal(frame.data, noise)
    assert frame.noise_mean == 4
    assert frame.noise_std == 2


def test_add_noise_gaussian_alias_uses_gaussian_distribution(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 0.5)

    monkeypatch.setattr(frame_module.distributions, "gaussian",
                        lambda x_mean, x_std, shape: noise)

    returned = frame.add_noise(x_mean=1, x_std=0.1, noise_type="gaussian")

    assert np.array_equal(returned, noise)
    assert np.array_equal(frame.data, noise)


def test_add_noise_normal_with_min_uses_truncated_gaussian(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 3.0)

    def fake_truncated_gaussian(x_mean, x_std, x_min, shape):
        assert x_mean == 6
        assert x_std == 1.5
        assert x_min == 0.2
        assert shape == frame.shape
        return noise

    monkeypatch.setattr(frame_module.distributions, "truncated_gaussian",
                        fake_truncated_gaussian)

    returned = frame.add_noise(x_mean=6, x_std=1.5, x_min=0.2, noise_type="normal")

    assert np.array_equal(returned, noise)
    assert np.array_equal(frame.data, noise)


def test_add_noise_invalid_type_exits():
    frame = make_frame()

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, x_std=1, noise_type="bogus")


def test_add_noise_normal_requires_x_std():
    frame = make_frame()

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, noise_type="normal")


def test_add_noise_updates_existing_noise_stats_via_sigma_clip(monkeypatch):
    frame = make_frame()
    frame.noise_mean = 1.0
    frame.noise_std = 1.0

    noise = np.arange(np.prod(frame.shape), dtype=float).reshape(frame.shape)

    monkeypatch.setattr(frame_module.distributions, "gaussian",
                        lambda x_mean, x_std, shape: noise)

    # Make the stats calculation deterministic and independent of astropy sigma_clip
    def fake_update():
        frame.noise_mean = float(np.mean(frame.data))
        frame.noise_std = float(np.std(frame.data))

    monkeypatch.setattr(frame, "_update_noise_frame_stats", fake_update)

    frame.add_noise(x_mean=0, x_std=1, noise_type="normal")

    assert np.array_equal(frame.data, noise)
    assert frame.noise_mean == np.mean(noise)
    assert frame.noise_std == np.std(noise)


def test_add_noise_uses_current_frame_shape(monkeypatch):
    frame = Frame(fchans=2, tchans=5, df=1, dt=1, fch1=10)
    captured = {}

    def fake_gaussian(x_mean, x_std, shape):
        captured["shape"] = shape
        return np.zeros(shape)

    monkeypatch.setattr(frame_module.distributions, "gaussian", fake_gaussian)

    frame.add_noise(x_mean=1, x_std=1, noise_type="normal")

    assert captured["shape"] == (5, 2)


def test_get_data_db_reflects_current_data_after_noise_addition(monkeypatch):
    frame = make_frame()

    monkeypatch.setattr(frame_module.distributions, "gaussian",
                        lambda x_mean, x_std, shape: np.ones(shape))

    frame.add_noise(x_mean=2, x_std=1, noise_type="normal")
    db = frame.get_data(use_db=True)

    assert np.allclose(db, 0.0)
    assert np.allclose(frame.get_data(), np.ones(frame.shape))


def test_add_noise_chi2_returns_noise_array_not_frame_data(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 9.0)

    monkeypatch.setattr(frame_module.distributions, "chi2",
                        lambda x_mean, df, shape: noise)

    returned = frame.add_noise(x_mean=3, noise_type="chi2")

    assert returned is noise
    assert returned is not frame.data
    assert np.array_equal(frame.data, noise)
