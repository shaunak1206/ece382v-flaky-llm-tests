import numpy as np
import pytest

from setigen.frame import Frame
import setigen.frame as frame_mod


def make_frame():
    return Frame(fchans=4, tchans=3, df=1.0, dt=2.0, fch1=10.0)


def test_get_data_returns_underlying_array():
    frame = make_frame()
    frame.data[:] = np.arange(frame.data.size).reshape(frame.shape)

    data = frame.get_data()

    assert data is frame.data
    np.testing.assert_array_equal(data, frame.data)


def test_get_data_db_returns_10log10_of_data():
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


def test_get_data_db_does_not_modify_frame_data():
    frame = make_frame()
    original = np.array(
        [
            [1.0, 2.0, 4.0, 8.0],
            [16.0, 32.0, 64.0, 128.0],
            [256.0, 512.0, 1024.0, 2048.0],
        ]
    )
    frame.data[:] = original

    _ = frame.get_data(use_db=True)

    np.testing.assert_array_equal(frame.data, original)


def test_get_data_db_with_zero_produces_negative_infinity():
    frame = make_frame()
    frame.data[:] = 0.0

    db = frame.get_data(use_db=True)

    assert np.isneginf(db).all()


def test_add_noise_chi2_calls_distribution_and_updates_data_and_stats(monkeypatch):
    frame = make_frame()

    called = {}

    def fake_chi2(x_mean, chi2_df, shape):
        called["args"] = (x_mean, chi2_df, shape)
        return np.full(shape, 7.0)

    monkeypatch.setattr(frame_mod.distributions, "chi2", fake_chi2)

    noise = frame.add_noise(x_mean=5.0, noise_type="chi2")

    assert called["args"] == (5.0, frame.chi2_df, frame.shape)
    np.testing.assert_array_equal(noise, np.full(frame.shape, 7.0))
    np.testing.assert_array_equal(frame.data, np.full(frame.shape, 7.0))
    assert frame.noise_mean == 5.0
    assert frame.noise_std == np.sqrt(2 * frame.chi2_df) * 5.0 / frame.chi2_df


def test_add_noise_chi2_uses_existing_noise_stats_when_nonzero(monkeypatch):
    frame = make_frame()
    frame.noise_mean = 1.0
    frame.noise_std = 2.0

    monkeypatch.setattr(frame_mod.distributions, "chi2",
                        lambda x_mean, chi2_df, shape: np.ones(shape))

    frame.add_noise(x_mean=3.0, noise_type="chi2")

    assert frame.noise_mean == np.mean(frame.data)
    assert frame.noise_std == np.std(frame.data)


def test_add_noise_normal_uses_gaussian_when_no_min(monkeypatch):
    frame = make_frame()

    called = {}

    def fake_gaussian(x_mean, x_std, shape):
        called["args"] = (x_mean, x_std, shape)
        return np.full(shape, 2.5)

    monkeypatch.setattr(frame_mod.distributions, "gaussian", fake_gaussian)

    noise = frame.add_noise(x_mean=1.25, x_std=0.5, noise_type="normal")

    assert called["args"] == (1.25, 0.5, frame.shape)
    np.testing.assert_array_equal(noise, np.full(frame.shape, 2.5))
    np.testing.assert_array_equal(frame.data, np.full(frame.shape, 2.5))
    assert frame.noise_mean == 1.25
    assert frame.noise_std == 0.5


def test_add_noise_gaussian_alias_uses_gaussian(monkeypatch):
    frame = make_frame()

    monkeypatch.setattr(frame_mod.distributions, "gaussian",
                        lambda x_mean, x_std, shape: np.full(shape, 4.0))

    noise = frame.add_noise(x_mean=2.0, x_std=1.0, noise_type="gaussian")

    np.testing.assert_array_equal(noise, np.full(frame.shape, 4.0))
    np.testing.assert_array_equal(frame.data, np.full(frame.shape, 4.0))


def test_add_noise_truncated_gaussian_uses_min_and_distribution(monkeypatch):
    frame = make_frame()

    called = {}

    def fake_truncated_gaussian(x_mean, x_std, x_min, shape):
        called["args"] = (x_mean, x_std, x_min, shape)
        return np.full(shape, 6.0)

    monkeypatch.setattr(frame_mod.distributions, "truncated_gaussian",
                        fake_truncated_gaussian)

    noise = frame.add_noise(x_mean=3.0, x_std=1.5, x_min=0.25, noise_type="normal")

    assert called["args"] == (3.0, 1.5, 0.25, frame.shape)
    np.testing.assert_array_equal(noise, np.full(frame.shape, 6.0))
    np.testing.assert_array_equal(frame.data, np.full(frame.shape, 6.0))


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()

    with pytest.raises(SystemExit) as excinfo:
        frame.add_noise(x_mean=1.0, x_std=1.0, noise_type="bogus")

    assert "not a valid noise type" in str(excinfo.value)


def test_add_noise_normal_without_x_std_exits():
    frame = make_frame()

    with pytest.raises(SystemExit) as excinfo:
        frame.add_noise(x_mean=1.0, noise_type="normal")

    assert "x_std must be given" in str(excinfo.value)


def test_add_noise_unknown_noise_type_does_not_modify_existing_data():
    frame = make_frame()
    original = frame.data.copy()

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1.0, x_std=1.0, noise_type="invalid")

    np.testing.assert_array_equal(frame.data, original)


def test_add_noise_adds_noise_to_existing_nonzero_data(monkeypatch):
    frame = make_frame()
    frame.data[:] = 3.0

    monkeypatch.setattr(frame_mod.distributions, "gaussian",
                        lambda x_mean, x_std, shape: np.full(shape, 1.5))

    frame.add_noise(x_mean=10.0, x_std=2.0, noise_type="normal")

    np.testing.assert_array_equal(frame.data, np.full(frame.shape, 4.5))
    assert frame.noise_mean == np.mean(frame.data)
    assert frame.noise_std == np.std(frame.data)
