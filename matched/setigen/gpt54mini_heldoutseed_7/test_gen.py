import numpy as np
import pytest

from setigen.frame import Frame


def make_frame():
    return Frame(
        fchans=4,
        tchans=3,
        df=1,
        dt=1,
        fch1=10,
    )


def test_get_data_returns_raw_array_by_default():
    frame = make_frame()
    frame.data[:] = np.arange(frame.data.size).reshape(frame.shape)

    out = frame.get_data()

    assert out is frame.data
    np.testing.assert_array_equal(out, frame.data)


def test_get_data_use_db_applies_log10():
    frame = make_frame()
    frame.data[:] = np.array(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [2.0, 20.0, 200.0, 2000.0],
            [3.0, 30.0, 300.0, 3000.0],
        ]
    )

    out = frame.get_data(use_db=True)

    np.testing.assert_allclose(out, 10 * np.log10(frame.data))


def test_get_data_use_db_preserves_shape():
    frame = make_frame()
    frame.data[:] = 7.0

    out = frame.get_data(use_db=True)

    assert out.shape == frame.shape


def test_add_noise_chi2_updates_data_and_returns_noise(monkeypatch):
    frame = make_frame()

    noise = np.full(frame.shape, 2.5)

    def fake_chi2(x_mean, df, shape):
        assert x_mean == 10
        assert df == frame.chi2_df
        assert shape == frame.shape
        return noise

    monkeypatch.setattr("setigen.frame.distributions.chi2", fake_chi2)

    returned = frame.add_noise(x_mean=10, noise_type="chi2")

    np.testing.assert_array_equal(returned, noise)
    np.testing.assert_array_equal(frame.data, noise)
    assert frame.noise_mean == 10
    assert frame.noise_std == np.sqrt(2 * frame.chi2_df) * 10 / frame.chi2_df


def test_add_noise_chi2_with_nonzero_existing_noise_stats_recomputes(monkeypatch):
    frame = make_frame()
    frame.noise_mean = 1.0
    frame.noise_std = 2.0

    noise = np.ones(frame.shape)

    monkeypatch.setattr("setigen.frame.distributions.chi2", lambda x_mean, df, shape: noise)

    called = {"count": 0}

    def fake_update():
        called["count"] += 1
        frame.noise_mean = 123
        frame.noise_std = 456

    monkeypatch.setattr(frame, "_update_noise_frame_stats", fake_update)

    frame.add_noise(x_mean=10, noise_type="chi2")

    assert called["count"] == 1
    assert frame.noise_mean == 123
    assert frame.noise_std == 456


def test_add_noise_normal_uses_gaussian_when_x_min_missing(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 3.0)

    def fake_gaussian(x_mean, x_std, shape):
        assert x_mean == 4
        assert x_std == 2
        assert shape == frame.shape
        return noise

    monkeypatch.setattr("setigen.frame.distributions.gaussian", fake_gaussian)

    returned = frame.add_noise(x_mean=4, x_std=2, noise_type="normal")

    np.testing.assert_array_equal(returned, noise)
    np.testing.assert_array_equal(frame.data, noise)
    assert frame.noise_mean == 4
    assert frame.noise_std == 2


def test_add_noise_normal_uses_truncated_gaussian_when_x_min_provided(monkeypatch):
    frame = make_frame()
    noise = np.full(frame.shape, 5.0)

    def fake_trunc(x_mean, x_std, x_min, shape):
        assert (x_mean, x_std, x_min) == (4, 2, 1)
        assert shape == frame.shape
        return noise

    monkeypatch.setattr("setigen.frame.distributions.truncated_gaussian", fake_trunc)

    returned = frame.add_noise(x_mean=4, x_std=2, x_min=1, noise_type="gaussian")

    np.testing.assert_array_equal(returned, noise)
    np.testing.assert_array_equal(frame.data, noise)


def test_add_noise_normal_requires_x_std():
    frame = make_frame()

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, noise_type="normal")


def test_add_noise_invalid_noise_type_exits():
    frame = make_frame()

    with pytest.raises(SystemExit):
        frame.add_noise(x_mean=1, x_std=1, noise_type="not-a-valid-type")


def test_add_noise_chi2_with_zero_baseline_sets_noise_stats():
    frame = make_frame()
    frame.zero_data()

    noise = np.ones(frame.shape) * 7.0
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("setigen.frame.distributions.chi2", lambda x_mean, df, shape: noise)

    try:
        frame.add_noise(x_mean=7, noise_type="chi2")
    finally:
        monkeypatch.undo()

    assert frame.noise_mean == 7
    assert frame.noise_std == np.sqrt(2 * frame.chi2_df) * 7 / frame.chi2_df


def test_add_noise_normal_with_zero_baseline_sets_noise_stats(monkeypatch):
    frame = make_frame()
    frame.zero_data()

    noise = np.ones(frame.shape) * 9.0
    monkeypatch.setattr("setigen.frame.distributions.gaussian", lambda x_mean, x_std, shape: noise)

    frame.add_noise(x_mean=9, x_std=3, noise_type="gaussian")

    assert frame.noise_mean == 9
    assert frame.noise_std == 3


def test_add_noise_adds_noise_to_existing_data(monkeypatch):
    frame = make_frame()
    frame.data[:] = 1.5

    noise = np.full(frame.shape, 0.5)
    monkeypatch.setattr("setigen.frame.distributions.gaussian", lambda x_mean, x_std, shape: noise)

    frame.add_noise(x_mean=0.5, x_std=0.1, noise_type="normal")

    np.testing.assert_array_equal(frame.data, np.full(frame.shape, 2.0))


def test_get_data_use_db_with_zeros_produces_infinities():
    frame = make_frame()
    frame.zero_data()

    out = frame.get_data(use_db=True)

    assert out.shape == frame.shape
    assert np.isneginf(out).all()


def test_add_noise_normal_with_x_min_and_share_index_false_uses_sample_from_obs(monkeypatch):
    frame = make_frame()

    sample_called = {"count": 0}

    def fake_sample_gaussian_params(x_mean_array, x_std_array, x_min_array=None):
        sample_called["count"] += 1
        assert np.array_equal(x_mean_array, np.array([1, 2, 3]))
        assert np.array_equal(x_std_array, np.array([4, 5, 6]))
        assert np.array_equal(x_min_array, np.array([7, 8, 9]))
        return 2, 5, 8

    def fake_trunc(x_mean, x_std, x_min, shape):
        assert (x_mean, x_std, x_min) == (2, 5, 8)
        assert shape == frame.shape
        return np.full(frame.shape, 11.0)

    monkeypatch.setattr("setigen.frame.sample_from_obs.sample_gaussian_params", fake_sample_gaussian_params)
    monkeypatch.setattr("setigen.frame.distributions.truncated_gaussian", fake_trunc)

    out = frame.add_noise(
        x_mean_array=np.array([1, 2, 3]),
        x_std_array=np.array([4, 5, 6]),
        x_min_array=np.array([7, 8, 9]),
        share_index=False,
        noise_type="normal",
    )

    assert sample_called["count"] == 1
    np.testing.assert_array_equal(out, np.full(frame.shape, 11.0))


def test_add_noise_normal_share_index_length_mismatch_raises_index_error():
    frame = make_frame()

    with pytest.raises(IndexError):
        frame.add_noise(
            x_mean_array=np.array([1, 2]),
            x_std_array=np.array([3]),
            x_min_array=np.array([4, 5]),
            share_index=True,
            noise_type="normal",
        )


def test_add_noise_normal_without_x_min_share_index_length_mismatch_raises_index_error():
    frame = make_frame()

    with pytest.raises(IndexError):
        frame.add_noise(
            x_mean_array=np.array([1, 2]),
            x_std_array=np.array([3]),
            share_index=True,
            noise_type="gaussian",
        )
