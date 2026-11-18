import numpy as np
import pytest
from pytsmp import utils


class TestSlidingDotProduct:

    def test_basic_dot_product(self):
        Q = np.array([1, 2, 3])
        T = np.array([1, 2, 3, 4, 5])
        # Manually calculate dot products:
        # [1*1 + 2*2 + 3*3, 2*1 + 3*2 + 4*3, 3*1 + 4*2 + 5*3] = [14, 20, 26]
        expected = np.array([14, 20, 26])
        result = utils.sliding_dot_product(Q, T)
        np.testing.assert_allclose(result, expected)

    def test_query_longer_than_time_series_raises(self):
        Q = np.array([1, 2, 3, 4])
        T = np.array([1, 2])
        with pytest.raises(ValueError):
            utils.sliding_dot_product(Q, T)

    def test_equal_length_query_and_time_series(self):
        Q = np.array([1, 2, 3])
        T = np.array([4, 5, 6])
        # Dot product is a single value
        expected = np.array([1*4 + 2*5 + 3*6])
        result = utils.sliding_dot_product(Q, T)
        np.testing.assert_allclose(result, expected)

    def test_dot_product_with_zeros(self):
        Q = np.array([0, 0, 0])
        T = np.array([1, 2, 3, 4, 5])
        expected = np.zeros(len(T) - len(Q) + 1)
        result = utils.sliding_dot_product(Q, T)
        np.testing.assert_array_equal(result, expected)

    def test_dot_product_negative_values(self):
        Q = np.array([1, -1, 2])
        T = np.array([-2, 0, 1, 3, 4])
        # subsequences: [-2,0,1], [0,1,3], [1,3,4]
        # dot products:
        # (1*-2) + (-1*0) + (2*1) = -2 + 0 + 2 = 0
        # (1*0) + (-1*1) + (2*3) = 0 -1 + 6 = 5
        # (1*1) + (-1*3) + (2*4) = 1 -3 + 8 = 6
        expected = np.array([0, 5, 6])
        result = utils.sliding_dot_product(Q, T)
        np.testing.assert_allclose(result, expected)


class TestRollingAvgSd:

    def test_rolling_avg_sd_basic(self):
        t = np.array([1, 2, 3, 4, 5])
        window = 3
        mean_expected = np.array([2., 3., 4.])
        sd_expected = np.array([np.std([1,2,3], ddof=0),
                                np.std([2,3,4], ddof=0),
                                np.std([3,4,5], ddof=0)])
        means, sds = utils.rolling_avg_sd(t, window)
        np.testing.assert_allclose(means, mean_expected)
        np.testing.assert_allclose(sds, sd_expected)

    def test_rolling_avg_sd_window_equals_length(self):
        t = np.array([1, 4, 7])
        window = 3
        mean_expected = np.array([4])
        sd_expected = np.array([np.std(t, ddof=0)])
        means, sds = utils.rolling_avg_sd(t, window)
        np.testing.assert_allclose(means, mean_expected)
        np.testing.assert_allclose(sds, sd_expected)

    def test_rolling_avg_sd_window_greater_than_length_raises(self):
        t = np.array([1, 2])
        with pytest.raises(ValueError):
            utils.rolling_avg_sd(t, 3)

    def test_rolling_avg_sd_constant_series(self):
        t = np.array([5, 5, 5, 5])
        window = 2
        means, sds = utils.rolling_avg_sd(t, window)
        # means all 5, sds all 0
        np.testing.assert_array_equal(means, np.array([5, 5, 5]))
        np.testing.assert_array_equal(sds, np.array([0, 0, 0]))

    def test_rolling_avg_sd_single_element_window(self):
        t = np.array([2, 4, 6, 8])
        window = 1
        means, sds = utils.rolling_avg_sd(t, window)
        np.testing.assert_array_equal(means, t)
        np.testing.assert_array_equal(sds, np.zeros_like(t))


class TestCalculateDistanceProfile:

    def test_distance_profile_basic(self):
        m = 3
        QT = np.array([6, 15, 24])
        mean_Q = 2
        sigma_Q = 1
        mean_T = np.array([1, 2, 3])
        sigma_T = np.array([1, 1, 1])
        # Using formula: sqrt(2 * m * (1 - (QT - m*mean_Q*mean_T) / (m*sigma_Q*sigma_T)))
        # intermediate = (QT - m*mean_Q*mean_T) / (m*sigma_Q*sigma_T)
        intermediate = (QT - m*mean_Q*mean_T) / (m*sigma_Q*sigma_T)
        expected = np.sqrt(np.maximum(2 * m * (1 - intermediate), 0))
        result = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
        np.testing.assert_allclose(result, expected)

    def test_distance_profile_sigmaQ_small(self):
        # sigma_Q close to zero triggers first np.where condition
        m = 4
        QT = np.array([1, 2, 3])
        mean_Q = 5
        sigma_Q = 1e-9  # below epsilon=1e-7
        mean_T = np.array([1, 2, 3])
        sigma_T = np.array([1, 2, 3])
        expected = np.full_like(QT, np.sqrt(m))
        result = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
        np.testing.assert_allclose(result, expected)

    def test_distance_profile_sigmaT_small(self):
        m = 2
        QT = np.array([10, 20])
        mean_Q = 1
        sigma_Q = 1
        mean_T = np.array([5, 5])
        sigma_T = np.array([1e-10, 1e-10])  # smaller than epsilon
        expected = np.full_like(QT, np.sqrt(m))
        result = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
        np.testing.assert_allclose(result, expected)

    def test_distance_profile_sigmaQ_and_sigmaT_small(self):
        m = 3
        QT = np.array([2, 4])
        mean_Q = 1
        sigma_Q = 1e-10
        mean_T = np.array([1, 1])
        sigma_T = np.array([1e-10, 1e-10])
        expected = np.zeros_like(QT)
        result = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
        np.testing.assert_allclose(result, expected)

    def test_distance_profile_dimension_mismatch_raises(self):
        m = 3
        QT = np.array([1, 2])
        mean_Q = np.array([1, 1])
        sigma_Q = np.array([1])
        mean_T = np.array([2, 2])
        sigma_T = np.array([1, 1])
        with pytest.raises(ValueError):
            utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)

    def test_distance_profile_all_ones(self):
        m = 2
        QT = np.array([2, 2, 2])
        mean_Q = 1
        sigma_Q = 1
        mean_T = np.array([1, 1, 1])
        sigma_T = np.array([1, 1, 1])
        expected = np.zeros_like(QT)
        result = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("Q,T", [
    (np.array([1,2,3]), np.array([1,2,3,4,5])),
    (np.array([0,0]), np.array([0,0,0])),
    (np.array([5]), np.array([5,6,7,8]))
])
def test_mass_returns_correct_size(Q, T):
    # The MASS function should return array of length len(T) - len(Q) + 1
    result = utils.mass(Q, T)
    expected_length = len(T) - len(Q) + 1
    assert result.shape[0] == expected_length


def test_mass_query_longer_than_time_series_raises():
    Q = np.array([1, 2, 3])
    T = np.array([1])
    with pytest.raises(ValueError):
        utils.mass(Q, T)


def test_mass_distance_profile_non_negative():
    Q = np.array([1, 2, 3])
    T = np.array([1, 2, 3, 4, 5])
    D = utils.mass(Q, T)
    assert np.all(D >= 0), "Distance profile contains negative values which is invalid"


def test_mass_distance_profile_constant_query():
    Q = np.array([7, 7, 7])
    T = np.array([1, 2, 3, 4, 5])
    # sigma_Q = 0 triggers distance profile special condition, so distances should be sqrt(m) or 0
    D = utils.mass(Q, T)
    # All distances should be either 0 or sqrt(len(Q))
    m = len(Q)
    for val in D:
        assert val == 0 or np.isclose(val, np.sqrt(m))


def test_mass_identical_series_zero_distance():
    Q = np.array([1, 2, 3])
    T = np.array([1, 2, 3, 4, 5])
    # Distance between Q and Q itself (subsequence at beginning) should be close to zero
    D = utils.mass(Q, T)
    # First element in D corresponds to distance of Q vs T[0:3]
    assert pytest.approx(D[0], abs=1e-7) == 0


def test_sliding_dot_product_large_input():
    # Test behavior on larger input arrays (performance not tested here)
    np.random.seed(0)
    Q = np.random.rand(100)
    T = np.random.rand(200)
    result = utils.sliding_dot_product(Q, T)
    assert len(result) == len(T) - len(Q) + 1
    # No exception and result is finite numbers
    assert np.all(np.isfinite(result))


def test_rolling_avg_sd_large_input():
    np.random.seed(1)
    t = np.random.rand(1000)
    window = 50
    means, sds = utils.rolling_avg_sd(t, window)
    # Length check
    assert len(means) == len(t) - window + 1
    assert len(sds) == len(t) - window + 1
    # Values are finite and standard deviation non-negative
    assert np.all(np.isfinite(means))
    assert np.all(sds >= 0)
