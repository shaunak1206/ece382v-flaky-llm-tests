import numpy as np
import pytest
from pytsmp import utils


class TestSlidingDotProduct:
    def test_basic_correctness(self):
        # Simple known case: Q dot subsequences in T
        T = np.array([1, 2, 3, 4, 5])
        Q = np.array([1, 0])
        result = utils.sliding_dot_product(Q, T)
        expected = np.array([1*1+0*2, 1*2+0*3, 1*3+0*4, 1*4+0*5])
        np.testing.assert_allclose(result, expected)

    def test_long_query_raises(self):
        T = np.array([1, 2, 3])
        Q = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError):
            utils.sliding_dot_product(Q, T)

    def test_output_length(self):
        T = np.arange(10)
        Q = np.arange(4)
        result = utils.sliding_dot_product(Q, T)
        assert result.shape[0] == len(T) - len(Q) + 1

    def test_dot_product_values_match_direct(self):
        # Verify the sliding dot product gives the same result as manual dot products
        T = np.array([2, -1, 0, 3, 1, 4])
        Q = np.array([1, 0, -1])
        result = utils.sliding_dot_product(Q, T)
        expected = np.array([
            np.dot(Q, T[0:3]),
            np.dot(Q, T[1:4]),
            np.dot(Q, T[2:5]),
            np.dot(Q, T[3:6]),
        ])
        np.testing.assert_allclose(result, expected)


class TestRollingAvgSD:
    def test_basic_correctness(self):
        t = np.array([1, 2, 3, 4, 5])
        window = 3
        means, sds = utils.rolling_avg_sd(t, window)
        expected_means = np.array([2.0, 3.0, 4.0])
        expected_sds = np.array([np.std(t[0:3], ddof=0), np.std(t[1:4], ddof=0), np.std(t[2:5], ddof=0)])
        np.testing.assert_allclose(means, expected_means)
        np.testing.assert_allclose(sds, expected_sds)

    def test_window_larger_than_series_raises(self):
        t = np.array([1, 2])
        with pytest.raises(ValueError):
            utils.rolling_avg_sd(t, 3)

    def test_constant_series_sd_zero(self):
        t = np.array([4, 4, 4, 4])
        means, sds = utils.rolling_avg_sd(t, 2)
        expected_means = np.array([4, 4, 4])
        expected_sds = np.array([0, 0, 0])
        np.testing.assert_allclose(means, expected_means)
        np.testing.assert_allclose(sds, expected_sds)

    def test_single_element_windows(self):
        t = np.array([10, 20, 30])
        means, sds = utils.rolling_avg_sd(t, 1)
        np.testing.assert_allclose(means, t)
        np.testing.assert_allclose(sds, np.zeros_like(t))


class TestCalculateDistanceProfile:
    def test_basic_distance_profile(self):
        # Create simple inputs that emulate Q and T and their statistics
        m = 3
        QT = np.array([10, 20, 30])
        mean_Q = 2.0
        sigma_Q = 1.0
        mean_T = np.array([1, 2, 3])
        sigma_T = np.array([1, 1, 1])

        D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
        expected = np.sqrt(2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)))
        np.testing.assert_allclose(D, expected)

    def test_constant_query_std_less_than_epsilon(self):
        m = 4
        n = 5
        QT = np.ones(n)
        mean_Q = 1.5
        sigma_Q = 0.0  # constant query
        mean_T = np.ones(n)
        sigma_T = np.ones(n) * 0.5

        D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
        # Distance should be sqrt(m) because sigma_Q < epsilon
        expected = np.full(n, np.sqrt(m))
        np.testing.assert_allclose(D, expected)

    def test_constant_subsequence_in_T(self):
        m = 3
        n = 4
        QT = np.arange(n)
        mean_Q = 1
        sigma_Q = 1
        mean_T = np.array([2, 2, 2, 2])
        sigma_T = np.zeros(n)  # constant subsequences in T

        D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
        # Distance should be sqrt(m) when sigma_T < epsilon but sigma_Q > epsilon
        expected = np.full(n, np.sqrt(m))
        np.testing.assert_allclose(D, expected)

    def test_both_constant_sequences_returns_zero(self):
        m = 3
        n = 3
        QT = np.array([0., 0., 0.])
        mean_Q = 5
        sigma_Q = 0
        mean_T = np.array([5, 5, 5])
        sigma_T = np.array([0, 0, 0])

        D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
        expected = np.zeros(n)
        np.testing.assert_allclose(D, expected)

    def test_dimension_mismatch_raises(self):
        QT = np.array([1, 2])
        mean_Q = 1
        sigma_Q = 1
        mean_T = np.array([1])
        sigma_T = np.array([1, 2])
        with pytest.raises(ValueError):
            utils.calculate_distance_profile(QT, 2, mean_Q, sigma_Q, mean_T, sigma_T)

    def test_vectorized_mean_Q_and_sigma_Q_valid(self):
        # mean_Q, sigma_Q as arrays matching mean_T length
        m = 3
        QT = np.array([5, 6, 7])
        mean_Q = np.array([2, 2, 2])
        sigma_Q = np.array([1, 1, 1])
        mean_T = np.array([1, 2, 3])
        sigma_T = np.array([1, 1, 1])

        D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
        expected = np.sqrt(2 * m * (1 - (QT - m * mean_Q * mean_T) / (m * sigma_Q * sigma_T)))
        np.testing.assert_allclose(D, expected)


class TestIntegrationMass:
    def test_mass_distance_profile_matches_manual(self):
        # Q: simple subsequence of T to have expected zero distance at first position
        T = np.array([1, 2, 3, 4, 5, 6])
        Q = T[1:4]  # [2,3,4]

        D = utils.mass(Q, T)
        # Distance profile length
        expected_length = len(T) - len(Q) + 1
        assert D.shape[0] == expected_length

        # The distance at subsequence Q itself should be near zero
        assert D[1] < 1e-10

    def test_mass_query_longer_than_T_raises(self):
        T = np.array([1, 2])
        Q = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            utils.mass(Q, T)
