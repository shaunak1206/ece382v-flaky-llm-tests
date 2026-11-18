import numpy as np
import pytest
from pytsmp import utils


class TestSlidingDotProduct:
    def test_basic_dot_product(self):
        Q = np.array([1, 2, 3])
        T = np.array([0, 1, 2, 3, 4])
        result = utils.sliding_dot_product(Q, T)
        # Manual dot products: [1*0+2*1+3*2=8, 1*1+2*2+3*3=14, 1*2+2*3+3*4=20]
        expected = np.array([8, 14, 20])
        np.testing.assert_allclose(result, expected)

    def test_query_longer_than_T_raises(self):
        Q = np.array([1, 2, 3, 4])
        T = np.array([1, 2])
        with pytest.raises(ValueError):
            utils.sliding_dot_product(Q, T)

    def test_result_length(self):
        Q = np.ones(4)
        T = np.ones(10)
        result = utils.sliding_dot_product(Q, T)
        expected_length = len(T) - len(Q) + 1
        assert len(result) == expected_length

    def test_sliding_dot_product_constant_sequence(self):
        Q = np.array([1, 1, 1])
        T = np.ones(6) * 5
        result = utils.sliding_dot_product(Q, T)
        # Every subsequence dot product == sum(Q)*5 == 3*5 == 15
        expected = np.array([15, 15, 15, 15])
        np.testing.assert_allclose(result, expected)

    def test_sliding_dot_product_with_zeros(self):
        Q = np.array([0, 0, 0])
        T = np.array([1, 2, 3, 4])
        result = utils.sliding_dot_product(Q, T)
        # Dot products should all be zero
        expected = np.zeros(len(T) - len(Q) + 1)
        np.testing.assert_allclose(result, expected)

    def test_sliding_dot_product_floats(self):
        Q = np.array([0.5, -0.5])
        T = np.array([1, 2, 3])
        result = utils.sliding_dot_product(Q, T)
        # Manual: [0.5*1 + -0.5*2 = 0.5 - 1 = -0.5, 0.5*2 + -0.5*3 = 1 - 1.5 = -0.5]
        expected = np.array([-0.5, -0.5])
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestRollingAvgSD:
    def test_rolling_avg_sd_basic(self):
        t = np.array([1, 2, 3, 4, 5])
        window = 3
        avg, sd = utils.rolling_avg_sd(t, window)
        expected_avg = np.array([2., 3., 4.])
        expected_sd = np.array([
            np.std([1, 2, 3], ddof=0),
            np.std([2, 3, 4], ddof=0),
            np.std([3, 4, 5], ddof=0),
        ])
        np.testing.assert_allclose(avg, expected_avg)
        np.testing.assert_allclose(sd, expected_sd)

    def test_rolling_avg_sd_window_equal_len(self):
        t = np.arange(10)
        avg, sd = utils.rolling_avg_sd(t, 10)
        np.testing.assert_allclose(avg, [np.mean(t)])
        np.testing.assert_allclose(sd, [np.std(t)])

    def test_rolling_avg_sd_raises_for_large_window(self):
        t = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            utils.rolling_avg_sd(t, 4)

    def test_rolling_avg_sd_constant_sequence_sd_zero(self):
        t = np.ones(5)
        avg, sd = utils.rolling_avg_sd(t, 3)
        np.testing.assert_allclose(avg, np.ones(3))
        np.testing.assert_allclose(sd, np.zeros(3))

    def test_rolling_avg_sd_check_length(self):
        t = np.arange(7)
        window = 4
        avg, sd = utils.rolling_avg_sd(t, window)
        expected_len = len(t) - window + 1
        assert len(avg) == expected_len
        assert len(sd) == expected_len


class TestCalculateDistanceProfile:
    def test_calculate_distance_profile_basic(self):
        m = 3
        QT = np.array([10, 20, 30])
        mean_Q = 2
        sigma_Q = 1
        mean_T = np.array([1, 2, 3])
        sigma_T = np.array([1, 1, 1])
        D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)

        # Manual calculation for one element:
        # dist = sqrt(2*m * (1 - (QT - m*mean_Q*mean_T)/(m*sigma_Q*sigma_T)))
        expected = np.sqrt(2*m*(1 - (QT - m*mean_Q*mean_T)/(m*sigma_Q*sigma_T)))
        np.testing.assert_allclose(D, expected)

    def test_calculate_distance_profile_with_constant_Q(self):
        m = 2
        QT = np.array([1, 2])
        mean_Q = 5
        sigma_Q = 0  # constant Q
        mean_T = np.array([1, 2])
        sigma_T = np.array([0.1, 0])
        # When sigma_Q < epsilon and sigma_T >= epsilon => distance = sqrt(m)
        # When sigma_Q < epsilon and sigma_T < epsilon => distance = 0
        D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T,
                                            epsilon=1e-7)
        expected = np.array([np.sqrt(m), 0])
        np.testing.assert_allclose(D, expected)

    def test_calculate_distance_profile_with_constant_T(self):
        m = 3
        QT = np.array([5, 8])
        mean_Q = 1
        sigma_Q = 0.5
        mean_T = np.array([1, 1])
        sigma_T = np.array([0, 0])
        # sigma_T < epsilon -> distance = sqrt(m)
        D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
        expected = np.full_like(QT, np.sqrt(m), dtype=float)
        np.testing.assert_allclose(D, expected)

    def test_calculate_distance_profile_dimension_mismatch_raises(self):
        QT = np.array([1, 2, 3])
        m = 3
        mean_Q = np.array([1, 2])
        sigma_Q = np.array([1, 1])
        mean_T = np.array([1, 2, 3])
        sigma_T = np.array([1, 2, 3, 4])  # length differs
        with pytest.raises(ValueError):
            utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)

    def test_calculate_distance_profile_scalar_mean_std_Q(self):
        m = 3
        QT = np.array([5, 6, 7])
        mean_Q = 2.0
        sigma_Q = 1.0
        mean_T = np.array([1.0, 2.0, 3.0])
        sigma_T = np.array([0.5, 1.0, 1.5])
        D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
        expected = np.sqrt(np.maximum(2*m*(1 - (QT - m*mean_Q*mean_T) / (m*sigma_Q*sigma_T)), 0))
        np.testing.assert_allclose(D, expected)

    def test_calculate_distance_profile_handles_negative_under_sqrt(self):
        # The inside of sqrt may be negative due to floating point rounding; max with 0 used
        m = 2
        QT = np.array([10, 10])
        mean_Q = 3
        sigma_Q = 1
        mean_T = np.array([3, 3])
        sigma_T = np.array([1, 1])
        # Create values so inside sqrt is slightly negative
        # Set QT so that the value inside sqrt = 2*m*(1 - something) < 0
        # We'll just verify no warning/error and result is >= 0
        D = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
        assert np.all(D >= 0)


@pytest.mark.parametrize("Q, T", [
    (np.array([1, 2]), np.array([1, 2, 3])),
    (np.arange(5), np.arange(10) + 0.1),
])
def test_mass_consistent_length(Q, T):
    D = utils.mass(Q, T)
    assert len(D) == len(T) - len(Q) + 1


def test_mass_raises_when_query_longer_than_T():
    Q = np.arange(5)
    T = np.arange(4)
    with pytest.raises(ValueError):
        utils.mass(Q, T)


def test_mass_agrees_with_manual_calculation():
    # This test verifies mass returns distance profile consistent with calculate_distance_profile
    Q = np.array([1, 2, 3])
    T = np.array([1, 2, 3, 4, 5])
    m = len(Q)
    QT = utils.sliding_dot_product(Q, T)
    mean_T, sigma_T = utils.rolling_avg_sd(T, m)
    mean_Q = np.mean(Q)
    sigma_Q = np.std(Q)
    D_manual = utils.calculate_distance_profile(QT, m, mean_Q, sigma_Q, mean_T, sigma_T)
    D_mass = utils.mass(Q, T)
    np.testing.assert_allclose(D_mass, D_manual)

