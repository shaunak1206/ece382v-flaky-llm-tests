import pytest

from pymlfunc import sub2ind
from pymlfunc import subind as subind_module


# --- Basic 2D conversions (column-major, 1-based, MATLAB semantics) ---

def test_sub2ind_first_element_is_one():
    assert sub2ind((3, 4), 1, 1) == 1


def test_sub2ind_first_column_walks_down_rows():
    assert sub2ind((3, 4), 2, 1) == 2
    assert sub2ind((3, 4), 3, 1) == 3


def test_sub2ind_second_column_offset_by_row_count():
    assert sub2ind((3, 4), 1, 2) == 4
    assert sub2ind((3, 4), 3, 2) == 6


def test_sub2ind_last_element_equals_total_count():
    assert sub2ind((3, 4), 3, 4) == 12


def test_sub2ind_square_matrix():
    assert sub2ind((2, 2), 1, 1) == 1
    assert sub2ind((2, 2), 2, 1) == 2
    assert sub2ind((2, 2), 1, 2) == 3
    assert sub2ind((2, 2), 2, 2) == 4


def test_sub2ind_single_row_matrix():
    assert sub2ind((1, 5), 1, 3) == 3
    assert sub2ind((1, 5), 1, 5) == 5


def test_sub2ind_single_column_matrix():
    assert sub2ind((5, 1), 4, 1) == 4


def test_sub2ind_single_element_matrix():
    assert sub2ind((1, 1), 1, 1) == 1


# --- 3D conversions ---

def test_sub2ind_3d_first_element():
    assert sub2ind((2, 3, 4), 1, 1, 1) == 1


def test_sub2ind_3d_along_each_axis():
    assert sub2ind((2, 3, 4), 2, 1, 1) == 2
    assert sub2ind((2, 3, 4), 1, 2, 1) == 3
    assert sub2ind((2, 3, 4), 1, 1, 2) == 7


def test_sub2ind_3d_last_element_equals_total_count():
    assert sub2ind((2, 3, 4), 2, 3, 4) == 24


# --- Consistency with the column-major formula ---

@pytest.mark.parametrize("dims", [(3, 4), (5, 2), (4, 4), (1, 7)])
def test_sub2ind_matches_column_major_formula_2d(dims):
    rows, cols = dims
    expected = 1
    for c in range(1, cols + 1):
        for r in range(1, rows + 1):
            assert sub2ind(dims, r, c) == expected
            expected += 1


def test_sub2ind_produces_unique_indices_over_full_grid():
    dims = (3, 5)
    results = {
        sub2ind(dims, r, c)
        for r in range(1, dims[0] + 1)
        for c in range(1, dims[1] + 1)
    }
    assert results == set(range(1, dims[0] * dims[1] + 1))


# --- Error handling ---

@pytest.mark.parametrize("row, col", [(0, 1), (1, 0), (-1, 1), (1, -2)])
def test_sub2ind_rejects_non_positive_subscripts(row, col):
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), row, col)


@pytest.mark.parametrize("row, col", [(4, 1), (1, 5), (10, 10)])
def test_sub2ind_rejects_out_of_bounds_subscripts(row, col):
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), row, col)


# --- Module wiring ---

def test_sub2ind_exported_from_package_is_module_function():
    assert sub2ind is subind_module.sub2ind
