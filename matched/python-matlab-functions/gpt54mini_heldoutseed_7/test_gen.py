import inspect

import pytest

from pymlfunc import sub2ind


def _call_sub2ind(shape, *subs):
    """
    Call sub2ind with a few common MATLAB-like signatures.

    The package may expose sub2ind as either:
      - sub2ind(shape, rows, cols, ...)
      - sub2ind(shape, (rows, cols, ...))
      - sub2ind(shape, [rows, cols, ...])
    """
    # First try the most likely form.
    try:
        return sub2ind(shape, *subs)
    except TypeError:
        pass

    # Try passing the subscripts as a single container.
    for packed in (subs, list(subs), tuple(subs)):
        try:
            return sub2ind(shape, packed)
        except TypeError:
            continue

    # Last resort: if signature is unusual, fail loudly with context.
    sig = None
    try:
        sig = str(inspect.signature(sub2ind))
    except Exception:
        sig = "<unavailable>"
    raise TypeError(f"Could not call sub2ind with shape={shape}, subs={subs}; signature={sig}")


def test_sub2ind_is_exported_from_package_init():
    assert callable(sub2ind)


def test_sub2ind_first_element_2d():
    assert _call_sub2ind((3, 4), 1, 1) == 1


def test_sub2ind_second_row_first_column_2d():
    assert _call_sub2ind((3, 4), 2, 1) == 2


def test_sub2ind_first_row_second_column_2d():
    # MATLAB/Fortran column-major linear indexing
    assert _call_sub2ind((3, 4), 1, 2) == 4


def test_sub2ind_last_element_2d():
    assert _call_sub2ind((3, 4), 3, 4) == 12


def test_sub2ind_middle_element_2d():
    assert _call_sub2ind((3, 4), 2, 3) == 8


def test_sub2ind_3d_first_element():
    assert _call_sub2ind((2, 3, 4), 1, 1, 1) == 1


def test_sub2ind_3d_crosses_first_dimension_fastest():
    assert _call_sub2ind((2, 3, 4), 2, 1, 1) == 2
    assert _call_sub2ind((2, 3, 4), 1, 2, 1) == 3
    assert _call_sub2ind((2, 3, 4), 1, 1, 2) == 7


def test_sub2ind_3d_last_element():
    assert _call_sub2ind((2, 3, 4), 2, 3, 4) == 24


def test_sub2ind_accepts_tuple_subscripts():
    assert _call_sub2ind((3, 4), (2, 3)) == 8


def test_sub2ind_accepts_list_subscripts():
    assert _call_sub2ind((3, 4), [2, 3]) == 8


def test_sub2ind_accepts_numpy_like_input_not_required_but_sequence_works():
    # Using plain Python sequences keeps the test dependency-light.
    # This still verifies that sequence-style inputs are accepted.
    assert _call_sub2ind([3, 4], [3, 4]) == 12


def test_sub2ind_rejects_zero_index():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 0, 1)


def test_sub2ind_rejects_negative_index():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), -1, 1)


def test_sub2ind_rejects_out_of_bounds_row():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 4, 1)


def test_sub2ind_rejects_out_of_bounds_column():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 1, 5)


def test_sub2ind_rejects_too_few_subscripts():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 1)


def test_sub2ind_rejects_too_many_subscripts():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 1, 1, 1)


def test_sub2ind_rejects_non_integer_subscripts():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 1.5, 2)


def test_sub2ind_rejects_non_integer_shape():
    with pytest.raises(Exception):
        _call_sub2ind((3.0, 4), 1, 1)
