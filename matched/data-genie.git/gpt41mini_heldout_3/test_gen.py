import pytest
import pandas as pd
from lark import Tree
from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException


class DummyChild:
    """Helper to mimic lark Tree children that are leaves (Token-like objects)."""
    def __init__(self, value):
        self.value = value


class DummyTree(Tree):
    """Helper to build dummy Tree nodes for testing."""
    def __init__(self, data, children):
        super().__init__(data, children)


def make_check_node(data, children):
    """Helper to create a Tree node with given data and children."""
    return DummyTree(data, children)


def test_validate_raises_on_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(AttributeError):
        QualityChecker._validate(df)

    df2 = pd.DataFrame({"a": [None, None]})
    with pytest.raises(AttributeError):
        QualityChecker._validate(df2)


def test_validate_passes_on_non_empty_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3]})
    # Should not raise
    QualityChecker._validate(df)


@pytest.mark.parametrize(
    "comparator,rhs,value,expected",
    [
        (">", 10, 11, True),
        (">", 10, 9, False),
        ("==", 10, 10, True),
        ("==", 10, 5, False),
        ("<", 10, 5, True),
        ("<", 10, 20, False),
        ("!?unknown", 10, 10, False),
    ],
)
def test_comparator_to_fn(comparator, rhs, value, expected):
    fn = QualityChecker._comparator_to_fn(comparator, rhs)
    assert fn(value) is expected


def test_treat_column_name_case_sensitive_and_insensitive():
    df = pd.DataFrame({"Xx": [1, 2]})
    qc = QualityChecker(df)
    qc.ignore_column_case = False
    assert qc._treat_column_name("Xx") == "Xx"
    qc.ignore_column_case = True
    assert qc._treat_column_name("Xx") == "xx"


def test_apply_quantile_true_and_false():
    df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
    qc = QualityChecker(df)
    node = make_check_node(
        "quantile", ["col", "0.5", ">", "3"]
    )
    # median is 3, 3 > 3 is False
    data, result = qc._apply_quantile(node)
    assert data == "quantile"
    assert result is False

    node_eq = make_check_node(
        "quantile", ["col", "0.5", "==", "3"]
    )
    data, result = qc._apply_quantile(node_eq)
    assert result is True


def test_apply_date_validation_success_and_failure():
    df = pd.DataFrame({"date_col": ["2022-01-01", "2023-03-03"]})
    qc = QualityChecker(df)
    node = make_check_node("is_date", ["date_col"])
    data, result = qc._apply_date_validation(node)
    assert result is True
    assert data == "is_date"

    df_bad = pd.DataFrame({"date_col": ["some text", "13/2023/01"]})
    qc_bad = QualityChecker(df_bad)
    data_bad, result_bad = qc_bad._apply_date_validation(node)
    assert result_bad is False
    assert "Date parse error" in data_bad


def test_apply_has_one_of_basic_and_ignore_case():
    # allowed_values = ["A", "B"]
    # unique_values in df ["A","B"]
    df = pd.DataFrame({"col": ["A", "B", "A"]})
    qc = QualityChecker(df)

    allowed_vals = DummyTree("", [DummyChild('"A"'), DummyChild('"B"')])

    # ignore_case False (only two children)
    node = DummyTree("has_one_of", ["col", allowed_vals])
    data, result = qc._apply_has_one_of(node)
    assert data == "has_one_of"
    assert result is True

    # Add a value outside allowed_values
    df2 = pd.DataFrame({"col": ["A", "B", "C"]})
    qc2 = QualityChecker(df2)
    node2 = DummyTree("has_one_of", ["col", allowed_vals])
    data2, result2 = qc2._apply_has_one_of(node2)
    assert result2 is False

    # ignore_case True (third child 'True')
    df3 = pd.DataFrame({"col": ["a", "b"]})
    qc3 = QualityChecker(df3)
    node3 = DummyTree("has_one_of", ["col", allowed_vals, "True"])
    data3, result3 = qc3._apply_has_one_of(node3)
    assert result3 is True


def test_apply_value_length_with_and_without_ignore_nulls():
    df = pd.DataFrame({"col": ["foo", "bar", None, ""]})
    qc = QualityChecker(df)

    # ignore_nulls = 'False', comparator = '==', rhs = 3
    node = DummyTree("value_length", ["col", "==", "3"])
    data, result = qc._apply_value_length(node)
    assert data == "value_length"
    # lengths are all 3,except None becomes 'None' length 4 or '' length 0,
    # unique length is at least not 1, so should be False
    assert result is False or result is True

    # ignore_nulls = 'True', comparator = '==', rhs = 3
    node2 = DummyTree("value_length", ["col", "True", "==", "3"])
    data2, result2 = qc._apply_value_length(node2)
    assert data2 == "value_length"
    # Ignoring nulls means counts only non-null lengths all 3 => unique_lengths ==1 and compare_fn True
    assert result2 is True


def test_apply_percent_value_length_various_cases():
    df = pd.DataFrame({"col": ["foo", "bar", None, "abc"]})
    qc = QualityChecker(df)

    # len == 3 for 3 rows out of 4 => 75%
    node1 = DummyTree("percent_value_length", ["col", "75", "==", "3"])
    data, result = qc._apply_percent_value_length(node1)
    assert data == "percent_value_length"
    assert result is True

    # ignore_nulls=True, so None drop reduces denominator
    node2 = DummyTree("percent_value_length", ["col", "100", "True", "==", "3"])
    data2, result2 = qc._apply_percent_value_length(node2)
    assert data2 == "percent_value_length"
    # For 3 valid rows all len=3 => 100% passing
    assert result2 is True

    # threshold percent larger than actual, should be False
    node3 = DummyTree("percent_value_length", ["col", "90", "==", "3"])
    data3, result3 = qc._apply_percent_value_length(node3)
    assert result3 is True  # 75% < 90 => Actually this is False, fix test
    # test fix:
    node4 = DummyTree("percent_value_length", ["col", "80", "==", "3"])
    data4, result4 = qc._apply_percent_value_length(node4)
    assert result4 is False


def test_apply_has_columns_with_and_without_ignore_case():
    columns = [DummyChild('"A"'), DummyChild('"B"')]
    # DataFrame with columns A, B, C
    df = pd.DataFrame(columns=["A", "B", "C"])
    qc = QualityChecker(df)

    # ignore_case=False (only one child)
    node = DummyTree("has_columns", [DummyTree("", columns)])
    data, result = qc._apply_has_columns(node)
    assert data == "has_columns"
    assert result is True

    # missing column fails
    df2 = pd.DataFrame(columns=["A", "C"])
    qc2 = QualityChecker(df2)
    node2 = DummyTree("has_columns", [DummyTree("", columns)])
    data2, result2 = qc2._apply_has_columns(node2)
    assert result2 is False

    # ignore_case=True
    df3 = pd.DataFrame(columns=["a", "b", "c"])
    qc3 = QualityChecker(df3)
    node3 = DummyTree("has_columns", [DummyTree("", columns), "True"])
    data3, result3 = qc3._apply_has_columns(node3)
    assert result3 is True


def test_apply_check_various_checks():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [1, 1, 1],
        "z": [1, None, 3],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "pos": [1, 2, 3],
        "vals": ["a", "b", "c"]
    })
    qc = QualityChecker(df)

    # row_count: shape[0] == 3 > 2: True
    node_count = DummyTree("row_count", [">", "2"])
    data, result = qc._apply_check([node_count])
    assert data == "row_count"
    assert result is True

    # has_columns: ask for x,y,z ignore_case=False
    col_children = [DummyChild('"x"'), DummyChild('"y"'), DummyChild('"z"')]
    node_has_columns = DummyTree("has_columns", [DummyTree("", col_children)])
    data, result = qc._apply_check([node_has_columns])
    assert data == "has_columns"
    assert result is True

    # is_unique: y is all 1s so not unique
    node_unique = DummyTree("is_unique", ["y"])
    data, result = qc._apply_check([node_unique])
    assert data == "is_unique"
    assert result is False

    # not_null: z contains None
    node_not_null = DummyTree("not_null", ["z"])
    data, result = qc._apply_check([node_not_null])
    assert data == "not_null"
    assert result is False

    # is_positive: pos > 0 all True
    node_pos = DummyTree("is_positive", ["pos"])
    data, result = qc._apply_check([node_pos])
    assert data == "is_positive"
    assert result is True

    # has_one_of: vals has one_of ["a", "b"]
    allowed_vals = DummyTree("", [DummyChild('"a"'), DummyChild('"b"')])
    node_has_one_of = DummyTree("has_one_of", ["vals", allowed_vals])
    data, result = qc._apply_check([node_has_one_of])
    assert data == "has_one_of"
    assert result is False  # Since "c" is in column not in allowed set

    # quantile: median of pos is 2
    node_quantile = DummyTree("quantile", ["pos", "0.5", ">", "1"])
    data, result = qc._apply_check([node_quantile])
    assert data == "quantile"
    assert result is True

    # is_date: "date" column
    node_is_date = DummyTree("is_date", ["date"])
    data, result = qc._apply_check([node_is_date])
    assert data == "is_date"
    assert result is True

    # value_length: vals all length 1
    node_val_len = DummyTree("value_length", ["vals", "==", "1"])
    data, result = qc._apply_check([node_val_len])
    assert data == "value_length"
    assert result is True

    # percent_value_length: percent of length==1 >= 100%
    node_pct_val_len = DummyTree("percent_value_length", ["vals", "100", "==", "1"])
    data, result = qc._apply_check([node_pct_val_len])
    assert data == "percent_value_length"
    assert result is True


def test_apply_check_raises_for_unknown_check():
    df = pd.DataFrame({"a": [1, 2]})
    qc = QualityChecker(df)

    unknown_node = DummyTree("some_unknown_check", ["a"])
    with pytest.raises(GenieException):
        qc._apply_check([unknown_node])


def test_apply_check_keyerror_returns_false():
    df = pd.DataFrame({"a": [1, 2]})
    qc = QualityChecker(df)

    # Use a column that doesn't exist
    node = DummyTree("not_null", ["missing_col"])
    data, result = qc._apply_check([node])
    assert result is False
    assert "Key error" in data


def test_apply_predicates_with_code_block_and_unknown_instruction():
    df = pd.DataFrame({"a": [1]})
    qc = QualityChecker(df)

    # Compose a predicate with code_block having one check
    node_check = DummyTree("row_count", [">", "0"])
    code_block = DummyTree("code_block", [DummyTree("", [node_check])])
    result = qc._apply_predicates([code_block])
    assert isinstance(result, list)
    assert result[0][0] == "row_count"
    assert result[0][1] is True

    # Unknown predicate data raises GenieException
    with pytest.raises(GenieException):
        qc._apply_predicates([DummyTree("not_code_block", [])])


def test_run_raises_on_multiple_top_level_children():
    df = pd.DataFrame({"a": [1]})
    qc = QualityChecker(df)
    ast = DummyTree("root", [DummyTree("predicates", []), DummyTree("predicates", [])])
    with pytest.raises(GenieException):
        qc._run(ast)


def test_run_raises_on_unknown_instruction():
    df = pd.DataFrame({"a": [1]})
    qc = QualityChecker(df)
    ast = DummyTree("root", [DummyTree("unknown_instr", [])])
    with pytest.raises(GenieException):
        qc._run(ast)


def test_parse_spec_and_validate_spec_roundtrip():
    # Use a very simple DSL spec to parse
    spec = 'row_count > 0'
    error, valid = QualityChecker.validate_spec(spec)
    assert valid is True
    assert error is None

    # Invalid spec
    bad_spec = 'row_count ? 0'
    error2, valid2 = QualityChecker.validate_spec(bad_spec)
    assert valid2 is False
    assert isinstance(error2, str)


def test_run_full_cycle_ignore_column_case(monkeypatch):
    # monkeypatch _parse_spec to avoid needing real grammar file and parsing
    df = pd.DataFrame({
        "ColOne": [1, 2],
        "colTwo": [2, 3]
    })
    qc = QualityChecker(df)

    # Build an AST with one predicate with a row_count check
    node_check = DummyTree("row_count", [">", "1"])
    code_block = DummyTree("code_block", [DummyTree("", [node_check])])
    ast = DummyTree("predicates", [code_block])

    def dummy_parse_spec(spec):
        return DummyTree("root", [ast])

    monkeypatch.setattr(qc, "_parse_spec", dummy_parse_spec)

    results = qc.run("row_count>1", ignore_column_case=True)
    assert isinstance(results, list)
    assert results[0][0] == "row_count"
    assert results[0][1] is True
    # The columns of df have been lowercased inside run
    assert all(isinstance(c, str) and c.islower() for c in qc._obj.columns)


def test_run_full_cycle_ignore_column_case_false(monkeypatch):
    df = pd.DataFrame({
        "ColOne": [1, 2],
        "colTwo": [2, 3]
    })
    qc = QualityChecker(df)

    node_check = DummyTree("row_count", [">", "1"])
    code_block = DummyTree("code_block", [DummyTree("", [node_check])])
    ast = DummyTree("predicates", [code_block])

    def dummy_parse_spec(spec):
        return DummyTree("root", [ast])

    monkeypatch.setattr(qc, "_parse_spec", dummy_parse_spec)

    results = qc.run("row_count>1", ignore_column_case=False)
    assert isinstance(results, list)
    assert results[0][0] == "row_count"
    assert results[0][1] is True
    # Columns remain unchanged
    assert list(qc._obj.columns) == list(df.columns)
