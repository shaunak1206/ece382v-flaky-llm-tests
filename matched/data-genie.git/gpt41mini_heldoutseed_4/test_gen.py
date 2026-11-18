import pytest
import pandas as pd
from lark import Tree
from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException


def test_validate_raises_on_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(AttributeError):
        QualityChecker._validate(df)

    df = pd.DataFrame({'A': [None, None]})
    with pytest.raises(AttributeError):
        QualityChecker._validate(df)

def test_comparator_to_fn_behaviour():
    fn = QualityChecker._comparator_to_fn(">", 5)
    assert fn(6) is True
    assert fn(5) is False

    fn = QualityChecker._comparator_to_fn("==", 3)
    assert fn(3) is True
    assert fn(4) is False

    fn = QualityChecker._comparator_to_fn("<", 10)
    assert fn(9) is True
    assert fn(10) is False

    # unknown comparator returns False always
    fn = QualityChecker._comparator_to_fn("!=", 1)
    assert fn(0) is False
    assert fn(1) is False

def test_treat_column_name_respects_ignore_case():
    df = pd.DataFrame({'Col': [1]})
    qc = QualityChecker(df)
    qc.ignore_column_case = False
    assert qc._treat_column_name("Col") == "Col"
    qc.ignore_column_case = True
    assert qc._treat_column_name("Col") == "col"

def make_node(data, children):
    return Tree(data, children)

def test_apply_quantile_true_false():
    df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
    qc = QualityChecker(df)
    # node: quantile col 0.5 > 3
    node = make_node('quantile', ['col', '0.5', '>', '3'])
    name, result = qc._apply_quantile(node)
    assert name == 'quantile'
    # median is 3, 3 > 3 is False
    assert result is False

    node = make_node('quantile', ['col', '0.5', '<', '4'])
    _, result = qc._apply_quantile(node)
    # median 3 < 4 True
    assert result is True

def test_apply_date_validation_success_and_failure():
    df = pd.DataFrame({'dt': ['2021-01-01', '2022-12-31']})
    qc = QualityChecker(df)
    node = make_node("is_date", ["dt"])
    name, result = qc._apply_date_validation(node)
    assert name == "is_date"
    assert result is True

    df = pd.DataFrame({'dt': ['bad-date', 'also-bad']})
    qc = QualityChecker(df)
    node = make_node("is_date", ["dt"])
    name, result = qc._apply_date_validation(node)
    assert "Date parse error" in name
    assert result is False

def test_apply_has_one_of_various_cases():
    df = pd.DataFrame({'col': ['A', 'B', 'C']})
    qc = QualityChecker(df)
    # allowed values: A, B, C, no ignore case
    allowed_vals = make_node('allowed_vals', [Tree('VALUE', [type('Token', (), {'value': '"A"'})()]),
                                             Tree('VALUE', [type('Token', (), {'value': '"B"'})()]),
                                             Tree('VALUE', [type('Token', (), {'value': '"C"'})()])])
    node = make_node('has_one_of', ['col', allowed_vals])
    name, result = qc._apply_has_one_of(node)
    assert name == 'has_one_of'
    assert result is True

    # Now add a value not in allowed values => should fail
    df = pd.DataFrame({'col': ['A', 'B', 'D']})
    qc = QualityChecker(df)
    node = make_node('has_one_of', ['col', allowed_vals])
    _, result = qc._apply_has_one_of(node)
    assert result is False

    # test ignore_case True, col has 'a', allowed vals uppercase, should be True
    df = pd.DataFrame({'col': ['a', 'b', 'c']})
    qc = QualityChecker(df)
    node = make_node('has_one_of', ['col', allowed_vals, 'True'])
    _, result = qc._apply_has_one_of(node)
    assert result is True

def test_apply_value_length_with_and_without_ignore_nulls():
    df = pd.DataFrame({'col': ['aaa', 'bbb', 'ccc']})
    qc = QualityChecker(df)

    # length all 3, test comparator == 3, ignore_nulls omitted (default 'False')
    node = make_node('value_length', ['col', '==', '3'])
    name, result = qc._apply_value_length(node)
    assert name == 'value_length'
    assert result is True

    # Use comparator > 2 (should be True)
    node = make_node('value_length', ['col', '>', '2'])
    _, result = qc._apply_value_length(node)
    assert result is True

    # Use comparator < 3 (should be False)
    node = make_node('value_length', ['col', '<', '3'])
    _, result = qc._apply_value_length(node)
    assert result is False

    # Add null values with empty strings, using ignore_nulls = 'True'
    df = pd.DataFrame({'col': ['aaa', '', 'bbb', '   ', None]})
    qc = QualityChecker(df)
    # length of non-null entries: 3 (aaa), 3 (bbb)
    node = make_node('value_length', ['col', 'True', '==', '3'])
    _, result = qc._apply_value_length(node)
    assert result is True

    # If lengths differ in data, should be False
    df = pd.DataFrame({'col': ['aaa', 'bb', 'bbb']})
    qc = QualityChecker(df)
    node = make_node('value_length', ['col', '==', '3'])
    _, result = qc._apply_value_length(node)
    assert result is False

def test_apply_percent_value_length_basic_cases():
    df = pd.DataFrame({'col': ['aaa', 'bb', 'ccc', 'dddd']})
    qc = QualityChecker(df)

    # percent=50, rhs=3, ignore_nulls omitted (False)
    node = make_node('percent_value_length', ['col', '50', '==', '3'])
    name, result = qc._apply_percent_value_length(node)
    assert name == 'percent_value_length'
    # Only 'aaa' and 'ccc' have length 3, 2/4=50%, so result True
    assert result is True

    # now percent=75, rhs=3, should fail
    node = make_node('percent_value_length', ['col', '75', '==', '3'])
    _, result = qc._apply_percent_value_length(node)
    assert result is False

    # Test with ignore_nulls='True', include empty string as null
    df = pd.DataFrame({'col': ['aaa', '', 'ccc', None]})
    qc = QualityChecker(df)
    node = make_node('percent_value_length', ['col', '50', 'True', '==', '3'])
    _, result = qc._apply_percent_value_length(node)
    assert result is True

def test_apply_has_columns_with_and_without_ignore_case():
    df = pd.DataFrame({'A': [1], 'b': [2], 'C': [3]})
    qc = QualityChecker(df)

    col_nodes = Tree('columns', [type('Token', (), {'value': '"A"'})(),
                                 type('Token', (), {'value': '"b"'})()])
    node = make_node('has_columns', [col_nodes])
    name, result = qc._apply_has_columns(node)
    assert name == 'has_columns'
    assert result is True

    # Add ignore_case True, expecting lowercase versions of columns
    node = make_node('has_columns', [col_nodes, 'True'])
    _, result = qc._apply_has_columns(node)
    assert result is True

    # Missing a column should fail
    col_missing_nodes = Tree('columns', [type('Token', (), {'value': '"A"'})(),
                                         type('Token', (), {'value': '"X"'})()])
    node = make_node('has_columns', [col_missing_nodes])
    _, result = qc._apply_has_columns(node)
    assert result is False

def test_apply_check_row_count():
    df = pd.DataFrame({'x': range(5)})
    qc = QualityChecker(df)
    node = make_node('row_count', ['>', '3'])
    # check argument to _apply_check is a list with a single Tree wrapping node
    check = [make_node("row_count", ['>', '3'])]
    name, result = qc._apply_check(check)
    assert name == 'row_count'
    assert result is True

    check = [make_node("row_count", ['<', '3'])]
    _, result = qc._apply_check(check)
    assert result is False

def test_apply_check_has_columns_and_not_null_and_is_unique_and_is_positive():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 2, 2], 'C': [1, 2, 3], 'D': [None, 1, 2], 'E': [1, 2, 3]})
    df['E'] = df['E'].astype(int)
    qc = QualityChecker(df)

    # has_columns
    col_nodes = Tree('columns', [type('Token', (), {'value': '"A"'})(),
                                 type('Token', (), {'value': '"B"'})()])
    node = make_node('has_columns', [col_nodes])
    name, result = qc._apply_check([node])
    assert name == 'has_columns'
    assert result is True

    # not_null passes
    node = make_node('not_null', ['A'])
    _, result = qc._apply_check([node])
    assert result is True

    # not_null fails (D contains None)
    node = make_node('not_null', ['D'])
    _, result = qc._apply_check([node])
    assert result is False

    # is_unique true for column C
    node = make_node('is_unique', ['C'])
    _, result = qc._apply_check([node])
    assert result is True

    # is_unique false for B (all same but duplicates)
    node = make_node('is_unique', ['B'])
    _, result = qc._apply_check([node])
    assert result is False

    # is_positive true for A
    node = make_node('is_positive', ['A'])
    _, result = qc._apply_check([node])
    assert result is True

    # add a zero to column and test is_positive false
    df.loc[0, 'A'] = 0
    qc = QualityChecker(df)
    node = make_node('is_positive', ['A'])
    _, result = qc._apply_check([node])
    assert result is False

def test_apply_check_raises_for_unknown_check():
    df = pd.DataFrame({'x': [1, 2]})
    qc = QualityChecker(df)
    node = make_node('unknown_check', ['x'])
    with pytest.raises(GenieException):
        qc._apply_check([node])

def test_apply_check_key_error_returns_false():
    df = pd.DataFrame({'a': [1, 2]})
    qc = QualityChecker(df)
    # referring to non-existing column
    node = make_node('not_null', ['nonexistent'])
    name, result = qc._apply_check([node])
    assert name.startswith("Key error")
    assert result is False

def test_apply_predicates_with_code_block():
    df = pd.DataFrame({'A': [1, 1], 'B': [2, 2]})
    qc = QualityChecker(df)

    # Build predicate tree with code_block that has one check: is_unique A (False)
    code_block = make_node('code_block', [make_node('is_unique', ['A'])])
    predicates = make_node('predicates', [code_block])
    results = qc._apply_predicates(predicates.children)
    assert isinstance(results, list)
    # The only check returns a tuple with ('is_unique', bool)
    assert results[0][0] == 'is_unique'
    assert results[0][1] is False

def test_apply_predicates_unknown_instruction_raises():
    df = pd.DataFrame({'A': [1]})
    qc = QualityChecker(df)
    unknown = make_node('unknown', [])
    predicates = make_node('predicates', [unknown])
    with pytest.raises(GenieException):
        qc._apply_predicates(predicates.children)

def test__run_accepts_predicates_instruction_and_unknown_raises():
    df = pd.DataFrame({'A': [1]})
    qc = QualityChecker(df)
    code_block = make_node('code_block', [make_node('is_unique', ['A'])])
    predicates = make_node('predicates', [code_block])
    pt = make_node('root', [predicates])
    results = qc._run(pt)
    assert isinstance(results, list)
    assert results[0][0] == 'is_unique'

    # Now test multiple children raises
    pt2 = make_node('root', [predicates, predicates])
    with pytest.raises(GenieException):
        qc._run(pt2)

    # test unknown instruction
    unknown_instr = make_node('something_else', [])
    pt3 = make_node('root', [unknown_instr])
    with pytest.raises(GenieException):
        qc._run(pt3)

def test_validate_spec_returns_none_and_true_on_valid_spec():
    valid_spec = "row_count > 0"
    err, ok = QualityChecker.validate_spec(valid_spec)
    assert ok is True
    assert err is None

def test_validate_spec_returns_error_and_false_on_invalid_spec():
    invalid_spec = "row_count >> 0"
    err, ok = QualityChecker.validate_spec(invalid_spec)
    assert ok is False
    assert isinstance(err, str)

def test_run_returns_results_and_sets_ignore_column_case(monkeypatch):
    df = pd.DataFrame({'A': [1], 'b': [2]})
    qc = QualityChecker(df.copy())

    spec = "has_columns [\"a\"]"
    # Patch _parse_spec to produce the corresponding parse tree instead of parsing for the test
    node_cols = Tree('columns', [type('Token', (), {'value': '"a"'})()])
    has_columns_node = Tree('has_columns', [node_cols])
    code_block_node = Tree('code_block', [has_columns_node])
    predicates_node = Tree('predicates', [code_block_node])
    ast = Tree('root', [predicates_node])

    def fake_parse(spec):
        return ast

    monkeypatch.setattr(QualityChecker, "_parse_spec", fake_parse)
    results = qc.run(spec, ignore_column_case=True)
    # When ignore_column_case is True, columns are lowered
    assert qc.ignore_column_case is True
    assert list(qc._obj.columns) == ['a', 'b']
    assert isinstance(results, list)
    assert results[0][0] == "has_columns"
    # 'a' is in lowered columns, so should be True
    assert results[0][1] is True
