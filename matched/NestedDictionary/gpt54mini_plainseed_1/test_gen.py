import pytest

from easy_dict import NestedDict, asciify


def test_init_and_basic_mapping_behavior():
    nd = NestedDict({"a": 1, "b": 2})
    assert len(nd) == 2
    assert set(iter(nd)) == {"a", "b"}
    assert "a" in nd
    assert nd["a"] == 1


def test_call_returns_underlying_dict_and_is_live():
    nd = NestedDict({"a": 1})
    backing = nd()
    assert isinstance(backing, dict)
    backing["b"] = 2
    assert nd["b"] == 2


def test_repr_includes_module_name_and_contents():
    nd = NestedDict({"a": 1})
    rep = repr(nd)
    assert "easy_dict" in rep
    assert "{'a': 1}" in rep


def test_getitem_nested_via_list_path():
    nd = NestedDict({"a": {"b": {"c": 3}}})
    assert nd[["a", "b", "c"]] == 3


def test_getitem_nested_via_slash_path():
    nd = NestedDict({"a": {"b": {"c": 3}}})
    assert nd["a/b/c"] == 3


def test_getitem_returns_tuple_for_multiple_matches():
    nd = NestedDict({"left": {"target": 1}, "right": {"target": 2}})
    assert nd["target"] == (1, 2)


def test_setitem_updates_nested_value_via_list_path():
    nd = NestedDict({"a": {"b": 1}})
    nd[["a", "b"]] = 99
    assert nd["a"]["b"] == 99
    assert nd["a/b"] == 99


def test_setitem_updates_nested_value_via_slash_path():
    nd = NestedDict({"a": {"b": 1}})
    nd["a/b"] = 42
    assert nd["a"]["b"] == 42


def test_setitem_creates_root_key_when_no_existing_branch_found():
    nd = NestedDict({"a": {"b": 1}})
    nd["new_key"] = "value"
    assert nd["new_key"] == "value"


def test_delitem_removes_root_key():
    nd = NestedDict({"a": 1, "b": 2})
    del nd["a"]
    assert "a" not in nd
    assert len(nd) == 1
    with pytest.raises(KeyError):
        _ = nd["a"]


def test_contains_and_iteration_over_keys():
    nd = NestedDict({"x": 1, "y": 2})
    assert "x" in nd
    assert sorted(list(nd)) == ["x", "y"]


def test_anchor_adds_nested_branch_under_existing_trunk():
    nd = NestedDict({"root": {"leaf": 1}})
    nd.anchor("root", "new_leaf", 2)
    assert nd["root"]["new_leaf"] == 2
    assert nd["root"]["leaf"] == 1


def test_anchor_on_missing_trunk_raises_key_error():
    nd = NestedDict({"root": {"leaf": 1}})
    with pytest.raises(KeyError):
        nd.anchor("missing", "child", 123)


def test_anchor_replaces_falsey_trunk_value_with_dict():
    nd = NestedDict({"root": 0})
    nd.anchor("root", "child", 123)
    assert nd["root"]["child"] == 123
    assert isinstance(nd["root"], dict)


def test_setdefault_simple_key_sets_default_when_missing():
    nd = NestedDict()
    nd.setdefault("a", 10)
    assert nd["a"] == 10


def test_setdefault_nested_list_path_sets_leaf_value():
    nd = NestedDict({"a": {}})
    nd.setdefault(["a", "b"], 7)
    assert nd["a"]["b"] == 7


def test_asciify_renders_expected_tree_structure():
    tree = {"a": {"b": 1}, "c": {"d": {"e": 2}}}
    lines = asciify(tree)
    assert isinstance(lines, list)
    assert "a" in lines[0]
    assert any("b" in line for line in lines)
    assert any("c" in line for line in lines)
    assert any("d" in line for line in lines)
    assert any("e" in line for line in lines)
