import pytest

from easy_dict import NestedDict, asciify


def test_init_and_basic_mapping_protocol():
    nd = NestedDict({"a": 1, "b": 2})
    assert len(nd) == 2
    assert list(iter(nd)) == ["a", "b"]
    assert "a" in nd
    assert "missing" not in nd
    assert nd() == {"a": 1, "b": 2}
    assert repr(nd).startswith("easy_dict")


def test_getitem_direct_key_returns_value():
    nd = NestedDict({"x": 10})
    assert nd["x"] == 10


def test_getitem_nested_search_returns_scalar_when_single_match():
    nd = NestedDict({"a": {"target": 123}, "b": {"other": 456}})
    assert nd["target"] == 123


def test_getitem_nested_search_returns_tuple_when_multiple_matches():
    nd = NestedDict({"a": {"target": 1}, "b": {"target": 2}})
    assert nd["target"] == (1, 2)


def test_getitem_slash_path_traversal():
    nd = NestedDict({"a": {"b": {"c": 7}}})
    assert nd["a/b/c"] == 7


def test_getitem_list_path_traversal():
    nd = NestedDict({"a": {"b": {"c": 8}}})
    assert nd[["a", "b", "c"]] == 8


def test_getitem_missing_key_raises_keyerror():
    nd = NestedDict({"a": {"b": 1}})
    with pytest.raises(KeyError):
        _ = nd["does_not_exist"]


def test_setitem_direct_key_updates_root_value():
    nd = NestedDict({"x": 1})
    nd["x"] = 99
    assert nd["x"] == 99
    assert nd() == {"x": 99}


def test_setitem_nested_path_updates_deep_value_with_slash_string():
    nd = NestedDict({"a": {"b": 1}, "other": 0})
    nd["a/b"] = 42
    assert nd["a/b"] == 42
    assert nd["b"] == 42  # nested lookup finds the updated deep value


def test_setitem_nested_path_updates_deep_value_with_list():
    nd = NestedDict({"a": {"b": 1}, "c": {"d": 2}})
    nd[["a", "b"]] = 11
    assert nd["a/b"] == 11


def test_delitem_removes_root_key():
    nd = NestedDict({"a": 1, "b": 2})
    del nd["a"]
    assert "a" not in nd
    assert nd() == {"b": 2}


def test_setdefault_simple_key_uses_default_when_missing():
    nd = NestedDict({"a": 1})
    nd.setdefault("b", 2)
    assert nd["b"] == 2
    assert nd() == {"a": 1, "b": 2}


def test_setdefault_list_path_creates_nested_branch():
    nd = NestedDict({"a": {"b": 1}})
    nd.setdefault(["a", "c"], 9)
    assert nd["a/c"] == 9


def test_setdefault_list_path_raises_on_empty_branches():
    nd = NestedDict({"a": 1})
    with pytest.raises(KeyError):
        nd.setdefault(["a"], 10)


def test_anchor_creates_nested_branch_under_existing_trunk():
    nd = NestedDict({"root": {}})
    nd.anchor("root", "leaf", 5)
    assert nd["root/leaf"] == 5


def test_anchor_replaces_falsy_non_dict_trunk_with_dict():
    nd = NestedDict({"root": None})
    nd.anchor("root", "leaf", 6)
    assert nd["root/leaf"] == 6


def test_anchor_raises_valueerror_on_truthy_non_dict_trunk():
    nd = NestedDict({"root": "not-a-dict"})
    with pytest.raises(ValueError):
        nd.anchor("root", "leaf", 7)


def test_anchor_raises_keyerror_when_trunk_is_missing():
    nd = NestedDict({"other": {}})
    with pytest.raises(KeyError):
        nd.anchor("missing", "leaf", 8)


def test_asciify_returns_expected_tree_lines():
    tree = {"a": {"b": 1}, "c": 2}
    result = asciify(tree)
    assert result == ["a", "`-- b", "c"]


def test_asciify_empty_dict_returns_empty_list():
    assert asciify({}) == []
