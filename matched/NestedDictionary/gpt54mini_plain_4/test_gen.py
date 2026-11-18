import pytest

from easy_dict import NestedDict, asciify


def test_init_and_basic_mapping_protocol():
    nd = NestedDict({"a": 1, "b": 2})
    assert len(nd) == 2
    assert "a" in nd
    assert nd["a"] == 1
    assert dict(iter(nd)) == {"a": 1, "b": 2}
    assert nd() == {"a": 1, "b": 2}


def test_repr_contains_module_name_and_values():
    nd = NestedDict({"x": 10})
    rep = repr(nd)
    assert "easy_dict" in rep
    assert "x" in rep
    assert "10" in rep


def test_getitem_returns_nested_value_by_key():
    nd = NestedDict({"outer": {"inner": "value"}})
    assert nd["outer"]["inner"] == "value"
    assert nd["inner"] == "value"


def test_getitem_with_slash_path_list_lookup():
    nd = NestedDict({"outer": {"inner": {"leaf": 42}}})
    assert nd["outer/inner/leaf"] == 42
    assert nd[["outer", "inner", "leaf"]] == 42


def test_getitem_missing_key_raises_keyerror():
    nd = NestedDict({"a": {"b": 1}})
    with pytest.raises(KeyError):
        _ = nd["missing"]


def test_setitem_existing_root_key_updates_value():
    nd = NestedDict({"a": 1})
    nd["a"] = 99
    assert nd["a"] == 99
    assert nd() == {"a": 99}


def test_setitem_nested_by_path_creates_update_in_deeper_dict():
    nd = NestedDict({"a": {"b": 1}})
    nd["a/b"] = 7
    assert nd["a"]["b"] == 7
    assert nd["a/b"] == 7


def test_setitem_list_path_updates_nested_branch():
    nd = NestedDict({"a": {"b": {"c": 1}}})
    nd[["a", "b", "c"]] = 11
    assert nd["a"]["b"]["c"] == 11


def test_setitem_missing_top_level_key_adds_key_at_root():
    nd = NestedDict({"a": {"b": 1}})
    nd["new_key"] = "created"
    assert nd["new_key"] == "created"
    assert nd() == {"a": {"b": 1}, "new_key": "created"}


def test_delitem_removes_key():
    nd = NestedDict({"a": 1, "b": 2})
    del nd["a"]
    assert "a" not in nd
    assert dict(nd()) == {"b": 2}


def test_anchor_adds_branch_to_existing_trunk():
    nd = NestedDict({"root": {"keep": True}})
    nd.anchor("root", "leaf", value=123)
    assert nd["root"]["leaf"] == 123
    assert nd["root"]["keep"] is True


def test_anchor_raises_keyerror_when_trunk_missing():
    nd = NestedDict({"other": {}})
    with pytest.raises(KeyError):
        nd.anchor("missing", "leaf", value=1)


def test_anchor_replaces_none_trunk_with_dict():
    nd = NestedDict({"root": None})
    nd.anchor("root", "leaf", value="x")
    assert nd["root"]["leaf"] == "x"
    assert nd() == {"root": {"leaf": "x"}}


def test_anchor_raises_valueerror_when_trunk_is_truthy_non_dict():
    nd = NestedDict({"root": 1})
    with pytest.raises(ValueError):
        nd.anchor("root", "leaf", value=2)


def test_setdefault_simple_key_sets_default_when_missing():
    nd = NestedDict({"a": 1})
    nd.setdefault("b", 5)
    assert nd["b"] == 5
    assert nd["a"] == 1


def test_setdefault_path_can_create_nested_default():
    nd = NestedDict({})
    nd.setdefault(["a", "b"], default={"c": 3})
    assert "a" in nd
    assert nd["a"] == {"c": 3}


def test_asciify_simple_tree_output():
    tree = {"a": {"b": {"c": 1}}, "d": 2}
    lines = asciify(tree)
    assert lines == ["+-- a", "|   `-- b", "|       `-- c", "`-- d"]


def test_asciify_empty_dict_returns_empty_list():
    assert asciify({}) == []
