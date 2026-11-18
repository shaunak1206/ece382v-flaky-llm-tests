import pytest

from easy_dict import NestedDict, asciify


def test_init_and_basic_mapping_protocol():
    nd = NestedDict({"a": 1, "b": 2})
    assert len(nd) == 2
    assert set(iter(nd)) == {"a", "b"}
    assert "a" in nd
    assert nd["a"] == 1
    assert nd() == {"a": 1, "b": 2}


def test_repr_includes_module_name_and_data():
    nd = NestedDict({"x": 10})
    rep = repr(nd)
    assert "easy_dict" in rep
    assert "'x': 10" in rep


def test_getitem_nested_key_lookup_by_leaf_name():
    nd = NestedDict({"root": {"child": "value"}})
    assert nd["child"] == "value"


def test_getitem_nested_key_lookup_with_slash_path():
    nd = NestedDict({"root": {"child": {"leaf": "value"}}})
    assert nd["root/child/leaf"] == "value"


def test_getitem_nested_key_lookup_with_list_path():
    nd = NestedDict({"root": {"child": {"leaf": "value"}}})
    assert nd[["root", "child", "leaf"]] == "value"


def test_getitem_raises_keyerror_for_missing_key():
    nd = NestedDict({"root": {"child": "value"}})
    with pytest.raises(KeyError):
        _ = nd["missing"]


def test_setitem_updates_existing_nested_key_via_list_path():
    nd = NestedDict({"root": {"child": {"leaf": "old"}}})
    nd[["root", "child", "leaf"]] = "new"
    assert nd["root/child/leaf"] == "new"


def test_setitem_updates_existing_nested_key_via_slash_path():
    nd = NestedDict({"root": {"child": {"leaf": "old"}}})
    nd["root/child/leaf"] = "new"
    assert nd["root/child/leaf"] == "new"


def test_setitem_updates_top_level_key():
    nd = NestedDict({"a": 1})
    nd["a"] = 3
    assert nd["a"] == 3


def test_delitem_removes_top_level_key():
    nd = NestedDict({"a": 1, "b": 2})
    del nd["a"]
    assert "a" not in nd
    assert len(nd) == 1


def test_contains_checks_only_top_level_keys():
    nd = NestedDict({"root": {"child": "value"}})
    assert "root" in nd
    assert "child" not in nd


def test_anchor_adds_branch_under_existing_trunk():
    nd = NestedDict({"root": {"child": "value"}})
    nd.anchor("root", "new_branch", 123)
    assert nd["root/new_branch"] == 123
    assert nd["root/child"] == "value"


def test_anchor_raises_keyerror_when_trunk_is_missing():
    nd = NestedDict({"root": {"child": "value"}})
    with pytest.raises(KeyError):
        nd.anchor("missing", "branch", 1)


def test_setdefault_top_level_key_uses_default():
    nd = NestedDict({"a": 1})
    nd.setdefault("b", 2)
    assert nd["b"] == 2
    assert nd["a"] == 1


def test_setdefault_nested_path_creates_leaf_value():
    nd = NestedDict({"root": {}})
    nd.setdefault(["root", "child"], "value")
    assert nd["root/child"] == "value"


def test_asciify_renders_nested_structure():
    data = {"a": {"b": {"c": 1}}, "d": 2}
    lines = asciify(data)
    assert lines == [
        "a",
        "`-- b",
        "    `-- c",
        "d",
    ]


def test_asciify_empty_dict_returns_empty_list():
    assert asciify({}) == []
