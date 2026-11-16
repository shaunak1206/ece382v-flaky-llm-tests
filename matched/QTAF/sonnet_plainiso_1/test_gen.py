# -*- coding: utf-8 -*-
import pytest

from testbase.conf import settings
from testbase.datadrive import (
    DataDrive,
    is_datadrive,
    get_datadrive,
    load_datadrive_tests,
)
from testbase.testcase import TestCase, Environ
from testbase.test import modify_settings


def _make_case(data):
    @DataDrive(data)
    class _DDCase(TestCase):
        """data drive case"""
        owner = "foo"
        priority = TestCase.EnumPriority.Normal
        status = TestCase.EnumStatus.Ready
        timeout = 1

        def run_test(self):
            pass

    return _DDCase


# ---------------------------------------------------------------- DataDrive

def test_datadrive_decorator_marks_class_and_returns_it():
    class Plain(TestCase):
        pass

    dd = DataDrive([1, 2])
    result = dd(Plain)
    assert result is Plain
    assert is_datadrive(Plain)
    assert get_datadrive(Plain) is dd


def test_datadrive_rejects_non_testcase_class():
    class NotCase(object):
        pass

    with pytest.raises(TypeError):
        DataDrive([1, 2, 3])(NotCase)


def test_is_datadrive_false_for_plain_testcase():
    class Plain(TestCase):
        pass

    assert not is_datadrive(Plain)


def test_datadrive_list_iterates_indices():
    dd = DataDrive(["a", "b", "c"])
    assert list(dd) == [0, 1, 2]
    assert len(dd) == 3
    assert dd[1] == "b"


def test_datadrive_tuple_iterates_indices():
    dd = DataDrive(("x", "y"))
    assert list(dd) == [0, 1]
    assert dd[0] == "x"


def test_datadrive_dict_iterates_keys():
    dd = DataDrive({"A": 1, "B": 2})
    assert sorted(dd) == ["A", "B"]
    assert dd["B"] == 2
    assert len(dd) == 2


def test_datadrive_generator_is_materialized():
    dd = DataDrive(x * 2 for x in range(3))
    assert list(dd) == [0, 1, 2]
    assert len(dd) == 3
    assert dd[2] == 4
    # iterating twice still works after conversion
    assert list(dd) == [0, 1, 2]


# --------------------------------------------------- load_datadrive_tests

def test_load_datadrive_tests_from_list():
    cls = _make_case([10, 20, 30])
    tests = load_datadrive_tests(cls)
    assert len(tests) == 3
    assert [t.casedata for t in tests] == [10, 20, 30]
    assert [t.casedataname for t in tests] == ["0", "1", "2"]
    assert all(isinstance(t, cls) for t in tests)


def test_load_datadrive_tests_from_dict_with_name():
    cls = _make_case({"one": 1, "two": 2})
    tests = load_datadrive_tests(cls, "two")
    assert len(tests) == 1
    assert tests[0].casedata == 2
    assert tests[0].casedataname == "two"
    assert tests[0].test_name.endswith("/two")


def test_load_datadrive_tests_list_by_index_name():
    cls = _make_case(["a", "b", "c"])
    tests = load_datadrive_tests(cls, 1)
    assert len(tests) == 1
    assert tests[0].casedata == "b"


def test_load_datadrive_tests_unknown_name_raises_value_error():
    cls = _make_case({"one": 1})
    with pytest.raises(ValueError):
        load_datadrive_tests(cls, "missing")


def test_load_datadrive_tests_non_datadrive_without_setting_raises():
    class Plain(TestCase):
        pass

    with modify_settings(DATA_DRIVE=False):
        with pytest.raises(RuntimeError):
            load_datadrive_tests(Plain)


def test_load_datadrive_tests_attrs_override():
    cls = _make_case({"k": {"__attrs__": {"owner": "bar", "timeout": 9, "bogus": 1}}})
    tests = load_datadrive_tests(cls)
    assert len(tests) == 1
    assert tests[0].owner == "bar"
    assert tests[0].timeout == 9
    assert not hasattr(tests[0], "bogus")


# ----------------------------------------------------------------- TestCase

def test_testcase_forbids_overloading_init():
    with pytest.raises(RuntimeError):
        class Bad(TestCase):
            def __init__(self):
                pass


def test_testcase_tags_inherit_and_instance_override():
    class Base(TestCase):
        tags = ["x"]

    class Child(Base):
        tags = "y"

    assert Base.tags == {"x"}
    assert Child.tags == {"x", "y"}
    assert Child.base_tags == {"x"}

    inst = Child(attrs={"tags": "z"})
    assert inst.tags == {"x", "z"}
    # class-level tags remain untouched
    assert Child.tags == {"x", "y"}


def test_testcase_without_tags_has_empty_set():
    class NoTags(TestCase):
        pass

    assert NoTags.tags == set()
    assert NoTags.base_tags == set()


def test_testcase_names_and_data():
    class Named(TestCase):
        pass

    plain = Named()
    assert plain.casedata is None
    assert plain.casedataname is None
    assert plain.test_class_name == Named.__module__ + ".Named"
    assert plain.test_name == plain.test_class_name

    named = Named("data", "n1")
    assert named.casedata == "data"
    assert named.casedataname == "n1"
    assert named.test_name == plain.test_class_name + "/n1"


def test_testcase_attrs_whitelist_and_doc_override():
    class Doc(TestCase):
        """  class doc  """
        owner = "a"

    inst = Doc(attrs={"owner": "b", "priority": "High", "evil": 1, "__doc__": "custom doc"})
    assert inst.owner == "b"
    assert inst.priority == "High"
    assert not hasattr(inst, "evil")
    assert inst.test_doc == "custom doc"
    # class attribute unchanged
    assert Doc.owner == "a"


def test_testcase_test_doc_is_stripped():
    class Doc(TestCase):
        """
        some description
        """

    assert Doc().test_doc == "some description"


def test_testcase_run_test_not_implemented_and_aliases():
    class Empty(TestCase):
        pass

    with pytest.raises(NotImplementedError):
        Empty().run_test()
    assert TestCase.runTest is TestCase.run_test
    assert TestCase.assertEqual is TestCase.assert_equal


def test_testcase_environ_is_singleton():
    class Env(TestCase):
        pass

    assert Env().environ is Environ()
    assert Env().environ is Env().environ


# ----------------------------------------------------------- modify_settings

def test_modify_settings_existing_key_restored():
    original = settings.DATA_DRIVE
    with modify_settings(DATA_DRIVE=not original):
        assert settings.DATA_DRIVE == (not original)
    assert settings.DATA_DRIVE == original


def test_modify_settings_new_key_added_then_removed():
    key = "QTAF_TEST_TEMP_KEY_XYZ"
    assert not hasattr(settings, key)
    with modify_settings(**{key: 42}):
        assert getattr(settings, key) == 42
    assert not hasattr(settings, key)


def test_modify_settings_restores_on_exception():
    original = settings.DATA_DRIVE
    with pytest.raises(ValueError):
        with modify_settings(DATA_DRIVE=not original):
            raise ValueError("boom")
    assert settings.DATA_DRIVE == original
