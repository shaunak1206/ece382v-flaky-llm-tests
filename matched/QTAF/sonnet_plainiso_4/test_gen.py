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


class _FakeResult(object):
    def __init__(self):
        self.records = []

    def log_record(self, level, msg, *args, **kwargs):
        self.records.append((level, msg))


def _make_list_case():
    @DataDrive([10, 20, 30])
    class ListCase(TestCase):
        """list case"""
        owner = "foo"
        priority = TestCase.EnumPriority.Normal
        status = TestCase.EnumStatus.Ready
        timeout = 1

        def run_test(self):
            pass

    return ListCase


def _make_dict_case():
    @DataDrive({"A": 1, "B": 2})
    class DictCase(TestCase):
        """dict case"""
        owner = "foo"
        priority = TestCase.EnumPriority.Normal
        status = TestCase.EnumStatus.Ready
        timeout = 1

        def run_test(self):
            pass

    return DictCase


def test_datadrive_rejects_non_testcase():
    with pytest.raises(TypeError):
        DataDrive([1, 2])(object)


def test_datadrive_decorator_marks_class():
    cls = _make_list_case()
    assert is_datadrive(cls)
    dd = get_datadrive(cls)
    assert isinstance(dd, DataDrive)
    assert len(dd) == 3


def test_non_decorated_class_is_not_datadrive():
    class Plain(TestCase):
        pass

    assert not is_datadrive(Plain)


def test_iter_list_yields_indices_and_getitem():
    dd = DataDrive(["a", "b", "c"])
    assert list(dd) == [0, 1, 2]
    assert dd[1] == "b"
    assert len(dd) == 3


def test_iter_tuple_yields_indices():
    dd = DataDrive(("x", "y"))
    assert list(dd) == [0, 1]


def test_iter_dict_yields_keys():
    dd = DataDrive({"k1": 1, "k2": 2})
    assert sorted(dd) == ["k1", "k2"]
    assert dd["k2"] == 2
    assert len(dd) == 2


def test_generator_data_converted_to_list():
    dd = DataDrive(i * 2 for i in range(3))
    assert list(dd) == [0, 1, 2]
    assert dd[2] == 4
    assert len(dd) == 3


def test_load_list_datadrive_tests():
    cls = _make_list_case()
    tests = load_datadrive_tests(cls)
    assert len(tests) == 3
    assert [t.casedataname for t in tests] == ["0", "1", "2"]
    assert [t.casedata for t in tests] == [10, 20, 30]
    assert all(isinstance(t, cls) for t in tests)


def test_load_dict_datadrive_tests_with_name():
    cls = _make_dict_case()
    tests = load_datadrive_tests(cls, "B")
    assert len(tests) == 1
    assert tests[0].casedata == 2
    assert tests[0].casedataname == "B"


def test_load_list_datadrive_by_string_index():
    cls = _make_list_case()
    tests = load_datadrive_tests(cls, "1")
    assert len(tests) == 1
    assert tests[0].casedata == 20
    assert tests[0].casedataname == "1"


def test_load_datadrive_unknown_name_raises_value_error():
    cls = _make_dict_case()
    with pytest.raises(ValueError):
        load_datadrive_tests(cls, "NOPE")


def test_load_non_datadrive_without_setting_raises_runtime_error():
    class Plain(TestCase):
        pass

    with modify_settings(DATA_DRIVE=False):
        with pytest.raises(RuntimeError):
            load_datadrive_tests(Plain)


def test_attrs_in_drive_data_override_attributes():
    @DataDrive({"one": {"__attrs__": {"owner": "bar", "priority": "High",
                                      "tags": "extra", "__doc__": "custom doc",
                                      "unknown_attr": 5}}})
    class AttrCase(TestCase):
        """orig doc"""
        owner = "foo"
        priority = TestCase.EnumPriority.Low
        status = TestCase.EnumStatus.Ready
        timeout = 1
        tags = "base"

    test = load_datadrive_tests(AttrCase)[0]
    assert test.owner == "bar"
    assert test.priority == "High"
    assert test.tags == {"base", "extra"}
    assert test.test_doc == "custom doc"
    assert not hasattr(test, "unknown_attr")
    # class-level values unchanged
    assert AttrCase.owner == "foo"


def test_testcase_names_and_casedata():
    class NameCase(TestCase):
        pass

    plain = NameCase()
    assert plain.casedata is None
    assert plain.casedataname is None
    assert plain.test_class_name.endswith("NameCase")
    assert plain.test_name == plain.test_class_name

    driven = NameCase({"v": 1}, "n1")
    assert driven.casedata == {"v": 1}
    assert driven.casedataname == "n1"
    assert driven.test_name == driven.test_class_name + "/n1"


def test_testcase_forbids_overriding_init():
    with pytest.raises(RuntimeError):
        class Bad(TestCase):
            def __init__(self):
                pass


def test_tags_merge_with_base_class():
    class Parent(TestCase):
        tags = ["p1", "p2"]

    class Child(Parent):
        tags = "c1"

    class NoTags(TestCase):
        pass

    assert Parent.tags == {"p1", "p2"}
    assert Child.tags == {"c1", "p1", "p2"}
    assert Child.base_tags == {"p1", "p2"}
    assert NoTags.tags == set()


def test_run_test_not_implemented_and_aliases():
    case = TestCase()
    with pytest.raises(NotImplementedError):
        case.run_test()
    assert TestCase.runTest is TestCase.run_test
    assert TestCase.preTest is TestCase.pre_test
    assert TestCase.assertEqual is TestCase.assert_equal


def test_test_doc_is_stripped():
    class DocCase(TestCase):
        """
        some description
        """

    assert DocCase().test_doc == "some description"


def test_test_extra_info():
    class ExtraCase(TestCase):
        test_extra_info_def = [("foo", "desc"), ("missing", "desc")]
        foo = 42

    assert ExtraCase().test_extra_info == {"foo": 42, "missing": None}


def test_environ_is_singleton_dict():
    case = TestCase()
    assert case.environ is Environ()
    assert isinstance(case.environ, dict)


def test_assert_equal_and_match_results():
    case = TestCase()
    result = _FakeResult()
    case.init_test(result)
    assert case.test_result is result

    assert case.assert_equal("same", 1, 1) is True
    assert case.assert_equal("str", "abc", "abc") is True
    assert result.records == []

    assert case.assert_equal("diff", 1, 2) is False
    assert len(result.records) == 1

    assert case.assert_match("match", "hello world", r"wor") is True
    assert case.assert_match("nomatch", "hello", r"^x") is False
    assert len(result.records) == 2


def test_modify_settings_existing_key_restored():
    original = settings.DATA_DRIVE
    with modify_settings(DATA_DRIVE=not original):
        assert settings.DATA_DRIVE == (not original)
    assert settings.DATA_DRIVE == original


def test_modify_settings_new_key_removed_after_exit():
    key = "QTAF_TEST_TEMP_KEY_XYZ"
    assert not hasattr(settings, key)
    with modify_settings(**{key: 123}):
        assert getattr(settings, key) == 123
    assert not hasattr(settings, key)


def test_modify_settings_restores_on_exception():
    original = settings.DATA_DRIVE
    with pytest.raises(ValueError):
        with modify_settings(DATA_DRIVE=not original):
            raise ValueError("boom")
    assert settings.DATA_DRIVE == original
