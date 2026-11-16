import sys

import pytest

from testbase import datadrive
from testbase.conf import settings
from testbase.datadrive import DataDrive, is_datadrive, get_datadrive, load_datadrive_tests
from testbase.test import modify_settings
from testbase.testcase import TestCase


class FakeResult(object):
    def __init__(self):
        self.records = []
        self.infos = []
        self.errors = []
        self.steps = []

    def log_record(self, level, msg, *args, **kwargs):
        self.records.append((level, msg))

    def info(self, msg):
        self.infos.append(msg)

    def error(self, msg):
        self.errors.append(msg)

    def begin_step(self, msg):
        self.steps.append(msg)


def _make_case(**attrs):
    class _Case(TestCase):
        """  sample doc  
        """
        owner = "foo"
        timeout = 1
        status = TestCase.EnumStatus.Ready
        priority = TestCase.EnumPriority.Normal

        def run_test(self):
            pass

    for k, v in attrs.items():
        setattr(_Case, k, v)
    return _Case


# ---------------- DataDrive ----------------

def test_datadrive_decorator_marks_class():
    cls = _make_case()
    assert not is_datadrive(cls)
    dd = DataDrive([1, 2, 3])
    decorated = dd(cls)
    assert decorated is cls
    assert is_datadrive(cls)
    assert get_datadrive(cls) is dd


def test_datadrive_rejects_non_testcase():
    class NotACase(object):
        pass

    with pytest.raises(TypeError):
        DataDrive([1])(NotACase)


def test_datadrive_list_iterates_indices():
    dd = DataDrive(["a", "b", "c"])
    assert list(dd) == [0, 1, 2]
    assert dd[1] == "b"
    assert len(dd) == 3


def test_datadrive_tuple_iterates_indices():
    dd = DataDrive(("x", "y"))
    assert list(dd) == [0, 1]
    assert dd[0] == "x"


def test_datadrive_dict_iterates_keys():
    dd = DataDrive({"K1": 1, "K2": 2})
    assert sorted(dd) == ["K1", "K2"]
    assert dd["K2"] == 2
    assert len(dd) == 2


def test_datadrive_generator_is_materialized():
    dd = DataDrive(x * 2 for x in [5, 6])
    assert list(dd) == [0, 1]
    assert dd[0] == 10
    assert dd[1] == 12
    assert len(dd) == 2


def test_is_datadrive_on_instance():
    cls = DataDrive([1])(_make_case())
    assert is_datadrive(cls(1, "0"))


# ---------------- load_datadrive_tests ----------------

def test_load_list_datadrive_creates_tests():
    cls = DataDrive(["a", "b"])(_make_case())
    tests = load_datadrive_tests(cls)
    assert len(tests) == 2
    assert [t.casedata for t in tests] == ["a", "b"]
    assert [t.casedataname for t in tests] == ["0", "1"]
    assert all(isinstance(t, cls) for t in tests)


def test_load_dict_datadrive_uses_keys_as_names():
    cls = DataDrive({"ONE": 1, "TWO": 2})(_make_case())
    tests = load_datadrive_tests(cls)
    result = {t.casedataname: t.casedata for t in tests}
    assert result == {"ONE": 1, "TWO": 2}


def test_load_with_specific_name():
    cls = DataDrive({"ONE": 1, "TWO": 2})(_make_case())
    tests = load_datadrive_tests(cls, "TWO")
    assert len(tests) == 1
    assert tests[0].casedata == 2
    assert tests[0].casedataname == "TWO"


def test_load_with_string_index_name_for_list():
    cls = DataDrive(["a", "b", "c"])(_make_case())
    tests = load_datadrive_tests(cls, "1")
    assert len(tests) == 1
    assert tests[0].casedata == "b"


def test_load_with_unknown_name_raises_value_error():
    cls = DataDrive({"ONE": 1})(_make_case())
    with pytest.raises(ValueError):
        load_datadrive_tests(cls, "MISSING")


def test_load_applies_attrs_override():
    data = {"CASE": {"__attrs__": {"owner": "bar", "priority": "High"}, "v": 1}}
    cls = DataDrive(data)(_make_case())
    tests = load_datadrive_tests(cls)
    assert len(tests) == 1
    assert tests[0].owner == "bar"
    assert tests[0].priority == "High"
    assert cls.owner == "foo"


def test_load_non_datadrive_without_setting_raises():
    cls = _make_case()
    with modify_settings(DATA_DRIVE=False):
        with pytest.raises(RuntimeError):
            load_datadrive_tests(cls)


# ---------------- TestCase ----------------

def test_testcase_forbids_overloading_init():
    with pytest.raises(RuntimeError):
        class Bad(TestCase):
            def __init__(self):
                pass


def test_testcase_tags_inherit_from_base():
    class A(TestCase):
        tags = ["x"]

    class B(A):
        tags = "y"

    assert A.tags == {"x"}
    assert B.tags == {"x", "y"}
    assert B.base_tags == {"x"}


def test_testcase_without_tags_has_empty_set():
    cls = _make_case()
    assert cls.tags == set()
    assert cls.base_tags == set()


def test_testcase_attrs_whitelist_overrides():
    cls = _make_case()
    tc = cls(None, None, {"owner": "bar", "tags": "t1", "__doc__": "custom doc", "unknown": 1})
    assert tc.owner == "bar"
    assert tc.tags == {"t1"}
    assert tc.test_doc == "custom doc"
    assert not hasattr(tc, "unknown")


def test_testcase_names_and_data():
    cls = _make_case()
    prefix = cls.__module__ + "." + cls.__name__
    tc = cls({"k": 1}, "name1")
    assert tc.casedata == {"k": 1}
    assert tc.casedataname == "name1"
    assert tc.test_class_name == prefix
    assert tc.test_name == prefix + "/name1"
    assert cls().test_name == prefix


def test_testcase_doc_is_stripped():
    cls = _make_case()
    assert cls().test_doc == "sample doc"


def test_testcase_extra_info():
    cls = _make_case(test_extra_info_def=[("foo_attr", "desc")], foo_attr=42)
    assert cls().test_extra_info == {"foo_attr": 42}


def test_testcase_base_run_test_not_implemented():
    with pytest.raises(NotImplementedError):
        TestCase().run_test()


def test_testcase_logging_helpers_use_result():
    tc = _make_case()()
    res = FakeResult()
    tc.init_test(res)
    assert tc.test_result is res
    tc.log_info(123)
    tc.fail("boom")
    tc.start_step("step1")
    assert res.infos == ["123"]
    assert res.errors == ["boom"]
    assert res.steps == ["step1"]


def test_testcase_assert_equal_and_match():
    tc = _make_case()()
    res = FakeResult()
    tc.init_test(res)
    assert tc.assert_equal("same", 1, 1) is True
    assert tc.assert_match("match", "hello world", "wor") is True
    assert res.records == []
    assert tc.assert_equal("diff", 1, 2) is False
    assert tc.assert_match("nomatch", "hello", "xyz") is False
    assert len(res.records) == 2


# ---------------- modify_settings ----------------

def test_modify_settings_existing_key_restored():
    orig = settings.DATA_DRIVE
    with modify_settings(DATA_DRIVE=not orig):
        assert settings.DATA_DRIVE == (not orig)
    assert settings.DATA_DRIVE == orig


def test_modify_settings_restores_on_exception():
    orig = settings.DATA_DRIVE
    with pytest.raises(KeyError):
        with modify_settings(DATA_DRIVE=not orig):
            raise KeyError("x")
    assert settings.DATA_DRIVE == orig


def test_modify_settings_new_key_removed_afterwards():
    key = "QTAF_TEST_TEMP_KEY_XYZ"
    assert getattr(settings, key, None) is None
    with modify_settings(**{key: "value"}):
        assert getattr(settings, key) == "value"
    assert getattr(settings, key, None) is None
