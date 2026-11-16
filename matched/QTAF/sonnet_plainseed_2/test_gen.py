# -*- coding: utf-8 -*-
import pytest

from testbase import datadrive
from testbase.conf import settings
from testbase.test import modify_settings
from testbase.testcase import TestCase, Environ


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


def _make_case():
    class _Case(TestCase):
        """  sample doc  
        """
        owner = "foo"
        priority = TestCase.EnumPriority.Normal
        status = TestCase.EnumStatus.Ready
        timeout = 1

        def run_test(self):
            pass
    return _Case


# ---------------- DataDrive ----------------

def test_datadrive_decorator_marks_class():
    cls = _make_case()
    assert not datadrive.is_datadrive(cls)
    dd = datadrive.DataDrive([1, 2, 3])
    result = dd(cls)
    assert result is cls
    assert datadrive.is_datadrive(cls)
    assert datadrive.get_datadrive(cls) is dd


def test_datadrive_rejects_non_testcase():
    class NotCase(object):
        pass

    with pytest.raises(TypeError):
        datadrive.DataDrive([1])(NotCase)


def test_datadrive_iter_list_yields_indexes():
    dd = datadrive.DataDrive(["a", "b", "c"])
    assert list(dd) == [0, 1, 2]
    assert dd[1] == "b"
    assert len(dd) == 3


def test_datadrive_iter_tuple_yields_indexes():
    dd = datadrive.DataDrive(("x", "y"))
    assert list(dd) == [0, 1]
    assert dd[0] == "x"


def test_datadrive_iter_dict_yields_keys():
    dd = datadrive.DataDrive({"A": 1, "B": 2})
    assert sorted(dd) == ["A", "B"]
    assert dd["B"] == 2
    assert len(dd) == 2


def test_datadrive_generator_is_materialized():
    dd = datadrive.DataDrive(i * 2 for i in range(4))
    assert list(dd) == [0, 1, 2, 3]
    assert dd[3] == 6
    assert len(dd) == 4
    # can iterate twice since generator converted to list
    assert list(dd) == [0, 1, 2, 3]


def test_load_datadrive_tests_from_list():
    cls = datadrive.DataDrive([10, 20, 30])(_make_case())
    tests = datadrive.load_datadrive_tests(cls)
    assert len(tests) == 3
    assert [t.casedata for t in tests] == [10, 20, 30]
    assert [t.casedataname for t in tests] == ["0", "1", "2"]
    assert all(isinstance(t, cls) for t in tests)


def test_load_datadrive_tests_from_dict():
    cls = datadrive.DataDrive({"TEST1": 1, "TEST2": 2})(_make_case())
    tests = datadrive.load_datadrive_tests(cls)
    by_name = dict((t.casedataname, t.casedata) for t in tests)
    assert by_name == {"TEST1": 1, "TEST2": 2}


def test_load_datadrive_tests_by_name_dict():
    cls = datadrive.DataDrive({"TEST1": 1, "TEST2": 2})(_make_case())
    tests = datadrive.load_datadrive_tests(cls, "TEST2")
    assert len(tests) == 1
    assert tests[0].casedata == 2
    assert tests[0].casedataname == "TEST2"


def test_load_datadrive_tests_by_name_for_list_index_string():
    cls = datadrive.DataDrive(["a", "b", "c"])(_make_case())
    tests = datadrive.load_datadrive_tests(cls, "1")
    assert len(tests) == 1
    assert tests[0].casedata == "b"
    assert tests[0].casedataname == "1"


def test_load_datadrive_tests_name_not_found():
    cls = datadrive.DataDrive({"A": 1})(_make_case())
    with pytest.raises(ValueError):
        datadrive.load_datadrive_tests(cls, "NOPE")


def test_load_datadrive_tests_non_datadrive_without_setting():
    cls = _make_case()
    with modify_settings(DATA_DRIVE=False):
        with pytest.raises(RuntimeError):
            datadrive.load_datadrive_tests(cls)


def test_load_datadrive_tests_applies_attrs():
    data = {"CASE": {"__attrs__": {"owner": "bar", "priority": "High",
                                   "unknown_attr": "ignored"}}}
    cls = datadrive.DataDrive(data)(_make_case())
    tests = datadrive.load_datadrive_tests(cls)
    assert len(tests) == 1
    t = tests[0]
    assert t.owner == "bar"
    assert t.priority == "High"
    assert not hasattr(t, "unknown_attr")
    # class attribute untouched
    assert cls.owner == "foo"


# ---------------- TestCase ----------------

def test_testcase_names_and_casedata():
    cls = _make_case()
    plain = cls()
    assert plain.casedata is None
    assert plain.casedataname is None
    expected_cls_name = cls.__module__ + "." + cls.__name__
    assert plain.test_class_name == expected_cls_name
    assert plain.test_name == expected_cls_name

    driven = cls([1, 2], "name1")
    assert driven.casedata == [1, 2]
    assert driven.casedataname == "name1"
    assert driven.test_name == expected_cls_name + "/name1"


def test_testcase_doc_is_stripped_and_overridable():
    cls = _make_case()
    assert cls().test_doc == "sample doc"
    t = cls(attrs={"__doc__": "custom doc"})
    assert t.test_doc == "custom doc"


def test_testcase_forbids_init_override():
    with pytest.raises(RuntimeError):
        class Bad(TestCase):
            def __init__(self):
                pass


def test_testcase_tags_metaclass_and_attrs():
    class Base(TestCase):
        tags = "base"

    class Child(Base):
        tags = ["child", "x"]

    assert Base.tags == {"base"}
    assert Child.tags == {"base", "child", "x"}
    assert Child.base_tags == {"base"}

    t = Child(attrs={"tags": "extra"})
    assert t.tags == {"base", "extra"}
    # class-level tags remain unchanged
    assert Child.tags == {"base", "child", "x"}

    class NoTags(TestCase):
        pass

    assert NoTags.tags == set()


def test_testcase_environ_is_singleton_and_extra_info():
    class Extra(TestCase):
        test_extra_info_def = [("alpha", "A"), ("beta", "B")]
        alpha = 5

    t = Extra()
    assert t.environ is Environ()
    assert t.test_extra_info == {"alpha": 5, "beta": None}


def test_testcase_run_test_not_implemented():
    class NoRun(TestCase):
        pass

    with pytest.raises(NotImplementedError):
        NoRun().run_test()


def test_testcase_assert_equal_and_match_with_result():
    t = _make_case()()
    res = FakeResult()
    t.init_test(res)
    assert t.test_result is res

    assert t.assert_equal("same", 1, 1) is True
    assert res.records == []
    assert t.assert_equal("diff", 1, 2) is False
    assert len(res.records) == 1
    assert "diff" in res.records[0][1]

    assert t.assert_match("match", "hello world", r"wor") is True
    assert t.assert_match("nomatch", "hello", r"^z") is False
    assert len(res.records) == 2


def test_testcase_log_info_fail_and_step():
    t = _make_case()()
    res = FakeResult()
    t.init_test(res)
    t.log_info(123)
    t.fail(456)
    t.start_step("step1")
    assert res.infos == ["123"]
    assert res.errors == ["456"]
    assert res.steps == ["step1"]


# ---------------- modify_settings ----------------

def test_modify_settings_overrides_and_restores_existing_key():
    original = settings.DATA_DRIVE
    with modify_settings(DATA_DRIVE=not original):
        assert settings.DATA_DRIVE == (not original)
    assert settings.DATA_DRIVE == original


def test_modify_settings_adds_and_removes_new_key():
    key = "QTAF_UNITTEST_TEMP_KEY"
    assert getattr(settings, key, None) is None
    with modify_settings(**{key: "value"}):
        assert getattr(settings, key) == "value"
    assert getattr(settings, key, None) is None


def test_modify_settings_restores_on_exception():
    original = settings.DATA_DRIVE
    with pytest.raises(ValueError):
        with modify_settings(DATA_DRIVE=not original):
            raise ValueError("boom")
    assert settings.DATA_DRIVE == original
