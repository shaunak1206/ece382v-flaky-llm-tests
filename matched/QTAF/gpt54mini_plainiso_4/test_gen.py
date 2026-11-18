import sys
import types

import pytest

from testbase.datadrive import (
    DataDrive,
    get_datadrive,
    is_datadrive,
    load_datadrive_tests,
)
from testbase.test import modify_settings
from testbase.testcase import TestCase
from testbase.conf import settings


class DummyResult:
    def __init__(self):
        self.calls = []

    def begin_step(self, msg):
        self.calls.append(("begin_step", msg))

    def info(self, msg):
        self.calls.append(("info", msg))

    def error(self, msg):
        self.calls.append(("error", msg))

    def log_record(self, level, msg, extra=None):
        self.calls.append(("log_record", level, msg, extra))


class DummyCase(TestCase):
    """dummy doc"""

    owner = "owner"
    priority = TestCase.EnumPriority.Normal
    status = TestCase.EnumStatus.Ready
    timeout = 1

    def run_test(self):
        return "ok"


def test_datadrive_decorator_sets_marker_and_helpers():
    dd = DataDrive([1, 2])

    @dd
    class TC(DummyCase):
        pass

    assert is_datadrive(TC) is True
    assert get_datadrive(TC) is dd
    assert TC.__qtaf_datadrive__ is dd


def test_datadrive_decorator_rejects_non_testcase():
    dd = DataDrive([1])
    with pytest.raises(TypeError):
        dd(object)


def test_datadrive_iterates_list_as_indices():
    dd = DataDrive(["a", "b", "c"])
    assert list(iter(dd)) == [0, 1, 2]
    assert len(dd) == 3
    assert dd[1] == "b"


def test_datadrive_iterates_tuple_as_indices():
    dd = DataDrive(("a", "b"))
    assert list(iter(dd)) == [0, 1]
    assert len(dd) == 2
    assert dd[0] == "a"


def test_datadrive_iterates_dict_as_keys():
    dd = DataDrive({"x": 10, "y": 20})
    assert list(iter(dd)) == ["x", "y"]
    assert len(dd) == 2
    assert dd["y"] == 20


def test_datadrive_generator_is_materialized_once():
    dd = DataDrive((i for i in range(3)))
    assert list(iter(dd)) == [0, 1, 2]
    assert len(dd) == 3
    assert list(iter(dd)) == [0, 1, 2]


def test_load_datadrive_tests_with_named_key_and_attrs(monkeypatch):
    warned = []

    class Logger:
        def warn(self, msg):
            warned.append(msg)

    monkeypatch.setattr("testbase.datadrive.logger", Logger())

    class TC(DummyCase):
        pass

    dd = DataDrive(
        {
            "CASE1": {"v": 1, "__attrs__": {"owner": "new_owner", "tags": ["t1"], "__doc__": "custom doc"}}
        }
    )
    TC.__qtaf_datadrive__ = dd

    tests = load_datadrive_tests(TC)
    assert len(tests) == 1
    t = tests[0]
    assert isinstance(t, TC)
    assert t.casedata == {"v": 1, "__attrs__": {"owner": "new_owner", "tags": ["t1"], "__doc__": "custom doc"}}
    assert t.casedataname == "CASE1"
    assert t.owner == "new_owner"
    assert t.test_doc == "custom doc"
    assert "t1" in t.tags
    assert not warned


def test_load_datadrive_tests_warns_for_bad_key(monkeypatch):
    warned = []

    class Logger:
        def warn(self, msg):
            warned.append(msg)

    monkeypatch.setattr("testbase.datadrive.logger", Logger())

    class TC(DummyCase):
        pass

    dd = DataDrive({"bad key": 1})
    TC.__qtaf_datadrive__ = dd

    tests = load_datadrive_tests(TC)
    assert len(tests) == 1
    assert tests[0].casedataname != "bad key"
    assert warned
    assert "bad key" in warned[0]


def test_load_datadrive_tests_name_lookup_by_translated_key(monkeypatch):
    class TC(DummyCase):
        pass

    dd = DataDrive({"abc_def": 1})
    TC.__qtaf_datadrive__ = dd

    # name is expected to match after translation of bad chars;
    # use a slightly different textual form that still should map through helper.
    tests = load_datadrive_tests(TC, name="abc_def")
    assert len(tests) == 1
    assert tests[0].casedata == 1
    assert tests[0].casedataname == "abc_def"


def test_load_datadrive_tests_requires_datadrive_when_setting_disabled(monkeypatch):
    class TC(DummyCase):
        pass

    if hasattr(settings, "DATA_DRIVE"):
        old = settings.DATA_DRIVE
    else:
        old = None
    with modify_settings(DATA_DRIVE=False):
        if hasattr(TC, "__qtaf_datadrive__"):
            delattr(TC, "__qtaf_datadrive__")
        with pytest.raises(RuntimeError):
            load_datadrive_tests(TC)
    if old is not None:
        assert settings.DATA_DRIVE == old


def test_testcase_class_name_and_name_properties():
    class TC(DummyCase):
        pass

    tc = TC()
    assert tc.test_class_name.endswith(".TC")
    assert tc.test_name == tc.test_class_name


def test_testcase_name_includes_casedataname():
    class TC(DummyCase):
        pass

    tc = TC(testdata={"x": 1}, testdataname="case1")
    assert tc.casedata == {"x": 1}
    assert tc.casedataname == "case1"
    assert tc.test_name.endswith("/case1")


def test_testcase_test_doc_uses_custom_doc_override():
    class TC(DummyCase):
        pass

    tc = TC(attrs={"__doc__": "hello"})
    assert tc.test_doc == "hello"


def test_testcase_test_extra_info_reads_defined_attributes():
    class TC(DummyCase):
        test_extra_info_def = [("x", ""), ("y", "")]

        def __init__(self):
            super().__init__()
            self.x = 10

    tc = TC()
    assert tc.test_extra_info == {"x": 10, "y": None}


def test_testcase_assert_equal_logs_failure_and_returns_false(monkeypatch):
    tc = DummyCase()
    res = DummyResult()
    tc.init_test(res)

    assert tc.assert_equal("msg", 1, 2) is False
    assert res.calls and res.calls[-1][0] == "log_record"
    assert "msg" in res.calls[-1][2]


def test_testcase_assert_match_success_and_failure(monkeypatch):
    tc = DummyCase()
    res = DummyResult()
    tc.init_test(res)

    assert tc.assert_match("ok", "hello123", r"\d+") is True
    assert tc.assert_match("bad", "hello", r"\d+") is False
    assert any(call[0] == "log_record" for call in res.calls)


def test_testcase_start_step_info_and_fail_forward_to_result():
    tc = DummyCase()
    res = DummyResult()
    tc.init_test(res)

    tc.start_step(123)
    tc.log_info(456)
    tc.fail(789)

    assert ("begin_step", "123") in res.calls
    assert ("info", "456") in res.calls
    assert ("error", "789") in res.calls


def test_modify_settings_temporarily_sets_and_restores_existing_and_new_keys():
    original_sealed = getattr(settings, "_Settings__sealed", True)
    original_keys = set(getattr(settings, "_Settings__keys", set()))
    original_value = getattr(settings, "SOME_EXISTING_KEY", None)
    had_existing = hasattr(settings, "SOME_EXISTING_KEY")

    if not hasattr(settings, "_Settings__ensure_loaded"):
        settings._Settings__ensure_loaded = lambda: None

    with modify_settings(SOME_EXISTING_KEY="new", SOME_NEW_KEY="value"):
        assert settings.SOME_EXISTING_KEY == "new"
        assert settings.SOME_NEW_KEY == "value"
        assert settings._Settings__sealed is False

    assert getattr(settings, "_Settings__sealed", True) is True
    if had_existing:
        assert settings.SOME_EXISTING_KEY == original_value
    else:
        assert not hasattr(settings, "SOME_EXISTING_KEY")
    assert not hasattr(settings, "SOME_NEW_KEY")
    assert set(getattr(settings, "_Settings__keys", set())) == original_keys


def test_modify_settings_sets_sealed_false_inside_context(monkeypatch):
    if not hasattr(settings, "_Settings__ensure_loaded"):
        settings._Settings__ensure_loaded = lambda: None

    with modify_settings(TEMP_KEY="x"):
        assert settings._Settings__sealed is False
        assert settings.TEMP_KEY == "x"
    assert settings._Settings__sealed is True
    assert not hasattr(settings, "TEMP_KEY")
