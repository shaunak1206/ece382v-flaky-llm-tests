import sys
import types

import pytest

from testbase import datadrive
from testbase import test as testmod
from testbase import testcase as tcmod
from testbase.conf import settings


class DummyResult:
    def __init__(self):
        self.calls = []
        self.records = []

    def begin_step(self, stepinfo):
        self.calls.append(("begin_step", stepinfo))

    def info(self, info):
        self.calls.append(("info", info))

    def error(self, message):
        self.calls.append(("error", message))

    def log_record(self, level, msg, extra=None):
        self.records.append((level, msg, extra))

    def begin_test(self, testcase):
        self.calls.append(("begin_test", testcase.test_name))

    def end_test(self):
        self.calls.append(("end_test", None))

    def warning(self, message):
        self.calls.append(("warning", message))

    @property
    def passed(self):
        return not any(c[0] == "error" for c in self.calls)


class DummyFactory:
    def __init__(self):
        self.created = []

    def create(self, testcase):
        self.created.append(testcase)
        return DummyResult()


class FakeRetry:
    def __init__(self, timeout=0, interval=0, raise_error=False):
        self.timeout = timeout
        self.interval = interval
        self.raise_error = raise_error

    def __iter__(self):
        for _ in range(2):
            yield object()


def make_testcase_class(name="MyCase", module_name="__main__", base=tcmod.TestCase):
    attrs = {
        "__module__": module_name,
        "__doc__": "  hello doc  ",
        "owner": "foo",
        "priority": base.EnumPriority.High,
        "status": base.EnumStatus.Ready,
        "timeout": 1,
    }
    return type(name, (base,), attrs)


def test_modify_settings_adds_and_removes_new_setting():
    assert not hasattr(settings, "QTAF_TEST_TEMP_SETTING")
    with testmod.modify_settings(QTAF_TEST_TEMP_SETTING="x"):
        assert settings.QTAF_TEST_TEMP_SETTING == "x"
        assert "QTAF_TEST_TEMP_SETTING" in settings._Settings__keys
    assert not hasattr(settings, "QTAF_TEST_TEMP_SETTING")
    assert "QTAF_TEST_TEMP_SETTING" not in settings._Settings__keys


def test_modify_settings_restores_existing_setting(monkeypatch):
    monkeypatch.setattr(settings, "QTAF_ASSERT_CONTINUE", True, raising=False)
    original = settings.QTAF_ASSERT_CONTINUE
    with testmod.modify_settings(QTAF_ASSERT_CONTINUE=not original):
        assert settings.QTAF_ASSERT_CONTINUE is (not original)
    assert settings.QTAF_ASSERT_CONTINUE is original


def test_modify_settings_restores_sealed_flag():
    settings._Settings__sealed = True
    with testmod.modify_settings(QTAF_TEMP_X=123):
        assert settings._Settings__sealed is False
    assert settings._Settings__sealed is True
    if hasattr(settings, "QTAF_TEMP_X"):
        delattr(settings, "QTAF_TEMP_X")
        settings._Settings__keys.discard("QTAF_TEMP_X")


def test_datadrive_decorator_marks_testcase_class():
    @datadrive.DataDrive([1, 2, 3])
    class MyCase(tcmod.TestCase):
        owner = "foo"
        priority = tcmod.TestCase.EnumPriority.High
        status = tcmod.TestCase.EnumStatus.Ready
        timeout = 1

        def run_test(self):
            pass

    assert datadrive.is_datadrive(MyCase)
    assert isinstance(datadrive.get_datadrive(MyCase), datadrive.DataDrive)


def test_datadrive_decorator_rejects_non_testcase():
    with pytest.raises(TypeError):
        datadrive.DataDrive([1])(object)


def test_datadrive_iterates_lists_as_indexes_and_len():
    dd = datadrive.DataDrive(["a", "b", "c"])
    assert list(dd) == [0, 1, 2]
    assert len(dd) == 3
    assert dd[1] == "b"


def test_datadrive_iterates_dict_keys_and_getitem():
    dd = datadrive.DataDrive({"k1": 10, "k2": 20})
    assert list(dd) == ["k1", "k2"]
    assert len(dd) == 2
    assert dd["k2"] == 20


def test_load_datadrive_tests_from_decorated_class_with_attrs(monkeypatch):
    warnings = []
    monkeypatch.setattr(tcmod.logger, "warn", lambda msg: warnings.append(msg))

    @datadrive.DataDrive({"bad-key": {"value": 1, "__attrs__": {"owner": "bar", "tags": ["x"], "__doc__": "doc2"}}})
    class MyCase(tcmod.TestCase):
        owner = "foo"
        priority = tcmod.TestCase.EnumPriority.High
        status = tcmod.TestCase.EnumStatus.Ready
        timeout = 1
        tags = {"base"}

        def run_test(self):
            pass

    tests = datadrive.load_datadrive_tests(MyCase)
    assert len(tests) == 1
    t = tests[0]
    assert t.casedata == {"value": 1, "__attrs__": {"owner": "bar", "tags": ["x"], "__doc__": "doc2"}}
    assert t.casedataname == "bad_key"
    assert t.owner == "bar"
    assert t.test_doc == "doc2"
    assert t.tags == {"base", "x"}
    assert warnings and "bad_key" in warnings[0]


def test_load_datadrive_tests_by_name_translation(monkeypatch):
    @datadrive.DataDrive({"bad-key": 7, "good": 8})
    class MyCase(tcmod.TestCase):
        owner = "foo"
        priority = tcmod.TestCase.EnumPriority.High
        status = tcmod.TestCase.EnumStatus.Ready
        timeout = 1

        def run_test(self):
            pass

    tests = datadrive.load_datadrive_tests(MyCase, "bad_key")
    assert len(tests) == 1
    assert tests[0].casedata == 7
    assert tests[0].casedataname == "bad_key"


def test_load_datadrive_tests_requires_flag_when_not_decorated(monkeypatch):
    monkeypatch.setattr(settings, "DATA_DRIVE", False, raising=False)

    class PlainCase(tcmod.TestCase):
        owner = "foo"
        priority = tcmod.TestCase.EnumPriority.High
        status = tcmod.TestCase.EnumStatus.Ready
        timeout = 1
        __doc__ = "doc"

        def run_test(self):
            pass

    with pytest.raises(RuntimeError):
        datadrive.load_datadrive_tests(PlainCase)


def test_testcase_properties_and_attr_overrides():
    cls = make_testcase_class()
    obj = cls(testdata={"x": 1}, testdataname="case1", attrs={"owner": "bar", "tags": "t1", "__doc__": "override"})
    assert obj.casedata == {"x": 1}
    assert obj.casedataname == "case1"
    assert obj.test_class_name == "__main__.MyCase"
    assert obj.test_name == "__main__.MyCase/case1"
    assert obj.test_doc == "override"
    assert obj.owner == "bar"
    assert obj.tags == {"t1"}


def test_testcase_test_doc_falls_back_to_class_doc_and_strips():
    cls = make_testcase_class(name="DocCase")
    obj = cls()
    assert obj.test_doc == "hello doc"


def test_testcase_start_log_fail_and_assertions_record_to_result(monkeypatch):
    cls = make_testcase_class(name="AssertCase")
    obj = cls()
    result = DummyResult()
    obj.init_test(result)

    obj.start_step("step1")
    obj.log_info("info1")
    obj.fail("boom")
    assert obj.assert_equal("eq", 1, 2) is False
    assert obj.assert_match("match", "abc", r"z+") is False

    assert ("begin_step", "step1") in result.calls
    assert ("info", "info1") in result.calls
    assert ("error", "boom") in result.calls
    assert len(result.records) >= 2


def test_assert_raises_when_continue_disabled(monkeypatch):
    monkeypatch.setattr(settings, "QTAF_ASSERT_CONTINUE", False, raising=False)
    cls = make_testcase_class(name="AssertStopCase")
    obj = cls()
    obj.init_test(DummyResult())
    with pytest.raises(RuntimeError):
        obj.assert_("msg", False)


def test_wait_for_equal_and_match_with_fake_retry(monkeypatch):
    monkeypatch.setattr(tcmod, "Retry", FakeRetry)

    class Obj:
        value = 3
        text = "HelloWorld"

    cls = make_testcase_class(name="WaitCase")
    obj = cls()
    result = DummyResult()
    obj.init_test(result)

    assert obj.wait_for_equal("eq", Obj(), "value", 3, timeout=0, interval=0) is True
    assert obj.wait_for_match("mt", Obj(), "text", r"hello", timeout=0, interval=0) is True

    o2 = Obj()
    o2.value = 4
    assert obj.wait_for_equal("eq2", o2, "value", 3, timeout=0, interval=0) is False
    assert obj.wait_for_match("mt2", o2, "text", r"nomatch", timeout=0, interval=0) is False


def test_testcase_runner_single_methods_mapping_warns_on_old_names():
    class LegacyCase(tcmod.TestCase):
        owner = "foo"
        priority = tcmod.TestCase.EnumPriority.High
        status = tcmod.TestCase.EnumStatus.Ready
        timeout = 1
        __doc__ = "doc"

        def runTest(self):
            return "old"

    runner = tcmod.TestCaseRunner()
    messages = []
    runner._testresult = DummyResult()
    runner._testresult.warning = lambda msg: messages.append(msg)
    runner._single_methods_mapping(LegacyCase)
    assert hasattr(LegacyCase, "run_test")
    assert LegacyCase.run_test == LegacyCase.runTest
    assert messages and "已被废弃" in messages[0]
