import os
import types

import pytest

from testbase import datadrive
from testbase import test as testmod
from testbase import testcase as tc


class DummySettings:
    def __init__(self):
        self._Settings__sealed = True
        self._Settings__keys = set()
        self._loaded = 0
        self.existing = "old"

    def _Settings__ensure_loaded(self):
        self._loaded += 1


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

    def begin_test(self, testcase):
        self.calls.append(("begin_test", testcase))

    def end_test(self):
        self.calls.append(("end_test",))

    def warning(self, msg):
        self.calls.append(("warning", msg))

    def exception(self, msg):
        self.calls.append(("exception", msg))

    @property
    def passed(self):
        return not any(c[0] in ("error", "exception") for c in self.calls)


class DummyFactory:
    def __init__(self):
        self.results = []

    def create(self, testcase):
        r = DummyResult()
        self.results.append((testcase, r))
        return r


def test_modify_settings_sets_and_restores_existing_and_new_keys(monkeypatch):
    dummy = DummySettings()
    monkeypatch.setattr(testmod, "settings", dummy, raising=False)

    assert dummy.existing == "old"
    assert not hasattr(dummy, "new_key")

    with testmod.modify_settings(existing="new", new_key=123):
        assert dummy._Settings__sealed is False
        assert dummy.existing == "new"
        assert dummy.new_key == 123
        assert "new_key" in dummy._Settings__keys
        assert dummy._loaded == 1

    assert dummy._Settings__sealed is True
    assert dummy.existing == "old"
    assert not hasattr(dummy, "new_key")
    assert "new_key" not in dummy._Settings__keys


def test_modify_settings_restores_state_when_exception_raised(monkeypatch):
    dummy = DummySettings()
    monkeypatch.setattr(testmod, "settings", dummy, raising=False)

    with pytest.raises(RuntimeError):
        with testmod.modify_settings(existing="x", temp="y"):
            raise RuntimeError("boom")

    assert dummy.existing == "old"
    assert not hasattr(dummy, "temp")
    assert dummy._Settings__sealed is True


def test_modify_environ_sets_and_restores_environment(monkeypatch):
    monkeypatch.setenv("KEEP_ME", "1")
    monkeypatch.delenv("NEW_ENV", raising=False)

    with testmod.modify_environ(KEEP_ME="2", NEW_ENV="abc"):
        assert os.environ["KEEP_ME"] == "2"
        assert os.environ["NEW_ENV"] == "abc"

    assert os.environ["KEEP_ME"] == "1"
    assert "NEW_ENV" not in os.environ


def test_modify_attributes_sets_and_restores_attributes():
    class Obj:
        a = 1
        b = 2

    with testmod.modify_attributes(Obj, {"a": 10, "b": 20}):
        assert Obj.a == 10
        assert Obj.b == 20

    assert Obj.a == 1
    assert Obj.b == 2


def test_datadrive_rejects_non_testcase_class():
    with pytest.raises(TypeError):
        datadrive.DataDrive([1])(object)


def test_datadrive_marks_testcase_class():
    @datadrive.DataDrive([1, 2])
    class Demo(tc.TestCase):
        pass

    assert datadrive.is_datadrive(Demo)
    assert isinstance(datadrive.get_datadrive(Demo), datadrive.DataDrive)


def test_datadrive_iterates_list_indexes():
    dd = datadrive.DataDrive(["a", "b", "c"])
    assert list(dd) == [0, 1, 2]
    assert len(dd) == 3
    assert dd[1] == "b"


def test_datadrive_iterates_dict_keys():
    dd = datadrive.DataDrive({"k1": 1, "k2": 2})
    assert list(dd) == ["k1", "k2"]
    assert len(dd) == 2
    assert dd["k2"] == 2


def test_datadrive_consumes_generator_and_caches_values():
    def gen():
        yield "x"
        yield "y"

    dd = datadrive.DataDrive(gen())
    assert list(dd) == [0, 1]
    assert len(dd) == 2
    assert dd[0] == "x"
    assert dd[1] == "y"
    assert list(dd) == [0, 1]


def test_get_translated_in_datadrive_matches_translated_key(monkeypatch):
    dd = {"a_b": 1, "c": 2}
    monkeypatch.setattr(datadrive, "smart_text", lambda x: x, raising=False)
    monkeypatch.setattr(datadrive, "translate_bad_char", lambda x: x.replace("-", "_"), raising=False)

    assert datadrive._get_translated_in_datadrive("a-b", dd) == 1


def test_get_translated_in_datadrive_raises_for_missing_key(monkeypatch):
    dd = {"a": 1}
    monkeypatch.setattr(datadrive, "smart_text", lambda x: x, raising=False)
    monkeypatch.setattr(datadrive, "translate_bad_char", lambda x: x, raising=False)

    with pytest.raises(ValueError):
        datadrive._get_translated_in_datadrive("missing", dd)


def test_load_datadrive_tests_creates_test_instances_and_applies_attrs(monkeypatch):
    monkeypatch.setattr(datadrive, "has_bad_char", lambda s: False, raising=False)
    monkeypatch.setattr(datadrive, "translate_bad_char", lambda s: s, raising=False)
    monkeypatch.setattr(datadrive.logger, "warn", lambda msg: None, raising=False)

    @datadrive.DataDrive(
        {
            "case1": {"x": 1, "__attrs__": {"owner": "alice", "timeout": 7, "__doc__": "custom doc"}},
            "case2": {"x": 2},
        }
    )
    class Demo(tc.TestCase):
        """base doc"""

        owner = "base_owner"
        priority = tc.TestCase.EnumPriority.Normal
        status = tc.TestCase.EnumStatus.Ready
        timeout = 1

    tests = datadrive.load_datadrive_tests(Demo)
    assert [t.casedataname for t in tests] == ["case1", "case2"]
    assert tests[0].casedata == {"x": 1, "__attrs__": {"owner": "alice", "timeout": 7, "__doc__": "custom doc"}}
    assert tests[0].owner == "alice"
    assert tests[0].timeout == 7
    assert tests[0].test_doc == "custom doc"
    assert tests[1].owner == "base_owner"


def test_load_datadrive_tests_warns_for_bad_key(monkeypatch):
    monkeypatch.setattr(datadrive, "has_bad_char", lambda s: True, raising=False)
    monkeypatch.setattr(datadrive, "translate_bad_char", lambda s: s.replace(" ", "_"), raising=False)

    warned = []
    monkeypatch.setattr(datadrive.logger, "warn", lambda msg: warned.append(msg), raising=False)

    @datadrive.DataDrive({"bad key": 1})
    class Demo(tc.TestCase):
        """doc"""

        owner = "o"
        priority = tc.TestCase.EnumPriority.Normal
        status = tc.TestCase.EnumStatus.Ready
        timeout = 1

    tests = datadrive.load_datadrive_tests(Demo)
    assert tests[0].casedataname == "bad_key"
    assert warned and "should use" in warned[0]


def test_load_datadrive_tests_can_select_single_key(monkeypatch):
    monkeypatch.setattr(datadrive, "has_bad_char", lambda s: False, raising=False)
    monkeypatch.setattr(datadrive.logger, "warn", lambda msg: None, raising=False)

    @datadrive.DataDrive({"A": 1, "B": 2})
    class Demo(tc.TestCase):
        """doc"""

        owner = "o"
        priority = tc.TestCase.EnumPriority.Normal
        status = tc.TestCase.EnumStatus.Ready
        timeout = 1

    tests = datadrive.load_datadrive_tests(Demo, "B")
    assert len(tests) == 1
    assert tests[0].casedataname == "B"
    assert tests[0].casedata == 2


def test_testcase_properties_and_extra_info():
    class Demo(tc.TestCase):
        """  hello doc  """

        owner = "o"
        priority = tc.TestCase.EnumPriority.High
        status = tc.TestCase.EnumStatus.Ready
        timeout = 3
        test_extra_info_def = [("owner", None), ("timeout", None)]

    case = Demo(testdata={"a": 1}, testdataname="name")
    assert case.casedata == {"a": 1}
    assert case.casedataname == "name"
    assert case.test_class_name.endswith(".Demo")
    assert case.test_name.endswith("/name")
    assert case.test_doc == "hello doc"
    assert case.test_extra_info == {"owner": "o", "timeout": 3}


def test_testcase_init_overrides_allowed_attrs():
    class Demo(tc.TestCase):
        """doc"""

        owner = "base"
        priority = tc.TestCase.EnumPriority.Normal
        status = tc.TestCase.EnumStatus.Ready
        timeout = 1
        tags = {"base"}

    case = Demo(attrs={"owner": "alice", "timeout": 9, "tags": ["x", "y"], "__doc__": "new doc", "nope": 1})
    assert case.owner == "alice"
    assert case.timeout == 9
    assert case.tags == {"base", "x", "y"}
    assert case.test_doc == "new doc"


def test_testcase_assert_equal_and_match_log_failures(monkeypatch):
    class Demo(tc.TestCase):
        """doc"""

        owner = "o"
        priority = tc.TestCase.EnumPriority.Normal
        status = tc.TestCase.EnumStatus.Ready
        timeout = 1

    case = Demo()
    result = DummyResult()
    case.init_test(result)

    assert case.assert_equal("msg", 1, 1) is True
    assert case.assert_equal("msg2", 1, 2) is False
    assert case.assert_match("msg3", "abc", r"^a") is True
    assert case.assert_match("msg4", "abc", r"^z") is False
    assert any(c[0] == "log_record" and c[1] == "ASSERT" for c in result.calls)


def test_testcase_start_log_fail_and_fail_convert_to_text():
    class Demo(tc.TestCase):
        """doc"""

        owner = "o"
        priority = tc.TestCase.EnumPriority.Normal
        status = tc.TestCase.EnumStatus.Ready
        timeout = 1

    case = Demo()
    result = DummyResult()
    case.init_test(result)

    case.start_step(123)
    case.log_info(456)
    case.fail(789)

    assert ("begin_step", "123") in result.calls
    assert ("info", "456") in result.calls
    assert ("error", "789") in result.calls
