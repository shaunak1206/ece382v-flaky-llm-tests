# -*- coding: utf-8 -*-
"""Unit tests for testbase.datadrive, testbase.testcase and testbase.test.modify_settings."""
import pytest

from testbase import datadrive
from testbase import testcase as tc
from testbase.conf import settings
from testbase.test import modify_settings
from testbase.testresult import EnumLogLevel


_MISSING = object()


class _FakeResult(object):
    """Minimal stand-in for a TestResult, recording calls made by TestCase."""

    def __init__(self):
        self.records = []
        self.errors = []
        self.infos = []
        self.steps = []

    def log_record(self, level, msg, *args, **kwargs):
        self.records.append((level, msg))

    def error(self, msg):
        self.errors.append(msg)

    def info(self, msg):
        self.infos.append(msg)

    def begin_step(self, stepinfo):
        self.steps.append(stepinfo)


# ---------------------------------------------------------------------------
# DataDrive
# ---------------------------------------------------------------------------

def test_call_registers_datadrive_on_testcase_subclass():
    deco = datadrive.DataDrive([1, 2, 3])

    class SampleCase(tc.TestCase):
        def run_test(self):
            pass

    decorated = deco(SampleCase)

    assert decorated is SampleCase
    assert datadrive.is_datadrive(SampleCase)
    assert datadrive.get_datadrive(SampleCase) is deco


def test_call_rejects_non_testcase_class():
    class NotACase(object):
        pass

    with pytest.raises(TypeError):
        datadrive.DataDrive([1])(NotACase)


def test_iter_list_yields_indices_and_getitem_len_work():
    dd = datadrive.DataDrive(["a", "b", "c"])

    assert list(dd) == [0, 1, 2]
    assert len(dd) == 3
    assert dd[0] == "a"
    assert dd[2] == "c"


def test_iter_dict_yields_keys_and_getitem_by_key():
    dd = datadrive.DataDrive({"TEST1": 1, "TEST2": 2})

    assert sorted(dd) == ["TEST1", "TEST2"]
    assert len(dd) == 2
    assert dd["TEST2"] == 2


def test_iter_tuple_yields_indices():
    dd = datadrive.DataDrive(("x", "y"))

    assert list(dd) == [0, 1]
    assert dd[1] == "y"


def test_generator_data_is_materialized_and_iterable_twice():
    def gen():
        for i in range(3):
            yield i * 10

    dd = datadrive.DataDrive(gen())

    assert list(dd) == [0, 1, 2]
    # Second iteration must still work because the generator was converted to a list.
    assert list(dd) == [0, 1, 2]
    assert dd[2] == 20


def test_is_datadrive_false_for_undecorated_class():
    class Plain(tc.TestCase):
        def run_test(self):
            pass

    assert not datadrive.is_datadrive(Plain)


# ---------------------------------------------------------------------------
# load_datadrive_tests
# ---------------------------------------------------------------------------

def test_load_list_creates_instances_named_by_index():
    @datadrive.DataDrive([10, 20])
    class ListCase(tc.TestCase):
        def run_test(self):
            pass

    tests = datadrive.load_datadrive_tests(ListCase)

    assert len(tests) == 2
    assert all(isinstance(t, ListCase) for t in tests)
    assert [t.casedataname for t in tests] == ["0", "1"]
    assert [t.casedata for t in tests] == [10, 20]


def test_load_dict_uses_keys_as_names_and_values_as_casedata():
    @datadrive.DataDrive({"TEST1": "first", "TEST2": "second"})
    class DictCase(tc.TestCase):
        def run_test(self):
            pass

    tests = datadrive.load_datadrive_tests(DictCase)
    mapping = dict((t.casedataname, t.casedata) for t in tests)

    assert mapping == {"TEST1": "first", "TEST2": "second"}


def test_load_with_name_selects_single_entry():
    @datadrive.DataDrive({"ALPHA": 1, "BETA": 2})
    class SelectCase(tc.TestCase):
        def run_test(self):
            pass

    tests = datadrive.load_datadrive_tests(SelectCase, "BETA")

    assert len(tests) == 1
    assert tests[0].casedataname == "BETA"
    assert tests[0].casedata == 2


def test_load_with_unknown_name_raises_value_error():
    @datadrive.DataDrive({"ALPHA": 1})
    class MissingCase(tc.TestCase):
        def run_test(self):
            pass

    with pytest.raises(ValueError):
        datadrive.load_datadrive_tests(MissingCase, "does_not_exist")


def test_load_extracts_attrs_from_testdata():
    @datadrive.DataDrive({
        "case1": {"__attrs__": {"owner": "alice", "timeout": 3}, "value": 5},
    })
    class AttrCase(tc.TestCase):
        owner = "default"
        timeout = 1

        def run_test(self):
            pass

    (test,) = datadrive.load_datadrive_tests(AttrCase)

    assert test.owner == "alice"
    assert test.timeout == 3
    assert test.casedata == {"__attrs__": {"owner": "alice", "timeout": 3}, "value": 5}


def test_load_undecorated_class_without_datadrive_setting_raises():
    class Undecorated(tc.TestCase):
        def run_test(self):
            pass

    with modify_settings(DATA_DRIVE=False):
        with pytest.raises(RuntimeError):
            datadrive.load_datadrive_tests(Undecorated)


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------

def test_default_attributes_and_test_name_without_casedataname():
    class PlainCase(tc.TestCase):
        """  plain doc  """
        owner = "bob"

        def run_test(self):
            pass

    case = PlainCase()

    assert case.casedata is None
    assert case.casedataname is None
    assert case.owner == "bob"
    assert case.test_name == case.test_class_name
    assert case.test_class_name == "%s.PlainCase" % __name__


def test_test_name_includes_casedataname():
    class NamedCase(tc.TestCase):
        def run_test(self):
            pass

    case = NamedCase(testdata={"k": 1}, testdataname="KEY1")

    assert case.casedata == {"k": 1}
    assert case.test_name == "%s/KEY1" % case.test_class_name


def test_attrs_overwrite_only_whitelisted_keys():
    class WhitelistCase(tc.TestCase):
        owner = None
        priority = None

        def run_test(self):
            pass

    case = WhitelistCase(attrs={
        "owner": "carol",
        "priority": WhitelistCase.EnumPriority.High,
        "not_whitelisted": "ignored",
    })

    assert case.owner == "carol"
    assert case.priority == "High"
    assert not hasattr(case, "not_whitelisted")


def test_test_doc_is_stripped_from_class_docstring():
    class DocCase(tc.TestCase):
        """
        Some description
        """

        def run_test(self):
            pass

    assert DocCase().test_doc == "Some description"


def test_run_test_default_raises_not_implemented():
    class EmptyCase(tc.TestCase):
        pass

    with pytest.raises(NotImplementedError):
        EmptyCase().run_test()


def test_extra_info_collects_declared_fields():
    class ExtraCase(tc.TestCase):
        test_extra_info_def = [("component", None), ("level", None)]
        component = "core"

        def run_test(self):
            pass

    info = ExtraCase().test_extra_info

    assert info == {"component": "core", "level": None}


def test_assert_equal_mismatch_records_assert_failure():
    class EqCase(tc.TestCase):
        def run_test(self):
            pass

    case = EqCase()
    result = _FakeResult()
    case.init_test(result)

    assert case.assert_equal("values differ", 1, 2) is False
    assert len(result.records) == 1
    level, _msg = result.records[0]
    assert level == EnumLogLevel.ASSERT


def test_assert_equal_match_returns_true_without_logging():
    class EqCase2(tc.TestCase):
        def run_test(self):
            pass

    case = EqCase2()
    result = _FakeResult()
    case.init_test(result)

    assert case.assert_equal("same", "abc", "abc") is True
    assert result.records == []


def test_assert_failure_raises_when_assert_continue_disabled():
    class StrictCase(tc.TestCase):
        def run_test(self):
            pass

    case = StrictCase()
    result = _FakeResult()
    case.init_test(result)

    with modify_settings(QTAF_ASSERT_CONTINUE=False):
        with pytest.raises(RuntimeError):
            case.assert_(False, "should fail")


def test_fail_log_info_and_start_step_delegate_to_result():
    class DelegateCase(tc.TestCase):
        def run_test(self):
            pass

    case = DelegateCase()
    result = _FakeResult()
    case.init_test(result)

    case.fail("boom")
    case.log_info(42)
    case.start_step("step one")

    assert result.errors == ["boom"]
    assert result.infos == ["42"]
    assert result.steps == ["step one"]


def test_wait_for_equal_returns_true_when_value_already_matches():
    class WaitCase(tc.TestCase):
        def run_test(self):
            pass

    class Obj(object):
        state = "ready"

    case = WaitCase()
    case.init_test(_FakeResult())

    assert case.wait_for_equal("state check", Obj(), "state", "ready", timeout=1, interval=0.01) is True


def test_metaclass_rejects_init_override():
    with pytest.raises(RuntimeError):
        class BadInit(tc.TestCase):
            def __init__(self):
                pass


def test_metaclass_merges_base_tags_into_subclass_tags():
    class BaseTagged(tc.TestCase):
        tags = ["a"]

        def run_test(self):
            pass

    class ChildTagged(BaseTagged):
        tags = "b"

    assert ChildTagged.tags == set(["a", "b"])
    assert ChildTagged.base_tags == set(["a"])


# ---------------------------------------------------------------------------
# modify_settings
# ---------------------------------------------------------------------------

def test_modify_settings_sets_new_key_and_removes_it_on_exit():
    key = "QTAF_UNITTEST_NEW_KEY"

    assert getattr(settings, key, _MISSING) is _MISSING

    with modify_settings(**{key: "temp"}):
        assert settings.QTAF_UNITTEST_NEW_KEY == "temp"
        assert settings._Settings__sealed is False

    assert getattr(settings, key, _MISSING) is _MISSING
    assert settings._Settings__sealed is True


def test_modify_settings_nested_restores_previous_value():
    key = "QTAF_UNITTEST_EXISTING"

    with modify_settings(**{key: 1}):
        assert settings.QTAF_UNITTEST_EXISTING == 1
        with modify_settings(**{key: 2}):
            assert settings.QTAF_UNITTEST_EXISTING == 2
        assert settings.QTAF_UNITTEST_EXISTING == 1

    assert getattr(settings, key, _MISSING) is _MISSING


def test_modify_settings_restores_state_when_block_raises():
    key = "QTAF_UNITTEST_EXC_KEY"

    with pytest.raises(ValueError):
        with modify_settings(**{key: "x"}):
            assert settings.QTAF_UNITTEST_EXC_KEY == "x"
            raise ValueError("boom")

    assert getattr(settings, key, _MISSING) is _MISSING
    assert settings._Settings__sealed is True
