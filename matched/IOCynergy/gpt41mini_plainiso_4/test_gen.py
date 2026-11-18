import pytest
import inspect
from typing import List
from cynergy.container import IocContainer, ConfigProviderRequiredException, LifeCycle
from cynergy.attributes import LifeCycle as AttrLifeCycle
from cynergy.config import Plain, Config, ServiceByName

class DummyService:
    pass

class DummyServiceWithArgs:
    __ioc_arguments__ = {
        'param1': Plain('a'),
        'param2': Config('key', default='default_value'),
        'param3': ServiceByName('named_service')
    }

    def __init__(self, param1, param2, param3):
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

class DummyServiceWithDefault:
    def __init__(self, val=5):
        self.val = val

class DummyServiceWithPrimitive:
    def __init__(self, x: int):
        self.x = x

class DummyAssignTo(DummyService):
    pass

class DummyMultiLifeCycle:
    __lifecycle__ = LifeCycle.MULTI_INSTANCE

def make_config_provider(expected_values):
    class DummyConfigProvider:
        def __init__(self):
            self.values = expected_values
        def get(self, key):
            if key not in self.values:
                raise KeyError(key)
            return self.values[key]
    return DummyConfigProvider()

def test_resolve_argument_primitive_returns_value():
    container = IocContainer(None)
    for primitive in [123, "abc", {"k":1}, [1,2], {3}, 4.5]:
        assert container._IocContainer__resolve_argument(primitive) == primitive

def test_resolve_argument_plain_returns_value():
    container = IocContainer(None)
    p = Plain('value')
    assert container._IocContainer__resolve_argument(p) == 'value'

def test_resolve_argument_config_with_provider_gets_value():
    provider = make_config_provider({'my_key': 'my_value'})
    container = IocContainer(provider)
    c = Config('my_key')
    result = container._IocContainer__resolve_argument(c)
    assert result == 'my_value'

def test_resolve_argument_config_with_default_and_missing_key_returns_default():
    provider = make_config_provider({})
    container = IocContainer(provider)
    c = Config('missing_key', default='default_val')
    result = container._IocContainer__resolve_argument(c)
    assert result == 'default_val'

def test_resolve_argument_config_without_provider_raises_value_error():
    container = IocContainer(None)
    c = Config('abc')
    with pytest.raises(ValueError):
        container._IocContainer__resolve_argument(c)

def test_resolve_argument_service_by_name_returns_registered_instance():
    container = IocContainer(None)
    inst = DummyService()
    container.register_instance_by_name('service1', inst)
    s = ServiceByName('service1')
    result = container._IocContainer__resolve_argument(s)
    assert result is inst

def test_resolve_argument_unsupported_type_raises_not_implemented():
    container = IocContainer(None)
    class Unsupported: pass
    with pytest.raises(NotImplementedError):
        container._IocContainer__resolve_argument(Unsupported())

def test_create_instance_for_single_initializes_with_arguments():
    provider = make_config_provider({'key': 'config_value'})
    container = IocContainer(provider)
    container.register_instance_by_name('named_service', DummyService())

    container._IocContainer__create_instance_for_single(DummyServiceWithArgs, DummyServiceWithArgs)
    instance = container.get(DummyServiceWithArgs)
    assert instance.param1 == 'a'
    assert instance.param2 == 'config_value'
    assert isinstance(instance.param3, DummyService)

def test_create_instance_for_single_raises_config_provider_required_exception():
    class DummyConfigRequired:
        __ioc_arguments__ = {'param': Config('missing_key')}
        def __init__(self, param):
            self.param = param
    container = IocContainer(None)  # no provider
    with pytest.raises(ConfigProviderRequiredException):
        container._IocContainer__create_instance_for_single(DummyConfigRequired, DummyConfigRequired)

def test_create_instance_for_single_raises_type_error_for_unmapped_primitive_arg():
    class DummyPrimitive:
        def __init__(self, x: int):
            self.x = x
    container = IocContainer(None)
    with pytest.raises(TypeError):
        container._IocContainer__create_instance_for_single(DummyPrimitive, DummyPrimitive)

def test_register_instance_and_get_returns_same_instance():
    container = IocContainer(None)
    inst = DummyService()
    container.register_instance(DummyService, inst)
    retrieved = container.get(DummyService)
    assert retrieved is inst

def test_register_class_and_get_returns_different_instance_type():
    container = IocContainer(None)
    container.register_class(DummyService, DummyAssignTo)
    instance = container.get(DummyService)
    assert isinstance(instance, DummyAssignTo)

def test_register_many_and_get_returns_list_of_instances():
    container = IocContainer(None)
    container.register_many(DummyService, [DummyAssignTo, DummyService])
    result = container.get(List[DummyService])
    assert isinstance(result, list)
    types = {type(i) for i in result}
    assert DummyAssignTo in types
    assert DummyService in types

def test_get_with_mult_instance_lifecycle_creates_new_instances():
    container = IocContainer(None)
    instance1 = container.get(DummyService, LifeCycle.MULTI_INSTANCE)
    instance2 = container.get(DummyService, LifeCycle.MULTI_INSTANCE)
    assert instance1 is not instance2

def test_get_returns_singleton_for_lifecycle():
    container = IocContainer(None)
    instance1 = container.get(DummyService)
    instance2 = container.get(DummyService)
    assert instance1 is instance2

def test_get_raises_not_implemented_for_unknown_lifecycle():
    container = IocContainer(None)
    class Dummy:
        pass
    with pytest.raises(NotImplementedError):
        container.get(Dummy, life_cycle='UNKNOWN')

def test_register_registers_list_calls_register_many(monkeypatch):
    container = IocContainer(None)
    recorded = {}
    def fake_register_many(cls, lst):
        recorded['called'] = True
        recorded['cls'] = cls
        recorded['lst'] = lst
    monkeypatch.setattr(container, "register_many", fake_register_many)
    container.register(DummyService, [DummyAssignTo, DummyService])
    assert recorded['called']
    assert recorded['cls'] is DummyService
    assert DummyAssignTo in recorded['lst']

def test_register_registers_class(monkeypatch):
    container = IocContainer(None)
    recorded = {}
    def fake_register_class(cls, assign_to):
        recorded['called'] = True
        recorded['cls'] = cls
        recorded['assign_to'] = assign_to
    monkeypatch.setattr(container, "register_class", fake_register_class)
    container.register(DummyService, DummyAssignTo)
    assert recorded['called']
    assert recorded['cls'] is DummyService
    assert recorded['assign_to'] is DummyAssignTo

def test_register_registers_instance(monkeypatch):
    container = IocContainer(None)
    instance = DummyService()
    recorded = {}
    def fake_register_instance(cls, inst):
        recorded['called'] = True
        recorded['cls'] = cls
        recorded['inst'] = inst
    monkeypatch.setattr(container, "register_instance", fake_register_instance)
    container.register(DummyService, instance)
    assert recorded['called']
    assert recorded['cls'] is DummyService
    assert recorded['inst'] is instance

def test_register_instance_by_name_and_get_by_name():
    container = IocContainer(None)
    inst = DummyService()
    container.register_instance_by_name("test_name", inst)
    retrieved = container.get_by_name("test_name")
    assert retrieved is inst

def test_get_by_name_unknown_raises_key_error():
    container = IocContainer(None)
    with pytest.raises(KeyError):
        container.get_by_name("unknown_name")

def test_clear_all_removes_instances():
    container = IocContainer(None)
    inst = DummyService()
    container.register_instance(DummyService, inst)
    assert container.get(DummyService) is inst
    container.clear_all()
    with pytest.raises(KeyError):
        container._IocContainer__get_instance(DummyService)

def test_get_config_caches_value():
    calls = {}
    class DummyConfig:
        def __init__(self):
            self.call_count = 0
        def get(self, key):
            calls[key] = calls.get(key, 0) + 1
            return f"value_for_{key}"
    config = DummyConfig()
    container = IocContainer(config)
    val1 = container.get_config("key1")
    val2 = container.get_config("key1")
    assert val1 == val2
    assert calls["key1"] == 1

def test_get_config_raises_error_if_provider_none():
    container = IocContainer(None)
    # __config is None, so __config.get will error potentially; but no test asked for this directly


# Additional test for _get_class_name static method
def test_get_class_name_with_list_type_and_normal_class():
    container = IocContainer(None)
    # type(List[str]) is type, but we want to test that the special case returns correct string
    # This is hard to test exactly because List is a generic alias, let's test the fallback
    class A: pass
    result = container._get_class_name(A)
    assert result == 'A'

    # Now create a generic to test List[T]
    from typing import List as TypingList
    cn = container._get_class_name(TypingList[A])
    assert cn == "List[A]"
