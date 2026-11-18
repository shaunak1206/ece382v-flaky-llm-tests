import pytest
import inspect
from typing import List

from cynergy.container import IocContainer, LifeCycle, ConfigProviderRequiredException
from cynergy.attributes import LifeCycle as AttrLifeCycle
from cynergy.container import register, get, register_many, get_by_name
from cynergy.attributes import IOC_ARGUMENTS_NAME

# Dummy service classes for testing
class ServiceA:
    def __init__(self):
        self.value = "A"

class ServiceB:
    def __init__(self):
        self.value = "B"

class ServiceC:
    def __init__(self, service_a: ServiceA):
        self.service_a = service_a

class ServiceD:
    def __init__(self, val: int = 42):
        self.val = val

class ServiceWithArgs:
    __annotations__ = {'val': int}
    def __init__(self, val):
        self.val = val

class ServiceWithIocArgs:
    setattr = None
    setattr = None

    def __init__(self, name: str, default: int):
        self.name = name
        self.default = default

# Mock ConfigProvider
class DummyConfigProvider:
    def __init__(self, mapping):
        self.mapping = mapping
    def get(self, key):
        if key not in self.mapping:
            raise KeyError(key)
        return self.mapping[key]

def test_register_and_get_instance_simple():
    container = IocContainer(None)
    instance = ServiceA()
    container.register(ServiceA, instance)
    got = container.get(ServiceA)
    assert got is instance

def test_register_class_and_get_instance():
    container = IocContainer(None)
    container.register(ServiceA, ServiceA)
    got = container.get(ServiceA)
    assert isinstance(got, ServiceA)

def test_register_many_and_get_list():
    container = IocContainer(None)
    container.register_many(ServiceA, [ServiceA, ServiceB])
    got = container.get(List[ServiceA])
    assert isinstance(got, list)
    assert any(isinstance(x, ServiceA) for x in got)
    assert any(isinstance(x, ServiceB) for x in got)

def test_get_by_name_and_register_instance_by_name():
    container = IocContainer(None)
    instance = ServiceA()
    container.register_instance_by_name("myservice", instance)
    got = container.get_by_name("myservice")
    assert got is instance

def test_get_by_name_raises_keyerror_if_not_registered():
    container = IocContainer(None)
    with pytest.raises(KeyError):
        container.get_by_name("nonexistent")

def test_get_with_lifecycle_singleton_same_instance():
    container = IocContainer(None)
    container.register(ServiceA, ServiceA)
    a1 = container.get(ServiceA, life_cycle=LifeCycle.SINGLETON)
    a2 = container.get(ServiceA, life_cycle=LifeCycle.SINGLETON)
    assert a1 is a2

def test_get_with_lifecycle_multi_instance_returns_different():
    container = IocContainer(None)
    container.register(ServiceA, ServiceA)
    a1 = container.get(ServiceA, life_cycle=LifeCycle.MULTI_INSTANCE)
    a2 = container.get(ServiceA, life_cycle=LifeCycle.MULTI_INSTANCE)
    assert a1 is not a2
    assert isinstance(a1, ServiceA)
    assert isinstance(a2, ServiceA)

def test_clear_all_removes_instances():
    container = IocContainer(None)
    container.register(ServiceA, ServiceA)
    _ = container.get(ServiceA)
    container.clear_all()
    # After clear, new instance will be different
    new_instance = container.get(ServiceA)
    assert new_instance is not None

def test_register_accepts_list_register_many():
    container = IocContainer(None)
    container.register(ServiceA, [ServiceA, ServiceB])
    got = container.get(List[ServiceA])
    assert any(isinstance(x, ServiceA) for x in got)
    assert any(isinstance(x, ServiceB) for x in got)

def test_register_accepts_class_register_class():
    container = IocContainer(None)
    container.register(ServiceA, ServiceB)
    instance = container.get(ServiceA)
    assert isinstance(instance, ServiceB)

def test_register_accepts_instance_register_instance():
    container = IocContainer(None)
    instance = ServiceA()
    container.register(ServiceA, instance)
    got = container.get(ServiceA)
    assert got is instance

def test__get_class_name_for_list():
    name = IocContainer._get_class_name(List[ServiceA])
    assert name == "List[ServiceA]"

def test_create_instance_for_single_with_argument_mapping(monkeypatch):
    class ArgClass:
        def __init__(self, val):
            self.val = val
    ArgClass.__dict__[IOC_ARGUMENTS_NAME] = {'val': 123}
    container = IocContainer(None)
    container.register_class(ArgClass, ArgClass)
    instance = container.get(ArgClass)
    assert instance.val == 123

def test_create_instance_for_single_without_argument_mapping_and_default():
    class DefaultArgClass:
        def __init__(self, val=99):
            self.val = val
    container = IocContainer(None)
    container.register_class(DefaultArgClass, DefaultArgClass)
    instance = container.get(DefaultArgClass)
    assert instance.val == 99

def test_create_instance_for_single_raises_when_primitive_arg_without_mapping():
    class PrimitiveArgClass:
        def __init__(self, x: int):
            self.x = x
    container = IocContainer(None)
    # Not registering mapping, expecting TypeError due to primitive arg
    with pytest.raises(TypeError):
        container.get(PrimitiveArgClass)

def test_create_instance_for_single_with_config_argument(monkeypatch):
    from cynergy.config import Plain, Config

    class ConfigArgClass:
        __dict__[IOC_ARGUMENTS_NAME] = {'val': Plain('hello')}
        def __init__(self, val):
            self.val = val

    config_provider = DummyConfigProvider({})
    container = IocContainer(config_provider)
    container.register_class(ConfigArgClass, ConfigArgClass)

    instance = container.get(ConfigArgClass)
    assert instance.val == 'hello'

def test_create_instance_for_single_with_config_argument_and_config(monkeypatch):
    from cynergy.config import Config

    class ConfigKeyClass:
        __dict__[IOC_ARGUMENTS_NAME] = {'val': Config('key1')}
        def __init__(self, val):
            self.val = val

    config_provider = DummyConfigProvider({'key1': 'value1'})
    container = IocContainer(config_provider)
    container.register_class(ConfigKeyClass, ConfigKeyClass)
    instance = container.get(ConfigKeyClass)
    assert instance.val == 'value1'

def test_create_instance_for_single_with_config_argument_and_keyerror_returns_default():
    from cynergy.config import Config

    class ConfigDefaultClass:
        __dict__[IOC_ARGUMENTS_NAME] = {'val': Config('nonexistent', default='def')}
        def __init__(self, val):
            self.val = val

    config_provider = DummyConfigProvider({})
    container = IocContainer(config_provider)
    container.register_class(ConfigDefaultClass, ConfigDefaultClass)

    instance = container.get(ConfigDefaultClass)
    assert instance.val == 'def'

def test_create_instance_for_single_with_config_argument_raises_valueerror_if_no_config(monkeypatch):
    from cynergy.config import Config

    class ConfigNoProviderClass:
        __dict__[IOC_ARGUMENTS_NAME] = {'val': Config('key1')}
        def __init__(self, val):
            self.val = val

    container = IocContainer(None)
    container.register_class(ConfigNoProviderClass, ConfigNoProviderClass)
    with pytest.raises(ValueError):
        container.get(ConfigNoProviderClass)

def test_get_raises_not_implemented_lifecycle():
    container = IocContainer(None)
    class Dummy:
        pass
    with pytest.raises(NotImplementedError):
        container.get(Dummy, life_cycle="UNKNOWN")

def test_register_instance_by_name_adds_instance_and_get_by_name_returns_it():
    container = IocContainer(None)
    inst = ServiceA()
    container.register_instance_by_name("myname", inst)
    got = container.get_by_name("myname")
    assert got is inst

def test_register_class_overwrites_previous_mapping():
    container = IocContainer(None)
    container.register_class(ServiceA, ServiceA)
    container.register_class(ServiceA, ServiceB)
    got = container.get(ServiceA)
    assert isinstance(got, ServiceB)

def test_register_many_overwrites_previous_mapping():
    container = IocContainer(None)
    container.register_many(ServiceA, [ServiceA])
    container.register_many(ServiceA, [ServiceB])
    got = container.get(List[ServiceA])
    assert any(isinstance(x, ServiceB) for x in got)

def test_get_config_caches_value():
    class DummyConfig:
        def __init__(self):
            self.called = 0
        def get(self, key):
            self.called += 1
            return key + "_value"

    config_provider = DummyConfig()
    container = IocContainer(config_provider)

    val1 = container.get_config("test")
    val2 = container.get_config("test")
    assert val1 == "test_value"
    assert val2 == "test_value"
    assert config_provider.called == 1  # get called only once due to caching

def test_register_warns_on_deprecated_register_instance(monkeypatch):
    import warnings
    from cynergy.container import register_instance

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        register_instance(ServiceA, ServiceA())
        assert any(issubclass(warning.category, Warning) for warning in w)
