import pytest
import inspect
from typing import List
from cynergy.container import IocContainer, LifeCycle, ConfigProviderRequiredException
from cynergy.attributes import LifeCycle as AttrLifeCycle
from cynergy.container import get, register, register_many, get_by_name
from cynergy.config import Plain, Config, ServiceByName


# Dummy classes for testing
class A:
    pass

class B:
    pass

class C:
    pass

class D:
    def __init__(self, a: A):
        self.a = a

class E:
    def __init__(self, x: int = 42):
        self.x = x

class F:
    def __init__(self, val: Plain):
        self.val = val

class G:
    def __init__(self, config_val: Config):
        self.config_val = config_val

class H:
    def __init__(self, service: ServiceByName):
        self.service = service

class I:
    # simulate no args
    pass

def test_register_and_get_instance():
    container = IocContainer(None)
    instance = A()
    container.register_instance(A, instance)
    retrieved = container.get(A)
    assert retrieved is instance

def test_register_class_and_get_returns_instance_of_assigned_class():
    container = IocContainer(None)
    container.register_class(A, B)
    instance = container.get(A)
    assert isinstance(instance, B)

def test_register_many_and_get_returns_list_of_instances():
    container = IocContainer(None)
    
    container.register_many(A, [B, C])
    instances = container.get(List[A])
    assert isinstance(instances, list)
    assert len(instances) == 2
    types = {type(i) for i in instances}
    assert types == {B, C}

def test_get_creates_singleton_instance():
    container = IocContainer(None)
    instance1 = container.get(A)
    instance2 = container.get(A)
    assert instance1 is instance2

def test_get_with_multi_instance_life_cycle_returns_different_instances():
    container = IocContainer(None)
    instance1 = container.get(A, LifeCycle.MULTI_INSTANCE)
    instance2 = container.get(A, LifeCycle.MULTI_INSTANCE)
    assert instance1 is not instance2
    assert isinstance(instance1, A)
    assert isinstance(instance2, A)

def test_get_raises_not_implemented_for_unknown_lifecycle():
    container = IocContainer(None)
    class DummyLifeCycle:
        pass

    with pytest.raises(NotImplementedError):
        container.get(A, DummyLifeCycle)

def test_get_with_injected_dependency():
    container = IocContainer(None)
    instance = container.get(D)
    assert isinstance(instance, D)
    assert isinstance(instance.a, A)

def test_register_method_dispatches_to_register_many_for_list():
    container = IocContainer(None)
    container.register(A, [B,C])
    result = container.get(List[A])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(i, (B,C)) for i in result)

def test_register_method_dispatches_to_register_class_for_class():
    container = IocContainer(None)
    container.register(A, B)
    instance = container.get(A)
    assert isinstance(instance, B)

def test_register_method_dispatches_to_register_instance_for_instance():
    container = IocContainer(None)
    instance = A()
    container.register(A, instance)
    retrieved = container.get(A)
    assert retrieved is instance

def test_create_instance_with_plain_argument():
    container = IocContainer(None)
    class J:
        IOC_ARGUMENTS_NAME = {'val': Plain(123)}
        def __init__(self, val): 
            self.val = val

    container.register_class(J, J)
    obj = container.get(J)
    assert obj.val == 123

def test_create_instance_with_config_argument_and_configprovider():
    class DummyConfigProvider(ConfigProvider):
        def __init__(self):
            self.store = {'key': 'value'}
        def get(self, key):
            return self.store[key]

    provider = DummyConfigProvider()
    container = IocContainer(provider)

    class K:
        IOC_ARGUMENTS_NAME = {'config_val': Config('key')}
        def __init__(self, config_val):
            self.config_val = config_val

    container.register_class(K, K)
    instance = container.get(K)
    assert instance.config_val == 'value'

def test_create_instance_with_config_argument_raises_without_provider():
    container = IocContainer(None)

    class L:
        IOC_ARGUMENTS_NAME = {'config_val': Config('key')}
        def __init__(self, config_val):
            self.config_val = config_val

    container.register_class(L, L)
    with pytest.raises(ValueError):
        container.get(L)

def test_create_instance_with_config_argument_default_value_on_missing_key():
    class DummyConfigProvider(ConfigProvider):
        def get(self, key):
            raise KeyError

    provider = DummyConfigProvider()
    container = IocContainer(provider)

    class M:
        IOC_ARGUMENTS_NAME = {'config_val': Config('missing_key', default='default_value')}
        def __init__(self, config_val):
            self.config_val = config_val

    container.register_class(M, M)
    instance = container.get(M)
    assert instance.config_val == 'default_value'

def test_create_instance_with_service_by_name_argument():
    container = IocContainer(None)

    service_instance = B()
    container.register_instance_by_name('my_service', service_instance)

    class N:
        IOC_ARGUMENTS_NAME = {'service': ServiceByName('my_service')}
        def __init__(self, service):
            self.service = service

    container.register_class(N, N)
    instance = container.get(N)
    assert instance.service is service_instance

def test_get_by_name_not_registered_raises_keyerror():
    container = IocContainer(None)
    with pytest.raises(KeyError):
        container.get_by_name('not_registered')

def test_get_by_name_returns_registered_instance():
    container = IocContainer(None)
    inst = A()
    container.register_instance_by_name('myname', inst)
    retrieved = container.get_by_name('myname')
    assert retrieved is inst

def test_clear_all_removes_instances():
    container = IocContainer(None)
    inst = A()
    container.register_instance(A, inst)
    container.clear_all()
    # After clear, a new instance will be returned different from old if we get again
    new_inst = container.get(A)
    assert new_inst is not inst

def test_register_instance_warns_deprecated(monkeypatch):
    # monkeypatch the warnings.warn to capture warnings
    import warnings
    recorded = {}

    def fake_warn(msg, category=None):
        recorded['msg'] = msg
        recorded['category'] = category
    monkeypatch.setattr(warnings, "warn", fake_warn)

    container = IocContainer(None)
    inst = A()
    container.register_instance(A, inst)
    assert "deprecated" in recorded['msg']

def test_register_class_registers_mapping():
    container = IocContainer(None)
    container.register_class(A, B)
    class_name = container._get_class_name(A)
    assert container._IocContainer__class_mapping[class_name] == B

def test_register_many_registers_list_mapping():
    container = IocContainer(None)
    container.register_many(A, [B, C])
    class_name = container._get_class_name(List[A])
    assert container._IocContainer__class_mapping[class_name] == [B,C]

def test_get_config_caches_values():
    called = {}

    class DummyCP(ConfigProvider):
        def get(self, key):
            called[key] = called.get(key, 0) + 1
            return f"value_of_{key}"

    provider = DummyCP()
    container = IocContainer(provider)
    val1 = container.get_config("key1")
    val2 = container.get_config("key1")
    assert val1 == 'value_of_key1'
    assert val2 == 'value_of_key1'
    assert called["key1"] == 1  # cached, so only 1 call

def test_get_with_lifecycle_attribute_on_class(monkeypatch):
    container = IocContainer(None)

    class WithLifeCycle:
        LifeCycle = AttrLifeCycle.MULTI_INSTANCE

    # Set lifecycle attribute so get uses it
    setattr(WithLifeCycle, 'IOC_LIFECYCLE', AttrLifeCycle.MULTI_INSTANCE)

    container.register_class(WithLifeCycle, WithLifeCycle)

    # Use life_cycle=None to trigger attribute usage
    inst1 = container.get(WithLifeCycle, life_cycle=None)
    inst2 = container.get(WithLifeCycle, life_cycle=None)
    # Because lifecycle is MULTI_INSTANCE attribute, should get different instance
    assert inst1 is not inst2
