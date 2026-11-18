import pytest
import inspect
from typing import List

import cynergy.container as container
from cynergy.attributes import LifeCycle
from cynergy.config import Plain, Config, ServiceByName
from cynergy.errors.ContainerException import ConfigProviderRequiredException


class DummyConfigProvider(container.ConfigProvider):
    def __init__(self, data=None):
        self.data = data or {}

    def get(self, key):
        if key not in self.data:
            raise KeyError(key)
        return self.data[key]


# Sample classes for testing
class ClassA:
    def __init__(self):
        self.value = "A"


class ClassB:
    def __init__(self, a: ClassA):
        self.a = a


class ClassC:
    def __init__(self, val: int = 123):
        self.val = val


class ClassWithPlain:
    __cynergy_arguments__ = {'txt': Plain("hello")}

    def __init__(self, txt: str):
        self.txt = txt


class ClassWithConfig:
    __cynergy_arguments__ = {'param': Config('some.key', default="fallback")}

    def __init__(self, param: str):
        self.param = param


class ClassWithServiceByName:
    __cynergy_arguments__ = {'named': ServiceByName('myservice')}

    def __init__(self, named: ClassA):
        self.named = named


class ClassWithLifecycle:
    _life_cycle = LifeCycle.MULTI_INSTANCE

    def __init__(self):
        self.value = 1


def test_register_and_get_instance_singleton():
    c = container.IocContainer(None)
    inst = ClassA()
    c.register_instance(ClassA, inst)
    retrieved = c.get(ClassA)
    assert retrieved is inst


def test_register_class_and_get_instance():
    c = container.IocContainer(None)
    c.register_class(ClassA, ClassA)
    inst = c.get(ClassA)
    assert isinstance(inst, ClassA)


def test_register_class_map_and_get_instance():
    c = container.IocContainer(None)

    class ImplA(ClassA):
        def __init__(self):
            super().__init__()
            self.impl = True

    c.register_class(ClassA, ImplA)
    inst = c.get(ClassA)
    assert isinstance(inst, ImplA)
    assert getattr(inst, "impl", False) is True


def test_register_many_and_get_list():
    c = container.IocContainer(None)
    c.register_many(ClassA, [ClassA, ClassA])
    lst = c.get(List[ClassA])
    assert isinstance(lst, list)
    assert all(isinstance(x, ClassA) for x in lst)
    assert len(lst) == 2


def test_register_instance_by_name_and_get_by_name():
    c = container.IocContainer(None)
    inst = ClassA()
    c.register_instance_by_name("myservice", inst)
    retrieved = c.get_by_name("myservice")
    assert retrieved is inst


def test_get_by_name_raises_if_not_registered():
    c = container.IocContainer(None)
    with pytest.raises(KeyError):
        c.get_by_name("not_registered")


def test_get_with_primitives_and_plain_and_config_and_service_by_name():
    # Setup config provider with data
    config_data = {"some.key": "CONFIG_VALUE"}
    config_provider = DummyConfigProvider(config_data)
    c = container.IocContainer(config_provider)

    # Register instance for service by name
    svc_instance = ClassA()
    c.register_instance_by_name("myservice", svc_instance)

    # Register ClassWithPlain, ClassWithConfig, ClassWithServiceByName
    inst_plain = c.get(ClassWithPlain)
    assert inst_plain.txt == "hello"

    inst_config = c.get(ClassWithConfig)
    assert inst_config.param == "CONFIG_VALUE"

    inst_service = c.get(ClassWithServiceByName)
    assert inst_service.named is svc_instance


def test_get_with_config_raises_if_no_provider():
    c = container.IocContainer(None)
    with pytest.raises(ValueError):
        c.get(ClassWithConfig)


def test_get_with_config_returns_default_if_key_missing():
    config_provider = DummyConfigProvider({})
    c = container.IocContainer(config_provider)
    inst = c.get(ClassWithConfig)
    assert inst.param == "fallback"


def test_get_creates_multi_instance_each_call():
    c = container.IocContainer(None)

    class MultiInst:
        _life_cycle = LifeCycle.MULTI_INSTANCE
        def __init__(self):
            pass

    # Register MultiInst to itself
    c.register_class(MultiInst, MultiInst)

    first = c.get(MultiInst, life_cycle=LifeCycle.MULTI_INSTANCE)
    second = c.get(MultiInst, life_cycle=LifeCycle.MULTI_INSTANCE)
    assert first is not second


def test_get_reuses_singleton_instance():
    c = container.IocContainer(None)
    c.register_class(ClassA, ClassA)
    first = c.get(ClassA)
    second = c.get(ClassA)
    assert first is second


def test_register_selects_method_correctly():
    c = container.IocContainer(None)
    c.register(ClassA, ClassA)
    assert c._IocContainer__class_mapping[container.IocContainer._get_class_name(ClassA)] == ClassA

    inst = ClassA()
    c.register(ClassA, inst)
    assert c._IocContainer__instances[container.IocContainer._get_class_name(ClassA)] == inst

    many = [ClassA, ClassB]
    c.register(List[ClassA], many)
    key = container.IocContainer._get_class_name(List[ClassA])
    assert c._IocContainer__class_mapping[key] == many


def test_clear_all_removes_all_instances():
    c = container.IocContainer(None)
    inst = ClassA()
    c.register_instance(ClassA, inst)
    assert c.get(ClassA) is inst
    c.clear_all()
    with pytest.raises(KeyError):
        c._IocContainer__get_instance(ClassA)


def test_get_raises_not_implemented_lifecycle():
    c = container.IocContainer(None)

    class UnknownLifeCycleClass:
        pass

    with pytest.raises(NotImplementedError):
        c.get(UnknownLifeCycleClass, life_cycle="unknown")


def test_private_method__get_class_name_with_list_type():
    name = container.IocContainer._get_class_name(List[ClassA])
    assert name == f"List[{ClassA.__name__}]"

def test_private_method__get_class_name_with_normal_class():
    name = container.IocContainer._get_class_name(ClassA)
    assert name == ClassA.__name__


def test___create_instance_for_single_with_unregistered_dependency():
    c = container.IocContainer(None)

    class CDependent:
        def __init__(self, a: ClassA):
            self.a = a

    # Register CDependent only. ClassA is not registered.
    c.register_class(CDependent, CDependent)
    instance = c.get(CDependent)
    assert isinstance(instance, CDependent)
    assert isinstance(instance.a, ClassA)


def test___create_instance_for_single_raises_on_module_registration():
    c = container.IocContainer(None)

    import inspect

    # Try registering a module instead of class to trigger exception
    with pytest.raises(Exception, match="Maybe you tried to register a module instead of a class?"):
        c._IocContainer__create_instance_for_single(inspect, inspect)

def test_resolve_argument_unsupported_type_raises():
    c = container.IocContainer(None)

    class Unsupported:
        pass

    with pytest.raises(NotImplementedError):
        # access private method
        c._IocContainer__resolve_argument(Unsupported())

def test_register_instance_by_name_stores_name_with_format():
    c = container.IocContainer(None)
    inst = ClassA()
    c.register_instance_by_name("testname", inst)
    key = c._IocContainer__BY_NAME_FORMAT.format("testname")
    assert c._IocContainer__instances[key] is inst

def test_get_config_caches_value():
    data = {"foo": 42}
    config_provider = DummyConfigProvider(data)
    c = container.IocContainer(config_provider)
    val1 = c.get_config("foo")
    val2 = c.get_config("foo")
    assert val1 == 42
    assert val1 is val2  # cached instance


def test_register_deprecated_warns(monkeypatch):
    c = container.IocContainer(None)
    called = {}

    def fake_warn(msg):
        called["warned"] = True
        assert "deprecated" in msg.lower()

    monkeypatch.setattr(container, "warn", fake_warn)
    c.register_instance(ClassA, ClassA())
    assert called.get("warned", False)
