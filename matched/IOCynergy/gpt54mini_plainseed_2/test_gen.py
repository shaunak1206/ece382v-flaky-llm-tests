import inspect
from typing import List

import pytest

from cynergy import container as container_module
from cynergy.attributes import LifeCycle
from cynergy.errors.ContainerException import ConfigProviderRequiredException


class DummyConfigProvider:
    def __init__(self, values=None):
        self.values = values or {}
        self.calls = []

    def get(self, key):
        self.calls.append(key)
        return self.values[key]


class ServiceA:
    def __init__(self):
        self.value = "A"


class ServiceB:
    def __init__(self):
        self.value = "B"


class Dep:
    def __init__(self):
        self.value = "dep"


class NeedsDep:
    def __init__(self, dep: Dep):
        self.dep = dep


class NeedsPrimitive:
    def __init__(self, name: str):
        self.name = name


class NeedsConfig:
    def __init__(self, host: str):
        self.host = host


class NeedsDefault:
    def __init__(self, a: int = 5):
        self.a = a


class MultiCtor:
    def __init__(self):
        self.created = True


@pytest.fixture(autouse=True)
def reset_container():
    container_module.initialize()
    yield
    container_module.initialize()


def test_register_instance_and_get_returns_same_object():
    obj = ServiceA()
    container_module.register(ServiceA, obj)

    result = container_module.get(ServiceA)

    assert result is obj


def test_register_class_and_get_creates_mapped_class():
    container_module.register(ServiceA, ServiceB)

    result = container_module.get(ServiceA)

    assert isinstance(result, ServiceB)
    assert result.value == "B"


def test_register_many_and_get_returns_list_of_registered_services():
    container_module.register_many(ServiceA, [ServiceA, ServiceB])

    result = container_module.get(List[ServiceA])

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], ServiceA)
    assert isinstance(result[1], ServiceB)


def test_register_many_preserves_singleton_instances_for_list_mapping():
    container_module.register_many(ServiceA, [ServiceA, ServiceB])

    first = container_module.get(List[ServiceA])
    second = container_module.get(List[ServiceA])

    assert first is second
    assert first[0] is second[0]
    assert first[1] is second[1]


def test_register_dispatches_to_register_instance_for_object():
    obj = ServiceA()

    container_module.register(ServiceA, obj)

    assert container_module.get(ServiceA) is obj


def test_register_dispatches_to_register_class_for_class_argument():
    container_module.register(ServiceA, ServiceB)

    assert isinstance(container_module.get(ServiceA), ServiceB)


def test_register_dispatches_to_register_many_for_list_argument():
    container_module.register(ServiceA, [ServiceA, ServiceB])

    result = container_module.get(List[ServiceA])

    assert [type(x) for x in result] == [ServiceA, ServiceB]


def test_get_singleton_returns_same_instance_on_repeated_calls():
    first = container_module.get(ServiceA)
    second = container_module.get(ServiceA)

    assert first is second


def test_get_multi_instance_returns_different_instances_each_call():
    first = container_module.get(ServiceA, life_cycle=LifeCycle.MULTI_INSTANCE)
    second = container_module.get(ServiceA, life_cycle=LifeCycle.MULTI_INSTANCE)

    assert first is not second


def test_get_uses_lifecycle_attribute_when_life_cycle_argument_is_false():
    class MultiService:
        __ioc_lifecycle__ = LifeCycle.MULTI_INSTANCE

        def __init__(self):
            pass

    first = container_module.get(MultiService, life_cycle=False)
    second = container_module.get(MultiService, life_cycle=False)

    assert first is not second


def test_get_uses_existing_instance_for_registered_class_mapping():
    container_module.register_class(ServiceA, ServiceB)

    result = container_module.get(ServiceA)

    assert isinstance(result, ServiceB)
    assert result.value == "B"


def test_get_creates_dependency_by_type_annotation():
    container_module.register(Dep, Dep)

    result = container_module.get(NeedsDep)

    assert isinstance(result, NeedsDep)
    assert isinstance(result.dep, Dep)


def test_get_raises_type_error_for_unmapped_primitive_argument():
    with pytest.raises(TypeError) as exc:
        container_module.get(NeedsPrimitive)

    assert "primitive argument [name]" in str(exc.value)


def test_get_raises_config_provider_required_exception_when_config_missing():
    class ConfigArgService:
        def __init__(self, host: str):
            self.host = host

    ConfigArgService.__ioc_arguments__ = {"host": container_module.Config("missing_key")}

    with pytest.raises(ConfigProviderRequiredException):
        container_module.get(ConfigArgService)


def test_initialize_with_config_provider_allows_config_injection():
    class ConfigArgService:
        def __init__(self, host: str):
            self.host = host

    provider = DummyConfigProvider({"host_key": "example.org"})
    ConfigArgService.__ioc_arguments__ = {"host": container_module.Config("host_key")}

    container_module.initialize(provider)

    result = container_module.get(ConfigArgService)

    assert result.host == "example.org"
    assert provider.calls == ["host_key"]


def test_get_config_caches_provider_values():
    provider = DummyConfigProvider({"port": 8080})
    container_module.initialize(provider)

    first = container_module.get_config("port")
    second = container_module.get_config("port")

    assert first == 8080
    assert second == 8080
    assert provider.calls == ["port"]


def test_clear_all_removes_cached_instances():
    instance = container_module.get(ServiceA)
    assert instance is container_module.get(ServiceA)

    container_module._clear_all()
    new_instance = container_module.get(ServiceA)

    assert new_instance is not instance


def test_register_many_uses_typing_list_key_name():
    container_module.register_many(ServiceA, [ServiceA, ServiceB])

    assert container_module.get(List[ServiceA])[0].__class__ is ServiceA
    assert container_module.get(List[ServiceA])[1].__class__ is ServiceB
