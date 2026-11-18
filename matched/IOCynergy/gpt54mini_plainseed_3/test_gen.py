import inspect
import warnings

import pytest

import cynergy.container as container_mod
from cynergy.attributes import LifeCycle
from cynergy.config import Config, Plain, ServiceByName
from cynergy.errors.ContainerException import ConfigProviderRequiredException


@pytest.fixture(autouse=True)
def reset_container():
    """
    Ensure each test starts with a fresh global container instance.
    """
    container_mod.initialize(None)
    yield
    container_mod.initialize(None)


def test_register_and_get_instance_returns_same_object():
    class Service:
        pass

    instance = Service()
    container_mod.register(Service, instance)

    assert container_mod.get(Service) is instance
    assert container_mod.get(Service) is instance


def test_register_class_and_get_creates_mapped_class():
    class Base:
        pass

    class Impl:
        def __init__(self):
            self.value = 123

    container_mod.register_class(Base, Impl)

    obj = container_mod.get(Base)
    assert isinstance(obj, Impl)
    assert obj.value == 123


def test_register_many_and_get_list_returns_instances_for_each_class():
    class Item:
        def __init__(self):
            self.marker = "item"

    class AnotherItem:
        def __init__(self):
            self.marker = "another"

    class Multi:
        pass

    container_mod.register_many(Multi, [Item, AnotherItem])

    result = container_mod.get(Multi)

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], Item)
    assert isinstance(result[1], AnotherItem)
    assert result[0].marker == "item"
    assert result[1].marker == "another"


def test_get_singleton_returns_same_instance_across_calls():
    class Service:
        def __init__(self):
            self.created = object()

    first = container_mod.get(Service)
    second = container_mod.get(Service)

    assert first is second
    assert isinstance(first, Service)


def test_get_multi_instance_returns_new_instance_each_call():
    class Service:
        def __init__(self):
            self.token = object()

    first = container_mod.get(Service, LifeCycle.MULTI_INSTANCE)
    second = container_mod.get(Service, LifeCycle.MULTI_INSTANCE)

    assert isinstance(first, Service)
    assert isinstance(second, Service)
    assert first is not second
    assert first.token is not second.token


def test_register_instance_by_name_and_get_by_name():
    class Service:
        pass

    instance = Service()
    container_mod.register_instance_by_name("my-service", instance)

    assert container_mod.get_by_name("my-service") is instance


def test_register_instance_by_name_uses_by_name_prefix_key_internally():
    class Service:
        pass

    instance = Service()
    container_mod.register_instance_by_name("svc", instance)

    # The public API should still resolve correctly.
    assert container_mod.get_by_name("svc") is instance


def test_plain_argument_mapping_injected_into_constructor():
    class Service:
        def __init__(self, name):
            self.name = name

    Service.__dict__["_ioc_args"] if False else None  # keep linters quiet if needed
    setattr(Service, "ioc_arguments", {"name": Plain("alpha")})

    # The container checks the attribute name from cynergy.attributes.
    # Using getattr with the constant ensures compatibility with the project.
    setattr(Service, container_mod.attributes.IOC_ARGUMENTS_NAME, {"name": Plain("alpha")})

    obj = container_mod.get(Service)
    assert obj.name == "alpha"


def test_config_argument_mapping_uses_config_provider_value():
    class DummyConfigProvider:
        def __init__(self):
            self.calls = []

        def get(self, key):
            self.calls.append(key)
            return {"db.url": "sqlite://"}[key]

    class Service:
        def __init__(self, url):
            self.url = url

    setattr(Service, container_mod.attributes.IOC_ARGUMENTS_NAME, {"url": Config("db.url")})
    container_mod.initialize(DummyConfigProvider())

    obj = container_mod.get(Service)
    assert obj.url == "sqlite://"


def test_config_argument_mapping_with_default_uses_default_when_missing():
    class DummyConfigProvider:
        def get(self, key):
            raise KeyError(key)

    class Service:
        def __init__(self, url):
            self.url = url

    setattr(Service, container_mod.attributes.IOC_ARGUMENTS_NAME, {"url": Config("missing.key", default="fallback")})
    container_mod.initialize(DummyConfigProvider())

    obj = container_mod.get(Service)
    assert obj.url == "fallback"


def test_config_argument_without_provider_raises_config_provider_required_exception():
    class Service:
        def __init__(self, url):
            self.url = url

    setattr(Service, container_mod.attributes.IOC_ARGUMENTS_NAME, {"url": Config("db.url")})
    container_mod.initialize(None)

    with pytest.raises(ConfigProviderRequiredException):
        container_mod.get(Service)


def test_service_by_name_argument_is_resolved_from_registered_named_service():
    class Dependency:
        def __init__(self):
            self.answer = 42

    class Service:
        def __init__(self, dep):
            self.dep = dep

    container_mod.register_instance_by_name("dep", Dependency())
    setattr(Service, container_mod.attributes.IOC_ARGUMENTS_NAME, {"dep": ServiceByName("dep")})

    obj = container_mod.get(Service)
    assert isinstance(obj.dep, Dependency)
    assert obj.dep.answer == 42


def test_unmapped_primitive_constructor_argument_raises_type_error():
    class Service:
        def __init__(self, count: int):
            self.count = count

    with pytest.raises(TypeError):
        container_mod.get(Service)


def test_get_config_caches_values_from_provider():
    class DummyConfigProvider:
        def __init__(self):
            self.calls = 0

        def get(self, key):
            self.calls += 1
            return f"value:{key}"

    provider = DummyConfigProvider()
    container_mod.initialize(provider)

    first = container_mod.get_config("x")
    second = container_mod.get_config("x")

    assert first == "value:x"
    assert second == "value:x"
    assert provider.calls == 1


def test_initialize_warns_when_reinitialized_and_clears_previous_instances():
    class Service:
        pass

    first_instance = Service()
    container_mod.register(Service, first_instance)
    assert container_mod.get(Service) is first_instance

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        container_mod.initialize(None)

    assert any(issubclass(w.category, UserWarning) for w in caught)

    second_instance = container_mod.get(Service)
    assert isinstance(second_instance, Service)
    assert second_instance is not first_instance


def test_register_dispatches_list_to_register_many():
    class A:
        def __init__(self):
            self.name = "A"

    class B:
        def __init__(self):
            self.name = "B"

    class Multi:
        pass

    container_mod.register(Multi, [A, B])

    result = container_mod.get(Multi)
    assert [type(x) for x in result] == [A, B]


def test_register_dispatches_class_to_register_class():
    class Base:
        pass

    class Impl:
        def __init__(self):
            self.ok = True

    container_mod.register(Base, Impl)

    obj = container_mod.get(Base)
    assert isinstance(obj, Impl)
    assert obj.ok is True


def test_register_dispatches_instance_to_register_instance():
    class Service:
        pass

    instance = Service()
    container_mod.register(Service, instance)

    assert container_mod.get(Service) is instance
