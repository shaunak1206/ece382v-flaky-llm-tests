import warnings
from typing import List

import pytest

from cynergy import attributes
from cynergy import container as ioc_module
from cynergy.container import IocContainer


class Base(object):
    pass


class ImplA(Base):
    pass


class ImplB(Base):
    pass


class Service(object):
    pass


class Consumer(object):
    def __init__(self, service: Service):
        self.service = service


class NeedsPrimitive(object):
    def __init__(self, name: str):
        self.name = name


class WithDefault(object):
    def __init__(self, retries: int = 3):
        self.retries = retries


@pytest.fixture
def container():
    return IocContainer(None)


@pytest.fixture
def module_container():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ioc_module.initialize()
    yield
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ioc_module.initialize()


# ---------- get ----------

def test_get_default_singleton_returns_same_instance(container):
    assert container.get(Service) is container.get(Service)


def test_get_multi_instance_returns_new_objects(container):
    first = container.get(Service, attributes.LifeCycle.MULTI_INSTANCE)
    second = container.get(Service, attributes.LifeCycle.MULTI_INSTANCE)
    assert isinstance(first, Service)
    assert first is not second


def test_get_resolves_annotated_dependency(container):
    consumer = container.get(Consumer)
    assert isinstance(consumer, Consumer)
    assert isinstance(consumer.service, Service)


def test_get_uses_default_value_for_parameter(container):
    instance = container.get(WithDefault)
    assert instance.retries == 3


def test_get_primitive_without_mapping_raises_type_error(container):
    with pytest.raises(TypeError):
        container.get(NeedsPrimitive)


def test_get_unknown_lifecycle_raises_not_implemented(container):
    with pytest.raises(NotImplementedError):
        container.get(Service, "unknown-lifecycle")


def test_get_none_lifecycle_uses_class_lifecycle_attribute(container):
    class Local(object):
        pass

    setattr(Local, attributes.LIFECYCLE_ARGUMENT_NAME, attributes.LifeCycle.MULTI_INSTANCE)
    first = container.get(Local, None)
    second = container.get(Local, None)
    assert first is not second


def test_get_none_lifecycle_defaults_to_singleton_without_attribute(container):
    class Plain(object):
        pass

    assert container.get(Plain, None) is container.get(Plain, None)


# ---------- register / register_class ----------

def test_register_class_maps_base_to_implementation(container):
    container.register(Base, ImplA)
    instance = container.get(Base)
    assert type(instance) is ImplA


def test_register_class_replaces_previous_mapping(container):
    container.register(Base, ImplA)
    container.register(Base, ImplB)
    assert type(container.get(Base)) is ImplB


def test_register_instance_returns_registered_object(container):
    obj = Service()
    container.register(Service, obj)
    assert container.get(Service) is obj


def test_register_list_dispatches_to_register_many(container):
    container.register(Base, [ImplA, ImplB])
    result = container.get(List[Base])
    assert [type(item) for item in result] == [ImplA, ImplB]


# ---------- register_many ----------

def test_register_many_preserves_order(container):
    container.register_many(Base, [ImplB, ImplA])
    result = container.get(List[Base])
    assert [type(item) for item in result] == [ImplB, ImplA]


def test_register_many_list_is_cached_as_singleton(container):
    container.register_many(Base, [ImplA, ImplB])
    first = container.get(List[Base])
    second = container.get(List[Base])
    assert first is second
    assert first[0] is second[0]


def test_register_many_empty_list_returns_empty_list(container):
    container.register_many(Base, [])
    assert container.get(List[Base]) == []


def test_register_many_single_class(container):
    container.register_many(Base, [ImplA])
    result = container.get(List[Base])
    assert len(result) == 1
    assert type(result[0]) is ImplA


# ---------- _get_class_name ----------

def test_get_class_name_for_plain_class(container):
    assert IocContainer._get_class_name(Service) == "Service"


def test_get_class_name_for_generic_list(container):
    assert IocContainer._get_class_name(List[Base]) == "List[Base]"


# ---------- module-level functions ----------

def test_module_register_many_and_get(module_container):
    ioc_module.register_many(Base, [ImplA, ImplB])
    result = ioc_module.get(List[Base])
    assert [type(item) for item in result] == [ImplA, ImplB]


def test_module_register_instance_then_get(module_container):
    obj = Service()
    ioc_module.register(Service, obj)
    assert ioc_module.get(Service) is obj


def test_module_get_default_singleton(module_container):
    assert ioc_module.get(Service) is ioc_module.get(Service)
