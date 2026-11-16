import pytest
from typing import List

from cynergy import attributes
from cynergy.attributes import LifeCycle
from cynergy.container import IocContainer


class Base(object):
    pass


class Impl(Base):
    pass


class OtherImpl(Base):
    pass


class Greeter(object):
    def greet(self):
        return "hi"


class Dep(object):
    pass


class Consumer(object):
    def __init__(self, dep: Dep):
        self.dep = dep


class Holder(object):
    def __init__(self, base: Base):
        self.base = base


class WithDefault(object):
    def __init__(self, value=42):
        self.value = value


class NeedsStr(object):
    def __init__(self, name: str):
        self.name = name


class ServiceA(object):
    def __init__(self, dep: Dep):
        self.dep = dep


class ServiceB(object):
    def __init__(self, dep: Dep):
        self.dep = dep


def test_get_resolves_unregistered_concrete_class():
    container = IocContainer(None)
    obj = container.get(Impl)
    assert isinstance(obj, Impl)


def test_get_default_lifecycle_is_singleton():
    container = IocContainer(None)
    first = container.get(Impl)
    second = container.get(Impl)
    assert first is second


def test_get_explicit_singleton_returns_same_instance():
    container = IocContainer(None)
    first = container.get(Impl, LifeCycle.SINGLETON)
    second = container.get(Impl, LifeCycle.SINGLETON)
    assert first is second


def test_get_multi_instance_returns_new_instances():
    container = IocContainer(None)
    first = container.get(Impl, LifeCycle.MULTI_INSTANCE)
    second = container.get(Impl, LifeCycle.MULTI_INSTANCE)
    assert first is not second
    assert isinstance(first, Impl)
    assert isinstance(second, Impl)


def test_get_with_none_lifecycle_defaults_to_singleton():
    container = IocContainer(None)
    first = container.get(Impl, None)
    second = container.get(Impl, None)
    assert first is second


def test_get_with_none_lifecycle_honors_class_attribute():
    class Transient(object):
        pass

    setattr(Transient, attributes.LIFECYCLE_ARGUMENT_NAME, LifeCycle.MULTI_INSTANCE)
    container = IocContainer(None)
    first = container.get(Transient, None)
    second = container.get(Transient, None)
    assert first is not second


def test_get_unsupported_lifecycle_raises_not_implemented():
    container = IocContainer(None)
    with pytest.raises(NotImplementedError):
        container.get(Impl, "unsupported")


def test_register_class_maps_base_to_implementation():
    container = IocContainer(None)
    container.register(Base, Impl)
    obj = container.get(Base)
    assert type(obj) is Impl


def test_register_class_later_registration_overrides_previous():
    container = IocContainer(None)
    container.register(Base, Impl)
    container.register(Base, OtherImpl)
    obj = container.get(Base)
    assert type(obj) is OtherImpl


def test_register_instance_returns_registered_object():
    container = IocContainer(None)
    greeter = Greeter()
    container.register(Greeter, greeter)
    assert container.get(Greeter) is greeter


def test_register_instance_is_returned_on_repeated_get():
    container = IocContainer(None)
    greeter = Greeter()
    container.register(Greeter, greeter)
    assert container.get(Greeter) is container.get(Greeter)


def test_register_many_maps_list_key_to_classes():
    container = IocContainer(None)
    container.register_many(Base, [Impl])
    result = container.get(List[Base])
    assert isinstance(result, list)
    assert len(result) == 1
    assert type(result[0]) is Impl


def test_register_with_list_returns_instances_of_each_class():
    container = IocContainer(None)
    container.register(Base, [Impl, OtherImpl])
    result = container.get(List[Base])
    assert [type(item) for item in result] == [Impl, OtherImpl]


def test_register_many_list_is_cached_singleton():
    container = IocContainer(None)
    container.register_many(Base, [Impl, OtherImpl])
    first = container.get(List[Base])
    second = container.get(List[Base])
    assert first is second


def test_register_many_items_shared_with_direct_get():
    container = IocContainer(None)
    container.register_many(Base, [Impl, OtherImpl])
    items = container.get(List[Base])
    assert container.get(Impl) is items[0]
    assert container.get(OtherImpl) is items[1]


def test_get_injects_dependency_by_annotation():
    container = IocContainer(None)
    consumer = container.get(Consumer)
    assert isinstance(consumer.dep, Dep)


def test_get_injects_dependency_using_registered_mapping():
    container = IocContainer(None)
    container.register(Base, Impl)
    holder = container.get(Holder)
    assert type(holder.base) is Impl


def test_get_uses_default_argument_when_not_registered():
    container = IocContainer(None)
    obj = container.get(WithDefault)
    assert obj.value == 42


def test_get_primitive_argument_without_mapping_raises_type_error():
    container = IocContainer(None)
    with pytest.raises(TypeError):
        container.get(NeedsStr)


def test_singleton_dependency_is_shared_between_consumers():
    container = IocContainer(None)
    a = container.get(ServiceA)
    b = container.get(ServiceB)
    assert a.dep is b.dep


def test_containers_do_not_share_registrations():
    first = IocContainer(None)
    second = IocContainer(None)
    first.register(Base, Impl)
    obj = second.get(Base)
    assert type(obj) is Base
