import importlib
import sys

import pytest

import uhashring
import uhashring.ring as ring_mod


def test_package_exports_hashring_name():
    assert "HashRing" in uhashring.__all__


def test_package_exports_monkey_name():
    assert "monkey" in uhashring.__all__


def test_package_has_hashring_attribute():
    assert hasattr(uhashring, "HashRing")


def test_package_hashring_is_ring_hashring():
    assert uhashring.HashRing is ring_mod.HashRing


def test_import_from_package_matches_attribute():
    namespace = {}
    exec("from uhashring import HashRing", namespace)
    assert namespace["HashRing"] is uhashring.HashRing


def test_import_module_returns_same_hashring_class():
    mod = importlib.import_module("uhashring")
    assert mod.HashRing is uhashring.HashRing


def test_reimport_does_not_change_hashring_object():
    first = uhashring.HashRing
    second = importlib.reload(uhashring).HashRing
    assert first is second


def test_reloading_package_keeps_all_same():
    original_all = list(uhashring.__all__)
    reloaded = importlib.reload(uhashring)
    assert reloaded.__all__ == original_all


def test_reloading_package_after_monkeypatching_ring_hashring_updates_export(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ring_mod, "HashRing", sentinel)
    reloaded = importlib.reload(uhashring)
    assert reloaded.HashRing is sentinel


def test_package_attribute_tracks_ring_module_after_reload(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ring_mod, "HashRing", sentinel)
    reloaded = importlib.reload(uhashring)
    assert reloaded.HashRing is ring_mod.HashRing


def test_star_import_brings_hashring_into_namespace():
    namespace = {}
    exec("from uhashring import *", namespace)
    assert namespace["HashRing"] is uhashring.HashRing


def test_package_hashring_name_is_bound_in_globals():
    assert uhashring.__dict__["HashRing"] is uhashring.HashRing
