import importlib
import sys
import types

import pytest

mpr121 = importlib.import_module("piripherals.mpr121")


class FakeDevice:
    def __init__(self):
        self.read_bytes = {}
        self.read_words = {}
        self.read_blocks = {}
        self.write_byte_calls = []
        self.write_word_calls = []
        self.write_block_calls = []
        self.default_read_byte = 0
        self.default_read_word = 0
        self.default_read_block = []

    def read_byte(self, reg):
        return self.read_bytes.get(reg, self.default_read_byte)

    def read_word(self, reg):
        return self.read_words.get(reg, self.default_read_word)

    def read_block(self, reg, length):
        value = self.read_blocks.get((reg, length), self.default_read_block)
        return list(value)

    def write_byte(self, reg, value):
        self.write_byte_calls.append((reg, value))

    def write_word(self, reg, value):
        self.write_word_calls.append((reg, value))

    def write_block(self, reg, values):
        self.write_block_calls.append((reg, list(values)))


class FakeBusFactory:
    def __init__(self, device):
        self.device_obj = device
        self.calls = []

    def __call__(self, bus_num):
        self.calls.append(bus_num)
        return self

    def device(self, addr):
        self.calls.append(("device", addr))
        return self.device_obj


def make_mpr(monkeypatch, device=None, handlers=False, setup=False, reset=False):
    if device is None:
        device = FakeDevice()
    factory = FakeBusFactory(device)
    monkeypatch.setattr(mpr121, "noop", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(mpr121, "not_raising", lambda f: f, raising=False)
    monkeypatch.setattr(mpr121, "IRQHandler", lambda irq, cb: None, raising=False)
    monkeypatch.setattr(mpr121, "Poller", lambda cb: None, raising=False)
    monkeypatch.setattr(mpr121, "Bus", factory, raising=False)
    obj = mpr121.MPR121(bus=1, addr=0x5a, irq=0, handlers=handlers, setup=setup, reset=reset)
    return obj, device, factory


def test_init_uses_bus_factory_when_bus_is_not_device(monkeypatch):
    obj, device, factory = make_mpr(monkeypatch)
    assert obj.addr == 0x5A
    assert factory.calls == [1, ("device", 0x5A)]
    assert obj._bus is device


def test_init_accepts_existing_bus_device(monkeypatch):
    monkeypatch.setattr(mpr121, "noop", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(mpr121, "not_raising", lambda f: f, raising=False)
    monkeypatch.setattr(mpr121, "IRQHandler", lambda irq, cb: None, raising=False)
    monkeypatch.setattr(mpr121, "Poller", lambda cb: None, raising=False)

    class BusLike:
        def __init__(self):
            self.device = lambda addr: FakeDevice()

    bus = BusLike()
    obj = mpr121.MPR121(bus=bus, handlers=False, setup=False, reset=False)
    assert obj._bus.__class__ is FakeDevice


def test_reset_writes_soft_reset_register(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    obj.reset()
    assert device.write_byte_calls == [(mpr121.SRESET, 0x63)]


def test_configure_writes_expected_ecr_value(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    obj.configure(cl=2, prox=3, touch=11)
    assert device.write_byte_calls[-1] == (mpr121.ECR, (2 << 6) | (3 << 4) | 11)


def test_touched_returns_mask_and_checks_failure_bits(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    device.read_words[mpr121.ETS] = 0x8123
    device.read_words[mpr121.OOR] = 0x0000
    assert obj.touched() == 0x123
    assert device.read_words[mpr121.ETS] == 0x8123


def test_touched_raises_on_overcurrent(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    device.read_words[mpr121.ETS] = 1 << 15
    device.read_words[mpr121.OOR] = 0
    with pytest.raises(Exception, match="overcurrent"):
        obj.touched()


def test_touched_raises_on_out_of_range(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    device.read_words[mpr121.ETS] = 0
    device.read_words[mpr121.OOR] = 1 << 15
    with pytest.raises(Exception, match="out of range"):
        obj.touched()


def test_out_of_range_returns_mask(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    device.read_words[mpr121.OOR] = 0xC321
    assert obj.out_of_range() == 0x1321


def test_out_of_range_raises_on_auto_config_failed(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    device.read_words[mpr121.OOR] = 1 << 15
    with pytest.raises(Exception, match="auto config failed"):
        obj.out_of_range()


def test_out_of_range_raises_on_auto_reconfig_failed(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    device.read_words[mpr121.OOR] = 1 << 14
    with pytest.raises(Exception, match="auto reconfig failed"):
        obj.out_of_range()


def test_electrode_data_decodes_all_channels(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    device.read_blocks[(mpr121.EFD, 2 * mpr121.NCH)] = []
    raw = []
    for i in range(mpr121.NCH):
        raw.extend([i, i + 0x10])
    device.read_blocks[(mpr121.EFD, 2 * mpr121.NCH)] = raw
    expected = [(i + 0x10) << 8 | i for i in range(mpr121.NCH)]
    assert obj.electrode_data() == expected


def test_baseline_reads_and_scales_values(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    device.read_blocks[(mpr121.EBL, mpr121.NCH)] = list(range(mpr121.NCH))
    assert obj.baseline() == [i << 2 for i in range(mpr121.NCH)]


def test_baseline_writes_correct_registers_for_touch_and_prox(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    obj.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20, prox=0)
    obj.baseline(rft=2, nhd=7, ncl=8, fdl=9, prox=1)
    assert device.write_block_calls[0] == (mpr121.MHD, [5, 1, 3, 20])
    assert device.write_block_calls[1] == (mpr121.MHDX + 8, [7, 8, 9])


def test_threshold_applies_to_all_channels_and_proximity(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    obj.threshold(50)
    assert len(device.write_byte_calls) == mpr121.NCH * 2
    assert device.write_byte_calls[0] == (mpr121.TTH, 50)
    assert device.write_byte_calls[1] == (mpr121.RTH, int(0.6 * 50))
    assert device.write_byte_calls[-2] == (mpr121.TTH + 2 * 12, 50)
    assert device.write_byte_calls[-1] == (mpr121.RTH + 2 * 12, int(0.6 * 50))


def test_debounce_defaults_release_to_touch(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    obj.debounce(touch=3)
    assert device.write_byte_calls == [(mpr121.DEB, (3 << 4) | 3)]


def test_filter_writes_combined_word(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    obj.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    lb = (1 << 6) | 30
    hb = (1 << 5) | (1 << 3) | 0
    assert device.write_word_calls == [(mpr121.AFE1, (hb << 8) | lb)]


def test_charge_updates_cdc_and_packed_cdt_bits(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    device.read_bytes[mpr121.CDT + 0] = 0xAB
    obj.charge(0, cdc=9, cdt=4)
    assert device.write_byte_calls[0] == (mpr121.CDC + 0, 9)
    assert device.write_byte_calls[1] == (mpr121.CDT + 0, (0xAB & ~0x0F) | 4)


def test_auto_config_writes_expected_registers(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    obj.auto_config(ace=1, are=0, bva=2, retry=3, afes=1, scts=1, acfie=0, arfie=1, oorie=0, usl=201, lsl=131, tl=181)
    lb = (1 << 6) | (3 << 4) | (2 << 2) | (0 << 1) | 1
    hb = (1 << 7) | (0 << 2) | (1 << 1) | 0
    assert device.write_word_calls[0] == (mpr121.ACNF_C0, (hb << 8) + lb)
    assert device.write_byte_calls[-3:] == [
        (mpr121.ACNF_USL, 201),
        (mpr121.ACNF_LSL, 131),
        (mpr121.ACNF_TL, 181),
    ]


def test_is_touched_reflects_cached_state(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    obj._touched = 0b1010
    assert obj.is_touched(1) is True
    assert obj.is_touched(0) is False
    assert obj.is_touched(3) is True


def test_handle_touch_supports_one_and_two_arg_handlers(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    events = []

    def h2(touched, channel):
        events.append(("two", touched, channel))

    def h1(touched):
        events.append(("one", touched))

    obj._handlers[1] = h2
    obj._handlers[2] = h1
    obj._handle_touch((1 << 1) | (1 << 2), (1 << 1) | (1 << 2))
    assert events == [("two", True, 1), ("one", True)]


def test_update_touch_state_calls_handler_only_when_state_changes(monkeypatch):
    obj, device, _ = make_mpr(monkeypatch)
    calls = []
    obj.touched = lambda raise_on_failure=1: 0b0011 if not calls else 0b0001
    obj._handlers[0] = lambda touched, channel=None: calls.append((channel, touched))
    obj._handlers[1] = lambda touched, channel=None: calls.append((channel, touched))

    obj.update_touch_state()
    obj.update_touch_state()
    assert calls == [(0, True), (1, True), (1, False)]
