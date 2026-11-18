import importlib
import sys

import pytest

mpr121 = importlib.import_module("piripherals.mpr121")


class FakeDevice:
    def __init__(self, addr=0x5A):
        self.addr = addr
        self.read_byte_values = {}
        self.read_word_values = {}
        self.read_block_values = {}
        self.writes = []
        self.byte_reads = []
        self.word_reads = []
        self.block_reads = []

    def read_byte(self, reg):
        self.byte_reads.append(reg)
        return self.read_byte_values.get(reg, 0)

    def write_byte(self, reg, val):
        self.writes.append(("byte", reg, val))

    def read_word(self, reg):
        self.word_reads.append(reg)
        return self.read_word_values.get(reg, 0)

    def write_word(self, reg, val):
        self.writes.append(("word", reg, val))

    def read_block(self, reg, count):
        self.block_reads.append((reg, count))
        if reg in self.read_block_values:
            return list(self.read_block_values[reg])
        return [0] * count

    def write_block(self, reg, values):
        self.writes.append(("block", reg, list(values)))


class FakeBus:
    def __init__(self, busnum):
        self.busnum = busnum
        self.device_calls = []
        self.device_obj = FakeDevice()

    def device(self, addr):
        self.device_calls.append(addr)
        self.device_obj.addr = addr
        return self.device_obj


@pytest.fixture
def fake_bus(monkeypatch):
    bus = FakeBus(1)
    monkeypatch.setattr(mpr121, "noop", lambda *args, **kwargs: None)
    monkeypatch.setattr(mpr121, "IRQHandler", lambda irq, cb: None)
    monkeypatch.setattr(mpr121, "Poller", lambda cb: None)
    monkeypatch.setattr(mpr121, "Bus", lambda busnum: bus)
    return bus


def make_mpr(monkeypatch, device=None):
    monkeypatch.setattr(mpr121, "noop", lambda *args, **kwargs: None)
    monkeypatch.setattr(mpr121, "IRQHandler", lambda irq, cb: None)
    monkeypatch.setattr(mpr121, "Poller", lambda cb: None)
    if device is None:
        device = FakeDevice()
    bus = FakeBus(1)
    bus.device_obj = device
    monkeypatch.setattr(mpr121, "Bus", lambda busnum: bus)
    return mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0), device, bus


def test_init_uses_bus_device_when_bus_is_int(monkeypatch):
    device = FakeDevice()
    bus = FakeBus(2)
    bus.device_obj = device
    monkeypatch.setattr(mpr121, "noop", lambda *args, **kwargs: None)
    monkeypatch.setattr(mpr121, "IRQHandler", lambda irq, cb: None)
    monkeypatch.setattr(mpr121, "Poller", lambda cb: None)
    monkeypatch.setattr(mpr121, "Bus", lambda busnum: bus)

    m = mpr121.MPR121(bus=2, addr=0x5b, handlers=0, setup=0, reset=0)

    assert m.addr == 0x5b
    assert m._bus is device
    assert bus.device_calls == [0x5b]


def test_init_with_existing_bus_object_skips_bus_wrapper(monkeypatch):
    device = FakeDevice()
    fake_bus_obj = type("B", (), {"device": lambda self, addr: device})()
    monkeypatch.setattr(mpr121, "noop", lambda *args, **kwargs: None)
    monkeypatch.setattr(mpr121, "IRQHandler", lambda irq, cb: None)
    monkeypatch.setattr(mpr121, "Poller", lambda cb: None)

    m = mpr121.MPR121(bus=fake_bus_obj, handlers=0, setup=0, reset=0)

    assert m._bus is device


def test_init_calls_irqhandler_when_irq_and_handlers_enabled(monkeypatch):
    calls = []

    def fake_irqhandler(irq, cb):
        calls.append((irq, cb))

    monkeypatch.setattr(mpr121, "noop", lambda *args, **kwargs: None)
    monkeypatch.setattr(mpr121, "IRQHandler", fake_irqhandler)
    monkeypatch.setattr(mpr121, "Poller", lambda cb: None)
    monkeypatch.setattr(mpr121, "Bus", lambda busnum: FakeBus(busnum))

    mpr121.MPR121(bus=1, irq=4, handlers=1, setup=0, reset=0)

    assert len(calls) == 1
    assert calls[0][0] == 4
    assert callable(calls[0][1])


def test_init_calls_poller_when_no_irq_and_handlers_enabled(monkeypatch):
    calls = []

    def fake_poller(cb):
        calls.append(cb)

    monkeypatch.setattr(mpr121, "noop", lambda *args, **kwargs: None)
    monkeypatch.setattr(mpr121, "IRQHandler", lambda irq, cb: None)
    monkeypatch.setattr(mpr121, "Poller", fake_poller)
    monkeypatch.setattr(mpr121, "Bus", lambda busnum: FakeBus(busnum))

    mpr121.MPR121(bus=1, irq=0, handlers=1, setup=0, reset=0)

    assert len(calls) == 1
    assert callable(calls[0])


def test_init_calls_setup_when_enabled(monkeypatch):
    device = FakeDevice()
    bus = FakeBus(1)
    bus.device_obj = device
    monkeypatch.setattr(mpr121, "noop", lambda *args, **kwargs: None)
    monkeypatch.setattr(mpr121, "IRQHandler", lambda irq, cb: None)
    monkeypatch.setattr(mpr121, "Poller", lambda cb: None)
    monkeypatch.setattr(mpr121, "Bus", lambda busnum: bus)

    m = mpr121.MPR121(bus=1, handlers=0, setup=1, reset=0)

    assert m._bus is device
    assert any(w for w in device.writes if w[0] == "byte" and w[1] == mpr121.ECR)


def test_reset_writes_soft_reset(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)
    m.reset()
    assert ("byte", mpr121.SRESET, 0x63) in device.writes


def test_configure_writes_expected_ecr_value(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)
    m.configure(cl=2, prox=3, touch=12)
    assert ("byte", mpr121.ECR, (2 << 6) | (3 << 4) | 12) in device.writes


def test_touched_returns_mask_without_failure_checks(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)
    device.read_word_values[mpr121.ETS] = 0x9abc
    device.read_block_values[mpr121.EBL] = [0] * mpr121.NCH
    device.read_word_values[mpr121.OOR] = 0

    assert m.touched(raise_on_failure=0) == (0x9abc & 0x1fff)


def test_touched_raises_on_overcurrent(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)
    device.read_word_values[mpr121.ETS] = 0x8001

    with pytest.raises(Exception, match="overcurrent"):
        m.touched()


def test_touched_raises_on_out_of_range(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)
    device.read_word_values[mpr121.ETS] = 0x0001
    monkeypatch.setattr(m, "out_of_range", lambda raise_on_failure=1: 1)

    with pytest.raises(Exception, match="out of range"):
        m.touched()


def test_electrode_data_parses_block_into_13_values(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)
    data = []
    for i in range(mpr121.NCH):
        lo = i
        hi = 2 * i
        data.extend([lo, hi])
    device.read_block_values[mpr121.EFD] = data

    assert m.electrode_data() == [(2 * i << 8) + i for i in range(mpr121.NCH)]


def test_baseline_reads_and_shifts_values(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)
    device.read_block_values[mpr121.EBL] = list(range(mpr121.NCH))

    assert m.baseline() == [i << 2 for i in range(mpr121.NCH)]


def test_baseline_writes_rising_and_falling_regs(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)

    m.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20, prox=0)
    m.baseline(rft=1, mhd=6, nhd=2, ncl=4, fdl=21, prox=1)

    assert ("block", mpr121.MHD, [5, 1, 3, 20]) in device.writes
    assert ("block", mpr121.MHDX + 4, [6, 2, 4, 21]) in device.writes


def test_baseline_writes_touch_mode_short_form(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)

    m.baseline(rft=2, nhd=7, ncl=8, fdl=9, prox=0)

    assert ("block", mpr121.MHD + 8, [7, 8, 9]) in device.writes


def test_threshold_applies_default_release_and_all_channels(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)

    m.threshold(50)

    expected_release = int(0.6 * 50)
    for ch in range(mpr121.NCH):
        assert ("byte", mpr121.TTH + 2 * ch, 50) in device.writes
        assert ("byte", mpr121.RTH + 2 * ch, expected_release) in device.writes


def test_debounce_writes_encoded_value(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)

    m.debounce(touch=3, release=5)

    assert ("byte", mpr121.DEB, (5 << 4) | 3) in device.writes


def test_filter_writes_expected_word(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)

    m.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)

    lb = (1 << 6) | 30
    hb = (1 << 5) | (1 << 3) | 0
    assert ("word", mpr121.AFE1, (hb << 8) | lb) in device.writes


def test_charge_updates_cdc_and_cdt_bits(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)
    device.read_byte_values[mpr121.CDT + 0] = 0b11110000

    m.charge(channel=1, cdc=12, cdt=5)

    assert ("byte", mpr121.CDC + 1, 12) in device.writes
    assert ("byte", mpr121.CDT + 0, (0b11110000 & ~(0xF << 4)) | (5 << 4)) in device.writes


def test_auto_config_writes_control_and_limits(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)

    m.auto_config(ace=1, are=0, bva=2, retry=3, afes=1, scts=1, acfie=0, arfie=1, oorie=0, usl=201, lsl=131, tl=181)

    lb = (1 << 6) | (3 << 4) | (2 << 2) | (0 << 1) | 1
    hb = (1 << 7) | (0 << 2) | (1 << 1) | 0
    assert ("word", mpr121.ACNF_C0, (hb << 8) + lb) in device.writes
    assert ("byte", mpr121.ACNF_USL, 201) in device.writes
    assert ("byte", mpr121.ACNF_LSL, 131) in device.writes
    assert ("byte", mpr121.ACNF_TL, 181) in device.writes


def test_out_of_range_raises_on_failure_bits(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)
    device.read_word_values[mpr121.OOR] = 0x8001

    with pytest.raises(Exception, match="auto config failed"):
        m.out_of_range()


def test_gpio_status_reads_dat_byte(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)
    device.read_byte_values[mpr121.GPIO_DAT] = 0xAA

    assert m.gpio_status() == 0xAA


def test_gpio_set_writes_set_or_clear_register(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)

    m.gpio_set(4, True)
    m.gpio_set(11, False)

    assert ("byte", mpr121.GPIO_SET, 1 << 0) in device.writes
    assert ("byte", mpr121.GPIO_CLR, 1 << 7) in device.writes


def test_update_touch_state_invokes_handler_on_change(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)
    calls = []
    m._handlers[0] = lambda touched, ch=None: calls.append((touched, ch))
    m._touched = 0
    monkeypatch.setattr(m, "touched", lambda raise_on_failure=1: 1)

    m.update_touch_state()

    assert calls == [(True, 0)]


def test_handle_touch_falls_back_to_single_argument_handler(monkeypatch):
    m, device, _ = make_mpr(monkeypatch)
    calls = []

    def one_arg_handler(touched):
        calls.append(touched)

    m._handlers[2] = one_arg_handler
    m._handle_touch(touched=1 << 2, changed=1 << 2)

    assert calls == [True]
