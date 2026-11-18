import importlib
from types import SimpleNamespace

import pytest

import piripherals.mpr121 as mpr121


class FakeDevice:
    def __init__(self):
        self.writes = []
        self.read_bytes = {}
        self.read_words = {}
        self.read_blocks = {}

    def write_byte(self, reg, value):
        self.writes.append(("byte", reg, value))

    def write_word(self, reg, value):
        self.writes.append(("word", reg, value))

    def write_block(self, reg, values):
        self.writes.append(("block", reg, list(values)))

    def read_byte(self, reg):
        return self.read_bytes.get(reg, 0)

    def read_word(self, reg):
        return self.read_words.get(reg, 0)

    def read_block(self, reg, length):
        value = self.read_blocks.get((reg, length))
        if value is None:
            return [0] * length
        return list(value)


class FakeBusFactory:
    def __init__(self, device):
        self.device_obj = device
        self.calls = []

    def device(self, addr):
        self.calls.append(addr)
        return self.device_obj


@pytest.fixture
def device():
    return FakeDevice()


@pytest.fixture
def mpr(monkeypatch, device):
    # prevent IRQ/poller side effects
    monkeypatch.setattr(mpr121, "IRQHandler", lambda *a, **k: None)
    monkeypatch.setattr(mpr121, "Poller", lambda *a, **k: None)
    bus = FakeBusFactory(device)
    obj = mpr121.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    return obj


def test_init_uses_bus_device_and_sets_defaults(monkeypatch, device):
    monkeypatch.setattr(mpr121, "IRQHandler", lambda *a, **k: None)
    monkeypatch.setattr(mpr121, "Poller", lambda *a, **k: None)
    bus = FakeBusFactory(device)
    obj = mpr121.MPR121(bus=bus, addr=0x5b, handlers=0, setup=0, reset=0)
    assert obj.addr == 0x5b
    assert obj._bus is device
    assert bus.calls == [0x5b]
    assert len(obj._handlers) == mpr121.NCH
    assert obj._touched == 0
    assert obj.overcurrent is False


def test_init_registers_irq_handler_when_irq_enabled(monkeypatch, device):
    calls = []

    monkeypatch.setattr(mpr121, "Poller", lambda *a, **k: None)

    def fake_irq_handler(irq, cb):
        calls.append((irq, cb))

    monkeypatch.setattr(mpr121, "IRQHandler", fake_irq_handler)
    bus = FakeBusFactory(device)
    mpr121.MPR121(bus=bus, irq=4, handlers=1, setup=0, reset=0)
    assert calls and calls[0][0] == 4
    assert callable(calls[0][1])


def test_init_registers_poller_when_no_irq(monkeypatch, device):
    calls = []

    def fake_poller(cb):
        calls.append(cb)

    monkeypatch.setattr(mpr121, "IRQHandler", lambda *a, **k: None)
    monkeypatch.setattr(mpr121, "Poller", fake_poller)
    bus = FakeBusFactory(device)
    mpr121.MPR121(bus=bus, irq=0, handlers=1, setup=0, reset=0)
    assert len(calls) == 1
    assert callable(calls[0])


def test_reset_writes_soft_reset(device, mpr):
    mpr.reset()
    assert device.writes == [("byte", mpr121.SRESET, 0x63)]


def test_configure_writes_expected_ecr_value(device, mpr):
    mpr.configure(cl=3, prox=2, touch=12)
    assert device.writes[-1] == ("byte", mpr121.ECR, (3 << 6) | (2 << 4) | 12)


def test_touched_returns_mask_and_checks_failure_bits(mpr, device):
    device.read_words[mpr121.ETS] = 0x8005
    device.read_words[mpr121.OOR] = 0x0000
    assert mpr.touched(raise_on_failure=0) == 0x0005
    with pytest.raises(Exception, match="overcurrent"):
        mpr.touched()


def test_touched_raises_on_out_of_range_when_requested(mpr, device):
    device.read_words[mpr121.ETS] = 0x0007
    device.read_words[mpr121.OOR] = 0x0001
    with pytest.raises(Exception, match="out of range"):
        mpr.touched()


def test_out_of_range_returns_mask_and_raises_on_auto_config_failure(mpr, device):
    device.read_words[mpr121.OOR] = 0x0009
    assert mpr.out_of_range(raise_on_failure=0) == 0x0009
    device.read_words[mpr121.OOR] = 0x8000
    with pytest.raises(Exception, match="auto config failed"):
        mpr.out_of_range()


def test_out_of_range_raises_on_auto_reconfig_failure(mpr, device):
    device.read_words[mpr121.OOR] = 0x4000
    with pytest.raises(Exception, match="auto reconfig failed"):
        mpr.out_of_range()


def test_electrode_data_reads_and_combines_bytes(mpr, device):
    device.read_blocks[(mpr121.EFD, 2 * mpr121.NCH)] = list(range(2 * mpr121.NCH))
    data = mpr.electrode_data()
    assert len(data) == mpr121.NCH
    assert data[0] == (1 << 8) + 0
    assert data[1] == (3 << 8) + 2
    assert data[-1] == ((2 * mpr121.NCH - 1) << 8) + (2 * mpr121.NCH - 2)


def test_baseline_read_mode_shifts_each_byte(mpr, device):
    device.read_blocks[(mpr121.EBL, mpr121.NCH)] = list(range(mpr121.NCH))
    assert mpr.baseline() == [b << 2 for b in range(mpr121.NCH)]


def test_baseline_write_mode_uses_correct_registers(mpr, device):
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20, prox=0)
    mpr.baseline(rft=2, mhd=0, nhd=9, ncl=8, fdl=7, prox=1)
    assert device.writes[0] == ("block", mpr121.MHD, [5, 1, 3, 20])
    assert device.writes[1] == ("block", mpr121.MHDX + 8, [9, 8, 7])


def test_threshold_applies_to_all_channels_with_default_release(mpr, device):
    mpr.threshold(50)
    # 13 channels * 2 writes each
    assert len(device.writes) == mpr121.NCH * 2
    assert device.writes[0] == ("byte", mpr121.TTH, 50)
    assert device.writes[1] == ("byte", mpr121.RTH, int(0.6 * 50))
    assert device.writes[-2] == ("byte", mpr121.TTH + 2 * 12, 50)
    assert device.writes[-1] == ("byte", mpr121.RTH + 2 * 12, int(0.6 * 50))


def test_debounce_uses_touch_when_release_missing(mpr, device):
    mpr.debounce(touch=3)
    assert device.writes[-1] == ("byte", mpr121.DEB, (3 << 4) | 3)


def test_filter_writes_word_with_expected_bit_packing(mpr, device):
    mpr.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    lb = (1 << 6) | 30
    hb = (1 << 5) | (1 << 3) | 0
    assert device.writes[-1] == ("word", mpr121.AFE1, (hb << 8) | lb)


def test_charge_updates_current_and_charge_time_registers(mpr, device):
    device.read_bytes[mpr121.CDT + 0] = 0xAB
    mpr.charge(channel=1, cdc=22, cdt=7)
    assert device.writes[0] == ("byte", mpr121.CDC + 1, 22)
    assert device.writes[1] == ("byte", mpr121.CDT + 0, (0xAB & ~(0xF << 4)) | (7 << 4))


def test_auto_config_writes_control_and_limits(mpr, device):
    mpr.auto_config(ace=1, are=0, bva=2, retry=3, afes=1, scts=1, acfie=0, arfie=1, oorie=0, usl=201, lsl=131, tl=181)
    lb = (1 << 6) | (3 << 4) | (2 << 2) | (0 << 1) | 1
    hb = (1 << 7) | (0 << 2) | (1 << 1) | 0
    assert device.writes[0] == ("word", mpr121.ACNF_C0, (hb << 8) + lb)
    assert ("byte", mpr121.ACNF_USL, 201) in device.writes
    assert ("byte", mpr121.ACNF_LSL, 131) in device.writes
    assert ("byte", mpr121.ACNF_TL, 181) in device.writes


def test_gpio_status_reads_data_byte(mpr, device):
    device.read_bytes[mpr121.GPIO_DAT] = 0xA5
    assert mpr.gpio_status() == 0xA5


def test_gpio_set_high_and_low_use_correct_registers(mpr, device):
    mpr.gpio_set(4, True)
    mpr.gpio_set(7, False)
    assert device.writes[0] == ("byte", mpr121.GPIO_SET, 1 << 0)
    assert device.writes[1] == ("byte", mpr121.GPIO_CLR, 1 << 3)


def test_is_touched_reflects_internal_state(mpr):
    mpr._touched = (1 << 2) | (1 << 5)
    assert mpr.is_touched(2) is True
    assert mpr.is_touched(5) is True
    assert mpr.is_touched(1) is False
