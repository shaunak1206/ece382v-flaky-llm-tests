import types
import pytest

import piripherals.mpr121 as mpr


class FakeDevice:
    def __init__(self, reads=None):
        self.reads = list(reads or [])
        self.writes = []
        self.memory = {}

    def read_word(self, reg):
        self.writes.append(("read_word", reg))
        if self.reads:
            return self.reads.pop(0)
        return self.memory.get(("word", reg), 0)

    def read_block(self, reg, size):
        self.writes.append(("read_block", reg, size))
        return list(self.memory.get(("block", reg, size), [0] * size))

    def read_byte(self, reg):
        self.writes.append(("read_byte", reg))
        return self.memory.get(("byte", reg), 0)

    def write_byte(self, reg, val):
        self.writes.append(("write_byte", reg, val))
        self.memory[("byte", reg)] = val

    def write_word(self, reg, val):
        self.writes.append(("write_word", reg, val))
        self.memory[("word", reg)] = val

    def write_block(self, reg, vals):
        self.writes.append(("write_block", reg, list(vals)))
        self.memory[("block", reg, len(vals))] = list(vals)


class FakeBus:
    def __init__(self, device=None):
        self._device = device or FakeDevice()
        self.calls = []

    def device(self, addr):
        self.calls.append(("device", addr))
        return self._device


@pytest.fixture
def bus_and_device():
    dev = FakeDevice()
    bus = FakeBus(dev)
    return bus, dev


def test_init_uses_existing_bus_device_and_sets_defaults(monkeypatch, bus_and_device):
    bus, dev = bus_and_device
    monkeypatch.setattr(mpr, "IRQHandler", lambda *a, **k: pytest.fail("IRQHandler should not be called"))
    monkeypatch.setattr(mpr, "Poller", lambda *a, **k: pytest.fail("Poller should not be called"))
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0, addr=0x5b)
    assert obj.addr == 0x5b
    assert obj._bus is dev
    assert obj._touched == 0
    assert len(obj._handlers) == mpr.NCH


def test_init_invokes_irq_handler_when_irq_enabled(monkeypatch, bus_and_device):
    bus, _ = bus_and_device
    called = {}

    def fake_irq(pin, cb):
        called["pin"] = pin
        called["cb"] = cb

    monkeypatch.setattr(mpr, "IRQHandler", fake_irq)
    monkeypatch.setattr(mpr, "Poller", lambda *a, **k: pytest.fail("Poller should not be called"))
    mpr.MPR121(bus=bus, irq=4, handlers=1, setup=0, reset=0)
    assert called["pin"] == 4
    assert callable(called["cb"])


def test_init_invokes_poller_when_no_irq(monkeypatch, bus_and_device):
    bus, _ = bus_and_device
    called = {}

    def fake_poller(cb):
        called["cb"] = cb

    monkeypatch.setattr(mpr, "IRQHandler", lambda *a, **k: pytest.fail("IRQHandler should not be called"))
    monkeypatch.setattr(mpr, "Poller", fake_poller)
    mpr.MPR121(bus=bus, irq=0, handlers=1, setup=0, reset=0)
    assert callable(called["cb"])


def test_reset_writes_soft_reset_register(bus_and_device):
    bus, dev = bus_and_device
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    obj.reset()
    assert ("write_byte", mpr.SRESET, 0x63) in dev.writes


def test_configure_packs_bits_correctly(bus_and_device):
    bus, dev = bus_and_device
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    obj.configure(cl=3, prox=2, touch=9)
    assert ("write_byte", mpr.ECR, (3 << 6) | (2 << 4) | 9) in dev.writes


def test_touched_returns_mask_and_checks_failure_bits(monkeypatch, bus_and_device):
    bus, dev = bus_and_device
    dev.reads = [0x1234]
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    monkeypatch.setattr(obj, "out_of_range", lambda raise_on_failure=1: 0)
    assert obj.touched() == 0x1234 & 0x1FFF


def test_touched_raises_on_overcurrent(bus_and_device):
    bus, dev = bus_and_device
    dev.reads = [0x8001]
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    with pytest.raises(Exception, match="overcurrent"):
        obj.touched()


def test_touched_raises_on_out_of_range(monkeypatch, bus_and_device):
    bus, dev = bus_and_device
    dev.reads = [0x0001]
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    monkeypatch.setattr(obj, "out_of_range", lambda raise_on_failure=1: 1)
    with pytest.raises(Exception, match="out of range"):
        obj.touched()


def test_out_of_range_returns_mask(bus_and_device):
    bus, dev = bus_and_device
    dev.reads = [0x3456]
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    assert obj.out_of_range() == 0x3456 & 0x1FFF


def test_out_of_range_raises_on_auto_config_failed(bus_and_device):
    bus, dev = bus_and_device
    dev.reads = [0x8000]
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    with pytest.raises(Exception, match="auto config failed"):
        obj.out_of_range()


def test_out_of_range_raises_on_auto_reconfig_failed(bus_and_device):
    bus, dev = bus_and_device
    dev.reads = [0x4000]
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    with pytest.raises(Exception, match="auto reconfig failed"):
        obj.out_of_range()


def test_electrode_data_decodes_block_into_words(bus_and_device):
    bus, dev = bus_and_device
    data = []
    for i in range(mpr.NCH):
        data.extend([i, i + 1])
    dev.memory[("block", mpr.EFD, 2 * mpr.NCH)] = data
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    assert obj.electrode_data() == [(i + 1) << 8 | i for i in range(mpr.NCH)]


def test_baseline_reads_and_shifts_values(bus_and_device):
    bus, dev = bus_and_device
    dev.memory[("block", mpr.EBL, mpr.NCH)] = list(range(mpr.NCH))
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    assert obj.baseline() == [i << 2 for i in range(mpr.NCH)]


def test_baseline_write_mode_uses_correct_registers(bus_and_device):
    bus, dev = bus_and_device
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    obj.baseline(rft=2, nhd=7, ncl=8, fdl=9, prox=1)
    assert ("write_block", mpr.MHDX + 8, [7, 8, 9]) in dev.writes
    dev.writes.clear()
    obj.baseline(rft=1, mhd=5, nhd=6, ncl=7, fdl=8, prox=0)
    assert ("write_block", mpr.MHD + 4, [5, 6, 7, 8]) in dev.writes


def test_threshold_applies_default_release_and_all_channels(bus_and_device):
    bus, dev = bus_and_device
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    obj.threshold(50)
    expected_release = int(0.6 * 50)
    for i in range(mpr.NCH):
        assert ("write_byte", mpr.TTH + 2 * i, 50) in dev.writes
        assert ("write_byte", mpr.RTH + 2 * i, expected_release) in dev.writes


def test_debounce_packs_touch_and_release(bus_and_device):
    bus, dev = bus_and_device
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    obj.debounce(touch=3, release=5)
    assert ("write_byte", mpr.DEB, (5 << 4) | 3) in dev.writes


def test_filter_packs_word_correctly(bus_and_device):
    bus, dev = bus_and_device
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    obj.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    lb = (1 << 6) | 30
    hb = (1 << 5) | (1 << 3) | 0
    assert ("write_word", mpr.AFE1, (hb << 8) | lb) in dev.writes


def test_charge_updates_current_and_time_bits(bus_and_device):
    bus, dev = bus_and_device
    dev.memory[("byte", mpr.CDT)] = 0xAA
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    obj.charge(channel=1, cdc=12, cdt=7)
    assert ("write_byte", mpr.CDC + 1, 12) in dev.writes
    assert ("write_byte", mpr.CDT + 0, (0xAA & ~(0xF << 4)) | (7 << 4)) in dev.writes


def test_auto_config_writes_registers(bus_and_device):
    bus, dev = bus_and_device
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    obj.auto_config(ace=1, are=1, bva=3, retry=2, afes=1, scts=0, acfie=1, arfie=1, oorie=1, usl=201, lsl=131, tl=181)
    lb = (1 << 6) | (2 << 4) | (3 << 2) | (1 << 1) | 1
    hb = (0 << 7) | (1 << 2) | (1 << 1) | 1
    assert ("write_word", mpr.ACNF_C0, (hb << 8) + lb) in dev.writes
    assert ("write_byte", mpr.ACNF_USL, 201) in dev.writes
    assert ("write_byte", mpr.ACNF_LSL, 131) in dev.writes
    assert ("write_byte", mpr.ACNF_TL, 181) in dev.writes


def test_is_touched_reflects_cached_state(bus_and_device):
    bus, _ = bus_and_device
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    obj._touched = (1 << 2) | (1 << 5)
    assert obj.is_touched(2) is True
    assert obj.is_touched(3) is False
    assert obj.is_touched(5) is True


def test_on_touch_wraps_handler_and_uses_channel_when_supported(bus_and_device):
    bus, _ = bus_and_device
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    calls = []

    def handler(touched, channel=None):
        calls.append((touched, channel))

    obj.on_touch(3, handler)
    obj._handle_touch(touched=(1 << 3), changed=(1 << 3))
    assert calls == [(True, 3)]


def test_update_touch_state_invokes_handler_on_change(monkeypatch, bus_and_device):
    bus, _ = bus_and_device
    obj = mpr.MPR121(bus=bus, handlers=0, setup=0, reset=0)
    obj._handlers[1] = lambda touched, channel=None: obj._handlers.append((touched, channel))
    monkeypatch.setattr(obj, "touched", lambda raise_on_failure=1: (1 << 1))
    obj.update_touch_state()
    assert obj._touched == (1 << 1)
    assert (True, 1) in obj._handlers
