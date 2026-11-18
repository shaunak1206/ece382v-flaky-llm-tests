import types

import pytest

import piripherals.mpr121 as mpr


class FakeDevice:
    def __init__(self):
        self.calls = []
        self.read_word_values = {}
        self.read_byte_values = {}
        self.read_block_values = {}

    def write_byte(self, reg, value):
        self.calls.append(("write_byte", reg, value))

    def write_word(self, reg, value):
        self.calls.append(("write_word", reg, value))

    def write_block(self, reg, values):
        self.calls.append(("write_block", reg, list(values)))

    def read_word(self, reg):
        self.calls.append(("read_word", reg))
        return self.read_word_values.get(reg, 0)

    def read_byte(self, reg):
        self.calls.append(("read_byte", reg))
        return self.read_byte_values.get(reg, 0)

    def read_block(self, reg, length):
        self.calls.append(("read_block", reg, length))
        return self.read_block_values.get((reg, length), [0] * length)


class FakeBusFactory:
    def __init__(self, device):
        self.device_obj = device
        self.args = []

    def __call__(self, bus):
        self.args.append(bus)
        return self

    def device(self, addr):
        self.args.append(("device", addr))
        return self.device_obj


@pytest.fixture(autouse=True)
def patch_util(monkeypatch):
    monkeypatch.setattr(mpr, "noop", lambda *a, **k: None)

    def not_raising(handler):
        return handler

    monkeypatch.setattr(mpr, "not_raising", not_raising)
    monkeypatch.setattr(mpr, "IRQHandler", lambda irq, cb: ("irq", irq, cb))
    monkeypatch.setattr(mpr, "Poller", lambda cb: ("poll", cb))
    monkeypatch.setattr(mpr, "fork", lambda fn: fn())
    monkeypatch.setattr(mpr, "on_change", lambda *a, **k: None)


def test_init_uses_existing_bus_device(monkeypatch):
    dev = FakeDevice()
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    assert m._bus is dev
    assert m.addr == 0x5A
    assert m._handlers == [mpr.noop] * mpr.NCH


def test_init_wraps_integer_bus_with_bus_factory(monkeypatch):
    dev = FakeDevice()
    factory = FakeBusFactory(dev)
    monkeypatch.setattr(mpr, "Bus", factory)
    m = mpr.MPR121(bus=1, handlers=0, setup=0, reset=0)
    assert m._bus is dev
    assert factory.args == [1, ("device", 0x5A)]
    assert m.addr == 0x5A


def test_init_installs_irq_handler_when_irq_given(monkeypatch):
    dev = FakeDevice()
    calls = []

    def irqhandler(irq, cb):
        calls.append((irq, cb))

    monkeypatch.setattr(mpr, "IRQHandler", irqhandler)
    mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), irq=4, handlers=1, setup=0, reset=0)
    assert len(calls) == 1
    assert calls[0][0] == 4
    assert callable(calls[0][1])


def test_init_installs_poller_when_no_irq(monkeypatch):
    dev = FakeDevice()
    calls = []

    def poller(cb):
        calls.append(cb)

    monkeypatch.setattr(mpr, "Poller", poller)
    mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), irq=0, handlers=1, setup=0, reset=0)
    assert len(calls) == 1
    assert callable(calls[0])


def test_init_calls_setup_when_enabled(monkeypatch):
    dev = FakeDevice()
    called = []

    class X(mpr.MPR121):
        def setup(self, **kwargs):
            called.append(kwargs)

    X(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=1, reset=0, foo=1)
    assert called == [{"reset": 0, "foo": 1}]


def test_init_calls_reset_when_setup_disabled_and_reset_enabled():
    dev = FakeDevice()
    mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=1)
    assert ("write_byte", mpr.SRESET, 0x63) in dev.calls


def test_update_touch_state_triggers_handler_on_change():
    dev = FakeDevice()
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    sensor._touched = 0
    sensor.touched = lambda raise_on_failure=1: 0b101
    seen = []
    sensor.on_touch(0, lambda touched, ch=None: seen.append((0, touched, ch)))
    sensor.on_touch(2, lambda touched, ch=None: seen.append((2, touched, ch)))
    sensor.update_touch_state()
    assert (0, True, 0) in seen
    assert (2, True, 2) in seen


def test_update_touch_state_does_not_call_handlers_when_unchanged():
    dev = FakeDevice()
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    sensor._touched = 0b11
    sensor.touched = lambda raise_on_failure=1: 0b11
    seen = []
    sensor.on_touch(0, lambda *a: seen.append(a))
    sensor.update_touch_state()
    assert seen == []


def test_is_touched_checks_bitmask():
    dev = FakeDevice()
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    sensor._touched = 0b1010
    assert sensor.is_touched(1) is True
    assert sensor.is_touched(0) is False


def test_on_touch_none_removes_handler():
    dev = FakeDevice()
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    sensor.on_touch(3, None)
    assert sensor._handlers[3] is mpr.noop


def test_touched_returns_status_bits_and_checks_failure_bits():
    dev = FakeDevice()
    dev.read_word_values[mpr.ETS] = 0x8005
    dev.read_word_values[mpr.OOR] = 0
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    assert sensor.touched(raise_on_failure=0) == 0x0005
    with pytest.raises(Exception, match="overcurrent"):
        sensor.touched()


def test_touched_raises_on_out_of_range_when_enabled():
    dev = FakeDevice()
    dev.read_word_values[mpr.ETS] = 0x0001
    dev.read_word_values[mpr.OOR] = 0x0001
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    with pytest.raises(Exception, match="out of range"):
        sensor.touched()


def test_out_of_range_handles_auto_config_failure_bits():
    dev = FakeDevice()
    dev.read_word_values[mpr.OOR] = 0x8000
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    with pytest.raises(Exception, match="auto config failed"):
        sensor.out_of_range()

    dev.read_word_values[mpr.OOR] = 0x4001
    with pytest.raises(Exception, match="auto reconfig failed"):
        sensor.out_of_range()


def test_electrode_data_reads_and_decodes_all_channels():
    dev = FakeDevice()
    raw = []
    for i in range(mpr.NCH):
        val = 100 + i
        raw.extend([val & 0xFF, (val >> 8) & 0xFF])
    dev.read_block_values[(mpr.EFD, 2 * mpr.NCH)] = raw
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    assert sensor.electrode_data() == [100 + i for i in range(mpr.NCH)]


def test_baseline_reads_and_writes_configuration():
    dev = FakeDevice()
    dev.read_block_values[(mpr.EBL, mpr.NCH)] = list(range(mpr.NCH))
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    assert sensor.baseline() == [x << 2 for x in range(mpr.NCH)]
    sensor.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    sensor.baseline(rft=2, nhd=9, ncl=8, fdl=7, prox=1)
    assert ("write_block", mpr.MHD, [5, 1, 3, 20]) in dev.calls
    assert ("write_block", mpr.MHDX + 8, [9, 8, 7]) in dev.calls


def test_threshold_applies_to_all_channels_when_channel_omitted():
    dev = FakeDevice()
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    sensor.threshold(50)
    for ch in range(mpr.NCH):
        assert ("write_byte", mpr.TTH + 2 * ch, 50) in dev.calls
        assert ("write_byte", mpr.RTH + 2 * ch, 30) in dev.calls


def test_debounce_and_filter_and_auto_config_write_expected_registers():
    dev = FakeDevice()
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    sensor.debounce(touch=3, release=5)
    sensor.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    sensor.auto_config(ace=1, are=1, bva=3, retry=2, afes=1, scts=0, acfie=1, arfie=1, oorie=1, usl=200, lsl=130, tl=180)
    assert ("write_byte", mpr.DEB, (5 << 4) | 3) in dev.calls
    assert ("write_word", mpr.AFE1, ((0x28 << 8) | 0x7E)) in dev.calls
    assert ("write_word", mpr.ACNF_C0, ((0x0E << 8) | 0x7F)) in dev.calls
    assert ("write_byte", mpr.ACNF_USL, 200) in dev.calls
    assert ("write_byte", mpr.ACNF_LSL, 130) in dev.calls
    assert ("write_byte", mpr.ACNF_TL, 180) in dev.calls


def test_charge_writes_current_and_updates_charge_time_bits():
    dev = FakeDevice()
    dev.read_byte_values[mpr.CDT + 0] = 0b10110011
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    sensor.charge(0, cdc=12, cdt=5)
    assert ("write_byte", mpr.CDC + 0, 12) in dev.calls
    assert ("write_byte", mpr.CDT + 0, (0b10110011 & ~0x0F) | 5) in dev.calls


def test_gpio_setup_and_gpio_set_use_expected_registers():
    dev = FakeDevice()
    dev.read_byte_values.update({
        mpr.GPIO_CTL0: 0,
        mpr.GPIO_CTL1: 0,
        mpr.GPIO_DIR: 0,
        mpr.GPIO_EN: 0,
    })
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    sensor.gpio_setup(channel=4, output=1, mode=2, enable=1)
    sensor.gpio_set(7, True)
    sensor.gpio_set(8, False)
    assert ("write_byte", mpr.GPIO_CTL0, 1) in dev.calls
    assert ("write_byte", mpr.GPIO_CTL1, 0) in dev.calls
    assert ("write_byte", mpr.GPIO_DIR, 1) in dev.calls
    assert ("write_byte", mpr.GPIO_EN, 1) in dev.calls
    assert ("write_byte", mpr.GPIO_SET, 1 << 3) in dev.calls
    assert ("write_byte", mpr.GPIO_CLR, 1 << 4) in dev.calls


def test_configure_and_reset_write_expected_registers():
    dev = FakeDevice()
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    sensor.reset()
    sensor.configure(cl=3, prox=2, touch=12)
    assert ("write_byte", mpr.SRESET, 0x63) in dev.calls
    assert ("write_byte", mpr.ECR, (3 << 6) | (2 << 4) | 12) in dev.calls


def test_setup_sequence_programs_expected_registers():
    dev = FakeDevice()
    sensor = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), handlers=0, setup=0, reset=0)
    sensor.setup(reset=1, channels=12, prox=0, threshold=50, debounce=2, auto_config=1)
    assert ("write_byte", mpr.SRESET, 0x63) in dev.calls
    assert ("write_byte", mpr.ECR, 0) in dev.calls
    assert ("write_word", mpr.AFE1, ((0x21 << 8) | 30)) in dev.calls
    assert ("write_byte", mpr.TTH + 2 * 12, 50 >> 2) in dev.calls
    assert ("write_byte", mpr.DEB, (2 << 4) | 2) in dev.calls
    assert ("write_byte", mpr.ECR, 12) in dev.calls
