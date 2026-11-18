import pytest

import piripherals.mpr121 as mpr121


class FakeDevice:
    def __init__(self):
        self.writes = []
        self.read_bytes = {}
        self.read_words = {}
        self.read_blocks = {}

    def write_byte(self, reg, value):
        self.writes.append(("write_byte", reg, value))
        self.read_bytes[reg] = value

    def write_word(self, reg, value):
        self.writes.append(("write_word", reg, value))
        self.read_words[reg] = value

    def write_block(self, reg, values):
        self.writes.append(("write_block", reg, list(values)))
        self.read_blocks[reg] = list(values)

    def read_byte(self, reg):
        return self.read_bytes.get(reg, 0)

    def read_word(self, reg):
        return self.read_words.get(reg, 0)

    def read_block(self, reg, n):
        if reg in self.read_blocks:
            data = self.read_blocks[reg]
            return data[:n] + [0] * max(0, n - len(data))
        return [0] * n


class FakeBusFactory:
    def __init__(self, device=None):
        self.calls = []
        self.device_obj = device or FakeDevice()

    def device(self, addr):
        self.calls.append(addr)
        return self.device_obj


class BusCtorStub:
    def __init__(self, device):
        self._device = device
        self.calls = []

    def __call__(self, busnum):
        self.calls.append(busnum)
        return self

    def device(self, addr):
        self.calls.append(("device", addr))
        return self._device


@pytest.fixture(autouse=True)
def patch_util_helpers(monkeypatch):
    monkeypatch.setattr(mpr121, "noop", lambda *a, **k: None)
    monkeypatch.setattr(mpr121, "not_raising", lambda handler: handler)
    monkeypatch.setattr(mpr121, "IRQHandler", lambda irq, cb: ("irq", irq, cb))
    monkeypatch.setattr(mpr121, "Poller", lambda cb: ("poller", cb))


def make_sensor(monkeypatch, device=None, bus_obj=None, **kwargs):
    device = device or FakeDevice()
    if bus_obj is None:
        bus_obj = FakeBusFactory(device)
        monkeypatch.setattr(mpr121, "Bus", lambda busnum: bus_obj)
    return mpr121.MPR121(bus=bus_obj if hasattr(bus_obj, "device") else 1, handlers=0, setup=0, reset=0, **kwargs), device, bus_obj


def test_init_uses_existing_bus_device(monkeypatch):
    device = FakeDevice()
    bus_obj = FakeBusFactory(device)
    sensor = mpr121.MPR121(bus=bus_obj, handlers=0, setup=0, reset=0)
    assert sensor._bus is device
    assert bus_obj.calls == [0x5A]
    assert sensor.addr == 0x5A


def test_init_wraps_integer_bus_with_Bus(monkeypatch):
    device = FakeDevice()
    stub = BusCtorStub(device)
    monkeypatch.setattr(mpr121, "Bus", stub)
    sensor = mpr121.MPR121(bus=7, handlers=0, setup=0, reset=0, addr=0x42)
    assert stub.calls == [7, ("device", 0x42)]
    assert sensor._bus is device
    assert sensor.addr == 0x42


def test_init_with_irq_registers_irqhandler(monkeypatch):
    device = FakeDevice()
    bus_obj = FakeBusFactory(device)
    created = []

    def fake_irq(irq, cb):
        created.append((irq, cb))
        return None

    monkeypatch.setattr(mpr121, "IRQHandler", fake_irq)
    mpr121.MPR121(bus=bus_obj, irq=4, handlers=1, setup=0, reset=0)
    assert len(created) == 1
    assert created[0][0] == 4
    assert callable(created[0][1])


def test_init_without_irq_uses_poller(monkeypatch):
    device = FakeDevice()
    bus_obj = FakeBusFactory(device)
    created = []

    def fake_poller(cb):
        created.append(cb)
        return None

    monkeypatch.setattr(mpr121, "Poller", fake_poller)
    mpr121.MPR121(bus=bus_obj, irq=0, handlers=1, setup=0, reset=0)
    assert len(created) == 1
    assert callable(created[0])


def test_init_with_reset_only_calls_reset(monkeypatch):
    device = FakeDevice()
    bus_obj = FakeBusFactory(device)
    sensor = mpr121.MPR121(bus=bus_obj, handlers=0, setup=0, reset=1)
    assert sensor._bus.writes == [("write_byte", mpr121.SRESET, 0x63)]


def test_update_touch_state_invokes_handler_on_change(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    sensor._handlers = [lambda touched, ch: events.append((touched, ch)) for _ in range(mpr121.NCH)]
    events = []
    sensor._handlers[1] = lambda touched, ch: events.append((touched, ch))
    monkeypatch.setattr(sensor, "touched", lambda raise_on_failure=1: 0b10)
    sensor.update_touch_state()
    assert sensor._touched == 0b10
    assert events == [(True, 1)]


def test_update_touch_state_no_change_no_handler_call(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    events = []
    sensor._handlers[0] = lambda *a: events.append(a)
    sensor._touched = 0b1
    monkeypatch.setattr(sensor, "touched", lambda raise_on_failure=1: 0b1)
    sensor.update_touch_state()
    assert events == []
    assert sensor._touched == 0b1


def test_handle_touch_falls_back_to_single_argument_handler(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    events = []

    def one_arg(touched):
        events.append(touched)

    sensor._handlers[2] = one_arg
    sensor._handle_touch(touched=0b100, changed=0b100)
    assert events == [True]


def test_is_touched_checks_bitmask(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    sensor._touched = 0b1010
    assert sensor.is_touched(1) is True
    assert sensor.is_touched(0) is False
    assert sensor.is_touched(3) is True


def test_on_touch_none_removes_handler(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    sensor.on_touch(5, None)
    assert sensor._handlers[5] is None or sensor._handlers[5] == mpr121.noop


def test_reset_writes_soft_reset(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    sensor.reset()
    assert device.writes == [("write_byte", mpr121.SRESET, 0x63)]


def test_configure_writes_ecr(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    sensor.configure(cl=2, prox=3, touch=12)
    assert device.writes == [("write_byte", mpr121.ECR, (2 << 6) | (3 << 4) | 12)]


def test_touched_returns_status_bits(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    device.read_words[mpr121.ETS] = 0xA123
    assert sensor.touched(raise_on_failure=0) == 0x0123


def test_touched_raises_on_overcurrent(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    device.read_words[mpr121.ETS] = (1 << 15) | 0x0007
    with pytest.raises(Exception, match="overcurrent"):
        sensor.touched()


def test_touched_raises_on_out_of_range(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    device.read_words[mpr121.ETS] = 0x0001
    monkeypatch.setattr(sensor, "out_of_range", lambda raise_on_failure=1: 0x0001)
    with pytest.raises(Exception, match="out of range"):
        sensor.touched()


def test_out_of_range_raises_auto_config_failed(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    device.read_words[mpr121.OOR] = (1 << 15) | 0x0001
    with pytest.raises(Exception, match="auto config failed"):
        sensor.out_of_range()


def test_out_of_range_raises_auto_reconfig_failed(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    device.read_words[mpr121.OOR] = (1 << 14) | 0x0001
    with pytest.raises(Exception, match="auto reconfig failed"):
        sensor.out_of_range()


def test_electrode_data_reads_and_decodes(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    device.read_blocks[mpr121.EFD] = []
    payload = []
    for i in range(mpr121.NCH):
        payload.extend([i, i + 1])
    device.read_blocks[mpr121.EFD] = payload
    assert sensor.electrode_data() == [((i + 1) << 8) + i for i in range(mpr121.NCH)]


def test_baseline_read_shifts_values(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    device.read_blocks[mpr121.EBL] = list(range(mpr121.NCH))
    assert sensor.baseline() == [i << 2 for i in range(mpr121.NCH)]


def test_baseline_write_regular_and_prox(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    sensor.baseline(rft=1, mhd=5, nhd=6, ncl=7, fdl=8, prox=0)
    sensor.baseline(rft=2, nhd=9, ncl=10, fdl=11, prox=1)
    assert device.writes[0] == ("write_block", mpr121.MHD + 4, [5, 6, 7, 8])
    assert device.writes[1] == ("write_block", mpr121.MHDX + 8, [9, 10, 11])


def test_threshold_applies_to_all_channels_with_default_release(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    sensor.threshold(100)
    assert len(device.writes) == mpr121.NCH * 2
    assert device.writes[0] == ("write_byte", mpr121.TTH, 100)
    assert device.writes[1] == ("write_byte", mpr121.RTH, 60)
    assert device.writes[-2] == ("write_byte", mpr121.TTH + 2 * 12, 100)
    assert device.writes[-1] == ("write_byte", mpr121.RTH + 2 * 12, 60)


def test_debounce_and_filter_and_auto_config(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    sensor.debounce(touch=3, release=5)
    sensor.filter(cdc=30, cdt=1, ffi=2, sfi=3, esi=4)
    sensor.auto_config(ace=1, are=0, bva=2, retry=3, afes=1, scts=1, acfie=0, arfie=1, oorie=0, usl=201, lsl=131, tl=181)

    assert ("write_byte", mpr121.DEB, (5 << 4) | 3) in device.writes
    assert ("write_word", mpr121.AFE1, (((1 << 5) | (3 << 3) | 4) << 8) | ((2 << 6) | 30)) in device.writes
    expected_lb = (1 << 6) | (3 << 4) | (2 << 2) | (0 << 1) | 1
    expected_hb = (1 << 7) | (0 << 2) | (1 << 1) | 0
    assert ("write_word", mpr121.ACNF_C0, (expected_hb << 8) + expected_lb) in device.writes
    assert ("write_byte", mpr121.ACNF_USL, 201) in device.writes
    assert ("write_byte", mpr121.ACNF_LSL, 131) in device.writes
    assert ("write_byte", mpr121.ACNF_TL, 181) in device.writes


def test_gpio_status_reads_byte(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    device.read_bytes[mpr121.GPIO_DAT] = 0xA5
    assert sensor.gpio_status() == 0xA5


def test_gpio_set_writes_set_or_clear(monkeypatch):
    sensor, device, _ = make_sensor(monkeypatch)
    sensor.gpio_set(4, True)
    sensor.gpio_set(11, False)
    assert device.writes == [
        ("write_byte", mpr121.GPIO_SET, 1 << 0),
        ("write_byte", mpr121.GPIO_CLR, 1 << 7),
    ]
