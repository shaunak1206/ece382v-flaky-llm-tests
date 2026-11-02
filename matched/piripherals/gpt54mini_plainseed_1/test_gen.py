import pytest

import piripherals.mpr121 as mpr_mod


class FakeDevice:
    def __init__(self):
        self.read_byte_values = {}
        self.read_word_values = {}
        self.read_block_values = {}
        self.writes = []
        self.read_byte_calls = []
        self.read_word_calls = []
        self.read_block_calls = []

    def read_byte(self, reg):
        self.read_byte_calls.append(reg)
        return self.read_byte_values.get(reg, 0)

    def write_byte(self, reg, value):
        self.writes.append(("byte", reg, value))
        self.read_byte_values[reg] = value

    def read_word(self, reg):
        self.read_word_calls.append(reg)
        return self.read_word_values.get(reg, 0)

    def write_word(self, reg, value):
        self.writes.append(("word", reg, value))
        self.read_word_values[reg] = value

    def read_block(self, reg, count):
        self.read_block_calls.append((reg, count))
        return list(self.read_block_values.get((reg, count), [0] * count))

    def write_block(self, reg, values):
        self.writes.append(("block", reg, list(values)))
        self.read_block_values[(reg, len(values))] = list(values)


class FakeBus:
    def __init__(self, device):
        self._device = device

    def device(self, addr):
        self.addr = addr
        return self._device


@pytest.fixture
def device():
    return FakeDevice()


@pytest.fixture
def bus_factory(device):
    return FakeBus(device)


@pytest.fixture(autouse=True)
def patch_helpers(monkeypatch):
    # Make initialization and handler registration deterministic and side-effect free.
    monkeypatch.setattr(mpr_mod, "noop", lambda *a, **k: None)

    def not_raising(handler):
        return handler

    monkeypatch.setattr(mpr_mod, "not_raising", not_raising)
    monkeypatch.setattr(mpr_mod, "IRQHandler", lambda irq, cb: ("irq", irq, cb))
    monkeypatch.setattr(mpr_mod, "Poller", lambda cb: ("poller", cb))


def test_init_uses_existing_bus_device_and_setup(bus_factory, device, monkeypatch):
    calls = []

    def fake_setup(self, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(mpr_mod.MPR121, "setup", fake_setup)
    m = mpr_mod.MPR121(bus=bus_factory, addr=0x5B, handlers=0, setup=1, reset=1, foo=1)

    assert m._bus is device
    assert m.addr == 0x5B
    assert calls == [{"reset": 1, "foo": 1}]


def test_init_constructs_bus_when_needed(monkeypatch, device):
    created = []

    class BusCls:
        def __init__(self, busnum):
            created.append(busnum)

        def device(self, addr):
            created.append(("device", addr))
            return device

    monkeypatch.setattr(mpr_mod, "Bus", BusCls, raising=False)
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)

    mpr_mod.MPR121(bus=2, addr=0x5A, handlers=0, setup=1, reset=0)
    assert created == [2, ("device", 0x5A)]


def test_init_registers_irq_handler_when_irq_enabled(bus_factory, device, monkeypatch):
    recorded = []

    def fake_irq(irq, cb):
        recorded.append((irq, cb))
        return None

    monkeypatch.setattr(mpr_mod, "IRQHandler", fake_irq)
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    mpr_mod.MPR121(bus=bus_factory, irq=4, handlers=1, setup=1, reset=0)

    assert len(recorded) == 1
    assert recorded[0][0] == 4
    assert callable(recorded[0][1])


def test_init_registers_poller_when_irq_disabled(bus_factory, device, monkeypatch):
    recorded = []

    def fake_poller(cb):
        recorded.append(cb)
        return None

    monkeypatch.setattr(mpr_mod, "Poller", fake_poller)
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    mpr_mod.MPR121(bus=bus_factory, irq=0, handlers=1, setup=1, reset=0)

    assert len(recorded) == 1
    assert callable(recorded[0])


def test_update_touch_state_calls_handlers_only_on_change(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    events = []
    m._handlers[0] = lambda touched, channel=None: events.append((touched, channel))
    m.touched = lambda raise_on_failure=1: 0b1

    m.update_touch_state()
    assert m._touched == 0b1
    assert events == [(True, 0)]

    m.update_touch_state()
    assert events == [(True, 0)]


def test_handle_touch_supports_handlers_with_and_without_channel(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    events = []
    m._handlers[0] = lambda touched, channel=None: events.append(("with", touched, channel))
    m._handlers[1] = lambda touched: events.append(("without", touched))

    m._handle_touch(touched=0b11, changed=0b11)
    assert events == [("with", True, 0), ("without", True)]


def test_is_touched_uses_internal_state(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    m._touched = 0b1010
    assert m.is_touched(1) is True
    assert m.is_touched(0) is False
    assert m.is_touched(3) is True


def test_on_touch_wraps_handler_and_none_removes(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    m.on_touch(2, lambda touched, channel=None: None)
    assert callable(m._handlers[2])

    m.on_touch(2, None)
    assert callable(m._handlers[2])


def test_reset_writes_soft_reset_register(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    m.reset()
    assert ("byte", mpr_mod.SRESET, 0x63) in device.writes


def test_configure_writes_expected_ecr(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    m.configure(cl=2, prox=3, touch=12)
    assert ("byte", mpr_mod.ECR, (2 << 6) | (3 << 4) | 12) in device.writes


def test_touched_masks_status_and_raises_on_overcurrent(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    device.read_word_values[mpr_mod.ETS] = 0x8005
    device.read_word_values[mpr_mod.OOR] = 0

    with pytest.raises(Exception, match="overcurrent"):
        m.touched()

    assert m.touched(raise_on_failure=0) == 0x0005


def test_out_of_range_raises_for_autoconfig_and_reconfig_flags(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    device.read_word_values[mpr_mod.OOR] = 0x8001
    with pytest.raises(Exception, match="auto config failed"):
        m.out_of_range()

    device.read_word_values[mpr_mod.OOR] = 0x4001
    with pytest.raises(Exception, match="auto reconfig failed"):
        m.out_of_range()

    device.read_word_values[mpr_mod.OOR] = 0x0007
    assert m.out_of_range() == 0x0007


def test_electrode_data_reads_and_combines_bytes(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    data = []
    for i in range(mpr_mod.NCH):
        data.extend([i, i + 1])
    device.read_block_values[(mpr_mod.EFD, 2 * mpr_mod.NCH)] = data

    res = m.electrode_data()
    assert len(res) == mpr_mod.NCH
    assert res[0] == (1 << 8) + 0
    assert res[1] == (2 << 8) + 1


def test_baseline_reads_and_scales_values(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    device.read_block_values[(mpr_mod.EBL, mpr_mod.NCH)] = list(range(mpr_mod.NCH))
    assert m.baseline() == [i << 2 for i in range(mpr_mod.NCH)]


def test_baseline_writes_correct_registers_for_touch_and_prox(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    m.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    m.baseline(rft=2, nhd=7, ncl=8, fdl=9)
    m.baseline(rft=1, mhd=4, nhd=2, ncl=6, fdl=7, prox=1)

    assert ("block", mpr_mod.MHD, [5, 1, 3, 20]) in device.writes
    assert ("block", mpr_mod.MHD + 8, [7, 8, 9]) in device.writes
    assert ("block", mpr_mod.MHDX + 4, [4, 2, 6, 7]) in device.writes


def test_threshold_defaults_release_and_applies_to_all_channels(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    m.threshold(50)
    expected_release = int(0.6 * 50)
    for ch in range(mpr_mod.NCH):
        assert ("byte", mpr_mod.TTH + 2 * ch, 50) in device.writes
        assert ("byte", mpr_mod.RTH + 2 * ch, expected_release) in device.writes


def test_debounce_and_filter_and_auto_config_write_expected_values(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    m.debounce(touch=3)
    m.filter(cdc=30, cdt=1, ffi=1, sfi=2, esi=4)
    m.auto_config(ace=1, are=0, bva=2, retry=3, afes=1, scts=1, acfie=0, arfie=1, oorie=0, usl=200, lsl=130, tl=180)

    assert ("byte", mpr_mod.DEB, (3 << 4) | 3) in device.writes
    assert ("word", mpr_mod.AFE1, (((1 << 5) | (2 << 3) | 4) << 8) | (((1 << 6) | 30))) in device.writes
    assert ("word", mpr_mod.ACNF_C0, (((1 << 7) | (0 << 2) | (1 << 1) | 0) << 8) + ((1 << 6) | (3 << 4) | (2 << 2) | (0 << 1) | 1)) in device.writes
    assert ("byte", mpr_mod.ACNF_USL, 200) in device.writes
    assert ("byte", mpr_mod.ACNF_LSL, 130) in device.writes
    assert ("byte", mpr_mod.ACNF_TL, 180) in device.writes


def test_charge_updates_cdc_and_cdt_bits(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    device.read_byte_values[mpr_mod.CDT + 0] = 0xA5
    m.charge(channel=1, cdc=12, cdt=3)

    assert ("byte", mpr_mod.CDC + 1, 12) in device.writes
    assert ("byte", mpr_mod.CDT + 0, (0xA5 & ~(0xF << 4)) | (3 << 4)) in device.writes


def test_gpio_setup_and_set_and_status(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    device.read_byte_values[mpr_mod.GPIO_CTL0] = 0
    device.read_byte_values[mpr_mod.GPIO_CTL1] = 0
    device.read_byte_values[mpr_mod.GPIO_DIR] = 0
    device.read_byte_values[mpr_mod.GPIO_EN] = 0
    device.read_byte_values[mpr_mod.GPIO_DAT] = 0x55

    m.gpio_setup(channel=4, output=1, mode=3, enable=1)
    m.gpio_set(channel=11, value=1)
    assert m.gpio_status() == 0x55

    assert ("byte", mpr_mod.GPIO_CTL0, 0) in device.writes
    assert ("byte", mpr_mod.GPIO_CTL1, 1) in device.writes
    assert ("byte", mpr_mod.GPIO_DIR, 1) in device.writes
    assert ("byte", mpr_mod.GPIO_EN, 1) in device.writes
    assert ("byte", mpr_mod.GPIO_SET, 1 << 7) in device.writes


def test_setup_calls_initial_configuration_sequence(bus_factory, device, monkeypatch):
    monkeypatch.setattr(mpr_mod.MPR121, "setup", lambda self, **kwargs: None)
    m = mpr_mod.MPR121(bus=bus_factory, handlers=0, setup=1, reset=0)

    calls = []

    monkeypatch.setattr(m, "reset", lambda: calls.append("reset"))
    monkeypatch.setattr(m, "configure", lambda **kwargs: calls.append(("configure", kwargs)))
    monkeypatch.setattr(m, "filter", lambda **kwargs: calls.append(("filter", kwargs)))
    monkeypatch.setattr(m, "auto_config", lambda **kwargs: calls.append(("auto_config", kwargs)))
    monkeypatch.setattr(m, "threshold", lambda *args, **kwargs: calls.append(("threshold", args, kwargs)))
    monkeypatch.setattr(m, "debounce", lambda *args, **kwargs: calls.append(("debounce", args, kwargs)))
    monkeypatch.setattr(m, "baseline", lambda *args, **kwargs: calls.append(("baseline", args, kwargs)))

    m.setup(reset=1, channels=8, prox=2, threshold=60, debounce=3, auto_config=1)

    assert calls[0] == "reset"
    assert calls[1] == ("configure", {"prox": 0, "touch": 0})
    assert calls[2][0] == "filter"
    assert calls[3][0] == "auto_config"
    assert calls[4] == ("threshold", (), {"touch": 60})
    assert calls[5] == ("threshold", (), {"channel": 12, "touch": 15})
    assert calls[6] == ("debounce", (), {"debounce": 3}) or calls[6][0] == "debounce"
    assert calls[-1] == ("configure", {"prox": 2, "touch": 8})
