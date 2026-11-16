"""Unit tests for piripherals.mpr121.MPR121.

No hardware is touched: a fake I2C bus records every call and serves
register values from plain dicts. The code under test uses no random
numbers and no hash(), so no generators are seeded here.
"""
import sys
import types

import pytest

import piripherals.mpr121 as mod
from piripherals.mpr121 import MPR121


class FakeBus:
    """Stands in for a piripherals Bus; device() returns the same object."""

    def __init__(self):
        self.calls = []
        self.bytes = {}   # register -> byte value
        self.words = {}   # register -> 16 bit word value
        self.addr = None

    def device(self, addr):
        self.addr = addr
        return self

    def write_byte(self, reg, value):
        self.calls.append(('write_byte', reg, value))
        self.bytes[reg] = value

    def read_byte(self, reg):
        return self.bytes.get(reg, 0)

    def write_word(self, reg, value):
        self.calls.append(('write_word', reg, value))
        self.words[reg] = value

    def read_word(self, reg):
        return self.words.get(reg, 0)

    def write_block(self, reg, data):
        data = list(data)
        self.calls.append(('write_block', reg, data))
        for i, value in enumerate(data):
            self.bytes[reg + i] = value

    def read_block(self, reg, count):
        return [self.bytes.get(reg + i, 0) for i in range(count)]


@pytest.fixture
def bus():
    return FakeBus()


def make(bus, **kwargs):
    """Create a MPR121 without handlers, setup or reset unless asked for."""
    kwargs.setdefault('handlers', 0)
    kwargs.setdefault('setup', 0)
    kwargs.setdefault('reset', 0)
    return MPR121(bus=bus, **kwargs)


def test_construction_without_setup_or_reset_writes_nothing(bus):
    make(bus)
    assert bus.calls == []


def test_reset_writes_soft_reset_command(bus):
    m = make(bus, reset=1)
    assert bus.calls == [('write_byte', mod.SRESET, 0x63)]
    bus.calls.clear()
    m.reset()
    assert bus.calls == [('write_byte', mod.SRESET, 0x63)]


def test_bus_number_creates_bus_when_device_missing(monkeypatch):
    created = []

    class Bus:
        def __init__(self, number):
            created.append(number)
            self.fake = FakeBus()

        def device(self, addr):
            return self.fake.device(addr)

    fake_module = types.ModuleType('piripherals.bus')
    fake_module.Bus = Bus
    # monkeypatch restores sys.modules even if the test fails
    monkeypatch.setitem(sys.modules, 'piripherals.bus', fake_module)

    m = MPR121(bus=3, addr=0x5b, handlers=0, setup=0, reset=0)

    assert created == [3]
    assert m._bus.addr == 0x5b


def test_handlers_without_irq_uses_poller(bus, monkeypatch):
    created = []

    class FakePoller:
        def __init__(self, callback):
            created.append(callback)

    monkeypatch.setattr(mod, 'Poller', FakePoller, raising=False)
    m = MPR121(bus=bus, handlers=1, setup=0, reset=0)

    assert len(created) == 1
    assert created[0] == m.update_touch_state


def test_irq_pin_registers_irq_handler(bus, monkeypatch):
    registered = []
    polled = []

    class FakeIRQHandler:
        def __init__(self, pin, callback):
            registered.append((pin, callback))

    class FakePoller:
        def __init__(self, callback):
            polled.append(callback)

    monkeypatch.setattr(mod, 'IRQHandler', FakeIRQHandler, raising=False)
    monkeypatch.setattr(mod, 'Poller', FakePoller, raising=False)
    m = MPR121(bus=bus, irq=4, handlers=1, setup=0, reset=0)

    assert polled == []
    assert len(registered) == 1
    pin, callback = registered[0]
    assert pin == 4
    assert callback == m.update_touch_state


def test_configure_packs_calibration_proximity_and_touch(bus):
    m = make(bus)
    m.configure(cl=3, prox=3, touch=12)
    # (3 << 6) | (3 << 4) | 12 == 0xfc
    assert bus.bytes[mod.ECR] == 0xfc

    m.configure(prox=0, touch=0)
    # stop mode keeps cl=3 default: (3 << 6) == 0xc0
    assert bus.bytes[mod.ECR] == 0xc0


def test_touched_returns_low_13_bits_of_touch_word(bus):
    m = make(bus)
    bus.words[mod.ETS] = 0x0a05
    assert m.touched() == 0x0a05


def test_touched_raises_on_overcurrent(bus):
    m = make(bus)
    bus.words[mod.ETS] = 0x8000
    with pytest.raises(Exception, match='overcurrent'):
        m.touched()


def test_touched_raises_when_out_of_range(bus):
    m = make(bus)
    bus.words[mod.OOR] = 0x0001
    with pytest.raises(Exception, match='out of range'):
        m.touched()


def test_touched_without_raise_ignores_failure_bits(bus):
    m = make(bus)
    bus.words[mod.ETS] = 0x8001
    bus.words[mod.OOR] = 0x4001
    assert m.touched(raise_on_failure=0) == 1


def test_out_of_range_raises_on_auto_config_failure(bus):
    m = make(bus)
    bus.words[mod.OOR] = 0x8000
    with pytest.raises(Exception, match='auto config failed'):
        m.out_of_range()


def test_out_of_range_raises_on_auto_reconfig_failure(bus):
    m = make(bus)
    bus.words[mod.OOR] = 0x4000
    with pytest.raises(Exception, match='auto reconfig failed'):
        m.out_of_range()


def test_update_touch_state_calls_handlers_only_for_changed_channels(bus):
    m = make(bus)
    events = []
    m.on_touch(1, lambda state, ch: events.append((state, ch)))
    m.on_touch(2, lambda state, ch: events.append((state, ch)))

    bus.words[mod.ETS] = 0b0110  # channels 1 and 2 touched
    m.update_touch_state()
    assert events == [(True, 1), (True, 2)]

    bus.words[mod.ETS] = 0b0100  # channel 1 released, channel 2 unchanged
    m.update_touch_state()
    assert events[2:] == [(False, 1)]


def test_is_touched_reflects_last_update(bus):
    m = make(bus)
    bus.words[mod.ETS] = 0x0005  # channels 0 and 2
    m.update_touch_state()
    assert m.is_touched(0) is True
    assert m.is_touched(1) is False
    assert m.is_touched(2) is True


def test_electrode_data_combines_little_endian_byte_pairs(bus):
    m = make(bus)
    flat = []
    for i in range(13):
        flat += [i, 0x03]  # low byte, high byte
    bus.write_block(mod.EFD, flat)
    assert m.electrode_data() == [(3 << 8) + i for i in range(13)]


def test_baseline_read_shifts_each_value_by_two(bus):
    m = make(bus)
    bus.write_block(mod.EBL, list(range(13)))
    assert m.baseline() == [i << 2 for i in range(13)]


def test_baseline_rising_and_falling_use_consecutive_registers(bus):
    m = make(bus)
    m.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    m.baseline(rft=1, mhd=5, nhd=1, ncl=3, fdl=20)
    assert ('write_block', mod.MHD, [5, 1, 3, 20]) in bus.calls
    assert ('write_block', mod.MHD + 4, [5, 1, 3, 20]) in bus.calls


def test_baseline_touched_scenario_writes_three_bytes(bus):
    m = make(bus)
    m.baseline(rft=2, nhd=1, ncl=3, fdl=20)
    assert bus.calls == [('write_block', mod.MHD + 8, [1, 3, 20])]


def test_baseline_proximity_uses_proximity_registers(bus):
    m = make(bus)
    m.baseline(rft=0, mhd=1, nhd=1, ncl=3, fdl=20, prox=1)
    assert bus.calls == [('write_block', mod.MHDX, [1, 1, 3, 20])]


def test_threshold_default_release_is_sixty_percent_of_touch(bus):
    m = make(bus)
    m.threshold(touch=100, channel=3)
    assert bus.bytes[mod.TTH + 6] == 100
    assert bus.bytes[mod.RTH + 6] == 60


def test_threshold_without_channel_applies_to_all_thirteen_channels(bus):
    m = make(bus)
    m.threshold(touch=40, release=10)
    for i in range(mod.NCH):
        assert bus.bytes[mod.TTH + 2 * i] == 40
        assert bus.bytes[mod.RTH + 2 * i] == 10


def test_debounce_packs_release_in_high_nibble(bus):
    m = make(bus)
    m.debounce(touch=2, release=5)
    assert bus.bytes[mod.DEB] == 0x52

    m.debounce(touch=3)  # release defaults to touch
    assert bus.bytes[mod.DEB] == 0x33


def test_filter_packs_afe1_word(bus):
    m = make(bus)
    m.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    # lb = (1 << 6) | 30 = 0x5e, hb = (1 << 5) | (1 << 3) | 0 = 0x28
    assert bus.words[mod.AFE1] == 0x285e


def test_charge_updates_only_its_own_nibble(bus):
    m = make(bus)
    bus.bytes[mod.CDT] = 0x5c

    m.charge(channel=1, cdc=20, cdt=3)  # high nibble of CDT
    assert bus.bytes[mod.CDC + 1] == 20
    assert bus.bytes[mod.CDT] == 0x3c

    m.charge(channel=0, cdc=0, cdt=7)  # low nibble, high nibble kept
    assert bus.bytes[mod.CDT] == 0x37


def test_auto_config_default_register_values(bus):
    m = make(bus)
    m.auto_config()
    # lb = 64 + 32 + 12 + 2 + 1 = 0x6f, hb = 4 + 2 + 1 = 0x07
    assert bus.words[mod.ACNF_C0] == 0x076f
    assert bus.bytes[mod.ACNF_USL] == 200
    assert bus.bytes[mod.ACNF_LSL] == 130
    assert bus.bytes[mod.ACNF_TL] == 180


def test_gpio_setup_sets_mode_direction_and_enable_bits(bus):
    m = make(bus)
    m.gpio_setup(channel=5, output=1, mode=2, enable=1)
    bit = 1 << (5 - 4)
    assert bus.bytes[mod.GPIO_CTL0] == bit      # mode bit 1 set
    assert bus.bytes[mod.GPIO_CTL1] == 0        # mode bit 0 clear
    assert bus.bytes[mod.GPIO_DIR] == bit       # output
    assert bus.bytes[mod.GPIO_EN] == bit        # enabled


def test_gpio_set_writes_set_or_clear_register_and_checks_channel(bus):
    m = make(bus)
    m.gpio_set(4, True)
    m.gpio_set(11, False)
    assert bus.calls == [
        ('write_byte', mod.GPIO_SET, 1),
        ('write_byte', mod.GPIO_CLR, 1 << 7),
    ]
    with pytest.raises(AssertionError):
        m.gpio_set(3, True)


def test_gpio_status_returns_data_register(bus):
    m = make(bus)
    bus.bytes[mod.GPIO_DAT] = 0x81
    assert m.gpio_status() == 0x81
