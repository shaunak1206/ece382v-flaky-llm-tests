"""Unit tests for piripherals.mpr121.MPR121.

The sensor is driven through an in-memory fake I2C device, so no hardware,
smbus or RPi.GPIO is needed. Background IRQ and polling threads are never
started: tests either pass handlers=0 or replace IRQHandler/Poller with stubs.
"""

import pytest

import piripherals.mpr121 as mpr121_mod
from piripherals.mpr121 import MPR121

ETS = mpr121_mod.ETS
OOR = mpr121_mod.OOR
EFD = mpr121_mod.EFD
EBL = mpr121_mod.EBL
TTH = mpr121_mod.TTH
RTH = mpr121_mod.RTH
DEB = mpr121_mod.DEB
AFE1 = mpr121_mod.AFE1
ECR = mpr121_mod.ECR
CDC = mpr121_mod.CDC
CDT = mpr121_mod.CDT
ACNF_C0 = mpr121_mod.ACNF_C0
ACNF_USL = mpr121_mod.ACNF_USL
ACNF_LSL = mpr121_mod.ACNF_LSL
ACNF_TL = mpr121_mod.ACNF_TL
GPIO_CTL0 = mpr121_mod.GPIO_CTL0
GPIO_CTL1 = mpr121_mod.GPIO_CTL1
GPIO_DIR = mpr121_mod.GPIO_DIR
GPIO_EN = mpr121_mod.GPIO_EN
GPIO_SET = mpr121_mod.GPIO_SET
GPIO_CLR = mpr121_mod.GPIO_CLR


class FakeDevice:
    """Byte-addressed register file that records every write."""

    def __init__(self):
        self.regs = {}
        self.writes = []

    def read_byte(self, reg):
        return self.regs.get(reg, 0)

    def read_word(self, reg):
        return self.read_byte(reg) | (self.read_byte(reg + 1) << 8)

    def read_block(self, reg, n):
        return [self.read_byte(reg + i) for i in range(n)]

    def write_byte(self, reg, value):
        self.writes.append(('byte', reg, value))
        self.regs[reg] = value & 0xff

    def write_word(self, reg, value):
        self.writes.append(('word', reg, value))
        self.regs[reg] = value & 0xff
        self.regs[reg + 1] = (value >> 8) & 0xff

    def write_block(self, reg, values):
        self.writes.append(('block', reg, list(values)))
        for i, v in enumerate(values):
            self.regs[reg + i] = v & 0xff


class FakeBus:
    """Bus stand-in: MPR121 skips wrapping it because it has a ``device`` attribute."""

    def __init__(self):
        self.device_obj = FakeDevice()
        self.addrs = []

    def device(self, addr):
        self.addrs.append(addr)
        return self.device_obj


def set_word(dev, reg, value):
    dev.regs[reg] = value & 0xff
    dev.regs[reg + 1] = (value >> 8) & 0xff


def _fail(*args, **kwargs):
    raise AssertionError('unexpected call')


@pytest.fixture
def bus():
    return FakeBus()


@pytest.fixture
def dev(bus):
    return bus.device_obj


@pytest.fixture
def sensor(bus):
    return MPR121(bus, handlers=0, setup=0, reset=0)


def test_constructor_binds_device_to_given_address(bus):
    sensor = MPR121(bus, addr=0x5b, handlers=0, setup=0, reset=0)
    assert bus.addrs == [0x5b]
    assert sensor.addr == 0x5b


def test_constructor_reset_only_sends_soft_reset(bus, monkeypatch):
    monkeypatch.setattr(mpr121_mod, 'IRQHandler', _fail)
    monkeypatch.setattr(mpr121_mod, 'Poller', _fail)
    MPR121(bus, handlers=0, setup=0, reset=0)
    assert bus.device_obj.writes == []
    MPR121(bus, handlers=0, setup=0, reset=1)
    assert bus.device_obj.writes == [('byte', 0x80, 0x63)]


def test_default_constructor_configures_and_enables_all_electrodes(bus):
    MPR121(bus, handlers=0)
    ecr = [value for kind, reg, value in bus.device_obj.writes if reg == ECR]
    assert ecr[0] == 0xc0   # stop mode while configuring
    assert ecr[-1] == 0xcc  # run mode, 12 touch electrodes


def test_irq_pin_registers_irq_handler(bus, monkeypatch):
    calls = []
    monkeypatch.setattr(mpr121_mod, 'IRQHandler',
                        lambda pin, cb: calls.append((pin, cb)))
    monkeypatch.setattr(mpr121_mod, 'Poller', _fail)
    sensor = MPR121(bus, irq=4, setup=0, reset=0)
    assert len(calls) == 1
    pin, callback = calls[0]
    assert pin == 4
    assert callback.__self__ is sensor
    assert callback.__func__ is MPR121.update_touch_state


def test_touched_returns_low_13_bits(dev, sensor):
    set_word(dev, ETS, 0x3a05)
    assert sensor.touched() == 0x1a05


def test_touched_raises_on_overcurrent(dev, sensor):
    set_word(dev, ETS, 0x8000)
    with pytest.raises(Exception, match='overcurrent'):
        sensor.touched()


def test_out_of_range_failure_bits_raise(dev, sensor):
    set_word(dev, OOR, 0x8000)
    with pytest.raises(Exception, match='auto config failed'):
        sensor.touched()
    set_word(dev, OOR, 0x4000)
    with pytest.raises(Exception, match='auto reconfig failed'):
        sensor.out_of_range()


def test_failure_checks_can_be_disabled(dev, sensor):
    set_word(dev, ETS, 0x8001)
    set_word(dev, OOR, 0xc000)
    assert sensor.touched(raise_on_failure=0) == 0x0001
    assert sensor.out_of_range(raise_on_failure=0) == 0x0000


def test_on_touch_handler_receives_state_and_channel(dev, sensor):
    events = []
    sensor.on_touch(3, lambda state, ch: events.append((state, ch)))
    assert sensor.is_touched(3) is False

    set_word(dev, ETS, 1 << 3)
    sensor.update_touch_state()
    assert sensor.is_touched(3) is True

    set_word(dev, ETS, 0)
    sensor.update_touch_state()
    assert sensor.is_touched(3) is False
    assert events == [(True, 3), (False, 3)]


def test_handler_fires_only_when_state_changes(dev, sensor):
    events = []
    sensor.on_touch(3, lambda state, ch: events.append(state))
    set_word(dev, ETS, 1 << 3)
    sensor.update_touch_state()
    sensor.update_touch_state()
    assert events == [True]


def test_only_changed_channels_invoke_their_handlers(dev, sensor):
    ch0_events = []
    ch1_events = []
    sensor.on_touch(0, lambda state, ch: ch0_events.append(state))
    sensor.on_touch(1, lambda state, ch: ch1_events.append(state))

    set_word(dev, ETS, 0b01)
    sensor.update_touch_state()
    assert ch0_events == [True]
    assert ch1_events == []

    set_word(dev, ETS, 0b11)
    sensor.update_touch_state()
    assert ch0_events == [True]
    assert ch1_events == [True]


def test_handler_exceptions_do_not_escape_update(dev, sensor):
    def boom(state, ch):
        raise RuntimeError('handler failed')

    sensor.on_touch(0, boom)
    set_word(dev, ETS, 1)
    sensor.update_touch_state()  # must not raise
    assert sensor.is_touched(0) is True


def test_electrode_data_combines_little_endian_pairs(dev, sensor):
    for ch in range(13):
        dev.regs[EFD + 2 * ch] = ch
        dev.regs[EFD + 2 * ch + 1] = 0x02
    assert sensor.electrode_data() == [c + 0x200 for c in range(13)]


def test_baseline_read_scales_raw_values(dev, sensor):
    for i in range(13):
        dev.regs[EBL + i] = i + 1
    assert sensor.baseline() == [(i + 1) << 2 for i in range(13)]


def test_baseline_writes_one_block_per_scenario(dev, sensor):
    sensor.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    sensor.baseline(rft=1, mhd=7, nhd=2, ncl=4, fdl=21)
    sensor.baseline(rft=2, nhd=1, ncl=3, fdl=20)
    assert ('block', 0x2b, [5, 1, 3, 20]) in dev.writes
    assert ('block', 0x2f, [7, 2, 4, 21]) in dev.writes
    assert ('block', 0x33, [1, 3, 20]) in dev.writes


def test_threshold_defaults_release_to_60_percent_on_all_channels(dev, sensor):
    sensor.threshold(50)
    for ch in range(13):
        assert dev.regs[TTH + 2 * ch] == 50
        assert dev.regs[RTH + 2 * ch] == 30


def test_debounce_packs_release_and_touch_nibbles(dev, sensor):
    sensor.debounce(touch=2, release=5)
    assert dev.regs[DEB] == 0x52
    sensor.debounce(3)
    assert dev.regs[DEB] == 0x33


def test_filter_packs_afe1_word(dev, sensor):
    sensor.filter()
    assert ('word', AFE1, 0x2410) in dev.writes


def test_charge_updates_only_its_nibble(dev, sensor):
    dev.regs[CDT + 1] = 0x05  # channel 2 nibble, must survive
    sensor.charge(3, cdc=10, cdt=7)
    assert dev.regs[CDC + 3] == 10
    assert dev.regs[CDT + 1] == 0x75


def test_auto_config_default_word_layout(dev, sensor):
    sensor.auto_config()
    assert ('word', ACNF_C0, 0x076f) in dev.writes
    assert dev.regs[ACNF_USL] == 200
    assert dev.regs[ACNF_LSL] == 130
    assert dev.regs[ACNF_TL] == 180


def test_gpio_setup_sets_mode_and_direction_bits(dev, sensor):
    sensor.gpio_setup(5, output=True, mode=2, enable=1)
    assert dev.regs[GPIO_CTL0] == 0b10
    assert dev.regs.get(GPIO_CTL1, 0) == 0
    assert dev.regs[GPIO_DIR] == 0b10
    assert dev.regs[GPIO_EN] == 0b10


def test_gpio_set_writes_set_or_clear_register(dev, sensor):
    sensor.gpio_set(6, True)
    sensor.gpio_set(6, False)
    assert ('byte', GPIO_SET, 4) in dev.writes
    assert ('byte', GPIO_CLR, 4) in dev.writes
    with pytest.raises(AssertionError):
        sensor.gpio_set(12, True)


def test_setup_programs_proximity_threshold_and_baselines(dev, sensor):
    sensor.setup(threshold=40)
    assert dev.regs[TTH] == 40
    assert dev.regs[TTH + 2 * 12] == 10  # proximity channel uses threshold >> 2
    assert dev.regs[RTH + 2 * 12] == 6
    assert ('block', 0x2b, [5, 1, 3, 20]) in dev.writes
    assert dev.regs[ECR] == 0xcc
