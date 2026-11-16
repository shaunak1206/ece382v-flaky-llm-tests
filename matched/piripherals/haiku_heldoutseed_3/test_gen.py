import pytest

from piripherals.mpr121 import (
    ACNF_C0,
    ACNF_LSL,
    ACNF_TL,
    ACNF_USL,
    CDC,
    CDT,
    DEB,
    EBL,
    EFD,
    ETS,
    GPIO_CLR,
    GPIO_SET,
    MHD,
    NCH,
    OOR,
    RTH,
    SRESET,
    TTH,
    AFE1,
    ECR,
    MPR121,
)


class FakeDevice:
    """In-memory stand-in for a smbus device at one I2C address."""

    def __init__(self):
        self.regs = {}    # byte register -> value
        self.words = {}   # word register -> value (for read_word)
        self.log = []     # ordered record of write operations

    def write_byte(self, reg, value):
        self.log.append(("wb", reg, value))
        self.regs[reg] = value & 0xFF

    def write_word(self, reg, value):
        self.log.append(("ww", reg, value))
        self.words[reg] = value

    def write_block(self, reg, data):
        data = list(data)
        self.log.append(("wblk", reg, data))
        for i, v in enumerate(data):
            self.regs[reg + i] = v & 0xFF

    def read_byte(self, reg):
        return self.regs.get(reg, 0)

    def read_word(self, reg):
        return self.words.get(reg, 0)

    def read_block(self, reg, length):
        return [self.regs.get(reg + i, 0) for i in range(length)]

    def writes_to(self, reg):
        return [entry for entry in self.log if entry[1] == reg]


class FakeBus:
    """Mimics the object passed as ``bus``: it only needs a ``device(addr)`` method."""

    def __init__(self):
        self.device_obj = FakeDevice()
        self.requested_addresses = []

    def device(self, addr):
        self.requested_addresses.append(addr)
        return self.device_obj


def make_sensor(**kwargs):
    """Create an MPR121 on a fake bus with polling, setup and reset disabled by default."""
    bus = FakeBus()
    kwargs.setdefault("handlers", 0)
    kwargs.setdefault("setup", 0)
    kwargs.setdefault("reset", 0)
    sensor = MPR121(bus=bus, **kwargs)
    return sensor, bus.device_obj


def test_init_binds_device_at_default_and_custom_address():
    bus = FakeBus()
    sensor = MPR121(bus=bus, handlers=0, setup=0, reset=0)
    assert bus.requested_addresses == [0x5A]
    assert sensor.addr == 0x5A

    bus2 = FakeBus()
    sensor2 = MPR121(bus=bus2, addr=0x5B, handlers=0, setup=0, reset=0)
    assert bus2.requested_addresses == [0x5B]
    assert sensor2.addr == 0x5B


def test_init_without_setup_or_reset_sends_nothing():
    _, dev = make_sensor()
    assert dev.log == []


def test_init_with_reset_only_sends_soft_reset_magic():
    _, dev = make_sensor(reset=1, setup=0)
    assert dev.log == [("wb", SRESET, 0x63)]


def test_reset_writes_soft_reset_magic_value():
    sensor, dev = make_sensor()
    sensor.reset()
    assert dev.writes_to(SRESET) == [("wb", SRESET, 0x63)]


def test_configure_packs_ecr_fields_and_defaults():
    sensor, dev = make_sensor()
    sensor.configure(cl=1, prox=2, touch=5)
    # (1 << 6) | (2 << 4) | 5 = 101
    assert dev.regs[ECR] == 101

    sensor.configure()
    # defaults cl=3, prox=0, touch=12 -> (3 << 6) | 12 = 204
    assert dev.regs[ECR] == 204


def test_is_touched_reads_internal_touch_state():
    sensor, _ = make_sensor()
    sensor._touched = 0b101
    assert sensor.is_touched(0) is True
    assert sensor.is_touched(1) is False
    assert sensor.is_touched(2) is True


def test_touched_masks_out_failure_bits_from_status_word():
    sensor, dev = make_sensor()
    dev.words[ETS] = 0x2005  # bit 13 set, low 13 bits = 0x005
    assert sensor.touched(raise_on_failure=0) == 0x0005


def test_touched_raises_on_overcurrent():
    sensor, dev = make_sensor()
    dev.words[ETS] = 1 << 15
    with pytest.raises(Exception, match="overcurrent"):
        sensor.touched()


def test_touched_raises_when_out_of_range():
    sensor, dev = make_sensor()
    dev.words[ETS] = 0x0001
    dev.words[OOR] = 0x0001
    with pytest.raises(Exception, match="out of range"):
        sensor.touched()


def test_touched_skips_checks_when_raise_disabled():
    sensor, dev = make_sensor()
    dev.words[ETS] = 0x8001
    dev.words[OOR] = 0x0001
    assert sensor.touched(raise_on_failure=0) == 0x0001


def test_out_of_range_raises_on_auto_config_failure():
    sensor, dev = make_sensor()
    dev.words[OOR] = 1 << 15
    with pytest.raises(Exception, match="auto config failed"):
        sensor.out_of_range()


def test_out_of_range_raises_on_auto_reconfig_failure():
    sensor, dev = make_sensor()
    dev.words[OOR] = 1 << 14
    with pytest.raises(Exception, match="auto reconfig failed"):
        sensor.out_of_range()


def test_update_touch_state_invokes_handler_on_change():
    sensor, dev = make_sensor()
    calls = []

    def handler(state, channel):
        calls.append((state, channel))

    sensor.on_touch(3, handler)

    dev.words[ETS] = 1 << 3
    sensor.update_touch_state()
    assert calls == [(True, 3)]

    dev.words[ETS] = 0
    sensor.update_touch_state()
    assert calls == [(True, 3), (False, 3)]


def test_update_touch_state_is_silent_when_nothing_changed():
    sensor, dev = make_sensor()
    calls = []
    sensor.on_touch(0, lambda state, channel: calls.append(state))

    dev.words[ETS] = 0x0001
    sensor.update_touch_state()
    sensor.update_touch_state()
    assert calls == [True]


def test_electrode_data_assembles_little_endian_pairs():
    sensor, dev = make_sensor()
    # channel 0: low byte 0x34, high byte 0x02 -> 0x234
    dev.regs[EFD] = 0x34
    dev.regs[EFD + 1] = 0x02
    # channel 12: low byte 0xFF, high byte 0x03 -> 0x3FF
    dev.regs[EFD + 2 * 12] = 0xFF
    dev.regs[EFD + 2 * 12 + 1] = 0x03

    data = sensor.electrode_data()
    assert len(data) == NCH
    assert data[0] == 0x234
    assert data[12] == 0x3FF


def test_baseline_read_shifts_values_left_by_two():
    sensor, dev = make_sensor()
    dev.regs[EBL] = 0x10
    dev.regs[EBL + 1] = 0x01

    baseline = sensor.baseline()
    assert len(baseline) == NCH
    assert baseline[0] == 0x40
    assert baseline[1] == 0x04


def test_threshold_single_channel_writes_touch_and_release():
    sensor, dev = make_sensor()
    sensor.threshold(touch=40, channel=2)
    assert dev.regs[TTH + 4] == 40
    # release defaults to int(0.6 * 40) = 24
    assert dev.regs[RTH + 4] == 24


def test_threshold_all_channels_writes_every_electrode():
    sensor, dev = make_sensor()
    sensor.threshold(touch=50)
    touch_writes = [e for e in dev.log if e[1] >= TTH and e[1] < RTH and (e[1] - TTH) % 2 == 0]
    release_writes = [e for e in dev.log if e[1] >= RTH and (e[1] - RTH) % 2 == 0 and e[1] < RTH + 2 * NCH]
    assert len(touch_writes) == NCH
    assert len(release_writes) == NCH
    assert all(e[2] == 50 for e in touch_writes)
    assert all(e[2] == int(0.6 * 50) for e in release_writes)


def test_debounce_packs_touch_and_release_nibbles():
    sensor, dev = make_sensor()
    sensor.debounce(touch=2, release=5)
    assert dev.regs[DEB] == (5 << 4) | 2

    sensor.debounce(3)
    # release defaults to touch
    assert dev.regs[DEB] == (3 << 4) | 3


def test_filter_packs_afe1_word():
    sensor, dev = make_sensor()
    sensor.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    # lb = (1 << 6) | 30 = 0x5e ; hb = (1 << 5) | (1 << 3) | 0 = 0x28
    assert dev.words[AFE1] == 0x285E


def test_charge_updates_only_the_channel_nibble():
    sensor, dev = make_sensor()
    # channel 2 lives in CDT + 1, low nibble (s = 0); high nibble must be preserved
    dev.regs[CDT + 1] = 0xA0
    sensor.charge(channel=2, cdc=7, cdt=3)
    assert dev.regs[CDC + 2] == 7
    assert dev.regs[CDT + 1] == 0xA3


def test_auto_config_packs_registers_with_defaults():
    sensor, dev = make_sensor()
    sensor.auto_config()
    # lb = (1<<6)|(2<<4)|(3<<2)|(1<<1)|1 = 0x6f
    # hb = (0<<7)|(1<<2)|(1<<1)|1 = 0x07
    assert dev.words[ACNF_C0] == 0x076F
    assert dev.regs[ACNF_USL] == 200
    assert dev.regs[ACNF_LSL] == 130
    assert dev.regs[ACNF_TL] == 180


def test_gpio_set_and_clear_use_separate_registers():
    sensor, dev = make_sensor()
    sensor.gpio_set(5, True)
    assert dev.writes_to(GPIO_SET) == [("wb", GPIO_SET, 1 << 1)]

    sensor.gpio_set(5, False)
    assert dev.writes_to(GPIO_CLR) == [("wb", GPIO_CLR, 1 << 1)]


def test_gpio_set_rejects_channel_outside_4_to_11():
    sensor, _ = make_sensor()
    with pytest.raises(AssertionError):
        sensor.gpio_set(3, True)
    with pytest.raises(AssertionError):
        sensor.gpio_set(12, True)
