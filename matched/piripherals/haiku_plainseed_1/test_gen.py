"""Unit tests for piripherals.mpr121.MPR121.

Every test drives the sensor through an in-memory fake I2C bus, so no hardware,
smbus, RPi.GPIO or background thread is needed. Handlers are disabled
(handlers=0) so no IRQ handler or poller thread is started.
"""

import pytest

from piripherals.mpr121 import (
    ACNF_C0, ACNF_LSL, ACNF_TL, ACNF_USL, AFE1, CDC, CDT, DEB, ECR, EBL, EFD,
    ETS, GPIO_CLR, GPIO_CTL0, GPIO_CTL1, GPIO_DAT, GPIO_DIR, GPIO_EN, GPIO_SET,
    MHD, MHDX, NCH, OOR, RTH, SRESET, TTH, MPR121,
)


class FakeDevice:
    """Stands in for one I2C slave and records every write."""

    def __init__(self):
        self.regs = {}    # byte registers, written and read back
        self.words = {}   # value returned by read_word(reg)
        self.blocks = {}  # value returned by read_block(reg, n)
        self.log = []     # (kind, register, value) for each write

    def write_byte(self, reg, value):
        self.log.append(('byte', reg, value))
        self.regs[reg] = value

    def read_byte(self, reg):
        return self.regs.get(reg, 0)

    def write_word(self, reg, value):
        self.log.append(('word', reg, value))

    def read_word(self, reg):
        return self.words.get(reg, 0)

    def write_block(self, reg, values):
        self.log.append(('block', reg, list(values)))

    def read_block(self, reg, length):
        return list(self.blocks.get(reg, [0] * length))


class FakeBus:
    """Quacks like piripherals.bus.Bus. MPR121 uses it because it has a ``device`` attribute."""

    def __init__(self):
        self.addresses = []
        self.device_obj = FakeDevice()

    def device(self, addr):
        self.addresses.append(addr)
        return self.device_obj


def make_sensor(**kwargs):
    """Build an MPR121 on a fake bus with no IRQ/poller, and no setup or reset unless asked."""
    kwargs.setdefault('handlers', 0)
    kwargs.setdefault('setup', 0)
    kwargs.setdefault('reset', 0)
    bus = FakeBus()
    sensor = MPR121(bus=bus, **kwargs)
    return sensor, bus.device_obj


def test_init_without_setup_or_reset_writes_nothing():
    sensor, dev = make_sensor(setup=0, reset=0)
    assert dev.log == []
    assert sensor.overcurrent is False
    assert sensor.is_touched(0) is False


def test_init_binds_requested_address():
    bus = FakeBus()
    MPR121(bus=bus, addr=0x5b, handlers=0, setup=0, reset=0)
    assert bus.addresses == [0x5b]


def test_init_reset_only_writes_soft_reset():
    _, dev = make_sensor(setup=0, reset=1)
    assert dev.log == [('byte', SRESET, 0x63)]


def test_configure_packs_cl_prox_touch_bits():
    sensor, dev = make_sensor()
    sensor.configure(cl=1, prox=2, touch=5)
    assert dev.log == [('byte', ECR, (1 << 6) | (2 << 4) | 5)]
    sensor.configure(cl=0, prox=0, touch=0)
    assert dev.regs[ECR] == 0


def test_touched_masks_to_twelve_channels():
    sensor, dev = make_sensor()
    dev.words[ETS] = 0xFFFF
    assert sensor.touched(raise_on_failure=0) == 0x1FFF


def test_touched_raises_on_overcurrent():
    sensor, dev = make_sensor()
    dev.words[ETS] = 0x8000
    with pytest.raises(Exception, match='overcurrent'):
        sensor.touched()


def test_touched_raises_when_out_of_range_bits_set():
    sensor, dev = make_sensor()
    dev.words[OOR] = 0x0004
    with pytest.raises(Exception, match='out of range'):
        sensor.touched()


def test_out_of_range_reports_auto_config_failure():
    sensor, dev = make_sensor()
    dev.words[OOR] = 0x8000
    with pytest.raises(Exception, match='auto config failed'):
        sensor.out_of_range()
    dev.words[OOR] = 0x4000
    with pytest.raises(Exception, match='auto reconfig failed'):
        sensor.out_of_range()
    assert sensor.out_of_range(raise_on_failure=0) == 0


def test_electrode_data_combines_little_endian_pairs():
    sensor, dev = make_sensor()
    raw = [0] * (2 * NCH)
    raw[0], raw[1] = 0x34, 0x02    # channel 0 -> 0x234
    raw[24], raw[25] = 0xFF, 0x03  # channel 12 -> 0x3FF
    dev.blocks[EFD] = raw
    data = sensor.electrode_data()
    assert len(data) == NCH
    assert data[0] == 0x234
    assert data[12] == 0x3FF
    assert data[1:12] == [0] * 11


def test_baseline_read_shifts_raw_values():
    sensor, dev = make_sensor()
    dev.blocks[EBL] = list(range(NCH))
    assert sensor.baseline() == [b << 2 for b in range(NCH)]


def test_baseline_write_uses_mhd_register():
    sensor, dev = make_sensor()
    sensor.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    sensor.baseline(rft=1, mhd=6, nhd=2, ncl=4, fdl=21)
    sensor.baseline(rft=0, mhd=1, nhd=1, ncl=3, fdl=20, prox=1)
    assert dev.log == [
        ('block', MHD, [5, 1, 3, 20]),
        ('block', MHD + 4, [6, 2, 4, 21]),
        ('block', MHDX, [1, 1, 3, 20]),
    ]


def test_threshold_default_release_is_60_percent():
    sensor, dev = make_sensor()
    sensor.threshold(touch=60)
    for ch in range(NCH):
        assert dev.regs[TTH + 2 * ch] == 60
        assert dev.regs[RTH + 2 * ch] == 36
    sensor.threshold(touch=40, release=10, channel=4)
    assert dev.regs[TTH + 8] == 40
    assert dev.regs[RTH + 8] == 10


def test_debounce_release_defaults_to_touch():
    sensor, dev = make_sensor()
    sensor.debounce(3)
    assert dev.regs[DEB] == (3 << 4) | 3
    sensor.debounce(touch=1, release=4)
    assert dev.regs[DEB] == (4 << 4) | 1


def test_filter_packs_afe1_word():
    sensor, dev = make_sensor()
    sensor.filter()  # defaults: cdc=16, cdt=1, ffi=0, sfi=0, esi=4
    assert dev.log == [('word', AFE1, 0x2410)]


def test_charge_preserves_neighbour_nibble():
    sensor, dev = make_sensor()
    dev.regs[CDT + 1] = 0x70
    sensor.charge(2, cdc=1, cdt=3)  # channel 2 uses the low nibble
    assert dev.regs[CDC + 2] == 1
    assert dev.regs[CDT + 1] == 0x73
    sensor.charge(3, cdc=9, cdt=5)  # channel 3 uses the high nibble
    assert dev.regs[CDC + 3] == 9
    assert dev.regs[CDT + 1] == 0x53


def test_auto_config_default_register_values():
    sensor, dev = make_sensor()
    sensor.auto_config()
    assert ('word', ACNF_C0, 0x076F) in dev.log
    assert dev.regs[ACNF_USL] == 200
    assert dev.regs[ACNF_LSL] == 130
    assert dev.regs[ACNF_TL] == 180


def test_gpio_setup_sets_direction_and_enable_bits():
    sensor, dev = make_sensor()
    dev.regs[GPIO_DIR] = 0x00
    dev.regs[GPIO_EN] = 0x00
    sensor.gpio_setup(channel=5, output=True, mode=0, enable=1)
    assert dev.regs[GPIO_DIR] == 0b10
    assert dev.regs[GPIO_EN] == 0b10
    assert (dev.regs[GPIO_CTL0] & 0b10) == 0
    assert (dev.regs[GPIO_CTL1] & 0b10) == 0

    dev.regs[GPIO_DIR] = 0xFF
    sensor.gpio_setup(channel=4, output=False, mode=2, enable=1)
    assert dev.regs[GPIO_DIR] == 0xFE
    assert (dev.regs[GPIO_CTL0] & 0b01) == 0b01
    assert (dev.regs[GPIO_CTL1] & 0b01) == 0


@pytest.mark.parametrize('channel', [0, 3, 12])
def test_gpio_setup_rejects_channels_outside_4_to_11(channel):
    sensor, dev = make_sensor()
    with pytest.raises(AssertionError):
        sensor.gpio_setup(channel, output=True)
    assert dev.log == []


def test_gpio_set_clear_and_status_use_channel_offset():
    sensor, dev = make_sensor()
    sensor.gpio_set(6, True)
    sensor.gpio_set(11, False)
    assert dev.log == [('byte', GPIO_SET, 1 << 2), ('byte', GPIO_CLR, 1 << 7)]
    dev.regs[GPIO_DAT] = 0xA5
    assert sensor.gpio_status() == 0xA5


def test_setup_defaults_run_mode_with_twelve_channels():
    sensor, dev = make_sensor(setup=1, reset=1)
    assert dev.log[0] == ('byte', SRESET, 0x63)
    ecr_writes = [v for (k, r, v) in dev.log if k == 'byte' and r == ECR]
    assert ecr_writes == [0x00, 0x0C]  # stop mode, then run with 12 electrodes
    assert dev.regs[TTH + 2 * 12] == 12
    assert dev.regs[RTH + 2 * 12] == 7
    assert dev.regs[DEB] == 0x22
    assert ('block', MHD, [5, 1, 3, 20]) in dev.log


def test_setup_channels_prox_and_no_auto_config():
    sensor, dev = make_sensor(setup=1, reset=0, auto_config=0, channels=6, prox=1)
    assert not [e for e in dev.log if e[0] == 'word' and e[1] == ACNF_C0]
    assert dev.regs.get(ACNF_USL) is None
    assert dev.regs[ECR] == (1 << 4) | 6


def test_update_calls_handler_only_on_change():
    sensor, dev = make_sensor()
    calls = []
    sensor.on_touch(0, lambda state, ch: calls.append((state, ch)))
    dev.words[ETS] = 0b1
    sensor.update_touch_state()
    sensor.update_touch_state()  # same state, no new callback
    assert calls == [(True, 0)]
    assert sensor.is_touched(0) is True
    dev.words[ETS] = 0
    sensor.update_touch_state()
    assert calls == [(True, 0), (False, 0)]
    assert sensor.is_touched(0) is False


def test_handler_exception_is_contained():
    sensor, dev = make_sensor()

    def broken(state, channel):
        raise ValueError('boom')

    sensor.on_touch(2, broken)
    dev.words[ETS] = 0b100
    sensor.update_touch_state()  # must not propagate
    assert sensor.is_touched(2) is True
