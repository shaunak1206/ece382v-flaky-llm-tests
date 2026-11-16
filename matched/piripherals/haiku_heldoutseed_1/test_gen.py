"""Unit tests for piripherals.mpr121.MPR121.

The tests use a fake I2C bus so no hardware, smbus or GPIO library is needed.
The device is constructed with handlers=0, setup=0 and reset=0, which avoids
IRQ handler and polling threads and keeps construction side-effect free.
"""

import pytest

from piripherals.mpr121 import (
    MPR121,
    ETS,
    OOR,
    EFD,
    EBL,
    MHD,
    MHDX,
    TTH,
    RTH,
    DEB,
    AFE1,
    ECR,
    CDC,
    CDT,
    GPIO_CTL0,
    GPIO_CTL1,
    GPIO_DIR,
    GPIO_EN,
    GPIO_DAT,
    GPIO_SET,
    GPIO_CLR,
    ACNF_C0,
    ACNF_USL,
    ACNF_LSL,
    ACNF_TL,
    SRESET,
    NCH,
)


class FakeDevice:
    """Records all bus traffic and serves reads from simple dictionaries."""

    def __init__(self):
        self.bytes = {}
        self.words = {}
        self.byte_writes = []
        self.word_writes = []
        self.block_writes = []

    def write_byte(self, reg, value):
        self.byte_writes.append((reg, value))
        self.bytes[reg] = value & 0xFF

    def read_byte(self, reg):
        return self.bytes.get(reg, 0)

    def write_word(self, reg, value):
        self.word_writes.append((reg, value))
        self.words[reg] = value

    def read_word(self, reg):
        return self.words.get(reg, 0)

    def read_block(self, reg, length):
        return [self.bytes.get(reg + i, 0) for i in range(length)]

    def write_block(self, reg, data):
        self.block_writes.append((reg, list(data)))


class FakeBus:
    """Mimics the object returned by piripherals.bus.Bus (has a device() method)."""

    def __init__(self):
        self.dev = FakeDevice()
        self.addresses = []

    def device(self, addr):
        self.addresses.append(addr)
        return self.dev


@pytest.fixture
def bus():
    return FakeBus()


@pytest.fixture
def dev(bus):
    return bus.dev


@pytest.fixture
def mpr(bus):
    return MPR121(bus=bus, handlers=0, setup=0, reset=0)


def test_constructor_binds_device_at_given_address(bus):
    m = MPR121(bus=bus, addr=0x5b, handlers=0, setup=0, reset=0)
    assert bus.addresses == [0x5b]
    assert m.addr == 0x5b
    assert m.overcurrent is False


def test_constructor_reset_only_writes_soft_reset(bus, dev):
    MPR121(bus=bus, handlers=0, setup=0, reset=1)
    assert dev.byte_writes == [(SRESET, 0x63)]


def test_constructor_without_reset_or_setup_writes_nothing(bus, dev):
    MPR121(bus=bus, handlers=0, setup=0, reset=0)
    assert dev.byte_writes == []
    assert dev.word_writes == []
    assert dev.block_writes == []


def test_reset_writes_soft_reset_register(mpr, dev):
    mpr.reset()
    assert dev.byte_writes == [(SRESET, 0x63)]


def test_configure_run_mode_packs_fields_into_ecr(mpr, dev):
    mpr.configure(cl=3, prox=0, touch=12)
    assert dev.byte_writes == [(ECR, 0xCC)]


def test_configure_stop_mode_writes_zero_touch_and_prox(mpr, dev):
    mpr.configure(cl=0, prox=0, touch=0)
    assert dev.byte_writes == [(ECR, 0x00)]


def test_touched_masks_to_13_bits_without_raising(mpr, dev):
    dev.words[ETS] = 0xE00F
    # Bits 15 and 14 set would raise with raise_on_failure, so ignore them here.
    assert mpr.touched(raise_on_failure=0) == 0x000F


def test_touched_returns_touch_bits(mpr, dev):
    dev.words[ETS] = 0x0005
    assert mpr.touched() == 0x0005


def test_touched_raises_on_overcurrent(mpr, dev):
    dev.words[ETS] = 0x8001
    with pytest.raises(Exception, match='overcurrent'):
        mpr.touched()


def test_touched_raises_when_out_of_range(mpr, dev):
    dev.words[ETS] = 0x0000
    dev.words[OOR] = 0x0001
    with pytest.raises(Exception, match='out of range'):
        mpr.touched()


def test_out_of_range_raises_on_auto_config_failure(mpr, dev):
    dev.words[OOR] = 0x8000
    with pytest.raises(Exception, match='auto config failed'):
        mpr.out_of_range()


def test_out_of_range_raises_on_auto_reconfig_failure(mpr, dev):
    dev.words[OOR] = 0x4000
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr.out_of_range()


def test_out_of_range_masks_status_when_not_raising(mpr, dev):
    dev.words[OOR] = 0xC003
    assert mpr.out_of_range(raise_on_failure=0) == 0x0003


def test_electrode_data_combines_little_endian_pairs(mpr, dev):
    dev.bytes[EFD] = 0x34
    dev.bytes[EFD + 1] = 0x02
    dev.bytes[EFD + 2] = 0xFF
    dev.bytes[EFD + 3] = 0x03
    data = mpr.electrode_data()
    assert len(data) == NCH
    assert data[0] == 0x234
    assert data[1] == 0x3FF
    assert all(v == 0 for v in data[2:])


def test_baseline_read_returns_values_shifted_left_by_two(mpr, dev):
    dev.bytes[EBL] = 0x10
    dev.bytes[EBL + 1] = 0x3F
    baseline = mpr.baseline()
    assert len(baseline) == NCH
    assert baseline[0] == 0x40
    assert baseline[1] == 0xFC


def test_baseline_write_rft0_uses_mhd_block(mpr, dev):
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert dev.block_writes == [(MHD, [5, 1, 3, 20])]


def test_baseline_write_rft2_omits_mhd_and_uses_offset_register(mpr, dev):
    mpr.baseline(rft=2, nhd=7, ncl=8, fdl=9)
    assert dev.block_writes == [(MHD + 4 * 2, [7, 8, 9])]


def test_baseline_write_proximity_uses_mhdx_register(mpr, dev):
    mpr.baseline(rft=1, mhd=1, nhd=1, ncl=3, fdl=20, prox=1)
    assert dev.block_writes == [(MHDX + 4, [1, 1, 3, 20])]


def test_threshold_single_channel_derives_release(mpr, dev):
    mpr.threshold(touch=50, channel=3)
    assert (TTH + 6, 50) in dev.byte_writes
    assert (RTH + 6, 30) in dev.byte_writes  # int(0.6 * 50)


def test_threshold_all_channels_with_explicit_release(mpr, dev):
    mpr.threshold(touch=40, release=10)
    for ch in range(NCH):
        assert (TTH + 2 * ch, 40) in dev.byte_writes
        assert (RTH + 2 * ch, 10) in dev.byte_writes


def test_debounce_packs_touch_and_release_counts(mpr, dev):
    mpr.debounce(touch=2)
    assert dev.byte_writes[-1] == (DEB, 0x22)
    mpr.debounce(touch=2, release=5)
    assert dev.byte_writes[-1] == (DEB, 0x52)


def test_filter_packs_afe1_word(mpr, dev):
    mpr.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    assert dev.word_writes == [(AFE1, 0x285E)]


def test_charge_even_channel_sets_cdt_low_nibble(mpr, dev):
    dev.bytes[CDT + 2] = 0x0F
    mpr.charge(channel=4, cdc=10, cdt=2)
    assert (CDC + 4, 10) in dev.byte_writes
    assert dev.byte_writes[-1] == (CDT + 2, 0x0F | 0x02)


def test_charge_odd_channel_sets_cdt_high_nibble(mpr, dev):
    dev.bytes[CDT + 2] = 0x0F
    mpr.charge(channel=5, cdc=20, cdt=3)
    assert (CDC + 5, 20) in dev.byte_writes
    assert dev.byte_writes[-1] == (CDT + 2, (0x0F & ~0xF0) | (3 << 4))


def test_auto_config_defaults_write_expected_registers(mpr, dev):
    mpr.auto_config()
    assert dev.word_writes == [(ACNF_C0, 0x076F)]
    assert (ACNF_USL, 200) in dev.byte_writes
    assert (ACNF_LSL, 130) in dev.byte_writes
    assert (ACNF_TL, 180) in dev.byte_writes


def test_gpio_set_high_writes_set_register_with_channel_bit(mpr, dev):
    mpr.gpio_set(channel=6, value=True)
    assert dev.byte_writes == [(GPIO_SET, 1 << 2)]


def test_gpio_set_low_writes_clear_register_with_channel_bit(mpr, dev):
    mpr.gpio_set(channel=6, value=False)
    assert dev.byte_writes == [(GPIO_CLR, 1 << 2)]


def test_gpio_set_rejects_channel_outside_gpio_range(mpr):
    with pytest.raises(AssertionError):
        mpr.gpio_set(channel=3, value=True)


def test_gpio_status_reads_data_register(mpr, dev):
    dev.bytes[GPIO_DAT] = 0xA5
    assert mpr.gpio_status() == 0xA5


def test_gpio_setup_output_sets_direction_and_enable(mpr, dev):
    mpr.gpio_setup(channel=4, output=1, mode=0, enable=1)
    assert dev.bytes[GPIO_DIR] & 0x01 == 1
    assert dev.bytes[GPIO_EN] & 0x01 == 1
    assert dev.bytes[GPIO_CTL0] & 0x01 == 0
    assert dev.bytes[GPIO_CTL1] & 0x01 == 0


def test_gpio_setup_open_drain_mode_sets_mode_bits(mpr, dev):
    mpr.gpio_setup(channel=5, output=1, mode=3, enable=1)
    assert dev.bytes[GPIO_CTL0] == 0b10
    assert dev.bytes[GPIO_CTL1] == 0b10


def test_is_touched_reflects_cached_touch_state(mpr):
    mpr._touched = 0b101
    assert mpr.is_touched(0) is True
    assert mpr.is_touched(1) is False
    assert mpr.is_touched(2) is True


def test_update_touch_state_invokes_handler_on_change(mpr, dev):
    calls = []
    mpr.on_touch(2, lambda *args: calls.append(args))

    dev.words[ETS] = 1 << 2
    mpr.update_touch_state()
    assert calls == [(True, 2)]

    dev.words[ETS] = 0
    mpr.update_touch_state()
    assert calls == [(True, 2), (False, 2)]


def test_update_touch_state_only_calls_handlers_for_changed_channels(mpr, dev):
    calls_2, calls_7 = [], []
    mpr.on_touch(2, lambda *args: calls_2.append(args))
    mpr.on_touch(7, lambda *args: calls_7.append(args))

    dev.words[ETS] = (1 << 2) | (1 << 5)
    mpr.update_touch_state()
    assert calls_2 == [(True, 2)]
    assert calls_7 == []
