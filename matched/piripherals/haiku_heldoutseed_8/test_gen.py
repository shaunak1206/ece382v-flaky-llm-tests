"""Unit tests for piripherals.mpr121.MPR121.

The tests use an in-memory fake of the SMBus device so that no real I2C
hardware (and no RPi.GPIO) is required.
"""

import pytest

from piripherals.mpr121 import (
    MPR121,
    NCH,
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
    GPIO_DAT,
    GPIO_DIR,
    GPIO_EN,
    GPIO_SET,
    GPIO_CLR,
    ACNF_C0,
    ACNF_USL,
    ACNF_LSL,
    ACNF_TL,
    SRESET,
)


class FakeDevice:
    """In-memory stand-in for an SMBus device at a single address."""

    def __init__(self):
        self.mem = {}
        self.writes = []

    # helpers for test setup
    def set_byte(self, reg, value):
        self.mem[reg] = value & 0xFF

    def set_word(self, reg, value):
        self.mem[reg] = value & 0xFF
        self.mem[reg + 1] = (value >> 8) & 0xFF

    def get_word(self, reg):
        return self.mem.get(reg, 0) | (self.mem.get(reg + 1, 0) << 8)

    # SMBus-like API used by MPR121
    def read_byte(self, reg):
        return self.mem.get(reg, 0)

    def write_byte(self, reg, value):
        self.writes.append(("byte", reg, value))
        self.mem[reg] = value & 0xFF

    def read_word(self, reg):
        return self.get_word(reg)

    def write_word(self, reg, value):
        self.writes.append(("word", reg, value))
        self.set_word(reg, value)

    def read_block(self, reg, count):
        return [self.mem.get(reg + i, 0) for i in range(count)]

    def write_block(self, reg, data):
        self.writes.append(("block", reg, list(data)))
        for i, v in enumerate(data):
            self.mem[reg + i] = v & 0xFF

    def written_bytes_to(self, reg):
        return [w[2] for w in self.writes if w[0] == "byte" and w[1] == reg]


class FakeBus:
    """Mimics piripherals.bus.Bus, which exposes ``device(addr)``."""

    def __init__(self, device):
        self._device = device
        self.addresses = []

    def device(self, addr):
        self.addresses.append(addr)
        return self._device


@pytest.fixture
def dev():
    return FakeDevice()


@pytest.fixture
def mpr(dev):
    # handlers=0 avoids IRQ/poller threads, setup=0 and reset=0 avoid
    # initial bus traffic so each test controls its own writes.
    return MPR121(bus=FakeBus(dev), addr=0x5A, handlers=0, setup=0, reset=0)


def test_constructor_stores_address_and_uses_bus_device(dev):
    bus = FakeBus(dev)
    m = MPR121(bus=bus, addr=0x5B, handlers=0, setup=0, reset=0)
    assert m.addr == 0x5B
    assert bus.addresses == [0x5B]
    assert m.overcurrent is False


def test_constructor_without_setup_or_reset_does_no_writes(dev):
    MPR121(bus=FakeBus(dev), handlers=0, setup=0, reset=0)
    assert dev.writes == []


def test_constructor_with_reset_only_soft_resets(dev):
    MPR121(bus=FakeBus(dev), handlers=0, setup=0, reset=1)
    assert dev.writes == [("byte", SRESET, 0x63)]


def test_constructor_default_setup_resets_and_enables_12_channels(dev):
    MPR121(bus=FakeBus(dev), handlers=0)
    # first write is the soft reset
    assert dev.writes[0] == ("byte", SRESET, 0x63)
    # last electrode configuration enables 12 channels in run mode
    assert dev.written_bytes_to(ECR)[-1] == 12


def test_setup_with_custom_channels_and_prox(mpr, dev):
    mpr.setup(reset=0, channels=5, prox=1, auto_config=0)
    assert dev.written_bytes_to(ECR)[-1] == (3 << 6) | (1 << 4) | 5
    assert not any(w[1] == ACNF_C0 for w in dev.writes)


def test_reset_writes_soft_reset_register(mpr, dev):
    mpr.reset()
    assert dev.writes == [("byte", SRESET, 0x63)]


def test_configure_packs_calibration_proximity_and_touch_bits(mpr, dev):
    mpr.configure(cl=3, prox=2, touch=5)
    assert dev.writes == [("byte", ECR, (3 << 6) | (2 << 4) | 5)]


def test_configure_stop_mode_writes_zero(mpr, dev):
    mpr.configure(cl=0, prox=0, touch=0)
    assert dev.written_bytes_to(ECR) == [0]


def test_touched_masks_to_12_channels_without_raising(mpr, dev):
    dev.set_word(ETS, 0xFFFF)  # overcurrent bit set, but raise disabled
    assert mpr.touched(raise_on_failure=0) == 0x1FFF


def test_touched_returns_status_bits(mpr, dev):
    dev.set_word(ETS, 0x0A05)
    assert mpr.touched() == 0x0A05


def test_touched_raises_on_overcurrent(mpr, dev):
    dev.set_word(ETS, 1 << 15)
    with pytest.raises(Exception, match="overcurrent"):
        mpr.touched()


def test_touched_raises_when_out_of_range(mpr, dev):
    dev.set_word(ETS, 0)
    dev.set_word(OOR, 0x0004)
    with pytest.raises(Exception, match="out of range"):
        mpr.touched()


def test_out_of_range_raises_on_auto_config_failure(mpr, dev):
    dev.set_word(OOR, 1 << 15)
    with pytest.raises(Exception, match="auto config failed"):
        mpr.out_of_range()


def test_out_of_range_raises_on_auto_reconfig_failure(mpr, dev):
    dev.set_word(OOR, 1 << 14)
    with pytest.raises(Exception, match="auto reconfig failed"):
        mpr.out_of_range()


def test_update_touch_state_invokes_handlers_only_on_change(mpr, dev):
    calls = []
    mpr.on_touch(0, lambda t, c: calls.append((c, t)))
    mpr.on_touch(2, lambda t, c: calls.append((c, t)))

    dev.set_word(ETS, 0b101)
    mpr.update_touch_state()
    assert sorted(calls) == [(0, True), (2, True)]

    calls.clear()
    mpr.update_touch_state()  # no change, no handler calls
    assert calls == []

    dev.set_word(ETS, 0b001)  # channel 2 released
    mpr.update_touch_state()
    assert calls == [(2, False)]
    assert mpr.is_touched(0) is True
    assert mpr.is_touched(2) is False


def test_handler_taking_only_state_gets_called_with_one_argument(mpr, dev):
    received = []
    mpr.on_touch(3, lambda state: received.append(state))
    dev.set_word(ETS, 1 << 3)
    mpr.update_touch_state()
    assert received == [True]


def test_is_touched_reflects_last_update(mpr, dev):
    dev.set_word(ETS, 1 << 7)
    mpr.update_touch_state()
    assert mpr.is_touched(7) is True
    assert mpr.is_touched(6) is False


def test_electrode_data_decodes_little_endian_10bit_values(mpr, dev):
    for i in range(NCH):
        dev.set_byte(EFD + 2 * i, i)
        dev.set_byte(EFD + 2 * i + 1, 0x02)
    data = mpr.electrode_data()
    assert len(data) == NCH
    assert data[0] == 0x200
    assert data[5] == 0x205


def test_baseline_read_returns_values_shifted_to_10_bits(mpr, dev):
    for i in range(NCH):
        dev.set_byte(EBL + i, 0x10 + i)
    result = mpr.baseline()
    assert result == [(0x10 + i) << 2 for i in range(NCH)]


def test_baseline_write_for_rising_scenario_uses_mhd_block(mpr, dev):
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert dev.writes == [("block", MHD, [5, 1, 3, 20])]


def test_baseline_write_for_falling_proximity_uses_mhdx_block(mpr, dev):
    mpr.baseline(rft=1, mhd=1, nhd=2, ncl=4, fdl=30, prox=1)
    assert dev.writes == [("block", MHDX + 4, [1, 2, 4, 30])]


def test_baseline_touched_scenario_omits_mhd(mpr, dev):
    mpr.baseline(rft=2, nhd=7, ncl=8, fdl=9)
    assert dev.writes == [("block", MHD + 8, [7, 8, 9])]


def test_threshold_all_channels_uses_default_release(mpr, dev):
    mpr.threshold(40)
    touch_writes = [w for w in dev.writes if w[0] == "byte" and w[1] >= TTH and w[1] < TTH + 2 * NCH and (w[1] - TTH) % 2 == 0]
    assert len(touch_writes) == NCH
    assert dev.mem[TTH] == 40
    assert dev.mem[RTH] == int(0.6 * 40)
    assert dev.mem[TTH + 2 * 12] == 40


def test_threshold_single_channel_with_explicit_release(mpr, dev):
    mpr.threshold(10, release=6, channel=12)
    assert ("byte", TTH + 24, 10) in dev.writes
    assert ("byte", RTH + 24, 6) in dev.writes


def test_debounce_packs_release_and_touch_nibbles(mpr, dev):
    mpr.debounce(touch=2)
    assert dev.writes == [("byte", DEB, 0x22)]
    dev.writes.clear()
    mpr.debounce(touch=3, release=5)
    assert dev.writes == [("byte", DEB, 0x53)]


def test_filter_packs_afe1_word(mpr, dev):
    mpr.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    assert dev.writes == [("word", AFE1, 0x285E)]


def test_charge_preserves_other_channel_nibble(mpr, dev):
    # channel 5 uses the upper nibble of CDT + 2; lower nibble belongs to channel 4
    dev.set_byte(CDT + 2, 0x05)
    mpr.charge(5, cdc=10, cdt=3)
    assert ("byte", CDC + 5, 10) in dev.writes
    assert dev.mem[CDT + 2] == 0x35


def test_auto_config_default_encoding(mpr, dev):
    mpr.auto_config()
    assert ("word", ACNF_C0, 0x076F) in dev.writes
    assert dev.written_bytes_to(ACNF_USL) == [200]
    assert dev.written_bytes_to(ACNF_LSL) == [130]
    assert dev.written_bytes_to(ACNF_TL) == [180]


def test_gpio_set_high_uses_set_register(mpr, dev):
    mpr.gpio_set(6, True)
    assert dev.writes == [("byte", GPIO_SET, 1 << 2)]


def test_gpio_set_low_uses_clear_register(mpr, dev):
    mpr.gpio_set(11, False)
    assert dev.writes == [("byte", GPIO_CLR, 1 << 7)]


def test_gpio_set_rejects_channel_outside_range(mpr):
    with pytest.raises(AssertionError):
        mpr.gpio_set(3, True)


def test_gpio_setup_configures_direction_and_enable(mpr, dev, capsys):
    mpr.gpio_setup(5, output=True, mode=0, enable=1)
    capsys.readouterr()  # discard debug prints
    assert dev.mem[GPIO_DIR] & (1 << 1)
    assert dev.mem[GPIO_EN] & (1 << 1)
    assert not dev.mem.get(GPIO_CTL0, 0) & (1 << 1)
    assert not dev.mem.get(GPIO_CTL1, 0) & (1 << 1)


def test_gpio_status_reads_data_register(mpr, dev):
    dev.set_byte(GPIO_DAT, 0b1010_0001)
    assert mpr.gpio_status() == 0b1010_0001
