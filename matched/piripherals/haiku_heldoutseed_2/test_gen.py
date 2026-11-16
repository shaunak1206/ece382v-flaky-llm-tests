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
    NCH,
)


class FakeDevice:
    """Minimal stand-in for an smbus device that records all traffic."""

    def __init__(self):
        self.regs = {}      # byte registers
        self.words = {}     # word reads, keyed by register
        self.blocks = {}    # block reads, keyed by register
        self.writes = []    # (kind, reg, value) tuples in order

    def write_byte(self, reg, value):
        self.writes.append(('byte', reg, value))
        self.regs[reg] = value

    def write_word(self, reg, value):
        self.writes.append(('word', reg, value))

    def write_block(self, reg, data):
        self.writes.append(('block', reg, list(data)))

    def read_byte(self, reg):
        return self.regs.get(reg, 0)

    def read_word(self, reg):
        return self.words.get(reg, 0)

    def read_block(self, reg, length):
        return list(self.blocks.get(reg, [0] * length))[:length]

    def writes_to(self, reg, kind=None):
        return [w[2] for w in self.writes
                if w[1] == reg and (kind is None or w[0] == kind)]


class FakeBus:
    """Stand-in for piripherals.bus.Bus; MPR121 only calls ``device(addr)``."""

    def __init__(self, device):
        self._device = device
        self.addrs = []

    def device(self, addr):
        self.addrs.append(addr)
        return self._device


@pytest.fixture
def dev():
    return FakeDevice()


@pytest.fixture
def bus(dev):
    return FakeBus(dev)


@pytest.fixture
def mpr(bus):
    # No setup, no reset, no IRQ/poller so no hardware or threads are involved.
    return MPR121(bus=bus, setup=0, reset=0, handlers=0)


def test_init_binds_device_to_address(bus, mpr):
    assert bus.addrs == [0x5a]
    assert mpr.addr == 0x5a
    assert mpr.overcurrent is False


def test_init_without_setup_or_reset_writes_nothing(dev, mpr):
    assert dev.writes == []


def test_init_with_reset_only_sends_soft_reset(bus, dev):
    MPR121(bus=bus, setup=0, reset=1, handlers=0)
    assert dev.writes_to(SRESET, 'byte') == [0x63]
    assert len(dev.writes) == 1


def test_init_with_default_setup_configures_run_mode(bus, dev):
    MPR121(bus=bus, handlers=0)
    # setup() first configures stop mode (0) then run mode with 12 channels.
    assert dev.writes_to(ECR, 'byte') == [0, 12]
    assert dev.writes_to(SRESET, 'byte') == [0x63]


def test_reset_writes_soft_reset_register(dev, mpr):
    mpr.reset()
    assert dev.writes == [('byte', SRESET, 0x63)]


def test_configure_packs_cl_prox_touch_into_ecr(dev, mpr):
    mpr.configure(cl=1, prox=2, touch=5)
    assert dev.writes_to(ECR, 'byte') == [(1 << 6) | (2 << 4) | 5]


def test_update_touch_state_detects_channels_and_is_touched(dev, mpr):
    dev.words[ETS] = 0b101
    mpr.update_touch_state()
    assert mpr.is_touched(0) is True
    assert mpr.is_touched(1) is False
    assert mpr.is_touched(2) is True


def test_touched_masks_to_12_bits(dev, mpr):
    dev.words[ETS] = 0x2001  # bit 13 is outside the 12 electrode bits
    assert mpr.touched() == 0x0001


def test_touched_raises_on_overcurrent(dev, mpr):
    dev.words[ETS] = 0x8000
    with pytest.raises(Exception, match='overcurrent'):
        mpr.touched()


def test_touched_raises_when_out_of_range(dev, mpr):
    dev.words[ETS] = 0x0000
    dev.words[OOR] = 0x0001
    with pytest.raises(Exception, match='out of range'):
        mpr.touched()


def test_touched_without_raise_ignores_failures(dev, mpr):
    dev.words[ETS] = 0x8001
    assert mpr.touched(raise_on_failure=0) == 0x0001


def test_out_of_range_raises_on_auto_config_failure(dev, mpr):
    dev.words[OOR] = 0x8000
    with pytest.raises(Exception, match='auto config failed'):
        mpr.out_of_range()


def test_out_of_range_raises_on_auto_reconfig_failure(dev, mpr):
    dev.words[OOR] = 0x4000
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr.out_of_range()


def test_out_of_range_returns_masked_status(dev, mpr):
    dev.words[OOR] = 0x4003
    assert mpr.out_of_range(raise_on_failure=0) == 0x0003


def test_electrode_data_decodes_little_endian_pairs(dev, mpr):
    raw = []
    for i in range(NCH):
        raw += [(i * 3) & 0xff, (i & 0x03)]  # low byte, high byte
    dev.blocks[EFD] = raw
    data = mpr.electrode_data()
    assert len(data) == NCH
    assert data[0] == 0
    assert data[1] == (1 << 8) | 3
    assert data[5] == (2 << 8) | 15


def test_baseline_read_shifts_values_left_by_two(dev, mpr):
    dev.blocks[EBL] = [1] * NCH
    assert mpr.baseline() == [4] * NCH


def test_baseline_write_rising_uses_mhd_register(dev, mpr):
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert dev.writes == [('block', MHD, [5, 1, 3, 20])]


def test_baseline_write_touched_uses_three_values(dev, mpr):
    mpr.baseline(rft=2, mhd=5, nhd=1, ncl=3, fdl=20)
    assert dev.writes == [('block', MHD + 8, [1, 3, 20])]


def test_baseline_write_proximity_uses_mhdx_register(dev, mpr):
    mpr.baseline(rft=1, mhd=7, nhd=2, ncl=4, fdl=9, prox=1)
    assert dev.writes == [('block', MHDX + 4, [7, 2, 4, 9])]


def test_threshold_single_channel_defaults_release_to_60_percent(dev, mpr):
    mpr.threshold(touch=50, channel=3)
    assert dev.writes_to(TTH + 6) == [50]
    assert dev.writes_to(RTH + 6) == [30]


def test_threshold_without_channel_applies_to_all_channels(dev, mpr):
    mpr.threshold(touch=10)
    for i in range(NCH):
        assert dev.writes_to(TTH + 2 * i) == [10]
        assert dev.writes_to(RTH + 2 * i) == [6]


def test_debounce_packs_touch_and_release_nibbles(dev, mpr):
    mpr.debounce(touch=2, release=5)
    assert dev.writes_to(DEB) == [(5 << 4) | 2]


def test_debounce_release_defaults_to_touch(dev, mpr):
    mpr.debounce(3)
    assert dev.writes_to(DEB) == [0x33]


def test_filter_packs_afe1_word(dev, mpr):
    mpr.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    lb = (1 << 6) | 30
    hb = (1 << 5) | (1 << 3) | 0
    assert dev.writes_to(AFE1, 'word') == [(hb << 8) | lb]


def test_charge_updates_only_its_nibble(dev, mpr):
    dev.regs[CDT + 1] = 0xff  # channel 3 uses the upper nibble of CDT+1
    mpr.charge(channel=3, cdc=10, cdt=2)
    assert dev.writes_to(CDC + 3) == [10]
    assert dev.writes_to(CDT + 1) == [0x2f]


def test_auto_config_default_writes(dev, mpr):
    mpr.auto_config()
    lb = (1 << 6) | (2 << 4) | (3 << 2) | (1 << 1) | 1
    hb = (0 << 7) | (1 << 2) | (1 << 1) | 1
    assert dev.writes_to(ACNF_C0, 'word') == [(hb << 8) | lb]
    assert dev.writes_to(ACNF_USL) == [200]
    assert dev.writes_to(ACNF_LSL) == [130]
    assert dev.writes_to(ACNF_TL) == [180]


def test_gpio_set_high_and_low_use_set_and_clear_registers(dev, mpr):
    mpr.gpio_set(5, True)
    mpr.gpio_set(5, False)
    assert dev.writes_to(GPIO_SET) == [1 << 1]
    assert dev.writes_to(GPIO_CLR) == [1 << 1]


def test_gpio_set_rejects_non_gpio_channel(mpr):
    with pytest.raises(AssertionError):
        mpr.gpio_set(3, True)


def test_gpio_setup_output_mode_sets_direction_and_enable(dev, mpr):
    mpr.gpio_setup(channel=4, output=1, mode=3, enable=1)
    assert dev.regs[GPIO_CTL0] & 1 == 1
    assert dev.regs[GPIO_CTL1] & 1 == 1
    assert dev.regs[GPIO_DIR] & 1 == 1
    assert dev.regs[GPIO_EN] & 1 == 1


def test_gpio_setup_maps_mode_1_to_cmos_output(dev, mpr):
    mpr.gpio_setup(channel=4, output=1, mode=1, enable=1)
    assert dev.regs[GPIO_CTL0] & 1 == 0
    assert dev.regs[GPIO_CTL1] & 1 == 0


def test_gpio_status_returns_data_register(dev, mpr):
    dev.regs[GPIO_DAT] = 0x0a
    assert mpr.gpio_status() == 0x0a


def test_touch_handler_receives_state_and_channel(dev, mpr):
    calls = []
    mpr.on_touch(0, lambda state, channel: calls.append((state, channel)))

    dev.words[ETS] = 0b1
    mpr.update_touch_state()
    dev.words[ETS] = 0b0
    mpr.update_touch_state()

    assert calls == [(True, 0), (False, 0)]


def test_no_handler_call_when_touch_state_unchanged(dev, mpr):
    calls = []
    mpr.on_touch(1, lambda state, channel: calls.append(state))

    dev.words[ETS] = 0b10
    mpr.update_touch_state()
    mpr.update_touch_state()

    assert calls == [True]
