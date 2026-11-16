import pytest

from piripherals.mpr121 import (
    MPR121,
    ACNF_C0,
    ACNF_LSL,
    ACNF_TL,
    ACNF_USL,
    AFE1,
    CDC,
    CDT,
    DEB,
    ECR,
    EBL,
    EFD,
    ETS,
    GPIO_CLR,
    GPIO_CTL0,
    GPIO_CTL1,
    GPIO_DIR,
    GPIO_EN,
    GPIO_SET,
    MHD,
    MHDX,
    NCH,
    OOR,
    RTH,
    SRESET,
    TTH,
)


class FakeDevice:
    """Minimal in-memory stand-in for a bus device, keyed by register."""

    def __init__(self):
        self.bytes = {}
        self.words = {}
        self.blocks = {}
        self.log = []

    def write_byte(self, reg, value):
        self.log.append(('write_byte', reg, value))
        self.bytes[reg] = value

    def read_byte(self, reg):
        return self.bytes.get(reg, 0)

    def write_word(self, reg, value):
        self.log.append(('write_word', reg, value))
        self.words[reg] = value

    def read_word(self, reg):
        return self.words.get(reg, 0)

    def write_block(self, reg, data):
        self.log.append(('write_block', reg, list(data)))

    def read_block(self, reg, length):
        return list(self.blocks.get(reg, [0] * length))


class FakeBus:
    """Bus object exposing ``device(addr)``, so MPR121 skips creating a real Bus."""

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
    # No IRQ/polling, no setup and no reset, so the device is quiet until a test drives it.
    return MPR121(bus=FakeBus(dev), handlers=0, setup=0, reset=0)


def test_reset_writes_soft_reset_command(mpr, dev):
    mpr.reset()
    assert dev.log == [('write_byte', SRESET, 0x63)]


def test_constructor_reset_only_when_setup_disabled(dev):
    MPR121(bus=FakeBus(dev), handlers=0, setup=0, reset=1)
    assert dev.log == [('write_byte', SRESET, 0x63)]


def test_constructor_setup_applies_default_configuration(dev):
    MPR121(bus=FakeBus(dev), handlers=0, setup=1, reset=1)
    assert dev.log[0] == ('write_byte', SRESET, 0x63)
    # ECR is written once to stop measurement and once to start with 12 channels.
    ecr_writes = [v for (op, reg, v) in dev.log
                  if op == 'write_byte' and reg == ECR]
    assert ecr_writes == [0, 12]


def test_configure_packs_cl_prox_touch_into_ecr(mpr, dev):
    mpr.configure(cl=1, prox=3, touch=5)
    assert dev.bytes[ECR] == (1 << 6) | (3 << 4) | 5


def test_touched_masks_result_to_13_bits(mpr, dev):
    dev.words[ETS] = 0xA123
    assert mpr.touched(raise_on_failure=0) == 0x0123


def test_touched_raises_on_overcurrent(mpr, dev):
    dev.words[ETS] = 0x8000
    with pytest.raises(Exception, match='overcurrent'):
        mpr.touched()


def test_touched_raises_when_channel_out_of_range(mpr, dev):
    dev.words[OOR] = 0x0010
    with pytest.raises(Exception, match='out of range'):
        mpr.touched()


def test_out_of_range_raises_auto_config_failed(mpr, dev):
    dev.words[OOR] = 0x8000
    with pytest.raises(Exception, match='auto config failed'):
        mpr.out_of_range()


def test_out_of_range_raises_auto_reconfig_failed(mpr, dev):
    dev.words[OOR] = 0x4000
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr.out_of_range()


def test_electrode_data_combines_little_endian_pairs(mpr, dev):
    block = [0x34, 0x02, 0xff, 0x03] + [0] * (2 * NCH - 4)
    dev.blocks[EFD] = block
    data = mpr.electrode_data()
    assert len(data) == NCH
    assert data[0] == 0x234
    assert data[1] == 0x3ff
    assert data[2:] == [0] * (NCH - 2)


def test_baseline_read_shifts_values_to_10_bit(mpr, dev):
    dev.blocks[EBL] = list(range(NCH))
    assert mpr.baseline() == [i << 2 for i in range(NCH)]


def test_baseline_rising_writes_mhd_block(mpr, dev):
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert ('write_block', MHD, [5, 1, 3, 20]) in dev.log


def test_baseline_falling_proximity_uses_prox_register(mpr, dev):
    mpr.baseline(rft=1, mhd=2, nhd=3, ncl=4, fdl=5, prox=1)
    assert ('write_block', MHDX + 4, [2, 3, 4, 5]) in dev.log


def test_threshold_default_release_is_sixty_percent(mpr, dev):
    mpr.threshold(50, channel=3)
    assert dev.bytes[TTH + 6] == 50
    assert dev.bytes[RTH + 6] == 30


def test_threshold_without_channel_writes_all_channels(mpr, dev):
    mpr.threshold(touch=10)
    for ch in range(NCH):
        assert dev.bytes[TTH + 2 * ch] == 10
        assert dev.bytes[RTH + 2 * ch] == 6


def test_debounce_release_defaults_to_touch(mpr, dev):
    mpr.debounce(3)
    assert dev.bytes[DEB] == 0x33
    mpr.debounce(touch=3, release=1)
    assert dev.bytes[DEB] == 0x13


def test_filter_packs_afe1_word(mpr, dev):
    mpr.filter()  # cdc=16, cdt=1, ffi=0, sfi=0, esi=4
    assert dev.words[AFE1] == 0x2410


def test_charge_even_channel_preserves_high_nibble(mpr, dev):
    dev.bytes[CDT + 2] = 0xf0
    mpr.charge(4, cdc=7, cdt=3)
    assert dev.bytes[CDC + 4] == 7
    assert dev.bytes[CDT + 2] == 0xf3


def test_charge_odd_channel_preserves_low_nibble(mpr, dev):
    dev.bytes[CDT + 2] = 0x0f
    mpr.charge(5, cdc=0, cdt=3)
    assert dev.bytes[CDT + 2] == 0x3f


def test_auto_config_default_register_values(mpr, dev):
    mpr.auto_config()
    assert dev.words[ACNF_C0] == 0x076f
    assert dev.bytes[ACNF_USL] == 200
    assert dev.bytes[ACNF_LSL] == 130
    assert dev.bytes[ACNF_TL] == 180


@pytest.mark.parametrize('channel', [3, 12])
def test_gpio_setup_rejects_channel_outside_range(mpr, channel):
    with pytest.raises(AssertionError):
        mpr.gpio_setup(channel, output=1)


def test_gpio_setup_configures_mode_direction_and_enable_bits(mpr, dev):
    # Channel 11 maps to bit 7 of the GPIO registers.
    mpr.gpio_setup(11, output=0, mode=3, enable=1)
    assert dev.bytes[GPIO_CTL0] == 0x80
    assert dev.bytes[GPIO_CTL1] == 0x80
    assert dev.bytes[GPIO_DIR] == 0x00
    assert dev.bytes[GPIO_EN] == 0x80


def test_gpio_set_uses_set_and_clear_registers(mpr, dev):
    mpr.gpio_set(6, True)
    mpr.gpio_set(6, False)
    assert ('write_byte', GPIO_SET, 1 << 2) in dev.log
    assert ('write_byte', GPIO_CLR, 1 << 2) in dev.log


def test_update_touch_state_reports_press_and_release(mpr, dev):
    events = []

    def recorder(state, channel=None):
        events.append((channel, state))

    mpr.on_touch(0, recorder)
    mpr.on_touch(1, recorder)
    mpr.on_touch(2, recorder)

    dev.words[ETS] = 0b101
    mpr.update_touch_state()
    assert events == [(0, True), (2, True)]
    assert mpr.is_touched(0)
    assert not mpr.is_touched(1)
    assert mpr.is_touched(2)

    # Unchanged state must not invoke any handler.
    mpr.update_touch_state()
    assert events == [(0, True), (2, True)]

    # Releasing channel 2 must report a release only for that channel.
    dev.words[ETS] = 0b001
    mpr.update_touch_state()
    assert events == [(0, True), (2, True), (2, False)]
    assert not mpr.is_touched(2)
