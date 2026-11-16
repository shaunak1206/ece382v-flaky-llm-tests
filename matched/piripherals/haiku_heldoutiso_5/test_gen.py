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
)


class FakeDevice:
    """Records every write and serves canned reads, keyed by register."""

    def __init__(self):
        self.words = {}
        self.bytes = {}
        self.blocks = {}
        self.writes = []

    def read_word(self, reg):
        return self.words.get(reg, 0)

    def read_byte(self, reg):
        return self.bytes.get(reg, 0)

    def read_block(self, reg, length):
        return list(self.blocks.get(reg, [0] * length))

    def write_byte(self, reg, value):
        self.writes.append(('byte', reg, value))

    def write_word(self, reg, value):
        self.writes.append(('word', reg, value))

    def write_block(self, reg, values):
        self.writes.append(('block', reg, list(values)))

    def writes_to(self, reg):
        return [w for w in self.writes if w[1] == reg]


class FakeBus:
    """Mimics piripherals.bus.Bus: exposes device(addr) returning a FakeDevice."""

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
    bus = FakeBus(dev)
    return MPR121(bus=bus, handlers=0, setup=0, reset=0)


def test_init_requests_device_at_given_address(dev):
    bus = FakeBus(dev)
    MPR121(bus=bus, addr=0x5b, handlers=0, setup=0, reset=0)
    assert bus.addresses == [0x5b]


def test_init_without_setup_or_reset_writes_nothing(dev):
    bus = FakeBus(dev)
    MPR121(bus=bus, handlers=0, setup=0, reset=0)
    assert dev.writes == []


def test_init_reset_only_soft_resets(dev):
    bus = FakeBus(dev)
    MPR121(bus=bus, handlers=0, setup=0, reset=1)
    assert dev.writes == [('byte', SRESET, 0x63)]


def test_reset_writes_magic_value_to_sreset(mpr, dev):
    mpr.reset()
    assert dev.writes == [('byte', SRESET, 0x63)]


def test_configure_packs_cl_prox_and_touch_into_ecr(mpr, dev):
    mpr.configure(cl=3, prox=0, touch=12)
    assert dev.writes_to(ECR) == [('byte', ECR, 0xcc)]


def test_configure_stop_mode_writes_only_calibration_lock(mpr, dev):
    mpr.configure(prox=0, touch=0)
    assert dev.writes_to(ECR) == [('byte', ECR, 0xc0)]


def test_is_touched_reflects_last_update(mpr, dev):
    dev.words[ETS] = 0b101
    mpr.update_touch_state()
    assert mpr.is_touched(0) is True
    assert mpr.is_touched(1) is False
    assert mpr.is_touched(2) is True


def test_touched_ignores_bits_above_12_when_not_raising(mpr, dev):
    dev.words[ETS] = 0x8000 | 0x0005
    assert mpr.touched(raise_on_failure=0) == 0x0005


def test_touched_raises_on_overcurrent(mpr, dev):
    dev.words[ETS] = 0x8000
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


def test_out_of_range_without_raise_returns_masked_status(mpr, dev):
    dev.words[OOR] = 0x8000 | 0x0100
    assert mpr.out_of_range(raise_on_failure=0) == 0x0100


def test_handler_receives_state_and_channel_on_change(mpr, dev):
    calls = []
    mpr.on_touch(3, lambda state, ch: calls.append((state, ch)))

    dev.words[ETS] = 1 << 3
    mpr.update_touch_state()
    dev.words[ETS] = 0
    mpr.update_touch_state()

    assert calls == [(True, 3), (False, 3)]


def test_handler_not_called_when_state_unchanged(mpr, dev):
    calls = []
    mpr.on_touch(0, lambda state, ch: calls.append((state, ch)))

    dev.words[ETS] = 1
    mpr.update_touch_state()
    mpr.update_touch_state()

    assert calls == [(True, 0)]


def test_handler_only_fires_for_changed_channel(mpr, dev):
    calls_a = []
    calls_b = []
    mpr.on_touch(0, lambda s, c: calls_a.append(s))
    mpr.on_touch(1, lambda s, c: calls_b.append(s))

    dev.words[ETS] = 0b01
    mpr.update_touch_state()
    dev.words[ETS] = 0b11
    mpr.update_touch_state()

    assert calls_a == [True]
    assert calls_b == [True]


def test_electrode_data_combines_little_endian_pairs(mpr, dev):
    raw = [0] * (2 * 13)
    raw[0] = 0x34
    raw[1] = 0x02
    dev.blocks[EFD] = raw
    data = mpr.electrode_data()
    assert len(data) == 13
    assert data[0] == 0x234


def test_baseline_read_shifts_values_left_by_two(mpr, dev):
    dev.blocks[EBL] = [1] * 13
    assert mpr.baseline() == [4] * 13


def test_baseline_rising_writes_four_parameters(mpr, dev):
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert dev.writes == [('block', MHD, [5, 1, 3, 20])]


def test_baseline_touched_scenario_writes_three_parameters(mpr, dev):
    mpr.baseline(rft=2, nhd=1, ncl=3, fdl=20)
    assert dev.writes == [('block', MHD + 8, [1, 3, 20])]


def test_baseline_proximity_uses_proximity_registers(mpr, dev):
    mpr.baseline(rft=1, mhd=2, nhd=1, ncl=3, fdl=20, prox=1)
    assert dev.writes == [('block', MHDX + 4, [2, 1, 3, 20])]


def test_threshold_all_channels_defaults_release_to_60_percent(mpr, dev):
    mpr.threshold(touch=50)
    assert ('byte', TTH, 50) in dev.writes
    assert ('byte', RTH, 30) in dev.writes
    # proximity channel 12 is the last one
    assert ('byte', TTH + 24, 50) in dev.writes
    assert ('byte', RTH + 24, 30) in dev.writes
    assert len(dev.writes) == 13 * 2


def test_threshold_single_channel_with_explicit_release(mpr, dev):
    mpr.threshold(touch=10, release=5, channel=3)
    assert dev.writes == [
        ('byte', TTH + 6, 10),
        ('byte', RTH + 6, 5),
    ]


def test_debounce_packs_release_in_high_nibble(mpr, dev):
    mpr.debounce(touch=1, release=3)
    assert dev.writes == [('byte', DEB, 0x31)]


def test_debounce_release_defaults_to_touch(mpr, dev):
    mpr.debounce(touch=2)
    assert dev.writes == [('byte', DEB, 0x22)]


def test_filter_packs_afe1_word(mpr, dev):
    mpr.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    assert dev.writes == [('word', AFE1, 0x285e)]


def test_charge_updates_only_the_target_nibble(mpr, dev):
    # channel 1 lives in the high nibble of CDT + 0
    dev.bytes[CDT] = 0xab
    mpr.charge(channel=1, cdc=10, cdt=3)
    assert ('byte', CDC + 1, 10) in dev.writes
    assert ('byte', CDT, 0x3b) in dev.writes


def test_auto_config_writes_control_word_and_limits(mpr, dev):
    mpr.auto_config()
    assert dev.writes == [
        ('word', ACNF_C0, 0x076f),
        ('byte', ACNF_USL, 200),
        ('byte', ACNF_LSL, 130),
        ('byte', ACNF_TL, 180),
    ]


def test_gpio_set_high_and_low_use_set_and_clear_registers(mpr, dev):
    mpr.gpio_set(5, True)
    mpr.gpio_set(5, False)
    assert dev.writes == [
        ('byte', GPIO_SET, 2),
        ('byte', GPIO_CLR, 2),
    ]


def test_gpio_set_rejects_channel_outside_gpio_range(mpr):
    with pytest.raises(AssertionError):
        mpr.gpio_set(3, True)


def test_gpio_setup_configures_output_and_enable_bits(mpr, dev):
    mpr.gpio_setup(channel=5, output=1, mode=0, enable=1)
    assert ('byte', GPIO_CTL0, 0) in dev.writes
    assert ('byte', GPIO_CTL1, 0) in dev.writes
    assert ('byte', GPIO_DIR, 2) in dev.writes
    assert ('byte', GPIO_EN, 2) in dev.writes


def test_gpio_status_reads_data_register(mpr, dev):
    dev.bytes[GPIO_DAT] = 0x5a
    assert mpr.gpio_status() == 0x5a


def test_setup_ends_by_enabling_twelve_channels(mpr, dev):
    mpr.setup(reset=1, channels=12, prox=0, threshold=50)
    assert dev.writes[0] == ('byte', SRESET, 0x63)
    assert dev.writes[-1] == ('byte', ECR, 0xcc)
