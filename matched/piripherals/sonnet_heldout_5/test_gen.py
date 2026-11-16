from collections import defaultdict

import pytest

from piripherals import mpr121 as mod
from piripherals.mpr121 import MPR121


class FakeDevice:
    """In-memory register file emulating the bus device interface."""

    def __init__(self):
        self.regs = defaultdict(int)
        self.log = []

    def read_byte(self, reg):
        return self.regs[reg]

    def write_byte(self, reg, value):
        self.log.append(('b', reg, value))
        self.regs[reg] = value

    def read_word(self, reg):
        return self.regs[reg] | (self.regs[reg + 1] << 8)

    def write_word(self, reg, value):
        self.log.append(('w', reg, value))
        self.regs[reg] = value & 0xff
        self.regs[reg + 1] = (value >> 8) & 0xff

    def read_block(self, reg, n):
        return [self.regs[reg + i] for i in range(n)]

    def write_block(self, reg, values):
        self.log.append(('blk', reg, list(values)))
        for i, v in enumerate(values):
            self.regs[reg + i] = v


class FakeBus:
    def __init__(self):
        self.dev = FakeDevice()
        self.addr = None

    def device(self, addr):
        self.addr = addr
        return self.dev


@pytest.fixture
def bus():
    return FakeBus()


@pytest.fixture
def mpr(bus):
    return MPR121(bus=bus, handlers=0, setup=0, reset=0)


def test_init_without_setup_or_reset_writes_nothing(bus, mpr):
    assert bus.addr == 0x5a
    assert bus.dev.log == []
    assert mpr.addr == 0x5a


def test_init_custom_address(bus):
    MPR121(bus=bus, addr=0x5b, handlers=0, setup=0, reset=0)
    assert bus.addr == 0x5b


def test_init_reset_only(bus):
    MPR121(bus=bus, handlers=0, setup=0, reset=1)
    assert bus.dev.log == [('b', mod.SRESET, 0x63)]


def test_reset_writes_soft_reset_register(bus, mpr):
    mpr.reset()
    assert bus.dev.log == [('b', 0x80, 0x63)]


def test_configure_default_and_custom(bus, mpr):
    mpr.configure()
    assert bus.dev.regs[mod.ECR] == (3 << 6) | 12
    mpr.configure(cl=1, prox=2, touch=5)
    assert bus.dev.regs[mod.ECR] == (1 << 6) | (2 << 4) | 5
    mpr.configure(cl=0, prox=0, touch=0)
    assert bus.dev.regs[mod.ECR] == 0


def test_threshold_all_channels_default_release(bus, mpr):
    mpr.threshold(50)
    for i in range(mod.NCH):
        assert bus.dev.regs[mod.TTH + 2 * i] == 50
        assert bus.dev.regs[mod.RTH + 2 * i] == 30


def test_threshold_single_channel_explicit_release(bus, mpr):
    mpr.threshold(40, release=10, channel=3)
    assert bus.dev.regs[mod.TTH + 6] == 40
    assert bus.dev.regs[mod.RTH + 6] == 10
    assert bus.dev.regs[mod.TTH] == 0
    assert len(bus.dev.log) == 2


def test_debounce(bus, mpr):
    mpr.debounce(3)
    assert bus.dev.regs[mod.DEB] == 0x33
    mpr.debounce(2, 5)
    assert bus.dev.regs[mod.DEB] == 0x52


def test_filter_defaults_pack_word(bus, mpr):
    mpr.filter()
    assert bus.dev.regs[mod.AFE1] == 16
    assert bus.dev.regs[mod.AFE2] == (1 << 5) | 4


def test_touched_reads_and_masks_status(bus, mpr):
    bus.dev.regs[mod.ETS] = 0x05
    bus.dev.regs[mod.ETS + 1] = 0x10  # bit 12 (proximity)
    assert mpr.touched() == 0x1005
    bus.dev.regs[mod.ETS + 1] = 0xe0 | 0x10
    assert mpr.touched(raise_on_failure=0) == 0x1005


def test_touched_raises_on_overcurrent(bus, mpr):
    bus.dev.regs[mod.ETS + 1] = 0x80
    with pytest.raises(Exception, match='overcurrent'):
        mpr.touched()
    # without checking, it returns the masked value
    assert mpr.touched(raise_on_failure=0) == 0


def test_touched_raises_on_out_of_range(bus, mpr):
    bus.dev.regs[mod.OOR + 1] = 0x80
    with pytest.raises(Exception, match='auto config failed'):
        mpr.touched()
    bus.dev.regs[mod.OOR + 1] = 0x40
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr.out_of_range()
    assert mpr.out_of_range(raise_on_failure=0) == 0


def test_update_touch_state_invokes_handlers_on_change(bus, mpr):
    events = []
    mpr.on_touch(0, lambda s, c: events.append((c, s)))
    mpr.on_touch(2, lambda s, c: events.append((c, s)))

    bus.dev.regs[mod.ETS] = 0b101
    mpr.update_touch_state()
    assert sorted(events) == [(0, True), (2, True)]
    assert mpr.is_touched(0)
    assert not mpr.is_touched(1)
    assert mpr.is_touched(2)

    # no change -> no new events
    events.clear()
    mpr.update_touch_state()
    assert events == []

    # release channel 0
    bus.dev.regs[mod.ETS] = 0b100
    mpr.update_touch_state()
    assert events == [(0, False)]
    assert not mpr.is_touched(0)


def test_on_touch_none_removes_handler(bus, mpr):
    events = []
    mpr.on_touch(1, lambda s, c: events.append((c, s)))
    mpr.on_touch(1, None)
    bus.dev.regs[mod.ETS] = 0b10
    mpr.update_touch_state()
    assert events == []
    assert mpr.is_touched(1)


def test_electrode_data_combines_bytes(bus, mpr):
    for i in range(mod.NCH):
        bus.dev.regs[mod.EFD + 2 * i] = i
        bus.dev.regs[mod.EFD + 2 * i + 1] = 0x01 if i % 2 else 0x00
    data = mpr.electrode_data()
    assert len(data) == mod.NCH
    assert data[0] == 0
    assert data[1] == 0x101
    assert data[4] == 4
    assert data[11] == 0x10b


def test_baseline_read_shifts_values(bus, mpr):
    for i in range(mod.NCH):
        bus.dev.regs[mod.EBL + i] = i + 1
    assert mpr.baseline() == [(i + 1) << 2 for i in range(mod.NCH)]


def test_baseline_write_variants(bus, mpr):
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    mpr.baseline(rft=1, mhd=6, nhd=2, ncl=4, fdl=21, prox=1)
    mpr.baseline(rft=2, mhd=9, nhd=7, ncl=8, fdl=9)
    assert bus.dev.log == [
        ('blk', mod.MHD, [5, 1, 3, 20]),
        ('blk', mod.MHDX + 4, [6, 2, 4, 21]),
        ('blk', mod.MHD + 8, [7, 8, 9]),
    ]


def test_charge_sets_current_and_preserves_other_nibble(bus, mpr):
    bus.dev.regs[mod.CDT + 1] = 0x0f
    mpr.charge(3, cdc=20, cdt=5)
    assert bus.dev.regs[mod.CDC + 3] == 20
    assert bus.dev.regs[mod.CDT + 1] == 0x5f

    bus.dev.regs[mod.CDT] = 0xa0
    mpr.charge(0, cdc=1, cdt=2)
    assert bus.dev.regs[mod.CDT] == 0xa2


def test_auto_config_defaults(bus, mpr):
    mpr.auto_config()
    assert bus.dev.regs[mod.ACNF_C0] == 0x6f
    assert bus.dev.regs[mod.ACNF_C1] == 0x07
    assert bus.dev.regs[mod.ACNF_USL] == 200
    assert bus.dev.regs[mod.ACNF_LSL] == 130
    assert bus.dev.regs[mod.ACNF_TL] == 180


def test_gpio_set_and_clear(bus, mpr):
    mpr.gpio_set(4, 1)
    mpr.gpio_set(7, 0)
    assert bus.dev.log == [('b', mod.GPIO_SET, 1), ('b', mod.GPIO_CLR, 8)]
    with pytest.raises(AssertionError):
        mpr.gpio_set(3, 1)
    with pytest.raises(AssertionError):
        mpr.gpio_set(12, 1)


def test_gpio_status_reads_data_register(bus, mpr):
    bus.dev.regs[mod.GPIO_DAT] = 0xa5
    assert mpr.gpio_status() == 0xa5


def test_gpio_setup_output_open_drain(bus, mpr):
    mpr.gpio_setup(5, output=1, mode=2, enable=1)
    assert bus.dev.regs[mod.GPIO_CTL0] == 0b10
    assert bus.dev.regs[mod.GPIO_CTL1] == 0
    assert bus.dev.regs[mod.GPIO_DIR] == 0b10
    assert bus.dev.regs[mod.GPIO_EN] == 0b10
    with pytest.raises(AssertionError):
        mpr.gpio_setup(2, output=1)


def test_setup_defaults_configures_device(bus):
    MPR121(bus=bus, handlers=0, setup=1, reset=1)
    regs = bus.dev.regs
    assert bus.dev.log[0] == ('b', mod.SRESET, 0x63)
    # stop mode first, run mode last
    assert bus.dev.log[1] == ('b', mod.ECR, 0)
    assert bus.dev.log[-1] == ('b', mod.ECR, (3 << 6) | 12)
    for i in range(12):
        assert regs[mod.TTH + 2 * i] == 50
        assert regs[mod.RTH + 2 * i] == 30
    assert regs[mod.TTH + 24] == 50 >> 2
    assert regs[mod.DEB] == 0x22
    assert regs[mod.ACNF_USL] == 200
    assert regs[mod.MHD] == 5
    assert regs[mod.MHDX] == 1
