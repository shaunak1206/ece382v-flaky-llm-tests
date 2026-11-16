import pytest

from piripherals import mpr121 as m
from piripherals.mpr121 import MPR121


class FakeDevice:
    """In-memory register file emulating the bus device used by MPR121."""

    def __init__(self):
        self.regs = [0] * 256
        self.writes = []

    def write_byte(self, reg, val):
        self.regs[reg] = val & 0xff
        self.writes.append((reg, val))

    def read_byte(self, reg):
        return self.regs[reg]

    def write_word(self, reg, val):
        self.regs[reg] = val & 0xff
        self.regs[reg + 1] = (val >> 8) & 0xff

    def read_word(self, reg):
        return self.regs[reg] | (self.regs[reg + 1] << 8)

    def write_block(self, reg, data):
        for i, d in enumerate(data):
            self.regs[reg + i] = d

    def read_block(self, reg, n):
        return list(self.regs[reg:reg + n])


class FakeBus:
    def __init__(self):
        self.dev = FakeDevice()
        self.addrs = []

    def device(self, addr):
        self.addrs.append(addr)
        return self.dev


@pytest.fixture
def bus():
    return FakeBus()


@pytest.fixture
def mpr(bus):
    return MPR121(bus=bus, handlers=0, setup=0, reset=0)


def test_init_uses_address_on_bus(bus):
    obj = MPR121(bus=bus, addr=0x5b, handlers=0, setup=0, reset=0)
    assert bus.addrs == [0x5b]
    assert obj.addr == 0x5b
    assert bus.dev.writes == []


def test_init_reset_without_setup_writes_soft_reset(bus):
    MPR121(bus=bus, handlers=0, setup=0, reset=1)
    assert bus.dev.writes == [(m.SRESET, 0x63)]


def test_configure_writes_ecr(mpr, bus):
    mpr.configure(cl=3, prox=0, touch=12)
    assert bus.dev.regs[m.ECR] == (3 << 6) | 12
    mpr.configure(cl=1, prox=2, touch=5)
    assert bus.dev.regs[m.ECR] == (1 << 6) | (2 << 4) | 5


def test_touched_masks_status_bits(mpr, bus):
    bus.dev.write_word(m.ETS, 0x0a05)
    assert mpr.touched() == 0x0a05
    bus.dev.write_word(m.ETS, 0x6fff)
    assert mpr.touched(raise_on_failure=0) == 0x0fff


def test_touched_raises_on_overcurrent(mpr, bus):
    bus.dev.write_word(m.ETS, 1 << 15)
    with pytest.raises(Exception, match='overcurrent'):
        mpr.touched()
    # no exception when failure checking is disabled
    assert mpr.touched(raise_on_failure=0) == 0


def test_touched_raises_on_out_of_range(mpr, bus):
    bus.dev.write_word(m.OOR, 0x0001)
    with pytest.raises(Exception, match='out of range'):
        mpr.touched()


def test_out_of_range_failure_bits(mpr, bus):
    bus.dev.write_word(m.OOR, 1 << 15)
    with pytest.raises(Exception, match='auto config failed'):
        mpr.out_of_range()
    bus.dev.write_word(m.OOR, 1 << 14)
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr.out_of_range()
    assert mpr.out_of_range(raise_on_failure=0) == 0
    bus.dev.write_word(m.OOR, 0x0005)
    assert mpr.out_of_range() == 5


def test_update_touch_state_calls_handlers_and_is_touched(mpr, bus):
    calls = []
    mpr.on_touch(0, lambda state, ch: calls.append((ch, state)))
    mpr.on_touch(2, lambda state, ch: calls.append((ch, state)))

    bus.dev.write_word(m.ETS, 0b101)
    mpr.update_touch_state()
    assert calls == [(0, True), (2, True)]
    assert mpr.is_touched(0)
    assert not mpr.is_touched(1)
    assert mpr.is_touched(2)

    # release channel 0 only
    calls.clear()
    bus.dev.write_word(m.ETS, 0b100)
    mpr.update_touch_state()
    assert calls == [(0, False)]
    assert not mpr.is_touched(0)

    # no change -> no calls
    calls.clear()
    mpr.update_touch_state()
    assert calls == []


def test_on_touch_none_removes_handler(mpr, bus):
    calls = []
    mpr.on_touch(1, lambda state, ch: calls.append((ch, state)))
    mpr.on_touch(1, None)
    bus.dev.write_word(m.ETS, 0b10)
    mpr.update_touch_state()
    assert calls == []
    assert mpr.is_touched(1)


def test_electrode_data_combines_bytes(mpr, bus):
    for i in range(m.NCH):
        bus.dev.regs[m.EFD + 2 * i] = i
        bus.dev.regs[m.EFD + 2 * i + 1] = i % 4
    data = mpr.electrode_data()
    assert len(data) == m.NCH
    for i in range(m.NCH):
        assert data[i] == ((i % 4) << 8) + i


def test_baseline_read_shifts_left_by_two(mpr, bus):
    for i in range(m.NCH):
        bus.dev.regs[m.EBL + i] = i + 1
    assert mpr.baseline() == [(i + 1) << 2 for i in range(m.NCH)]


def test_baseline_write_rising_falling_touched_and_prox(mpr, bus):
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert bus.dev.regs[m.MHD:m.MHD + 4] == [5, 1, 3, 20]

    mpr.baseline(rft=1, mhd=6, nhd=2, ncl=4, fdl=21)
    assert bus.dev.regs[m.MHD + 4:m.MHD + 8] == [6, 2, 4, 21]

    mpr.baseline(rft=2, mhd=99, nhd=7, ncl=8, fdl=9)
    assert bus.dev.regs[m.MHD + 8:m.MHD + 11] == [7, 8, 9]

    mpr.baseline(rft=0, mhd=1, nhd=1, ncl=2, fdl=3, prox=1)
    assert bus.dev.regs[m.MHDX:m.MHDX + 4] == [1, 1, 2, 3]


def test_threshold_all_channels_default_release(mpr, bus):
    mpr.threshold(50)
    for ch in range(m.NCH):
        assert bus.dev.regs[m.TTH + 2 * ch] == 50
        assert bus.dev.regs[m.RTH + 2 * ch] == 30


def test_threshold_single_channel_explicit_release(mpr, bus):
    mpr.threshold(40, release=10, channel=3)
    assert bus.dev.regs[m.TTH + 6] == 40
    assert bus.dev.regs[m.RTH + 6] == 10
    assert bus.dev.regs[m.TTH + 4] == 0
    assert bus.dev.regs[m.TTH + 8] == 0


def test_debounce_packs_touch_and_release(mpr, bus):
    mpr.debounce(3)
    assert bus.dev.regs[m.DEB] == 0x33
    mpr.debounce(2, 5)
    assert bus.dev.regs[m.DEB] == (5 << 4) | 2


def test_filter_writes_word_little_endian(mpr, bus):
    mpr.filter()
    assert bus.dev.regs[m.AFE1] == 16
    assert bus.dev.regs[m.AFE2] == (1 << 5) | 4
    mpr.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    assert bus.dev.regs[m.AFE1] == (1 << 6) | 30
    assert bus.dev.regs[m.AFE2] == (1 << 5) | (1 << 3)


def test_charge_sets_current_and_preserves_other_nibble(mpr, bus):
    bus.dev.regs[m.CDT + 1] = 0x0f
    mpr.charge(3, cdc=10, cdt=5)
    assert bus.dev.regs[m.CDC + 3] == 10
    assert bus.dev.regs[m.CDT + 1] == 0x5f
    mpr.charge(2, cdc=1, cdt=2)
    assert bus.dev.regs[m.CDT + 1] == 0x5f & 0xf0 | 0x02


def test_auto_config_defaults(mpr, bus):
    mpr.auto_config()
    assert bus.dev.regs[m.ACNF_C0] == 111
    assert bus.dev.regs[m.ACNF_C0 + 1] == 7
    assert bus.dev.regs[m.ACNF_USL] == 200
    assert bus.dev.regs[m.ACNF_LSL] == 130
    assert bus.dev.regs[m.ACNF_TL] == 180


def test_gpio_set_and_status(mpr, bus):
    mpr.gpio_set(6, 1)
    assert bus.dev.writes[-1] == (m.GPIO_SET, 1 << 2)
    mpr.gpio_set(11, 0)
    assert bus.dev.writes[-1] == (m.GPIO_CLR, 1 << 7)
    bus.dev.regs[m.GPIO_DAT] = 0xa5
    assert mpr.gpio_status() == 0xa5
    with pytest.raises(AssertionError):
        mpr.gpio_set(3, 1)
    with pytest.raises(AssertionError):
        mpr.gpio_set(12, 1)


def test_gpio_setup_sets_bits(mpr, bus):
    mpr.gpio_setup(5, output=1, mode=3, enable=1)
    bit = 1 << 1
    assert bus.dev.regs[m.GPIO_CTL0] == bit
    assert bus.dev.regs[m.GPIO_CTL1] == bit
    assert bus.dev.regs[m.GPIO_DIR] == bit
    assert bus.dev.regs[m.GPIO_EN] == bit
    with pytest.raises(AssertionError):
        mpr.gpio_setup(3, output=1)


def test_setup_applies_defaults(bus):
    MPR121(bus=bus, handlers=0, setup=1, reset=1)
    r = bus.dev.regs
    assert bus.dev.writes[0] == (m.SRESET, 0x63)
    assert r[m.ECR] == (3 << 6) | 12
    assert r[m.TTH] == 50 and r[m.RTH] == 30
    assert r[m.TTH + 24] == 12 and r[m.RTH + 24] == 7
    assert r[m.DEB] == 0x22
    assert r[m.MHD:m.MHD + 4] == [5, 1, 3, 20]
    assert r[m.MHDX:m.MHDX + 4] == [1, 1, 3, 20]
    assert r[m.ACNF_USL] == 200


def test_setup_without_auto_config_and_with_prox(bus):
    MPR121(bus=bus, handlers=0, setup=1, reset=0, auto_config=0,
           channels=6, prox=3)
    r = bus.dev.regs
    assert r[m.ECR] == (3 << 6) | (3 << 4) | 6
    assert r[m.ACNF_USL] == 0
    assert (m.SRESET, 0x63) not in bus.dev.writes
