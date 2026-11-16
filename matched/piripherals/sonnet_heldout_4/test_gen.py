import pytest

from piripherals import mpr121
from piripherals.mpr121 import MPR121, NCH


class FakeDevice:
    """In-memory register file emulating a bus device (little endian words)."""

    def __init__(self):
        self.regs = [0] * 256
        self.writes = []

    def write_byte(self, reg, value):
        self.regs[reg] = value & 0xff
        self.writes.append((reg, value))

    def read_byte(self, reg):
        return self.regs[reg]

    def write_word(self, reg, value):
        self.regs[reg] = value & 0xff
        self.regs[reg + 1] = (value >> 8) & 0xff
        self.writes.append((reg, value))

    def read_word(self, reg):
        return self.regs[reg] | (self.regs[reg + 1] << 8)

    def write_block(self, reg, values):
        for i, v in enumerate(values):
            self.regs[reg + i] = v
        self.writes.append((reg, list(values)))

    def read_block(self, reg, n):
        return list(self.regs[reg:reg + n])


class FakeBus:
    def __init__(self):
        self.dev = FakeDevice()
        self.addr = None

    def device(self, addr):
        self.addr = addr
        return self.dev


def make(**kwargs):
    kwargs.setdefault('handlers', 0)
    kwargs.setdefault('setup', 0)
    kwargs.setdefault('reset', 0)
    bus = FakeBus()
    m = MPR121(bus=bus, **kwargs)
    return m, bus.dev


def test_init_uses_address_and_does_nothing_else():
    bus = FakeBus()
    m = MPR121(bus=bus, addr=0x5b, handlers=0, setup=0, reset=0)
    assert bus.addr == 0x5b
    assert m.addr == 0x5b
    assert bus.dev.writes == []


def test_init_reset_only_writes_soft_reset():
    m, dev = make(reset=1)
    assert dev.writes == [(mpr121.SRESET, 0x63)]


def test_configure_writes_ecr():
    m, dev = make()
    m.configure(cl=2, prox=1, touch=5)
    assert dev.regs[mpr121.ECR] == (2 << 6) | (1 << 4) | 5
    m.configure()
    assert dev.regs[mpr121.ECR] == (3 << 6) | 12


def test_touched_masks_status_bits():
    m, dev = make()
    dev.write_word(mpr121.ETS, 0x0a05)
    assert m.touched() == 0x0a05
    dev.write_word(mpr121.ETS, 0xffff & ~(1 << 15))
    assert m.touched() == 0x1fff


def test_touched_overcurrent_raises():
    m, dev = make()
    dev.write_word(mpr121.ETS, 1 << 15)
    with pytest.raises(Exception, match='overcurrent'):
        m.touched()
    # failure bits ignored when requested
    assert m.touched(raise_on_failure=0) == 0


def test_touched_raises_on_out_of_range():
    m, dev = make()
    dev.write_word(mpr121.OOR, 1 << 15)
    with pytest.raises(Exception):
        m.touched()


def test_out_of_range_failure_flags():
    m, dev = make()
    dev.write_word(mpr121.OOR, 1 << 15)
    with pytest.raises(Exception, match='auto config failed'):
        m.out_of_range()
    dev.write_word(mpr121.OOR, 1 << 14)
    with pytest.raises(Exception, match='auto reconfig failed'):
        m.out_of_range()
    dev.write_word(mpr121.OOR, (1 << 14) | 0x0003)
    assert m.out_of_range(raise_on_failure=0) == 0x0003
    dev.write_word(mpr121.OOR, 0x0005)
    assert m.out_of_range() == 5


def test_update_touch_state_calls_handlers_on_change():
    m, dev = make()
    calls = []
    m.on_touch(3, lambda state, ch: calls.append((state, ch)))
    dev.write_word(mpr121.ETS, 1 << 3)
    m.update_touch_state()
    assert calls == [(True, 3)]
    assert m.is_touched(3)
    assert not m.is_touched(2)

    # no change -> no call
    m.update_touch_state()
    assert calls == [(True, 3)]

    dev.write_word(mpr121.ETS, 0)
    m.update_touch_state()
    assert calls == [(True, 3), (False, 3)]
    assert not m.is_touched(3)


def test_handlers_only_called_for_changed_channels():
    m, dev = make()
    calls = []
    for i in range(NCH):
        m.on_touch(i, lambda s, c: calls.append((c, s)))
    dev.write_word(mpr121.ETS, 0b101)
    m.update_touch_state()
    assert sorted(calls) == [(0, True), (2, True)]
    del calls[:]
    dev.write_word(mpr121.ETS, 0b100)
    m.update_touch_state()
    assert calls == [(0, False)]


def test_on_touch_none_removes_handler():
    m, dev = make()
    calls = []
    m.on_touch(1, lambda s, c: calls.append((s, c)))
    m.on_touch(1, None)
    dev.write_word(mpr121.ETS, 1 << 1)
    m.update_touch_state()
    assert calls == []
    assert m.is_touched(1)


def test_threshold_all_channels_default_release():
    m, dev = make()
    m.threshold(50)
    for i in range(NCH):
        assert dev.regs[mpr121.TTH + 2 * i] == 50
        assert dev.regs[mpr121.RTH + 2 * i] == 30


def test_threshold_single_channel_explicit_release():
    m, dev = make()
    m.threshold(40, release=10, channel=5)
    assert dev.regs[mpr121.TTH + 10] == 40
    assert dev.regs[mpr121.RTH + 10] == 10
    assert dev.regs[mpr121.TTH + 8] == 0


def test_debounce():
    m, dev = make()
    m.debounce(3)
    assert dev.regs[mpr121.DEB] == 0x33
    m.debounce(2, 5)
    assert dev.regs[mpr121.DEB] == 0x52


def test_filter_writes_afe_registers():
    m, dev = make()
    m.filter()
    assert dev.regs[mpr121.AFE1] == 16
    assert dev.regs[mpr121.AFE2] == (1 << 5) | 4
    m.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    assert dev.regs[mpr121.AFE1] == (1 << 6) | 30
    assert dev.regs[mpr121.AFE2] == (1 << 5) | (1 << 3)


def test_charge_sets_nibbles_independently():
    m, dev = make()
    m.charge(2, cdc=20, cdt=3)
    m.charge(3, cdc=21, cdt=5)
    assert dev.regs[mpr121.CDC + 2] == 20
    assert dev.regs[mpr121.CDC + 3] == 21
    assert dev.regs[mpr121.CDT + 1] == (5 << 4) | 3
    m.charge(2, cdc=20, cdt=1)
    assert dev.regs[mpr121.CDT + 1] == (5 << 4) | 1


def test_electrode_data_combines_bytes():
    m, dev = make()
    for i in range(NCH):
        dev.regs[mpr121.EFD + 2 * i] = i
        dev.regs[mpr121.EFD + 2 * i + 1] = 0x01 if i % 2 else 0x00
    data = m.electrode_data()
    assert len(data) == NCH
    for i in range(NCH):
        assert data[i] == i + (0x100 if i % 2 else 0)


def test_baseline_read_and_write():
    m, dev = make()
    for i in range(NCH):
        dev.regs[mpr121.EBL + i] = i + 1
    assert m.baseline() == [(i + 1) << 2 for i in range(NCH)]

    m.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert dev.regs[mpr121.MHD:mpr121.MHD + 4] == [5, 1, 3, 20]
    m.baseline(rft=1, mhd=6, nhd=2, ncl=4, fdl=21, prox=1)
    assert dev.regs[mpr121.MHDX + 4:mpr121.MHDX + 8] == [6, 2, 4, 21]
    m.baseline(rft=2, nhd=7, ncl=8, fdl=9)
    assert dev.regs[mpr121.MHD + 8:mpr121.MHD + 11] == [7, 8, 9]


def test_auto_config_registers():
    m, dev = make()
    m.auto_config()
    assert dev.regs[mpr121.ACNF_C0] == (1 << 6) | (2 << 4) | (3 << 2) | 0b11
    assert dev.regs[mpr121.ACNF_C0 + 1] == 0b111
    assert dev.regs[mpr121.ACNF_USL] == 200
    assert dev.regs[mpr121.ACNF_LSL] == 130
    assert dev.regs[mpr121.ACNF_TL] == 180


def test_gpio_set_and_clear():
    m, dev = make()
    m.gpio_set(4, 1)
    m.gpio_set(11, 0)
    assert dev.writes[-2] == (mpr121.GPIO_SET, 1)
    assert dev.writes[-1] == (mpr121.GPIO_CLR, 1 << 7)
    with pytest.raises(AssertionError):
        m.gpio_set(3, 1)
    with pytest.raises(AssertionError):
        m.gpio_set(12, 1)


def test_gpio_setup_output_and_status():
    m, dev = make()
    m.gpio_setup(5, output=1, mode=2)
    assert dev.regs[mpr121.GPIO_CTL0] == 1 << 1
    assert dev.regs[mpr121.GPIO_CTL1] == 0
    assert dev.regs[mpr121.GPIO_DIR] == 1 << 1
    assert dev.regs[mpr121.GPIO_EN] == 1 << 1
    dev.regs[mpr121.GPIO_DAT] = 0xa5
    assert m.gpio_status() == 0xa5
    with pytest.raises(AssertionError):
        m.gpio_setup(2, output=1)


def test_setup_defaults():
    m, dev = make(setup=1, reset=1)
    assert dev.writes[0] == (mpr121.SRESET, 0x63)
    assert dev.regs[mpr121.ECR] == (3 << 6) | 12
    for i in range(12):
        assert dev.regs[mpr121.TTH + 2 * i] == 50
        assert dev.regs[mpr121.RTH + 2 * i] == 30
    assert dev.regs[mpr121.TTH + 24] == 12
    assert dev.regs[mpr121.RTH + 24] == 7
    assert dev.regs[mpr121.DEB] == 0x22
    assert dev.regs[mpr121.ACNF_USL] == 200
    assert dev.regs[mpr121.MHD] == 5
    assert dev.regs[mpr121.MHDX] == 1


def test_setup_without_auto_config_skips_acnf():
    m, dev = make(setup=1, reset=0, auto_config=0, channels=6, prox=1)
    assert dev.regs[mpr121.ACNF_USL] == 0
    assert dev.regs[mpr121.ECR] == (3 << 6) | (1 << 4) | 6
    assert all(reg != mpr121.SRESET for reg, _ in dev.writes)
