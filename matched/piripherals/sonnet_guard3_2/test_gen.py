import pytest

from piripherals import mpr121 as m
from piripherals.mpr121 import MPR121


class FakeDevice:
    def __init__(self):
        self.mem = {}
        self.log = []

    def write_byte(self, reg, val):
        self.log.append(('wb', reg, val))
        self.mem[reg] = val

    def read_byte(self, reg):
        return self.mem.get(reg, 0)

    def write_word(self, reg, val):
        self.log.append(('ww', reg, val))
        self.mem[reg] = val

    def read_word(self, reg):
        return self.mem.get(reg, 0)

    def write_block(self, reg, vals):
        self.log.append(('wblk', reg, list(vals)))

    def read_block(self, reg, n):
        return [self.mem.get(reg + i, 0) for i in range(n)]


class FakeBus:
    def __init__(self):
        self.dev = FakeDevice()
        self.addrs = []

    def device(self, addr):
        self.addrs.append(addr)
        return self.dev


def make(**kwargs):
    bus = FakeBus()
    kw = dict(handlers=0, setup=0, reset=0)
    kw.update(kwargs)
    mpr = MPR121(bus=bus, **kw)
    return mpr, bus.dev, bus


def test_init_without_setup_does_no_bus_writes():
    mpr, dev, bus = make(addr=0x5b)
    assert bus.addrs == [0x5b]
    assert mpr.addr == 0x5b
    assert dev.log == []


def test_init_reset_only_writes_soft_reset():
    mpr, dev, _ = make(reset=1)
    assert dev.log == [('wb', m.SRESET, 0x63)]


def test_configure_encodes_register():
    mpr, dev, _ = make()
    mpr.configure()
    assert dev.log[-1] == ('wb', m.ECR, 0xC0 | 12)
    mpr.configure(cl=1, prox=2, touch=5)
    assert dev.log[-1] == ('wb', m.ECR, (1 << 6) | (2 << 4) | 5)


def test_touched_masks_status_bits():
    mpr, dev, _ = make()
    dev.mem[m.ETS] = 0x0abc
    assert mpr.touched() == 0x0abc
    dev.mem[m.ETS] = 0x1fff
    assert mpr.touched() == 0x1fff


def test_touched_raises_on_overcurrent():
    mpr, dev, _ = make()
    dev.mem[m.ETS] = 0x8001
    with pytest.raises(Exception, match='overcurrent'):
        mpr.touched()
    assert mpr.touched(raise_on_failure=0) == 0x0001


def test_touched_raises_on_out_of_range():
    mpr, dev, _ = make()
    dev.mem[m.ETS] = 0x0001
    dev.mem[m.OOR] = 0x8000
    with pytest.raises(Exception):
        mpr.touched()
    assert mpr.touched(raise_on_failure=0) == 1


def test_out_of_range_failure_bits():
    mpr, dev, _ = make()
    dev.mem[m.OOR] = 0x8000 | 0x3
    with pytest.raises(Exception, match='auto config failed'):
        mpr.out_of_range()
    dev.mem[m.OOR] = 0x4000 | 0x3
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr.out_of_range()
    assert mpr.out_of_range(raise_on_failure=0) == 0x3
    dev.mem[m.OOR] = 0x0005
    assert mpr.out_of_range() == 5


def test_electrode_data_combines_bytes():
    mpr, dev, _ = make()
    for i in range(m.NCH):
        dev.mem[m.EFD + 2 * i] = i + 1
        dev.mem[m.EFD + 2 * i + 1] = i % 4
    data = mpr.electrode_data()
    assert len(data) == m.NCH
    assert data == [((i % 4) << 8) + i + 1 for i in range(m.NCH)]


def test_baseline_read_shifts_values():
    mpr, dev, _ = make()
    for i in range(m.NCH):
        dev.mem[m.EBL + i] = i
    assert mpr.baseline() == [i << 2 for i in range(m.NCH)]


def test_baseline_write_registers():
    mpr, dev, _ = make()
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert dev.log[-1] == ('wblk', m.MHD, [5, 1, 3, 20])
    mpr.baseline(rft=1, mhd=2, nhd=3, ncl=4, fdl=5, prox=1)
    assert dev.log[-1] == ('wblk', m.MHDX + 4, [2, 3, 4, 5])
    mpr.baseline(rft=2, mhd=9, nhd=3, ncl=4, fdl=5)
    assert dev.log[-1] == ('wblk', m.MHD + 8, [3, 4, 5])


def test_threshold_all_channels_default_release():
    mpr, dev, _ = make()
    mpr.threshold(100)
    rel = int(0.6 * 100)
    for i in range(m.NCH):
        assert dev.mem[m.TTH + 2 * i] == 100
        assert dev.mem[m.RTH + 2 * i] == rel
    assert len(dev.log) == 2 * m.NCH


def test_threshold_single_channel_explicit_release():
    mpr, dev, _ = make()
    mpr.threshold(40, release=10, channel=3)
    assert dev.log == [('wb', m.TTH + 6, 40), ('wb', m.RTH + 6, 10)]


def test_debounce_encoding():
    mpr, dev, _ = make()
    mpr.debounce(2)
    assert dev.log[-1] == ('wb', m.DEB, 0x22)
    mpr.debounce(3, 5)
    assert dev.log[-1] == ('wb', m.DEB, 0x53)


def test_filter_default_word():
    mpr, dev, _ = make()
    mpr.filter()
    assert dev.log[-1] == ('ww', m.AFE1, 0x2410)


def test_charge_preserves_other_nibble():
    mpr, dev, _ = make()
    dev.mem[m.CDT + 1] = 0x05
    mpr.charge(3, cdc=20, cdt=2)
    assert dev.mem[m.CDC + 3] == 20
    assert dev.mem[m.CDT + 1] == 0x25
    dev.mem[m.CDT + 1] = 0x25
    mpr.charge(2, cdc=1, cdt=7)
    assert dev.mem[m.CDT + 1] == 0x27


def test_auto_config_defaults():
    mpr, dev, _ = make()
    mpr.auto_config()
    assert dev.mem[m.ACNF_C0] == 0x076F
    assert dev.mem[m.ACNF_USL] == 200
    assert dev.mem[m.ACNF_LSL] == 130
    assert dev.mem[m.ACNF_TL] == 180


def test_update_touch_state_calls_handlers_on_change():
    mpr, dev, _ = make()
    calls = []
    mpr.on_touch(1, lambda state, ch: calls.append((state, ch)))
    dev.mem[m.ETS] = 0b10
    mpr.update_touch_state()
    assert calls == [(True, 1)]
    assert mpr.is_touched(1)
    assert not mpr.is_touched(0)
    # no change -> no call
    mpr.update_touch_state()
    assert calls == [(True, 1)]
    dev.mem[m.ETS] = 0
    mpr.update_touch_state()
    assert calls == [(True, 1), (False, 1)]
    assert not mpr.is_touched(1)


def test_on_touch_none_removes_handler():
    mpr, dev, _ = make()
    calls = []
    mpr.on_touch(2, lambda state, ch: calls.append((state, ch)))
    mpr.on_touch(2, None)
    dev.mem[m.ETS] = 0b100
    mpr.update_touch_state()
    assert calls == []
    assert mpr.is_touched(2)


def test_gpio_set_and_status():
    mpr, dev, _ = make()
    mpr.gpio_set(4, True)
    assert dev.log[-1] == ('wb', m.GPIO_SET, 1)
    mpr.gpio_set(11, False)
    assert dev.log[-1] == ('wb', m.GPIO_CLR, 1 << 7)
    dev.mem[m.GPIO_DAT] = 0xa5
    assert mpr.gpio_status() == 0xa5
    with pytest.raises(AssertionError):
        mpr.gpio_set(3, True)


def test_gpio_setup_sets_bits(capsys):
    mpr, dev, _ = make()
    mpr.gpio_setup(5, output=1, mode=2)
    assert dev.mem[m.GPIO_CTL0] == 0b10
    assert dev.mem[m.GPIO_CTL1] == 0
    assert dev.mem[m.GPIO_DIR] == 0b10
    assert dev.mem[m.GPIO_EN] == 0b10
    with pytest.raises(AssertionError):
        mpr.gpio_setup(12, output=1)


def test_setup_sequence():
    mpr, dev, _ = make(setup=1, reset=1, threshold=50, channels=12)
    assert dev.log[0] == ('wb', m.SRESET, 0x63)
    assert dev.log[1] == ('wb', m.ECR, 0xC0)
    assert dev.log[-1] == ('wb', m.ECR, 0xC0 | 12)
    assert dev.mem[m.TTH] == 50
    assert dev.mem[m.TTH + 24] == 50 >> 2
    assert dev.mem[m.DEB] == 0x22
    assert dev.mem[m.AFE1] == ((((1 << 5) | (1 << 3) | 0) << 8) | (1 << 6) | 30)
