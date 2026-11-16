import pytest

from piripherals import mpr121 as m
from piripherals.mpr121 import MPR121


class FakeDevice:
    def __init__(self):
        self.mem = [0] * 256
        self.log = []

    def write_byte(self, reg, val):
        self.log.append(('byte', reg, val))
        self.mem[reg] = val

    def read_byte(self, reg):
        return self.mem[reg]

    def write_word(self, reg, val):
        self.log.append(('word', reg, val))
        self.mem[reg] = val & 0xff
        self.mem[reg + 1] = (val >> 8) & 0xff

    def read_word(self, reg):
        return self.mem[reg] | (self.mem[reg + 1] << 8)

    def write_block(self, reg, data):
        self.log.append(('block', reg, list(data)))
        for i, d in enumerate(data):
            self.mem[reg + i] = d

    def read_block(self, reg, n):
        return list(self.mem[reg:reg + n])

    def set_word(self, reg, val):
        self.mem[reg] = val & 0xff
        self.mem[reg + 1] = (val >> 8) & 0xff


class FakeBus:
    def __init__(self):
        self.dev = FakeDevice()
        self.addrs = []

    def device(self, addr):
        self.addrs.append(addr)
        return self.dev


def make(**kw):
    bus = FakeBus()
    kw.setdefault('handlers', 0)
    kw.setdefault('setup', 0)
    kw.setdefault('reset', 0)
    mpr = MPR121(bus=bus, **kw)
    return mpr, bus.dev, bus


def test_init_without_setup_does_not_write():
    mpr, dev, bus = make(addr=0x5b)
    assert bus.addrs == [0x5b]
    assert mpr.addr == 0x5b
    assert dev.log == []


def test_init_reset_only():
    mpr, dev, _ = make(reset=1)
    assert dev.log == [('byte', m.SRESET, 0x63)]


def test_configure_writes_ecr():
    mpr, dev, _ = make()
    mpr.configure(cl=3, prox=0, touch=12)
    assert dev.log == [('byte', m.ECR, (3 << 6) | 12)]
    mpr.configure(cl=1, prox=2, touch=5)
    assert dev.log[-1] == ('byte', m.ECR, (1 << 6) | (2 << 4) | 5)


def test_threshold_all_channels_default_release():
    mpr, dev, _ = make()
    mpr.threshold(50)
    assert len(dev.log) == 2 * m.NCH
    for i in range(m.NCH):
        assert dev.mem[m.TTH + 2 * i] == 50
        assert dev.mem[m.RTH + 2 * i] == 30


def test_threshold_single_channel_explicit_release():
    mpr, dev, _ = make()
    mpr.threshold(80, release=10, channel=3)
    assert dev.log == [('byte', m.TTH + 6, 80), ('byte', m.RTH + 6, 10)]


def test_debounce():
    mpr, dev, _ = make()
    mpr.debounce(2)
    assert dev.log == [('byte', m.DEB, 0x22)]
    mpr.debounce(1, 3)
    assert dev.log[-1] == ('byte', m.DEB, 0x31)


def test_filter_word_encoding():
    mpr, dev, _ = make()
    mpr.filter()
    assert dev.log == [('word', m.AFE1, 0x2410)]


def test_touched_masks_status_bits():
    mpr, dev, _ = make()
    dev.set_word(m.ETS, 0x0abc)
    assert mpr.touched() == 0x0abc
    dev.set_word(m.ETS, 0x1fff)
    assert mpr.touched() == 0x1fff


def test_touched_overcurrent_raises_only_when_requested():
    mpr, dev, _ = make()
    dev.set_word(m.ETS, 0x8005)
    with pytest.raises(Exception, match='overcurrent'):
        mpr.touched()
    assert mpr.touched(raise_on_failure=0) == 5


def test_touched_out_of_range_raises():
    mpr, dev, _ = make()
    dev.set_word(m.ETS, 0x0001)
    dev.set_word(m.OOR, 0x8000)
    with pytest.raises(Exception, match='auto config failed'):
        mpr.touched()


def test_out_of_range_flags():
    mpr, dev, _ = make()
    dev.set_word(m.OOR, 0x4003)
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr.out_of_range()
    assert mpr.out_of_range(raise_on_failure=0) == 3
    dev.set_word(m.OOR, 0x0010)
    assert mpr.out_of_range() == 0x10


def test_electrode_data_combines_bytes():
    mpr, dev, _ = make()
    for i in range(m.NCH):
        dev.mem[m.EFD + 2 * i] = i
        dev.mem[m.EFD + 2 * i + 1] = i % 4
    data = mpr.electrode_data()
    assert data == [i + ((i % 4) << 8) for i in range(m.NCH)]


def test_baseline_read_shifts_values():
    mpr, dev, _ = make()
    for i in range(m.NCH):
        dev.mem[m.EBL + i] = i + 1
    assert mpr.baseline() == [(i + 1) << 2 for i in range(m.NCH)]


def test_baseline_write_registers():
    mpr, dev, _ = make()
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    mpr.baseline(rft=1, mhd=2, nhd=3, ncl=4, fdl=5, prox=1)
    mpr.baseline(rft=2, mhd=9, nhd=6, ncl=7, fdl=8)
    assert dev.log == [
        ('block', m.MHD, [5, 1, 3, 20]),
        ('block', m.MHDX + 4, [2, 3, 4, 5]),
        ('block', m.MHD + 8, [6, 7, 8]),
    ]


def test_charge_updates_only_relevant_nibble():
    mpr, dev, _ = make()
    dev.mem[m.CDT + 1] = 0x0a
    mpr.charge(3, cdc=10, cdt=5)
    assert dev.mem[m.CDC + 3] == 10
    assert dev.mem[m.CDT + 1] == 0x5a
    mpr.charge(2, cdc=1, cdt=7)
    assert dev.mem[m.CDT + 1] == 0x5a
    assert dev.mem[m.CDT + 1] & 0x0f == 0x0a
    assert dev.mem[m.CDT + 1] >> 4 == 5


def test_gpio_set_and_clear_and_bounds():
    mpr, dev, _ = make()
    mpr.gpio_set(4, 1)
    mpr.gpio_set(11, 0)
    assert dev.log == [('byte', m.GPIO_SET, 1), ('byte', m.GPIO_CLR, 0x80)]
    with pytest.raises(AssertionError):
        mpr.gpio_set(3, 1)
    with pytest.raises(AssertionError):
        mpr.gpio_set(12, 1)


def test_gpio_setup_sets_bits_and_status(capsys):
    mpr, dev, _ = make()
    mpr.gpio_setup(5, output=1, mode=3, enable=1)
    assert dev.mem[m.GPIO_CTL0] == 0b10
    assert dev.mem[m.GPIO_CTL1] == 0b10
    assert dev.mem[m.GPIO_DIR] == 0b10
    assert dev.mem[m.GPIO_EN] == 0b10
    mpr.gpio_setup(5, output=0, mode=1, enable=0)
    assert dev.mem[m.GPIO_CTL0] == 0
    assert dev.mem[m.GPIO_CTL1] == 0
    assert dev.mem[m.GPIO_DIR] == 0
    assert dev.mem[m.GPIO_EN] == 0
    dev.mem[m.GPIO_DAT] = 0x42
    assert mpr.gpio_status() == 0x42
    with pytest.raises(AssertionError):
        mpr.gpio_setup(3, 1)


def test_update_touch_state_invokes_handlers_on_changes():
    mpr, dev, _ = make()
    calls = []
    mpr.on_touch(0, lambda *a: calls.append((0, a[0])))
    mpr.on_touch(2, lambda *a: calls.append((2, a[0])))

    dev.set_word(m.ETS, 0b101)
    mpr.update_touch_state()
    assert calls == [(0, True), (2, True)]
    assert mpr.is_touched(0)
    assert not mpr.is_touched(1)
    assert mpr.is_touched(2)

    del calls[:]
    mpr.update_touch_state()  # no change, no calls
    assert calls == []

    dev.set_word(m.ETS, 0b100)
    mpr.update_touch_state()
    assert calls == [(0, False)]
    assert not mpr.is_touched(0)


def test_setup_sequence():
    mpr, dev, _ = make()
    mpr.setup(reset=1, channels=12, threshold=50, debounce=2)
    assert dev.log[0] == ('byte', m.SRESET, 0x63)
    assert dev.log[1] == ('byte', m.ECR, 0)
    assert dev.log[-1] == ('byte', m.ECR, (3 << 6) | 12)
    # proximity threshold is a quarter of the touch threshold
    assert dev.mem[m.TTH + 24] == 12
    assert dev.mem[m.TTH] == 50
    assert dev.mem[m.DEB] == 0x22
    assert dev.mem[m.ACNF_USL] == 200
    assert dev.mem[m.ACNF_LSL] == 130
    assert dev.mem[m.ACNF_TL] == 180


def test_setup_without_auto_config_skips_acnf():
    mpr, dev, _ = make()
    mpr.setup(reset=0, auto_config=0)
    regs = [e[1] for e in dev.log]
    assert m.SRESET not in regs
    assert m.ACNF_C0 not in regs
    assert m.ACNF_USL not in regs
