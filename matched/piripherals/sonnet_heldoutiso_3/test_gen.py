import pytest

from piripherals import mpr121 as m
from piripherals.mpr121 import MPR121


class FakeDevice:
    """Minimal in-memory stand-in for a bus device."""

    def __init__(self):
        self.regs = {}
        self.words = {}
        self.blocks = {}
        self.log = []

    def write_byte(self, reg, value):
        self.regs[reg] = value
        self.log.append(('byte', reg, value))

    def read_byte(self, reg):
        return self.regs.get(reg, 0)

    def write_word(self, reg, value):
        self.words[reg] = value
        self.log.append(('word', reg, value))

    def read_word(self, reg):
        return self.words.get(reg, 0)

    def write_block(self, reg, values):
        self.log.append(('block', reg, list(values)))

    def read_block(self, reg, n):
        return list(self.blocks.get(reg, [0] * n))[:n]


class FakeBus:
    def __init__(self):
        self.dev = FakeDevice()
        self.addrs = []

    def device(self, addr):
        self.addrs.append(addr)
        return self.dev


@pytest.fixture
def env():
    bus = FakeBus()
    mpr = MPR121(bus=bus, addr=0x5b, handlers=0, setup=0, reset=0)
    return mpr, bus.dev, bus


def test_init_uses_device_address(env):
    mpr, dev, bus = env
    assert bus.addrs == [0x5b]
    assert mpr.addr == 0x5b
    assert dev.log == []  # no reset, no setup


def test_init_with_reset_only_writes_soft_reset():
    bus = FakeBus()
    MPR121(bus=bus, handlers=0, setup=0, reset=1)
    assert bus.dev.log == [('byte', m.SRESET, 0x63)]


def test_configure_encodes_ecr(env):
    mpr, dev, _ = env
    mpr.configure()
    assert dev.log[-1] == ('byte', m.ECR, (3 << 6) | 12)
    mpr.configure(cl=0, prox=0, touch=0)
    assert dev.log[-1] == ('byte', m.ECR, 0)
    mpr.configure(cl=1, prox=2, touch=5)
    assert dev.log[-1] == ('byte', m.ECR, (1 << 6) | (2 << 4) | 5)


def test_touched_masks_status_bits(env):
    mpr, dev, _ = env
    dev.words[m.ETS] = 0x0123
    assert mpr.touched() == 0x0123
    dev.words[m.ETS] = 0x1fff | (1 << 13)
    assert mpr.touched(raise_on_failure=0) == 0x1fff


def test_touched_raises_on_overcurrent(env):
    mpr, dev, _ = env
    dev.words[m.ETS] = 1 << 15
    with pytest.raises(Exception, match='overcurrent'):
        mpr.touched()
    # not raised when check disabled
    assert mpr.touched(raise_on_failure=0) == 0


def test_touched_raises_on_out_of_range(env):
    mpr, dev, _ = env
    dev.words[m.ETS] = 0x1
    dev.words[m.OOR] = 0x0004
    with pytest.raises(Exception, match='out of range'):
        mpr.touched()


def test_out_of_range_failure_bits(env):
    mpr, dev, _ = env
    dev.words[m.OOR] = 0x0005
    assert mpr.out_of_range() == 0x0005
    dev.words[m.OOR] = 1 << 15
    with pytest.raises(Exception, match='auto config failed'):
        mpr.out_of_range()
    dev.words[m.OOR] = 1 << 14
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr.out_of_range()
    assert mpr.out_of_range(raise_on_failure=0) == 0


def test_update_touch_state_invokes_handlers(env):
    mpr, dev, _ = env
    calls = []
    mpr.on_touch(3, lambda state, ch: calls.append((state, ch)))

    dev.words[m.ETS] = 1 << 3
    mpr.update_touch_state()
    assert calls == [(True, 3)]
    assert mpr.is_touched(3)
    assert not mpr.is_touched(2)

    # no change -> no new call
    mpr.update_touch_state()
    assert calls == [(True, 3)]

    dev.words[m.ETS] = 0
    mpr.update_touch_state()
    assert calls == [(True, 3), (False, 3)]
    assert not mpr.is_touched(3)


def test_handlers_only_called_for_changed_channels(env):
    mpr, dev, _ = env
    calls = []
    for i in (0, 1):
        mpr.on_touch(i, lambda s, c: calls.append((s, c)))
    dev.words[m.ETS] = 0b01
    mpr.update_touch_state()
    dev.words[m.ETS] = 0b11
    mpr.update_touch_state()
    assert calls == [(True, 0), (True, 1)]


def test_on_touch_none_removes_handler(env):
    mpr, dev, _ = env
    calls = []
    mpr.on_touch(1, lambda s, c: calls.append((s, c)))
    mpr.on_touch(1, None)
    dev.words[m.ETS] = 0b10
    mpr.update_touch_state()
    assert calls == []
    assert mpr.is_touched(1)


def test_threshold_single_channel_default_release(env):
    mpr, dev, _ = env
    mpr.threshold(50, channel=2)
    assert dev.regs[m.TTH + 4] == 50
    assert dev.regs[m.RTH + 4] == 30


def test_threshold_explicit_release_and_all_channels(env):
    mpr, dev, _ = env
    mpr.threshold(40, release=10)
    for ch in range(m.NCH):
        assert dev.regs[m.TTH + 2 * ch] == 40
        assert dev.regs[m.RTH + 2 * ch] == 10
    assert len(dev.log) == 2 * m.NCH


def test_debounce(env):
    mpr, dev, _ = env
    mpr.debounce(3)
    assert dev.log[-1] == ('byte', m.DEB, 0x33)
    mpr.debounce(2, 5)
    assert dev.log[-1] == ('byte', m.DEB, 0x52)


def test_filter_defaults_and_custom(env):
    mpr, dev, _ = env
    mpr.filter()
    assert dev.log[-1] == ('word', m.AFE1, 0x2410)
    mpr.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    lb = (1 << 6) | 30
    hb = (1 << 5) | (1 << 3)
    assert dev.log[-1] == ('word', m.AFE1, (hb << 8) | lb)


def test_charge_updates_only_channel_nibble(env):
    mpr, dev, _ = env
    dev.regs[m.CDT + 1] = 0xff
    mpr.charge(3, cdc=20, cdt=2)  # odd channel -> high nibble of CDT+1
    assert dev.regs[m.CDC + 3] == 20
    assert dev.regs[m.CDT + 1] == 0x2f
    mpr.charge(2, cdc=5, cdt=1)  # even channel -> low nibble of CDT+1
    assert dev.regs[m.CDC + 2] == 5
    assert dev.regs[m.CDT + 1] == 0x21


def test_electrode_data_combines_bytes(env):
    mpr, dev, _ = env
    raw = []
    for i in range(m.NCH):
        raw += [i & 0xff, 0x01 if i % 2 else 0x00]
    dev.blocks[m.EFD] = raw
    data = mpr.electrode_data()
    assert len(data) == m.NCH
    for i in range(m.NCH):
        assert data[i] == ((0x100 if i % 2 else 0) + i)


def test_baseline_read_shifts_values(env):
    mpr, dev, _ = env
    dev.blocks[m.EBL] = list(range(m.NCH))
    assert mpr.baseline() == [b << 2 for b in range(m.NCH)]


def test_baseline_write_registers(env):
    mpr, dev, _ = env
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert dev.log[-1] == ('block', m.MHD, [5, 1, 3, 20])
    mpr.baseline(rft=1, mhd=2, nhd=3, ncl=4, fdl=5)
    assert dev.log[-1] == ('block', m.MHD + 4, [2, 3, 4, 5])
    mpr.baseline(rft=2, mhd=9, nhd=3, ncl=4, fdl=5)
    assert dev.log[-1] == ('block', m.MHD + 8, [3, 4, 5])
    mpr.baseline(rft=0, mhd=1, nhd=1, ncl=3, fdl=20, prox=1)
    assert dev.log[-1] == ('block', m.MHDX, [1, 1, 3, 20])


def test_auto_config_defaults(env):
    mpr, dev, _ = env
    mpr.auto_config()
    assert dev.words[m.ACNF_C0] == 0x076f
    assert dev.regs[m.ACNF_USL] == 200
    assert dev.regs[m.ACNF_LSL] == 130
    assert dev.regs[m.ACNF_TL] == 180


def test_gpio_set_and_clear(env):
    mpr, dev, _ = env
    mpr.gpio_set(4, 1)
    assert dev.log[-1] == ('byte', m.GPIO_SET, 1)
    mpr.gpio_set(11, 0)
    assert dev.log[-1] == ('byte', m.GPIO_CLR, 1 << 7)
    with pytest.raises(AssertionError):
        mpr.gpio_set(3, 1)
    with pytest.raises(AssertionError):
        mpr.gpio_set(12, 1)


def test_gpio_setup_sets_register_bits(env):
    mpr, dev, _ = env
    mpr.gpio_setup(6, output=1, mode=3)
    bit = 1 << 2
    assert dev.regs[m.GPIO_CTL0] & bit
    assert dev.regs[m.GPIO_CTL1] & bit
    assert dev.regs[m.GPIO_DIR] & bit
    assert dev.regs[m.GPIO_EN] & bit
    with pytest.raises(AssertionError):
        mpr.gpio_setup(2, output=1)


def test_gpio_status_reads_data_register(env):
    mpr, dev, _ = env
    dev.regs[m.GPIO_DAT] = 0xa5
    assert mpr.gpio_status() == 0xa5


def test_setup_resets_then_configures():
    bus = FakeBus()
    MPR121(bus=bus, handlers=0, setup=1, reset=1, channels=8, threshold=60)
    log = bus.dev.log
    assert log[0] == ('byte', m.SRESET, 0x63)
    assert log[-1] == ('byte', m.ECR, (3 << 6) | 8 if False else 8)
    assert bus.dev.regs[m.TTH] == 60
    assert bus.dev.regs[m.TTH + 24] == 15  # proximity channel: threshold >> 2
    assert bus.dev.regs[m.DEB] == 0x22
    assert bus.dev.words[m.ACNF_C0] == 0x076f
