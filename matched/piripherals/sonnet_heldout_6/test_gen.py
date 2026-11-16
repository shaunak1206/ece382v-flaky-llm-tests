import pytest

from piripherals import mpr121
from piripherals.mpr121 import MPR121


class FakeDevice:
    """In-memory register file emulating an I2C device."""

    def __init__(self):
        self.regs = {}
        self.log = []

    def write_byte(self, reg, val):
        self.log.append(('byte', reg, val))
        self.regs[reg] = val

    def read_byte(self, reg):
        return self.regs.get(reg, 0)

    def write_word(self, reg, val):
        self.log.append(('word', reg, val))
        self.regs[reg] = val & 0xff
        self.regs[reg + 1] = (val >> 8) & 0xff

    def read_word(self, reg):
        return self.regs.get(reg, 0) | (self.regs.get(reg + 1, 0) << 8)

    def write_block(self, reg, data):
        self.log.append(('block', reg, list(data)))
        for i, v in enumerate(data):
            self.regs[reg + i] = v

    def read_block(self, reg, n):
        return [self.regs.get(reg + i, 0) for i in range(n)]


class FakeBus:
    def __init__(self):
        self.dev = FakeDevice()
        self.addr = None

    def device(self, addr):
        self.addr = addr
        return self.dev


@pytest.fixture
def env():
    bus = FakeBus()
    m = MPR121(bus=bus, handlers=0, setup=0, reset=0)
    return m, bus.dev


def test_init_without_setup_or_reset_writes_nothing():
    bus = FakeBus()
    m = MPR121(bus=bus, addr=0x5b, handlers=0, setup=0, reset=0)
    assert bus.addr == 0x5b
    assert m.addr == 0x5b
    assert bus.dev.log == []


def test_init_reset_only_writes_soft_reset():
    bus = FakeBus()
    MPR121(bus=bus, handlers=0, setup=0, reset=1)
    assert bus.dev.log == [('byte', mpr121.SRESET, 0x63)]


def test_reset(env):
    m, dev = env
    m.reset()
    assert dev.log[-1] == ('byte', 0x80, 0x63)


def test_configure_defaults_and_stop_mode(env):
    m, dev = env
    m.configure()
    assert dev.regs[mpr121.ECR] == (3 << 6) | 12
    m.configure(cl=0, prox=0, touch=0)
    assert dev.regs[mpr121.ECR] == 0
    m.configure(cl=1, prox=2, touch=5)
    assert dev.regs[mpr121.ECR] == (1 << 6) | (2 << 4) | 5


def test_touched_masks_status_bits(env):
    m, dev = env
    dev.regs[mpr121.ETS] = 0x23
    dev.regs[mpr121.ETS + 1] = 0x01
    assert m.touched() == 0x123


def test_touched_raises_on_overcurrent(env):
    m, dev = env
    dev.regs[mpr121.ETS + 1] = 0x80  # bit 15
    with pytest.raises(Exception, match='overcurrent'):
        m.touched()
    # no raise when disabled, failure bit masked away
    assert m.touched(raise_on_failure=0) == 0


def test_touched_raises_on_out_of_range(env):
    m, dev = env
    dev.regs[mpr121.OOR + 1] = 0x80
    with pytest.raises(Exception, match='auto config failed'):
        m.touched()


def test_out_of_range_failure_bits(env):
    m, dev = env
    dev.regs[mpr121.OOR] = 0x05
    assert m.out_of_range() == 0x05
    dev.regs[mpr121.OOR + 1] = 0x40  # bit 14
    with pytest.raises(Exception, match='auto reconfig failed'):
        m.out_of_range()
    assert m.out_of_range(raise_on_failure=0) == 0x05
    dev.regs[mpr121.OOR + 1] = 0x80  # bit 15
    with pytest.raises(Exception, match='auto config failed'):
        m.out_of_range()


def test_update_touch_state_calls_handlers_on_change(env):
    m, dev = env
    events = []
    m.on_touch(1, lambda state, ch: events.append((ch, state)))
    m.on_touch(3, lambda state, ch: events.append((ch, state)))

    dev.regs[mpr121.ETS] = 0b0010
    m.update_touch_state()
    assert events == [(1, True)]
    assert m.is_touched(1)
    assert not m.is_touched(3)

    dev.regs[mpr121.ETS] = 0b1000
    m.update_touch_state()
    assert events == [(1, True), (1, False), (3, True)]
    assert not m.is_touched(1)
    assert m.is_touched(3)


def test_update_touch_state_no_event_when_unchanged(env):
    m, dev = env
    events = []
    m.on_touch(0, lambda state, ch: events.append((ch, state)))
    dev.regs[mpr121.ETS] = 1
    m.update_touch_state()
    m.update_touch_state()
    assert events == [(0, True)]


def test_on_touch_remove_handler(env):
    m, dev = env
    events = []
    m.on_touch(0, lambda state, ch: events.append((ch, state)))
    m.on_touch(0, None)
    dev.regs[mpr121.ETS] = 1
    m.update_touch_state()
    assert events == []
    assert m.is_touched(0)


def test_threshold_single_channel_default_release(env):
    m, dev = env
    m.threshold(50, channel=2)
    assert dev.regs[mpr121.TTH + 4] == 50
    assert dev.regs[mpr121.RTH + 4] == 30  # int(0.6 * 50)


def test_threshold_all_channels(env):
    m, dev = env
    m.threshold(40, release=10)
    for i in range(mpr121.NCH):
        assert dev.regs[mpr121.TTH + 2 * i] == 40
        assert dev.regs[mpr121.RTH + 2 * i] == 10


def test_debounce(env):
    m, dev = env
    m.debounce(2)
    assert dev.regs[mpr121.DEB] == 0x22
    m.debounce(3, 5)
    assert dev.regs[mpr121.DEB] == 0x53


def test_filter_defaults_pack_into_word(env):
    m, dev = env
    m.filter()
    assert dev.regs[mpr121.AFE1] == 16
    assert dev.regs[mpr121.AFE2] == (1 << 5) | 4
    m.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    assert dev.regs[mpr121.AFE1] == (1 << 6) | 30
    assert dev.regs[mpr121.AFE2] == (1 << 5) | (1 << 3)


def test_electrode_data_combines_bytes(env):
    m, dev = env
    dev.regs[mpr121.EFD] = 0x34
    dev.regs[mpr121.EFD + 1] = 0x02
    dev.regs[mpr121.EFD + 2 * 12] = 0xff
    dev.regs[mpr121.EFD + 2 * 12 + 1] = 0x03
    data = m.electrode_data()
    assert len(data) == mpr121.NCH
    assert data[0] == 0x234
    assert data[1] == 0
    assert data[12] == 0x3ff


def test_baseline_read_and_write(env):
    m, dev = env
    dev.regs[mpr121.EBL] = 10
    dev.regs[mpr121.EBL + 5] = 0xff
    bl = m.baseline()
    assert len(bl) == mpr121.NCH
    assert bl[0] == 40
    assert bl[5] == 0xff << 2

    m.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert dev.log[-1] == ('block', mpr121.MHD, [5, 1, 3, 20])
    m.baseline(rft=1, mhd=1, nhd=2, ncl=3, fdl=4, prox=1)
    assert dev.log[-1] == ('block', mpr121.MHDX + 4, [1, 2, 3, 4])
    m.baseline(rft=2, mhd=9, nhd=1, ncl=2, fdl=3)
    assert dev.log[-1] == ('block', mpr121.MHD + 8, [1, 2, 3])


def test_charge_sets_current_and_time_nibble(env):
    m, dev = env
    dev.regs[mpr121.CDT + 1] = 0x05
    m.charge(3, cdc=20, cdt=2)
    assert dev.regs[mpr121.CDC + 3] == 20
    assert dev.regs[mpr121.CDT + 1] == 0x25
    m.charge(2, cdc=1, cdt=7)
    assert dev.regs[mpr121.CDT + 1] == 0x27


def test_auto_config_defaults(env):
    m, dev = env
    m.auto_config()
    assert dev.regs[mpr121.ACNF_C0] == 0x6f
    assert dev.regs[mpr121.ACNF_C0 + 1] == 0x07
    assert dev.regs[mpr121.ACNF_USL] == 200
    assert dev.regs[mpr121.ACNF_LSL] == 130
    assert dev.regs[mpr121.ACNF_TL] == 180


def test_gpio_set_and_status(env, capsys):
    m, dev = env
    m.gpio_set(5, 1)
    assert dev.log[-1] == ('byte', mpr121.GPIO_SET, 0b10)
    m.gpio_set(11, 0)
    assert dev.log[-1] == ('byte', mpr121.GPIO_CLR, 1 << 7)
    dev.regs[mpr121.GPIO_DAT] = 0xa5
    assert m.gpio_status() == 0xa5
    with pytest.raises(AssertionError):
        m.gpio_set(3, 1)
    with pytest.raises(AssertionError):
        m.gpio_set(12, 1)


def test_gpio_setup_sets_bits(env, capsys):
    m, dev = env
    m.gpio_setup(5, output=1, mode=3, enable=1)
    bit = 1 << 1
    assert dev.regs[mpr121.GPIO_CTL0] & bit
    assert dev.regs[mpr121.GPIO_CTL1] & bit
    assert dev.regs[mpr121.GPIO_DIR] & bit
    assert dev.regs[mpr121.GPIO_EN] & bit
    with pytest.raises(AssertionError):
        m.gpio_setup(2, output=1)


def test_setup_resets_then_enables_channels(env):
    m, dev = env
    m.setup(channels=8, threshold=60, debounce=3)
    assert dev.log[0] == ('byte', mpr121.SRESET, 0x63)
    ecr_writes = [e for e in dev.log if e[0] == 'byte' and e[1] == mpr121.ECR]
    assert ecr_writes[0][2] == 0  # stopped during configuration
    assert ecr_writes[-1][2] == (3 << 6) | 8
    assert dev.regs[mpr121.TTH] == 60
    assert dev.regs[mpr121.TTH + 2 * 12] == 60 >> 2
    assert dev.regs[mpr121.DEB] == 0x33
    assert dev.regs[mpr121.ACNF_USL] == 200


def test_setup_without_reset_or_auto_config(env):
    m, dev = env
    m.setup(reset=0, auto_config=0)
    assert ('byte', mpr121.SRESET, 0x63) not in dev.log
    assert mpr121.ACNF_USL not in dev.regs
