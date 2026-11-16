import pytest

from piripherals import mpr121
from piripherals.mpr121 import MPR121


class FakeDevice:
    """In-memory register map emulating the device wrapper used by MPR121."""

    def __init__(self):
        self.regs = bytearray(256)
        self.writes = []

    def read_byte(self, reg):
        return self.regs[reg]

    def write_byte(self, reg, value):
        self.writes.append((reg, value))
        self.regs[reg] = value & 0xff

    def read_word(self, reg):
        return self.regs[reg] | (self.regs[reg + 1] << 8)

    def write_word(self, reg, value):
        self.writes.append((reg, value))
        self.regs[reg] = value & 0xff
        self.regs[reg + 1] = (value >> 8) & 0xff

    def read_block(self, reg, n):
        return list(self.regs[reg:reg + n])

    def write_block(self, reg, data):
        self.writes.append((reg, list(data)))
        for i, d in enumerate(data):
            self.regs[reg + i] = d & 0xff


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


def test_init_uses_given_address_and_does_not_touch_device():
    bus = FakeBus()
    m = MPR121(bus=bus, addr=0x5b, handlers=0, setup=0, reset=0)
    assert bus.addr == 0x5b
    assert m.addr == 0x5b
    assert bus.dev.writes == []


def test_init_with_reset_only_writes_soft_reset():
    bus = FakeBus()
    MPR121(bus=bus, handlers=0, setup=0, reset=1)
    assert bus.dev.writes == [(mpr121.SRESET, 0x63)]


def test_configure_encodes_ecr(env):
    m, dev = env
    m.configure(cl=2, prox=1, touch=5)
    assert dev.regs[mpr121.ECR] == (2 << 6) | (1 << 4) | 5
    m.configure(prox=0, touch=0)
    assert dev.regs[mpr121.ECR] == 0xc0


def test_touched_masks_status_bits(env):
    m, dev = env
    dev.regs[mpr121.ETS] = 0xff
    dev.regs[mpr121.ETS + 1] = 0x1f
    assert m.touched() == 0x1fff
    dev.regs[mpr121.ETS] = 0x05
    dev.regs[mpr121.ETS + 1] = 0x00
    assert m.touched() == 0x05


def test_touched_raises_on_overcurrent(env):
    m, dev = env
    dev.regs[mpr121.ETS + 1] = 0x80
    with pytest.raises(Exception, match='overcurrent'):
        m.touched()
    # without raising, the flag bit is masked out
    assert m.touched(raise_on_failure=0) == 0


def test_touched_raises_on_out_of_range(env):
    m, dev = env
    dev.regs[mpr121.OOR] = 0x01
    with pytest.raises(Exception, match='out of range'):
        m.touched()
    assert m.touched(raise_on_failure=0) == 0


def test_out_of_range_failure_flags(env):
    m, dev = env
    dev.regs[mpr121.OOR + 1] = 0x80
    with pytest.raises(Exception, match='auto config failed'):
        m.out_of_range()
    dev.regs[mpr121.OOR + 1] = 0x40
    with pytest.raises(Exception, match='auto reconfig failed'):
        m.out_of_range()
    assert m.out_of_range(raise_on_failure=0) == 0


def test_update_touch_state_invokes_handlers_on_change(env):
    m, dev = env
    calls = []
    m.on_touch(3, lambda state, ch: calls.append((state, ch)))

    dev.regs[mpr121.ETS] = 1 << 3
    m.update_touch_state()
    assert calls == [(True, 3)]
    assert m.is_touched(3)
    assert not m.is_touched(2)

    # no change -> no new call
    m.update_touch_state()
    assert calls == [(True, 3)]

    dev.regs[mpr121.ETS] = 0
    m.update_touch_state()
    assert calls == [(True, 3), (False, 3)]
    assert not m.is_touched(3)


def test_handler_only_called_for_its_channel(env):
    m, dev = env
    calls = []
    m.on_touch(0, lambda state, ch: calls.append(('a', state, ch)))
    m.on_touch(1, lambda state, ch: calls.append(('b', state, ch)))
    dev.regs[mpr121.ETS] = 0b10
    m.update_touch_state()
    assert calls == [('b', True, 1)]


def test_handle_touch_falls_back_to_single_argument_handler(env):
    m, _ = env
    calls = []
    m._handlers[2] = lambda state: calls.append(state)
    m._handle_touch(0b100, 0b100)
    assert calls == [True]
    m._handle_touch(0, 0b100)
    assert calls == [True, False]


def test_on_touch_none_removes_handler(env):
    m, dev = env
    calls = []
    m.on_touch(4, lambda s, c: calls.append(s))
    m.on_touch(4, None)
    dev.regs[mpr121.ETS] = 1 << 4
    m.update_touch_state()
    assert calls == []


def test_threshold_all_channels_default_release(env):
    m, dev = env
    m.threshold(50)
    for ch in range(mpr121.NCH):
        assert dev.regs[mpr121.TTH + 2 * ch] == 50
        assert dev.regs[mpr121.RTH + 2 * ch] == 30


def test_threshold_single_channel_explicit_release(env):
    m, dev = env
    m.threshold(40, release=10, channel=5)
    assert dev.regs[mpr121.TTH + 10] == 40
    assert dev.regs[mpr121.RTH + 10] == 10
    assert dev.regs[mpr121.TTH + 12] == 0
    assert dev.regs[mpr121.TTH + 8] == 0


def test_debounce_encoding(env):
    m, dev = env
    m.debounce(touch=3, release=5)
    assert dev.regs[mpr121.DEB] == (5 << 4) | 3
    m.debounce(2)
    assert dev.regs[mpr121.DEB] == 0x22


def test_filter_writes_afe_registers(env):
    m, dev = env
    m.filter()
    assert dev.regs[mpr121.AFE1] == 16
    assert dev.regs[mpr121.AFE2] == (1 << 5) | 4


def test_charge_preserves_other_nibble(env):
    m, dev = env
    dev.regs[mpr121.CDT + 1] = 0x0f
    m.charge(3, cdc=10, cdt=5)
    assert dev.regs[mpr121.CDC + 3] == 10
    assert dev.regs[mpr121.CDT + 1] == 0x5f

    dev.regs[mpr121.CDT + 1] = 0xf0
    m.charge(2, cdc=7, cdt=3)
    assert dev.regs[mpr121.CDC + 2] == 7
    assert dev.regs[mpr121.CDT + 1] == 0xf3


def test_electrode_data_combines_bytes(env):
    m, dev = env
    for i in range(mpr121.NCH):
        dev.regs[mpr121.EFD + 2 * i] = i
        dev.regs[mpr121.EFD + 2 * i + 1] = 0x01 if i == 2 else 0
    data = m.electrode_data()
    assert len(data) == mpr121.NCH
    assert data[0] == 0
    assert data[1] == 1
    assert data[2] == 0x102


def test_baseline_read_shifts_values(env):
    m, dev = env
    for i in range(mpr121.NCH):
        dev.regs[mpr121.EBL + i] = i + 1
    assert m.baseline() == [(i + 1) << 2 for i in range(mpr121.NCH)]


def test_baseline_write_registers(env):
    m, dev = env
    m.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert list(dev.regs[mpr121.MHD:mpr121.MHD + 4]) == [5, 1, 3, 20]
    m.baseline(rft=1, mhd=6, nhd=2, ncl=4, fdl=21, prox=1)
    assert list(dev.regs[mpr121.MHDX + 4:mpr121.MHDX + 8]) == [6, 2, 4, 21]
    m.baseline(rft=2, mhd=99, nhd=7, ncl=8, fdl=9)
    # touched scenario has no mhd, three values written
    assert list(dev.regs[mpr121.MHD + 8:mpr121.MHD + 11]) == [7, 8, 9]


def test_auto_config_defaults(env):
    m, dev = env
    m.auto_config()
    assert dev.regs[mpr121.ACNF_C0] == 0x6f
    assert dev.regs[mpr121.ACNF_C1] == 0x07
    assert dev.regs[mpr121.ACNF_USL] == 200
    assert dev.regs[mpr121.ACNF_LSL] == 130
    assert dev.regs[mpr121.ACNF_TL] == 180


def test_gpio_setup_and_status(env):
    m, dev = env
    m.gpio_setup(5, output=1, mode=3)
    assert dev.regs[mpr121.GPIO_CTL0] == 0b10
    assert dev.regs[mpr121.GPIO_CTL1] == 0b10
    assert dev.regs[mpr121.GPIO_DIR] == 0b10
    assert dev.regs[mpr121.GPIO_EN] == 0b10

    m.gpio_setup(5, output=0, mode=2, enable=1)
    assert dev.regs[mpr121.GPIO_CTL0] == 0b10
    assert dev.regs[mpr121.GPIO_CTL1] == 0
    assert dev.regs[mpr121.GPIO_DIR] == 0

    dev.regs[mpr121.GPIO_DAT] = 0xa5
    assert m.gpio_status() == 0xa5


def test_gpio_setup_and_set_validate_channel(env):
    m, _ = env
    with pytest.raises(AssertionError):
        m.gpio_setup(3, output=1)
    with pytest.raises(AssertionError):
        m.gpio_set(12, 1)


def test_gpio_set_uses_set_and_clear_registers(env):
    m, dev = env
    m.gpio_set(6, 1)
    m.gpio_set(7, 0)
    assert (mpr121.GPIO_SET, 1 << 2) in dev.writes
    assert (mpr121.GPIO_CLR, 1 << 3) in dev.writes


def test_setup_configures_device(env):
    m, dev = env
    m.setup(reset=1, channels=12, threshold=50, debounce=2)
    assert (mpr121.SRESET, 0x63) in dev.writes
    assert dev.regs[mpr121.ECR] == (3 << 6) | 12 or dev.regs[mpr121.ECR] == 12
    assert dev.regs[mpr121.ECR] & 0x0f == 12
    assert dev.regs[mpr121.TTH] == 50
    assert dev.regs[mpr121.RTH] == 30
    assert dev.regs[mpr121.TTH + 24] == 12
    assert dev.regs[mpr121.RTH + 24] == 7
    assert dev.regs[mpr121.DEB] == 0x22
    assert list(dev.regs[mpr121.MHD:mpr121.MHD + 4]) == [5, 1, 3, 20]
    assert list(dev.regs[mpr121.MHDX:mpr121.MHDX + 4]) == [1, 1, 3, 20]
    assert dev.regs[mpr121.ACNF_USL] == 200
