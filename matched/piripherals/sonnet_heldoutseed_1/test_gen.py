import pytest

from piripherals import mpr121
from piripherals.mpr121 import MPR121


class FakeDevice:
    """In-memory register file mimicking the bus device API used by MPR121."""

    def __init__(self):
        self.mem = {}
        self.log = []

    def write_byte(self, reg, val):
        self.log.append(('byte', reg, val))
        self.mem[reg] = val

    def read_byte(self, reg):
        return self.mem.get(reg, 0)

    def write_word(self, reg, val):
        self.log.append(('word', reg, val))
        self.mem[reg] = val & 0xff
        self.mem[reg + 1] = (val >> 8) & 0xff

    def read_word(self, reg):
        return self.mem.get(reg, 0) | (self.mem.get(reg + 1, 0) << 8)

    def write_block(self, reg, data):
        self.log.append(('block', reg, list(data)))
        for i, d in enumerate(data):
            self.mem[reg + i] = d

    def read_block(self, reg, n):
        return [self.mem.get(reg + i, 0) for i in range(n)]


class FakeBus:
    def __init__(self):
        self.dev = FakeDevice()
        self.addr = None

    def device(self, addr):
        self.addr = addr
        return self.dev


@pytest.fixture
def fake():
    return FakeBus()


@pytest.fixture
def mpr(fake):
    return MPR121(bus=fake, handlers=0, setup=0, reset=0)


def test_init_without_setup_or_reset_does_not_touch_device(fake):
    m = MPR121(bus=fake, addr=0x5b, handlers=0, setup=0, reset=0)
    assert m.addr == 0x5b
    assert fake.addr == 0x5b
    assert fake.dev.log == []


def test_init_reset_only_writes_soft_reset(fake):
    MPR121(bus=fake, handlers=0, setup=0, reset=1)
    assert fake.dev.log == [('byte', mpr121.SRESET, 0x63)]


def test_configure_writes_electrode_control_register(mpr, fake):
    mpr.configure(cl=3, prox=1, touch=12)
    assert fake.dev.mem[mpr121.ECR] == (3 << 6) | (1 << 4) | 12
    mpr.configure(cl=0, prox=0, touch=0)
    assert fake.dev.mem[mpr121.ECR] == 0


def test_touched_masks_status_bits(mpr, fake):
    fake.dev.mem[mpr121.ETS] = 0xff
    fake.dev.mem[mpr121.ETS + 1] = 0x1f
    assert mpr.touched() == 0x1fff


def test_touched_raises_on_overcurrent(mpr, fake):
    fake.dev.mem[mpr121.ETS + 1] = 0x80
    with pytest.raises(Exception, match='overcurrent'):
        mpr.touched()
    # no raise when disabled
    assert mpr.touched(raise_on_failure=0) == 0


def test_touched_raises_when_out_of_range_flagged(mpr, fake):
    fake.dev.mem[mpr121.OOR + 1] = 0x80
    with pytest.raises(Exception, match='auto config failed'):
        mpr.touched()


def test_out_of_range_failure_bits(mpr, fake):
    fake.dev.mem[mpr121.OOR + 1] = 0x40
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr.out_of_range()
    assert mpr.out_of_range(raise_on_failure=0) == 0
    fake.dev.mem[mpr121.OOR + 1] = 0x00
    fake.dev.mem[mpr121.OOR] = 0x05
    assert mpr.out_of_range() == 0x05


def test_update_touch_state_calls_handler_on_touch_and_release(mpr, fake):
    calls = []
    mpr.on_touch(2, lambda state, ch: calls.append((state, ch)))

    fake.dev.mem[mpr121.ETS] = 1 << 2
    mpr.update_touch_state()
    assert calls == [(True, 2)]
    assert mpr.is_touched(2)
    assert not mpr.is_touched(1)

    # no change -> no new call
    mpr.update_touch_state()
    assert calls == [(True, 2)]

    fake.dev.mem[mpr121.ETS] = 0
    mpr.update_touch_state()
    assert calls == [(True, 2), (False, 2)]
    assert not mpr.is_touched(2)


def test_handlers_only_called_for_their_channel(mpr, fake):
    a, b = [], []
    mpr.on_touch(0, lambda s, c: a.append((s, c)))
    mpr.on_touch(1, lambda s, c: b.append((s, c)))
    fake.dev.mem[mpr121.ETS] = 0b10
    mpr.update_touch_state()
    assert a == []
    assert b == [(True, 1)]


def test_removing_handler_with_none_is_safe(mpr, fake):
    calls = []
    mpr.on_touch(0, lambda s, c: calls.append(s))
    mpr.on_touch(0, None)
    fake.dev.mem[mpr121.ETS] = 1
    mpr.update_touch_state()
    assert calls == []
    assert mpr.is_touched(0)


def test_threshold_all_channels_default_release(mpr, fake):
    mpr.threshold(100)
    for ch in range(mpr121.NCH):
        assert fake.dev.mem[mpr121.TTH + 2 * ch] == 100
        assert fake.dev.mem[mpr121.RTH + 2 * ch] == 60


def test_threshold_single_channel_explicit_release(mpr, fake):
    mpr.threshold(80, release=10, channel=3)
    assert fake.dev.mem[mpr121.TTH + 6] == 80
    assert fake.dev.mem[mpr121.RTH + 6] == 10
    assert mpr121.TTH + 8 not in fake.dev.mem


def test_debounce_packs_release_and_touch(mpr, fake):
    mpr.debounce(touch=2, release=5)
    assert fake.dev.mem[mpr121.DEB] == (5 << 4) | 2
    mpr.debounce(3)
    assert fake.dev.mem[mpr121.DEB] == (3 << 4) | 3


def test_filter_writes_afe_word(mpr, fake):
    mpr.filter(cdc=16, cdt=1, ffi=0, sfi=0, esi=4)
    assert fake.dev.mem[mpr121.AFE1] == 0x10
    assert fake.dev.mem[mpr121.AFE2] == 0x24
    mpr.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    assert fake.dev.mem[mpr121.AFE1] == (1 << 6) | 30
    assert fake.dev.mem[mpr121.AFE2] == (1 << 5) | (1 << 3)


def test_electrode_data_combines_bytes(mpr, fake):
    fake.dev.mem[mpr121.EFD] = 0x34
    fake.dev.mem[mpr121.EFD + 1] = 0x02
    fake.dev.mem[mpr121.EFD + 2 * 12] = 0xff
    fake.dev.mem[mpr121.EFD + 2 * 12 + 1] = 0x03
    data = mpr.electrode_data()
    assert len(data) == mpr121.NCH
    assert data[0] == 0x234
    assert data[1] == 0
    assert data[12] == 0x3ff


def test_baseline_read_shifts_values(mpr, fake):
    fake.dev.mem[mpr121.EBL] = 0x10
    fake.dev.mem[mpr121.EBL + 5] = 0xff
    bl = mpr.baseline()
    assert len(bl) == mpr121.NCH
    assert bl[0] == 0x40
    assert bl[5] == 0x3fc
    assert bl[1] == 0


def test_baseline_write_variants(mpr, fake):
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    mpr.baseline(rft=1, mhd=6, nhd=2, ncl=4, fdl=21, prox=1)
    mpr.baseline(rft=2, mhd=9, nhd=7, ncl=8, fdl=9)
    blocks = [e for e in fake.dev.log if e[0] == 'block']
    assert blocks[0] == ('block', mpr121.MHD, [5, 1, 3, 20])
    assert blocks[1] == ('block', mpr121.MHDX + 4, [6, 2, 4, 21])
    # touched scenario has no mhd
    assert blocks[2] == ('block', mpr121.MHD + 8, [7, 8, 9])


def test_charge_sets_current_and_time_nibbles(mpr, fake):
    fake.dev.mem[mpr121.CDT + 1] = 0xff
    mpr.charge(3, cdc=20, cdt=2)
    assert fake.dev.mem[mpr121.CDC + 3] == 20
    # odd channel -> high nibble replaced, low nibble kept
    assert fake.dev.mem[mpr121.CDT + 1] == 0x2f
    mpr.charge(2, cdc=1, cdt=5)
    assert fake.dev.mem[mpr121.CDT + 1] == 0x25


def test_gpio_set_and_clear_and_bounds(mpr, fake):
    mpr.gpio_set(4, 1)
    mpr.gpio_set(11, 0)
    assert ('byte', mpr121.GPIO_SET, 1) in fake.dev.log
    assert ('byte', mpr121.GPIO_CLR, 1 << 7) in fake.dev.log
    with pytest.raises(AssertionError):
        mpr.gpio_set(3, 1)
    with pytest.raises(AssertionError):
        mpr.gpio_set(12, 1)


def test_gpio_setup_output_open_drain_high_side(mpr, fake):
    mpr.gpio_setup(5, output=1, mode=3, enable=1)
    bit = 1 << 1
    assert fake.dev.mem[mpr121.GPIO_CTL0] & bit
    assert fake.dev.mem[mpr121.GPIO_CTL1] & bit
    assert fake.dev.mem[mpr121.GPIO_DIR] & bit
    assert fake.dev.mem[mpr121.GPIO_EN] & bit
    with pytest.raises(AssertionError):
        mpr.gpio_setup(3, output=1)


def test_gpio_status_reads_data_register(mpr, fake):
    fake.dev.mem[mpr121.GPIO_DAT] = 0xa5
    assert mpr.gpio_status() == 0xa5


def test_auto_config_defaults(mpr, fake):
    mpr.auto_config()
    assert fake.dev.mem[mpr121.ACNF_C0] == 0x6f
    assert fake.dev.mem[mpr121.ACNF_C0 + 1] == 0x07
    assert fake.dev.mem[mpr121.ACNF_USL] == 200
    assert fake.dev.mem[mpr121.ACNF_LSL] == 130
    assert fake.dev.mem[mpr121.ACNF_TL] == 180


def test_setup_full_sequence(fake):
    MPR121(bus=fake, handlers=0, setup=1, reset=1, threshold=50, channels=12)
    log = fake.dev.log
    mem = fake.dev.mem
    assert log[0] == ('byte', mpr121.SRESET, 0x63)
    assert log[1] == ('byte', mpr121.ECR, 3 << 6)  # stopped during config
    assert log[-1] == ('byte', mpr121.ECR, (3 << 6) | 12)  # running
    assert mem[mpr121.TTH] == 50
    assert mem[mpr121.RTH] == 30
    # proximity channel gets threshold >> 2
    assert mem[mpr121.TTH + 24] == 12
    assert mem[mpr121.DEB] == (2 << 4) | 2
    assert mem[mpr121.ACNF_USL] == 200


def test_setup_without_auto_config_skips_auto_config_registers(fake):
    MPR121(bus=fake, handlers=0, setup=1, reset=0, auto_config=0)
    assert mpr121.ACNF_USL not in fake.dev.mem
    assert mpr121.ACNF_C0 not in fake.dev.mem
