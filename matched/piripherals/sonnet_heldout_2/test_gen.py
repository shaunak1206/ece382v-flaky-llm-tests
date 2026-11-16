import pytest

from piripherals import mpr121 as m
from piripherals.mpr121 import MPR121


class FakeDevice:
    """In-memory register file emulating the bus device interface."""

    def __init__(self):
        self.regs = {}
        self.writes = []
        self.block_writes = []

    def read_byte(self, reg):
        return self.regs.get(reg, 0)

    def write_byte(self, reg, value):
        self.writes.append((reg, value))
        self.regs[reg] = value

    def read_word(self, reg):
        return self.regs.get(reg, 0) | (self.regs.get(reg + 1, 0) << 8)

    def write_word(self, reg, value):
        self.writes.append((reg, value))
        self.regs[reg] = value & 0xff
        self.regs[reg + 1] = (value >> 8) & 0xff

    def read_block(self, reg, n):
        return [self.regs.get(reg + i, 0) for i in range(n)]

    def write_block(self, reg, data):
        self.block_writes.append((reg, list(data)))
        for i, v in enumerate(data):
            self.regs[reg + i] = v


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
    dev = MPR121(bus=bus, **kwargs)
    return dev, bus.dev, bus


def test_init_uses_given_address():
    dev, fake, bus = make(addr=0x5b)
    assert bus.addr == 0x5b
    assert dev.addr == 0x5b


def test_init_without_reset_or_setup_writes_nothing():
    dev, fake, _ = make()
    assert fake.writes == []
    assert fake.block_writes == []


def test_init_reset_only_writes_soft_reset():
    dev, fake, _ = make(reset=1)
    assert fake.writes == [(m.SRESET, 0x63)]


def test_configure_writes_ecr():
    dev, fake, _ = make()
    dev.configure(cl=3, prox=1, touch=12)
    assert fake.regs[m.ECR] == (3 << 6) | (1 << 4) | 12
    dev.configure(prox=0, touch=0)
    assert fake.regs[m.ECR] == 3 << 6


def test_touched_masks_status_bits():
    dev, fake, _ = make()
    fake.regs[m.ETS] = 0x05
    fake.regs[m.ETS + 1] = 0x00
    assert dev.touched() == 0x05
    fake.regs[m.ETS + 1] = 0x10  # bit 12 (proximity)
    assert dev.touched() == 0x1005


def test_touched_raises_on_overcurrent():
    dev, fake, _ = make()
    fake.regs[m.ETS + 1] = 0x80
    with pytest.raises(Exception, match='overcurrent'):
        dev.touched()
    # no raise when disabled, flag bits masked out
    assert dev.touched(raise_on_failure=0) == 0


def test_out_of_range_failures():
    dev, fake, _ = make()
    fake.regs[m.OOR + 1] = 0x80
    with pytest.raises(Exception, match='auto config failed'):
        dev.out_of_range()
    fake.regs[m.OOR + 1] = 0x40
    with pytest.raises(Exception, match='auto reconfig failed'):
        dev.out_of_range()
    assert dev.out_of_range(raise_on_failure=0) == 0


def test_touched_raises_when_auto_config_failed():
    dev, fake, _ = make()
    fake.regs[m.OOR + 1] = 0x80
    with pytest.raises(Exception):
        dev.touched()


def test_out_of_range_returns_status_bits():
    dev, fake, _ = make()
    fake.regs[m.OOR] = 0x0a
    assert dev.out_of_range() == 0x0a


def test_update_touch_state_invokes_handlers_on_change():
    dev, fake, _ = make()
    calls = []
    dev.on_touch(0, lambda s, c: calls.append((s, c)))
    dev.on_touch(2, lambda s, c: calls.append((s, c)))

    fake.regs[m.ETS] = 0b101
    dev.update_touch_state()
    assert sorted(calls) == [(True, 0), (True, 2)]
    assert dev.is_touched(0)
    assert not dev.is_touched(1)
    assert dev.is_touched(2)

    # no change -> no new calls
    del calls[:]
    dev.update_touch_state()
    assert calls == []

    # release channel 0
    fake.regs[m.ETS] = 0b100
    dev.update_touch_state()
    assert calls == [(False, 0)]
    assert not dev.is_touched(0)


def test_on_touch_none_removes_handler():
    dev, fake, _ = make()
    calls = []
    dev.on_touch(1, lambda s, c: calls.append((s, c)))
    dev.on_touch(1, None)
    fake.regs[m.ETS] = 0b10
    dev.update_touch_state()
    assert calls == []


def test_threshold_single_channel_default_release():
    dev, fake, _ = make()
    dev.threshold(50, channel=3)
    assert fake.regs[m.TTH + 6] == 50
    assert fake.regs[m.RTH + 6] == 30  # int(0.6 * 50)


def test_threshold_all_channels():
    dev, fake, _ = make()
    dev.threshold(40, release=10)
    for i in range(m.NCH):
        assert fake.regs[m.TTH + 2 * i] == 40
        assert fake.regs[m.RTH + 2 * i] == 10


def test_debounce():
    dev, fake, _ = make()
    dev.debounce(3)
    assert fake.regs[m.DEB] == (3 << 4) | 3
    dev.debounce(2, 5)
    assert fake.regs[m.DEB] == (5 << 4) | 2


def test_filter_word_layout():
    dev, fake, _ = make()
    dev.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    assert fake.regs[m.AFE1] == (1 << 6) | 30
    assert fake.regs[m.AFE2] == (1 << 5) | (1 << 3) | 0


def test_charge_sets_current_and_time_nibbles():
    dev, fake, _ = make()
    dev.charge(2, cdc=10, cdt=3)
    assert fake.regs[m.CDC + 2] == 10
    assert fake.regs[m.CDT + 1] == 3
    dev.charge(3, cdc=11, cdt=5)
    assert fake.regs[m.CDC + 3] == 11
    assert fake.regs[m.CDT + 1] == (5 << 4) | 3


def test_baseline_read_and_write():
    dev, fake, _ = make()
    for i in range(m.NCH):
        fake.regs[m.EBL + i] = i + 1
    assert dev.baseline() == [(i + 1) << 2 for i in range(m.NCH)]

    dev.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    dev.baseline(rft=2, mhd=9, nhd=2, ncl=4, fdl=6)
    dev.baseline(rft=1, mhd=1, nhd=1, ncl=3, fdl=20, prox=1)
    assert fake.block_writes == [
        (m.MHD, [5, 1, 3, 20]),
        (m.MHD + 8, [2, 4, 6]),
        (m.MHDX + 4, [1, 1, 3, 20]),
    ]


def test_electrode_data_combines_bytes():
    dev, fake, _ = make()
    fake.regs[m.EFD] = 0x34
    fake.regs[m.EFD + 1] = 0x02
    fake.regs[m.EFD + 24] = 0xff
    fake.regs[m.EFD + 25] = 0x03
    data = dev.electrode_data()
    assert len(data) == m.NCH
    assert data[0] == 0x234
    assert data[1] == 0
    assert data[12] == 0x3ff


def test_gpio_set_and_clear():
    dev, fake, _ = make()
    dev.gpio_set(5, 1)
    dev.gpio_set(7, 0)
    assert fake.writes == [(m.GPIO_SET, 1 << 1), (m.GPIO_CLR, 1 << 3)]
    with pytest.raises(AssertionError):
        dev.gpio_set(3, 1)
    with pytest.raises(AssertionError):
        dev.gpio_set(12, 1)


def test_gpio_setup_output_open_drain_and_status():
    dev, fake, _ = make()
    dev.gpio_setup(5, output=1, mode=3, enable=1)
    bit = 1 << 1
    assert fake.regs[m.GPIO_CTL0] & bit
    assert fake.regs[m.GPIO_CTL1] & bit
    assert fake.regs[m.GPIO_DIR] & bit
    assert fake.regs[m.GPIO_EN] & bit
    fake.regs[m.GPIO_DAT] = 0x42
    assert dev.gpio_status() == 0x42
    with pytest.raises(AssertionError):
        dev.gpio_setup(2, output=0)


def test_auto_config_registers():
    dev, fake, _ = make()
    dev.auto_config()
    assert fake.regs[m.ACNF_C0] == (1 << 6) | (2 << 4) | (3 << 2) | (1 << 1) | 1
    assert fake.regs[m.ACNF_C1] == 0b111
    assert fake.regs[m.ACNF_USL] == 200
    assert fake.regs[m.ACNF_LSL] == 130
    assert fake.regs[m.ACNF_TL] == 180


def test_setup_configures_device_and_starts_measurement():
    dev, fake, _ = make(setup=1, reset=1, threshold=60, channels=10)
    assert fake.writes[0] == (m.SRESET, 0x63)
    assert fake.regs[m.ECR] == 10
    assert fake.regs[m.TTH] == 60
    assert fake.regs[m.TTH + 24] == 60 >> 2
    assert fake.regs[m.DEB] == (2 << 4) | 2
    assert fake.regs[m.ACNF_USL] == 200
    assert (m.MHD, [5, 1, 3, 20]) in fake.block_writes
    assert (m.MHDX, [1, 1, 3, 20]) in fake.block_writes
