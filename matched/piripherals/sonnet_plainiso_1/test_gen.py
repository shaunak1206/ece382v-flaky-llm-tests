import pytest

from piripherals import mpr121 as m
from piripherals.mpr121 import MPR121


class FakeDevice:
    """In-memory stand-in for a bus.Device bound to one address."""

    def __init__(self):
        self.regs = {}
        self.words = {}
        self.blocks = {}
        self.writes = []

    def read_byte(self, reg):
        return self.regs.get(reg, 0)

    def write_byte(self, reg, value):
        self.regs[reg] = value
        self.writes.append(('byte', reg, value))

    def read_word(self, reg):
        return self.words.get(reg, 0)

    def write_word(self, reg, value):
        self.writes.append(('word', reg, value))

    def read_block(self, reg, n):
        return self.blocks.get(reg, [0] * n)

    def write_block(self, reg, data):
        self.writes.append(('block', reg, list(data)))


class FakeBus:
    def __init__(self):
        self.device_dev = FakeDevice()
        self.addrs = []

    def device(self, addr):
        self.addrs.append(addr)
        return self.device_dev


def make(reset=0, setup=0, **kw):
    bus = FakeBus()
    dev = MPR121(bus=bus, handlers=0, setup=setup, reset=reset, **kw)
    return dev, bus.device_dev, bus


def test_init_binds_device_at_address():
    dev, _, bus = make(addr=0x5b)
    assert bus.addrs == [0x5b]
    assert dev.addr == 0x5b
    assert dev.overcurrent is False


def test_init_without_setup_or_reset_writes_nothing():
    _, fake, _ = make()
    assert fake.writes == []


def test_init_with_reset_only_sends_soft_reset():
    _, fake, _ = make(reset=1)
    assert fake.writes == [('byte', m.SRESET, 0x63)]


def test_is_touched_initially_false_for_all_channels():
    dev, _, _ = make()
    assert not any(dev.is_touched(i) for i in range(m.NCH))


def test_update_touch_state_sets_touched_bits():
    dev, fake, _ = make()
    fake.words[m.ETS] = 0b101
    dev.update_touch_state()
    assert dev.is_touched(0)
    assert not dev.is_touched(1)
    assert dev.is_touched(2)


def test_handler_called_on_touch_and_release():
    dev, fake, _ = make()
    calls = []
    dev.on_touch(3, lambda state, ch: calls.append((state, ch)))
    fake.words[m.ETS] = 1 << 3
    dev.update_touch_state()
    fake.words[m.ETS] = 0
    dev.update_touch_state()
    assert calls == [(True, 3), (False, 3)]


def test_handler_not_called_without_change():
    dev, fake, _ = make()
    calls = []
    dev.on_touch(1, lambda state, ch: calls.append((state, ch)))
    fake.words[m.ETS] = 1 << 1
    dev.update_touch_state()
    dev.update_touch_state()
    assert calls == [(True, 1)]


def test_handler_only_called_for_changed_channel():
    dev, fake, _ = make()
    calls = []
    dev.on_touch(0, lambda s, c: calls.append(('a', s, c)))
    dev.on_touch(1, lambda s, c: calls.append(('b', s, c)))
    fake.words[m.ETS] = 1 << 1
    dev.update_touch_state()
    assert calls == [('b', True, 1)]


def test_handler_exception_is_swallowed():
    dev, fake, _ = make()

    def bad(state, ch):
        raise ValueError('boom')

    dev.on_touch(2, bad)
    fake.words[m.ETS] = 1 << 2
    dev.update_touch_state()  # must not raise
    assert dev.is_touched(2)


def test_touched_masks_to_13_bits():
    dev, fake, _ = make()
    fake.words[m.ETS] = 0xffff & ~(1 << 15)
    assert dev.touched(raise_on_failure=0) == 0x1fff


def test_touched_raises_on_overcurrent():
    dev, fake, _ = make()
    fake.words[m.ETS] = 1 << 15
    with pytest.raises(Exception, match='overcurrent'):
        dev.touched()


def test_out_of_range_failure_bits():
    dev, fake, _ = make()
    fake.words[m.OOR] = 1 << 15
    with pytest.raises(Exception, match='auto config failed'):
        dev.out_of_range()
    fake.words[m.OOR] = 1 << 14
    with pytest.raises(Exception, match='auto reconfig failed'):
        dev.out_of_range()
    fake.words[m.OOR] = (1 << 14) | 0x7
    assert dev.out_of_range(raise_on_failure=0) == 0x7


def test_touched_raises_when_out_of_range_reports_failure():
    dev, fake, _ = make()
    fake.words[m.OOR] = 1 << 15
    with pytest.raises(Exception, match='auto config failed'):
        dev.touched()


def test_electrode_data_combines_low_and_high_bytes():
    dev, fake, _ = make()
    raw = [0] * (2 * m.NCH)
    raw[0], raw[1] = 0x34, 0x02
    raw[2], raw[3] = 0xff, 0x03
    fake.blocks[m.EFD] = raw
    data = dev.electrode_data()
    assert len(data) == m.NCH
    assert data[0] == 0x234
    assert data[1] == 0x3ff
    assert data[2] == 0


def test_baseline_read_shifts_values_left_by_two():
    dev, fake, _ = make()
    fake.blocks[m.EBL] = list(range(m.NCH))
    assert dev.baseline() == [b << 2 for b in range(m.NCH)]


def test_baseline_write_rising_and_touched_registers():
    dev, fake, _ = make()
    dev.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    dev.baseline(rft=2, mhd=9, nhd=2, ncl=4, fdl=6)
    dev.baseline(rft=1, mhd=1, nhd=1, ncl=1, fdl=1, prox=1)
    assert fake.writes == [
        ('block', m.MHD, [5, 1, 3, 20]),
        ('block', m.MHD + 8, [2, 4, 6]),
        ('block', m.MHDX + 4, [1, 1, 1, 1]),
    ]


def test_threshold_single_channel_and_default_release():
    dev, fake, _ = make()
    dev.threshold(50, channel=2)
    assert fake.regs[m.TTH + 4] == 50
    assert fake.regs[m.RTH + 4] == 30


def test_threshold_all_channels_when_channel_omitted():
    dev, fake, _ = make()
    dev.threshold(100, release=40)
    for i in range(m.NCH):
        assert fake.regs[m.TTH + 2 * i] == 100
        assert fake.regs[m.RTH + 2 * i] == 40
    assert len(fake.writes) == 2 * m.NCH


def test_debounce_defaults_release_to_touch():
    dev, fake, _ = make()
    dev.debounce(3)
    assert fake.regs[m.DEB] == 0x33
    dev.debounce(touch=1, release=5)
    assert fake.regs[m.DEB] == 0x51


def test_configure_and_filter_register_values():
    dev, fake, _ = make()
    dev.configure()
    assert fake.regs[m.ECR] == (3 << 6) | 12
    dev.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    lb = (1 << 6) | 30
    hb = (1 << 5) | (1 << 3)
    assert fake.writes[-1] == ('word', m.AFE1, (hb << 8) | lb)


def test_auto_config_defaults():
    dev, fake, _ = make()
    dev.auto_config()
    assert fake.writes[0] == ('word', m.ACNF_C0, 0x076f)
    assert fake.regs[m.ACNF_USL] == 200
    assert fake.regs[m.ACNF_LSL] == 130
    assert fake.regs[m.ACNF_TL] == 180


def test_charge_updates_only_selected_nibble():
    dev, fake, _ = make()
    fake.regs[m.CDT + 1] = 0xa5
    dev.charge(3, cdc=20, cdt=2)
    assert fake.regs[m.CDC + 3] == 20
    # odd channel => high nibble replaced, low nibble preserved
    assert fake.regs[m.CDT + 1] == (0x05 | (2 << 4))


def test_gpio_set_and_status():
    dev, fake, _ = make()
    dev.gpio_set(6, 1)
    dev.gpio_set(7, 0)
    assert fake.writes == [('byte', m.GPIO_SET, 1 << 2),
                           ('byte', m.GPIO_CLR, 1 << 3)]
    fake.regs[m.GPIO_DAT] = 0xa5
    assert dev.gpio_status() == 0xa5
    with pytest.raises(AssertionError):
        dev.gpio_set(3, 1)


def test_gpio_setup_sets_bits_and_validates_channel(capsys):
    dev, fake, _ = make()
    dev.gpio_setup(5, output=1, mode=3, enable=1)
    capsys.readouterr()
    assert fake.regs[m.GPIO_CTL0] == 0b10
    assert fake.regs[m.GPIO_CTL1] == 0b10
    assert fake.regs[m.GPIO_DIR] == 0b10
    assert fake.regs[m.GPIO_EN] == 0b10
    with pytest.raises(AssertionError):
        dev.gpio_setup(12, output=1)


def test_setup_resets_then_configures_and_starts_measurement():
    dev, fake, _ = make()
    dev.setup(reset=1, channels=8, prox=0, threshold=40)
    assert fake.writes[0] == ('byte', m.SRESET, 0x63)
    assert fake.writes[1] == ('byte', m.ECR, 3 << 6)
    assert fake.writes[-1] == ('byte', m.ECR, (3 << 6) | 8)
    assert fake.regs[m.TTH] == 40
    assert fake.regs[m.TTH + 24] == 10
