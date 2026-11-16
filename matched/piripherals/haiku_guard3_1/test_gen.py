import pytest

from piripherals.mpr121 import (
    MPR121,
    NCH,
    ETS,
    OOR,
    EFD,
    EBL,
    MHD,
    MHDX,
    TTH,
    RTH,
    DEB,
    AFE1,
    ECR,
    CDC,
    CDT,
    SRESET,
    GPIO_SET,
    GPIO_CLR,
    GPIO_DAT,
)


class FakeDevice:
    """In-memory stand-in for one I2C device. Records every write."""

    def __init__(self):
        self.mem = {}
        self.log = []

    def write_byte(self, reg, value):
        self.log.append(('byte', reg, value))
        self.mem[reg] = value & 0xff

    def read_byte(self, reg):
        return self.mem.get(reg, 0)

    def write_word(self, reg, value):
        # SMBus word writes are little-endian: low byte first.
        self.log.append(('word', reg, value))
        self.mem[reg] = value & 0xff
        self.mem[reg + 1] = (value >> 8) & 0xff

    def read_word(self, reg):
        return self.mem.get(reg, 0) | (self.mem.get(reg + 1, 0) << 8)

    def read_block(self, reg, n):
        return [self.mem.get(reg + i, 0) for i in range(n)]

    def write_block(self, reg, data):
        self.log.append(('block', reg, list(data)))
        for i, value in enumerate(data):
            self.mem[reg + i] = value & 0xff


class FakeBus:
    """Stand-in for piripherals.bus.Bus; hands out one FakeDevice."""

    def __init__(self):
        self.addrs = []
        self.device_obj = FakeDevice()

    def device(self, addr):
        self.addrs.append(addr)
        return self.device_obj


def make(setup=0, reset=0, handlers=0, **kwargs):
    """Build an MPR121 on a fake bus with no IRQ/poller and no setup."""
    bus = FakeBus()
    m = MPR121(bus=bus, handlers=handlers, setup=setup, reset=reset, **kwargs)
    return m, bus.device_obj, bus


def test_init_without_setup_or_reset_writes_nothing():
    m, dev, bus = make()
    assert dev.log == []
    assert bus.addrs == [0x5a]
    assert m.addr == 0x5a
    assert m.overcurrent is False
    assert m.is_touched(0) is False


def test_init_with_reset_only_sends_soft_reset():
    m, dev, _ = make(reset=1)
    assert dev.log == [('byte', SRESET, 0x63)]
    m.reset()
    assert dev.log[-1] == ('byte', SRESET, 0x63)


def test_configure_packs_mode_bits_and_stop_mode():
    m, dev, _ = make()
    m.configure(cl=3, prox=0, touch=12)
    assert dev.log[-1] == ('byte', ECR, 0xcc)
    m.configure(prox=0, touch=0)
    assert dev.log[-1] == ('byte', ECR, 0x00)
    m.configure(cl=1, prox=3, touch=0)
    assert dev.log[-1] == ('byte', ECR, 0x70)


def test_setup_resets_and_ends_in_run_mode():
    m, dev, _ = make(setup=1, reset=1)
    assert dev.log[0] == ('byte', SRESET, 0x63)
    # threshold=50 gives channel 12 a touch threshold of 50 >> 2 = 12
    assert ('byte', TTH + 2 * 12, 12) in dev.log
    assert dev.log[-1] == ('byte', ECR, 0xcc)


def test_touched_masks_status_word_to_12_bits():
    m, dev, _ = make()
    dev.mem[ETS] = 0x05
    dev.mem[ETS + 1] = 0x4a  # word 0x4a05: bit 14 set, not overcurrent
    assert m.touched() == 0x0a05


def test_touched_raises_on_overcurrent():
    m, dev, _ = make()
    dev.mem[ETS + 1] = 0x80  # word 0x8000
    with pytest.raises(Exception, match='overcurrent'):
        m.touched()


def test_touched_raises_when_a_channel_is_out_of_range():
    m, dev, _ = make()
    dev.mem[OOR] = 0x08  # out-of-range bit for channel 3
    with pytest.raises(Exception, match='out of range'):
        m.touched()


def test_touched_without_raise_ignores_failure_bits():
    m, dev, _ = make()
    dev.mem[ETS] = 0x01
    dev.mem[ETS + 1] = 0x80  # overcurrent bit set
    dev.mem[OOR] = 0x08      # out-of-range bit set
    assert m.touched(raise_on_failure=0) == 0x0001


def test_out_of_range_raises_auto_config_failed():
    m, dev, _ = make()
    dev.mem[OOR + 1] = 0x80  # word 0x8000
    with pytest.raises(Exception, match='auto config failed'):
        m.out_of_range()


def test_out_of_range_raises_auto_reconfig_failed():
    m, dev, _ = make()
    dev.mem[OOR + 1] = 0x40  # word 0x4000
    with pytest.raises(Exception, match='auto reconfig failed'):
        m.out_of_range()


def test_update_touch_state_calls_handlers_in_channel_order_and_on_release():
    m, dev, _ = make()
    calls = []
    m.on_touch(3, lambda s, c: calls.append((c, s)))
    m.on_touch(0, lambda s, c: calls.append((c, s)))

    dev.mem[ETS] = 0x09  # channels 0 and 3 touched
    m.update_touch_state()
    assert calls == [(0, True), (3, True)]

    dev.mem[ETS] = 0x01  # channel 3 released
    m.update_touch_state()
    assert calls[2:] == [(3, False)]


def test_update_touch_state_is_silent_when_nothing_changed():
    m, dev, _ = make()
    calls = []
    m.on_touch(2, lambda s, c: calls.append((c, s)))
    dev.mem[ETS] = 0x04
    m.update_touch_state()
    m.update_touch_state()
    assert calls == [(2, True)]


def test_on_touch_with_none_removes_handler():
    m, dev, _ = make()
    calls = []
    m.on_touch(0, lambda s, c: calls.append((c, s)))
    m.on_touch(0, None)
    dev.mem[ETS] = 0x01
    m.update_touch_state()
    assert calls == []


def test_handler_taking_only_the_state_argument_still_works():
    m, dev, _ = make()
    seen = []

    def handler(state):
        seen.append(state)

    m.on_touch(5, handler)
    dev.mem[ETS] = 1 << 5
    m.update_touch_state()
    assert seen == [True]


def test_is_touched_reflects_latest_update():
    m, dev, _ = make()
    dev.mem[ETS] = 0x04
    dev.mem[ETS + 1] = 0x04  # word 0x0400: channel 10 touched
    m.update_touch_state()
    assert m.is_touched(10) is True
    assert m.is_touched(0) is False


def test_electrode_data_decodes_little_endian_10_bit_values():
    m, dev, _ = make()
    dev.mem[EFD + 0] = 0x34
    dev.mem[EFD + 1] = 0x02        # channel 0 -> 0x234
    dev.mem[EFD + 22] = 0xff
    dev.mem[EFD + 23] = 0x03       # channel 11 -> 0x3ff
    dev.mem[EFD + 24] = 0x01       # channel 12 -> 1
    data = m.electrode_data()
    assert len(data) == NCH
    assert data[0] == 0x234
    assert data[1] == 0
    assert data[11] == 0x3ff
    assert data[12] == 1


def test_baseline_read_shifts_values_left_by_two():
    m, dev, _ = make()
    for i in range(NCH):
        dev.mem[EBL + i] = i
    assert m.baseline() == [i << 2 for i in range(NCH)]


def test_baseline_write_selects_register_per_scenario():
    m, dev, _ = make()
    m.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert dev.log[-1] == ('block', MHD, [5, 1, 3, 20])
    m.baseline(rft=1, mhd=6, nhd=2, ncl=4, fdl=21)
    assert dev.log[-1] == ('block', MHD + 4, [6, 2, 4, 21])
    m.baseline(rft=2, nhd=1, ncl=3, fdl=20)
    assert dev.log[-1] == ('block', MHD + 8, [1, 3, 20])
    m.baseline(rft=1, mhd=1, nhd=1, ncl=3, fdl=20, prox=1)
    assert dev.log[-1] == ('block', MHDX + 4, [1, 1, 3, 20])


def test_threshold_single_channel_writes_touch_and_release():
    m, dev, _ = make()
    m.threshold(touch=50, release=30, channel=3)
    assert dev.log == [('byte', TTH + 6, 50), ('byte', RTH + 6, 30)]

    dev.log.clear()
    m.threshold(touch=50, channel=3)  # release defaults to int(0.6 * touch)
    assert ('byte', RTH + 6, int(0.6 * 50)) in dev.log


def test_threshold_all_channels_writes_every_channel():
    m, dev, _ = make()
    m.threshold(touch=10, release=5)
    expected = []
    for i in range(NCH):
        expected.append(('byte', TTH + 2 * i, 10))
        expected.append(('byte', RTH + 2 * i, 5))
    assert dev.log == expected


def test_debounce_packs_touch_and_release_nibbles():
    m, dev, _ = make()
    m.debounce(touch=2, release=5)
    assert dev.log[-1] == ('byte', DEB, 0x52)
    m.debounce(touch=3)  # release defaults to touch
    assert dev.log[-1] == ('byte', DEB, 0x33)


def test_filter_packs_afe1_word():
    m, dev, _ = make()
    m.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    # low byte: ffi<<6 | cdc = 0x5e; high byte: cdt<<5 | sfi<<3 | esi = 0x28
    assert dev.log[-1] == ('word', AFE1, 0x285e)


def test_charge_updates_only_the_channel_nibble():
    m, dev, _ = make()
    dev.mem[CDT] = 0xab
    m.charge(channel=1, cdc=7, cdt=5)  # odd channel -> high nibble
    assert ('byte', CDC + 1, 7) in dev.log
    assert dev.mem[CDT] == 0x5b

    m.charge(channel=0, cdc=0, cdt=3)  # even channel -> low nibble
    assert dev.mem[CDT] == 0x53


def test_gpio_set_writes_set_or_clear_register_and_status_reads_data():
    m, dev, _ = make()
    m.gpio_set(5, True)
    assert dev.log[-1] == ('byte', GPIO_SET, 1 << 1)
    m.gpio_set(5, False)
    assert dev.log[-1] == ('byte', GPIO_CLR, 1 << 1)
    with pytest.raises(AssertionError):
        m.gpio_set(3, True)

    dev.mem[GPIO_DAT] = 0x5a
    assert m.gpio_status() == 0x5a
