"""Unit tests for piripherals.mpr121.MPR121 using a fake in-memory bus."""

import pytest

from piripherals import mpr121
from piripherals.mpr121 import MPR121


class FakeDevice:
    """Minimal register-file device recording every write."""

    def __init__(self):
        self.regs = {}
        self.writes = []
        self.read_values = {}  # (kind, reg) -> value overrides
        self.block_data = {}

    def write_byte(self, reg, value):
        self.regs[reg] = value
        self.writes.append(('byte', reg, value))

    def read_byte(self, reg):
        return self.regs.get(reg, 0)

    def write_word(self, reg, value):
        self.regs[reg] = value
        self.writes.append(('word', reg, value))

    def read_word(self, reg):
        return self.read_values.get(('word', reg), self.regs.get(reg, 0))

    def write_block(self, reg, values):
        self.writes.append(('block', reg, list(values)))

    def read_block(self, reg, n):
        return self.block_data.get(reg, [0] * n)


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


def test_init_uses_device_address_and_does_not_touch_hardware(fake):
    m = MPR121(bus=fake, addr=0x5b, handlers=0, setup=0, reset=0)
    assert fake.addr == 0x5b
    assert m.addr == 0x5b
    assert m.overcurrent is False
    assert fake.dev.writes == []


def test_init_selects_poller_without_irq(fake, monkeypatch):
    calls = []
    monkeypatch.setattr(mpr121, 'Poller', lambda cb: calls.append(('poll', cb)))
    monkeypatch.setattr(mpr121, 'IRQHandler',
                        lambda pin, cb: calls.append(('irq', pin, cb)))
    m = MPR121(bus=fake, irq=0, handlers=1, setup=0, reset=0)
    assert len(calls) == 1
    assert calls[0][0] == 'poll'
    assert calls[0][1] == m.update_touch_state


def test_init_selects_irq_handler_with_irq_pin(fake, monkeypatch):
    calls = []
    monkeypatch.setattr(mpr121, 'Poller', lambda cb: calls.append(('poll', cb)))
    monkeypatch.setattr(mpr121, 'IRQHandler',
                        lambda pin, cb: calls.append(('irq', pin, cb)))
    m = MPR121(bus=fake, irq=4, handlers=1, setup=0, reset=0)
    assert calls == [('irq', 4, m.update_touch_state)]


def test_reset_writes_soft_reset_register(mpr, fake):
    mpr.reset()
    assert fake.dev.writes[-1] == ('byte', mpr121.SRESET, 0x63)


def test_configure_encodes_ecr(mpr, fake):
    mpr.configure(cl=2, prox=1, touch=5)
    assert fake.dev.regs[mpr121.ECR] == (2 << 6) | (1 << 4) | 5
    mpr.configure()
    assert fake.dev.regs[mpr121.ECR] == (3 << 6) | 12


def test_touched_masks_status_to_13_bits(mpr, fake):
    fake.dev.read_values[('word', mpr121.ETS)] = 0x0abc
    assert mpr.touched() == 0x0abc & 0x1fff
    fake.dev.read_values[('word', mpr121.ETS)] = 0x1fff
    assert mpr.touched() == 0x1fff


def test_touched_raises_on_overcurrent(mpr, fake):
    fake.dev.read_values[('word', mpr121.ETS)] = 1 << 15
    with pytest.raises(Exception, match='overcurrent'):
        mpr.touched()
    # failure bits are ignored when raising is disabled
    assert mpr.touched(raise_on_failure=0) == 0


def test_out_of_range_failure_bits(mpr, fake):
    fake.dev.read_values[('word', mpr121.OOR)] = (1 << 14) | 0x3
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr.out_of_range()
    assert mpr.out_of_range(raise_on_failure=0) == 0x3


def test_update_touch_state_invokes_handler_with_state_and_channel(mpr, fake):
    seen = []
    mpr.on_touch(3, lambda state, ch: seen.append((state, ch)))
    fake.dev.read_values[('word', mpr121.ETS)] = 1 << 3
    mpr.update_touch_state()
    assert seen == [(True, 3)]
    assert mpr.is_touched(3)
    assert not mpr.is_touched(2)

    fake.dev.read_values[('word', mpr121.ETS)] = 0
    mpr.update_touch_state()
    assert seen == [(True, 3), (False, 3)]
    assert not mpr.is_touched(3)


def test_handler_not_called_when_state_unchanged(mpr, fake):
    seen = []
    mpr.on_touch(0, lambda state, ch: seen.append(state))
    fake.dev.read_values[('word', mpr121.ETS)] = 1
    mpr.update_touch_state()
    mpr.update_touch_state()
    mpr.update_touch_state()
    assert seen == [True]


def test_only_changed_channels_are_dispatched(mpr, fake):
    seen = []
    for ch in (0, 1, 2):
        mpr.on_touch(ch, lambda state, c: seen.append((c, state)))
    fake.dev.read_values[('word', mpr121.ETS)] = 0b011
    mpr.update_touch_state()
    assert sorted(seen) == [(0, True), (1, True)]
    seen.clear()
    fake.dev.read_values[('word', mpr121.ETS)] = 0b110
    mpr.update_touch_state()
    assert sorted(seen) == [(0, False), (2, True)]


def test_raising_or_removed_handlers_do_not_propagate(mpr, fake):
    def boom(*a):
        raise RuntimeError('handler failure')

    mpr.on_touch(1, boom)
    mpr.on_touch(2, None)
    fake.dev.read_values[('word', mpr121.ETS)] = 0b110
    mpr.update_touch_state()  # must not raise
    assert mpr.is_touched(1) and mpr.is_touched(2)


def test_threshold_defaults_release_to_sixty_percent(mpr, fake):
    mpr.threshold(50, channel=2)
    assert fake.dev.regs[mpr121.TTH + 4] == 50
    assert fake.dev.regs[mpr121.RTH + 4] == 30


def test_threshold_applies_to_all_channels_by_default(mpr, fake):
    mpr.threshold(40, release=10)
    for ch in range(mpr121.NCH):
        assert fake.dev.regs[mpr121.TTH + 2 * ch] == 40
        assert fake.dev.regs[mpr121.RTH + 2 * ch] == 10


def test_debounce_packs_release_and_touch(mpr, fake):
    mpr.debounce(touch=3, release=5)
    assert fake.dev.regs[mpr121.DEB] == (5 << 4) | 3
    mpr.debounce(touch=2)
    assert fake.dev.regs[mpr121.DEB] == (2 << 4) | 2


def test_filter_packs_afe_word(mpr, fake):
    mpr.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    lb = (1 << 6) | 30
    hb = (1 << 5) | (1 << 3) | 0
    assert fake.dev.regs[mpr121.AFE1] == (hb << 8) | lb


def test_charge_updates_only_the_right_nibble(mpr, fake):
    fake.dev.regs[mpr121.CDT + 1] = 0xAB
    mpr.charge(3, cdc=10, cdt=5)
    assert fake.dev.regs[mpr121.CDC + 3] == 10
    assert fake.dev.regs[mpr121.CDT + 1] == 0x5B  # odd channel -> high nibble


def test_electrode_data_combines_little_endian_bytes(mpr, fake):
    raw = [0] * (2 * mpr121.NCH)
    raw[0], raw[1] = 0x34, 0x02
    raw[24], raw[25] = 0xff, 0x03
    fake.dev.block_data[mpr121.EFD] = raw
    data = mpr.electrode_data()
    assert len(data) == mpr121.NCH
    assert data[0] == 0x234
    assert data[12] == 0x3ff
    assert data[1] == 0


def test_baseline_read_shifts_to_ten_bit(mpr, fake):
    fake.dev.block_data[mpr121.EBL] = list(range(mpr121.NCH))
    assert mpr.baseline() == [b << 2 for b in range(mpr121.NCH)]


def test_baseline_write_selects_register_and_payload(mpr, fake):
    mpr.baseline(rft=1, mhd=5, nhd=1, ncl=3, fdl=20)
    mpr.baseline(rft=2, mhd=9, nhd=2, ncl=4, fdl=6)
    mpr.baseline(rft=0, mhd=1, nhd=1, ncl=2, fdl=3, prox=1)
    assert ('block', mpr121.MHD + 4, [5, 1, 3, 20]) in fake.dev.writes
    # touched scenario has no mhd
    assert ('block', mpr121.MHD + 8, [2, 4, 6]) in fake.dev.writes
    assert ('block', mpr121.MHDX, [1, 1, 2, 3]) in fake.dev.writes


def test_auto_config_defaults(mpr, fake):
    mpr.auto_config()
    lb = (1 << 6) | (2 << 4) | (3 << 2) | (1 << 1) | 1
    hb = (0 << 7) | (1 << 2) | (1 << 1) | 1
    assert fake.dev.regs[mpr121.ACNF_C0] == (hb << 8) + lb
    assert fake.dev.regs[mpr121.ACNF_USL] == 200
    assert fake.dev.regs[mpr121.ACNF_LSL] == 130
    assert fake.dev.regs[mpr121.ACNF_TL] == 180


def test_gpio_set_writes_set_or_clear_register(mpr, fake):
    mpr.gpio_set(6, 1)
    assert fake.dev.writes[-1] == ('byte', mpr121.GPIO_SET, 1 << 2)
    mpr.gpio_set(11, 0)
    assert fake.dev.writes[-1] == ('byte', mpr121.GPIO_CLR, 1 << 7)


def test_gpio_channel_range_is_asserted(mpr):
    with pytest.raises(AssertionError):
        mpr.gpio_set(3, 1)
    with pytest.raises(AssertionError):
        mpr.gpio_set(12, 1)
    with pytest.raises(AssertionError):
        mpr.gpio_setup(2, 1)


def test_gpio_setup_sets_expected_register_bits(mpr, fake, capsys):
    mpr.gpio_setup(5, output=1, mode=2, enable=1)
    capsys.readouterr()
    bit = 1 << 1
    assert fake.dev.regs[mpr121.GPIO_CTL0] & bit
    assert not fake.dev.regs[mpr121.GPIO_CTL1] & bit
    assert fake.dev.regs[mpr121.GPIO_DIR] & bit
    assert fake.dev.regs[mpr121.GPIO_EN] & bit


def test_setup_programs_defaults_and_starts_measurement(mpr, fake):
    mpr.setup(reset=1, channels=8, threshold=40)
    writes = fake.dev.writes
    assert writes[0] == ('byte', mpr121.SRESET, 0x63)
    assert fake.dev.regs[mpr121.TTH] == 40
    assert fake.dev.regs[mpr121.RTH] == 24
    assert fake.dev.regs[mpr121.TTH + 24] == 10  # proximity channel
    assert fake.dev.regs[mpr121.DEB] == (2 << 4) | 2
    assert fake.dev.regs[mpr121.ECR] == (3 << 6) | 8
    assert writes[-1] == ('byte', mpr121.ECR, (3 << 6) | 8)
