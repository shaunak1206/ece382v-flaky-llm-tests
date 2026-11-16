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
    GPIO_DAT,
    GPIO_SET,
    GPIO_CLR,
    ACNF_C0,
    ACNF_USL,
    ACNF_LSL,
    ACNF_TL,
    SRESET,
)


class FakeDevice:
    """Records bus writes and serves preset read values."""

    def __init__(self):
        self.writes = []
        self.bytes = {}
        self.blocks = {}
        self.ets_word = 0
        self.oor_word = 0

    def write_byte(self, reg, value):
        self.writes.append(('write_byte', reg, value))

    def read_byte(self, reg):
        return self.bytes.get(reg, 0)

    def write_word(self, reg, value):
        self.writes.append(('write_word', reg, value))

    def read_word(self, reg):
        if reg == ETS:
            return self.ets_word
        if reg == OOR:
            return self.oor_word
        return 0

    def read_block(self, reg, length):
        return list(self.blocks.get(reg, [0] * length))

    def write_block(self, reg, data):
        self.writes.append(('write_block', reg, list(data)))


class FakeBus:
    """Stands in for piripherals.bus.Bus; has a 'device' attribute so Bus is not used."""

    def __init__(self, device):
        self._device = device
        self.addr = None

    def device(self, addr):
        self.addr = addr
        return self._device


def make(setup=0, reset=0, **kwargs):
    dev = FakeDevice()
    m = MPR121(bus=FakeBus(dev), addr=0x5a, irq=0, handlers=0,
               setup=setup, reset=reset, **kwargs)
    return m, dev


def writes_to(dev, reg, kind='write_byte'):
    return [w[2] for w in dev.writes if w[0] == kind and w[1] == reg]


def test_init_without_setup_or_reset_writes_nothing():
    m, dev = make(setup=0, reset=0)
    assert dev.writes == []
    assert m.addr == 0x5a
    assert m.overcurrent is False


def test_init_with_reset_only_soft_resets():
    m, dev = make(setup=0, reset=1)
    assert writes_to(dev, SRESET) == [0x63]
    assert len(dev.writes) == 1


def test_init_with_setup_configures_run_mode_with_12_channels():
    m, dev = make(setup=1, reset=0)
    ecr_values = writes_to(dev, ECR)
    assert ecr_values[0] == 0          # stop mode first
    assert ecr_values[-1] == 12        # run mode, electrodes 0-11
    assert writes_to(dev, AFE1, 'write_word') == [0x285e]


def test_bus_device_is_requested_with_address():
    dev = FakeDevice()
    bus = FakeBus(dev)
    MPR121(bus=bus, addr=0x5b, handlers=0, setup=0, reset=0)
    assert bus.addr == 0x5b


def test_reset_writes_soft_reset_register():
    m, dev = make()
    m.reset()
    assert dev.writes == [('write_byte', SRESET, 0x63)]


def test_configure_packs_cl_prox_and_touch_bits():
    m, dev = make()
    m.configure(cl=3, prox=2, touch=5)
    assert writes_to(dev, ECR) == [(3 << 6) | (2 << 4) | 5]


def test_is_touched_reflects_internal_state():
    m, _ = make()
    m._touched = 0b101
    assert m.is_touched(0) is True
    assert m.is_touched(1) is False
    assert m.is_touched(2) is True


def test_touched_masks_to_13_bits_without_raising():
    m, dev = make()
    dev.ets_word = 0xffff
    assert m.touched(raise_on_failure=0) == 0x1fff


def test_touched_raises_on_overcurrent():
    m, dev = make()
    dev.ets_word = 1 << 15
    with pytest.raises(Exception, match='overcurrent'):
        m.touched()


def test_touched_raises_when_out_of_range_auto_config_failed():
    m, dev = make()
    dev.oor_word = 1 << 15
    with pytest.raises(Exception, match='auto config failed'):
        m.touched()


def test_out_of_range_raises_on_auto_reconfig_failure():
    m, dev = make()
    dev.oor_word = 1 << 14
    with pytest.raises(Exception, match='auto reconfig failed'):
        m.out_of_range()


def test_out_of_range_returns_masked_status_when_not_raising():
    m, dev = make()
    dev.oor_word = (1 << 15) | 0x0a0a
    assert m.out_of_range(raise_on_failure=0) == 0x0a0a


def test_update_touch_state_invokes_handlers_on_change():
    m, dev = make()
    calls = []
    m.on_touch(0, lambda state, ch: calls.append((ch, state)))
    m.on_touch(1, lambda state, ch: calls.append((ch, state)))
    dev.ets_word = 0b01
    m.update_touch_state()
    assert calls == [(0, True)]
    assert m.is_touched(0) is True


def test_update_touch_state_does_not_fire_when_unchanged():
    m, dev = make()
    calls = []
    m.on_touch(0, lambda state, ch: calls.append(state))
    dev.ets_word = 0b1
    m.update_touch_state()
    m.update_touch_state()
    assert calls == [True]


def test_handle_touch_reports_release_on_change():
    m, _ = make()
    calls = []
    m.on_touch(2, lambda state, ch: calls.append((ch, state)))
    m._touched = 0b100
    m._handle_touch(0, 0b100)
    assert calls == [(2, False)]


def test_handle_touch_falls_back_to_single_argument_handler():
    m, _ = make()
    calls = []
    m._handlers[3] = lambda state: calls.append(state)
    m._handle_touch(1 << 3, 1 << 3)
    assert calls == [True]


def test_on_touch_none_removes_handler_without_error():
    m, dev = make()
    m.on_touch(4, None)
    m._handle_touch(1 << 4, 1 << 4)  # must not raise


def test_electrode_data_combines_little_endian_pairs():
    m, dev = make()
    raw = [0] * (2 * NCH)
    raw[0], raw[1] = 0x34, 0x02        # channel 0 -> 0x234
    raw[2 * 12], raw[2 * 12 + 1] = 0xff, 0x03  # channel 12 -> 0x3ff
    dev.blocks[EFD] = raw
    data = m.electrode_data()
    assert len(data) == NCH
    assert data[0] == 0x234
    assert data[12] == 0x3ff


def test_baseline_read_returns_shifted_values():
    m, dev = make()
    dev.blocks[EBL] = [1] * NCH
    assert m.baseline() == [4] * NCH


def test_baseline_write_for_rising_scenario_uses_mhd_block():
    m, dev = make()
    m.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert ('write_block', MHD, [5, 1, 3, 20]) in dev.writes


def test_baseline_proximity_touched_scenario_uses_three_byte_block():
    m, dev = make()
    m.baseline(rft=2, nhd=1, ncl=3, fdl=20, prox=1)
    assert ('write_block', MHDX + 8, [1, 3, 20]) in dev.writes


def test_threshold_applies_to_all_channels_with_default_release():
    m, dev = make()
    m.threshold(touch=100)
    assert writes_to(dev, TTH + 0) == [100]
    assert len(writes_to(dev, TTH + 2 * 0)) == 1
    touch_writes = [w for w in dev.writes if w[0] == 'write_byte' and w[1] >= TTH and w[1] <= TTH + 2 * (NCH - 1) and (w[1] - TTH) % 2 == 0]
    release_writes = [w for w in dev.writes if w[0] == 'write_byte' and w[1] >= RTH and w[1] <= RTH + 2 * (NCH - 1) and (w[1] - RTH) % 2 == 0]
    assert len(touch_writes) == NCH
    assert all(w[2] == 100 for w in touch_writes)
    assert len(release_writes) == NCH
    assert all(w[2] == int(0.6 * 100) for w in release_writes)


def test_threshold_single_channel_with_explicit_release():
    m, dev = make()
    m.threshold(touch=80, release=30, channel=3)
    assert ('write_byte', TTH + 6, 80) in dev.writes
    assert ('write_byte', RTH + 6, 30) in dev.writes


def test_debounce_sets_nibbles_with_release_defaulting_to_touch():
    m, dev = make()
    m.debounce(touch=3)
    assert writes_to(dev, DEB) == [0x33]
    m.debounce(touch=1, release=4)
    assert writes_to(dev, DEB)[-1] == 0x41


def test_filter_packs_afe1_word():
    m, dev = make()
    m.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    assert writes_to(dev, AFE1, 'write_word') == [0x285e]


def test_charge_writes_current_and_merges_time_nibble():
    m, dev = make()
    reg = CDT + 5 // 2
    dev.bytes[reg] = 0x0f
    m.charge(channel=5, cdc=10, cdt=3)
    assert writes_to(dev, CDC + 5) == [10]
    assert writes_to(dev, reg) == [0x3f]


def test_auto_config_packs_control_words_and_limits():
    m, dev = make()
    m.auto_config()
    assert writes_to(dev, ACNF_C0, 'write_word') == [0x76f]
    assert writes_to(dev, ACNF_USL) == [200]
    assert writes_to(dev, ACNF_LSL) == [130]
    assert writes_to(dev, ACNF_TL) == [180]


def test_gpio_set_drives_bit_for_channel():
    m, dev = make()
    m.gpio_set(6, True)
    m.gpio_set(6, False)
    assert ('write_byte', GPIO_SET, 1 << 2) in dev.writes
    assert ('write_byte', GPIO_CLR, 1 << 2) in dev.writes


def test_gpio_set_rejects_non_gpio_channel():
    m, _ = make()
    with pytest.raises(AssertionError):
        m.gpio_set(3, True)


def test_gpio_status_reads_data_register():
    m, dev = make()
    dev.bytes[GPIO_DAT] = 0xa5
    assert m.gpio_status() == 0xa5
