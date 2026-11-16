import pytest

from piripherals.mpr121 import (
    MPR121,
    ACNF_C0,
    ACNF_LSL,
    ACNF_TL,
    ACNF_USL,
    AFE1,
    CDC,
    CDT,
    DEB,
    ECR,
    EBL,
    EFD,
    ETS,
    GPIO_CLR,
    GPIO_CTL0,
    GPIO_CTL1,
    GPIO_DAT,
    GPIO_DIR,
    GPIO_EN,
    GPIO_SET,
    MHD,
    MHDX,
    NCH,
    OOR,
    RTH,
    SRESET,
    TTH,
)


class FakeDevice:
    """In-memory stand-in for a single I2C device (byte-addressed, little endian words)."""

    def __init__(self):
        self.mem = {}
        self.writes = []

    def write_byte(self, reg, value):
        self.mem[reg] = value & 0xFF
        self.writes.append(("byte", reg, value))

    def read_byte(self, reg):
        return self.mem.get(reg, 0)

    def write_word(self, reg, value):
        self.mem[reg] = value & 0xFF
        self.mem[reg + 1] = (value >> 8) & 0xFF
        self.writes.append(("word", reg, value))

    def read_word(self, reg):
        return self.mem.get(reg, 0) | (self.mem.get(reg + 1, 0) << 8)

    def read_block(self, reg, n):
        return [self.mem.get(reg + i, 0) for i in range(n)]

    def write_block(self, reg, values):
        for i, v in enumerate(values):
            self.mem[reg + i] = v & 0xFF
        self.writes.append(("block", reg, list(values)))

    def poke(self, reg, values):
        """Set device memory directly without recording a write."""
        if isinstance(values, int):
            values = [values]
        for i, v in enumerate(values):
            self.mem[reg + i] = v & 0xFF


class FakeBus:
    """Stand-in for the bus object; exposes a ``device(addr)`` method like the real one."""

    def __init__(self):
        self.device_obj = FakeDevice()
        self.addresses = []

    def device(self, addr):
        self.addresses.append(addr)
        return self.device_obj


@pytest.fixture
def fake_bus():
    return FakeBus()


def make_sensor(bus, **kwargs):
    """Create a sensor with no IRQ/polling, no setup and no reset by default."""
    kwargs.setdefault("handlers", 0)
    kwargs.setdefault("setup", 0)
    kwargs.setdefault("reset", 0)
    return MPR121(bus=bus, **kwargs)


def test_init_without_side_effects_binds_device_address(fake_bus):
    sensor = make_sensor(fake_bus, addr=0x5B)

    assert fake_bus.addresses == [0x5B]
    assert sensor.addr == 0x5B
    assert sensor.overcurrent is False
    assert fake_bus.device_obj.writes == []


def test_init_with_reset_only_sends_soft_reset(fake_bus):
    make_sensor(fake_bus, reset=1, setup=0)

    assert ("byte", SRESET, 0x63) in fake_bus.device_obj.writes
    assert len(fake_bus.device_obj.writes) == 1


def test_reset_writes_soft_reset_command(fake_bus):
    sensor = make_sensor(fake_bus)
    sensor.reset()

    assert fake_bus.device_obj.writes == [("byte", SRESET, 0x63)]


def test_configure_packs_run_mode_bits(fake_bus):
    sensor = make_sensor(fake_bus)
    sensor.configure(cl=3, prox=0, touch=12)

    # (3 << 6) | (0 << 4) | 12
    assert fake_bus.device_obj.mem[ECR] == 0xCC


def test_configure_packs_proximity_and_touch_fields(fake_bus):
    sensor = make_sensor(fake_bus)
    sensor.configure(cl=1, prox=2, touch=5)

    assert fake_bus.device_obj.mem[ECR] == (1 << 6) | (2 << 4) | 5


def test_touched_returns_masked_touch_bits(fake_bus):
    dev = fake_bus.device_obj
    dev.poke(ETS, [0x05, 0xFF])  # word 0xff05 -> masked to 0x1f05
    sensor = make_sensor(fake_bus)

    assert sensor.touched() == 0x1F05


def test_touched_raises_on_overcurrent(fake_bus):
    fake_bus.device_obj.poke(ETS, [0x01, 0x80])  # bit 15 set
    sensor = make_sensor(fake_bus)

    with pytest.raises(Exception, match="overcurrent"):
        sensor.touched()


def test_touched_without_raise_ignores_failures(fake_bus):
    fake_bus.device_obj.poke(ETS, [0x01, 0x80])
    fake_bus.device_obj.poke(OOR, [0x04, 0x00])
    sensor = make_sensor(fake_bus)

    assert sensor.touched(raise_on_failure=0) == 0x0001


def test_touched_raises_when_out_of_range_bits_set(fake_bus):
    fake_bus.device_obj.poke(OOR, [0x02, 0x00])  # channel 1 out of range
    sensor = make_sensor(fake_bus)

    with pytest.raises(Exception, match="out of range"):
        sensor.touched()


def test_out_of_range_raises_on_auto_config_failure(fake_bus):
    fake_bus.device_obj.poke(OOR, [0x00, 0x80])  # bit 15
    sensor = make_sensor(fake_bus)

    with pytest.raises(Exception, match="auto config failed"):
        sensor.out_of_range()


def test_out_of_range_raises_on_auto_reconfig_failure(fake_bus):
    fake_bus.device_obj.poke(OOR, [0x00, 0x40])  # bit 14
    sensor = make_sensor(fake_bus)

    with pytest.raises(Exception, match="auto reconfig failed"):
        sensor.out_of_range()


def test_update_touch_state_reflects_is_touched(fake_bus):
    fake_bus.device_obj.poke(ETS, [0x05, 0x00])  # channels 0 and 2
    sensor = make_sensor(fake_bus)
    sensor.update_touch_state()

    assert sensor.is_touched(0) is True
    assert sensor.is_touched(1) is False
    assert sensor.is_touched(2) is True


def test_handler_called_with_state_and_channel_on_change(fake_bus):
    dev = fake_bus.device_obj
    sensor = make_sensor(fake_bus)
    events = []
    sensor.on_touch(3, lambda touched, channel: events.append((touched, channel)))

    dev.poke(ETS, [0x08, 0x00])
    sensor.update_touch_state()
    dev.poke(ETS, [0x00, 0x00])
    sensor.update_touch_state()

    assert events == [(True, 3), (False, 3)]


def test_handler_not_called_when_state_unchanged(fake_bus):
    dev = fake_bus.device_obj
    sensor = make_sensor(fake_bus)
    calls = []
    sensor.on_touch(0, lambda touched, channel: calls.append(touched))

    dev.poke(ETS, [0x01, 0x00])
    sensor.update_touch_state()
    sensor.update_touch_state()

    assert calls == [True]


def test_electrode_data_decodes_little_endian_pairs(fake_bus):
    dev = fake_bus.device_obj
    dev.poke(EFD, [0x34, 0x02, 0xFF, 0x03])  # channel 0 = 0x234, channel 1 = 0x3ff
    sensor = make_sensor(fake_bus)

    data = sensor.electrode_data()

    assert len(data) == NCH
    assert data[0] == 0x234
    assert data[1] == 0x3FF
    assert all(v == 0 for v in data[2:])


def test_baseline_read_returns_values_shifted_by_two(fake_bus):
    dev = fake_bus.device_obj
    dev.poke(EBL, list(range(1, NCH + 1)))
    sensor = make_sensor(fake_bus)

    assert sensor.baseline() == [(i + 1) << 2 for i in range(NCH)]


def test_baseline_write_rising_scenario_uses_mhd_block(fake_bus):
    sensor = make_sensor(fake_bus)
    sensor.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)

    assert fake_bus.device_obj.writes == [("block", MHD, [5, 1, 3, 20])]


def test_baseline_write_touched_scenario_uses_three_fields_and_prox_register(fake_bus):
    sensor = make_sensor(fake_bus)
    sensor.baseline(rft=2, nhd=4, ncl=5, fdl=6)
    sensor.baseline(rft=1, mhd=2, nhd=0, ncl=0, fdl=0, prox=1)

    writes = fake_bus.device_obj.writes
    # rft=2 -> MHD + 8, only nhd/ncl/fdl written
    assert writes[0] == ("block", MHD + 8, [4, 5, 6])
    # prox with rft=1 -> MHDX + 4, full four-field block
    assert writes[1] == ("block", MHDX + 4, [2, 0, 0, 0])


def test_threshold_single_channel_writes_touch_and_default_release(fake_bus):
    sensor = make_sensor(fake_bus)
    sensor.threshold(touch=50, channel=5)

    mem = fake_bus.device_obj.mem
    assert mem[TTH + 10] == 50
    assert mem[RTH + 10] == int(0.6 * 50)


def test_threshold_all_channels_writes_every_channel_including_proximity(fake_bus):
    sensor = make_sensor(fake_bus)
    sensor.threshold(touch=100, release=40)

    mem = fake_bus.device_obj.mem
    for ch in range(NCH):
        assert mem[TTH + 2 * ch] == 100
        assert mem[RTH + 2 * ch] == 40


def test_debounce_packs_release_and_touch_nibbles(fake_bus):
    sensor = make_sensor(fake_bus)
    sensor.debounce(touch=3, release=1)
    assert fake_bus.device_obj.mem[DEB] == 0x13

    sensor.debounce(touch=2)  # release defaults to touch
    assert fake_bus.device_obj.mem[DEB] == 0x22


def test_filter_defaults_pack_into_afe1_word(fake_bus):
    sensor = make_sensor(fake_bus)
    sensor.filter()

    mem = fake_bus.device_obj.mem
    # lb = cdc 16 = 0x10; hb = (cdt 1 << 5) | esi 4 = 0x24
    assert mem[AFE1] == 0x10
    assert mem[AFE1 + 1] == 0x24


def test_charge_sets_current_and_preserves_other_nibble(fake_bus):
    dev = fake_bus.device_obj
    dev.poke(CDT + 1, 0xF0)  # channel 2 uses low nibble of CDT+1
    sensor = make_sensor(fake_bus)

    sensor.charge(channel=2, cdc=10, cdt=5)

    assert dev.mem[CDC + 2] == 10
    assert dev.mem[CDT + 1] == 0xF5


def test_auto_config_defaults_write_config_and_limits(fake_bus):
    sensor = make_sensor(fake_bus)
    sensor.auto_config()

    mem = fake_bus.device_obj.mem
    # lb = afes1<<6 | retry2<<4 | bva3<<2 | are1<<1 | ace1 = 0x6f
    # hb = oorie1<<2 | arfie1<<1 | acfie1 = 0x07
    assert mem[ACNF_C0] == 0x6F
    assert mem[ACNF_C0 + 1] == 0x07
    assert mem[ACNF_USL] == 200
    assert mem[ACNF_LSL] == 130
    assert mem[ACNF_TL] == 180


def test_gpio_set_high_and_low_use_set_and_clear_registers(fake_bus):
    sensor = make_sensor(fake_bus)
    sensor.gpio_set(5, True)
    sensor.gpio_set(5, False)

    assert fake_bus.device_obj.writes == [
        ("byte", GPIO_SET, 1 << 1),
        ("byte", GPIO_CLR, 1 << 1),
    ]


def test_gpio_set_rejects_non_gpio_channel(fake_bus):
    sensor = make_sensor(fake_bus)

    with pytest.raises(AssertionError):
        sensor.gpio_set(3, True)


def test_gpio_status_returns_data_register(fake_bus):
    fake_bus.device_obj.poke(GPIO_DAT, 0xA5)
    sensor = make_sensor(fake_bus)

    assert sensor.gpio_status() == 0xA5


def test_gpio_setup_configures_mode_direction_and_enable_bits(fake_bus, capsys):
    dev = fake_bus.device_obj
    dev.poke(GPIO_DIR, 0x08)  # channel 7 currently output, should become input
    sensor = make_sensor(fake_bus)

    sensor.gpio_setup(channel=7, output=False, mode=3, enable=1)

    assert (dev.mem[GPIO_CTL0] >> 3) & 1 == 1
    assert (dev.mem[GPIO_CTL1] >> 3) & 1 == 1
    assert (dev.mem[GPIO_DIR] >> 3) & 1 == 0
    assert (dev.mem[GPIO_EN] >> 3) & 1 == 1
