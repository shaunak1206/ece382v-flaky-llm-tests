import piripherals.mpr121 as mpr_mod


class FakeDevice:
    def __init__(self):
        self.writes = []
        self.read_bytes = {}
        self.read_words = {}
        self.read_blocks = {}

    def write_byte(self, reg, value):
        self.writes.append(("byte", reg, value))

    def write_word(self, reg, value):
        self.writes.append(("word", reg, value))

    def write_block(self, reg, values):
        self.writes.append(("block", reg, list(values)))

    def read_byte(self, reg):
        return self.read_bytes.get(reg, 0)

    def read_word(self, reg):
        if reg in self.read_words:
            return self.read_words[reg]
        return 0

    def read_block(self, reg, n):
        return list(self.read_blocks.get((reg, n), [0] * n))


class FakeBus:
    def __init__(self, bus_no):
        self.bus_no = bus_no
        self.dev = FakeDevice()

    def device(self, addr):
        self.addr = addr
        return self.dev


def make_mpr(monkeypatch, bus_obj=None, handlers=0, setup=False, reset=False, **kwargs):
    if bus_obj is None:
        bus_obj = FakeBus(1)
    monkeypatch.setattr(mpr_mod, "Bus", lambda bus_no: bus_obj)
    monkeypatch.setattr(mpr_mod, "IRQHandler", lambda irq, cb: None)
    monkeypatch.setattr(mpr_mod, "Poller", lambda cb: None)
    return mpr_mod.MPR121(bus=bus_obj, handlers=handlers, setup=setup, reset=reset, **kwargs), bus_obj.dev


def test_init_uses_existing_bus_device(monkeypatch):
    bus = FakeBus(2)
    mpr, dev = make_mpr(monkeypatch, bus_obj=bus)
    assert mpr._bus is dev
    assert mpr.addr == 0x5A
    assert mpr.overcurrent is False


def test_init_calls_bus_factory_when_no_device_attr(monkeypatch):
    created = {}

    class BusFactory:
        def __init__(self, bus_no):
            created["bus_no"] = bus_no

        def device(self, addr):
            created["addr"] = addr
            return FakeDevice()

    monkeypatch.setattr(mpr_mod, "Bus", BusFactory)
    monkeypatch.setattr(mpr_mod, "IRQHandler", lambda irq, cb: None)
    monkeypatch.setattr(mpr_mod, "Poller", lambda cb: None)
    mpr = mpr_mod.MPR121(bus=1, handlers=0, setup=False, reset=False)
    assert created == {"bus_no": 1, "addr": 0x5A}
    assert isinstance(mpr._bus, FakeDevice)


def test_reset_writes_soft_reset(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    mpr.reset()
    assert dev.writes == [("byte", mpr_mod.SRESET, 0x63)]


def test_configure_writes_expected_ecr_value(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    mpr.configure(cl=2, prox=3, touch=12)
    assert dev.writes == [("byte", mpr_mod.ECR, (2 << 6) | (3 << 4) | 12)]


def test_touched_returns_mask_without_errors(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    dev.read_words[mpr_mod.ETS] = 0x1234
    assert mpr.touched() == (0x1234 & 0x1FFF)


def test_touched_raises_on_overcurrent(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    dev.read_words[mpr_mod.ETS] = 0x8001
    dev.read_words[mpr_mod.OOR] = 0x0000
    with pytest.raises(Exception, match="overcurrent"):
        mpr.touched()


def test_touched_can_skip_failure_checks(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    dev.read_words[mpr_mod.ETS] = 0x8001
    dev.read_words[mpr_mod.OOR] = 0xC000
    assert mpr.touched(raise_on_failure=0) == 0x0001


def test_out_of_range_raises_on_auto_config_failed(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    dev.read_words[mpr_mod.OOR] = 0x8002
    with pytest.raises(Exception, match="auto config failed"):
        mpr.out_of_range()


def test_out_of_range_raises_on_auto_reconfig_failed(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    dev.read_words[mpr_mod.OOR] = 0x4002
    with pytest.raises(Exception, match="auto reconfig failed"):
        mpr.out_of_range()


def test_electrode_data_decodes_bytes_little_endian(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    raw = []
    for i in range(mpr_mod.NCH):
        raw.extend([i, i + 1])
    dev.read_blocks[(mpr_mod.EFD, 2 * mpr_mod.NCH)] = raw
    assert mpr.electrode_data() == [(i + 1) << 8 | i for i in range(mpr_mod.NCH)]


def test_baseline_reads_and_scales_values(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    dev.read_blocks[(mpr_mod.EBL, mpr_mod.NCH)] = list(range(mpr_mod.NCH))
    assert mpr.baseline() == [i << 2 for i in range(mpr_mod.NCH)]


def test_baseline_writes_rising_registers(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert dev.writes == [("block", mpr_mod.MHD, [5, 1, 3, 20])]


def test_baseline_writes_falling_prox_registers(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    mpr.baseline(rft=1, nhd=7, ncl=8, fdl=9, prox=1)
    assert dev.writes == [("block", mpr_mod.MHDX + 4, [7, 8, 9])]


def test_threshold_applies_to_all_channels_when_no_channel(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    mpr.threshold(50)
    expected = []
    release = int(0.6 * 50)
    for i in range(mpr_mod.NCH):
        expected.append(("byte", mpr_mod.TTH + 2 * i, 50))
        expected.append(("byte", mpr_mod.RTH + 2 * i, release))
    assert dev.writes == expected


def test_debounce_defaults_release_to_touch(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    mpr.debounce(touch=3)
    assert dev.writes == [("byte", mpr_mod.DEB, (3 << 4) | 3)]


def test_filter_writes_expected_word(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    mpr.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    lb = (1 << 6) | 30
    hb = (1 << 5) | (1 << 3) | 0
    assert dev.writes == [("word", mpr_mod.AFE1, (hb << 8) | lb)]


def test_charge_updates_cdc_and_cdt_bits(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    dev.read_bytes[mpr_mod.CDT + 0] = 0xAB
    mpr.charge(channel=0, cdc=9, cdt=2)
    assert dev.writes == [
        ("byte", mpr_mod.CDC + 0, 9),
        ("byte", mpr_mod.CDT + 0, (0xAB & ~0x0F) | (2 << 0)),
    ]


def test_auto_config_writes_word_and_limits(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    mpr.auto_config(ace=1, are=0, bva=2, retry=3, afes=1, scts=1, acfie=0, arfie=1, oorie=0, usl=201, lsl=131, tl=181)
    lb = (1 << 6) | (3 << 4) | (2 << 2) | (0 << 1) | 1
    hb = (1 << 7) | (0 << 2) | (1 << 1) | 0
    assert dev.writes == [
        ("word", mpr_mod.ACNF_C0, (hb << 8) + lb),
        ("byte", mpr_mod.ACNF_USL, 201),
        ("byte", mpr_mod.ACNF_LSL, 131),
        ("byte", mpr_mod.ACNF_TL, 181),
    ]


def test_gpio_setup_sets_register_bits(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    dev.read_bytes.update({
        mpr_mod.GPIO_CTL0: 0b10101010,
        mpr_mod.GPIO_CTL1: 0b01010101,
        mpr_mod.GPIO_DIR: 0b00001111,
        mpr_mod.GPIO_EN: 0b11110000,
    })
    mpr.gpio_setup(channel=4, output=1, mode=3, enable=1)
    assert ("byte", mpr_mod.GPIO_CTL0, 0b10101011) in dev.writes
    assert ("byte", mpr_mod.GPIO_CTL1, 0b01010101) in dev.writes
    assert ("byte", mpr_mod.GPIO_DIR, 0b00001111) in dev.writes
    assert ("byte", mpr_mod.GPIO_EN, 0b11110001) in dev.writes


def test_gpio_set_selects_set_or_clear_register(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    mpr.gpio_set(4, True)
    mpr.gpio_set(11, False)
    assert dev.writes == [
        ("byte", mpr_mod.GPIO_SET, 1 << 0),
        ("byte", mpr_mod.GPIO_CLR, 1 << 7),
    ]


def test_update_touch_state_triggers_handler_on_change(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    calls = []
    mpr.on_touch(0, lambda touched, channel=None: calls.append((touched, channel)))
    mpr.touched = lambda raise_on_failure=1: 0x0001
    mpr.update_touch_state()
    assert calls == [(True, 0)]


def test_handle_touch_supports_handler_without_channel_argument(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    calls = []
    mpr._handlers[1] = lambda touched: calls.append(touched)
    mpr._handle_touch(touched=0x0002, changed=0x0002)
    assert calls == [True]


def test_setup_performs_expected_register_writes(monkeypatch):
    mpr, dev = make_mpr(monkeypatch)
    mpr.setup(reset=1, channels=12, prox=1, threshold=50, debounce=2, auto_config=1)
    assert ("byte", mpr_mod.SRESET, 0x63) in dev.writes
    assert ("byte", mpr_mod.ECR, (0 << 6) | (1 << 4) | 0) in dev.writes
    assert ("byte", mpr_mod.ECR, (0 << 6) | (1 << 4) | 12) in dev.writes
    assert ("byte", mpr_mod.DEB, (2 << 4) | 2) in dev.writes
    assert ("byte", mpr_mod.TTH + 24, 12) in dev.writes
    assert ("byte", mpr_mod.RTH + 24, int(0.6 * 12)) in dev.writes
