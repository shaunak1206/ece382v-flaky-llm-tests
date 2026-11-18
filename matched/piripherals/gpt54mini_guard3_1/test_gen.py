import piripherals.mpr121 as mpr121


class FakeDevice:
    def __init__(self):
        self.writes = []
        self.read_bytes = {}
        self.read_words = {}
        self.read_blocks = {}

    def write_byte(self, reg, value):
        self.writes.append(("write_byte", reg, value))
        self.read_bytes[reg] = value

    def write_word(self, reg, value):
        self.writes.append(("write_word", reg, value))
        self.read_words[reg] = value

    def write_block(self, reg, data):
        self.writes.append(("write_block", reg, list(data)))
        self.read_blocks[reg] = list(data)

    def read_byte(self, reg):
        return self.read_bytes.get(reg, 0)

    def read_word(self, reg):
        return self.read_words.get(reg, 0)

    def read_block(self, reg, length):
        block = self.read_blocks.get(reg, [0] * length)
        return list(block[:length]) + [0] * max(0, length - len(block))


class FakeBusFactory:
    def __init__(self, device=None):
        self.device_obj = device or FakeDevice()
        self.calls = []

    def device(self, addr):
        self.calls.append(addr)
        return self.device_obj


def test_init_uses_existing_bus_device_and_skips_setup(monkeypatch):
    bus = FakeBusFactory()
    monkeypatch.setattr(mpr121, "IRQHandler", lambda *a, **k: None)
    monkeypatch.setattr(mpr121, "Poller", lambda *a, **k: None)

    m = mpr121.MPR121(bus=bus, addr=0x5a, handlers=0, setup=0, reset=0)

    assert m.addr == 0x5a
    assert m._bus is bus.device_obj
    assert bus.calls == [0x5a]
    assert m._handlers == [mpr121.noop] * mpr121.NCH


def test_init_with_integer_bus_constructs_bus_and_resets(monkeypatch):
    created = []

    class DummyBus:
        def __init__(self, busnum):
            created.append(busnum)
            self.device_obj = FakeDevice()

        def device(self, addr):
            self.addr = addr
            return self.device_obj

    monkeypatch.setattr(mpr121, "IRQHandler", lambda *a, **k: None)
    monkeypatch.setattr(mpr121, "Poller", lambda *a, **k: None)
    monkeypatch.setattr(mpr121, "Bus", DummyBus)

    m = mpr121.MPR121(bus=1, addr=0x5a, handlers=0, setup=0, reset=1)

    assert created == [1]
    assert m._bus.writes == [("write_byte", mpr121.SRESET, 0x63)]


def test_init_installs_irq_handler_when_irq_enabled(monkeypatch):
    calls = []

    monkeypatch.setattr(mpr121, "Poller", lambda *a, **k: calls.append("poller"))
    monkeypatch.setattr(
        mpr121, "IRQHandler", lambda irq, cb: calls.append(("irq", irq, cb))
    )

    m = mpr121.MPR121(
        bus=FakeBusFactory(),
        irq=17,
        handlers=1,
        setup=0,
        reset=0,
    )

    assert calls and calls[0][0] == "irq"
    assert calls[0][1] == 17
    assert calls[0][2] == m.update_touch_state
    assert "poller" not in calls


def test_init_installs_poller_when_no_irq(monkeypatch):
    calls = []

    monkeypatch.setattr(mpr121, "IRQHandler", lambda *a, **k: calls.append("irq"))
    monkeypatch.setattr(
        mpr121, "Poller", lambda cb: calls.append(("poller", cb))
    )

    m = mpr121.MPR121(bus=FakeBusFactory(), irq=0, handlers=1, setup=0, reset=0)

    assert calls == [("poller", m.update_touch_state)]


def test_reset_writes_soft_reset_command():
    dev = FakeDevice()
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    m.reset()

    assert dev.writes == [("write_byte", mpr121.SRESET, 0x63)]


def test_configure_writes_expected_ecr_value():
    dev = FakeDevice()
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    m.configure(cl=3, prox=2, touch=7)

    assert dev.writes[-1] == ("write_byte", mpr121.ECR, (3 << 6) | (2 << 4) | 7)


def test_touched_returns_bitmask_without_exception_when_no_failures():
    dev = FakeDevice()
    dev.read_words[mpr121.ETS] = 0x1234
    dev.read_words[mpr121.OOR] = 0
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    result = m.touched()

    assert result == (0x1234 & 0x1FFF)


def test_touched_raises_on_overcurrent_bit():
    dev = FakeDevice()
    dev.read_words[mpr121.ETS] = 1 << 15
    dev.read_words[mpr121.OOR] = 0
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    try:
        m.touched()
        assert False, "expected exception"
    except Exception as exc:
        assert str(exc) == "overcurrent"


def test_touched_raises_on_out_of_range_when_requested():
    dev = FakeDevice()
    dev.read_words[mpr121.ETS] = 0
    dev.read_words[mpr121.OOR] = 1
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    try:
        m.touched()
        assert False, "expected exception"
    except Exception as exc:
        assert str(exc) == "out of range"


def test_electrode_data_decodes_13_words_from_block():
    dev = FakeDevice()
    block = []
    for i in range(mpr121.NCH):
        value = 100 + i
        block.extend([value & 0xFF, (value >> 8) & 0xFF])
    dev.read_blocks[mpr121.EFD] = block
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    assert m.electrode_data() == [100 + i for i in range(mpr121.NCH)]


def test_baseline_read_shifts_each_byte_left_by_two():
    dev = FakeDevice()
    dev.read_blocks[mpr121.EBL] = list(range(mpr121.NCH))
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    assert m.baseline() == [i << 2 for i in range(mpr121.NCH)]


def test_baseline_write_uses_mhd_register_for_rft_0():
    dev = FakeDevice()
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    m.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20, prox=0)

    assert dev.writes[-1] == (
        "write_block",
        mpr121.MHD,
        [5, 1, 3, 20],
    )


def test_baseline_write_uses_mhdx_register_for_prox_and_rft_1():
    dev = FakeDevice()
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    m.baseline(rft=1, mhd=5, nhd=1, ncl=3, fdl=20, prox=1)

    assert dev.writes[-1] == (
        "write_block",
        mpr121.MHDX + 4 * 1,
        [5, 1, 3, 20],
    )


def test_baseline_write_uses_three_byte_form_for_rft_2():
    dev = FakeDevice()
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    m.baseline(rft=2, nhd=7, ncl=8, fdl=9, prox=0)

    assert dev.writes[-1] == (
        "write_block",
        mpr121.MHD + 4 * 2,
        [7, 8, 9],
    )


def test_threshold_applies_default_release_and_all_channels():
    dev = FakeDevice()
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    m.threshold(50)

    expected = []
    release = int(0.6 * 50)
    for i in range(mpr121.NCH):
        expected.append(("write_byte", mpr121.TTH + 2 * i, 50))
        expected.append(("write_byte", mpr121.RTH + 2 * i, release))
    assert dev.writes == expected


def test_debounce_defaults_release_to_touch():
    dev = FakeDevice()
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    m.debounce(touch=3)

    assert dev.writes[-1] == ("write_byte", mpr121.DEB, (3 << 4) | 3)


def test_filter_writes_combined_word():
    dev = FakeDevice()
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    m.filter(cdc=30, cdt=1, ffi=1, sfi=2, esi=4)

    lb = (1 << 6) | 30
    hb = (1 << 5) | (2 << 3) | 4
    assert dev.writes[-1] == ("write_word", mpr121.AFE1, (hb << 8) | lb)


def test_charge_updates_current_and_preserves_other_cdt_nibble():
    dev = FakeDevice()
    dev.read_bytes[mpr121.CDT + 0] = 0xAB
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    m.charge(channel=0, cdc=17, cdt=5)

    assert ("write_byte", mpr121.CDC + 0, 17) in dev.writes
    assert ("write_byte", mpr121.CDT + 0, (0xAB & ~0x0F) | 5) in dev.writes


def test_auto_config_writes_expected_registers():
    dev = FakeDevice()
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    m.auto_config(
        ace=1,
        are=1,
        bva=3,
        retry=2,
        afes=1,
        scts=0,
        acfie=1,
        arfie=1,
        oorie=1,
        usl=200,
        lsl=130,
        tl=180,
    )

    lb = (1 << 6) | (2 << 4) | (3 << 2) | (1 << 1) | 1
    hb = (0 << 7) | (1 << 2) | (1 << 1) | 1
    assert ("write_word", mpr121.ACNF_C0, (hb << 8) + lb) in dev.writes
    assert ("write_byte", mpr121.ACNF_USL, 200) in dev.writes
    assert ("write_byte", mpr121.ACNF_LSL, 130) in dev.writes
    assert ("write_byte", mpr121.ACNF_TL, 180) in dev.writes


def test_out_of_range_returns_bitmask_without_exception():
    dev = FakeDevice()
    dev.read_words[mpr121.OOR] = 0x1234
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    assert m.out_of_range() == (0x1234 & 0x1FFF)


def test_update_touch_state_triggers_handlers_on_change(monkeypatch):
    dev = FakeDevice()
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    called = []

    def handler(*args):
        called.append(args)

    m.on_touch(2, handler)
    m._touched = 0
    monkeypatch.setattr(m, "touched", lambda: 1 << 2)

    m.update_touch_state()

    assert m._touched == (1 << 2)
    assert called == [((True, 2))] or called == [(True, 2)]


def test_handle_touch_falls_back_to_one_argument_handler():
    dev = FakeDevice()
    m = mpr121.MPR121(bus=FakeBusFactory(dev), handlers=0, setup=0, reset=0)

    called = []

    def handler(state):
        called.append(state)

    m._handlers[1] = handler
    m._handle_touch(touched=(1 << 1), changed=(1 << 1))

    assert called == [True]
