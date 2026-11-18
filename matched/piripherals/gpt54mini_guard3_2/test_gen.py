import types
import pytest

import piripherals.mpr121 as mpr


class FakeDevice:
    def __init__(self, byte_reads=None, word_reads=None, block_reads=None):
        self.byte_reads = dict(byte_reads or {})
        self.word_reads = dict(word_reads or {})
        self.block_reads = dict(block_reads or {})
        self.bytes_written = []
        self.words_written = []
        self.blocks_written = []
        self.read_byte_calls = []
        self.read_word_calls = []
        self.read_block_calls = []

    def read_byte(self, reg):
        self.read_byte_calls.append(reg)
        return self.byte_reads.get(reg, 0)

    def write_byte(self, reg, value):
        self.bytes_written.append((reg, value))

    def read_word(self, reg):
        self.read_word_calls.append(reg)
        return self.word_reads.get(reg, 0)

    def write_word(self, reg, value):
        self.words_written.append((reg, value))

    def read_block(self, reg, size):
        self.read_block_calls.append((reg, size))
        if reg in self.block_reads:
            return list(self.block_reads[reg])
        return [0] * size

    def write_block(self, reg, values):
        self.blocks_written.append((reg, list(values)))


class FakeBusFactory:
    def __init__(self, device):
        self.device_obj = device
        self.calls = []

    def __call__(self, bus):
        self.calls.append(bus)
        return types.SimpleNamespace(device=lambda addr: self.device_obj)


def test_init_uses_existing_bus_device_and_runs_setup(monkeypatch):
    dev = FakeDevice()
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    assert m._bus is dev
    assert m.addr == 0x5A
    assert m._handlers == [mpr.noop] * mpr.NCH
    assert m._touched == 0
    assert m.overcurrent is False


def test_init_creates_bus_when_needed_and_calls_device(monkeypatch):
    dev = FakeDevice()
    factory = FakeBusFactory(dev)
    monkeypatch.setattr(mpr, "Bus", factory)
    m = mpr.MPR121(bus=2, setup=False, reset=False, handlers=False, addr=0x33)
    assert factory.calls == [2]
    assert m._bus is dev


def test_init_with_irq_installs_irq_handler(monkeypatch):
    dev = FakeDevice()
    monkeypatch.setattr(mpr, "IRQHandler", lambda irq, cb: ("irq", irq, cb))
    monkeypatch.setattr(mpr, "Poller", lambda cb: pytest.fail("Poller should not be used"))
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), irq=7, handlers=True, setup=False, reset=False)
    assert m._bus is dev


def test_init_without_irq_installs_poller(monkeypatch):
    dev = FakeDevice()
    monkeypatch.setattr(mpr, "IRQHandler", lambda irq, cb: pytest.fail("IRQHandler should not be used"))
    seen = {}
    monkeypatch.setattr(mpr, "Poller", lambda cb: seen.setdefault("cb", cb))
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), irq=0, handlers=True, setup=False, reset=False)
    assert seen["cb"] == m.update_touch_state


def test_reset_writes_soft_reset_command():
    dev = FakeDevice()
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    m.reset()
    assert dev.bytes_written == [(mpr.SRESET, 0x63)]


def test_configure_writes_expected_ecr_value():
    dev = FakeDevice()
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    m.configure(cl=2, prox=3, touch=5)
    assert dev.bytes_written == [(mpr.ECR, (2 << 6) | (3 << 4) | 5)]


def test_touched_returns_bits_without_raising_when_disabled():
    dev = FakeDevice(word_reads={mpr.ETS: 0x8ABC})
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    assert m.touched(raise_on_failure=False) == (0x8ABC & 0x1FFF)


def test_touched_raises_on_overcurrent():
    dev = FakeDevice(word_reads={mpr.ETS: 1 << 15})
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    with pytest.raises(Exception, match="overcurrent"):
        m.touched()


def test_touched_raises_on_out_of_range(monkeypatch):
    dev = FakeDevice(word_reads={mpr.ETS: 0x1234, mpr.OOR: 1})
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    with pytest.raises(Exception, match="out of range"):
        m.touched()


def test_out_of_range_returns_bits_without_raising():
    dev = FakeDevice(word_reads={mpr.OOR: 0x9ABC})
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    assert m.out_of_range(raise_on_failure=False) == (0x9ABC & 0x1FFF)


def test_electrode_data_decodes_little_endian_words():
    block = []
    for i in range(mpr.NCH):
        value = 100 + i
        block.extend([value & 0xFF, (value >> 8) & 0xFF])
    dev = FakeDevice(block_reads={mpr.EFD: block})
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    assert m.electrode_data() == [100 + i for i in range(mpr.NCH)]


def test_baseline_reads_and_scales_values():
    raw = list(range(mpr.NCH))
    dev = FakeDevice(block_reads={mpr.EBL: raw})
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    assert m.baseline() == [b << 2 for b in raw]


def test_baseline_write_uses_correct_registers_for_touch_and_proximity():
    dev = FakeDevice()
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    m.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20, prox=False)
    m.baseline(rft=2, nhd=7, ncl=8, fdl=9, prox=True)
    assert dev.blocks_written == [
        (mpr.MHD + 0, [5, 1, 3, 20]),
        (mpr.MHDX + 8, [7, 8, 9]),
    ]


def test_threshold_applies_to_all_channels_and_defaults_release():
    dev = FakeDevice()
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    m.threshold(50)
    expected = []
    release = int(0.6 * 50)
    for i in range(mpr.NCH):
        expected.append((mpr.TTH + 2 * i, 50))
        expected.append((mpr.RTH + 2 * i, release))
    assert dev.bytes_written == expected


def test_debounce_defaults_release_to_touch():
    dev = FakeDevice()
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    m.debounce(touch=3)
    assert dev.bytes_written == [(mpr.DEB, (3 << 4) | 3)]


def test_filter_writes_combined_word():
    dev = FakeDevice()
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    m.filter(cdc=30, cdt=1, ffi=1, sfi=2, esi=4)
    lb = (1 << 6) | 30
    hb = (1 << 5) | (2 << 3) | 4
    assert dev.words_written == [(mpr.AFE1, (hb << 8) | lb)]


def test_charge_writes_current_and_updates_existing_ctd_bits():
    dev = FakeDevice(byte_reads={mpr.CDT + 0: 0b10110011})
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    m.charge(channel=1, cdc=22, cdt=9)
    assert dev.bytes_written[0] == (mpr.CDC + 1, 22)
    assert dev.bytes_written[1] == (mpr.CDT + 0, (0b10110011 & ~(0xF << 4)) | (9 << 4))


def test_auto_config_writes_expected_registers():
    dev = FakeDevice()
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    m.auto_config(ace=1, are=0, bva=2, retry=3, afes=1, scts=1, acfie=0, arfie=1, oorie=0, usl=201, lsl=131, tl=181)
    lb = (1 << 6) | (3 << 4) | (2 << 2) | (0 << 1) | 1
    hb = (1 << 7) | (0 << 2) | (1 << 1) | 0
    assert dev.words_written == [(mpr.ACNF_C0, (hb << 8) + lb)]
    assert dev.bytes_written == [(mpr.ACNF_USL, 201), (mpr.ACNF_LSL, 131), (mpr.ACNF_TL, 181)]


def test_gpio_status_reads_data_register():
    dev = FakeDevice(byte_reads={mpr.GPIO_DAT: 0xA5})
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    assert m.gpio_status() == 0xA5


def test_gpio_set_maps_channel_to_bit_and_register():
    dev = FakeDevice()
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    m.gpio_set(4, True)
    m.gpio_set(11, False)
    assert dev.bytes_written == [(mpr.GPIO_SET, 1 << 0), (mpr.GPIO_CLR, 1 << 7)]


def test_update_touch_state_invokes_handlers_for_changed_bits():
    dev = FakeDevice(word_reads={mpr.ETS: 0b101})
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    events = []
    m.on_touch(0, lambda touched, channel=None: events.append((0, touched, channel)))
    m.on_touch(2, lambda touched, channel=None: events.append((2, touched, channel)))
    m.update_touch_state()
    assert m._touched == 0b101
    assert events == [(0, True, 0), (2, True, 2)]


def test_on_touch_none_removes_handler():
    dev = FakeDevice()
    m = mpr.MPR121(bus=types.SimpleNamespace(device=lambda addr: dev), setup=False, reset=False, handlers=False)
    m.on_touch(1, lambda *a: pytest.fail("should not be called"))
    m.on_touch(1, None)
    assert m._handlers[1] is mpr.noop or m._handlers[1] == mpr.noop
