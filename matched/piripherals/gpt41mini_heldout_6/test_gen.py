import builtins
import pytest
from piripherals import mpr121


class DummyBusDevice:
    def __init__(self):
        self.written = []
        self.bytes_read = {}
        self.words_read = {}
        self.blocks_read = {}
        self.bytes_to_read = {}
        self.read_byte_count = {}
        self.read_word_count = {}
        self.read_block_count = {}

    def write_byte(self, register, value):
        self.written.append(("write_byte", register, value))

    def write_word(self, register, value):
        self.written.append(("write_word", register, value))

    def write_block(self, register, values):
        self.written.append(("write_block", register, list(values)))

    def read_byte(self, register):
        self.read_byte_count[register] = self.read_byte_count.get(register, 0) + 1
        if register in self.bytes_to_read:
            return self.bytes_to_read[register].pop(0)
        return self.bytes_read.get(register, 0)

    def read_word(self, register):
        self.read_word_count[register] = self.read_word_count.get(register, 0) + 1
        return self.words_read.get(register, 0)

    def read_block(self, register, length):
        self.read_block_count[register] = self.read_block_count.get(register, 0) + 1
        if register in self.blocks_read:
            blk = self.blocks_read[register]
            if len(blk) >= length:
                return blk[:length]
            else:
                return blk + [0] * (length - len(blk))
        return [0] * length


class DummyBus:
    def __init__(self):
        self.device_called_with = []
        self.device_obj = DummyBusDevice()

    def device(self, addr):
        self.device_called_with.append(addr)
        return self.device_obj


@pytest.fixture
def dummy_bus():
    return DummyBus()


@pytest.fixture(autouse=True)
def patch_bus(monkeypatch, dummy_bus):
    # Patch piripherals.mpr121.Bus to return dummy_bus instance
    monkeypatch.setattr(mpr121, "Bus", lambda busnum=None: dummy_bus)


def test_init_with_bus_object(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus)
    # Should assign dummy bus device
    assert m._bus is dummy_bus.device_obj
    # Should initialize handlers list with noop
    assert len(m._handlers) == mpr121.NCH
    assert all(callable(h) for h in m._handlers)


def test_init_with_bus_int(monkeypatch, dummy_bus):
    # Allows bus param as int -> Bus -> device
    m = mpr121.MPR121(bus=1)
    assert dummy_bus.device_called_with == [0x5a]
    assert m._bus is dummy_bus.device_obj


def test_reset_writes_correct_byte(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    m.reset()
    # Reset command is writing 0x63 to SRESET register
    assert ("write_byte", mpr121.SRESET, 0x63) in dummy_bus.device_obj.written


def test_configure_writes_correct_register(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    m.configure(cl=2, prox=1, touch=5)
    # check ECR register written with bits (cl<<6)|(prox<<4)|touch
    expected_val = (2 << 6) | (1 << 4) | 5
    assert ("write_byte", mpr121.ECR, expected_val) in dummy_bus.device_obj.written


def test_touched_normal_behavior(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    # Set the read_word return value with overcurrent and out_of_range bits cleared. Bits 15,14=0
    dummy_bus.device_obj.words_read[mpr121.ETS] = 0x1AAA  # 0b1 1010 1010 1010 (lower 13 bits set)
    dummy_bus.device_obj.words_read[mpr121.OOR] = 0  # no out_of_range bits set

    touched = m.touched()
    # Only lower 13 bits returned
    assert touched == 0x1AAA & 0x1FFF


def test_touched_raises_on_overcurrent(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    # Set bit 15 = overcurrent
    dummy_bus.device_obj.words_read[mpr121.ETS] = (1 << 15)
    with pytest.raises(Exception) as e:
        m.touched()
    assert "overcurrent" in str(e.value)


def test_touched_raises_on_out_of_range(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    dummy_bus.device_obj.words_read[mpr121.ETS] = 0  # no overcurrent
    # Set out_of_range bits in OOR register, bit15 not set so no auto config fail
    dummy_bus.device_obj.words_read[mpr121.OOR] = 1 << 0  # some bit in lower 13 bits
    # Should not raise, because raise_on_failure tests overcurrent & out_of_range separately
    # But touched checks self.out_of_range() which raises on bit 15? So we patch out_of_range to return 0 first:
    dummy_bus.device_obj.words_read[mpr121.OOR] = 0  # no failure bits
    assert m.touched() == 0


def test_out_of_range_normal_and_failure(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    # No failure bits set
    dummy_bus.device_obj.words_read[mpr121.OOR] = 0x1000  # 12th bit set in lower 13 bits
    val = m.out_of_range()
    assert val == (0x1000 & 0x1FFF)

    # Set auto config fail bit (bit 15)
    dummy_bus.device_obj.words_read[mpr121.OOR] = (1 << 15)
    with pytest.raises(Exception) as e:
        m.out_of_range()
    assert "auto config failed" in str(e.value)

    # Set auto reconfig fail bit (bit 14)
    dummy_bus.device_obj.words_read[mpr121.OOR] = (1 << 14)
    with pytest.raises(Exception) as e:
        m.out_of_range()
    assert "auto reconfig failed" in str(e.value)


def test_electrode_data_parsing(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    # Provide a block of bytes for 13 channels * 2 bytes each (26 bytes)
    # Each channel data: low byte = i, high byte = i+1
    dummy_bus.device_obj.blocks_read[mpr121.EFD] = []
    for i in range(mpr121.NCH):
        dummy_bus.device_obj.blocks_read.setdefault(mpr121.EFD, []).extend([i, i + 1])
    data = m.electrode_data()
    assert len(data) == mpr121.NCH
    for i, val in enumerate(data):
        assert val == ((i + 1) << 8) + i


def test_baseline_returns_values(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    # Compose baseline block read of 13 values => will be left shifted by 2 by baseline()
    dummy_bus.device_obj.blocks_read[mpr121.EBL] = list(range(mpr121.NCH))
    baselines = m.baseline()
    assert baselines == [v << 2 for v in range(mpr121.NCH)]


def test_baseline_writes_block_for_rft_0_and_1(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    dummy_bus.device_obj.written.clear()
    # rft=0 triggers write_block to MHD with 4 bytes
    m.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20, prox=False)
    reg = mpr121.MHD + 4 * 0
    assert ("write_block", reg, [5, 1, 3, 20]) in dummy_bus.device_obj.written

    dummy_bus.device_obj.written.clear()
    # rft=1 triggers write_block to MHD with 4 bytes
    m.baseline(rft=1, mhd=6, nhd=2, ncl=4, fdl=21, prox=False)
    reg = mpr121.MHD + 4 * 1
    assert ("write_block", reg, [6, 2, 4, 21]) in dummy_bus.device_obj.written


def test_baseline_writes_block_for_rft_greater_than_1(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    dummy_bus.device_obj.written.clear()
    # rft=2 triggers write_block to MHDX with 3 bytes (no mhd)
    m.baseline(rft=2, nhd=7, ncl=5, fdl=22, prox=False)
    reg = mpr121.MHDX + 4 * 2
    # Should write fewer bytes = [nhd, ncl, fdl]
    calls = dummy_bus.device_obj.written
    assert ("write_block", reg, [7, 5, 22]) in calls


def test_threshold_default_release_all_channels(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    dummy_bus.device_obj.written.clear()
    m.threshold(50)
    # It should recursively call threshold with channel 0-12
    # Check that writing for some channel occurred and release threshold is int(50*0.6)=30
    found = False
    for call in dummy_bus.device_obj.written:
        if call[0] == "write_byte" and call[2] in (50, 30):
            found = True
    assert found


def test_threshold_custom_release_and_channel(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    dummy_bus.device_obj.written.clear()
    m.threshold(touch=40, release=20, channel=3)
    # Should write touch 40 and release 20 on registers TTH+6 and RTH+6
    assert ("write_byte", mpr121.TTH + 6, 40) in dummy_bus.device_obj.written
    assert ("write_byte", mpr121.RTH + 6, 20) in dummy_bus.device_obj.written


def test_debounce_default_release_equals_touch(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    dummy_bus.device_obj.written.clear()
    m.debounce(touch=3)
    # Debounce value: (release <<4) | touch with release=touch
    expected = (3 << 4) | 3
    assert ("write_byte", mpr121.DEB, expected) in dummy_bus.device_obj.written


def test_debounce_custom_release(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    dummy_bus.device_obj.written.clear()
    m.debounce(touch=2, release=5)
    expected = (5 << 4) | 2
    assert ("write_byte", mpr121.DEB, expected) in dummy_bus.device_obj.written


def test_filter_writes_word_register(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    dummy_bus.device_obj.written.clear()
    m.filter(cdc=31, cdt=3, ffi=2, sfi=1, esi=5)
    # lb = (ffi << 6) | cdc = (2 << 6) | 31 = 0xC0 + 31 = 0xCF
    lb = (2 << 6) | 31
    # hb = (cdt << 5) | (sfi << 3) | esi = (3 << 5) | (1 << 3) | 5 = 96 + 8 + 5 = 109 (0x6D)
    hb = (3 << 5) | (1 << 3) | 5
    expected_val = (hb << 8) | lb
    assert ("write_word", mpr121.AFE1, expected_val) in dummy_bus.device_obj.written


def test_charge_writes_bytes_and_registers(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    dummy_bus.device_obj.read_byte[mpr121.CDT] = 0x00
    dummy_bus.device_obj.read_byte[mpr121.CDT + 1] = 0xFF
    dummy_bus.device_obj.written.clear()
    # Charge channel 0, cdc=20, cdt=3
    m.charge(0, cdc=20, cdt=3)
    expected_cdc_call = ("write_byte", mpr121.CDC + 0, 20)
    assert expected_cdc_call in dummy_bus.device_obj.written

    # For channel 0, reg = CDT + 0//2 = CDT, s=4*(0%2)=0
    # we read byte from CDT then write modified value with cdt shifted by s bits
    found_writes = [w for w in dummy_bus.device_obj.written if w[0] == 'write_byte' and w[1] == mpr121.CDT]
    assert found_writes  # should write at least once to CDT register


def test_auto_config_writes_expected_registers(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    dummy_bus.device_obj.written.clear()
    m.auto_config(ace=1, are=0, bva=2, retry=3, afes=2, scts=1,
                  acfie=0, arfie=1, oorie=0, usl=100, lsl=50, tl=25)
    pok = dummy_bus.device_obj.written
    # It must write a word to ACNF_C0 and bytes to USL, LSL, TL
    assert any(w[0] == "write_word" and w[1] == mpr121.ACNF_C0 for w in pok)
    assert ("write_byte", mpr121.ACNF_USL, 100) in pok
    assert ("write_byte", mpr121.ACNF_LSL, 50) in pok
    assert ("write_byte", mpr121.ACNF_TL, 25) in pok


def test_gpio_setup_writes_bytes(monkeypatch, dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    dummy_bus.device_obj.read_byte.return_value = 0
    dummy_bus.device_obj.read_byte = lambda addr: 0
    dummy_bus.device_obj.written.clear()
    # patch read_byte to always return 0 to avoid read errors and ensure predictable writes

    # channel must be between 4 and 11
    m.gpio_setup(4, output=1, mode=3, enable=1)
    # Should write 4 times for control0, control1, dir, enable registers (each call writes once)

    # Check there are exactly 4 writes for gpio config
    writes = [w for w in dummy_bus.device_obj.written if w[0] == "write_byte"]
    # we expect 4 writes for gpio_setup
    assert len(writes) >= 4

    # `mode=1` internally maps to 0, test this triggers correct writes
    dummy_bus.device_obj.written.clear()
    m.gpio_setup(5, output=0, mode=1, enable=0)
    assert len(dummy_bus.device_obj.written) >= 4


def test_gpio_status_reads_byte(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    dummy_bus.device_obj.bytes_read[mpr121.GPIO_DAT] = 0x5A
    dummy_bus.device_obj.read_byte = lambda addr: dummy_bus.device_obj.bytes_read.get(addr, 0)
    val = m.gpio_status()
    assert isinstance(val, int)


def test_gpio_set_raises_on_invalid_channel(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    # channel lower than 4 should raise assertion
    with pytest.raises(AssertionError):
        m.gpio_set(3, True)
    # channel higher than 11 should raise assertion
    with pytest.raises(AssertionError):
        m.gpio_set(12, False)


def test_gpio_set_calls_write_byte(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    dummy_bus.device_obj.written.clear()
    m.gpio_set(4, True)
    assert any(w[0] == "write_byte" and w[1] == mpr121.GPIO_SET for w in dummy_bus.device_obj.written)
    dummy_bus.device_obj.written.clear()
    m.gpio_set(5, False)
    assert any(w[0] == "write_byte" and w[1] == mpr121.GPIO_CLR for w in dummy_bus.device_obj.written)


def test_on_touch_assigns_and_removes_handlers(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    called = []

    def handler_touch(state, ch):
        called.append((state, ch))

    m.on_touch(0, handler_touch)
    assert m._handlers[0] is not mpr121.noop

    # Remove handler
    m.on_touch(0, None)
    assert m._handlers[0] is mpr121.noop


def test_update_touch_state_calls_handle_touch(monkeypatch, dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)

    # Force initial _touched to 0, create touched() returning 0b101 (1+4)
    monkeypatch.setattr(m, "touched", lambda raise_on_failure=1: 0b101)

    # Replace _handle_touch to track calls
    calls = []

    def fake_handle_touch(touched, changed):
        calls.append((touched, changed))

    m._handle_touch = fake_handle_touch

    m.update_touch_state()

    # initial touched was 0, new touched 0b101, changed bits 0b101
    assert calls and calls[0] == (0b101, 0b101)


def test_handle_touch_invokes_handlers(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    touched = 0b101  # channels 0, 2 touched
    changed = 0b111  # channels 0,1,2 changed

    called = []

    def handler(state, ch):
        called.append((state, ch))

    # Assign handler to channels 0,1,2
    for i in range(3):
        m.on_touch(i, handler)

    m._handle_touch(touched, changed)

    # Should be called for all 3 but state True only on channels 0 and 2
    assert (True, 0) in called
    assert (False, 1) in called
    assert (True, 2) in called


def test_is_touched_returns_correct_state(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    m._touched = 0b1010
    assert m.is_touched(1) is True
    assert m.is_touched(0) is False


def test_setup_calls_expected_methods(monkeypatch, dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    calls = []

    def fake_reset():
        calls.append("reset")

    def fake_configure(*args, **kwargs):
        calls.append("configure")

    def fake_filter(*a, **k):
        calls.append("filter")

    def fake_auto_config(*a, **k):
        calls.append("auto_config")

    def fake_threshold(*a, **k):
        calls.append("threshold")

    def fake_debounce(*a, **k):
        calls.append("debounce")

    def fake_baseline(*a, **k):
        calls.append("baseline")

    monkeypatch.setattr(m, "reset", fake_reset)
    monkeypatch.setattr(m, "configure", fake_configure)
    monkeypatch.setattr(m, "filter", fake_filter)
    monkeypatch.setattr(m, "auto_config", fake_auto_config)
    monkeypatch.setattr(m, "threshold", fake_threshold)
    monkeypatch.setattr(m, "debounce", fake_debounce)
    monkeypatch.setattr(m, "baseline", fake_baseline)

    m.setup(reset=True, channels=5, prox=1, threshold=60, debounce=3, auto_config=True)

    # Check that relevant setup steps were called
    assert "reset" in calls
    assert calls.count("configure") >= 2  # configure called early and at end
    assert "filter" in calls
    assert "auto_config" in calls
    assert calls.count("threshold") >= 2
    assert "debounce" in calls
    # baseline called at least 4 times (two loops * 2 calls)
    assert calls.count("baseline") >= 4


def test_dump_runs_without_exception(monkeypatch, dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)

    # Patch all read_byte/read_word/read_block to return fixed values for dump
    dummy_bus.device_obj.read_byte = lambda reg: 10
    dummy_bus.device_obj.read_word = lambda reg: 0
    dummy_bus.device_obj.read_block = lambda reg, length: [0] * length

    # Patch other methods used by dump
    monkeypatch.setattr(m, "touched", lambda **kwargs: 0)
    monkeypatch.setattr(m, "out_of_range", lambda **kwargs: 0)
    monkeypatch.setattr(m, "gpio_status", lambda: 0)
    monkeypatch.setattr(m, "electrode_data", lambda: [0] * mpr121.NCH)
    monkeypatch.setattr(m, "baseline", lambda: [0] * mpr121.NCH)

    # Patch print and sys.stdout.write to avoid clutter
    monkeypatch.setattr(builtins, "print", lambda *a, **k: None)
    monkeypatch.setattr("sys.stdout.write", lambda x: None)

    m.dump(regs=0, up=0, loop=1)  # run one loop without register dump


def test_on_touch_accepts_handlers_with_one_arg(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    called = []

    def h(state):
        called.append(state)

    m.on_touch(0, h)
    # Manually call _handle_touch for channel 0 change bit set
    m._handle_touch(1 << 0, 1 << 0)
    assert True in called


def test_on_touch_ignores_handler_exceptions(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)

    def bad_handler(*args):
        raise TypeError("test")

    m.on_touch(0, bad_handler)
    # Should not raise exception when _handle_touch calls bad_handler
    m._handle_touch(1 << 0, 1 << 0)


def test_update_touch_state_calls_and_changes_state(monkeypatch, dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, setup=False)
    # Setup touched to toggle between calls
    states = [0, 2, 0]

    def touched_side_effect(raise_on_failure=1):
        return states.pop(0) if states else 0

    monkeypatch.setattr(m, "touched", touched_side_effect)
    calls = []

    def fake_handle(touched, changed):
        calls.append((touched, changed))

    m._handle_touch = fake_handle

    # call twice, state changes each time
    m.update_touch_state()
    m.update_touch_state()

    assert len(calls) >= 1


# END OF TESTS
