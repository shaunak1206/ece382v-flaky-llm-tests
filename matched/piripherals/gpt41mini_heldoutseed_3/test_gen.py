import builtins
import pytest
from piripherals import mpr121

class DummyBusDevice:
    def __init__(self):
        self._writes = []
        self._bytes = {}
        self._words = {}
        self._blocks = {}
        self._read_byte_calls = []
        self._last_write_byte = None

    def write_byte(self, reg, val):
        self._writes.append((reg, val))
        self._last_write_byte = (reg, val)

    def write_word(self, reg, val):
        self._writes.append((reg, val))

    def write_block(self, reg, vals):
        self._writes.append((reg, vals))

    def read_byte(self, reg):
        self._read_byte_calls.append(reg)
        # Return preset value or 0 if none preset
        return self._bytes.get(reg, 0)

    def read_word(self, reg):
        return self._words.get(reg, 0)

    def read_block(self, reg, length):
        return self._blocks.get(reg, [0] * length)

class DummyBus:
    def device(self, addr):
        return DummyBusDevice()

# Helper to replace bus.device with our dummy device
class DummyBusFactory:
    def __init__(self):
        self.device_inst = DummyBusDevice()
    def device(self, addr):
        return self.device_inst

def noop(*args, **kwargs):
    pass

def not_raising(f):
    if f is None:
        return None
    return f

# Patch .bus.Bus to avoid real hardware accesses
@pytest.fixture(autouse=True)
def patch_bus(monkeypatch):
    # Patch piripherals.mpr121.Bus to DummyBus
    monkeypatch.setattr(mpr121, "Bus", lambda busnum: DummyBusFactory())
    yield

def test_init_creates_bus_device(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    assert hasattr(m, "_bus")
    # _handlers is a list length NCH
    assert isinstance(m._handlers, list)
    assert len(m._handlers) == mpr121.NCH
    # _touched initialized 0
    assert m._touched == 0

def test_reset_calls_write_byte(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    m.reset()
    found = any(w[0] == mpr121.SRESET and w[1] == 0x63 for w in m._bus._writes)
    assert found

def test_configure_writes_correct_byte(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    # default configure prox=0 touch=0 cl=3
    m.configure(cl=3, prox=2, touch=5)
    expected_val = (3 << 6) | (2 << 4) | 5
    assert (mpr121.ECR, expected_val) in m._bus._writes

def test_touch_handlers_called_on_change(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)

    triggered = []

    def handler(touched, ch):
        triggered.append((touched, ch))

    m.on_touch(0, handler)
    m._touched = 0b0000
    # Simulate touched returns bit 0 touched
    monkeypatch.setattr(m, "touched", lambda raise_on_failure=1: 0b0001)
    # update_touch_state triggers _handle_touch -> handler called with True for channel 0
    m.update_touch_state()
    assert triggered == [(True, 0)]

def test_touch_handlers_typeerror_2args(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    triggered = []

    def handler(touched):
        triggered.append(touched)

    m.on_touch(1, handler)
    m._touched = 0b0000
    monkeypatch.setattr(m, "touched", lambda raise_on_failure=1: 0b0010)
    m.update_touch_state()
    # handler should be called with True or False boolean only
    assert True in triggered or False in triggered

def test_is_touched_returns_correct(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    m._touched = 0b1010
    assert m.is_touched(1) is True  # bit 1 is set
    assert m.is_touched(0) is False  # bit 0 is not set

def test_on_touch_assign_none_removes_handler(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    def dummy(touched, ch): pass
    m.on_touch(5, dummy)
    m.on_touch(5, None)
    # After removal, handler is noop
    assert m._handlers[5] == mpr121.noop

def test_touched_reads_word_and_checks_errors(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    # Provide a word with overcurrent bit set (bit 15)
    m._bus._words[mpr121.ETS] = 1 << 15
    with pytest.raises(Exception, match='overcurrent'):
        m.touched()
    # Provide word with no overcurrent but out_of_range True triggers exception
    m._bus._words[mpr121.ETS] = 0
    # simulate out_of_range to True by patching it to raise
    def oor(raise_on_failure=1):
        raise Exception('out of range')
    monkeypatch.setattr(m, "out_of_range", oor)
    with pytest.raises(Exception, match='out of range'):
        m.touched()

def test_touched_returns_lower_13_bits(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    rawword = 0xFFFF
    m._bus._words[mpr121.ETS] = rawword
    # patch out_of_range returns False to not raise
    monkeypatch.setattr(m, "out_of_range", lambda raise_on_failure=1: 0)
    val = m.touched()
    # val lower 13 bits of rawword
    assert val == (rawword & 0x1fff)

def test_electrode_data_reads_block(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    # Provide block of bytes 2*NCH, alternating pattern
    data_bytes = []
    for i in range(mpr121.NCH):
        data_bytes.extend([i, i+1])
    m._bus._blocks[mpr121.EFD] = data_bytes
    electrode_data = m.electrode_data()
    assert isinstance(electrode_data, list)
    assert len(electrode_data) == mpr121.NCH
    # check first value equals bytes combined little endian (low + high)
    assert electrode_data[0] == (data_bytes[1] << 8) + data_bytes[0]

def test_baseline_returns_list(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    base_vals = [i for i in range(mpr121.NCH)]
    m._bus._blocks[mpr121.EBL] = base_vals
    # baseline() with no args returns scaled baseline (b<<2)
    returned = m.baseline()
    assert all(r == b << 2 for r, b in zip(returned, base_vals))

def test_baseline_writes_block_for_values(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    # For rft > 1 writes 3 bytes to MHDX+4*rft (prox or no prox)
    m.baseline(rft=2, nhd=10, ncl=20, fdl=30, prox=False)
    found = any(type(vals) == list for reg, vals in m._bus._writes if reg == mpr121.MHDX + 8)
    assert found
    m._bus._writes.clear()
    m.baseline(rft=0, mhd=5, nhd=10, ncl=20, fdl=30, prox=True)
    found2 = any(type(vals) == list for reg, vals in m._bus._writes if reg == mpr121.MHD + 0)
    assert found2

def test_threshold_sets_bytes_for_all_channels(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    m.threshold(100)
    # There should be 2*NCH writes: touch and release for each channel
    touch_writes = [w for w in m._bus._writes if w[0] >= mpr121.TTH and w[0] < mpr121.TTH + 2 * mpr121.NCH]
    release_writes = [w for w in m._bus._writes if w[0] >= mpr121.RTH and w[0] < mpr121.RTH + 2 * mpr121.NCH]
    assert len(touch_writes) >= mpr121.NCH
    assert len(release_writes) >= mpr121.NCH

def test_threshold_sets_bytes_for_one_channel(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    m.threshold(50, 30, channel=3)
    assert (mpr121.TTH + 6, 50) in m._bus._writes
    assert (mpr121.RTH + 6, 30) in m._bus._writes

def test_debounce_writes_byte(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    m.debounce(touch=3, release=5)
    found = any(w[0] == mpr121.DEB and w[1] == (5 << 4) | 3 for w in m._bus._writes)
    assert found
    # Release omitted defaults to touch
    m.debounce(touch=4, release=-1)
    found2 = any(w[0] == mpr121.DEB and w[1] == (4 << 4) | 4 for w in m._bus._writes)
    assert found2

def test_filter_writes_word(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    m.filter(cdc=20, cdt=3, ffi=2, sfi=1, esi=5)
    # Value should be composed as per filter()
    low_byte = (2 << 6) | 20
    high_byte = (3 << 5) | (1 << 3) | 5
    val = (high_byte << 8) | low_byte
    assert (mpr121.AFE1, val) in m._bus._writes

def test_charge_writes_bytes(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    # Setup read_byte to return 0xFF for CDT register
    m._bus._bytes[mpr121.CDT] = 0xFF
    m.charge(channel=2, cdc=12, cdt=3)
    # It should write to CDC + 2 register with 12
    assert (mpr121.CDC + 2, 12) in m._bus._writes
    # It should also write to CDT + 1 (2//2) with correct masking
    write_calls = [w for w in m._bus._writes if w[0] >= mpr121.CDT and w[0] < mpripherals.CDT + mpr121.NCH//2 + 1]
    assert len(write_calls) > 0

def test_auto_config_writes_correctly(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    m.auto_config(ace=1, are=0, bva=2, retry=1, afes=3, scts=1,
                  acfie=0, arfie=1, oorie=0, usl=150, lsl=100, tl=75)
    # Check a word write to ACNF_C0
    found = any(w[0] == mpr121.ACNF_C0 for w in m._bus._writes)
    assert found
    # Check bytes writes to USL, LSL, TL
    regs_written = [w[0] for w in m._bus._writes]
    assert mpr121.ACNF_USL in regs_written
    assert mpr121.ACNF_LSL in regs_written
    assert mpr121.ACNF_TL in regs_written

def test_out_of_range_raises(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    # Set bits to trigger exceptions
    m._bus._words[mpr121.OOR] = (1 << 15)  # auto_config_failed bit
    with pytest.raises(Exception, match='auto config failed'):
        m.out_of_range()
    # Clear bit 15, set bit 14
    m._bus._words[mpr121.OOR] = (1 << 14)
    with pytest.raises(Exception, match='auto reconfig failed'):
        m.out_of_range()
    # If no bits set, return lower bits masked
    m._bus._words[mpr121.OOR] = 0x1abc
    assert m.out_of_range() == 0x1abc

def test_gpio_setup_writes(monkeypatch, capsys):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    # Set _bus._bytes to control read_byte returns for GPIO_CTL*/DIR/EN
    def dummy_read_byte(reg):
        return 0
    m._bus.read_byte = dummy_read_byte
    # We should not raise for valid channel
    m.gpio_setup(4, output=1, mode=1, enable=1)
    # Test for assertion error on invalid channel
    with pytest.raises(AssertionError):
        m.gpio_setup(3, output=0)
    # Should call print - capture output
    out = capsys.readouterr().out
    assert '0x' in out

def test_gpio_status_returns_read_byte(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    m._bus._bytes[mpr121.GPIO_DAT] = 0xAB
    assert m.gpio_status() == 0xAB

def test_gpio_set_calls_write_byte(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    # Set GPIO set
    m.gpio_set(5, True)
    assert (mpr121.GPIO_SET, 1 << (5 - 4)) in m._bus._writes
    # Clear GPIO
    m.gpio_set(5, False)
    assert (mpr121.GPIO_CLR, 1 << (5 - 4)) in m._bus._writes
    # assertion error for invalid channels
    with pytest.raises(AssertionError):
        m.gpio_set(3, True)

def test_update_touch_state_no_change(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    m._touched = 0b1010
    monkeypatch.setattr(m, "touched", lambda raise_on_failure=1: 0b1010)
    # _handle_touch should not be called => no error
    m.update_touch_state()

def test_setup_calls_reset_and_configuration(monkeypatch):
    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    # patch reset/configure/filter/auto_config/threshold/debounce/baseline called flags
    calls = {}
    monkeypatch.setattr(m, "reset", lambda: calls.setdefault("reset", True))
    monkeypatch.setattr(m, "configure", lambda **kwargs: calls.setdefault("configure", True))
    monkeypatch.setattr(m, "filter", lambda **kwargs: calls.setdefault("filter", True))
    monkeypatch.setattr(m, "auto_config", lambda **kwargs: calls.setdefault("auto_config", True))
    monkeypatch.setattr(m, "threshold", lambda **kwargs: calls.setdefault("threshold", True))
    monkeypatch.setattr(m, "debounce", lambda **kwargs: calls.setdefault("debounce", True))
    monkeypatch.setattr(m, "baseline", lambda **kwargs: calls.setdefault(f"baseline_{kwargs.get('rft')}_{'prox' if kwargs.get('prox') else 'no_prox'}", True))
    m.setup(reset=True, channels=12, prox=0, threshold=50, debounce=2, auto_config=True)
    assert calls.get("reset") is True
    assert calls.get("configure") is True
    assert calls.get("filter") is True
    assert calls.get("auto_config") is True
    assert calls.get("threshold") is True
    assert calls.get("debounce") is True
    assert calls.get("baseline_0_no_prox") is True
    assert calls.get("baseline_1_no_prox") is True or calls.get("baseline_1_prox") is True

