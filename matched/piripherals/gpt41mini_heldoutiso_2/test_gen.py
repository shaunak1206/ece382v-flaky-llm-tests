import pytest
from piripherals import mpr121
from types import SimpleNamespace

# Patch the .bus.Bus and IRQHandler, Poller to avoid hardware usage
class DummyBusDevice:
    def __init__(self):
        self.writes = []
        self.reads = {}
        self.read_word_values = {}
        self.read_block_values = {}
        self.read_byte_values = {}
    def write_byte(self, reg, val):
        self.writes.append((reg, val))
    def write_word(self, reg, val):
        self.writes.append((reg, val))
    def write_block(self, reg, data):
        self.writes.append((reg, tuple(data)))
    def read_word(self, reg):
        return self.read_word_values.get(reg, 0)
    def read_block(self, reg, length):
        vals = self.read_block_values.get(reg, [0]*length)
        if len(vals) < length:
            vals = vals + [0]*(length - len(vals))
        return vals[:length]
    def read_byte(self, reg):
        return self.read_byte_values.get(reg, 0)

class DummyBus:
    def device(self, addr):
        self.addr = addr
        self.device_obj = DummyBusDevice()
        return self.device_obj

def dummy_irqhandler(irq, callback):
    # Just store into list for testing called with correct callback?
    dummy_irqhandler.last_irq = irq
    dummy_irqhandler.last_callback = callback
    return SimpleNamespace(close=lambda: None)
def dummy_poller(callback):
    dummy_poller.last_callback = callback
    return SimpleNamespace(close=lambda: None)

mpr121.noop = lambda *a, **k: None
mpr121.not_raising = lambda h: h if h else mpr121.noop

# Patch IRQHandler and Poller globally for all tests
mpr121.IRQHandler = dummy_irqhandler
mpr121.Poller = dummy_poller

@pytest.fixture
def mpr():
    bus = DummyBus()
    m = mpr121.MPR121(bus=bus, irq=0, handlers=0, setup=0, reset=0)
    m._bus = bus.device_obj
    return m

def test_init_with_bus_object_calls_device():
    bus = DummyBus()
    m = mpr121.MPR121(bus=bus, irq=0, handlers=0, setup=0, reset=0)
    assert m._bus is bus.device_obj
    assert m.addr == 0x5a

def test_init_with_bus_int_creates_bus_and_device(monkeypatch):
    monkeypatch.setattr(mpr121, 'Bus', lambda busnum: DummyBus())
    m = mpr121.MPR121(bus=5, irq=0, handlers=0, setup=0, reset=0)
    assert hasattr(m._bus, 'write_byte')

def test_irq_handler_used_when_irq_and_handlers(monkeypatch):
    called = {}
    def fake_irq(irq, callback):
        called['irq'] = irq
        called['callback'] = callback
        return SimpleNamespace()
    monkeypatch.setattr(mpr121, 'IRQHandler', fake_irq)
    m = mpr121.MPR121(irq=7, handlers=1, setup=0, reset=0)
    assert called['irq'] == 7
    assert callable(called['callback'])

def test_poller_used_when_no_irq_and_handlers(monkeypatch):
    called = {}
    def fake_poll(callback):
        called['callback'] = callback
        return SimpleNamespace()
    monkeypatch.setattr(mpr121, 'Poller', fake_poll)
    m = mpr121.MPR121(irq=0, handlers=1, setup=0, reset=0)
    assert callable(called['callback'])

def test_no_handlers_if_handlers_flag_false(monkeypatch):
    monkeypatch.setattr(mpr121, 'IRQHandler', dummy_irqhandler)
    monkeypatch.setattr(mpr121, 'Poller', dummy_poller)
    m = mpr121.MPR121(irq=0, handlers=0, setup=0, reset=0)
    # No IRQHandler or Poller called
    assert not hasattr(dummy_irqhandler, 'last_irq') or dummy_irqhandler.last_irq != 0
    assert not hasattr(dummy_poller, 'last_callback')

def test_reset_calls_write_byte(mpr):
    mpr.reset()
    assert (mpr121.SRESET, 0x63) in mpr._bus.writes

def test_configure_writes_expected_byte(mpr):
    mpr.configure(cl=2, prox=1, touch=3)
    expected = (2 << 6) | (1 << 4) | 3
    assert (mpr121.ECR, expected) in mpr._bus.writes

def test_touched_returns_correct_bits_and_raises_on_overcurrent(mpr):
    # Set read_word to simulate touched bits, no exceptions
    mpr._bus.read_word_values[mpr121.ETS] = 0b1010101
    mpr._bus.read_word_values[mpr121.OOR] = 0
    # patch out_of_range not to raise
    mpr.out_of_range = lambda raise_on_failure=1: False
    val = mpr.touched()
    assert val == 0b1010101
    # Test raises for overcurrent
    mpr._bus.read_word_values[mpr121.ETS] = (1 << 15)  # overcurrent bit set
    with pytest.raises(Exception, match='overcurrent'):
        mpr.touched()

def test_touched_raises_if_out_of_range(mpr):
    mpr._bus.read_word_values[mpr121.ETS] = 0
    mpr.out_of_range = lambda raise_on_failure=1: (_ for _ in ()).throw(Exception('out of range'))
    with pytest.raises(Exception, match='out of range'):
        mpr.touched()

def test_electrode_data_returns_correct_length_and_values(mpr):
    # Provide 2*NCH bytes representing 16-bit little endian values
    example_data = []
    for i in range(mpr121.NCH):
        example_data.extend([i & 0xff, (i >> 8) & 0xff])
    mpr._bus.read_block_values[mpr121.EFD] = example_data
    data = mpr.electrode_data()
    assert len(data) == mpr121.NCH
    # Values reconstructed as low byte + (high byte << 8)
    for i, val in enumerate(data):
        assert val == i

def test_baseline_return_values_and_write_block_calls(mpr):
    # Setup baseline register read returns 13 bytes with known values
    mpr._bus.read_block_values[mpr121.EBL] = list(range(mpr121.NCH))
    # Calling without args returns baseline shifted left 2 bits
    baseline_vals = mpr.baseline()
    for i, val in enumerate(baseline_vals):
        assert val == i << 2
    # Calling with rft=2 triggers write_block prox=False
    mpr._bus.writes.clear()
    mpr.baseline(rft=2, mhd=3, nhd=4, ncl=5, fdl=6, prox=False)
    assert (mpr121.MHD + 8, (3, 4, 5, 6)) not in mpr._bus.writes  # Should not write 4 bytes because rft>1 expects 3 bytes
    # Actually it writes 3 bytes in this condition - so check write_block call
    assert mpr._bus.writes and mpr._bus.writes[-1][0] == mpr121.MHD + 8
    assert tuple(mpr._bus.writes[-1][1]) == (4, 5, 6)
    # Calling with rft=0 prox=True triggers write_block at MHDX + 0
    mpr._bus.writes.clear()
    mpr.baseline(rft=0, mhd=7, nhd=8, ncl=9, fdl=10, prox=True)
    assert mpr._bus.writes and mpr._bus.writes[-1][0] == mpr121.MHDX + 0
    assert tuple(mpr._bus.writes[-1][1]) == (7, 8, 9, 10)

def test_threshold_sets_all_channels_when_channel_minus_one(mpr):
    mpr._bus.writes.clear()
    mpr.threshold(100)
    # For all channels writes happen
    expected_count = mpr121.NCH * 2  # each channel writes 2 bytes
    # Each threshold() calls write_byte twice per channel
    assert len(mpr._bus.writes) >= expected_count
    # Verify first write is touch threshold, second is release threshold
    for i in range(mpr121.NCH):
        tth_reg = mpr121.TTH + 2*i
        rth_reg = mpr121.RTH + 2*i
        found_touch = any(w[0] == tth_reg and w[1] == 100 for w in mpr._bus.writes)
        found_release = any(w[0] == rth_reg and w[1] == int(0.6 * 100) for w in mpr._bus.writes)
        assert found_touch is True
        assert found_release is True

def test_threshold_sets_specific_channel(mpr):
    mpr._bus.writes.clear()
    mpr.threshold(70, 30, 5)
    found_touch = (mpr121.TTH + 2*5, 70) in mpr._bus.writes
    found_release = (mpr121.RTH + 2*5, 30) in mpr._bus.writes
    assert found_touch and found_release

def test_debounce_sets_default_release(mpr):
    mpr._bus.writes.clear()
    mpr.debounce(3)
    # release should be touch default, so 3
    val = (3 << 4) | 3
    found = (mpr121.DEB, val) in mpr._bus.writes
    assert found

def test_debounce_sets_release_if_specified(mpr):
    mpr._bus.writes.clear()
    mpr.debounce(1, 2)
    val = (2 << 4) | 1
    assert (mpr121.DEB, val) in mpr._bus.writes

def test_filter_writes_correct_word(mpr):
    mpr._bus.writes.clear()
    mpr.filter(cdc=21, cdt=3, ffi=2, sfi=1, esi=4)
    # calculate expected low byte and high byte per filter()
    lb = (2 << 6) | 21
    hb = (3 << 5) | (1 << 3) | 4
    expected_val = (hb << 8) | lb
    found = (mpr121.AFE1, expected_val) in mpr._bus.writes
    assert found

def test_charge_writes_bytes_correctly(mpr):
    mpr._bus.read_byte_values[mpr121.CDT + 1] = 0b10101010
    mpr._bus.writes.clear()
    mpr.charge(3, cdc=15, cdt=7)
    # cdc write
    assert (mpr121.CDC + 3, 15) in mpr._bus.writes
    # read_byte called for charge time register
    # There should be a write byte with cdt value set in correct bits
    writes = [w for w in mpr._bus.writes if w[0] == mpr121.CDT + 1]
    assert writes
    val_written = writes[-1][1]
    # bits corresponding to cdt are bits 4-7 for channel 3 (third channel)
    # channel 3 means index 3//2=1, shift: 4*(3%2)=4 bits, so bits 4-7
    # So val_written should have cdt shifted left 4 bits and merged with low bits 0b1010
    cdt_shifted = 7 << 4
    other_bits_mask = ~(0xf << 4) & 0xff
    expected_val = (0b10101010 & other_bits_mask) | cdt_shifted
    assert val_written == expected_val

def test_auto_config_writes_word_and_bytes(mpr):
    mpr._bus.writes.clear()
    mpr.auto_config(ace=1, are=0, bva=2, retry=3, afes=1, scts=1, acfie=0, arfie=1, oorie=1, usl=255, lsl=128, tl=64)
    # Compose lb and hb according to auto_config
    lb = (1 << 6) | (3 << 4) | (2 << 2) | (0 << 1) | 1
    hb = (1 << 7) | (1 << 2) | (1 << 1) | 0
    expected_word = (hb << 8) + lb
    # Check that write_word and bytes are called with expected values
    assert (mpr121.ACNF_C0, expected_word) in mpr._bus.writes
    assert (mpr121.ACNF_USL, 255) in mpr._bus.writes
    assert (mpr121.ACNF_LSL, 128) in mpr._bus.writes
    assert (mpr121.ACNF_TL, 64) in mpr._bus.writes

def test_out_of_range_returns_bits_and_raises_properly(mpr):
    # No failure bits set
    mpr._bus.read_word_values[mpr121.OOR] = 0b1010101010101
    val = mpr.out_of_range()
    assert val == (0b1010101010101 & 0x1fff)
    # Set auto config failed bit 15
    mpr._bus.read_word_values[mpr121.OOR] = (1 << 15)
    with pytest.raises(Exception, match='auto config failed'):
        mpr.out_of_range()
    # Set auto reconfig failed bit 14
    mpr._bus.read_word_values[mpr121.OOR] = (1 << 14)
    # Must reset to no bit 15 to test this separately
    mpr._bus.read_word_values[mpr121.OOR] = (1 << 14)
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr.out_of_range()

def test_gpio_setup_writes_bytes(mpr):
    # Patch _bus.read_byte to return 0xff always (to see bits cleared)
    mpr._bus.read_byte_values[mpr121.GPIO_CTL0] = 0b11111111
    mpr._bus.read_byte_values[mpr121.GPIO_CTL1] = 0b11111111
    mpr._bus.read_byte_values[mpr121.GPIO_DIR] = 0b11111111
    mpr._bus.read_byte_values[mpr121.GPIO_EN] = 0b11111111
    mpr._bus.writes.clear()
    mpr.gpio_setup(channel=6, output=1, mode=3, enable=1)
    # Check writes to the 4 registers for channel 6-4=2 bits
    reg_addrs = [mpr121.GPIO_CTL0, mpr121.GPIO_CTL1, mpr121.GPIO_DIR, mpr121.GPIO_EN]
    assert any(w[0] == mpr121.GPIO_CTL0 for w in mpr._bus.writes)
    assert any(w[0] == mpr121.GPIO_CTL1 for w in mpr._bus.writes)
    assert any(w[0] == mpr121.GPIO_DIR for w in mpr._bus.writes)
    assert any(w[0] == mpr121.GPIO_EN for w in mpr._bus.writes)

def test_gpio_setup_mode_1_treated_as_0(mpr):
    # mode=1 should be treated as 0
    mpr._bus.read_byte_values[mpr121.GPIO_CTL0] = 0
    mpr._bus.read_byte_values[mpr121.GPIO_CTL1] = 0
    mpr._bus.read_byte_values[mpr121.GPIO_DIR] = 0
    mpr._bus.read_byte_values[mpr121.GPIO_EN] = 0
    mpr._bus.writes.clear()
    mpr.gpio_setup(channel=4, output=0, mode=1, enable=1)
    # Extract value written to GPIO_CTL0 and GPIO_CTL1 for bit 0
    ctl0_write = next(w[1] for w in mpr._bus.writes if w[0] == mpr121.GPIO_CTL0)
    ctl1_write = next(w[1] for w in mpr._bus.writes if w[0] == mpr121.GPIO_CTL1)
    # Mode 1 treated as 0 means both bits cleared (0b00), so bits 1 & 0 (for channel 0) should be 0
    assert (ctl0_write & 0b1) == 0
    assert (ctl1_write & 0b1) == 0

def test_gpio_status_reads_byte(mpr):
    mpr._bus.read_byte_values[mpr121.GPIO_DAT] = 0x5a
    status = mpr.gpio_status()
    assert status == 0x5a

def test_gpio_set_writes_correct_register_and_value(mpr):
    mpr._bus.writes.clear()
    # set True writes to GPIO_SET, bit 2 (channel 6 - 4 = 2)
    mpr.gpio_set(6, True)
    assert (mpr121.GPIO_SET, 1 << 2) in mpr._bus.writes or (mpr121.GPIO_SET, 4) in mpr._bus.writes
    mpr._bus.writes.clear()
    # set False writes to GPIO_CLR
    mpr.gpio_set(7, False)
    assert (mpr121.GPIO_CLR, 1 << 3) in mpr._bus.writes or (mpr121.GPIO_CLR, 8) in mpr._bus.writes

def test_is_touched_returns_correct_bool(mpr):
    mpr._touched = 0b1010  # bits 3 and 1 set
    assert mpr.is_touched(1) is True
    assert mpr.is_touched(2) is False

def test_on_touch_registers_and_removes_handler(mpr):
    called = {}
    def handler1(state, ch):
        called['called'] = (state, ch)
    mpr.on_touch(3, handler1)
    # Replace the handler with None to remove
    mpr.on_touch(3, None)
    # Provide invalid handler that raises TypeError with two params
    def bad_handler(state, ch):
        raise TypeError
    # Should fallback to calling handler with one param
    called2 = {}
    def bad_handler_one(state):
        called2['called'] = state
    mpr.on_touch(2, bad_handler_one)
    # Trigger _handle_touch to call handler
    mpr._handlers[2] = bad_handler_one
    # Emulate a touch changed on channel 2
    mpr._handle_touch(touched=0b100, changed=0b100)
    assert called2['called'] is True

def test_update_touch_state_calls_handle_touch(mpr):
    called = {}
    def handler(touched, changed):
        called['touched'] = touched
        called['changed'] = changed
    mpr._handlers[0] = handler
    mpr._touched = 0
    # Patch touched to simulate channel 0 touched now
    mpr.touched = lambda raise_on_failure=1: 0b1
    mpr._handle_touch = lambda touched, changed: called.update({'touched': touched, 'changed': changed})
    mpr.update_touch_state()
    assert called['touched'] == 0b1
    assert called['changed'] == 0b1
