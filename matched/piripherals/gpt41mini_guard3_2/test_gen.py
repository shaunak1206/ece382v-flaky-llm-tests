import builtins
import types
import pytest

from piripherals import mpr121

class DummyBusDevice:
    """Mock the bus device with controllable read/write behaviors."""
    def __init__(self):
        # Storage for registers: addr -> value or list of values
        self.reg_byte = {}
        self.reg_word = {}
        self.reg_block = {}
        self.calls = []  # record calls: tuples of (method, args)
        # Pre-fill some registers with fixed values for read tests
        # ETS (touch status) at 0x00: 16-bit, bits 0-12 touch
        self.reg_word[mpr121.ETS] = 0x0055  # arbitrary touch bits pattern, no error bits
        # OOR register (out of range)
        self.reg_word[mpr121.OOR] = 0x0000  # no errors
        # Baseline register block with values [0..12), baseline shifted by 2 in code
        self.reg_block[mpr121.EBL] = list(range(mpr121.NCH))
        # Electrode Filtered Data block (2 bytes per NCH)
        # We'll return 0, 1, 2, ... for low byte, and 0 for high byte for simplicity
        self.reg_block[mpr121.EFD] = [i & 0xFF if idx % 2 == 0 else 0 for idx, i in enumerate(range(2 * mpr121.NCH))]
        # Threshold touch (TTH) and release (RTH)
        self.reg_byte.update({mpr121.TTH + 2*i: 50 for i in range(mpr121.NCH)})
        self.reg_byte.update({mpr121.RTH + 2*i: 30 for i in range(mpr121.NCH)})
        # GPIO related registers: all zero initially
        self.reg_byte[mpr121.GPIO_DAT] = 0xAA  # 0b10101010 arbitrary pattern for testing gpio_status
        # Charge current CDC 
        self.reg_byte.update({mpr121.CDC + ch: 10 + ch for ch in range(mpr121.NCH)})
        # Charge time CDT registers: half byte for each channel (2 channels per byte)
        # Let's say all zero initially
        self.reg_byte.update({mpr121.CDT + i: 0 for i in range((mpr121.NCH + 1) // 2)})
        # Auto config registers
        self.reg_word[mpr121.ACNF_C0] = 0
        self.reg_byte[mpr121.ACNF_USL] = 0
        self.reg_byte[mpr121.ACNF_LSL] = 0
        self.reg_byte[mpr121.ACNF_TL] = 0
        # Debounce register
        self.reg_byte[mpr121.DEB] = 0

    def write_byte(self, addr, val):
        self.calls.append(('write_byte', addr, val))
        self.reg_byte[addr] = val

    def read_byte(self, addr):
        self.calls.append(('read_byte', addr))
        return self.reg_byte.get(addr, 0)

    def write_word(self, addr, val):
        self.calls.append(('write_word', addr, val))
        self.reg_word[addr] = val

    def read_word(self, addr):
        self.calls.append(('read_word', addr))
        return self.reg_word.get(addr, 0)

    def write_block(self, addr, values):
        self.calls.append(('write_block', addr, list(values)))
        self.reg_block[addr] = list(values)

    def read_block(self, addr, length):
        self.calls.append(('read_block', addr, length))
        blk = self.reg_block.get(addr)
        if blk is None:
            # Return a list of zeros for the requested length if unspecified
            return [0] * length
        # Return slice if length is less or equal, else zero pad
        if len(blk) >= length:
            return blk[:length]
        else:
            return blk + [0] * (length - len(blk))

# Dummy noop handler for direct comparison in tests
def dummy_handler(a, b=None):
    return (a, b)

def dummy_handler_one_arg(a):
    return (a,)

@pytest.fixture
def mpr():
    # Provide an MPR121 instance with dummy bus and no IRQ or Poller handlers to avoid side effects
    dummy_bus = DummyBusDevice()
    # monkeypatch bus.device to return dummy_bus
    class DummyBus:
        def device(self, addr):
            return dummy_bus
    # replace piripherals.mpr121.Bus with DummyBus to avoid accessing hardware
    return mpr121.MPR121(bus=DummyBus(), irq=0, handlers=0, setup=0, reset=0), dummy_bus

def test_init_default_calls_setup(monkeypatch):
    dummy_bus = DummyBusDevice()
    class DummyBus:
        def device(self, addr):
            return dummy_bus
    monkeypatch.setattr(mpr121, "Bus", lambda bus: DummyBus())
    # Patch setup to capture call args
    called = {}
    def fake_setup(self, **kwargs):
        called.update(kwargs)
    monkeypatch.setattr(mpr121.MPR121, "setup", fake_setup)
    m = mpr121.MPR121()
    assert called.get("reset", True) is True
    # Check _bus attribute is our dummy_bus
    assert m._bus is dummy_bus
    assert m.addr == 0x5a

def test_update_touch_state_calls_handle_touch(monkeypatch, mpr):
    m, dummy_bus = mpr
    # Setup _touched to 0, force touched() to return 0b101 (5)
    m._touched = 0
    # monkeypatch m.touched to return fixed value
    monkeypatch.setattr(m, "touched", lambda raise_on_failure=1: 0b101)
    # patch handler for channels 0,2
    called = []
    def handler(status, channel):
        called.append((status, channel))
    m.on_touch(0, handler)
    m.on_touch(2, handler)
    m.on_touch(1, None)  # no handler
    m.update_touch_state()
    # We expect calls for channel 0 and 2 changes
    assert (True, 0) in called
    assert (True, 2) in called
    # channel 1 not changed, so no call
    for call in called:
        assert call[1] in (0, 2)
    # Now update state again to same touched bits - no calls should happen
    called.clear()
    m.update_touch_state()
    assert called == []

def test_handle_touch_with_handler_one_arg(monkeypatch, mpr):
    m, dummy_bus = mpr
    called = []
    def handler_one_arg(status):
        called.append(status)
    m.on_touch(0, handler_one_arg)
    # Changed bit 0 only, touched bit 0 set to True
    m._handle_touch(touched=1, changed=1)
    assert called == [True]

def test_is_touched(mpr):
    m, dummy_bus = mpr
    m._touched = 0b101010
    for ch in range(mpr121.NCH):
        expected = (m._touched & (1 << ch)) > 0
        assert m.is_touched(ch) == expected

def test_on_touch_registers_and_unregisters_handler(mpr):
    m, dummy_bus = mpr
    def f1(status, ch): pass
    m.on_touch(0, f1)
    # The stored handler should not be f1 directly but wrapped by not_raising or noop
    assert callable(m._handlers[0])
    m.on_touch(0, None)
    # When None passed, handler resets to noop function
    assert m._handlers[0] == mpr121.noop

def test_reset_writes_reset_byte(mpr):
    m, dummy_bus = mpr
    m.reset()
    # Check that SRESET register is written with 0x63
    assert ("write_byte", mpr121.SRESET, 0x63) in dummy_bus.calls

def test_configure_writes_correct_register(mpr):
    m, dummy_bus = mpr
    m.configure(cl=1, prox=2, touch=3)
    # The value written to ECR is (cl<<6) | (prox<<4) | touch
    expected_value = (1 << 6) | (2 << 4) | 3
    assert ("write_byte", mpr121.ECR, expected_value) in dummy_bus.calls

def test_touched_returns_touch_bits_and_raises(monkeypatch, mpr):
    m, dummy_bus = mpr
    # Set word to a value with overcurrent bit 15 set
    dummy_bus.reg_word[mpr121.ETS] = 1 << 15 | 0x0345
    with pytest.raises(Exception, match='overcurrent'):
        m.touched(raise_on_failure=True)
    # Clear overcurrent but set out_of_range bit (read from out_of_range uses OOR register)
    dummy_bus.reg_word[mpr121.ETS] = 0x0345
    dummy_bus.reg_word[mpr121.OOR] = 1 << 15
    # monkeypatch out_of_range() to read dummy_bus.reg_word and raise
    with pytest.raises(Exception, match='out of range'):
        m.touched(raise_on_failure=True)
    # Test no exception and correct bits returned
    dummy_bus.reg_word[mpr121.OOR] = 0
    dummy_bus.reg_word[mpr121.ETS] = 0x1234
    result = m.touched(raise_on_failure=True)
    assert result == 0x1234 & 0x1fff

def test_electrode_data_returns_list_of_ints(mpr):
    m, dummy_bus = mpr
    # Setup read_block to return pairs of bytes: low byte = i, high byte = (i<<1)
    dummy_bus.reg_block[mpr121.EFD] = []
    for i in range(mpr121.NCH):
        dummy_bus.reg_block[mpr121.EFD].append(i & 0xFF)       # low byte
        dummy_bus.reg_block[mpr121.EFD].append((i << 1) & 0xFF) # high byte
    data = m.electrode_data()
    assert isinstance(data, list)
    assert len(data) == mpr121.NCH
    for i, val in enumerate(data):
        expected = (dummy_bus.reg_block[mpr121.EFD][2*i+1]<<8) + dummy_bus.reg_block[mpr121.EFD][2*i]
        assert val == expected

def test_baseline_without_args_returns_scaled_block(mpr):
    m, dummy_bus = mpr
    # baseline() called with no args returns baseline block shifted left 2 bits
    dummy_bus.reg_block[mpr121.EBL] = [1]*mpr121.NCH
    result = m.baseline()
    assert all(x == 4 for x in result)  # 1 << 2 == 4
    assert len(result) == mpr121.NCH

def test_baseline_with_args_writes_blocks(monkeypatch, mpr):
    m, dummy_bus = mpr
    # baseline with rft > 1 (e.g. 2) writes 3 bytes to a prox or non-prox reg
    reg_written = []
    def fake_write_block(addr, vals):
        reg_written.append((addr, list(vals)))
    monkeypatch.setattr(m._bus, "write_block", fake_write_block)
    m.baseline(rft=2, nhd=1, ncl=2, fdl=3, prox=0)
    assert reg_written
    reg, vals = reg_written[0]
    assert reg >= mpr121.MHD  # MHD or MHDX
    # baseline with rft 0-1 writes 4 bytes
    reg_written.clear()
    m.baseline(rft=1, mhd=5, nhd=6, ncl=7, fdl=8, prox=1)
    assert reg_written
    reg, vals = reg_written[0]
    # vals should be length 4 for rft 0 or 1
    assert len(vals) == 4

def test_threshold_sets_all_or_single_channels(monkeypatch, mpr):
    m, dummy_bus = mpr
    called = []
    def fake_write_byte(addr, val):
        called.append((addr, val))
    monkeypatch.setattr(m._bus, "write_byte", fake_write_byte)
    # call threshold for all channels
    m.threshold(touch=100)
    # Expect 2 writes per channel for touch and release
    assert all(isinstance(addr, int) and 0 <= val <= 255 for addr, val in called)
    assert len(called) >= mpr121.NCH * 2
    called.clear()
    # call threshold for single channel
    m.threshold(touch=50, release=30, channel=5)
    # Should produce exactly two writes (touch and release for channel 5)
    assert len(called) == 2
    assert called[0][0] == mpr121.TTH + 2 * 5
    assert called[1][0] == mpr121.RTH + 2 * 5
    # release threshold should be equal to given value
    assert called[0][1] == 50
    assert called[1][1] == 30

def test_debounce_writes_byte(monkeypatch, mpr):
    m, dummy_bus = mpr
    calls = []
    def fake_write_byte(addr, val):
        calls.append((addr, val))
    monkeypatch.setattr(m._bus, "write_byte", fake_write_byte)
    # release omitted → release == touch
    m.debounce(touch=3)
    assert calls
    addr, val = calls[-1]
    assert addr == mpr121.DEB
    # bits should be (release << 4) | touch meaning (3<<4) | 3 = 0x33 if release=touch=3
    assert val == (3 << 4) | 3
    calls.clear()
    m.debounce(touch=2, release=5)
    addr, val = calls[-1]
    assert val == (5 << 4) | 2

def test_filter_writes_word(monkeypatch, mpr):
    m, dummy_bus = mpr
    calls = []
    def fake_write_word(addr, val):
        calls.append((addr, val))
    monkeypatch.setattr(m._bus, "write_word", fake_write_word)
    m.filter(cdc=10, cdt=3, ffi=2, sfi=1, esi=4)
    lb = (2 << 6) | 10
    hb = (3 << 5) | (1 << 3) | 4
    assert ("write_word", mpr121.AFE1, (hb << 8) | lb) in calls

def test_charge_writes_bytes(monkeypatch, mpr):
    m, dummy_bus = mpr
    calls = []
    # We will track reads and writes to verify CDT transformation
    def fake_write_byte(addr, val):
        calls.append(('w', addr, val))
        dummy_bus.reg_byte[addr] = val
    def fake_read_byte(addr):
        calls.append(('r', addr))
        # return 0xFF for CDT registers to test masking
        return 0xFF
    monkeypatch.setattr(m._bus, "write_byte", fake_write_byte)
    monkeypatch.setattr(m._bus, "read_byte", fake_read_byte)
    # call charge for channel 3 with cdc=1 and cdt=2
    m.charge(3, cdc=1, cdt=2)
    # Expect one write_byte for CDC+channel and one for CDT+channel//2
    writes = [c for c in calls if c[0] == 'w']
    reads = [c for c in calls if c[0] == 'r']
    assert any(c[1] == mpr121.CDC + 3 for c in writes)
    cdt_reg = mpr121.CDT + 3 // 2
    assert any(c[1] == cdt_reg for c in writes)
    assert any(c[1] == cdt_reg for c in reads)
    # The CDT written value should have only 4 bits (nibble) modified at pos s = 4*(channel%2)
    modified = None
    for w in writes:
        if w[1] == cdt_reg:
            modified = w[2]
            break
    assert modified is not None
    # The bits at position s=4*(3%2=1) = 4 must be equal to dct shifted
    s = 4 * (3 % 2)
    # Mask is 0xf shifted left by s
    mask = 0xf << s
    # Extract bits in modified at that mask
    bits = (modified & mask) >> s
    assert bits == 2

def test_auto_config_writes_correct_regs(monkeypatch, mpr):
    m, dummy_bus = mpr
    written_words = []
    written_bytes = []
    def fake_write_word(addr, val):
        written_words.append((addr, val))
    def fake_write_byte(addr, val):
        written_bytes.append((addr, val))
    monkeypatch.setattr(m._bus, "write_word", fake_write_word)
    monkeypatch.setattr(m._bus, "write_byte", fake_write_byte)
    m.auto_config(ace=1, are=0, bva=2, retry=3, afes=2, scts=1, acfie=0, arfie=1, oorie=0,
                  usl=201, lsl=131, tl=181)
    assert any(w[0] == mpr121.ACNF_C0 for w in written_words)
    # Check that some bytes written to USL, LSL, TL
    addrs = [b[0] for b in written_bytes]
    assert mpr121.ACNF_USL in addrs
    assert mpr121.ACNF_LSL in addrs
    assert mpr121.ACNF_TL in addrs

def test_out_of_range_raises(monkeypatch, mpr):
    m, dummy_bus = mpr
    dummy_bus.reg_word[mpr121.OOR] = 1 << 15  # auto config failed bit set
    with pytest.raises(Exception, match="auto config failed"):
        m.out_of_range(raise_on_failure=1)

    dummy_bus.reg_word[mpr121.OOR] = 1 << 14  # auto reconfig failed
    with pytest.raises(Exception, match="auto reconfig failed"):
        m.out_of_range(raise_on_failure=1)

    dummy_bus.reg_word[mpr121.OOR] = 0x0  # no error
    val = m.out_of_range(raise_on_failure=1)
    assert val == (dummy_bus.reg_word[mpr121.OOR] & 0x1fff)

def test_gpio_setup_bits_set(monkeypatch, mpr):
    m, dummy_bus = mpr
    # Set initial dummy read_byte values for GPIO_CTL0, GPIO_CTL1, GPIO_DIR, GPIO_EN
    dummy_bus.reg_byte[mpr121.GPIO_CTL0] = 0b00000000
    dummy_bus.reg_byte[mpr121.GPIO_CTL1] = 0b00000000
    dummy_bus.reg_byte[mpr121.GPIO_DIR] = 0b00000000
    dummy_bus.reg_byte[mpr121.GPIO_EN] = 0b00000000
    written_calls = []
    def fake_write_byte(addr, val):
        written_calls.append((addr, val))
        dummy_bus.reg_byte[addr] = val
    monkeypatch.setattr(m._bus, "write_byte", fake_write_byte)
    monkeypatch.setattr(m._bus, "read_byte", lambda addr: dummy_bus.reg_byte.get(addr, 0))
    # Test pin 6 (channel=6) with output True, mode=3 (open drain high side), enable True
    m.gpio_setup(channel=6, output=1, mode=3, enable=1)
    # We expect four writes (CTL0, CTL1, DIR, EN)
    addrs = [c[0] for c in written_calls]
    expected_addrs = [mpr121.GPIO_CTL0, mpr121.GPIO_CTL1, mpr121.GPIO_DIR, mpr121.GPIO_EN]
    for ea in expected_addrs:
        assert ea in addrs
    # The bits written for channel 6 (channel-4=2) in CTL0 and CTL1 are correct
    ctl0_val = next(val for (a, val) in written_calls if a == mpr121.GPIO_CTL0)
    ctl1_val = next(val for (a, val) in written_calls if a == mpr121.GPIO_CTL1)
    # mode=3 is binary 11, ctl0 bit = mode&0b10 -> bit1, ctl1 bit=mode&0b01 -> bit0
    # channel=2 so bits at pos 2
    assert ((ctl0_val >> 2) & 1) == 1  # bit for ctl0 at channel
    assert ((ctl1_val >> 2) & 1) == 1  # bit for ctl1 at channel
    # DIR and EN bits written similarly
    dir_val = next(val for (a, val) in written_calls if a == mpr121.GPIO_DIR)
    en_val = next(val for (a, val) in written_calls if a == mpr121.GPIO_EN)
    assert ((dir_val >> 2) & 1) == 1
    assert ((en_val >> 2) & 1) == 1

def test_gpio_status_returns_byte(mpr):
    m, dummy_bus = mpr
    val = m.gpio_status()
    assert val == dummy_bus.reg_byte[mpr121.GPIO_DAT]
    # Also test that the call to read_byte is registered
    assert ('read_byte', mpr121.GPIO_DAT) in dummy_bus.calls

def test_gpio_set_writes_correct_register(monkeypatch, mpr):
    m, dummy_bus = mpr
    written = []
    monkeypatch.setattr(m._bus, "write_byte", lambda addr, val: written.append((addr, val)))
    # Set channel 7 to True => write_byte to GPIO_SET, bit (7-4)=3
    m.gpio_set(7, True)
    addr, val = written[-1]
    assert addr == mpr121.GPIO_SET
    assert val == 1 << (7 - 4)
    # Set channel 7 to False => write_byte to GPIO_CLR, bit (7-4)=3
    m.gpio_set(7, False)
    addr, val = written[-1]
    assert addr == mpr121.GPIO_CLR
    assert val == 1 << (7 - 4)

def test_setup_calls_methods_correctly(monkeypatch):
    dummy_bus = DummyBusDevice()
    class DummyBus:
        def device(self, addr):
            return dummy_bus
    monkeypatch.setattr(mpr121, "Bus", lambda bus: DummyBus())
    calls = []
    def fake_reset(self):
        calls.append("reset")
    def fake_configure(self, **kwargs):
        calls.append(("configure", kwargs))
    def fake_filter(self, **kwargs):
        calls.append(("filter", kwargs))
    def fake_auto_config(self, **kwargs):
        calls.append("auto_config")
    def fake_threshold(self, **kwargs):
        calls.append(("threshold", kwargs))
    def fake_debounce(self, **kwargs):
        calls.append(("debounce", kwargs))
    def fake_baseline(self, **kwargs):
        calls.append(("baseline", kwargs))
    m = mpr121.MPR121(bus=1, irq=0, handlers=0, setup=0, reset=0)
    monkeypatch.setattr(m, "reset", fake_reset)
    monkeypatch.setattr(m, "configure", fake_configure)
    monkeypatch.setattr(m, "filter", fake_filter)
    monkeypatch.setattr(m, "auto_config", fake_auto_config)
    monkeypatch.setattr(m, "threshold", fake_threshold)
    monkeypatch.setattr(m, "debounce", fake_debounce)
    monkeypatch.setattr(m, "baseline", fake_baseline)
    m.setup(reset=True, channels=12, prox=0, threshold=50, debounce=2, auto_config=True)
    # We expect calls:
    assert "reset" in calls
    assert any(c[0] == "configure" for c in calls)
    assert any(c[0] == "filter" for c in calls)
    assert "auto_config" in calls
    assert any(c[0] == "threshold" for c in calls)
    assert any(c[0] == "debounce" for c in calls)
    # baseline called multiple times (loop twice)
    baseline_calls = [c for c in calls if c[0] == "baseline"]
    assert len(baseline_calls) >= 2


@pytest.mark.parametrize("channel", [0, 5, 12])
def test_on_touch_accepts_and_normalizes(monkeypatch, channel):
    dummy_bus = DummyBusDevice()
    class DummyBus:
        def device(self, addr):
            return dummy_bus
    monkeypatch.setattr(mpr121, "Bus", lambda bus: DummyBus())
    m = mpr121.MPR121(bus=1, irq=0, handlers=0, setup=0, reset=0)
    # register handler with two params
    called = []
    def handler(status, ch):
        called.append((status, ch))
    m.on_touch(channel, handler)
    # forcibly call _handle_touch to invoke this handler
    changed = 1 << channel
    touched = changed
    m._handle_touch(touched, changed)
    assert called == [(True, channel)]
    called.clear()
    # register handler with one param
    def handler1param(status):
        called.append((status,))
    m.on_touch(channel, handler1param)
    m._handle_touch(0, changed)
    assert called == [(False,)]

def test_dump_runs_and_prints(monkeypatch, capsys, mpr):
    m, dummy_bus = mpr
    # monkeypatch touches, out_of_range, gpio_status, electrode_data, baseline to fixed values
    monkeypatch.setattr(m, "touched", lambda raise_on_failure=1: 0x1FFF)
    monkeypatch.setattr(m, "out_of_range", lambda raise_on_failure=1: 0)
    monkeypatch.setattr(m, "gpio_status", lambda: 0xFF)
    monkeypatch.setattr(m, "electrode_data", lambda: [1023] * mpr121.NCH)
    monkeypatch.setattr(m, "baseline", lambda: [512] * mpr121.NCH)
    # monkeypatch _bus.read_byte to return fixed thresholds (50)
    monkeypatch.setattr(m._bus, "read_byte", lambda addr: 50)
    # call dump with no loop (loop=1)
    m.dump(regs=False, up=False, loop=1)
    out = capsys.readouterr().out
    assert " E:" in out
    # Should print line for each electrode
    for i in range(mpr121.NCH):
        assert f"{i}:" in out

