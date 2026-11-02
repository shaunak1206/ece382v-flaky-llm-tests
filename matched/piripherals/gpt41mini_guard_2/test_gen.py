import pytest
from piripherals.mpr121 import MPR121, NCH, ETS, OOR, EFD, EBL, MHD, MHDX, TTH, RTH, DEB, AFE1, CDC, CDT, ACNF_C0, ACNF_USL, ACNF_LSL, ACNF_TL, GPIO_CTL0, GPIO_CTL1, GPIO_DIR, GPIO_EN, GPIO_SET, GPIO_CLR
from types import SimpleNamespace

@pytest.fixture
def bus_mock():
    # Mock bus.device(addr) returns an object with read_byte/read_word/read_block/write_byte/write_word/write_block methods
    class DeviceMock:
        def __init__(self):
            self.written = {}
            self.byte_registers = {}
            self.word_registers = {}
            self.block_registers = {}
            self.read_callbacks = {}

        def write_byte(self, reg, val):
            self.written[(reg, 'byte')] = val
            self.byte_registers[reg] = val

        def write_word(self, reg, val):
            self.written[(reg, 'word')] = val
            self.word_registers[reg] = val

        def write_block(self, reg, vals):
            self.written[(reg, 'block')] = tuple(vals)
            self.block_registers[reg] = tuple(vals)

        def read_byte(self, reg):
            # Return previously written byte or 0
            return self.byte_registers.get(reg, 0)

        def read_word(self, reg):
            return self.word_registers.get(reg, 0)

        def read_block(self, reg, length):
            # Return list of zeros if no block data preset
            data = self.block_registers.get(reg)
            if data is None:
                return [0] * length
            # Return requested length if enough data available, else pad with zeros
            if len(data) >= length:
                return list(data[:length])
            else:
                return list(data) + [0] * (length - len(data))

    class BusMock:
        def device(self, addr):
            return DeviceMock()

    return BusMock()

@pytest.fixture
def mpr121(bus_mock):
    # Construct MPR121 with mocked bus, disable irq and handlers to avoid side effects
    return MPR121(bus=bus_mock, irq=0, handlers=0, setup=0, reset=0)

def test_reset_writes_correct_value(mpr121):
    mpr121.reset()
    assert mpr121._bus.byte_registers.get(0x80) == 0x63

def test_configure_writes_correct_byte(mpr121):
    mpr121.configure(cl=2, prox=1, touch=5)
    val = mpr121._bus.byte_registers.get(0x5e)  # ECR
    assert val == (2 << 6) | (1 << 4) | 5

def test_touched_returns_correct_bits(mpr121):
    # prepare the word to return: bits 0-12 arbitrary, ensure no error bits
    expected_bits = 0x0abc
    mpr121._bus.read_word = lambda reg: expected_bits
    touched = mpr121.touched(raise_on_failure=False)
    assert touched == expected_bits & 0x1FFF

def test_touched_raises_on_overcurrent(mpr121):
    # Set bit 15 to 1 to simulate overcurrent
    mpr121._bus.read_word = lambda reg: 1 << 15
    with pytest.raises(Exception) as e:
        mpr121.touched()
    assert 'overcurrent' in str(e.value)

def test_touched_raises_on_out_of_range(mpr121):
    # out_of_range() is called by touched, so we patch that to raise
    def oor_raise():
        raise Exception('out of range')
    mpr121._bus.read_word = lambda reg: 0  # no overcurrent
    mpr121.out_of_range = lambda raise_on_failure=1: (_ for _ in ()).throw(Exception('out of range'))
    with pytest.raises(Exception) as e:
        mpr121.touched()
    assert 'out of range' in str(e.value)

def test_electrode_data_returns_correct_format(mpr121):
    # Return block data of 2*NCH bytes with pattern: low byte i, high byte i+1
    block_data = []
    for i in range(NCH):
        block_data.append(i & 0xFF)      # low byte
        block_data.append((i + 1) & 0xFF)  # high byte
    mpr121._bus.read_block = lambda reg, length: block_data.copy()
    data = mpr121.electrode_data()
    assert len(data) == NCH
    for i, val in enumerate(data):
        expected = ((i + 1) << 8) + (i & 0xFF)
        assert val == expected

def test_baseline_reads_when_rft_negative(mpr121):
    # populate block with baseline data (NCH bytes)
    baseline_bytes = list(range(NCH))
    mpr121._bus.read_block = lambda reg, length: baseline_bytes.copy()
    result = mpr121.baseline()
    assert result == [b << 2 for b in baseline_bytes]

@pytest.mark.parametrize("rft", [0,1])
def test_baseline_write_calls_write_block_4vals(mpr121, rft):
    calls = []
    def write_block_mock(reg, vals):
        calls.append((reg, tuple(vals)))
    mpr121._bus.write_block = write_block_mock
    mpr121.baseline(rft=rft, mhd=7, nhd=8, ncl=9, fdl=10, prox=False)
    # For rft 0 or 1, 4 vals written
    assert calls
    reg, vals = calls[0]
    assert reg == MHD + 4*rft
    assert vals == (7, 8, 9, 10)

def test_baseline_write_calls_write_block_3vals_prox(mpr121):
    calls = []
    def write_block_mock(reg, vals):
        calls.append((reg, tuple(vals)))
    mpr121._bus.write_block = write_block_mock
    mpr121.baseline(rft=2, nhd=1, ncl=2, fdl=3, prox=True)
    assert calls
    reg, vals = calls[0]
    assert reg == MHDX + 8
    assert vals == (1, 2, 3)

def test_threshold_sets_all_channels_by_default(mpr121):
    called = []
    orig_write = mpr121._bus.write_byte
    def write_byte_mock(reg, val):
        called.append((reg, val))
        orig_write(reg, val)
    mpr121._bus.write_byte = write_byte_mock
    mpr121.threshold(touch=30)
    # It should call threshold recursively for channels 0..12 (13 calls * 2 = 26 writes)
    # But our mock just logs calls
    regs = [reg for reg, _ in called]
    values = [val for _, val in called]
    # There should be writes to registers TTH, TTH+2,... and RTH, RTH+2,...
    # Check that every channel is covered
    channels_set = set()
    for reg in regs:
        if (reg - TTH) % 2 == 0:
            channel = (reg - TTH) // 2
            channels_set.add(channel)
    assert channels_set == set(range(NCH))
    # Check release values set properly with release=0.6*touch (integer)
    release_expected = int(0.6 * 30)
    # RTH registers start at RTH = 0x42
    rth_vals = [val for reg, val in called if reg >= RTH and (reg - RTH) % 2 == 0]
    assert all(v == release_expected for v in rth_vals if v is not None)

def test_threshold_sets_specific_channel(mpr121):
    writes = []
    mpr121._bus.write_byte = lambda reg, val: writes.append((reg, val))
    mpr121.threshold(touch=40, release=20, channel=5)
    assert (TTH + 2*5, 40) in writes
    assert (RTH + 2*5, 20) in writes
    # No other channels written
    other_channels = [ch for reg, val in writes for ch in [(reg - TTH) // 2, (reg - RTH) // 2] if 0 <= ch < NCH and ch != 5]
    assert other_channels == []

def test_debounce_writes_combined_byte(mpr121):
    writes = []
    mpr121._bus.write_byte = lambda reg, val: writes.append((reg, val))
    mpr121.debounce(touch=3, release=-1)
    # release defaults to touch
    assert writes[-1] == (DEB, (3 << 4) | 3)
    mpr121.debounce(touch=2, release=7)
    assert writes[-1] == (DEB, (7 << 4) | 2)

def test_filter_writes_word_correctly(mpr121):
    writes = []
    mpr121._bus.write_word = lambda reg, val: writes.append((reg,val))
    mpr121._bus.write_byte = lambda r,v: None  # ignore writes in this test
    mpr121.filter(cdc=20, cdt=3, ffi=2, sfi=1, esi=4)
    # lb = (ffi<<6)|cdc = (2<<6)|20 = 128+20=148=0x94
    # hb = (cdt<<5)|(sfi<<3)|esi = (3<<5)|(1<<3)|4=96+8+4=108=0x6C
    val_expected = (108 << 8) | 148
    assert writes[-1] == (AFE1, val_expected)

def test_charge_writes_bytes_properly(mpr121):
    calls = []
    dummy_reg_values = {CDT: 0b00000000, CDT+1: 0b11110000}

    def read_byte(reg):
        return dummy_reg_values.get(reg, 0)

    def write_byte(reg, val):
        calls.append((reg, val))
        dummy_reg_values[reg] = val

    mpr121._bus.read_byte = read_byte
    mpr121._bus.write_byte = write_byte

    mpr121.charge(channel=0, cdc=5, cdt=3)
    # For channel 0:
    # write CDC+0 = 5
    # write CDT + 0 = update bits 4 * (0%2) = 0
    assert (CDC + 0, 5) in calls
    # read old val is 0, update with (3 << 0) & ~(0xf << 0)
    # so final byte = (0 & ~0xf) | 3 = 3
    assert (CDT, 3) in calls

    mpr121.charge(channel=3, cdc=6, cdt=7)
    # CDC + 3 = 6
    # CDT + 1 (3//2=1), bit shift 4*(3%2)=4
    # old val read 0xf0, clear 0xf <<4 = 0xf0 mask, so new = (0xf0 & ~0xf0) | (7<<4) = 0x70
    assert (CDC + 3, 6) in calls
    assert (CDT + 1, 0x70) in calls

def test_auto_config_writes_regs_and_bytes(mpr121):
    calls_word = []
    calls_byte = []
    mpr121._bus.write_word = lambda reg, val: calls_word.append((reg,val))
    mpr121._bus.write_byte = lambda reg, val: calls_byte.append((reg,val))
    mpr121.auto_config(
        ace=1, are=0, bva=2, retry=3, afes=1, scts=1, acfie=0, arfie=1,
        oorie=0, usl=11, lsl=22, tl=33)
    # lb = (afes<<6) | (retry<<4) | (bva<<2) | (are<<1) | (ace)
    lb = (1 << 6) | (3 << 4) | (2 << 2) | (0 << 1) | 1
    # hb = (scts<<7) | (oorie<<2) | (arfie<<1) | acfie
    hb = (1 << 7) | (0 << 2) | (1 << 1) | 0
    expected_word = (hb << 8) + lb
    assert calls_word and calls_word[-1] == (ACNF_C0, expected_word)
    expected_bytes = {ACNF_USL: 11, ACNF_LSL: 22, ACNF_TL: 33}
    for reg, val in expected_bytes.items():
        assert (reg, val) in calls_byte

@pytest.mark.parametrize("raise_fail,error_msg,bit", [
    (True, 'auto config failed', 15),
    (True, 'auto reconfig failed', 14),
    (False, None, 15),
])
def test_out_of_range_behavior(mpr121, raise_fail, error_msg, bit):
    val = (1 << bit) if error_msg else 0
    mpr121._bus.read_word = lambda reg: val
    if raise_fail:
        with pytest.raises(Exception) as e:
            mpr121.out_of_range(raise_on_failure=True)
        assert error_msg in str(e.value)
    else:
        # Should just return bits masked
        result = mpr121.out_of_range(raise_on_failure=False)
        assert result == val & 0x1fff

def test_gpio_setup_calls_set_bit(monkeypatch, mpr121):
    # We patch _bus.read_byte and write_byte to track values
    calls = []

    reg_values = {
        GPIO_CTL0: 0x00,
        GPIO_CTL1: 0x00,
        GPIO_DIR: 0x00,
        GPIO_EN: 0x00,
    }

    def read_byte_mock(addr):
        return reg_values.get(addr, 0)

    def write_byte_mock(addr, val):
        calls.append((addr, val))
        reg_values[addr] = val

    monkeypatch.setattr(mpr121._bus, "read_byte", read_byte_mock)
    monkeypatch.setattr(mpr121._bus, "write_byte", write_byte_mock)

    # Call with channel=6 (valid 4-11), output=1, mode=3 (0b11), enable=1
    mpr121.gpio_setup(channel=6, output=1, mode=3, enable=1)

    # channel adjusted to 6-4=2, bit index 2
    bit = 2

    # There should be 4 writes, one each for each GPIO_* register
    expected_regs = {GPIO_CTL0, GPIO_CTL1, GPIO_DIR, GPIO_EN}
    written_regs = {reg for reg, _ in calls}
    assert expected_regs.issubset(written_regs)

    # Check bits are set/unset correctly in the final values:
    # mode=3 (0b11) -> bit1 and bit0 set
    # GPIO_CTL0 bit is bit1 (mode&0b10) -> bit1 = 1
    # GPIO_CTL1 bit is bit0 (mode&0b01) -> bit0 = 1
    ctl0_val = reg_values[GPIO_CTL0]
    ctl1_val = reg_values[GPIO_CTL1]
    # bit at position 2
    assert (ctl0_val >> bit) & 1 == 1  # bit1 of mode is set -> 1 in GPIO_CTL0
    assert (ctl1_val >> bit) & 1 == 1  # bit0 of mode is set -> 1 in GPIO_CTL1
    # GPIO_DIR bit (output=1) at bit position 2 must be 1
    dir_val = reg_values[GPIO_DIR]
    assert (dir_val >> bit) & 1 == 1
    # GPIO_EN bit (enable=1) at bit position 2 must be 1
    en_val = reg_values[GPIO_EN]
    assert (en_val >> bit) & 1 == 1

def test_gpio_setup_mode_1_maps_to_0(monkeypatch, mpr121):
    # mode=1 should be treated as 0
    reg_values = {
        GPIO_CTL0: 0,
        GPIO_CTL1: 0,
        GPIO_DIR: 0,
        GPIO_EN: 0,
    }
    calls = []

    def read_byte_mock(addr):
        return reg_values.get(addr, 0)
    def write_byte_mock(addr, val):
        calls.append((addr,val))
        reg_values[addr] = val

    monkeypatch.setattr(mpr121._bus, "read_byte", read_byte_mock)
    monkeypatch.setattr(mpr121._bus, "write_byte", write_byte_mock)

    mpr121.gpio_setup(channel=4, output=0, mode=1, enable=1)
    # channel = 4 => bit 0
    bit = 0
    ctl0_val = reg_values[GPIO_CTL0]
    ctl1_val = reg_values[GPIO_CTL1]
    # mode=1 maps to 0, bits 1 and 0 both zero
    assert ((ctl0_val >> bit) & 1) == 0
    assert ((ctl1_val >> bit) & 1) == 0
    # GPIO_DIR = output=0 -> bit cleared
    dir_val = reg_values[GPIO_DIR]
    assert ((dir_val >> bit) & 1) == 0
    # GPIO_EN = 1 -> bit set
    assert ((reg_values[GPIO_EN] >> bit) & 1) == 1

def test_gpio_status_reads_byte(mpr121):
    mpr121._bus.read_byte = lambda reg: 0xAB
    val = mpr121.gpio_status()
    assert val == 0xAB

def test_gpio_set_writes_set_and_clear_correctly(mpr121):
    calls = []
    mpr121._bus.write_byte = lambda reg, val: calls.append((reg, val))
    mpr121.gpio_set(channel=5, value=True)
    assert any(reg == GPIO_SET and val == (1 << (5 - 4)) for reg, val in calls)
    calls.clear()
    mpr121.gpio_set(channel=10, value=False)
    assert any(reg == GPIO_CLR and val == (1 << (10 - 4)) for reg, val in calls)

def test_on_touch_assigns_and_removes_handlers(mpr121):
    called = []
    def handler(touched, channel):
        called.append((touched, channel))
    mpr121.on_touch(1, handler)
    # Replace handler with None disables it
    assert mpr121._handlers[1] is not None
    mpr121.on_touch(1, None)
    # Should be default noop function (not None)
    assert mpr121._handlers[1] is not None

def test_handle_touch_calls_handler_with_correct_args(mpr121):
    states = []
    def h1(touched, channel):
        states.append((touched, channel))

    def h0(touched):
        states.append((touched, None))

    mpr121._handlers[0] = h1
    mpr121._handlers[1] = h0
    # touched=bits with bits 0 and 1 set, changed=bit 0 and 1 changed
    t = (1 << 0) | (1 << 1)
    mpr121._handle_touch(touched=t, changed=t)
    # h1 called with (True, 0)
    # h0 called with (True)
    assert (True, 0) in states
    assert (True, None) in states

def test_update_touch_state_triggers_handle_touch(monkeypatch, mpr121):
    # Setup initial _touched as 0 and touched() to return 1<<2 (channel 2)
    mpr121._touched = 0
    monkeypatch.setattr(mpr121, 'touched', lambda: 1 << 2)
    called_args = {}
    def fake_handle_touch(touched, changed):
        called_args['touched'] = touched
        called_args['changed'] = changed
    monkeypatch.setattr(mpr121, '_handle_touch', fake_handle_touch)
    mpr121.update_touch_state()
    assert called_args.get('touched') == (1 << 2)
    assert called_args.get('changed') == (1 << 2)
    # Call again with same touched, _handle_touch should not be called again (changed=0)
    called_args.clear()
    mpr121.update_touch_state()
    # _handle_touch not called when no change - keys stay empty
    assert called_args == {}

def test_is_touched_returns_correct_boolean(mpr121):
    mpr121._touched = 0b1010  # channels 1 and 3 touched
    assert mpr121.is_touched(1) is True
    assert mpr121.is_touched(3) is True
    assert mpr121.is_touched(0) is False
    assert mpr121.is_touched(2) is False
