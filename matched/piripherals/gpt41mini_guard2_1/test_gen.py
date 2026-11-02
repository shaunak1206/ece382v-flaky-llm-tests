import builtins
import pytest
from piripherals import MPR121


class DummyBusDevice:
    """Mock of the I2C bus device, recording writes and allowing
    setup of read responses."""

    def __init__(self):
        self._writes = []
        self._bytes = {}
        self._words = {}
        self._blocks = {}
        self._read_byte_calls = []
        self._read_word_calls = []
        self._read_block_calls = []

    def write_byte(self, reg, val):
        self._writes.append((reg, val))
        self._bytes[reg] = val

    def write_word(self, reg, val):
        self._writes.append((reg, val))
        self._words[reg] = val

    def write_block(self, reg, data):
        self._writes.append((reg, tuple(data)))
        self._blocks[reg] = list(data)

    def read_byte(self, reg):
        self._read_byte_calls.append(reg)
        return self._bytes.get(reg, 0)

    def read_word(self, reg):
        self._read_word_calls.append(reg)
        return self._words.get(reg, 0)

    def read_block(self, reg, length):
        self._read_block_calls.append((reg, length))
        data = self._blocks.get(reg)
        if data is not None:
            # Return original data truncated or padded to length
            if len(data) >= length:
                return data[:length]
            else:
                return data + [0] * (length - len(data))
        # Default fallback: return zeros
        return [0] * length


class DummyBus:
    def device(self, addr):
        return DummyBusDevice()


@pytest.fixture
def dummy_bus():
    return DummyBus()


@pytest.fixture
def mpr121(dummy_bus):
    # Initialize MPR121 with dummy bus and no IRQ/poller handlers (handlers=0)
    return MPR121(bus=dummy_bus, irq=0, handlers=0, setup=0, reset=0)


def test_reset_calls_write_byte(mpr121):
    # Reset should send 0x63 to SRESET
    mpr121.reset()
    writes = mpr121._bus._writes
    assert (mpr121.SRESET, 0x63) in writes


def test_configure_sets_register_byte(mpr121):
    # configure with cl=1, prox=2, touch=3 -> written byte should be
    val = (1 << 6) | (2 << 4) | 3
    mpr121.configure(cl=1, prox=2, touch=3)
    assert (mpr121.ECR, val) in mpr121._bus._writes


def test_touched_returns_touch_bits_no_errors(mpr121):
    # Setup device to return word without error bits set
    touch_bits = 0x123  # some bits in lower 13 bits set
    mpr121._bus._words[MPR121.ETS] = touch_bits
    # out_of_range returns 0 (no error)
    mpr121._bus._words[MPR121.OOR] = 0
    result = mpr121.touched()
    assert result == (touch_bits & 0x1fff)


def test_touched_raises_on_overcurrent(mpr121):
    # Bit 15 set means overcurrent
    val = (1 << 15) | 0x1
    mpr121._bus._words[MPR121.ETS] = val
    mpr121._bus._words[MPR121.OOR] = 0
    with pytest.raises(Exception, match='overcurrent'):
        mpr121.touched()


def test_touched_raises_on_out_of_range(monkeypatch, mpr121):
    # Setup touched returns word without error bits
    mpr121._bus._words[MPR121.ETS] = 0x1
    # Make out_of_range raise Exception 'auto config failed'
    def fake_out_of_range(raise_on_failure=1):
        raise Exception('out of range')

    monkeypatch.setattr(mpr121, "out_of_range", fake_out_of_range)
    with pytest.raises(Exception, match='out of range'):
        mpr121.touched()


def test_out_of_range_returns_bits_no_errors(mpr121):
    val = 0x123
    mpr121._bus._words[MPR121.OOR] = val
    assert mpr121.out_of_range() == (val & 0x1fff)


def test_out_of_range_raises_on_auto_config_fail(mpr121):
    val = (1 << 15) | 0x01
    mpr121._bus._words[MPR121.OOR] = val
    with pytest.raises(Exception, match='auto config failed'):
        mpr121.out_of_range()


def test_out_of_range_raises_on_auto_reconfig_fail(mpr121):
    val = (1 << 14) | 0x01
    mpr121._bus._words[MPR121.OOR] = val
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr121.out_of_range()


def test_baseline_reads_when_no_args(mpr121):
    # For rft < 0, baseline should read NCH bytes from EBL
    expected = [i for i in range(MPR121.NCH)]
    mpr121._bus._blocks[MPR121.EBL] = expected
    result = mpr121.baseline()
    # Each baseline byte shifted left by 2 bits
    assert result == [b << 2 for b in expected]


def test_baseline_write_block_rft_0_or_1(mpr121):
    # should write 4 bytes to MHD register offset by 4 * rft
    rft = 1
    reg = MPR121.MHD + 4 * rft
    mpr121.baseline(rft=rft, mhd=2, nhd=3, ncl=4, fdl=5)
    assert (reg, (2, 3, 4, 5)) in mpr121._bus._writes


def test_baseline_write_block_rft_2_or_above(mpr121):
    rft = 2
    reg = MPR121.MHDX + 4 * rft
    mpr121.baseline(rft=rft, nhd=6, ncl=7, fdl=8, prox=True)
    assert (reg, (6, 7, 8)) in mpr121._bus._writes


def test_threshold_sets_per_channel_bytes(mpr121):
    # For all channels when channel < 0
    mpr121._bus._writes.clear()
    mpr121.threshold(touch=100, release=50, channel=-1)
    # There should be 13*2 writes (touch and release for each channel 0-12)
    writes = mpr121._bus._writes
    touch_regs = {MPR121.TTH + 2 * ch for ch in range(MPR121.NCH)}
    release_regs = {MPR121.RTH + 2 * ch for ch in range(MPR121.NCH)}
    touched_written_regs = set()
    release_written_regs = set()
    for reg, val in writes:
        if reg in touch_regs:
            touched_written_regs.add(reg)
            assert val == 100
        if reg in release_regs:
            release_written_regs.add(reg)
            assert val == 50
    assert touched_written_regs == touch_regs
    assert release_written_regs == release_regs


def test_threshold_sets_single_channel_bytes(mpr121):
    ch = 5
    mpr121._bus._writes.clear()
    mpr121.threshold(touch=120, release=60, channel=ch)
    writes = mpr121._bus._writes
    # Should write exactly 2 bytes for channel ch
    expected_touch_reg = MPR121.TTH + 2 * ch
    expected_release_reg = MPR121.RTH + 2 * ch
    assert (expected_touch_reg, 120) in writes
    assert (expected_release_reg, 60) in writes


def test_debounce_write_byte(mpr121):
    mpr121.debounce(touch=3, release=4)
    val = (4 << 4) | 3
    assert (MPR121.DEB, val) in mpr121._bus._writes


def test_debounce_release_default_equals_touch(mpr121):
    mpr121.debounce(touch=5)
    val = (5 << 4) | 5
    assert (MPR121.DEB, val) in mpr121._bus._writes


def test_filter_write_word(mpr121):
    # cdc=10, cdt=3, ffi=2, sfi=1, esi=4
    # lb = (ffi << 6) | cdc = (2<<6) | 10 = 128 + 10 = 138
    # hb = (cdt <<5) | (sfi <<3) | esi = (3<<5) | (1<<3) | 4 = 96 + 8 +4=108
    expected_val = (108 << 8) | 138
    mpr121.filter(cdc=10, cdt=3, ffi=2, sfi=1, esi=4)
    assert (MPR121.AFE1, expected_val) in mpr121._bus._writes


def test_charge_writes_bytes_and_reads_byte(mpr121):
    # Setup the bus to have initial CDT register bytes
    reg = MPR121.CDT + 3 // 2  # channel 3 -> reg CDT+1
    initial_val = 0b1111000011110000
    mpr121._bus._bytes[reg] = initial_val
    mpr121.charge(channel=3, cdc=7, cdt=5)
    # Check CDC + channel write
    assert (MPR121.CDC + 3, 7) in mpr121._bus._writes
    # Check that read_byte for CDT+1 was called
    assert reg in mpr121._bus._read_byte_calls
    # Check that write_byte to CDT+1 occurred and cdt bits align correctly
    # Extract the last write to CDT+1
    writes_to_reg = [w for w in mpr121._bus._writes if w[0] == reg]
    assert writes_to_reg
    new_val = writes_to_reg[-1][1]
    # cdt inserted in bits 4 to 7 (4 bits shifted by s=4)
    # Check only relevant bits: (after masking out 0xf << 4)
    masked_old = initial_val & ~(0xf << 4)
    expected_val = masked_old | (5 << 4)
    assert new_val == expected_val


def test_auto_config_writes_word_and_bytes(mpr121):
    # Default parameters leads to specific lb and hb values
    lb = (1 << 6) | (2 << 4) | (3 << 2) | (1 << 1) | 1
    hb = (0 << 7) | (1 << 2) | (1 << 1) | 1
    expected_val = (hb << 8) + lb
    mpr121.auto_config()
    # Check that the word write is as expected
    found_word_write = False
    for reg, val in mpr121._bus._writes:
        if reg == MPR121.ACNF_C0:
            assert val == expected_val
            found_word_write = True
            break
    assert found_word_write
    # Check that the bytes at USL, LSL, TL were written and have default values
    assert (MPR121.ACNF_USL, 200) in mpr121._bus._writes
    assert (MPR121.ACNF_LSL, 130) in mpr121._bus._writes
    assert (MPR121.ACNF_TL, 180) in mpr121._bus._writes


def test_is_touched_reads_correct_bit_and_returns_bool(mpr121):
    # Set _touched to a pattern
    mpr121._touched = 0b1010  # channels 1 and 3 touched
    assert mpr121.is_touched(1) is True
    assert mpr121.is_touched(3) is True
    assert mpr121.is_touched(0) is False
    assert mpr121.is_touched(2) is False


def test_on_touch_sets_handler_and_executes_on_state_change(mpr121):
    called = []

    def handler(state, channel):
        called.append((state, channel))

    mpr121.on_touch(2, handler)
    # Simulate touch state changing on channel 2 from 0 to 1<<2
    mpr121._touched = 0
    # Monkeypatch touched method to return bit 2 touched
    monkeypatched_val = 1 << 2
    mpr121.touched = lambda raise_on_failure=1: monkeypatched_val
    # Call update_touch_state should detect change and call handler
    mpr121.update_touch_state()
    # We expect the handler to be called with state True and channel 2
    assert called == [(True, 2)]


def test_on_touch_with_handler_accepting_one_arg(mpr121):
    called = []

    def handler(state):
        called.append(state)

    mpr121.on_touch(0, handler)
    # Force touch state bit 0 changed
    mpr121._touched = 0
    monkeypatched_val = 1 << 0
    mpr121.touched = lambda raise_on_failure=1: monkeypatched_val
    mpr121.update_touch_state()
    # Handler called with single boolean argument
    assert called == [True]


def test_update_touch_state_no_change_calls_no_handler(mpr121):
    called = []

    def handler(state, channel):
        called.append((state, channel))

    mpr121.on_touch(0, handler)
    # touched returns same value as current _touched, so no change
    mpr121._touched = 0b101
    mpr121.touched = lambda raise_on_failure=1: mpr121._touched
    mpr121.update_touch_state()
    assert not called


def test_gpio_setup_asserts_channel_bounds(mpr121):
    with pytest.raises(AssertionError):
        mpr121.gpio_setup(channel=3, output=True)
    with pytest.raises(AssertionError):
        mpr121.gpio_setup(channel=12, output=True)


def test_gpio_setup_calls_write_byte(mpr121):
    # Setup initial read_byte return values for GPIO_CTL0, GPIO_CTL1, GPIO_DIR, GPIO_EN
    for reg in (MPR121.GPIO_CTL0, MPR121.GPIO_CTL1, MPR121.GPIO_DIR, MPR121.GPIO_EN):
        mpr121._bus._bytes[reg] = 0b00000000
    # Call gpio_setup for channel 5 (maps to bit 1)
    # mode=3 (binary '11'), output=1, enable=1
    mpr121.gpio_setup(channel=5, output=1, mode=3, enable=1)
    # There should be writes to each of the four registers with bits set accordingly
    writes = mpr121._bus._writes
    # The channel bit to set is (channel - 4) == 1
    bit = 1
    def check_bit_set(reg, expected_bit_value):
        # Find most recent write to reg
        for wreg, val in reversed(writes):
            if wreg == reg:
                return ((val >> bit) & 1) == expected_bit_value
        return None

    # mode=3 = binary 11, bits: 1 (LSB), 1 (bit1)
    # GPIO_CTL0 bit is mode & 0b10 >> 1 == bit 1 => 1
    assert check_bit_set(MPR121.GPIO_CTL0, 1) is True
    # GPIO_CTL1 bit is mode & 0b01 == bit 0 => 1
    assert check_bit_set(MPR121.GPIO_CTL1, 1) is True
    # GPIO_DIR should have bit set to output & 0b1 == 1
    assert check_bit_set(MPR121.GPIO_DIR, 1) is True
    # GPIO_EN should have bit set to enable & 0b1 == 1
    assert check_bit_set(MPR121.GPIO_EN, 1) is True


def test_gpio_status_reads_byte(mpr121):
    expected_val = 0xAA
    mpr121._bus._bytes[MPR121.GPIO_DAT] = expected_val
    result = mpr121.gpio_status()
    assert result == expected_val


def test_gpio_set_calls_write_byte_correct_reg(mpr121):
    mpr121._bus._writes.clear()
    # channel=4, value=True -> write to GPIO_SET register bit 0 (4 - 4 =0)
    mpr121.gpio_set(4, True)
    assert (MPR121.GPIO_SET, 1 << 0) in mpr121._bus._writes
    mpr121._bus._writes.clear()
    # channel=5, value=False -> write to GPIO_CLR register bit 1 (5-4=1)
    mpr121.gpio_set(5, False)
    assert (MPR121.GPIO_CLR, 1 << 1) in mpr121._bus._writes
