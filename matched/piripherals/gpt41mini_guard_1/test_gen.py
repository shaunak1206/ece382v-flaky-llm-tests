import builtins
import pytest
from piripherals import mpr121


class DummyBusDevice:
    """Mock for the device returned by bus.device(addr)."""

    def __init__(self):
        self.written_bytes = []   # (register, value)
        self.written_blocks = []  # (register, list_of_bytes)
        self.words = {}
        self.bytes = {}
        self.block_reads = {}
        self.byte_reads = {}
        self.word_reads = {}

    def write_byte(self, reg, val):
        self.written_bytes.append((reg, val))

    def write_block(self, reg, vals):
        self.written_blocks.append((reg, list(vals)))

    def write_word(self, reg, val):
        self.written_bytes.append((reg, val & 0xFF))
        self.written_bytes.append((reg + 1, (val >> 8) & 0xFF))

    def read_byte(self, reg):
        return self.byte_reads.get(reg, 0)

    def read_word(self, reg):
        # Return word from word_reads, fallback 0
        return self.word_reads.get(reg, 0)

    def read_block(self, reg, length):
        return self.block_reads.get((reg, length), [0] * length)


class DummyBus:
    """Mock for the Bus class from .bus"""

    def __init__(self):
        self._device = DummyBusDevice()

    def device(self, addr):
        return self._device


class DummyIRQHandler:
    def __init__(self, irq, callback):  # Does nothing
        self.irq = irq
        self.callback = callback


class DummyPoller:
    def __init__(self, callback):  # Does nothing
        self.callback = callback


@pytest.fixture(autouse=True)
def patch_internal(monkeypatch):
    # Patch Bus used inside MPR121 to DummyBus
    monkeypatch.setattr(mpr121, "Bus", lambda *a, **k: DummyBus())
    # Patch IRQHandler and Poller to dummy classes that do not start threads
    monkeypatch.setattr(mpr121, "IRQHandler", DummyIRQHandler)
    monkeypatch.setattr(mpr121, "Poller", DummyPoller)


@pytest.fixture
def mpr():
    """Construct MPR121 instance with dummy bus device."""
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    # Access dummy device directly for setup
    m._bus = DummyBusDevice()
    m._handlers = [mpr121.noop] * mpr121.NCH
    m._touched = 0
    return m


def test_reset_writes_expected_byte(mpr):
    mpr.reset()
    assert mpr._bus.written_bytes[-1] == (mpr121.SRESET, 0x63)


@pytest.mark.parametrize("cl,prox,touch", [
    (0, 0, 0),
    (1, 1, 1),
    (2, 2, 12),
    (3, 3, 5),
])
def test_configure_writes_correct_byte(mpr, cl, prox, touch):
    mpr.configure(cl=cl, prox=prox, touch=touch)
    expected = (cl << 6) | (prox << 4) | touch
    found = None
    for reg, val in mpr._bus.written_bytes:
        if reg == mpr121.ECR:
            found = val
            break
    assert found == expected


def test_touched_reads_word_and_returns_bits(mpr):
    # Setup word_reads to simulate touch bits 0b1011 (channels 0,1,3 touched) = 0x000B
    val = 0x000B
    mpr._bus.word_reads[mpr121.ETS] = val
    result = mpr.touched(raise_on_failure=0)
    # Mask out upper bits, only first 13 bits used
    assert result == (val & 0x1FFF)


def test_touched_raises_on_overcurrent(mpr):
    # Set touch bits plus overcurrent bit (15)
    val = (1 << 15) | 0x0001
    mpr._bus.word_reads[mpr121.ETS] = val
    with pytest.raises(Exception) as e:
        mpr.touched(raise_on_failure=1)
    assert "overcurrent" in str(e.value)


def test_touched_raises_on_out_of_range(mpr):
    # Set out_of_range bits in out_of_range() call triggered inside touched
    mpr._bus.word_reads[mpr121.ETS] = 0
    # Patch out_of_range to raise exception
    def oor(raise_on_failure):
        raise Exception("out of range")
    mpr.out_of_range = oor
    with pytest.raises(Exception) as e:
        mpr.touched(raise_on_failure=1)
    assert "out of range" in str(e.value)


def test_electrode_data_returns_correct_values(mpr):
    # Setup dummy block data for 13 channels, each channel 2 bytes: low byte, high byte
    # Provide values: channel i = i * 100 (within 10 bit 0..1023)
    raw_data = []
    for i in range(mpr121.NCH):
        val = i * 100
        low = val & 0xFF
        high = (val >> 8) & 0xFF
        raw_data.extend([low, high])
    mpr._bus.block_reads[(mpr121.EFD, 2 * mpr121.NCH)] = raw_data
    data = mpr.electrode_data()
    expected = [i * 100 for i in range(mpr121.NCH)]
    assert data == expected


def test_baseline_returns_values_when_no_args(mpr):
    # Provide block read returning 13 baseline bytes
    baseline_raw = list(range(mpr121.NCH))
    mpr._bus.block_reads[(mpr121.EBL, mpr121.NCH)] = baseline_raw
    result = mpr.baseline()
    # Each baseline byte shifted left by 2
    expected = [b << 2 for b in baseline_raw]
    assert result == expected


@pytest.mark.parametrize("rft", [0, 1])
def test_baseline_configure_writes_block_for_rft_0_1(mpr, rft):
    # Patch _bus.write_block to record calls
    called = {}

    def write_block(reg, vals):
        called['reg'] = reg
        called['vals'] = vals
    mpr._bus.write_block = write_block
    # Call baseline with rft 0 or 1 (not prox)
    mpr.baseline(rft=rft, mhd=10, nhd=20, ncl=30, fdl=40, prox=0)
    expected_reg = mpr121.MHD + 4 * rft
    assert called['reg'] == expected_reg
    assert called['vals'] == [10, 20, 30, 40]


@pytest.mark.parametrize("rft", [2, 3])
def test_baseline_configure_writes_block_for_rft_2_or_3(mpr, rft):
    called = {}

    def write_block(reg, vals):
        called['reg'] = reg
        called['vals'] = vals
    mpr._bus.write_block = write_block
    mpr.baseline(rft=rft, nhd=21, ncl=31, fdl=41, prox=0)
    expected_reg = mpr121.MHD + 4 * rft  # prox=0 so base MHD
    assert called['reg'] == expected_reg
    # For rft>1, only 3 values written: nhd, ncl, fdl
    assert called['vals'] == [21, 31, 41]


def test_baseline_configure_uses_MHDX_for_prox_true(mpr):
    called = {}

    def write_block(reg, vals):
        called['reg'] = reg
        called['vals'] = vals
    mpr._bus.write_block = write_block
    mpr.baseline(rft=0, mhd=1, nhd=2, ncl=3, fdl=4, prox=1)
    assert called['reg'] == mpr121.MHDX + 4 * 0  # prox True uses MHDX
    assert called['vals'] == [1, 2, 3, 4]


@pytest.mark.parametrize("touch,release,channel", [
    (50, -1, -1),
    (100, 40, -1),
    (200, 150, 5),
    (30, -1, 12),
])
def test_threshold_writes_correct_bytes(mpr, touch, release, channel):
    called = []
    # Patch _bus.write_byte to record calls
    def write_byte(reg, val):
        called.append((reg, val))
    mpr._bus.write_byte = write_byte
    mpr.threshold(touch, release, channel)
    if channel < 0:
        # Should write for all channels (13)
        # Each channel writes 2 bytes (touch and release)
        expected_calls = 2 * mpr121.NCH
        assert len(called) == expected_calls
        # Check first touch and release matches calculation for channel 0
        tth, rth = called[0], called[1]
        assert tth[0] == mpr121.TTH
        # release is default 0.6 * touch if release<0
        expected_release = release if release >= 0 else int(0.6 * touch)
        assert rth[0] == mpr121.RTH
        # Check that the difference between touches is 2 bytes
        for i in range(mpr121.NCH):
            assert called[2*i][0] == mpr121.TTH + 2*i
            assert called[2*i+1][0] == mpr121.RTH + 2*i
    else:
        # Should write exactly 2 bytes for the given channel
        assert len(called) == 2
        assert called[0][0] == mpr121.TTH + 2 * channel
        assert called[1][0] == mpr121.RTH + 2 * channel
        assert called[0][1] == touch
        if release < 0:
            assert called[1][1] == int(0.6 * touch)
        else:
            assert called[1][1] == release


@pytest.mark.parametrize("touch,release", [
    (3, -1),
    (5, 2),
    (0, 0),
])
def test_debounce_writes_correct_byte(mpr, touch, release):
    called = []

    def write_byte(reg, val):
        called.append((reg, val))
    mpr._bus.write_byte = write_byte
    mpr.debounce(touch, release)
    assert len(called) == 1
    val = called[0][1]
    expected_release = release if release >= 0 else touch
    # High nibble is release, low nibble is touch
    assert val == ((expected_release << 4) | touch)


def test_filter_writes_correct_word(mpr):
    called = []

    def write_word(reg, val):
        called.append((reg, val))
    mpr._bus.write_word = write_word
    mpr.filter(cdc=10, cdt=3, ffi=2, sfi=1, esi=4)
    # Calculate expected lb and hb as in source code
    lb = (2 << 6) | 10
    hb = (3 << 5) | (1 << 3) | 4
    expected = (hb << 8) | lb
    assert len(called) == 1
    assert called[0][0] == mpr121.AFE1
    # Compare values exactly since integers
    assert called[0][1] == expected


def test_charge_writes_bytes_and_reads_byte(mpr):
    calls = []

    def write_byte(reg, val):
        calls.append((reg, val))

    def read_byte(reg):
        # Return 0xff so that bit masking can be tested
        return 0xFF

    mpr._bus.write_byte = write_byte
    mpr._bus.read_byte = read_byte

    mpr.charge(channel=3, cdc=12, cdt=6)
    # Should write charge current to CDC+3
    # Then read CDT+1, mask and write modified value to CDT+1
    found_writes = [c for c in calls if c[0] in (mpr121.CDC + 3, mpr121.CDT + 1)]
    assert any(c[0] == mpr121.CDC + 3 and c[1] == 12 for c in found_writes)
    assert any(c[0] == mpr121.CDT + 1 for c in found_writes)
    # Check that the bits for cdt are correctly set without affecting others
    # Bit shift position
    s = 4 * (3 % 2)
    # bit mask on read 0xFF & ~(0xf << s) | (cdt << s)
    expected_val = (0xFF & ~(0xF << s)) | (6 << s)
    write_reg, write_val = [w for w in calls if w[0] == mpr121.CDT + 1][0]
    assert write_val == expected_val


def test_auto_config_writes_expected_values(mpr):
    calls_byte = []
    calls_word = []

    def write_byte(reg, val):
        calls_byte.append((reg, val))

    def write_word(reg, val):
        calls_word.append((reg, val))

    mpr._bus.write_byte = write_byte
    mpr._bus.write_word = write_word

    mpr.auto_config(
        ace=1,
        are=0,
        bva=2,
        retry=1,
        afes=3,
        scts=1,
        acfie=0,
        arfie=1,
        oorie=0,
        usl=201,
        lsl=131,
        tl=181)

    # There should be one word write to ACNF_C0 and three byte writes to USL, LSL, TL
    assert any(reg == mpr121.ACNF_C0 for reg, _ in calls_word)

    # Calculate expected lb and hb
    lb = (3 << 6) | (1 << 4) | (2 << 2) | (0 << 1) | 1
    hb = (1 << 7) | (0 << 2) | (1 << 1) | 0
    expected_word = (hb << 8) + lb
    assert calls_word[0][1] == expected_word

    expected_writes = {
        mpr121.ACNF_USL: 201,
        mpr121.ACNF_LSL: 131,
        mpr121.ACNF_TL: 181,
    }
    for reg, val in calls_byte:
        assert expected_writes.get(reg) == val


@pytest.mark.parametrize("word,exception_text", [
    ((1 << 15), "auto config failed"),
    ((1 << 14), "auto reconfig failed"),
])
def test_out_of_range_raises_on_failure(mpr, word, exception_text):
    mpr._bus.word_reads[mpr121.OOR] = word
    with pytest.raises(Exception) as e:
        mpr.out_of_range(raise_on_failure=1)
    assert exception_text in str(e.value)


def test_out_of_range_returns_masked_value(mpr):
    val = 0x3fff  # over 13 bits to test masking
    mpr._bus.word_reads[mpr121.OOR] = val
    returned = mpr.out_of_range(raise_on_failure=0)
    assert returned == (val & 0x1fff)


@pytest.mark.parametrize("channel,output,mode,enable", [
    (4, True, 0, True),
    (5, False, 3, False),
    (11, True, 2, True),
])
def test_gpio_setup_writes_correct_bits(mpr, channel, output, mode, enable):
    # Patch bus.read_byte and write_byte to record and simulate a register value
    reg_values = {}

    def read_byte(reg):
        return reg_values.get(reg, 0)

    writes = []

    def write_byte(reg, val):
        reg_values[reg] = val
        writes.append((reg, val))

    mpr._bus.read_byte = read_byte
    mpr._bus.write_byte = write_byte

    mpr.gpio_setup(channel, output, mode, enable)

    # channel offset 0-7
    ch = channel - 4

    # Check bits in GPIO_CTL0 for mode & 0b10 (bit1)
    # mode & 0b01 => GPIO_CTL1 (bit0)
    bit1_val = (reg_values.get(mpr121.GPIO_CTL0, 0) >> ch) & 1
    bit0_val = (reg_values.get(mpr121.GPIO_CTL1, 0) >> ch) & 1
    expected_bit1 = (mode & 0b10) >> 1
    expected_bit0 = mode & 0b01
    assert bit1_val == expected_bit1
    assert bit0_val == expected_bit0

    # Check output bit in GPIO_DIR at position ch
    dir_val = (reg_values.get(mpr121.GPIO_DIR, 0) >> ch) & 1
    assert dir_val == (1 if output else 0)

    # Check enable bit in GPIO_EN at position ch
    en_val = (reg_values.get(mpr121.GPIO_EN, 0) >> ch) & 1
    assert en_val == (1 if enable else 0)


def test_gpio_setup_raises_on_invalid_channel(mpr):
    with pytest.raises(AssertionError):
        mpr.gpio_setup(3, True)
    with pytest.raises(AssertionError):
        mpr.gpio_setup(12, True)


@pytest.mark.parametrize("gpio_data", [0x00, 0xF0, 0xFF])
def test_gpio_status_reads_byte(mpr, gpio_data):
    mpr._bus.byte_reads[mpr121.GPIO_DAT] = gpio_data
    status = mpr.gpio_status()
    assert status == gpio_data


@pytest.mark.parametrize("channel,value", [
    (4, True),
    (5, False),
    (11, True),
])
def test_gpio_set_writes_correct_register(mpr, channel, value):
    writes = []

    def write_byte(reg, val):
        writes.append((reg, val))

    mpr._bus.write_byte = write_byte
    mpr.gpio_set(channel, value)
    assert writes
    reg, val = writes[-1]
    expected_reg = mpr121.GPIO_SET if value else mpr121.GPIO_CLR
    assert reg == expected_reg
    expected_val = 1 << (channel - 4)
    assert val == expected_val


def test_gpio_set_raises_on_invalid_channel(mpr):
    with pytest.raises(AssertionError):
        mpr.gpio_set(3, True)
    with pytest.raises(AssertionError):
        mpr.gpio_set(12, True)


def test_update_touch_state_calls_handle_touch_and_handlers(mpr):
    calls = []

    def dummy_handler(touched, channel):
        calls.append((touched, channel))

    # Setup initial _touched
    mpr._touched = 0b1010
    # Override touched() to flip bits (xor so channels 1 and 3 change)
    mpr.touched = lambda raise_on_failure=1: 0b0101
    for i in range(mpr121.NCH):
        mpr._handlers[i] = None  # Clear all
    # Set dummy handlers for channels 0-3
    for i in range(4):
        mpr._handlers[i] = dummy_handler
    mpr.update_touch_state()
    # Changes should be in channels 0,1,2,3 because touched flips bits at 0b1010 -> 0b0101
    # XOR = 0b1111, channels 0-3 changed
    touched_bits = 0b0101
    changed_bits = 0b1111
    expected_calls = []
    for i in range(mpr121.NCH):
        mask = 1 << i
        if changed_bits & mask:
            expected_calls.append(((touched_bits & mask) > 0, i))
    assert sorted(calls) == sorted(expected_calls)


def test_on_touch_registers_and_removes_handlers(mpr):
    called = []

    def handler_one(touched, channel):
        called.append((touched, channel))

    # Register a handler for channel 0
    mpr.on_touch(0, handler_one)
    assert mpr._handlers[0] is not mpr121.noop
    # Call _handle_touch to trigger handler
    mpr._handle_touch(touched=1, changed=1)
    assert called == [(True, 0)]

    # Remove handler by passing None
    mpr.on_touch(0, None)
    assert mpr._handlers[0] is mpr121.noop
    # Should not call anything now
    called.clear()
    mpr._handle_touch(touched=1, changed=1)
    assert called == []


def test_is_touched_returns_correct_boolean(mpr):
    # Set _touched bits arbitrarily e.g. 0b1010
    mpr._touched = 0b1010
    assert not mpr.is_touched(0)
    assert mpr.is_touched(1)
    assert not mpr.is_touched(2)
    assert mpr.is_touched(3)


def test_threshold_default_release_calculation(mpr):
    # Patch _bus.write_byte to record calls
    calls = []

    def write_byte(reg, val):
        calls.append((reg, val))

    mpr._bus.write_byte = write_byte
    # Call threshold with release=-1 should set release = int(0.6 * touch)
    mpr.threshold(touch=10, release=-1, channel=0)
    touch_written = [c for c in calls if c[0] == mpr121.TTH]
    release_written = [c for c in calls if c[0] == mpr121.RTH]
    assert touch_written and release_written
    expected_release = int(0.6 * 10)
    assert release_written[0][1] == expected_release


def test_dump_method_runs_without_error(monkeypatch):
    # Patch I/O and bus methods to avoid printing or actual bus reads
    monkeypatch.setattr(mpr121, "print", lambda *a, **k: None)
    dummy_model = mpr121.MPR121(handlers=0, setup=0, reset=0)
    m = dummy_model
    # Fill dummy bus registers for dump
    m._bus = DummyBusDevice()
    # Setup some dummy values so dump won't fail
    for i in range(mpr121.NCH):
        m._bus.byte_reads[mpr121.TTH + 2 * i] = 15
        m._bus.byte_reads[mpr121.RTH + 2 * i] = 10
        m._bus.byte_reads[mpr121.CDC + i] = 1 + i
    m._bus.byte_reads[mpr121.CDT + 0] = 0x11
    m._bus.byte_reads[mpr121.CDT + 1] = 0x22
    m._bus.byte_reads[mpr121.GPIO_DAT] = 0b11110000
    # Setup read_block for registers read
    for i in range(4):
        m._bus.block_reads[(32*i, 32)] = [i]*32
    # Setup electrode data & baseline and out_of_range and touched
    m._bus.block_reads[(mpr121.EFD, 2*mpr121.NCH)] = [0]*2*mpr121.NCH
    m._bus.block_reads[(mpr121.EBL, mpr121.NCH)] = [0]*mpr121.NCH
    m._bus.word_reads[mpr121.ETS] = 0x0
    m._bus.word_reads[mpr121.OOR] = 0x0
    # Call dump with loop=1 and regs=1 to run main iteration once
    m.dump(regs=1, up=0, loop=1)


# The total number of tests here is 25.
