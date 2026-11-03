import pytest
import random
from piripherals.mpr121 import MPR121, NCH, SRESET, ECR, TTH, RTH, DEB, AFE1, ETS, OOR, EFD, EBL, CDC, CDT, ACNF_C0, ACNF_USL, ACNF_LSL, ACNF_TL, GPIO_CTL0, GPIO_CTL1, GPIO_DIR, GPIO_EN, GPIO_SET, GPIO_CLR, GPIO_DAT


class DummyBusDevice:
    """Dummy I2C device for testing."""

    def __init__(self):
        # Registers initialized with zeros or preset values
        self.registers = {}
        # To keep block data for reads/writes easily
        # For block read of EFD and EBL we preset some data
        self.registers[EBL] = [10] * NCH  # baseline raw data (bytes)
        self.read_word_vals = {
            ETS: 0x1234,  # example touch status with no failure bits set
            OOR: 0x0000,  # example out of range status with no failure bits set
            ACNF_C0: 0,
        }
        # Storage for bytes for TTH, RTH, DEB, CDC, CDT, etc
        for reg in range(TTH, TTH + 2 * NCH):
            self.registers[reg] = 0
        for reg in range(RTH, RTH + 2 * NCH):
            self.registers[reg] = 0
        self.registers[DEB] = 0
        for reg in range(CDC, CDC + 12):
            self.registers[reg] = 0
        for reg in range(CDT, CDT + 6):
            self.registers[reg] = 0
        self.registers[GPIO_DAT] = 0
        self.registers[GPIO_CTL0] = 0
        self.registers[GPIO_CTL1] = 0
        self.registers[GPIO_DIR] = 0
        self.registers[GPIO_EN] = 0

        # For block registers: store 2*NCH bytes for EFD (electrode filtered data)
        # Make data such that electrode_data() returns increment 0..NCH-1
        self.block_data = {}
        efd_bytes = []
        for i in range(NCH):
            val = i  # 10 bits sensor value
            efd_bytes.append(val & 0xff)
            efd_bytes.append((val >> 8) & 0xff)
        self.block_data[EFD] = efd_bytes

    def write_byte(self, reg, val):
        self.registers[reg] = val & 0xFF

    def read_byte(self, reg):
        return self.registers.get(reg, 0)

    def write_word(self, reg, val):
        # split val in two bytes low/high
        self.registers[reg] = val & 0xFFFF
        self.read_word_vals[reg] = val & 0xFFFF

    def read_word(self, reg):
        # return preset or zero if not preset
        return self.read_word_vals.get(reg, 0)

    def write_block(self, reg, data):
        # Save data list to registers or block_data as applicable by reg
        # For baseline configure, etc, just store in registers dict by offset
        self.registers[reg] = list(data)

    def read_block(self, reg, length):
        # For baseline returns NCH bytes at EBL
        if reg == EBL and length == NCH:
            return self.registers.get(EBL, [0]*NCH)
        # For electrode filtered data EFD returns 2*NCH bytes
        if reg == EFD and length == 2 * NCH:
            return self.block_data.get(EFD, [0]*(2*NCH))
        # If block_data has requested reg return that truncated or padded
        if reg in self.block_data:
            d = self.block_data[reg]
            if len(d) >= length:
                return d[:length]
            else:
                return d + [0] * (length - len(d))
        # Default: return zero bytes
        return [0] * length


class DummyBus:
    def __init__(self):
        self.called_device_addrs = []
        self.device_obj = DummyBusDevice()

    def device(self, addr):
        self.called_device_addrs.append(addr)
        return self.device_obj


@pytest.fixture
def dummy_mpr(monkeypatch):
    # Create dummy bus & monkeypatch piripherals.mpr121.Bus to return dummy bus
    dummy_bus = DummyBus()

    # Patch bus.Bus class so that MPR121 __init__ uses dummy bus.
    # The module uses "from .bus import Bus" so we patch "piripherals.mpr121.Bus"
    monkeypatch.setattr("piripherals.mpr121.Bus", lambda bus_id: dummy_bus)
    # Return dummy bus and device for direct manipulation in tests
    return dummy_bus, dummy_bus.device_obj


def test_init_defaults_no_handlers_no_setup_no_reset(dummy_mpr, monkeypatch):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    # Disable irq and handlers and reset & setup
    mpr = MPR121(bus=1, irq=0, handlers=0, reset=0, setup=0)
    assert mpr.addr == 0x5a
    # The device member should be dummy_device
    assert mpr._bus is dummy_device
    # Should have default handlers all noop (callable)
    for h in mpr._handlers:
        assert callable(h)
    # _touched init is zero
    assert mpr._touched == 0
    # No methods called on bus during init because handlers=0 and setup=0 and reset=0
    # But device creation called with address 0x5a
    assert dummy_bus.called_device_addrs == [0x5a]


def test_reset_calls_write_byte(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)
    mpr.reset()
    # Should write 0x63 to SRESET register
    assert dummy_device.registers.get(SRESET) == 0x63


def test_configure_writes_correct_byte(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)
    mpr.configure(cl=2, prox=3, touch=4)
    expected = (2 << 6) | (3 << 4) | 4
    assert dummy_device.registers[ECR] == expected


def test_touched_returns_low_bits_and_raises_on_failure(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    dummy_device.read_word_vals[ETS] = 0b1000000000000000 | 0b1  # overcurrent bit set + bit0 touched
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)
    with pytest.raises(Exception) as e:
        mpr.touched()
    assert 'overcurrent' in str(e.value)

    # Disable raise by argument
    val = mpr.touched(raise_on_failure=0)
    # Should return lower 13 bits (only bit0=1)
    assert val & 1 == 1


def test_out_of_range_raises_for_failure_bits(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    # Auto config failed set
    dummy_device.read_word_vals[OOR] = 1 << 15
    with pytest.raises(Exception) as e:
        mpr.out_of_range()
    assert 'auto config failed' in str(e.value)

    # Auto reconfig failed set
    dummy_device.read_word_vals[OOR] = 1 << 14
    with pytest.raises(Exception) as e:
        mpr.out_of_range()
    assert 'auto reconfig failed' in str(e.value)

    # normal no error
    dummy_device.read_word_vals[OOR] = 0x1234
    val = mpr.out_of_range(raise_on_failure=0)
    assert isinstance(val, int)
    # Only low 13 bits reported
    assert val == (0x1234 & 0x1FFF)


def test_electrode_data_returns_correct_values(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    data = mpr.electrode_data()
    # Data length should be NCH
    assert len(data) == NCH
    # Values should be increasing from 0 to NCH-1 based on dummy_device block_data setup
    for i, val in enumerate(data):
        assert val == i


def test_baseline_returns_raw_baseline_on_no_args(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    # Setup EBL register bytes in dummy
    dummy_device.registers[EBL] = [i for i in range(NCH)]
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)
    baseline_vals = mpr.baseline()
    assert len(baseline_vals) == NCH
    # baseline() shifts left by 2 bits, so values multiplied by 4
    for i, val in enumerate(baseline_vals):
        assert val == i << 2


def test_baseline_configure_calls_write_block_no_raise(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    # Call baseline with rft=0 (less than 2), prox=False
    mpr.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20, prox=False)
    reg = 0x2b + 4 * 0
    assert dummy_device.registers.get(reg) == [5, 1, 3, 20]

    # Call baseline with rft=2 (>1), prox=True
    mpr.baseline(rft=2, mhd=5, nhd=1, ncl=3, fdl=20, prox=True)
    regp = 0x36 + 4 * 2
    assert dummy_device.registers.get(regp) == [1, 3, 20]


def test_threshold_sets_all_channels_when_channel_not_specified(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    mpr.threshold(touch=100, release=80, channel=-1)
    # Each channel from 0 to 12 should have TTH and RTH set accordingly:
    for ch in range(NCH):
        tth_val = dummy_device.registers.get(TTH + 2 * ch)
        rth_val = dummy_device.registers.get(RTH + 2 * ch)
        assert tth_val == 100
        assert rth_val == 80


def test_threshold_release_default_is_0_6_times_touch(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    mpr.threshold(touch=50, release=-1, channel=5)
    # release default to int(0.6*50)
    expected_release = int(0.6 * 50)
    assert dummy_device.registers.get(TTH + 2 * 5) == 50
    assert dummy_device.registers.get(RTH + 2 * 5) == expected_release


def test_debounce_sets_correct_byte(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    mpr.debounce(touch=3, release=5)
    expected = (5 << 4) | 3
    assert dummy_device.registers[DEB] == expected

    # release default to touch
    mpr.debounce(touch=4, release=-1)
    expected2 = (4 << 4) | 4
    assert dummy_device.registers[DEB] == expected2


def test_filter_writes_word_with_correct_bits(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    mpr.filter(cdc=31, cdt=3, ffi=2, sfi=1, esi=4)
    lb = (2 << 6) | 31
    hb = (3 << 5) | (1 << 3) | 4
    expected_word = (hb << 8) | lb
    reg_val = dummy_device.registers[AFE1]
    assert reg_val == expected_word


def test_charge_writes_correct_bytes_and_masks(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    # Preset CDT register to known value for mask test
    dummy_device.registers[CDT] = 0b11110000
    dummy_device.registers[CDT + 1] = 0b10101010
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    # Channel 0 sets bits at s=0
    mpr.charge(channel=0, cdc=5, cdt=3)
    assert dummy_device.registers[CDC + 0] == 5
    # CDT register bits 3:0 replaced by 3 (0b0011), original high bits intact
    expected_cdt = (dummy_device.registers[CDT] & (~0xf)) | (3 << 0)
    assert dummy_device.registers[CDT] == expected_cdt

    # Channel 1 sets bits at s=4
    dummy_device.registers[CDT + 0] = 0  # reset CDT register
    mpr.charge(channel=1, cdc=7, cdt=4)
    assert dummy_device.registers[CDC + 1] == 7
    expected_cdt1 = (dummy_device.registers[CDT] & ~(0xf << 4)) | (4 << 4)
    assert dummy_device.registers[CDT] == expected_cdt1


def test_auto_config_writes_bytes_and_word(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    mpr.auto_config(ace=1, are=0, bva=2, retry=3, afes=1,
                    scts=1, acfie=0, arfie=1, oorie=0,
                    usl=201, lsl=131, tl=181)

    lb = (1 << 6) | (3 << 4) | (2 << 2) | (0 << 1) | (1)
    hb = (1 << 7) | (0 << 2) | (1 << 1) | (0)
    expected_word = (hb << 8) + lb
    assert dummy_device.registers[ACNF_C0] == expected_word
    assert dummy_device.registers[ACNF_USL] == 201
    assert dummy_device.registers[ACNF_LSL] == 131
    assert dummy_device.registers[ACNF_TL] == 181


def test_gpio_setup_calls_write_byte_and_sets_bits(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    # Initialize GPIO registers with known values
    dummy_device.registers[GPIO_CTL0] = 0b10101010
    dummy_device.registers[GPIO_CTL1] = 0b01010101
    dummy_device.registers[GPIO_DIR] = 0b11110000
    dummy_device.registers[GPIO_EN] = 0b00001111

    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    # channel 7 (valid 4 <= ch <= 11)
    ch = 7
    output = 1
    mode = 3  # open drain output, high side MOS only (mode=3)
    enable = 1

    mpr.gpio_setup(ch, output, mode=mode, enable=enable)

    bitpos = ch - 4
    # Bits for mode=3: binary 0b11 -> bits: ctl0 bit =1 (bit1), ctl1 bit=1 (bit0)
    ctl0_old = 0b10101010
    ctl1_old = 0b01010101

    # After setting bit at bitpos, bit should be set accordingly preserving other bits
    ctl0_expected = (ctl0_old & ~(1 << bitpos)) | (1 << bitpos)
    ctl1_expected = (ctl1_old & ~(1 << bitpos)) | (1 << bitpos)

    assert dummy_device.registers[GPIO_CTL0] == ctl0_expected
    assert dummy_device.registers[GPIO_CTL1] == ctl1_expected

    # Direction bit set at bitpos to output=1
    gpio_dir_old = 0b11110000
    gpio_dir_expected = (gpio_dir_old & ~(1 << bitpos)) | (1 << bitpos)
    assert dummy_device.registers[GPIO_DIR] == gpio_dir_expected

    # Enable bit set similarly
    gpio_en_old = 0b00001111
    gpio_en_expected = (gpio_en_old & ~(1 << bitpos)) | (1 << bitpos)
    assert dummy_device.registers[GPIO_EN] == gpio_en_expected


def test_gpio_status_reads_byte(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    dummy_device.registers[GPIO_DAT] = 0x5A
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)
    val = mpr.gpio_status()
    assert val == 0x5A


def test_gpio_set_writes_set_and_clr(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    # Setting channel 6 HIGH (1)
    mpr.gpio_set(6, True)
    # Should have written to GPIO_SET register
    # data should be 1 << (6 -4) = 1 << 2 = 4
    assert dummy_device.registers.get(GPIO_SET) == 4
    assert GPIO_CLR not in dummy_device.registers

    # Setting channel 8 LOW (0)
    mpr.gpio_set(8, False)
    # Should have written to GPIO_CLR register
    expected_val = 1 << (8 - 4)
    assert dummy_device.registers.get(GPIO_CLR) == expected_val


def test_is_touched_reflects_touched_bit(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)
    # Manually set _touched bitfield
    mpr._touched = 0b101  # channels 0 and 2 touched

    assert mpr.is_touched(0) is True
    assert mpr.is_touched(1) is False
    assert mpr.is_touched(2) is True


def test_on_touch_registers_handler_and_calls_properly(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    called = []

    def handler(state, channel=None):
        called.append((state, channel))

    mpr.on_touch(3, handler)
    # The stored handler for channel 3 should be the "not_raising" wrapped version, callable
    assert callable(mpr._handlers[3])

    # Trigger _handle_touch with change on channel 3 touched
    old_touched = mpr._touched
    new_touched = old_touched ^ (1 << 3)  # change bit 3 toggled

    mpr._handle_touch(new_touched, 1 << 3)
    # The handler should have been called with (True or False, 3)
    assert len(called) == 1
    state, channel = called[0]
    # state is True if bit 3 in new_touched is set, else False
    expected_state = (new_touched & (1 << 3)) > 0
    assert state == expected_state
    assert channel == 3


def test_update_touch_state_updates_and_calls_handlers(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    # Replace touched method to simulate changing touch bits
    states = [0b0001, 0b0011, 0b0001]  # touch bit 0 toggling with additional bit 1

    call_record = []

    def fake_touched(raise_on_failure=1):
        return states.pop(0)

    mpr.touched = fake_touched

    def handler(state, channel):
        call_record.append((channel, state))

    for ch in range(NCH):
        mpr.on_touch(ch, handler)

    # Initially _touched is 0
    mpr._touched = 0

    # First update: from 0 to 1 (bit0 changed)
    mpr.update_touch_state()
    # It should call handler for bit 0 (changed)
    assert (0, True) in call_record

    old_len = len(call_record)
    # Second update: from 1 to 3 (bits 1 and 0 changed)
    mpr.update_touch_state()
    # Should detect changes for bits that are changed:
    # changed = 1 ^ 3 = 2 (bit1 only)
    # So only bit 1 handler called
    assert (1, True) in call_record
    assert len(call_record) > old_len

    old_len = len(call_record)
    # Third update: from 3 to 1 (bit1 changed off)
    mpr.update_touch_state()
    # Changed = 3 ^ 1 = 2 (bit 1 toggled off)
    assert (1, False) in call_record
    # No extra calls for other bits
    assert len(call_record) > old_len


def test_handlers_do_not_raise_if_removing(dummy_mpr):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    mpr.on_touch(0, None)  # remove handler, should store noop callable
    # The handler is callable
    assert callable(mpr._handlers[0])


def test_dump_runs_without_error(dummy_mpr, monkeypatch):
    random.seed(0)
    dummy_bus, dummy_device = dummy_mpr
    mpr = MPR121(bus=1, handlers=0, setup=0, reset=0)

    # Monkeypatch sys.stdout.write to a no-op to avoid printing escape sequences
    import sys
    monkeypatch.setattr(sys.stdout, "write", lambda *args, **kwargs: None)

    # Monkeypatch print to dummy to suppress output
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    # This test just runs dump, it should not raise under normal dummy data
    mpr.dump(regs=True, up=True, loop=1)


# Ensure monkeypatch is used for patching imports and that test functions are properly seeded and isolated.
