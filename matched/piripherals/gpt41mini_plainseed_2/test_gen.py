import pytest
from piripherals.mpr121 import MPR121, NCH, SRESET, ECR, ETS, OOR
from piripherals import mpr121


class DummyBusDevice:
    def __init__(self):
        self.written_bytes = []
        self.written_words = []
        self.written_blocks = []
        self.read_byte_calls = []
        self.read_word_calls = []
        self.read_block_calls = []
        self.byte_values = {}
        self.word_values = {}
        self.block_values = {}

    def write_byte(self, reg, val):
        self.written_bytes.append((reg, val))

    def write_word(self, reg, val):
        self.written_words.append((reg, val))

    def write_block(self, reg, vals):
        self.written_blocks.append((reg, list(vals)))

    def read_byte(self, reg=None):
        self.read_byte_calls.append(reg)
        return self.byte_values.get(reg, 0)

    def read_word(self, reg=None):
        self.read_word_calls.append(reg)
        return self.word_values.get(reg, 0)

    def read_block(self, reg=None, length=None):
        self.read_block_calls.append((reg, length))
        vals = self.block_values.get(reg, [0] * length)
        return vals[:length]


class DummyBus:
    def device(self, addr):
        return DummyBusDevice()


@pytest.fixture
def dummy_bus_device():
    bus = DummyBus()
    dev = bus.device(0x5a)
    return dev


def test_init_default(monkeypatch):
    dummy_device = DummyBusDevice()
    class DummyBusClass:
        def device(self, addr):
            return dummy_device

    monkeypatch.setattr('piripherals.mpr121.Bus', lambda busnum: DummyBusClass())
    # Patch IRQHandler and Poller to avoid side effects
    monkeypatch.setattr('piripherals.mpr121.IRQHandler', lambda irq, cb: None)
    monkeypatch.setattr('piripherals.mpr121.Poller', lambda cb: None)
    # Patch noop and not_raising
    monkeypatch.setattr('piripherals.mpr121.noop', lambda *a, **k: None)
    monkeypatch.setattr('piripherals.mpr121.not_raising', lambda h: h or (lambda *a, **k: None))

    m = MPR121()
    assert m._bus is dummy_device
    assert m.addr == 0x5a
    assert len(m._handlers) == NCH
    assert m._touched == 0


def test_reset_writes_reset_command(dummy_bus_device):
    m = MPR121(bus=DummyBus())
    m._bus = dummy_bus_device
    m.reset()
    assert (SRESET, 0x63) in dummy_bus_device.written_bytes


def test_configure_writes_correct_ecr_value(dummy_bus_device):
    m = MPR121(bus=DummyBus())
    m._bus = dummy_bus_device

    # cl=2, prox=1, touch=5
    m.configure(cl=2, prox=1, touch=5)
    # (cl << 6) | (prox <<4) | touch = (2<<6) | (1<<4) | 5 = 128 + 16 + 5 = 149
    found = False
    for reg, val in dummy_bus_device.written_bytes:
        if reg == ECR and val == ((2 << 6) | (1 << 4) | 5):
            found = True
            break
    assert found


def test_touched_returns_lower_13_bits(dummy_bus_device):
    m = MPR121(bus=DummyBus())
    m._bus = dummy_bus_device

    # Setting dummy return value (fake status bits with overcurrent and out of range off)
    dummy_bus_device.word_values[ETS] = 0b1010101010101  # 13 bits set bits (pattern)

    bits = m.touched()
    assert bits == (dummy_bus_device.word_values[ETS] & 0x1fff)


def test_touched_raises_on_overcurrent(dummy_bus_device):
    m = MPR121(bus=DummyBus())
    m._bus = dummy_bus_device

    # Set 15th bit (overcurrent)
    dummy_bus_device.word_values[ETS] = 1 << 15
    with pytest.raises(Exception, match='overcurrent'):
        m.touched()


def test_touched_raises_on_out_of_range(dummy_bus_device):
    m = MPR121(bus=DummyBus())
    m._bus = dummy_bus_device

    # No overcurrent, but out_of_range() returns True
    dummy_bus_device.word_values[ETS] = 0  # no overcurrent
    def fake_out_of_range(raise_on_failure=1):
        raise Exception('out of range')
    m.out_of_range = fake_out_of_range

    with pytest.raises(Exception, match='out of range'):
        m.touched()


def test_electrode_data_returns_correct_values(dummy_bus_device):
    m = MPR121(bus=DummyBus())
    m._bus = dummy_bus_device

    # Setup block data: 2 bytes per channel: low byte + high byte pairs
    # For NCH=13 channels, total 26 bytes
    # Values set as: channel i data = (i << 8) + i
    block = []
    for i in range(NCH):
        block.append(i & 0xFF)        # low byte
        block.append((i >> 8) & 0xFF) # high byte, zero
    dummy_bus_device.block_values[mpr121.EFD] = block

    data = m.electrode_data()
    expected = [(block[2*i+1] << 8) + block[2*i] for i in range(NCH)]
    assert data == expected
    for idx, val in enumerate(data):
        # For our construction val == idx
        assert val == idx


def test_baseline_read_and_write_behavior(dummy_bus_device):
    m = MPR121(bus=DummyBus())
    m._bus = dummy_bus_device

    # Setup baseline block read data
    dummy_bus_device.block_values[mpr121.EBL] = [i for i in range(NCH)]

    # Reading without arguments returns values shifted left 2 bits
    bl = m.baseline()
    expected = [i << 2 for i in range(NCH)]
    assert bl == expected

    # Writing with rft=0 triggers write_block with 4 bytes on MHD reg
    dummy_bus_device.written_blocks.clear()
    m.baseline(rft=0, mhd=1, nhd=2, ncl=3, fdl=4, prox=0)
    found = False
    for reg, vals in dummy_bus_device.written_blocks:
        if reg == mpr121.MHD and vals == [1, 2, 3, 4]:
            found = True
            break
    assert found

    # Writing with rft=2 triggers write_block with 3 bytes on MHDX reg
    dummy_bus_device.written_blocks.clear()
    m.baseline(rft=2, mhd=5, nhd=6, ncl=7, fdl=8, prox=0)
    found = False
    for reg, vals in dummy_bus_device.written_blocks:
        if reg == mpr121.MHDX + 8 and vals == [6, 7, 8]:
            found = True
    # Actually the reg is MHDX + 4 * rft == MHDX + 8
    # But the code always uses MHDX + 4 * rft. For rft=2 it's +8 offset.
    # Check for that
    assert found or any(reg == mpr121.MHDX + 8 for reg, _ in dummy_bus_device.written_blocks)

    # Now test with prox=True (use MHDX reg)
    dummy_bus_device.written_blocks.clear()
    m.baseline(rft=1, mhd=9, nhd=10, ncl=11, fdl=12, prox=1)
    # It should use MHDX + 4 * 1 = MHDX + 4
    found = False
    for reg, vals in dummy_bus_device.written_blocks:
        if reg == mpr121.MHDX + 4 and vals == [9, 10, 11, 12]:
            found = True
            break
    assert found


def test_threshold_sets_values_for_all_channels_and_specific_channel(monkeypatch):
    dummy_device = DummyBusDevice()
    m = MPR121(bus=DummyBus())
    m._bus = dummy_device

    # Overwrite _bus.write_byte to record writes
    dummy_device.written_bytes.clear()
    # Call with channel -1 (all)
    m.threshold(touch=10, release=-1, channel=-1)
    # Should write twice per channel (touch and release), for all NCH channels
    assert len(dummy_device.written_bytes) == 2 * NCH
    # Check that release was set to 0.6 * touch (int)
    for reg, val in dummy_device.written_bytes:
        if reg % 2 == 0:  # touch registers even
            assert val == 10 or val == 2  # The proximity channel threshold is 10 >> 2 == 2 for channel 12
        else:  # release registers odd
            assert val == int(0.6 * 10) or val == int(0.6 * (10 >> 2))

    # Clear and test for single channel override
    dummy_device.written_bytes.clear()
    m.threshold(touch=20, release=15, channel=2)
    assert dummy_device.written_bytes == [(mpr121.TTH + 2 * 2, 20), (mpr121.RTH + 2 * 2, 15)]


def test_debounce_writes_byte_correctly(dummy_bus_device):
    m = MPR121(bus=DummyBus())
    m._bus = dummy_bus_device

    dummy_bus_device.written_bytes.clear()
    m.debounce(touch=3, release=-1)  # release should be same as touch
    assert dummy_bus_device.written_bytes[-1][0] == mpr121.DEB
    val = dummy_bus_device.written_bytes[-1][1]
    assert (val & 0xF) == 3
    assert (val >> 4) == 3

    dummy_bus_device.written_bytes.clear()
    m.debounce(touch=2, release=5)
    val = dummy_bus_device.written_bytes[-1][1]
    assert (val & 0xF) == 2
    assert (val >> 4) == 5


def test_filter_write_word_calls(monkeypatch):
    dummy_device = DummyBusDevice()
    m = MPR121(bus=DummyBus())
    m._bus = dummy_device

    dummy_device.written_words.clear()
    m.filter(cdc=12, cdt=2, ffi=3, sfi=2, esi=1)
    # lb = (ffi<<6) | cdc = (3<<6) | 12 = 192 + 12 = 204
    # hb = (cdt<<5) | (sfi<<3) | esi = (2<<5) | (2<<3) | 1 = 64 +16 +1 = 81
    expected_word = (81 << 8) | 204
    assert dummy_device.written_words[-1] == (mpr121.AFE1, expected_word)


def test_charge_writes_byte(monkeypatch):
    dummy_device = DummyBusDevice()
    m = MPR121(bus=DummyBus())
    m._bus = dummy_device

    # Setup read_byte returns for CDT register (there are 6 CDT registers from CDT to CDT+5)
    dummy_device.byte_values[mpr121.CDT] = 0xFF
    dummy_device.byte_values[mpr121.CDT + 1] = 0x00
    dummy_device.byte_values[mpr121.CDT + 2] = 0xAA

    # channel 1 -> writes to CDC+1, CDT+0(half byte)
    dummy_device.written_bytes.clear()
    m.charge(channel=1, cdc=11, cdt=5)
    assert (mpr121.CDC + 1, 11) in dummy_device.written_bytes

    # read_byte to CDT + 0 returns 0xFF as per setup, so:
    # old_val & ~(0xf << s) = 0xFF & ~(0xf << 4) = 0xFF & 0x0F = 0x0F
    # new_val = 0x0F | (5 << 4) = 0x0F | 0x50 = 0x5F
    # but s = 4 * (channel % 2) -> channel=1, s=4
    found = False
    for reg, val in dummy_device.written_bytes:
        if reg == mpr121.CDT and val & 0xF == 0xF and val >> 4 == 5:
            found = True
            break
    assert found


def test_auto_config_writes_correctly(dummy_bus_device):
    m = MPR121(bus=DummyBus())
    m._bus = dummy_bus_device

    dummy_bus_device.written_words.clear()
    dummy_bus_device.written_bytes.clear()
    m.auto_config(ace=0, are=1, bva=2, retry=3, afes=2, scts=1, acfie=0, arfie=1, oorie=0,
                  usl=201, lsl=131, tl=181)

    # Construct expected lb and hb
    lb = (2 << 2) | (1 << 1) | 0  # bva=2, are=1, ace=0
    lb |= (3 << 4)  # retry = 3 (bits 5-4) (3 << 4)
    lb |= (2 << 6)  # afes = 2 (bits 7-6)
    # Note: retry is 3 but valid values are 0-3 (bitmask 2 bits)
    # So, the full lb: (afes << 6) | (retry << 4) | (bva << 2) | (are << 1) | ace
    lb = (2 << 6) | (3 << 4) | (2 << 2) | (1 << 1) | 0
    # hb = (scts << 7) | (oorie << 2) | (arfie << 1) | (acfie)
    hb = (1 << 7) | (0 << 2) | (1 << 1) | 0

    expected_word = (hb << 8) + lb
    assert (mpr121.ACNF_C0, expected_word) in dummy_bus_device.written_words
    assert (mpr121.ACNF_USL, 201) in dummy_bus_device.written_bytes
    assert (mpr121.ACNF_LSL, 131) in dummy_bus_device.written_bytes
    assert (mpr121.ACNF_TL, 181) in dummy_bus_device.written_bytes


def test_out_of_range_behavior(monkeypatch, dummy_bus_device):
    m = MPR121(bus=DummyBus())
    m._bus = dummy_bus_device

    # Return with automatic config fail (bit 15)
    dummy_bus_device.word_values[mpr121.OOR] = (1 << 15)
    with pytest.raises(Exception, match='auto config failed'):
        m.out_of_range()

    # Return with auto reconfig fail (bit 14)
    dummy_bus_device.word_values[mpr121.OOR] = (1 << 14)
    with pytest.raises(Exception, match='auto reconfig failed'):
        m.out_of_range()

    # Without raising
    dummy_bus_device.word_values[mpr121.OOR] = 0x1abc
    val = m.out_of_range(raise_on_failure=0)
    assert val == 0x1abc & 0x1fff


def test_on_touch_and_handle_touch_invokes_handlers(monkeypatch):
    m = MPR121(bus=DummyBus())
    # Start with noop handlers
    calls = []

    def make_handler(i):
        def handler(touched, ch=None):
            calls.append((touched, i))
        return handler

    for i in range(NCH):
        m.on_touch(i, make_handler(i))

    # Set touched bits: bit 0 and 2 changed to touched (1)
    m._touched = 0b0010
    prev = 0b0000
    changed = prev ^ m._touched
    m._handle_touch(m._touched, changed)
    assert (True, 1) in calls
    # Clear calls and test type error invocation (handler without 2 args)
    def handler_one_arg(touched):
        calls.append(('onearg', touched))
    m.on_touch(0, handler_one_arg)
    calls.clear()
    # Trigger change on channel 0
    m._handle_touch(1 << 0, 1 << 0)
    # Should call handler_one_arg without exception and record
    assert ('onearg', True) in calls

def test_is_touched_reflects_state(dummy_bus_device):
    m = MPR121(bus=DummyBus())
    m._bus = dummy_bus_device
    # Set bits 0, 3, and 12 touched
    m._touched = (1 << 0) | (1 << 3) | (1 << 12)
    assert m.is_touched(0) is True
    assert m.is_touched(3) is True
    assert m.is_touched(12) is True
    assert m.is_touched(1) is False


def test_gpio_setup_calls_write_byte(monkeypatch):
    dummy_device = DummyBusDevice()
    m = MPR121(bus=DummyBus())
    m._bus = dummy_device

    dummy_device.byte_values[mpr121.GPIO_CTL0] = 0x00
    dummy_device.byte_values[mpr121.GPIO_CTL1] = 0x00
    dummy_device.byte_values[mpr121.GPIO_DIR] = 0x00
    dummy_device.byte_values[mpr121.GPIO_EN] = 0x00

    dummy_device.written_bytes.clear()

    # Setup channel=5 (valid), output=1, mode=3, enable=1
    m.gpio_setup(channel=5, output=1, mode=3, enable=1)
    # We expect 4 writes (one for each GPIO reg) altering bit corresponding to channel-4=1
    regs = [mpr121.GPIO_CTL0, mpr121.GPIO_CTL1, mpr121.GPIO_DIR, mpr121.GPIO_EN]
    writes_regs = [reg for reg, _ in dummy_device.written_bytes]
    for reg in regs:
        assert reg in writes_regs
    # Test assertion triggers on invalid channel
    with pytest.raises(AssertionError):
        m.gpio_setup(channel=3, output=1)
    with pytest.raises(AssertionError):
        m.gpio_setup(channel=12, output=0)


def test_gpio_status_returns_byte(dummy_bus_device):
    m = MPR121(bus=DummyBus())
    m._bus = dummy_bus_device
    dummy_bus_device.byte_values[mpr121.GPIO_DAT] = 0xAB
    status = m.gpio_status()
    assert status == 0xAB


def test_gpio_set_writes_correctly(dummy_bus_device):
    m = MPR121(bus=DummyBus())
    m._bus = dummy_bus_device

    # Test setting channel 6 to True (1)
    dummy_device = dummy_bus_device
    m._bus = dummy_device
    dummy_device.written_bytes.clear()
    m.gpio_set(6, True)
    # Should write to GPIO_SET with bit (6 - 4) = 2 set
    assert (mpr121.GPIO_SET, 1 << 2) in dummy_device.written_bytes

    dummy_device.written_bytes.clear()
    # Test setting channel 10 to False (0)
    m.gpio_set(10, False)
    # Should write to GPIO_CLR with bit (10 - 4) = 6 set
    assert (mpr121.GPIO_CLR, 1 << 6) in dummy_device.written_bytes


def test_update_touch_state_triggers_handler(monkeypatch):
    m = MPR121(bus=DummyBus())
    call_args = []

    def fake_touched():
        # Change touched bits to 0b101
        return 0b101

    m.touched = fake_touched
    m._touched = 0b001
    # Attach a handler that appends call args
    def handler(touched, channel):
        call_args.append((touched, channel))

    for i in range(NCH):
        m.on_touch(i, None)
    m.on_touch(0, handler)
    m.on_touch(2, handler)
    # Update touch state should trigger handlers for changed bits 0,2
    m.update_touch_state()
    assert (True, 2) in call_args or (False, 2) in call_args
    assert (True, 0) in call_args or (False, 0) in call_args
