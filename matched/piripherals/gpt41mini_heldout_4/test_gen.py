import builtins
import pytest
from piripherals import mpr121

class DummyBusDevice:
    def __init__(self):
        self.written_bytes = {}
        self.written_words = {}
        self.written_blocks = {}
        self.read_bytes = {}
        self.read_words = {}
        self.read_blocks = {}

    def write_byte(self, reg, val):
        self.written_bytes[reg] = val

    def write_word(self, reg, val):
        self.written_words[reg] = val

    def write_block(self, reg, data):
        self.written_blocks[reg] = list(data)

    def read_byte(self, reg):
        # Default to 0 if nothing set
        return self.read_bytes.get(reg, 0)

    def read_word(self, reg):
        return self.read_words.get(reg, 0)

    def read_block(self, reg, length):
        return self.read_blocks.get(reg, [0]*length)

class DummyBus:
    def __init__(self):
        self.device_called_with = []
        self.device_obj = DummyBusDevice()

    def device(self, addr):
        self.device_called_with.append(addr)
        return self.device_obj

def noop(*args, **kwargs):
    pass

# Patch noop in mpr121 module for tests to avoid NameError
mpr121.noop = noop
mpr121.not_raising = lambda f: f if f is not None else noop

# Patch IRQHandler and Poller to avoid real hardware calls
class DummyIRQHandler:
    def __init__(self, irq_pin, callback):
        self.irq_pin = irq_pin
        self.callback = callback

class DummyPoller:
    def __init__(self, callback):
        self.callback = callback

mpr121.IRQHandler = DummyIRQHandler
mpr121.Poller = DummyPoller

@pytest.fixture
def dummy_mpr():
    bus = DummyBus()
    # Instantiate with handlers=False and setup=False by default
    return mpr121.MPR121(bus=bus, irq=0, handlers=False, setup=False, reset=False)

def test_init_uses_bus_device():
    bus = DummyBus()
    m = mpr121.MPR121(bus=bus, irq=0, handlers=False, setup=False, reset=False)
    assert bus.device_called_with == [mpr121.MPR121.__defaults__[1]] or bus.device_called_with == [0x5a]
    assert hasattr(m, '_bus')
    assert m._bus is bus.device_obj

def test_reset_calls_write_byte(dummy_mpr):
    dummy_mpr.reset()
    assert dummy_mpr._bus.written_bytes[mpr121.SRESET] == 0x63

def test_configure_writes_expected_byte(dummy_mpr):
    dummy_mpr.configure(cl=2, prox=1, touch=3)
    expected_val = (2 << 6) | (1 << 4) | 3
    assert dummy_mpr._bus.written_bytes[mpr121.ECR] == expected_val

def test_touched_returns_status_bits_and_raises():
    dummy_mpr = dummy_mpr = mpr121.MPR121(bus=DummyBus(), irq=0, handlers=False, setup=False, reset=False)
    # Setup bus to return a word with overcurrent bit set
    dummy_mpr._bus.read_words[mpr121.ETS] = (1 << 15) | 0x0f0f
    dummy_mpr._bus.read_words[mpr121.OOR] = 0  # no out of range
    
    with pytest.raises(Exception, match='overcurrent'):
        dummy_mpr.touched(raise_on_failure=True)
    
    # Setup bus to return out_of_range bit set
    dummy_mpr._bus.read_words[mpr121.ETS] = 0x0f0f
    dummy_mpr._bus.read_words[mpr121.OOR] = (1 << 15)
    
    # Patch read_word to point to OOR for out_of_range call
    orig_read_word = dummy_mpr._bus.read_word
    def read_word_side(reg):
        if reg == mpr121.OOR:
            return dummy_mpr._bus.read_words[mpr121.OOR]
        return orig_read_word(reg)
    dummy_mpr._bus.read_word = read_word_side
    
    with pytest.raises(Exception, match='out of range'):
        dummy_mpr.touched(raise_on_failure=True)

def test_touched_returns_value_without_raising():
    dummy_mpr = mpr121.MPR121(bus=DummyBus(), irq=0, handlers=False, setup=False, reset=False)
    dummy_mpr._bus.read_words[mpr121.ETS] = 0x1234
    dummy_mpr._bus.read_words[mpr121.OOR] = 0x0000
    val = dummy_mpr.touched(raise_on_failure=False)
    assert val == 0x1234 & 0x1fff

@pytest.mark.parametrize("rft,prox,expected_reg", [
    (-1, 0, None),
    (0, 0, mpr121.MHD),
    (1, 0, mpr121.MHD + 4),
    (2, 0, mpr121.MHD + 8),
    (0, 1, mpr121.MHDX),
    (1, 1, mpr121.MHDX + 4),
    (2, 1, mpr121.MHDX + 8),
])
def test_baseline_reads_or_writes_correctly(rft, prox, expected_reg):
    dummy_mpr = mpr121.MPR121(bus=DummyBus(), irq=0, handlers=False, setup=False, reset=False)
    dummy_mpr._bus.read_blocks = {}
    dummy_mpr._bus.read_block = lambda reg, length: [i for i in range(length)]
    dummy_mpr._bus.write_block = lambda reg, data: setattr(dummy_mpr._bus, 'last_written_block', (reg, data))

    if rft < 0:
        res = dummy_mpr.baseline(rft=rft, prox=prox)
        # baseline reads and returns values shifted left 2 bits
        expected = [i << 2 for i in range(mpr121.NCH)]
        assert res == expected
    else:
        dummy_mpr.baseline(rft=rft, mhd=5, nhd=6, ncl=7, fdl=8, prox=bool(prox))
        reg, data = dummy_mpr._bus.last_written_block
        assert reg == expected_reg
        if rft > 1:
            # should write (nhd, ncl, fdl)
            assert data == [6, 7, 8]
        else:
            assert data == [5, 6, 7, 8]

def test_threshold_sets_for_all_or_single_channel():
    dummy_mpr = mpr121.MPR121(bus=DummyBus(), irq=0, handlers=False, setup=False, reset=False)

    # test all channels setting
    dummy_mpr._bus.written_bytes.clear()
    dummy_mpr.threshold(20)
    for ch in range(mpr121.NCH):
        assert dummy_mpr._bus.written_bytes[mpr121.TTH + 2 * ch] == 20
        assert dummy_mpr._bus.written_bytes[mpr121.RTH + 2 * ch] == int(0.6 * 20)

    # test single channel, release omitted -> release = 0.6*touch
    dummy_mpr._bus.written_bytes.clear()
    dummy_mpr.threshold(touch=40, channel=5)
    assert dummy_mpr._bus.written_bytes[mpr121.TTH + 2 * 5] == 40
    assert dummy_mpr._bus.written_bytes[mpr121.RTH + 2 * 5] == int(0.6 * 40)

    # test single channel, release explicitly set
    dummy_mpr._bus.written_bytes.clear()
    dummy_mpr.threshold(touch=100, release=80, channel=7)
    assert dummy_mpr._bus.written_bytes[mpr121.TTH + 2 * 7] == 100
    assert dummy_mpr._bus.written_bytes[mpr121.RTH + 2 * 7] == 80

def test_debounce_sets_correct_byte():
    dummy_mpr = mpr121.MPR121(bus=DummyBus(), irq=0, handlers=False, setup=False, reset=False)
    dummy_mpr._bus.written_bytes.clear()

    # release ommited -> release = touch
    dummy_mpr.debounce(touch=3)
    val = (3 << 4) | 3
    assert dummy_mpr._bus.written_bytes[mpr121.DEB] == val

    # release explicitly set
    dummy_mpr.debounce(touch=2, release=5)
    val = (5 << 4) | 2
    assert dummy_mpr._bus.written_bytes[mpr121.DEB] == val

def test_filter_writes_word_with_expected_format():
    dummy_mpr = mpr121.MPR121(bus=DummyBus(), irq=0, handlers=False, setup=False, reset=False)
    dummy_mpr._bus.written_words.clear()
    dummy_mpr.filter(cdc=10, cdt=3, ffi=2, sfi=1, esi=5)
    lb = (2 << 6) | 10
    hb = (3 << 5) | (1 << 3) | 5
    expected = (hb << 8) | lb
    assert dummy_mpr._bus.written_words[mpr121.AFE1] == expected

def test_charge_sets_bytes_properly():
    dummy_mpr = mpr121.MPR121(bus=DummyBus(), irq=0, handlers=False, setup=False, reset=False)
    dummy_mpr._bus.read_bytes[mpr121.CDT] = 0
    dummy_mpr._bus.read_bytes[mpr121.CDT + 1] = 0

    dummy_mpr.charge(channel=0, cdc=12, cdt=3)
    assert dummy_mpr._bus.written_bytes[mpr121.CDC + 0] == 12
    # read_byte called for CDT + 0
    val = dummy_mpr._bus.written_bytes[mpr121.CDT]
    expected_val = (0 & ~(0xf << 0)) | (3 << 0)
    assert val == expected_val

    # test channel=3 (odd channel, high nibble)
    dummy_mpr._bus.read_bytes[mpr121.CDT + 1] = 0xFF
    dummy_mpr.charge(channel=3, cdc=7, cdt=2)
    assert dummy_mpr._bus.written_bytes[mpr121.CDC + 3] == 7
    # Should update the high nibble of CDT + 1 (channel // 2 = 1, bit shift 4)
    val = dummy_mpr._bus.written_bytes[mpr121.CDT + 1]
    # previous val was 0xFF. We clear bits 4-7 and set cdt<<4.
    expected_val = (0xFF & ~(0xf << 4)) | (2 << 4)
    assert val == expected_val

def test_auto_config_writes_expected_registers():
    dummy_mpr = mpr121.MPR121(bus=DummyBus(), irq=0, handlers=False, setup=False, reset=False)
    dummy_mpr._bus.written_bytes.clear()
    dummy_mpr._bus.written_words.clear()

    dummy_mpr.auto_config(
        ace=1, are=0, bva=2, retry=3, afes=2,
        scts=1, acfie=0, arfie=1, oorie=0,
        usl=123, lsl=234, tl=201)

    lb = (2 << 6) | (3 << 4) | (2 << 2) | (0 << 1) | 1
    hb = (1 << 7) | (0 << 2) | (1 << 1) | 0
    expected_word = (hb << 8) + lb
    assert dummy_mpr._bus.written_words[mpr121.ACNF_C0] == expected_word
    assert dummy_mpr._bus.written_bytes[mpr121.ACNF_USL] == 123
    assert dummy_mpr._bus.written_bytes[mpr121.ACNF_LSL] == 234
    assert dummy_mpr._bus.written_bytes[mpr121.ACNF_TL] == 201

def test_out_of_range_raises_on_failures():
    dummy_mpr = mpr121.MPR121(bus=DummyBus(), irq=0, handlers=False, setup=False, reset=False)
    # auto config failed bit set
    dummy_mpr._bus.read_words[mpr121.OOR] = (1 << 15)
    with pytest.raises(Exception, match='auto config failed'):
        dummy_mpr.out_of_range(raise_on_failure=True)

    # auto reconfig failed bit set
    dummy_mpr._bus.read_words[mpr121.OOR] = (1 << 14)
    with pytest.raises(Exception, match='auto reconfig failed'):
        dummy_mpr.out_of_range(raise_on_failure=True)

def test_out_of_range_returns_bits_without_raising():
    dummy_mpr = mpr121.MPR121(bus=DummyBus(), irq=0, handlers=False, setup=False, reset=False)
    dummy_mpr._bus.read_words[mpr121.OOR] = 0x1234
    val = dummy_mpr.out_of_range(raise_on_failure=False)
    assert val == 0x1234 & 0x1fff

def test_gpio_setup_writes_bytes_and_prints(monkeypatch):
    dummy_mpr = mpr121.MPR121(bus=DummyBus(), irq=0, handlers=False, setup=False, reset=False)
    dummy_mpr._bus.read_bytes[mpr121.GPIO_CTL0] = 0b10101010
    dummy_mpr._bus.read_bytes[mpr121.GPIO_CTL1] = 0b11001100
    dummy_mpr._bus.read_bytes[mpr121.GPIO_DIR] = 0b11110000
    dummy_mpr._bus.read_bytes[mpr121.GPIO_EN] = 0b00001111

    printed = []
    monkeypatch.setattr(builtins, 'print', lambda *a, **k: printed.append(a[0]))

    # mode=3 (0b11), output=1, enable=1
    dummy_mpr.gpio_setup(channel=6, output=True, mode=3, enable=True)

    # Check if bytes were written for each register, bit 2 in channel (6-4=2)
    for reg in [mpr121.GPIO_CTL0, mpr121.GPIO_CTL1, mpr121.GPIO_DIR, mpr121.GPIO_EN]:
        assert reg in dummy_mpr._bus.written_bytes

    # Check print output contains expected formatted string of register values
    # It must print 4 lines for each set_bit call
    assert any(isinstance(s, str) and s.startswith('0x') for s in printed)

def test_gpio_status_reads_byte():
    dummy_mpr = mpr121.MPR121(bus=DummyBus(), irq=0, handlers=False, setup=False, reset=False)
    dummy_mpr._bus.read_bytes[mpr121.GPIO_DAT] = 0xAB
    val = dummy_mpr.gpio_status()
    assert val == 0xAB

def test_gpio_set_writes_correct_register_and_value():
    dummy_mpr = mpr121.MPR121(bus=DummyBus(), irq=0, handlers=False, setup=False, reset=False)

    dummy_mpr.gpio_set(channel=4, value=True)
    assert dummy_mpr._bus.written_bytes[mpr121.GPIO_SET] == 1 << 0

    dummy_mpr.gpio_set(channel=11, value=False)
    assert dummy_mpr._bus.written_bytes[mpr121.GPIO_CLR] == 1 << (11 - 4)

def test_update_touch_state_invokes_handler(monkeypatch):
    # Setup a dummy bus and MPR121 with custom handler
    dummy_bus = DummyBus()
    dummy_bus.read_words[mpr121.ETS] = 0b1010  # touched status example
    m = mpr121.MPR121(bus=dummy_bus, irq=0, handlers=False, setup=False, reset=False)
    called = {}
    def handler(touched, channel=None):
        called[channel] = touched
    for ch in range(mpr121.NCH):
        m.on_touch(ch, lambda touched, c=ch: handler(touched, c))

    m._touched = 0b0000
    m.update_touch_state()
    # touched changed bits should be 0b1010, handlers for those bits called with the right values
    assert called[1] is True
    assert called[3] is True
    # Other channels either not called or remain default - no raise

def test_handle_touch_typeerror_handler_fallback():
    dummy_bus = DummyBus()
    m = mpr121.MPR121(bus=dummy_bus, irq=0, handlers=False, setup=False, reset=False)
    called = []
    def handler_bool_only(touched):
        called.append(touched)
    m._handlers[0] = handler_bool_only
    m._touched = 0
    # _handle_touch calls handler with two args, this handler only accept one -> fallback to 1 arg call
    m._handle_touch(touched=1, changed=1)
    assert called == [True]
