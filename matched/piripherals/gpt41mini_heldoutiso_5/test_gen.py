import builtins
import pytest
from piripherals.mpr121 import MPR121, NCH, SRESET, ECR, TTH, RTH, DEB, AFE1, CDC, CDT, \
    ACNF_C0, ACNF_USL, ACNF_LSL, ACNF_TL, ETS, OOR, EFD, EBL, GPIO_CTL0, GPIO_CTL1, GPIO_DIR, GPIO_EN, GPIO_SET, GPIO_CLR, GPIO_DAT

class DummyBusDevice:
    def __init__(self):
        self.writes = []
        self.read_word_data = {}
        self.read_byte_data = {}
        self.read_block_data = {}
        self.read_byte_calls = []
        self.read_word_calls = []
        self.read_block_calls = []
        self.bytes_to_return = []
        self.byte_counter = 0

    def write_byte(self, reg, val):
        self.writes.append(('write_byte', reg, val))

    def write_word(self, reg, val):
        self.writes.append(('write_word', reg, val))

    def write_block(self, reg, vals):
        self.writes.append(('write_block', reg, tuple(vals)))

    def read_word(self, reg):
        self.read_word_calls.append(reg)
        # Provide default dummy values if not preset
        return self.read_word_data.get(reg, 0x0000)

    def read_byte(self, reg):
        self.read_byte_calls.append(reg)
        # Provide default dummy values if not preset
        return self.read_byte_data.get(reg, 0)

    def read_block(self, reg, length):
        self.read_block_calls.append((reg, length))
        # Return preset block or zeros if not preset
        return self.read_block_data.get(reg, [0]*length)


class DummyBus:
    def __init__(self):
        self.device_called_with = []
        self.device_instance = DummyBusDevice()

    def device(self, addr):
        self.device_called_with.append(addr)
        return self.device_instance


@pytest.fixture
def mpr121_with_dummybus(monkeypatch):
    """Create MPR121 instance with dummy bus device."""
    dummy_bus = DummyBus()
    monkeypatch.setattr('piripherals.mpr121.Bus', lambda bus=None: dummy_bus)
    mpr = MPR121(bus=1, irq=0, handlers=0, setup=0, reset=0)
    # Overwrite _bus to dummy device directly for further control
    mpr._bus = dummy_bus.device_instance
    # Replace handlers with identifiable noops to test handler calling
    mpr._handlers = [lambda *args, **kwargs: None for _ in range(NCH)]
    return mpr, dummy_bus.device_instance


def test_reset_writes_correct_byte(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    mpr.reset()
    assert ('write_byte', SRESET, 0x63) in device.writes


def test_configure_writes_correct_ecr_value(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    mpr.configure(cl=2, prox=3, touch=7)
    expected_val = (2 << 6) | (3 << 4) | 7
    assert ('write_byte', ECR, expected_val) in device.writes


def test_touched_returns_correct_bits_and_raises(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    # Setup read word to simulate touch bits and failure bits
    # Overcurrent bit set (bit 15)
    device.read_word_data[ETS] = (1 << 15) | 0x123
    with pytest.raises(Exception, match='overcurrent'):
        mpr.touched(raise_on_failure=True)
    # Remove overcurrent and set out_of_range to true
    device.read_word_data[ETS] = 0x123
    device.read_word_data[OOR] = (1 << 15)
    # Patch out_of_range to return raise exception
    orig_out_of_range = mpr.out_of_range
    def raise_oor():
        raise Exception("out of range")
    mpr.out_of_range = raise_oor
    with pytest.raises(Exception, match='out of range'):
        mpr.touched(raise_on_failure=True)
    # Reset patch and test normal return (mask 0x1fff)
    mpr.out_of_range = orig_out_of_range
    device.read_word_data[ETS] = 0xA5A5
    val = mpr.touched(raise_on_failure=False)
    assert val == (0xA5A5 & 0x1fff)


def test_out_of_range_returns_bits_and_raises(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    # test auto_config_failed bit - bit 15
    device.read_word_data[OOR] = (1 << 15)
    with pytest.raises(Exception, match='auto config failed'):
        mpr.out_of_range(raise_on_failure=True)
    # test auto_reconfig_failed bit - bit 14
    device.read_word_data[OOR] = (1 << 14)
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr.out_of_range(raise_on_failure=True)
    # normal return with masked bits
    device.read_word_data[OOR] = 0xABCD
    val = mpr.out_of_range(raise_on_failure=False)
    assert val == (0xABCD & 0x1fff)


def test_electrode_data_reads_block_and_returns_correct(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    # Prepare 26 bytes (2*NCH) with incrementing pattern
    bytes_block = []
    for i in range(NCH):
        low = i
        high = i + 1
        bytes_block.extend([low, high])
    device.read_block_data[EFD] = bytes_block
    data = mpr.electrode_data()
    # The data is constructed (high << 8) + low per channel
    for i, val in enumerate(data):
        expected = ((i+1) << 8) + i
        assert val == expected


def test_baseline_no_args_reads_correctly(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    block = [i for i in range(NCH)]
    device.read_block_data[EBL] = block
    baseline = mpr.baseline()
    # baseline returns bytes shifted left by 2 bits
    for i, b in enumerate(baseline):
        assert b == (block[i] << 2)


def test_baseline_with_args_writes_correctly(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    device.writes.clear()
    # rft 0 (rising), prox False: 4 writes to MHD + 4*rft = MHD
    mpr.baseline(rft=0, mhd=10, nhd=20, ncl=30, fdl=40, prox=False)
    assert any(w for w in device.writes if w[0] == 'write_block')
    reg, vals = device.writes[-1][1], device.writes[-1][2]
    assert reg == 0x2b  # MHD
    assert tuple(vals) == (10, 20, 30, 40)

    device.writes.clear()
    # rft 2 (touched), prox False: 3 byte write block to MHD+4*rft = MHD+8
    mpr.baseline(rft=2, nhd=21, ncl=31, fdl=41, prox=False)
    reg2 = 0x2b + 4 * 2
    assert any(w for w in device.writes if w[0] == 'write_block')
    reg, vals = device.writes[-1][1], device.writes[-1][2]
    assert reg == reg2
    # length 3 tuple
    assert len(vals) == 3
    assert vals == (21, 31, 41)

    device.writes.clear()
    # rft 1, prox True: reg is MHDX + 4*rft
    mpr.baseline(rft=1, mhd=5, nhd=6, ncl=7, fdl=8, prox=True)
    reg3 = 0x36 + 4 * 1
    reg_write = None
    for w in device.writes:
        if w[0] == 'write_block':
            reg_write = w[1]
    assert reg_write == reg3


def test_threshold_sets_all_and_single_channel(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    device.writes.clear()
    mpr.threshold(touch=100)
    # Should write touch and release for all 13 channels
    w = device.writes
    # 13 channels * 2 writes each = 26 writes
    assert len(w) == 26
    # touch and release pairs for channel 0 written first
    assert w[0][1] == (TTH + 0)
    assert w[1][1] == (RTH + 0)
    # Check default release = int(0.6 * touch) = 60
    assert w[0][2] == 100
    assert w[1][2] == 60

    device.writes.clear()
    # Set thresholds for channel 5 with given release
    mpr.threshold(touch=80, release=50, channel=5)
    # Should write only twice
    assert len(device.writes) == 2
    assert device.writes[0][1] == TTH + 10  # channel 5 offset *2
    assert device.writes[0][2] == 80
    assert device.writes[1][1] == RTH + 10
    assert device.writes[1][2] == 50


def test_debounce_writes_byte_correctly(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    device.writes.clear()
    mpr.debounce(touch=3)
    # release defaults to touch value
    expected_val = (3 << 4) | 3
    assert ('write_byte', DEB, expected_val) in device.writes

    device.writes.clear()
    mpr.debounce(touch=2, release=5)
    expected_val = (5 << 4) | 2
    assert ('write_byte', DEB, expected_val) in device.writes


def test_filter_writes_word_correctly(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    device.writes.clear()
    mpr.filter(cdc=10, cdt=3, ffi=2, sfi=1, esi=4)
    lb = (2 << 6) | 10
    hb = (3 << 5) | (1 << 3) | 4
    expected_val = (hb << 8) | lb
    assert ('write_word', AFE1, expected_val) in device.writes


def test_charge_writes_bytes_correctly(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    device.writes.clear()
    # Setup read_byte to return 0 for register reads in charge
    for reg in range(CDT, CDT + 6):
        device.read_byte_data[reg] = 0x00
    # channel even
    mpr.charge(channel=4, cdc=12, cdt=7)
    # Should write to CDC+channel and CDT+channel//2 registers
    assert ('write_byte', CDC + 4, 12) in device.writes
    # Find write_byte to CDT+2 (4//2=2) 
    write_to_cdt = [w for w in device.writes if w[0] == 'write_byte' and w[1] == CDT + 2]
    assert len(write_to_cdt) == 1
    # Check that written value has cdt in correct nibble shifted
    # cdt=7 << (4*(4%2)) = cdt << 0 = 7
    written_val = write_to_cdt[0][2]
    assert (written_val & 0x0f) == 7

    device.writes.clear()
    # channel odd
    device.read_byte_data[CDT + 3] = 0xFF
    mpr.charge(channel=7, cdc=8, cdt=3)
    assert ('write_byte', CDC + 7, 8) in device.writes
    write_to_cdt_3 = [w for w in device.writes if w[0] == 'write_byte' and w[1] == CDT + 3]
    assert len(write_to_cdt_3) == 1
    # cdt shifted by 4*(7%2) = 4*1=4 bits
    expected_nibble = 3 << 4
    # Lower nibble preserved, only upper nibble replaced with cdt * 16
    # read_byte_data returns 0xFF so mask & ~(0xf << s) = clears upper nibble then OR new
    written_val = write_to_cdt_3[0][2]
    assert (written_val & 0xF0) == expected_nibble


def test_auto_config_writes_bytes_and_word_correctly(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    device.writes.clear()
    mpr.auto_config(
        ace=1, are=1, bva=2, retry=3, afes=2, scts=1, acfie=0, arfie=1, oorie=1,
        usl=123, lsl=45, tl=67)
    # Check write_word to ACNF_C0 with proper bits set
    wword = [w for w in device.writes if w[0] == 'write_word' and w[1] == ACNF_C0]
    assert len(wword) == 1
    val = wword[0][2]
    lb = (2 << 2) | (3 << 4) | (1 << 1) | 1  # (afes <<6) deferred
    lb = (2 << 6) | (3 << 4) | (2 << 2) | (1 << 1) | 1
    # Let's recalc lb byte specifically:
    lb_expected = (2 << 6) | (3 << 4) | (2 << 2) | (1 << 1) | 1
    hb_expected = (1 << 7) | (1 << 2) | (1 << 1) | 0
    expected_val = (hb_expected << 8) + lb_expected
    assert val == expected_val
    # Check write_byte calls for USL, LSL, TL
    wbytes = [w for w in device.writes if w[0] == 'write_byte']
    regs = [w[1] for w in wbytes]
    assert ACNF_USL in regs
    assert ACNF_LSL in regs
    assert ACNF_TL in regs


def test_is_touched_reports_touch_correctly(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    # Set _touched with bits 0, 4 and 12 set
    mpr._touched = (1 << 0) | (1 << 4) | (1 << 12)
    assert mpr.is_touched(0) is True
    assert mpr.is_touched(4) is True
    assert mpr.is_touched(12) is True
    assert mpr.is_touched(1) is False
    assert mpr.is_touched(11) is False


def test_on_touch_registers_and_removes_handler(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus

    called = {}

    def handler1(touched, channel):
        called['handler1'] = (touched, channel)

    mpr.on_touch(3, handler1)
    # Trigger _handle_touch manually
    touched_bits = 1 << 3
    changed_bits = 1 << 3
    mpr._handle_touch(touched_bits, changed_bits)
    assert called.get('handler1') == (True, 3)

    # Remove handler by passing None - should replace with noop
    mpr.on_touch(3, None)
    # Replace handler with one not accepting two args (simulate old style)
    triggered = []

    def handler_oldstyle(touched):
        triggered.append(touched)

    mpr.on_touch(2, handler_oldstyle)
    # Touch channel 2: set bits 1<<2 in touched and changed
    mpr._handle_touch(1 << 2, 1 << 2)
    # handler_oldstyle should have been called with touched boolean
    assert triggered == [True]


def test_update_touch_state_calls_handler_correctly(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    called = []

    def handler(touched, channel):
        called.append((touched, channel))

    # Setup handler for channel 0
    mpr.on_touch(0, handler)
    # Setup _touched and patch touched() to return a changed bitmask
    mpr._touched = 0
    device.read_word_data[ETS] = 1  # channel 0 touched
    # Patch touched() method to call original based on dummy bus
    orig_touched = mpr.touched
    def touched_patch(*args, **kwargs):
        return 1
    mpr.touched = touched_patch
    mpr.update_touch_state()
    # Should call handler once with True,0
    assert called == [(True, 0)]
    # Calling again with same touch, no changes => no call
    called.clear()
    mpr.update_touch_state()
    assert called == []
    # Now patch touched to 0 (remove touch)
    mpr.touched = lambda *a, **k: 0
    mpr.update_touch_state()
    assert called == [(False, 0)]
    # Restore touched method for cleanup
    mpr.touched = orig_touched


def test_gpio_setup_writes_bits(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus

    # Setup for channel=5 (valid 4-11), output=True, mode=3, enable=True
    # We will check the calls were made for GPIO_CTL0, GPIO_CTL1, GPIO_DIR, GPIO_EN

    # Setup device.read_byte_data so bits can be manipulated
    device.read_byte_data[GPIO_CTL0] = 0b00000000
    device.read_byte_data[GPIO_CTL1] = 0b00000000
    device.read_byte_data[GPIO_DIR] = 0b00000000
    device.read_byte_data[GPIO_EN] = 0b00000000

    mpr.gpio_setup(channel=5, output=1, mode=3, enable=1)

    # Because mode=3, bits for GPIO_CTL0 and GPIO_CTL1 are set accordingly
    writes = device.writes
    # Expect writes to GPIO_CTL0, GPIO_CTL1, GPIO_DIR, GPIO_EN each once
    written_regs = [w[1] for w in writes if w[0] == 'write_byte']
    assert GPIO_CTL0 in written_regs
    assert GPIO_CTL1 in written_regs
    assert GPIO_DIR in written_regs
    assert GPIO_EN in written_regs

    # Also test that asserting mode=1 maps to mode=0
    device.read_byte_data[GPIO_CTL0] = 0
    device.read_byte_data[GPIO_CTL1] = 0
    device.read_byte_data[GPIO_DIR] = 0
    device.read_byte_data[GPIO_EN] = 0
    device.writes.clear()

    mpr.gpio_setup(channel=6, output=0, mode=1, enable=0)
    # mode=1 translates to mode=0, so bits should be zero
    writes = device.writes
    ctl0_write = next(w for w in writes if w[1] == GPIO_CTL0)
    ctl1_write = next(w for w in writes if w[1] == GPIO_CTL1)
    assert ctl0_write[2] & 0b10 == 0
    assert ctl1_write[2] & 0b01 == 0


def test_gpio_status_returns_read_byte(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    device.read_byte_data[GPIO_DAT] = 0xAB
    status = mpr.gpio_status()
    assert status == 0xAB


def test_gpio_set_writes_correct_register(mpr121_with_dummybus):
    mpr, device = mpr121_with_dummybus
    device.writes.clear()
    # channel=4, value True => write to GPIO_SET, bit 0 (channel-4=0)
    mpr.gpio_set(4, True)
    assert ('write_byte', GPIO_SET, 1) in device.writes
    device.writes.clear()
    # channel=11, value False => write to GPIO_CLR, bit 7 (11-4)
    mpr.gpio_set(11, False)
    expected = 1 << (11 - 4)
    assert ('write_byte', GPIO_CLR, expected) in device.writes


def test_init_with_irq_and_handlers_creates_irqhandler(monkeypatch):
    # Patch IRQHandler to track calls
    calls = {}
    def fake_IRQHandler(irq, callback):
        calls['irq'] = irq
        calls['callback'] = callback
    monkeypatch.setattr('piripherals.mpr121.IRQHandler', fake_IRQHandler)
    monkeypatch.setattr('piripherals.mpr121.Poller', lambda callback: None)
    # Patch Bus to DummyBus for testing
    monkeypatch.setattr('piripherals.mpr121.Bus', lambda bus=None: DummyBus())
    mpr = MPR121(bus=1, irq=42, handlers=1, setup=0, reset=0)
    assert calls.get('irq') == 42
    assert callable(calls.get('callback'))


def test_init_with_no_irq_uses_poller(monkeypatch):
    calls = {}
    def fake_Poller(callback):
        calls['poller'] = True
        calls['callback'] = callback
    monkeypatch.setattr('piripherals.mpr121.Poller', fake_Poller)
    monkeypatch.setattr('piripherals.mpr121.IRQHandler', lambda irq, cb: None)
    monkeypatch.setattr('piripherals.mpr121.Bus', lambda bus=None: DummyBus())
    mpr = MPR121(bus=1, irq=0, handlers=1, setup=0, reset=0)
    assert calls.get('poller') is True
    assert callable(calls.get('callback'))
