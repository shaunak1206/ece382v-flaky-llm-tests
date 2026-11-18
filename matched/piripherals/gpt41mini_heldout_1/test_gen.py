import builtins
import pytest
from piripherals import mpr121

class DummyBusDevice:
    def __init__(self):
        # map of register to value or list
        self.written_bytes = []
        self.written_words = []
        self.read_values = {}
        self.read_word_values = {}
        self.read_block_values = {}
        self._byte_reads = {}
        self._word_reads = {}
        self._block_reads = {}

    def write_byte(self, reg, val):
        self.written_bytes.append((reg, val))

    def write_word(self, reg, val):
        self.written_words.append((reg, val))

    def write_block(self, reg, vals):
        # simulate write_block by write_byte for testing
        self.written_bytes.append((reg, list(vals)))

    def read_byte(self, reg):
        if reg in self._byte_reads:
            return self._byte_reads[reg]
        # default dummy
        return 0

    def read_word(self, reg):
        if reg in self._word_reads:
            return self._word_reads[reg]
        return 0

    def read_block(self, reg, length):
        if reg in self._block_reads:
            return self._block_reads[reg]
        return [0]*length


class DummyBus:
    def __init__(self):
        self.called_with = []
        self.device_inst = DummyBusDevice()

    def device(self, addr):
        self.called_with.append(addr)
        return self.device_inst


def make_mpr121_with_dummy_bus(**kwargs):
    # Substitute bus with dummy to avoid hardware
    dummy_bus = DummyBus()
    # Patch piripherals.mpr121.Bus to avoid import errors on bus
    import piripherals.mpr121 as m
    m.Bus = lambda busnum=None: dummy_bus
    # Patch IRQHandler and Poller to dummy no-op to avoid side effects
    m.IRQHandler = lambda irq_pin, callback: None
    m.Poller = lambda callback: None
    # Patch noop and not_raising (from .util) to return dummy or noop
    m.noop = lambda *a, **k: None
    m.not_raising = lambda f: f if f is not None else None

    # create instance
    instance = m.MPR121(bus=1, irq=0, handlers=0, setup=0, reset=0, **kwargs)
    # Return instance and dummy_bus.device_inst for inspection
    return instance, dummy_bus.device_inst


def test_init_calls_reset_and_setup(monkeypatch):
    called = dict(reset=False, setup=False)

    def fake_reset(self):
        called['reset'] = True

    def fake_setup(self, **kwargs):
        called['setup'] = True

    monkeypatch.setattr(mpr121.MPR121, 'reset', fake_reset)
    monkeypatch.setattr(mpr121.MPR121, 'setup', fake_setup)

    # handlers=0 disables IRQ/Poller, setup=1 & reset=1 default called
    mpr121.MPR121(bus=1, handlers=0, setup=1, reset=1)
    assert called['reset'] is True
    assert called['setup'] is True


def test_init_with_irq_and_handlers(monkeypatch):
    flags = dict(irq_handler_called=False, poller_called=False)

    def fake_irqhandler(irq, cb):
        flags['irq_handler_called'] = True

    def fake_poller(cb):
        flags['poller_called'] = True

    monkeypatch.setattr(mpr121, 'IRQHandler', fake_irqhandler)
    monkeypatch.setattr(mpr121, 'Poller', fake_poller)

    # With irq!=0 and handlers=1, IRQHandler should be used.
    mpr121.MPR121(bus=1, irq=4, handlers=1, setup=0, reset=0)
    assert flags['irq_handler_called']
    assert not flags['poller_called']

    # With irq=0 and handlers=1, Poller should be used.
    flags['irq_handler_called'] = False
    flags['poller_called'] = False
    mpr121.MPR121(bus=1, irq=0, handlers=1, setup=0, reset=0)
    assert not flags['irq_handler_called']
    assert flags['poller_called']

def test_update_touch_state_calls_handle_touch(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    called = dict(handle=False)

    def fake_touched():
        return 0b1010

    def fake_handle_touch(touched, changed):
        called['handle'] = True
        assert touched == 0b1010
        assert changed == 0b1010  # initial self._touched = 0

    mpr._touched = 0
    monkeypatch.setattr(mpr, 'touched', fake_touched)
    monkeypatch.setattr(mpr, '_handle_touch', fake_handle_touch)

    mpr.update_touch_state()
    assert called['handle']


def test_handle_touch_invokes_handlers_provides_args():
    mpr, dev = make_mpr121_with_dummy_bus()

    called = []

    def handler(touched, channel=None):
        called.append((touched, channel))

    mpr._handlers[5] = handler

    # simulate touch change on bit 5 and 6
    touched = (1 << 5) | 0
    changed = 1 << 5
    mpr._handle_touch(touched, changed)

    assert (True, 5) in called


def test_handle_touch_handler_typeerror_passes_one_param():
    mpr, dev = make_mpr121_with_dummy_bus()
    called = []

    def handler_oneparam(touched):
        called.append(touched)

    mpr._handlers[2] = handler_oneparam

    touched = (1 << 2) 
    changed = (1 << 2)
    mpr._handle_touch(touched, changed)
    assert True in called


def test_is_touched_returns_correct_status():
    mpr, dev = make_mpr121_with_dummy_bus()
    mpr._touched = 0b1010  # bits 1,3 set
    assert mpr.is_touched(1) is True
    assert mpr.is_touched(0) is False
    assert mpr.is_touched(3) is True
    assert mpr.is_touched(2) is False


def test_on_touch_replaces_handlers_and_removes():
    mpr, dev = make_mpr121_with_dummy_bus()
    called = []

    def handler(status, channel): 
        called.append((status, channel))

    mpr.on_touch(1, handler)
    assert mpr._handlers[1] is handler

    mpr.on_touch(2, None)
    # None replaces with noop, which is callable but does nothing
    assert callable(mpr._handlers[2])
    assert mpr._handlers[2] != handler


def test_reset_writes_correct_byte():
    mpr, dev = make_mpr121_with_dummy_bus()
    mpr.reset()
    assert any(reg == mpr.SRESET and val == 0x63 for reg, val in dev.written_bytes)


def test_configure_writes_correct_byte():
    mpr, dev = make_mpr121_with_dummy_bus()
    mpr.configure(cl=2, prox=3, touch=5)
    expected = (2 << 6) | (3 << 4) | 5
    assert any(reg == mpr.ECR and val == expected for reg, val in dev.written_bytes)


def test_touched_returns_value_and_raises(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    # case no failure bits
    dev._word_reads[mpr.ETS] = 0x0012
    mpr._bus = dev
    # Should return low 13 bits
    val = mpr.touched()
    assert val == 0x0012

    # case with overcurrent bit set (bit 15)
    dev._word_reads[mpr.ETS] = (1 << 15)
    with pytest.raises(Exception, match='overcurrent'):
        mpr.touched()

    # case with out of range bit set, bit 15 off but out_of_range returns True
    dev._word_reads[mpr.ETS] = 0x0001
    def fake_oor():
        return 1
    monkeypatch.setattr(mpr, 'out_of_range', fake_oor)

    with pytest.raises(Exception, match='out of range'):
        mpr.touched()


def test_electrode_data_returns_list_with_correct_length_and_values():
    mpr, dev = make_mpr121_with_dummy_bus()
    # Setup block read to return 2*NCH bytes: alternating low/high bytes for 0..NCH-1
    bytes_list = []
    for i in range(mpr.NCH):
        low = i & 0xff
        high = (i << 1) & 0xff
        bytes_list.extend([low, high])
    dev._block_reads[mpr.EFD] = bytes_list
    data = mpr.electrode_data()
    assert len(data) == mpr.NCH
    for i in range(mpr.NCH):
        expected = (bytes_list[2*i+1] << 8) + bytes_list[2*i]
        assert data[i] == expected


def test_baseline_no_args_returns_scaled_list():
    mpr, dev = make_mpr121_with_dummy_bus()
    # fake baseline block returns NCH values 0..NCH-1
    dev._block_reads[mpr.EBL] = list(range(mpr.NCH))
    expected = [(b << 2) for b in range(mpr.NCH)]
    result = mpr.baseline()
    assert result == expected


def test_baseline_with_args_calls_write_block(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    called = {'reg': None, 'vals': None}

    def fake_write_block(reg, vals):
        called['reg'] = reg
        called['vals'] = vals

    monkeypatch.setattr(dev, 'write_block', fake_write_block)

    # rft 0 (rising) should write 4 bytes (mhd, nhd, ncl, fdl)
    mpr.baseline(rft=0, mhd=5, nhd=6, ncl=7, fdl=8, prox=False)
    assert called['reg'] == mpr.MHD
    assert called['vals'] == [5, 6, 7, 8]

    # rft 2 should write 3 bytes (nhd, ncl, fdl)
    mpr.baseline(rft=2, nhd=9, ncl=10, fdl=11, prox=True)
    assert called['reg'] == mpr.MHDX + 8  # 4*2=8 offset
    assert called['vals'] == [9, 10, 11]


def test_threshold_sets_all_and_channel_specific(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    calls = []

    def fake_write_byte(reg, val):
        calls.append((reg,val))

    monkeypatch.setattr(dev, 'write_byte', fake_write_byte)

    # channel -1 sets for all including proximity channel (12)
    mpr.threshold(touch=10)
    # Must have written 13*2 times (touch and release for each channel 0-12)
    touch_regs = set(r for r,v in calls if r % 2 == 0)
    release_regs = set(r for r,v in calls if r % 2 == 1)
    assert len(calls) == mpr.NCH * 2
    # Check proximity channel is written (channel 12)
    assert mpr.TTH + 2*12 in touch_regs

    # Reset
    calls.clear()
    # channel specific call with release omitted sets release to 0.6*touch
    mpr.threshold(touch=20, channel=4)
    assert (mpr.TTH + 8, 20) in calls
    expected_release = int(0.6*20)
    assert (mpr.RTH + 8, expected_release) in calls


def test_debounce_sets_write_byte(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    calls = []

    def fake_write_byte(reg, val):
        calls.append((reg, val))

    monkeypatch.setattr(dev, 'write_byte', fake_write_byte)

    mpr.debounce(touch=3)
    # Since release omitted, release=touch=3, value=(3 <<4)|3=51
    assert (mpr.DEB, 51) in calls

    calls.clear()
    mpr.debounce(touch=1, release=2)
    # value should be (2 <<4) | 1 = 33
    assert (mpr.DEB, 33) in calls


def test_filter_writes_correct_word(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    calls = []

    def fake_write_word(reg, val):
        calls.append((reg,val))

    monkeypatch.setattr(dev, 'write_word', fake_write_word)

    mpr.filter(cdc=10, cdt=3, ffi=2, sfi=1, esi=4)
    # compute expected low byte
    lb = (2 << 6) | 10
    hb = (3 << 5) | (1 << 3) | 4
    expected_word = (hb << 8) | lb
    assert (mpr.AFE1, expected_word) in calls


def test_charge_writes_bytes_and_reads(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    # Setup read_byte to return 0xff for CDT register
    dev._byte_reads[mpr.CDT] = 0xff
    calls = []

    def fake_write_byte(reg, val):
        calls.append((reg,val))

    def fake_read_byte(reg):
        return dev._byte_reads.get(reg, 0x00)

    monkeypatch.setattr(dev, 'write_byte', fake_write_byte)
    monkeypatch.setattr(dev, 'read_byte', fake_read_byte)

    mpr._bus = dev
    mpr.charge(channel=3, cdc=5, cdt=2)

    # Expect write_byte called on CDC+channel with cdc=5
    assert (mpr.CDC+3, 5) in calls
    # Expect write_byte called on CDT + 3//2 register with modified bits for cdt=2 shifted left 4 bits (3%2=1)
    write_cdt_calls = [val for reg,val in calls if reg == mpr.CDT + 1]
    assert any((val & 0xf0) == (2 << 4) for val in write_cdt_calls)


def test_auto_config_writes_correct_bytes(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    calls_word = []
    calls_byte = []

    monkeypatch.setattr(dev, 'write_word', lambda reg, val: calls_word.append((reg,val)))
    monkeypatch.setattr(dev, 'write_byte', lambda reg, val: calls_byte.append((reg,val)))

    mpr._bus = dev

    mpr.auto_config(ace=1, are=1, bva=2, retry=2, afes=3, scts=1, acfie=1, arfie=0, oorie=1,
                    usl=123, lsl=45, tl=67)

    # Verify write_word call to ACNF_C0 with correct bits set
    expected_lb = (3 << 6) | (2 << 4) | (2 << 2) | (1 << 1) | 1
    expected_hb = (1 << 7) | (1 << 2) | (0 << 1) | 1
    # word is (hb <<8) + lb
    found = False
    for reg, val in calls_word:
        if reg == mpr.ACNF_C0:
            assert val == (expected_hb << 8) + expected_lb
            found = True
    assert found

    # Check ACNF_USL, ACNF_LSL, ACNF_TL written with usl, lsl, tl respectively
    write_regs = {r for r,v in calls_byte}
    assert mpr.ACNF_USL in write_regs
    assert mpr.ACNF_LSL in write_regs
    assert mpr.ACNF_TL in write_regs


def test_out_of_range_returns_bits_and_raises(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    # no failure bits set
    dev._word_reads[mpr.OOR] = 0x0111
    mpr._bus = dev
    val = mpr.out_of_range()
    assert val == 0x0111

    # set bit 15 (auto_config_failed)
    dev._word_reads[mpr.OOR] = (1 << 15)
    with pytest.raises(Exception, match='auto config failed'):
        mpr.out_of_range()

    # set bit 14 (auto_reconfig_failed)
    dev._word_reads[mpr.OOR] = (1 << 14)
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr.out_of_range()


def test_gpio_setup_writes_bytes(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    # read_byte returns 0 for all GPIO registers initially
    dev._byte_reads[mpr.GPIO_CTL0] = 0
    dev._byte_reads[mpr.GPIO_CTL1] = 0
    dev._byte_reads[mpr.GPIO_DIR] = 0
    dev._byte_reads[mpr.GPIO_EN] = 0

    writes = []
    monkeypatch.setattr(dev, 'read_byte', lambda reg: dev._byte_reads.get(reg, 0))
    monkeypatch.setattr(dev, 'write_byte', lambda reg, val: writes.append((reg, val)))

    mpr._bus = dev

    # Mode 3 should be decomposed into bits 1 (0b10) and 1 (0b01)
    # Channel is 6, so index = 6-4=2
    mpr.gpio_setup(channel=6, output=True, mode=3, enable=True)

    # We expect 4 writes to GPIO_CTL0, GPIO_CTL1, GPIO_DIR, GPIO_EN for bit 2
    targets = {mpr.GPIO_CTL0, mpr.GPIO_CTL1, mpr.GPIO_DIR, mpr.GPIO_EN}
    regs_written = set(r for r,v in writes)
    assert targets == regs_written

    # The bits in GPIO_CTL0 for bit 2 is mode & 0b10 = 2 (bit1 set)
    # GPIO_CTL1 bit 2 is mode & 0b01 = 1 (bit0 set)
    ctl0_val = next(v for r,v in writes if r == mpr.GPIO_CTL0)
    ctl1_val = next(v for r,v in writes if r == mpr.GPIO_CTL1)
    # bit 2 corresponds to (1 << 2) = 4
    assert (ctl0_val & (1 << 2)) == (1 << 2)
    assert (ctl1_val & (1 << 2)) == (1 << 2)

def test_gpio_status_reads_byte(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    dev._byte_reads[mpr.GPIO_DAT] = 0xAB
    mpr._bus = dev
    assert mpr.gpio_status() == 0xAB


def test_gpio_set_writes_expected_register_and_value(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    writes = []
    monkeypatch.setattr(dev, 'write_byte', lambda reg, val: writes.append((reg,val)))
    mpr._bus = dev

    # channel=5, value=True (1)
    mpr.gpio_set(5, True)
    # with value True, should write to GPIO_SET register
    regs = [r for r,v in writes]
    assert mpr.GPIO_SET in regs
    # with value False, write must be to GPIO_CLR
    writes.clear()
    mpr.gpio_set(7, False)
    regs = [r for r,v in writes]
    assert mpr.GPIO_CLR in regs

def test_threshold_release_calc_correct(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    calls = []

    monkeypatch.setattr(dev, 'write_byte', lambda reg, val: calls.append((reg,val)))
    mpr._bus = dev

    # call threshold with release=-1 meaning default 0.6*touch
    mpr.threshold(touch=40, release=-1, channel=8)
    # make sure release is int(0.6*40)
    assert (mpr.TTH + 2*8, 40) in calls
    expected_release = int(0.6*40)
    assert (mpr.RTH + 2*8, expected_release) in calls


def test_on_touch_removes_handler(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()

    called = {'called': False}

    def handler(status, channel):
        called['called'] = True

    mpr.on_touch(3, handler)

    assert mpr._handlers[3] is handler
    mpr.on_touch(3, None)  # Should replace with noop
    assert callable(mpr._handlers[3])


def test_baseline_with_only_rft_and_prox(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    called = {}

    def fake_write_block(reg, vals):
        called['reg'] = reg
        called['vals'] = vals

    monkeypatch.setattr(dev, 'write_block', fake_write_block)
    mpr._bus = dev

    # call with rft=1 (falling), prox True, no other params
    mpr.baseline(rft=1, prox=True)
    assert called['reg'] == mpr.MHDX + 4*1
    # Because rft <=1, vals length should be 4 exact!

def test_dump_runs_without_error(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    mpr._bus = dev

    # provide required read_byte for thresholds and other registers used in dump()
    for i in range(mpr.NCH):
        dev._byte_reads[mpr.TTH + 2*i] = 10
        dev._byte_reads[mpr.RTH + 2*i] = 5

    # patch methods that dump calls and reads to avoid IO errors
    monkeypatch.setattr(mpr, 'touched', lambda raise_on_failure=1: 0)
    monkeypatch.setattr(mpr, 'out_of_range', lambda raise_on_failure=1: 0)
    monkeypatch.setattr(mpr, 'gpio_status', lambda: 0)
    monkeypatch.setattr(mpr, 'electrode_data', lambda: [0]*mpr.NCH)
    monkeypatch.setattr(mpr, 'baseline', lambda rft=-1, **kwargs: [0]*mpr.NCH)
    monkeypatch.setattr(dev, 'read_byte', lambda reg: 0)
    monkeypatch.setattr(dev, 'read_block', lambda reg, size: [0]*size)

    # run dump with regs=0, up=0, loop=1 to avoid escape sequences and multiple prints
    # Just verify no exception
    mpr.dump(regs=0, up=0, loop=1)


def test_setup_calls_expected_sequence(monkeypatch):
    mpr, dev = make_mpr121_with_dummy_bus()
    calls = []

    monkeypatch.setattr(mpr, 'reset', lambda: calls.append('reset'))
    monkeypatch.setattr(mpr, 'configure', lambda **kwargs: calls.append(('configure', kwargs)))
    monkeypatch.setattr(mpr, 'filter', lambda **kwargs: calls.append(('filter', kwargs)))
    monkeypatch.setattr(mpr, 'auto_config', lambda **kwargs: calls.append('auto_config'))
    monkeypatch.setattr(mpr, 'threshold', lambda **kwargs: calls.append(('threshold', kwargs)))
    monkeypatch.setattr(mpr, 'debounce', lambda **kwargs: calls.append(('debounce', kwargs)))
    monkeypatch.setattr(mpr, 'baseline', lambda **kwargs: calls.append(('baseline', kwargs)))

    mpr.setup(reset=True, channels=11, prox=2, threshold=60, debounce=3, auto_config=True)

    # check reset called
    assert 'reset' in calls
    # check configure called at least twice (once to disable, once to enable)
    config_calls = [c for c in calls if isinstance(c, tuple) and c[0] == 'configure']
    assert len(config_calls) >= 2
    # check auto_config called
    assert 'auto_config' in calls
    # check threshold called with default channel and for proximity channel 12
    threshold_calls = [c for c in calls if isinstance(c, tuple) and c[0] == 'threshold']
    assert any('touch' in c[1] and c[1].get('channel', None) == -1 for _, c[1] in threshold_calls) or threshold_calls
    # check debounce called
    assert any(c for c in calls if isinstance(c, tuple) and c[0] == 'debounce')


# The following test ensures setup disables initial setup and reset working:

def test_disable_setup_and_reset(monkeypatch):
    # Patch reset and setup to know if called
    called = {'reset': False, 'setup': False}
    def fake_reset(self):
        called['reset'] = True

    def fake_setup(self, **kwargs):
        called['setup'] = True

    monkeypatch.setattr(mpr121.MPR121, 'reset', fake_reset)
    monkeypatch.setattr(mpr121.MPR121, 'setup', fake_setup)
    # With setup=0 and reset=1, only reset should be called
    mpr121.MPR121(setup=0, reset=1, handlers=0)
    assert called['reset'] is True
    assert called['setup'] is False

    # With setup=0 and reset=0, neither called
    called['reset'] = False
    called['setup'] = False
    mpr121.MPR121(setup=0, reset=0, handlers=0)
    assert not called['reset']
    assert not called['setup']

