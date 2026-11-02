import pytest
from piripherals import mpr121

import random
import numpy as np
import sys

# A dummy bus device to simulate hardware bus reads and writes
class DummyBusDevice:
    def __init__(self):
        self.memory = {}
        self._read_word_return = 0x0000
        self._read_block_return = [0] * (2 * mpr121.NCH)
        self._read_byte_return = 0x00
        self._written_bytes = []
        self._written_words = []

    def write_byte(self, reg, value):
        self.memory[reg] = value
        self._written_bytes.append((reg, value))

    def write_word(self, reg, value):
        self.memory[reg] = value
        self._written_words.append((reg, value))

    def write_block(self, reg, values):
        for i, v in enumerate(values):
            self.memory[reg + i] = v
        self._written_bytes.append((reg, values))

    def read_word(self, reg):
        return self._read_word_return

    def read_block(self, reg, length):
        # return slice of the stored list or a repeated pattern:
        requested = self._read_block_return[:length]
        if len(requested) < length:
            requested = requested + [0] * (length - len(requested))
        return requested

    def read_byte(self, reg):
        return self._read_byte_return


class DummyBus:
    def __init__(self):
        self.device_calls = []

    def device(self, addr):
        # create a new DummyBusDevice for each addr (or same one if you want)
        self.device_calls.append(addr)
        return DummyBusDevice()


@pytest.fixture
def dummy_bus(monkeypatch):
    # Patch the piripherals.bus.Bus to a dummy factory returning dummy bus device
    class DummyBusFactory:
        def __init__(self):
            self._device = DummyBusDevice()

        def device(self, addr):
            return self._device

    factory = DummyBusFactory()
    monkeypatch.setattr(mpr121, "Bus", lambda bus_num=None: factory)
    return factory._device


def seed_all():
    random.seed(1234)
    np.random.seed(1234)
    try:
        import torch
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
    except ImportError:
        pass


def test_init_creates_bus_device(dummy_bus, monkeypatch):
    seed_all()
    # It should create a bus device from bus argument int
    m = mpr121.MPR121(bus=1, addr=0x5a, irq=0, handlers=0, setup=0, reset=0)
    assert m._bus is dummy_bus
    # default handlers list should be noop functions for each channel
    assert len(m._handlers) == mpr121.NCH
    for h in m._handlers:
        assert callable(h)


def test_reset_writes_reset_command(dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)
    dummy_bus._written_bytes.clear()
    m.reset()
    assert dummy_bus._written_bytes
    reg, val = dummy_bus._written_bytes[-1]
    assert reg == mpr121.SRESET
    assert val == 0x63


def test_configure_sets_register_correctly(dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)
    dummy_bus._written_bytes.clear()
    # cl=3, prox=2 (enable electrodes 0-3), touch=5 (electrodes 0-4 enabled)
    m.configure(cl=3, prox=2, touch=5)
    reg, val = dummy_bus._written_bytes[-1]
    assert reg == mpr121.ECR
    expected = (3 << 6) | (2 << 4) | 5
    assert val == expected


def test_touched_returns_mask_and_raises_on_overcurrent(monkeypatch):
    seed_all()
    dummy_bus_device = DummyBusDevice()

    # Simulate overcurrent bit (bit 15) set
    dummy_bus_device._read_word_return = 0x8000 | 0x000F

    # patch Bus to return dummy bus device
    monkeypatch.setattr(mpr121, "Bus", lambda bus: type("Bus", (), {"device": lambda s, addr=None: dummy_bus_device})())

    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)

    # By default, raises on failure bit set
    with pytest.raises(Exception) as e:
        m.touched()
    assert str(e.value) == 'overcurrent'

    # When raise_on_failure=False, returns touch bits even if overcurrent
    val = m.touched(raise_on_failure=False)
    assert (val & 0x1fff) == (0x8000 | 0x000F) & 0x1fff


def test_touched_returns_mask_and_raises_on_out_of_range(monkeypatch):
    seed_all()
    dummy_bus_device = DummyBusDevice()

    # No overcurrent, but out of range bit set in out_of_range()
    dummy_bus_device._read_word_return = 0x0001  # touched bits low=1

    # patch _bus read_word to return 0 for touched(), patch out_of_range to raise Exception
    def dummy_read_word(reg):
        if reg == mpr121.ETS:
            return 0x0001  # some touched bit set
        elif reg == mpr121.OOR:
            # poke in out_of_range bits with auto config failed (bit 15)
            return 0x8000
        return 0

    dummy_bus_device.read_word = dummy_read_word

    monkeypatch.setattr(mpr121, "Bus", lambda bus: type("Bus", (), {"device": lambda s, addr=None: dummy_bus_device})())

    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)

    # Patch out_of_range to raise Exception with 'out of range'
    def dummy_out_of_range(raise_on_failure=1):
        if raise_on_failure:
            raise Exception('out of range')
        return 0

    monkeypatch.setattr(m, "out_of_range", dummy_out_of_range)

    with pytest.raises(Exception) as e:
        m.touched()

    assert str(e.value) == 'out of range'


def test_electrode_data_reads_correctly(dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)
    data_raw = []
    for i in range(mpr121.NCH):
        # Make 2 bytes per channel, low byte = i, high byte = NCH - i
        data_raw.append(i)          # low byte
        data_raw.append(mpr121.NCH - i)   # high byte (max 13)
    dummy_bus._read_block_return = data_raw

    result = m.electrode_data()
    assert isinstance(result, list)
    assert len(result) == mpr121.NCH
    for i, v in enumerate(result):
        expected = (mpr121.NCH - i) << 8 | i
        assert v == expected


def test_baseline_get_and_set(monkeypatch):
    seed_all()
    dummy_bus_device = DummyBusDevice()
    # For get baseline, _bus.read_block returns 13 bytes with values 10
    dummy_bus_device._read_block_return = [10] * mpr121.NCH

    monkeypatch.setattr(mpr121, "Bus", lambda bus: type("Bus", (), {"device": lambda s, addr=None: dummy_bus_device})())

    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)

    # if no args, returns list of baseline values shifted left by 2
    baseline_vals = m.baseline()
    assert isinstance(baseline_vals, list)
    assert len(baseline_vals) == mpr121.NCH
    # Each should be 10 << 2 = 40
    assert all(v == 40 for v in baseline_vals)

    # Now test set calls write_block with correct length for rft=0 (rising)
    dummy_bus_device._written_bytes.clear()
    m.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    # It should write 4 bytes at reg MHD + 4*rft = MHD
    assert any(isinstance(w[1], list) and len(w[1]) == 4 for w in dummy_bus_device._written_bytes)

    dummy_bus_device._written_bytes.clear()
    m.baseline(rft=2, nhd=1, ncl=3, fdl=20, prox=True)
    reg_expected = mpr121.MHDX + 4 * 2
    # It should write 3 bytes for rft=2
    assert any(reg == reg_expected and isinstance(vals, list) and len(vals) == 3
               for reg, vals in dummy_bus_device._written_bytes if isinstance(vals, list))


def test_threshold_sets_bytes_for_all_channels(dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)
    dummy_bus._written_bytes.clear()

    touch_val = 100
    release_val = 40
    # channel < 0 applies thresholds to all channels
    m.threshold(touch=touch_val, release=release_val, channel=-1)

    # There should be NCH * 2 writes (touch and release for each)
    # but threshold writes one byte per call, but call is recursive.
    # So check at least for all channels a touch and release write happened.

    touch_regs = [mpr121.TTH + 2*i for i in range(mpr121.NCH)]
    release_regs = [mpr121.RTH + 2*i for i in range(mpr121.NCH)]
    regs_written = [r for r, v in dummy_bus._written_bytes]

    for tr in touch_regs:
        assert tr in regs_written
    for rr in release_regs:
        assert rr in regs_written

    # Now test call to threshold for a single channel sets correct bytes
    dummy_bus._written_bytes.clear()
    m.threshold(touch=200, release=120, channel=5)
    # The last two writes should be at TTH+10 and RTH+10 registers
    last_two = dummy_bus._written_bytes[-2:]
    assert last_two[0][0] == mpr121.TTH + 2*5
    assert last_two[0][1] == 200
    assert last_two[1][0] == mpr121.RTH + 2*5
    assert last_two[1][1] == 120


def test_debounce_sets_byte(dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)
    dummy_bus._written_bytes.clear()

    m.debounce(touch=3, release=5)
    # Check register and value
    reg, val = dummy_bus._written_bytes[-1]
    assert reg == mpr121.DEB
    # value is ((release << 4) | touch)
    assert val == (5 << 4) | 3

    # Test release default to touch when omitted
    dummy_bus._written_bytes.clear()
    m.debounce(touch=4)
    reg, val = dummy_bus._written_bytes[-1]
    assert val == (4 << 4) | 4


def test_filter_sets_word_correctly(dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)
    dummy_bus._written_words.clear()

    m.filter(cdc=10, cdt=3, ffi=2, sfi=1, esi=5)
    # write_word called with AFE1 reg and a 16-bit word value
    reg, val = dummy_bus._written_words[-1]
    assert reg == mpr121.AFE1
    # val low byte = (ffi << 6) | cdc
    expected_low = (2 << 6) | 10
    # val high byte = (cdt << 5) | (sfi << 3) | esi
    expected_high = (3 << 5) | (1 << 3) | 5
    expected_val = (expected_high << 8) | expected_low
    assert val == expected_val


def test_charge_writes_bytes_and_reads_correctly(dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)
    dummy_bus.memory[mpr121.CDT] = 0b00000000
    dummy_bus.memory[mpr121.CDT + 1] = 0b11110000
    dummy_bus._written_bytes.clear()

    # Set charge current and time for channel 3
    m.charge(channel=3, cdc=12, cdt=2)

    # Two writes: one for CDC + channel and one for CDT reg
    writes = dummy_bus._written_bytes
    regs_written = [w[0] for w in writes]
    assert mpr121.CDC + 3 in regs_written
    # Check read and write on CDT register (channel 3 => reg CDT+1, shift 4)
    assert mpr121.CDT + 1 in regs_written

    # Verify that written value keeps other bits unchanged except 4 bits shifted
    val_written = None
    for r, v in writes:
        if r == mpr121.CDT + 1:
            val_written = v
    assert val_written is not None
    # Lower nibble of val_written should be preserved as original
    orig_val = dummy_bus.read_byte(mpr121.CDT + 1)
    s = 4 * (3 % 2)
    # Bits not modified according to bitmask ~(0xf << 4)
    expected_val_masked = orig_val & ~(0xf << s)
    # Value bits set by cdt 2 << s
    expected_val = expected_val_masked | (2 << s)
    assert val_written == expected_val


def test_auto_config_writes_correctly(dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)
    dummy_bus._written_bytes.clear()
    dummy_bus._written_words.clear()

    m.auto_config(ace=1, are=0, bva=2, retry=1, afes=3, scts=1, acfie=1, arfie=0, oorie=1, usl=150, lsl=120, tl=100)

    # check last word write to ACNF_C0
    word_writes = [w for w in dummy_bus._written_words if w[0] == mpr121.ACNF_C0]
    assert word_writes

    reg, val = word_writes[-1]
    # unpack lb and hb
    lb = val & 0xff
    hb = (val >> 8) & 0xff

    # lb low nibble bits
    # ace=1 at bit0, are=0 at bit1, bva=2 at bits 2-3, retry=1 at bits 4-5, afes=3 at bits 6-7
    expected_lb = (3 << 6) | (1 << 4) | (2 << 2) | (0 << 1) | 1
    assert lb == expected_lb

    # hb bits: scts=1 at bit7, oorie=1 at bit2, arfie=0 bit1, acfie=1 bit0
    expected_hb = (1 << 7) | (1 << 2) | (0 << 1) | 1
    assert hb == expected_hb

    # check that 3 bytes written to the three USL, LSL, TL registers with given values
    written = dict(dummy_bus._written_bytes)
    assert written[mpr121.ACNF_USL] == 150
    assert written[mpr121.ACNF_LSL] == 120
    assert written[mpr121.ACNF_TL] == 100


def test_out_of_range_raises_correctly(monkeypatch):
    seed_all()
    dummy_bus_device = DummyBusDevice()

    # simulate word where bit 15 (auto config failed) is set:
    dummy_bus_device._read_word_return = 0x8000
    monkeypatch.setattr(mpr121, "Bus", lambda bus: type("Bus", (), {"device": lambda s, addr=None: dummy_bus_device})())

    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)
    # should raise 'auto config failed'
    with pytest.raises(Exception) as e:
        m.out_of_range()

    assert str(e.value) == 'auto config failed'

    # simulate bit 14 (auto reconfig failed)
    dummy_bus_device._read_word_return = 0x4000
    with pytest.raises(Exception) as e:
        m.out_of_range()

    assert str(e.value) == 'auto reconfig failed'

    # When raise_on_failure=False, returns value masked with 0x1fff
    dummy_bus_device._read_word_return = 0xFFFF
    out = m.out_of_range(raise_on_failure=False)
    assert out == 0x1fff


def test_gpio_setup_calls_write_byte_correctly(monkeypatch, dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)

    # Patch _bus.read_byte to return 0 always to avoid print garbage
    dummy_bus.read_byte = lambda reg: 0x00
    dummy_bus._written_bytes.clear()

    # Test with output=True, mode=0, enable=True, channel=5 (valid)
    m.gpio_setup(channel=5, output=1, mode=0, enable=1)

    # Should write 4 times, for GPIO_CTL0, GPIO_CTL1, GPIO_DIR, GPIO_EN
    regs_written = [w[0] for w in dummy_bus._written_bytes]
    # Expected channels is channel-4 = 1
    bit_pos = 1
    # For GPIO_CTL0, bit 1 must be mode & 0b10 (mode=0 here, so 0)
    # For GPIO_CTL1, bit 1 must be mode & 0b01 (mode=0)
    # GPIO_DIR bit 1 must be output (1)
    # GPIO_EN bit 1 must be enable (1)
    # Expect these registers written
    assert mpr121.GPIO_CTL0 in regs_written
    assert mpr121.GPIO_CTL1 in regs_written
    assert mpr121.GPIO_DIR in regs_written
    assert mpr121.GPIO_EN in regs_written


def test_gpio_status_reads_byte(monkeypatch, dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)

    dummy_bus._read_byte_return = 0xAB
    val = m.gpio_status()
    assert val == 0xAB


def test_gpio_set_writes_correct_byte(dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)
    dummy_bus._written_bytes.clear()

    # Set channel 6 high (value True)
    m.gpio_set(6, True)
    reg, val = dummy_bus._written_bytes[-1]
    assert reg == mpr121.GPIO_SET
    # val should be 1 << (6 - 4) = 1 << 2 = 4
    assert val == (1 << (6 - 4))

    dummy_bus._written_bytes.clear()
    # Set channel 7 low
    m.gpio_set(7, False)
    reg, val = dummy_bus._written_bytes[-1]
    assert reg == mpr121.GPIO_CLR
    assert val == (1 << (7 - 4))


def test_is_touched_property(dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)

    # Set the private field _touched manually
    m._touched = 0b1010101010101  # some bits set for 13 channels

    for ch in range(mpr121.NCH):
        val = m.is_touched(ch)
        expected = (m._touched & (1 << ch)) > 0
        assert val == expected


def test_on_touch_sets_handlers_and_calls(dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)

    calls = []

    def handler_with_channel(touched, ch):
        calls.append((touched, ch))

    def handler_without_channel(touched):
        calls.append((touched,))

    # Register handler with 2 parameters for channel 0
    m.on_touch(0, handler_with_channel)
    # Register handler with 1 parameter for channel 1
    m.on_touch(1, handler_without_channel)

    # Trigger change on channel 0 and 1
    # We'll simulate _handle_touch manually

    # touched bits: channels 0 and 1 touched (bits 0 and 1)
    touched = 0b11
    # changed bits: channels 0 and 1 changed (bits 0 and 1)
    changed = 0b11

    m._handle_touch(touched, changed)

    # Calls should contain both handlers called properly
    assert (True, 0) in calls or (False, 0) in calls
    assert any(len(c) in [1, 2] for c in calls)
    assert any(c[0] is True or c[0] is False for c in calls)

    calls.clear()
    # Remove handler by setting None, and test fallback to noop (no call)
    m.on_touch(0, None)
    m._handle_touch(touched, 1 << 0)  # only channel 0 changed
    # No calls should be made since noop does nothing
    assert calls == []


def test_update_touch_state_changes_and_calls_handlers(monkeypatch):
    seed_all()
    dummy_bus_device = DummyBusDevice()
    # Setup read_word ETS fake values for touched()
    # On first call, touching channels 0,1
    dummy_bus_device._read_word_return = 0b11

    monkeypatch.setattr(mpr121, "Bus", lambda bus: type("Bus", (), {"device": lambda s, addr=None: dummy_bus_device})())

    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)

    called = []

    def my_handler(touched, ch):
        called.append((touched, ch))

    m.on_touch(0, my_handler)
    m.on_touch(1, my_handler)

    # Initially _touched is zero, calling update_touch_state should detect change
    m._touched = 0

    m.update_touch_state()
    # Should detect touch on 0 and 1 (changed bits 0b11)
    # Handlers for channel 0 and 1 should be called
    assert len(called) == 2
    # on second call, touched bits same, so no change, no handlers called
    called.clear()
    m.update_touch_state()
    assert called == []


def test_setup_calls_reset_and_configure_and_thresh(dummy_bus):
    seed_all()
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)
    dummy_bus._written_bytes.clear()
    dummy_bus._written_words.clear()

    # call setup with reset enabled
    m.setup(reset=True, channels=5, prox=2, threshold=30, debounce=3, auto_config=False)

    # reset should have been called, i.e. write_byte to SRESET with 0x63
    reset_calls = [w for w in dummy_bus._written_bytes if w[0] == mpr121.SRESET and w[1] == 0x63]
    assert reset_calls

    # configure should have been called with prox=0 touch=0 and later prox=2 touch=5 (channels)
    configure_calls = [w for w in dummy_bus._written_bytes if w[0] == mpr121.ECR]
    assert configure_calls
    # touch threshold for channels and proximity channel (12) should also be set
    # check that touch threshold is written at two different calls: for normal channels and for proximity(12)
    touch_thresh_regs = [mpr121.TTH + 2*i for i in range(12)] + [mpr121.TTH + 2*12]
    tenured_regs = set(w[0] for w in dummy_bus._written_bytes)
    assert any(r in tenured_regs for r in touch_thresh_regs)

    # debounce register set
    debounce_calls = [w for w in dummy_bus._written_bytes if w[0] == mpr121.DEB]
    assert debounce_calls


@pytest.mark.parametrize("irq_val", [0, 4])
def test_init_irq_and_handlers(monkeypatch, irq_val):
    seed_all()
    class DummyIRQ:
        def __init__(self, pin, callback):
            self.pin = pin
            self.callback = callback
            self.called = True

    class DummyPoller:
        def __init__(self, callback):
            self.callback = callback
            self.called = True

    monkeypatch.setattr(mpr121, "IRQHandler", DummyIRQ)
    monkeypatch.setattr(mpr121, "Poller", DummyPoller)

    dummy_bus_device = DummyBusDevice()
    monkeypatch.setattr(mpr121, "Bus", lambda bus: type("Bus", (), {"device": lambda s, addr=None: dummy_bus_device})())

    if irq_val != 0:
        m = mpr121.MPR121(bus=1, irq=irq_val, handlers=1, setup=0, reset=0)
        # IRQHandler should be called
        # No easy way to check directly but no exception means it was set
    else:
        m = mpr121.MPR121(bus=1, irq=irq_val, handlers=1, setup=0, reset=0)
        # Poller should be called as fallback when irq=0


def test_dump_runs_without_exception(monkeypatch, capsys):
    seed_all()
    dummy_bus_device = DummyBusDevice()

    # Prepare dummy bus data for dump
    dummy_bus_device._read_byte_return = 0
    # For read_byte to TTH and RTH registers, return 50 for TTH, 40 for RTH
    def dummy_read_byte(reg):
        if reg >= mpr121.TTH and reg < mpr121.TTH + 2 * mpr121.NCH:
            # Touch threshold: even addr offsets are TTH, odd are RTH
            offset = reg - mpr121.TTH
            if offset % 2 == 0:
                return 50
            else:
                return 40
        elif reg in (mpr121.CDC + i for i in range(mpr121.NCH)):
            return 10
        elif mpr121.CDT <= reg <= mpr121.CDT + 5:
            return 6
        return 0
    dummy_bus_device.read_byte = dummy_read_byte

    # Prepare read_block to return dummy data for registers in dump
    # Return zeros for 128 bytes (4*32)
    dummy_bus_device._read_block_return = [0]* (32*4)

    # Setup touched and out_of_range to return arbitrary values without exceptions
    monkeypatch.setattr(mpr121.MPR121, "touched", lambda self, raise_on_failure=1: 0b1010)
    monkeypatch.setattr(mpr121.MPR121, "out_of_range", lambda self, raise_on_failure=1: 0b0101)
    monkeypatch.setattr(mpr121.MPR121, "gpio_status", lambda self: 0b00110011)
    monkeypatch.setattr(mpr121.MPR121, "electrode_data", lambda self: [100]*mpr121.NCH)
    monkeypatch.setattr(mpr121.MPR121, "baseline", lambda self, *a, **k: [200]*mpr121.NCH)

    monkeypatch.setattr(mpr121, "Bus", lambda bus: type("Bus", (), {"device": lambda s, addr=None: dummy_bus_device})())

    m = mpr121.MPR121(bus=1, handlers=0, setup=0, reset=0)

    # run dump with 1 loop and registers printed enabled
    # There should be no exceptions and some output captured
    m.dump(regs=True, up=True, loop=1)

    out = capsys.readouterr().out
    assert isinstance(out, str)
    # It should print lines that contain channel index (0-12):
    for i in range(mpr121.NCH):
        assert f"{i}:" in out


# End of test file
