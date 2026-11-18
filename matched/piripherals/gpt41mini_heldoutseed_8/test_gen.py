import builtins
import types
import pytest

import piripherals.mpr121 as mpr121


class DummyBusDevice:
    def __init__(self):
        self.written = []
        self.read_bytes = {}
        self.read_words = {}
        self.read_blocks = {}

    def write_byte(self, reg, val):
        self.written.append(('write_byte', reg, val))

    def write_word(self, reg, val):
        self.written.append(('write_word', reg, val))

    def write_block(self, reg, vals):
        self.written.append(('write_block', reg, tuple(vals)))

    def read_byte(self, reg):
        # Return previously set value or zero
        return self.read_bytes.get(reg, 0)

    def read_word(self, reg):
        # Return previously set value or zero
        return self.read_words.get(reg, 0)

    def read_block(self, reg, length):
        # Return previously set block or zeros
        return self.read_blocks.get(reg, [0] * length)


class DummyBus:
    def device(self, addr):
        return DummyBusDevice()


class DummyIRQHandler:
    def __init__(self, irq, callback):
        self.irq = irq
        self.callback = callback
        self.started = True

    def stop(self):
        self.started = False


class DummyPoller:
    def __init__(self, callback):
        self.callback = callback
        self.started = True

    def stop(self):
        self.started = False


@pytest.fixture(autouse=True)
def patch_bus_irq_poller(monkeypatch):
    # Patch Bus in piripherals.mpr121 to DummyBus
    monkeypatch.setattr(mpr121, "Bus", DummyBus)
    # Patch IRQHandler and Poller with dummy classes
    monkeypatch.setattr(mpr121, "IRQHandler", DummyIRQHandler)
    monkeypatch.setattr(mpr121, "Poller", DummyPoller)
    yield


def test_init_with_defaults_creates_bus_and_handlers(monkeypatch):
    m = mpr121.MPR121()
    assert isinstance(m._bus, DummyBusDevice)
    # Handlers list of length NCH
    assert len(m._handlers) == mpr121.NCH
    # Handlers initialized to noop (function)
    for h in m._handlers:
        assert callable(h)


def test_init_with_existing_bus(monkeypatch):
    dummy_bus = DummyBus()
    device = dummy_bus.device(0x5a)
    # Provide dummy bus.device to be returned on device() call
    monkeypatch.setattr(dummy_bus, "device", lambda addr: device)
    m = mpr121.MPR121(bus=dummy_bus, handlers=0, setup=0, reset=0)
    assert m._bus is device


def test_update_touch_state_calls_handle_touch(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    # Patch touched() to toggle bits on call
    states = [0b0011, 0b0101]
    m._touched = states[0]

    def fake_touched():
        return states.pop(0)

    m.touched = fake_touched
    called = []

    def fake_handle_touch(touched, changed):
        called.append((touched, changed))

    m._handle_touch = fake_handle_touch
    m.update_touch_state()
    # 0b0011 ^ 0b0101 = 0b0110 -> bits 1 and 2 changed
    # touched value is 0b0101
    assert called == [(0b0101, 0b0110)]
    # _touched updated
    assert m._touched == 0b0101


def test_handle_touch_calls_correct_handlers(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    calls = []

    # Set handlers that record calls, one for each channel
    for i in range(mpr121.NCH):
        def handler_factory(ch):
            def h(touched, *args):
                calls.append((ch, touched, args))
            return h

        m._handlers[i] = handler_factory(i)

    # touched bits: channels 1, 3 changed (bits 1 and 3)
    touched = 0b1010
    changed = 0b1010
    m._handle_touch(touched, changed)

    # Only channels 1 and 3 handlers called with correct touched boolean
    expected = [
        (1, bool(touched & (1 << 1)), (1,)),
        (3, bool(touched & (1 << 3)), (3,))
    ]
    # But _handle_touch calls h((touched & m) > 0, i), so handler gets (bool, channel)
    # Our handler accepts (touched, *args), so args has channel if passed
    # The code uses try except and tries to call with two args, fallback with one
    # Our handler gets called with two args, first a bool, second the channel int
    assert all(call[0] in [1, 3] for call in calls)
    assert len(calls) == 2
    for ch, touched_val, args in calls:
        assert typeof := isinstance(touched_val, bool)
        assert args == (ch,)


def test_is_touched_returns_correct_status(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    m._touched = 0b10101
    for i in range(mpr121.NCH):
        expected = ((m._touched & (1 << i)) > 0)
        assert m.is_touched(i) is expected


def test_on_touch_register_and_remove_handlers(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    called = []

    def handler(state, ch=None):
        called.append((state, ch))

    m.on_touch(0, handler)
    # The registered handler should call handler and add to called
    # Emulate _handle_touch invoking handler
    m._handlers[0](True, 0)
    assert called == [(True, 0)]

    # Remove handler by passing None (should replace with noop which doesn't raise)
    m.on_touch(0, None)
    # Calling handler now should do nothing
    m._handlers[0](False, 0)  # no exception


def test_reset_writes_soft_reset_command(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    m.reset()
    assert ("write_byte", mpr121.SRESET, 0x63) in m._bus.written


def test_configure_writes_correct_ecr_value(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    m.configure(cl=2, prox=3, touch=7)
    expected_val = (2 << 6) | (3 << 4) | 7
    assert ("write_byte", mpr121.ECR, expected_val) in m._bus.written


def test_touched_returns_value_and_raises(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)

    # Set read_word to return a value with overcurrent bit set (bit 15)
    m._bus.read_word = lambda reg: 1 << 15

    with pytest.raises(Exception, match="overcurrent"):
        m.touched()

    # Set to normal value but out_of_range reports True -> raises 'out of range'
    m._bus.read_word = lambda reg: 0x100
    m.out_of_range = lambda raise_on_failure=1: (_ for _ in ()).throw(Exception('out of range'))
    with pytest.raises(Exception, match="out of range"):
        m.touched()

    # Without raising on failure should not raise and return masked value
    m._bus.read_word = lambda reg: 0x1234
    m.out_of_range = lambda raise_on_failure=1: False
    val = m.touched(raise_on_failure=0)
    assert val == (0x1234 & 0x1fff)


def test_electrode_data_reads_and_parse(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    # Compose block of bytes 2 * NCH: for channel i, LSB=i, MSB=255 - i
    block = []
    for i in range(mpr121.NCH):
        block.append(i)
        block.append(255 - i)
    m._bus.read_block = lambda reg, length: block
    data = m.electrode_data()
    # check data matches (MSB << 8) + LSB for each channel
    for i, val in enumerate(data):
        expected = ((255 - i) << 8) + i
        assert val == expected


def test_baseline_no_args_returns_processed_block(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    raw = list(range(mpr121.NCH))
    m._bus.read_block = lambda reg, length: raw
    out = m.baseline()
    # baseline shifts each byte by 2 bits
    for r, o in zip(raw, out):
        assert o == (r << 2)


def test_baseline_with_args_calls_write_block(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    m._bus.written.clear()

    # rft=0 (rising), prox=False writes 4 bytes: mhd, nhd, ncl, fdl at MHD+0
    m.baseline(rft=0, mhd=4, nhd=5, ncl=6, fdl=7, prox=False)
    reg = mpr121.MHD + 4 * 0
    found = False
    for w in m._bus.written:
        if w[0] == "write_block" and w[1] == reg:
            assert w[2] == (4, 5, 6, 7)
            found = True
    assert found

    m._bus.written.clear()

    # rft=2 (touched), prox=True writes 3 bytes: nhd, ncl, fdl at MHDX + 8
    m.baseline(rft=2, nhd=8, ncl=9, fdl=10, prox=True)
    reg = mpr121.MHDX + 8
    found = False
    for w in m._bus.written:
        if w[0] == "write_block" and w[1] == reg:
            assert w[2] == (8, 9, 10)
            found = True
    assert found


@pytest.mark.parametrize("release,expected_release", [(None, 6), (5, 5)])
def test_threshold_writes_all_channels_and_single(monkeypatch, release, expected_release):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    m._bus.written.clear()
    touch = 10

    # When channel=-1: calls threshold on all channels (NCH)
    m.threshold(touch, release)
    # find all writes for all channels and release threshold
    touch_count = 0
    release_count = 0
    for call in m._bus.written:
        if call[0] == 'write_byte':
            reg, val = call[1], call[2]
            if reg >= mpr121.TTH and reg < mpr121.TTH + 2 * mpr121.NCH:
                if (reg - mpr121.TTH) % 2 == 0:
                    # touch registers
                    assert val == touch
                    touch_count += 1
                else:
                    # release registers
                    # release defaults to 0.6 * touch rounded down = 6 in this case
                    assert val == expected_release
                    release_count += 1
    assert touch_count == mpr121.NCH
    assert release_count == mpr121.NCH

    m._bus.written.clear()
    # Test with channel specific threshold writes single channel only
    m.threshold(touch=20, release=15, channel=5)
    found_touch = False
    found_release = False
    for call in m._bus.written:
        if call[0] == "write_byte":
            reg, val = call[1], call[2]
            if reg == mpr121.TTH + 2 * 5:
                assert val == 20
                found_touch = True
            if reg == mpr121.RTH + 2 * 5:
                assert val == 15
                found_release = True
    assert found_touch
    assert found_release


@pytest.mark.parametrize("release", [None, 3])
def test_debounce_writes_proper_byte(monkeypatch, release):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    m._bus.written.clear()
    touch = 4
    m.debounce(touch=touch, release=release if release is not None else -1)
    expected_release = release if release is not None else touch
    expected_val = (expected_release << 4) | touch
    found = False
    for c in m._bus.written:
        if c[0] == "write_byte" and c[1] == mpr121.DEB:
            assert c[2] == expected_val
            found = True
    assert found


def test_filter_writes_word_combining_bytes(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    m._bus.written.clear()
    cdc = 22
    cdt = 3
    ffi = 2
    sfi = 1
    esi = 5
    m.filter(cdc=cdc, cdt=cdt, ffi=ffi, sfi=sfi, esi=esi)
    # lb = (ffi << 6) | cdc
    lb = (ffi << 6) | cdc
    # hb = (cdt << 5) | (sfi << 3) | esi
    hb = (cdt << 5) | (sfi << 3) | esi
    expected_word = (hb << 8) | lb
    found = False
    for c in m._bus.written:
        if c[0] == "write_word" and c[1] == mpr121.AFE1:
            assert c[2] == expected_word
            found = True
    assert found


def test_charge_writes_correct_bytes(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    # We'll patch read_byte to return 0xff for CDT register
    m._bus.read_byte = lambda reg: 0xff
    m._bus.written.clear()
    channel = 5
    cdc = 12
    cdt = 7
    m.charge(channel, cdc, cdt)

    # Expect write_byte called for CDC + channel and CDT + channel//2
    assert ('write_byte', mpr121.CDC + channel, cdc) in m._bus.written
    reg = mpr121.CDT + channel // 2
    # Check that write_byte to reg occurs with proper bits
    found = False
    for call in m._bus.written:
        if call[0] == "write_byte" and call[1] == reg:
            # value should be (old_value & ~(0xf << s)) | (cdt << s)
            val = call[2]
            s = 4 * (channel % 2)
            old = 0xff
            expected_val = (old & ~(0xf << s)) | (cdt << s)
            assert val == expected_val
            found = True
    assert found


def test_auto_config_writes_bytes_and_word(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    m._bus.written.clear()
    m.auto_config(
        ace=1, are=0, bva=2, retry=1, afes=3,
        scts=1, acfie=0, arfie=1, oorie=0,
        usl=50, lsl=40, tl=60
    )
    # Compose expected lb and hb, word:
    lb = (3 << 6) | (1 << 4) | (2 << 2) | (0 << 1) | 1
    hb = (1 << 7) | (0 << 2) | (1 << 1) | 0
    expected_word = (hb << 8) + lb
    # Check write_word to ACNF_C0 with expected_word
    found_word = False
    found_usl = False
    found_lsl = False
    found_tl = False
    for call in m._bus.written:
        if call[0] == "write_word" and call[1] == mpr121.ACNF_C0:
            assert call[2] == expected_word
            found_word = True
        if call[0] == "write_byte":
            if call[1] == mpr121.ACNF_USL:
                assert call[2] == 50
                found_usl = True
            elif call[1] == mpr121.ACNF_LSL:
                assert call[2] == 40
                found_lsl = True
            elif call[1] == mpr121.ACNF_TL:
                assert call[2] == 60
                found_tl = True
    assert found_word and found_usl and found_lsl and found_tl


def test_out_of_range_raises_on_errors(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    # Bit 15 set triggers 'auto config failed'
    m._bus.read_word = lambda reg: (1 << 15)
    with pytest.raises(Exception, match='auto config failed'):
        m.out_of_range()

    # Bit 14 set triggers 'auto reconfig failed'
    m._bus.read_word = lambda reg: (1 << 14)
    with pytest.raises(Exception, match='auto reconfig failed'):
        m.out_of_range()

    # Without raise_on_failure returns masked value
    m._bus.read_word = lambda reg: 0xABCD
    r = m.out_of_range(raise_on_failure=0)
    assert r == (0xABCD & 0x1fff)


def test_gpio_setup_writes_correct_registers(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)

    # Controls to record register writes and reads
    reg_bytes = {}

    def dummy_read_byte(addr):
        return reg_bytes.get(addr, 0)

    def dummy_write_byte(addr, val):
        reg_bytes[addr] = val

    m._bus.read_byte = dummy_read_byte
    m._bus.write_byte = dummy_write_byte

    # Call gpio_setup for channel 7 with mode 3 (0b11), output True, enable True
    m.gpio_setup(channel=7, output=1, mode=3, enable=1)
    ch = 7 - 4
    # Bits for mode: GPIO_CTL0 -> high-bit, GPIO_CTL1 -> low-bit of mode
    # Expect for GPIO_CTL0 bit ch to be 1 (mode & 0b10)
    # Expect for GPIO_CTL1 bit ch to be 1 (mode & 0b01)
    assert ((reg_bytes[mpr121.GPIO_CTL0] >> ch) & 1) == ((3 & 0b10) >> 1)
    assert ((reg_bytes[mpr121.GPIO_CTL1] >> ch) & 1) == (3 & 0b01)
    # GPIO_DIR bit ch = 1 (output True)
    assert ((reg_bytes[mpr121.GPIO_DIR] >> ch) & 1) == 1
    # GPIO_EN bit ch = 1 (enable True)
    assert ((reg_bytes[mpr121.GPIO_EN] >> ch) & 1) == 1


def test_gpio_status_returns_bus_read_byte(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    m._bus.read_byte = lambda addr: 0xA5 if addr == mpr121.GPIO_DAT else 0x00
    val = m.gpio_status()
    assert val == 0xA5


def test_gpio_set_writes_set_or_clr(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    m._bus.written.clear()
    # Value True uses GPIO_SET register at channel 4
    m.gpio_set(channel=4, value=1)
    assert ('write_byte', mpr121.GPIO_SET, 1 << 0) in m._bus.written
    m._bus.written.clear()
    # Value False uses GPIO_CLR register at channel 7 (bit 3)
    m.gpio_set(channel=7, value=0)
    assert ('write_byte', mpr121.GPIO_CLR, 1 << 3) in m._bus.written


def test_setup_calls_methods_in_expected_order(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    calls = []

    m.reset = lambda: calls.append('reset')
    m.configure = lambda **kwargs: calls.append(('configure', kwargs))
    m.filter = lambda **kwargs: calls.append(('filter', kwargs))
    m.auto_config = lambda: calls.append('auto_config')
    m.threshold = lambda **kwargs: calls.append(('threshold', kwargs))
    m.debounce = lambda **kwargs: calls.append(('debounce', kwargs))
    m.baseline = lambda **kwargs: calls.append(('baseline', kwargs))

    m.setup(reset=True, channels=11, prox=1, threshold=44, debounce=3, auto_config=True)

    # Check that reset was called first
    assert calls[0] == 'reset'
    # Check that configure called first with prox=0 touch=0
    assert any(call[0] == 'configure' and call[1].get('prox') == 0 and call[1].get('touch') == 0 for call in calls)
    # filter called
    assert any(call[0] == 'filter' for call in calls)
    # auto_config called
    assert 'auto_config' in calls
    # threshold called at least twice (once for all channels, once for channel 12)
    thresholds = [call for call in calls if callable(call) is False and (call[0] == 'threshold')]
    # But we used lambdas so thresholds are ('threshold', kwargs)
    threshold_calls = [c for c in calls if isinstance(c, tuple) and c[0] == 'threshold']
    assert len(threshold_calls) >= 2
    # debounce called
    assert any(call[0] == 'debounce' for call in calls)
    # baseline called at least 4 times (two for rft=0,1 with prox False then prox True)
    baseline_calls = [c for c in calls if isinstance(c, tuple) and c[0] == 'baseline']
    assert len(baseline_calls) >= 4
    # configure called with prox=1 and touch=channels last
    assert calls[-1][0] == 'configure'
    assert calls[-1][1].get('prox') == 1
    assert calls[-1][1].get('touch') == 11


def test_on_touch_handler_callable_without_channel(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)

    called = []
    # Handler with single argument only
    def handler(touched):
        called.append(touched)

    m.on_touch(2, handler)
    # _handle_touch calls handler with two arguments, fallback on one argument due TypeError

    # We'll call _handle_touch to test fallback
    m._handle_touch(touched=0b100, changed=0b100)  # channel 2 bit set

    # Handler should be called with True (bit set)
    assert True in called


def test_dump_runs_without_error(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)

    # Patch methods to return simple values to avoid errors
    m._bus.read_byte = lambda reg: 5
    m._bus.read_block = lambda reg, length: [0] * length
    m._bus.read_word = lambda reg: 0
    m.touched = lambda raise_on_failure=1: 0
    m.out_of_range = lambda raise_on_failure=1: 0
    m.gpio_status = lambda: 0
    m.electrode_data = lambda: [10] * mpr121.NCH
    m.baseline = lambda rft=-1, mhd=0, nhd=0, ncl=0, fdl=0, prox=0: [20] * mpr121.NCH

    # Capture prints
    import io
    import sys

    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Run dump with minimal loop 1 and no register dump
    m.dump(regs=0, up=0, loop=1)

    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "E:" in output


def test_invalid_channel_gpio_setup_raises(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    # channel < 4 should assert
    with pytest.raises(AssertionError):
        m.gpio_setup(3, output=True)
    with pytest.raises(AssertionError):
        m.gpio_setup(12, output=True)


def test_gpio_set_invalid_channel_assert(monkeypatch):
    m = mpr121.MPR121(handlers=0, setup=0, reset=0)
    with pytest.raises(AssertionError):
        m.gpio_set(3, True)
    with pytest.raises(AssertionError):
        m.gpio_set(12, False)
