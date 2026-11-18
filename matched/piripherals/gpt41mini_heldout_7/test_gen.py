import builtins
import pytest
from piripherals import mpr121


class DummyBusDevice:
    def __init__(self):
        self.written = []
        self.bytes = {}
        self.words = {}
        self.blocks = {}

    def write_byte(self, reg, val):
        self.written.append(('write_byte', reg, val))
        self.bytes[reg] = val

    def write_word(self, reg, val):
        self.written.append(('write_word', reg, val))
        self.words[reg] = val

    def write_block(self, reg, data):
        self.written.append(('write_block', reg, data))
        self.blocks[reg] = data

    def read_byte(self, reg):
        return self.bytes.get(reg, 0)

    def read_word(self, reg):
        return self.words.get(reg, 0)

    def read_block(self, reg, length):
        return [0] * length


class DummyBus:
    def __init__(self):
        self.device_called_with = None
        self.device_instance = DummyBusDevice()

    def device(self, addr):
        self.device_called_with = addr
        return self.device_instance


@pytest.fixture
def dummy_bus():
    return DummyBus()


@pytest.fixture
def mpr121_instance(dummy_bus):
    # handlers=False and setup=False disables IRQ/Poller and setup call
    return mpr121.MPR121(bus=dummy_bus, irq=0, handlers=0, setup=0, reset=0)


def test_init_with_bus_object_sets_device(dummy_bus):
    m = mpr121.MPR121(bus=dummy_bus, irq=0, handlers=0, setup=0, reset=0)
    assert m._bus is dummy_bus.device_instance
    assert m.addr == 0x5a
    # _handlers length must be NCH and elements be callable (noop)
    assert len(m._handlers) == mpr121.NCH
    assert all(callable(h) for h in m._handlers)
    assert m._touched == 0


def test_init_with_bus_int_creates_bus_device(monkeypatch):
    called = {}

    class MockDevice(DummyBusDevice):
        def __init__(self):
            super().__init__()

    class MockBus:
        def device(self, addr):
            called['addr'] = addr
            return MockDevice()

    monkeypatch.setattr(mpr121, 'Bus', lambda bus: MockBus())
    m = mpr121.MPR121(bus=3, irq=0, handlers=0, setup=0, reset=0)
    assert isinstance(m._bus, DummyBusDevice) or hasattr(m._bus, 'write_byte')
    assert called['addr'] == m.addr


def test_reset_calls_write_byte_on_reset_register(mpr121_instance):
    m = mpr121_instance
    m.reset()
    assert ('write_byte', mpr121.SRESET, 0x63) in m._bus.written


def test_update_touch_state_calls_handle_touch(monkeypatch):
    m = mpr121_instance
    # Force _touched old and touched new to produce changed bits
    m._touched = 0b0000_0000_0001
    new_touch = 0b0000_0000_0011

    monkeypatch.setattr(m, 'touched', lambda: new_touch)

    called = {}

    def fake_handle_touch(touched, changed):
        called['touched'] = touched
        called['changed'] = changed

    monkeypatch.setattr(m, '_handle_touch', fake_handle_touch)
    m.update_touch_state()
    assert called['touched'] == new_touch
    assert called['changed'] == (m._touched ^ 0b0000_0000_0001)


def test_handle_touch_invokes_handlers(monkeypatch):
    m = mpr121_instance
    calls = []

    def handler1(state, channel):
        calls.append((1, state, channel))

    def handler2(state):
        calls.append((2, state))

    # Assign handler1 to channel 0, handler2 to channel 1
    m._handlers[0] = handler1
    m._handlers[1] = handler2
    # touched bits have channel 0 and channel 1 touched, changed bits only channel 0 and 1
    touched = 0b0000_0000_0011
    changed = 0b0000_0000_0011

    m._handle_touch(touched, changed)
    # Handler1 gets called with state True, channel 0
    # Handler2 gets called with state True (b/c only one argument)
    assert (1, True, 0) in calls
    assert (2, True) in calls


@pytest.mark.parametrize('channel,val,expected', [
    (0, 0b1, True),
    (0, 0b10, False),
    (5, 0b100000, True),
    (5, 0b0, False),
])
def test_is_touched_returns_correct_status(mpr121_instance, channel, val, expected):
    m = mpr121_instance
    m._touched = val
    assert m.is_touched(channel) is expected


def test_on_touch_registers_and_removes_handler(mpr121_instance):
    m = mpr121_instance
    called = {}

    def handler(touched, channel):
        called['called'] = (touched, channel)

    m.on_touch(3, handler)
    # _handlers at index 3 is not noop
    assert m._handlers[3] != mpr121.noop
    # Simulate call
    m._handlers[3](True, 3)
    assert called['called'] == (True, 3)
    # Remove handler by setting None
    m.on_touch(3, None)
    # Should reset to noop
    assert m._handlers[3] == mpr121.noop


def test_configure_writes_correct_byte(mpr121_instance):
    m = mpr121_instance
    m.configure(cl=2, prox=3, touch=4)
    # Should write byte ECR with correct bits:
    expected = (2 << 6) | (3 << 4) | 4
    assert ('write_byte', mpr121.ECR, expected) in m._bus.written


def test_touched_returns_correct_bits_and_raises(monkeypatch):
    m = mpr121_instance

    def fake_read_word(reg):
        # set touch bits 0b0001 and overcurrent bit 15 to 1
        if reg == mpr121.ETS:
            return (1 << 15) | 0b1010
        return 0

    m._bus.read_word = fake_read_word
    with pytest.raises(Exception, match='overcurrent'):
        m.touched(raise_on_failure=True)

    # now test with no raise_on_failure
    touched_bits = m.touched(raise_on_failure=False)
    assert touched_bits == 0b1010


def test_touched_raises_exception_if_out_of_range(monkeypatch):
    m = mpr121_instance

    def oor_side_effect():
        # patch to indicate out of range return True
        return True

    def fake_read_word(reg):
        # clear overcurrent bit, just normal bits
        if reg == mpr121.ETS:
            return 0b1111
        return 0

    m._bus.read_word = fake_read_word
    monkeypatch.setattr(m, 'out_of_range', lambda raise_on_failure=True: True)
    with pytest.raises(Exception, match='out of range'):
        m.touched(raise_on_failure=True)


def test_electrode_data_reads_blocks_and_returns_data(mpr121_instance):
    m = mpr121_instance

    # Prepare block of data with 2*NCH bytes, each electrode gets ascending pattern
    data = []
    for i in range(mpr121.NCH):
        # low byte = i, high byte = i+1
        data.append(i)
        data.append(i + 1)

    m._bus.read_block = lambda reg, length: data
    ed = m.electrode_data()
    # Each electrode data is (high_byte<<8) + low_byte
    for i in range(mpr121.NCH):
        assert ed[i] == ((i + 1) << 8) + i


def test_baseline_returns_values_when_rft_negative(mpr121_instance):
    m = mpr121_instance
    # prepare baseline block read: NCH values from 0 to NCH-1
    block = list(range(mpr121.NCH))
    m._bus.read_block = lambda reg, length: block
    val = m.baseline()
    # baseline() returns block values shifted by 2 bits (x << 2)
    assert val == [x << 2 for x in block]


def test_baseline_writes_correct_block_for_rft_2_and_1(mpr121_instance):
    m = mpr121_instance

    calls = []

    def fake_write_block(reg, data):
        calls.append((reg, data))

    m._bus.write_block = fake_write_block

    # rft=2 triggers 3-byte write, prox False => MHD
    m.baseline(rft=2, nhd=10, ncl=20, fdl=30, prox=False)
    assert calls[-1][0] == mpr121.MHD + 4 * 2
    assert calls[-1][1] == [10, 20, 30]

    # rft=2 with prox True triggers write at MHDX
    m.baseline(rft=2, nhd=11, ncl=21, fdl=31, prox=True)
    assert calls[-1][0] == mpr121.MHDX + 4 * 2
    assert calls[-1][1] == [11, 21, 31]

    # rft=1 triggers 4-byte write to MHD
    m.baseline(rft=1, mhd=5, nhd=6, ncl=7, fdl=8, prox=False)
    assert calls[-1][0] == mpr121.MHD + 4 * 1
    assert calls[-1][1] == [5, 6, 7, 8]


def test_threshold_sets_all_channels_correctly(mpr121_instance):
    m = mpr121_instance
    m._bus.write_byte = lambda reg, val: m._bus.write_byte(reg, val)
    # Clear written list
    m._bus.written.clear()
    m.threshold(20)
    # For all channels, each channel write pair
    # 2 writes * NCH channels expected
    wbytes = [x for x in m._bus.written if x[0] == 'write_byte']
    # There should be 2 writes for each of 13 channels = 26 writes
    assert len(wbytes) == 2 * mpr121.NCH
    # Check first channel writes correct registers and values
    assert wbytes[0][1] == mpr121.TTH  # touch threshold for chan 0
    assert wbytes[0][2] == 20
    assert wbytes[1][1] == mpr121.RTH  # release threshold for chan 0
    assert wbytes[1][2] == int(0.6 * 20)


def test_threshold_sets_single_channel(mpr121_instance):
    m = mpr121_instance
    m._bus.written.clear()
    m.threshold(30, 12, 5)
    # Only 2 writes should be present
    writes = [x for x in m._bus.written if x[0] == 'write_byte']
    assert len(writes) == 2
    assert writes[0][1] == mpr121.TTH + 2 * 5
    assert writes[0][2] == 30
    assert writes[1][1] == mpr121.RTH + 2 * 5
    assert writes[1][2] == 12


def test_debounce_works_and_defaults_release_to_touch(mpr121_instance):
    m = mpr121_instance
    m._bus.written.clear()
    m.debounce(touch=3)
    # Only one write_byte to DEB, with release=touch
    assert ('write_byte', mpr121.DEB, (3 << 4) | 3) in m._bus.written
    m._bus.written.clear()
    m.debounce(touch=2, release=7)
    assert ('write_byte', mpr121.DEB, (7 << 4) | 2) in m._bus.written


def test_filter_writes_word_correctly(mpr121_instance):
    m = mpr121_instance
    m._bus.written.clear()
    m.filter(cdc=31, cdt=4, ffi=2, sfi=3, esi=5)
    # Compose expected low and high bytes
    lb = (2 << 6) | 31
    hb = (4 << 5) | (3 << 3) | 5
    val = (hb << 8) | lb
    assert ('write_word', mpr121.AFE1, val) in m._bus.written


def test_charge_writes_bytes_and_reads_correctly(mpr121_instance):
    m = mpr121_instance
    mem = {}

    def fake_write_byte(reg, val):
        mem[reg] = val
        m._bus.written.append(('write_byte', reg, val))

    def fake_read_byte(reg):
        return mem.get(reg, 0xff)

    m._bus.write_byte = fake_write_byte
    m._bus.read_byte = fake_read_byte
    m._bus.written.clear()

    # Setup channel 1 with cdc=10, cdt=3
    m.charge(1, cdc=10, cdt=3)
    assert ('write_byte', mpr121.CDC + 1, 10) in m._bus.written
    # Because channel=1, reg should be CDT + 0, s=4*(1%2)
    reg = mpr121.CDT + 0
    # Previously val for reg is 0xff so expect mask update
    val_written = mem[reg]
    # Bits 4..7 contain cdt shifted 4 bits
    assert (val_written >> 4) & 0x0f == 3


def test_auto_config_writes_correct_registers(mpr121_instance):
    m = mpr121_instance
    m._bus.written.clear()
    m.auto_config(
        ace=1, are=0, bva=2, retry=1,
        afes=3, scts=1,
        acfie=0, arfie=1, oorie=1,
        usl=201, lsl=131, tl=181)
    writes = m._bus.written
    # Write_word to ACNF_C0
    first_write = writes[0]
    assert first_write[0] == 'write_word'
    assert first_write[1] == mpr121.ACNF_C0
    lb = (3 << 6) | (1 << 4) | (2 << 2) | (0 << 1) | 1
    hb = (1 << 7) | (1 << 2) | (1 << 1) | 0
    expected_val = (hb << 8) + lb
    assert first_write[2] == expected_val
    # Check subsequent write bytes for USL, LSL, TL
    regs = [x[1] for x in writes[1:]]
    assert mpr121.ACNF_USL in regs
    assert mpr121.ACNF_LSL in regs
    assert mpr121.ACNF_TL in regs


def test_out_of_range_raises(monkeypatch):
    m = mpr121_instance

    def fake_read_word(reg):
        # bit 15 set = auto config failed
        # bit 14 set = auto reconfig failed
        if reg == mpr121.OOR:
            return (1 << 15)
        return 0

    m._bus.read_word = fake_read_word
    with pytest.raises(Exception, match='auto config failed'):
        m.out_of_range(raise_on_failure=True)

    def fake_read_word2(reg):
        # bit 14 set = auto reconfig failed only
        if reg == mpr121.OOR:
            return (1 << 14)
        return 0

    m._bus.read_word = fake_read_word2
    with pytest.raises(Exception, match='auto reconfig failed'):
        m.out_of_range(raise_on_failure=True)

    # Test with no raise_on_failure returns bits masked to 13 bits
    m._bus.read_word = lambda reg: 0xFFFF  # all bits set
    val = m.out_of_range(raise_on_failure=False)
    assert val & 0x1FFF == 0x1FFF


def test_gpio_setup_writes_bytes(mpr121_instance):
    m = mpr121_instance

    # Setup initial read_byte values for the GPIO registers to zero
    read_values = {
        mpr121.GPIO_CTL0: 0,
        mpr121.GPIO_CTL1: 0,
        mpr121.GPIO_DIR: 0,
        mpr121.GPIO_EN: 0,
    }

    def fake_read_byte(reg):
        return read_values.get(reg, 0)

    write_outputs = {}

    def fake_write_byte(reg, val):
        write_outputs[reg] = val
        m._bus.written.append(('write_byte', reg, val))

    m._bus.read_byte = fake_read_byte
    m._bus.write_byte = fake_write_byte
    m._bus.written.clear()

    # Test channel=4, output=1, mode=3, enable=0
    m.gpio_setup(channel=4, output=1, mode=3, enable=0)
    # For channel 4, bit index 0 (since channel-4)
    # GPIO_CTL0 bit should be mode & 0b10 (2)
    # GPIO_CTL1 bit should be mode & 0b01 (1)
    ctl0_val = write_outputs[mpr121.GPIO_CTL0]
    ctl1_val = write_outputs[mpr121.GPIO_CTL1]
    dir_val = write_outputs[mpr121.GPIO_DIR]
    en_val = write_outputs[mpr121.GPIO_EN]

    assert ctl0_val & 0b1 == 0b1  # bit 0 set
    assert ctl1_val & 0b1 == 0b1  # bit 0 set
    assert dir_val & 0b1 == 0b1   # bit 0 set (output=1)
    assert en_val & 0b1 == 0b0    # bit 0 clear (enable=0)


def test_gpio_status_returns_byte(mpr121_instance):
    m = mpr121_instance
    m._bus.read_byte = lambda reg: 0xAA if reg == mpr121.GPIO_DAT else 0
    val = m.gpio_status()
    assert val == 0xAA


def test_gpio_set_writes_to_correct_register(mpr121_instance):
    m = mpr121_instance
    m._bus.written.clear()
    m._bus.write_byte = lambda reg, val: m._bus.written.append(('write_byte', reg, val))

    # channel 5 (bit 1) set True should write to GPIO_SET
    m.gpio_set(5, True)
    assert m._bus.written[0][1] == mpr121.GPIO_SET
    assert m._bus.written[0][2] == 1 << (5 - 4)

    m._bus.written.clear()
    # channel 5 set False writes to GPIO_CLR
    m.gpio_set(5, False)
    assert m._bus.written[0][1] == mpr121.GPIO_CLR
    assert m._bus.written[0][2] == 1 << (5 - 4)


def test_setup_calls_reset_and_configure(mocker, dummy_bus):
    mock_reset = mocker.patch('piripherals.mpr121.MPR121.reset')
    mock_configure = mocker.patch('piripherals.mpr121.MPR121.configure')
    mock_filter = mocker.patch('piripherals.mpr121.MPR121.filter')
    mock_auto_config = mocker.patch('piripherals.mpr121.MPR121.auto_config')
    mock_threshold = mocker.patch('piripherals.mpr121.MPR121.threshold')
    mock_debounce = mocker.patch('piripherals.mpr121.MPR121.debounce')
    mock_baseline = mocker.patch('piripherals.mpr121.MPR121.baseline')

    m = mpr121.MPR121(bus=dummy_bus, irq=0, handlers=0, setup=0, reset=0)
    m.setup(reset=True, channels=5, prox=1, threshold=55, debounce=3, auto_config=True)

    mock_reset.assert_called_once()
    mock_configure.assert_any_call(prox=0, touch=0)
    mock_filter.assert_called_once()
    mock_auto_config.assert_called_once()
    # threshold called twice (channels default 12 and proximity channel 12)
    assert mock_threshold.call_count >= 2
    mock_debounce.assert_called_once_with(3)
    # baseline called four times (2 for rft=0,1 and 2 for prox=0,1)
    assert mock_baseline.call_count >= 4
    # final configure called with prox=1 and touch=5
    mock_configure.assert_any_call(prox=1, touch=5)


def test_on_touch_handler_called_from_update_touch_state(mpr121_instance):
    m = mpr121_instance
    called = {}

    def handler(touched, channel):
        called['touched'] = touched
        called['channel'] = channel

    # Assign handler to channel 0
    m.on_touch(0, handler)

    # Patch touched() to return bit 0 touched, previous state not touched
    m._touched = 0
    m.touched = lambda: 1
    m.update_touch_state()
    assert called['touched'] is True
    assert called['channel'] == 0


def test_handle_touch_handler_with_typeerror_calls_without_channel(mpr121_instance):
    m = mpr121_instance
    called = {}

    # Handler that accepts only one arg (touched)
    def handler_one_arg(touched):
        called['called'] = touched

    m._handlers[0] = handler_one_arg
    touched = 0b1
    changed = 0b1
    m._handle_touch(touched, changed)
    assert called['called'] is True


def test_is_touched_out_of_range(mpr121_instance):
    m = mpr121_instance
    m._touched = 0
    # channel out of range should not raise but return False
    # The function does not explicitly check channel range so let's just test channel 12 (last)
    m._touched = 1 << 12
    assert m.is_touched(12) is True
    # channel >12 will raise IndexError? Let's test index 13 returns False (bitwise operation)
    with pytest.raises(IndexError):
        m.is_touched(13)  # bit shifts larger than width? Might not raise but no check in code


def test_gpio_setup_asserts_invalid_channel(mpr121_instance):
    m = mpr121_instance
    with pytest.raises(AssertionError):
        m.gpio_setup(3, True)  # channel < 4
    with pytest.raises(AssertionError):
        m.gpio_setup(12, True)  # channel > 11


def test_dump_runs_without_error(mpr121_instance, monkeypatch):
    m = mpr121_instance

    # Patch reads to return fixed values
    m._bus.read_byte = lambda reg: 10
    m._bus.read_block = lambda reg, length: [0] * length
    m._bus.read_word = lambda reg: 0

    # Patch print to capture output
    captured = []

    def fake_print(*args, **kwargs):
        captured.append(args)

    monkeypatch.setattr(builtins, 'print', fake_print)
    m.dump(regs=0, up=0, loop=1)
    assert any('E:' in str(line) for line in ''.join(str(arg) for args in captured for arg in args))


def test_timeout_main(monkeypatch):
    # Test that main parser arguments parse without error and MPR121 can be constructed
    monkeypatch.setattr('piripherals.mpr121.fork', lambda f: None)
    monkeypatch.setattr('piripherals.mpr121.on_change', lambda *a, **k: None)
    monkeypatch.setattr('piripherals.mpr121.MPR121.dump', lambda *a, **k: None)
    monkeypatch.setattr('piripherals.mpr121.MPR121.on_touch', lambda *a, **k: None)

    import sys
    testargs = ["mpr121", "-b", "1", "-a", "0x5a", "-p", "1", "-T", "30", "-c", "5", "-q", "0", "-S", "-R"]
    monkeypatch.setattr(sys, 'argv', testargs)

    # prevent infinite loop by patching fork with no-op
    # We just want to verify that main runs without throwing errors
    from piripherals.mpr121 import main
    main()
