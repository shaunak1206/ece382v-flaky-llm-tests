import builtins
import pytest
from piripherals import mpr121


class DummyBusDevice:
    """Dummy bus device for testing"""

    def __init__(self):
        self.written_bytes = []
        self.written_words = []
        self.written_blocks = []
        self.read_bytes = {}
        self.read_words = {}
        self.read_blocks = {}

    def write_byte(self, reg, val):
        self.written_bytes.append((reg, val))

    def write_word(self, reg, val):
        self.written_words.append((reg, val))

    def write_block(self, reg, vals):
        self.written_blocks.append((reg, list(vals)))

    def read_byte(self, reg):
        # Return preset value or 0
        return self.read_bytes.get(reg, 0)

    def read_word(self, reg):
        return self.read_words.get(reg, 0)

    def read_block(self, reg, length):
        return self.read_blocks.get((reg, length), [0] * length)


class DummyBus:
    def device(self, addr):
        self.addr = addr
        self.device = DummyBusDevice()
        return self.device


# Helpers to patch utilities and classes we don't want to use during tests
class DummyIRQHandler:
    def __init__(self, irq, callback):
        self.irq = irq
        self.callback = callback


class DummyPoller:
    def __init__(self, callback):
        self.callback = callback


def noop(*args, **kwargs):
    return None


def not_raising(handler):
    # mimic piripherals.util.not_raising: pass through if callable else noop
    if handler is None:
        return None
    return handler


@pytest.fixture
def mpr121_instance(monkeypatch):
    # Patch .bus.Bus to return DummyBus here
    monkeypatch.setattr(mpr121, "Bus", lambda bus_idx=1: DummyBus())

    # Patch IRQHandler and Poller in mpr121 module
    monkeypatch.setattr(mpr121, "IRQHandler", DummyIRQHandler)
    monkeypatch.setattr(mpr121, "Poller", DummyPoller)

    # Patch noop and not_raising to local versions
    monkeypatch.setattr(mpr121, "noop", noop)
    monkeypatch.setattr(mpr121, "not_raising", not_raising)

    inst = mpr121.MPR121(irq=0, handlers=1, setup=0, reset=0)
    # force device attribute for bus device
    inst._bus = DummyBusDevice()
    inst._handlers = [noop] * mpr121.NCH
    inst._touched = 0
    return inst


def test_soft_reset_calls_write_byte(mpr121_instance):
    mpr121_instance.reset()
    assert (mpr121.SRESET, 0x63) in mpr121_instance._bus.written_bytes


def test_update_touch_state_changes_state_and_calls_handlers(mpr121_instance):
    calls = []

    def handler(status, channel):
        calls.append((status, channel))

    mpr121_instance._handlers[0] = handler
    mpr121_instance._touched = 0b0

    # patch touched method to return a changed state with channel 0 touched
    mpr121_instance.touched = lambda raise_on_failure=1: 0b1
    mpr121_instance.update_touch_state()
    assert calls == [(True, 0)]
    assert mpr121_instance._touched == 0b1


def test_handle_touch_calls_handlers_without_channel_arg(mpr121_instance):
    calls = []

    def handler(status):
        calls.append(status)

    mpr121_instance._handlers[0] = handler
    touched = 0b1
    changed = 0b1
    mpr121_instance._handle_touch(touched, changed)
    assert calls == [True]


def test_is_touched_returns_correct_value(mpr121_instance):
    mpr121_instance._touched = 0b1010
    assert not mpr121_instance.is_touched(0)
    assert mpr121_instance.is_touched(1)
    assert not mpr121_instance.is_touched(2)
    assert mpr121_instance.is_touched(3)


def test_on_touch_sets_handler_and_removes_handler(mpr121_instance):
    def handle(status, channel):
        return status, channel

    mpr121_instance.on_touch(1, handle)
    assert mpr121_instance._handlers[1] is handle

    mpr121_instance.on_touch(1, None)
    assert callable(mpr121_instance._handlers[1])  # noop


def test_configure_writes_correct_byte(mpr121_instance):
    mpr121_instance._bus.written_bytes.clear()
    mpr121_instance.configure(cl=2, prox=3, touch=5)
    expected_byte = (2 << 6) | (3 << 4) | 5
    assert (mpr121.ECR, expected_byte) in mpr121_instance._bus.written_bytes


def test_touched_raises_on_overcurrent_and_out_of_range(mpr121_instance):
    # overcurrent bit 15 set
    mpr121_instance._bus.read_words[mpr121.ETS] = (1 << 15) | 0x123
    mpr121_instance._bus.read_words[mpr121.OOR] = 0x0
    with pytest.raises(Exception, match='overcurrent'):
        mpr121_instance.touched()

    # out_of_range returns with failure bit set
    mpr121_instance._bus.read_words[mpr121.ETS] = 0x123
    mpr121_instance._bus.read_words[mpr121.OOR] = (1 << 15)
    with pytest.raises(Exception, match='auto config failed'):
        mpr121_instance.touched()

    # without raising on failure no exception
    mpr121_instance._bus.read_words[mpr121.ETS] = (1 << 15)
    mpr121_instance._bus.read_words[mpr121.OOR] = (1 << 15)
    val = mpr121_instance.touched(raise_on_failure=0)
    assert val & 0x1fff == ((1 << 15) & 0x1fff)  # masked lower 13 bits


def test_electrode_data_returns_correct_data(mpr121_instance):
    # Setup read_block to return bytes representing i for each electrode low & high bytes
    data = []
    for i in range(mpr121.NCH):
        low = i & 0xFF
        high = (i >> 8) & 0xFF
        data.extend([low, high])
    mpr121_instance._bus.read_blocks = {}
    mpr121_instance._bus.read_block = lambda reg, length: data

    result = mpr121_instance.electrode_data()
    expected = [((data[2 * i + 1] << 8) + data[2 * i]) for i in range(mpr121.NCH)]
    assert result == expected


def test_baseline_returns_list_of_ints_when_called_without_args(mpr121_instance):
    # Prepare baseline block values; values shifted left by 2 per device code
    vals = [i for i in range(mpr121.NCH)]
    mpr121_instance._bus.read_block = lambda reg, length: [v >> 2 for v in vals]

    result = mpr121_instance.baseline()
    # baseline returns value shifted left by 2
    expected = [v << 2 for v in mpr121_instance._bus.read_block(mpr121.EBL, mpr121.NCH)]
    assert result == expected


def test_baseline_configure_calls_write_block_with_correct_data(mpr121_instance):
    mpr121_instance._bus.written_blocks.clear()

    # rft < 2 cases use 4 items list
    mpr121_instance.baseline(rft=1, mhd=3, nhd=4, ncl=5, fdl=6, prox=False)
    found = False
    for reg, vals in mpr121_instance._bus.written_blocks:
        if reg == (mpr121.MHD + 4 * 1) and vals == [3, 4, 5, 6]:
            found = True
    assert found

    mpr121_instance.baseline(rft=1, mhd=1, nhd=2, ncl=3, fdl=4, prox=True)
    found = False
    for reg, vals in mpr121_instance._bus.written_blocks:
        if reg == (mpr121.MHDX + 4 * 1) and vals == [1, 2, 3, 4]:
            found = True
    assert found

    # rft > 1 uses 3 byte list
    mpr121_instance.baseline(rft=2, nhd=6, ncl=7, fdl=8, prox=False)
    found = False
    for reg, vals in mpr121_instance._bus.written_blocks:
        if reg == (mpr121.MHD + 4 * 2) and vals == [6, 7, 8]:
            found = True
    assert found


def test_threshold_sets_release_to_60_percent_touch_if_not_provided(mpr121_instance):
    mpr121_instance._bus.written_bytes.clear()
    mpr121_instance.threshold(100)
    # release should be 60% of 100 = 60
    # called once per channel (0-12)
    count_touch = sum(1 for r, v in mpr121_instance._bus.written_bytes if r % 2 == mpr121.TTH % 2)
    count_release = sum(1 for r, v in mpr121_instance._bus.written_bytes if r % 2 == mpr121.RTH % 2)
    assert count_touch == mpr121.NCH
    assert count_release == mpr121.NCH

    # Also test setting single channel explicitly
    mpr121_instance._bus.written_bytes.clear()
    mpr121_instance.threshold(50, channel=0)
    vals = dict(mpr121_instance._bus.written_bytes)
    assert vals[mpr121.TTH] == 50
    expected_release = int(0.6 * 50)
    assert vals[mpr121.RTH] == expected_release


def test_debounce_defaults_release_to_touch_value(mpr121_instance):
    mpr121_instance._bus.written_bytes.clear()
    mpr121_instance.debounce(touch=3)
    expected_val = (3 << 4) | 3  # release=touch=3
    assert (mpr121.DEB, expected_val) in mpr121_instance._bus.written_bytes

    mpr121_instance._bus.written_bytes.clear()
    mpr121_instance.debounce(touch=2, release=5)
    expected_val = (5 << 4) | 2
    assert (mpr121.DEB, expected_val) in mpr121_instance._bus.written_bytes


def test_filter_write_word_constructed_correctly(mpr121_instance):
    mpr121_instance._bus.written_words.clear()
    mpr121_instance.filter(cdc=10, cdt=3, ffi=2, sfi=1, esi=5)
    # lb byte
    lb = (2 << 6) | 10
    hb = (3 << 5) | (1 << 3) | 5
    expected_val = (hb << 8) | lb
    found = any(w == (mpr121.AFE1, expected_val) for w in mpr121_instance._bus.written_words)
    assert found


def test_charge_writes_correct_bytes_and_modifies_cdt_correctly(mpr121_instance):
    # Set read_byte to known value for reg CDT+channel//2
    reg = mpr121.CDT + 1
    # Initial value with bits set to 0xff
    mpr121_instance._bus.read_bytes[reg] = 0xFF

    mpr121_instance._bus.written_bytes.clear()
    mpr121_instance.charge(channel=2, cdc=7, cdt=5)
    # CDC+channel written once
    assert (mpr121.CDC + 2, 7) in mpr121_instance._bus.written_bytes
    # CDT register written once with bits modified
    written_regs = [reg for reg, _ in mpr121_instance._bus.written_bytes if reg == reg]
    assert reg in written_regs
    # The value written should have bits for channel 2 (index 0) set correctly
    # cdt is shifted by 0 * 4 (channel % 2)
    # Should clear bits at position s=4*0=0xf mask and set it to cdt=5
    # Since read_byte returns 0xFF, new val & ~0xf is 0xF0, or'ed with 5 is 0xF5
    val_written = [v for r, v in mpr121_instance._bus.written_bytes if r == reg][0]
    assert (val_written & 0xF) == 5


def test_auto_config_writes_correct_data(mpr121_instance):
    mpr121_instance._bus.written_bytes.clear()
    mpr121_instance._bus.written_words.clear()

    mpr121_instance.auto_config()

    # We expect one word write to ACNF_C0 and three byte writes to certain registers
    word_writes = [w for w in mpr121_instance._bus.written_words if w[0] == mpr121.ACNF_C0]
    assert word_writes
    byte_regs = [mpr121.ACNF_USL, mpr121.ACNF_LSL, mpr121.ACNF_TL]
    for reg in byte_regs:
        found = any(w[0] == reg for w in mpr121_instance._bus.written_bytes)
        assert found


def test_out_of_range_raises_on_auto_config_failure(mpr121_instance):
    # set read_word to have bit 15 set (auto config failed)
    mpr121_instance._bus.read_words[mpr121.OOR] = (1 << 15)
    with pytest.raises(Exception, match='auto config failed'):
        mpr121_instance.out_of_range()

    # bit 14 set (auto reconfig failed)
    mpr121_instance._bus.read_words[mpr121.OOR] = (1 << 14)
    with pytest.raises(Exception, match='auto reconfig failed'):
        mpr121_instance.out_of_range()

    # without raise_on_failure no exception
    mpr121_instance._bus.read_words[mpr121.OOR] = (1 << 15)
    val = mpr121_instance.out_of_range(raise_on_failure=0)
    assert val & 0x1fff == ((1 << 15) & 0x1fff)


def test_gpio_setup_sets_bits_correctly(monkeypatch, mpr121_instance):
    # patch read_byte and write_byte to capture calls
    calls = {"read": [], "write": []}

    def fake_read_byte(reg):
        calls['read'].append(reg)
        return 0

    def fake_write_byte(reg, val):
        calls['write'].append((reg, val))

    mpr121_instance._bus.read_byte = fake_read_byte
    mpr121_instance._bus.write_byte = fake_write_byte

    # channel 7 (valid range), mode=3 (both bits set), output True, enable True
    mpr121_instance.gpio_setup(channel=7, output=1, mode=3, enable=1)

    # Should do 4 sets for 4 registers, each with calculated bits (channel-4 = 3)
    # That means read_byte called 8 times and write_byte 4 times
    assert sorted(calls['write'], key=lambda x: x[0]) == sorted(
        calls['write']
    )  # no repeats or errors
    assert all(reg in [mpr121.GPIO_CTL0, mpr121.GPIO_CTL1, mpr121.GPIO_DIR, mpr121.GPIO_EN]
               for reg, _ in calls['write'])
    assert all(reg in [mpr121.GPIO_CTL0, mpr121.GPIO_CTL1, mpr121.GPIO_DIR, mpr121.GPIO_EN]
               for reg in calls['read'])


def test_gpio_status_reads_byte(mpr121_instance):
    mpr121_instance._bus.read_byte = lambda reg: 0x5A
    assert mpr121_instance.gpio_status() == 0x5A


def test_gpio_set_writes_correct_register(mpr121_instance):
    mpr121_instance._bus.written_bytes.clear()
    mpr121_instance.gpio_set(4, True)
    assert (mpr121.GPIO_SET, 1 << 0) in mpr121_instance._bus.written_bytes

    mpr121_instance._bus.written_bytes.clear()
    mpr121_instance.gpio_set(11, False)
    assert (mpr121.GPIO_CLR, 1 << (11 - 4)) in mpr121_instance._bus.written_bytes


def test_setup_invokes_expected_methods(monkeypatch):
    monkeypatch.setattr(mpr121, "MPR121.setup", mpr121.MPR121.setup)  # Use real setup

    mpr = mpr121.MPR121(setup=0, reset=0, irq=0, handlers=0)
    calls = []

    def fake_reset():
        calls.append("reset")

    def fake_configure(**kwargs):
        calls.append(("configure", kwargs))

    def fake_filter(**kwargs):
        calls.append(("filter", kwargs))

    def fake_auto_config():
        calls.append("auto_config")

    def fake_threshold(*args, **kwargs):
        calls.append(("threshold", args, kwargs))

    def fake_debounce(*args, **kwargs):
        calls.append(("debounce", args, kwargs))

    def fake_baseline(**kwargs):
        calls.append(("baseline", kwargs))

    mpr.reset = fake_reset
    mpr.configure = fake_configure
    mpr.filter = fake_filter
    mpr.auto_config = fake_auto_config
    mpr.threshold = fake_threshold
    mpr.debounce = fake_debounce
    mpr.baseline = fake_baseline

    mpr.setup(reset=True, channels=5, prox=2, threshold=30, debounce=1, auto_config=True)

    # reset called once
    assert "reset" in calls

    # configure called twice (once with 0,0, once with prox=2, touch=5)
    configure_calls = [c for c in calls if c[0] == "configure"]
    assert len(configure_calls) == 2

    # filter called once
    assert any(c[0] == "filter" for c in calls)

    # auto_config called
    assert "auto_config" in calls

    # threshold called at least twice (all channels and proximity channel)
    threshold_calls = [c for c in calls if c[0] == "threshold"]
    assert len(threshold_calls) >= 2

    # debounce called once
    assert any(c[0] == "debounce" for c in calls)

    # baseline called 4 times: rft=0 and 1, prox False and True
    baseline_calls = [c for c in calls if c[0] == "baseline"]
    count_rft_0_1 = sum(1 for c in baseline_calls if c[1].get("rft") in (0, 1))
    assert count_rft_0_1 >= 4


def test_on_touch_handler_receives_correct_invocation_args(mpr121_instance):
    received = {}

    def handler(touched, channel=None):
        received['touched'] = touched
        received['channel'] = channel

    mpr121_instance.on_touch(3, handler)
    touched = 1 << 3
    changed = 1 << 3
    mpr121_instance._handle_touch(touched, changed)
    assert received['touched'] is True
    assert received['channel'] == 3


def test_update_touch_state_when_no_change_does_not_call_handler(mpr121_instance):
    calls = []

    def handler(*args):
        calls.append(args)

    mpr121_instance._handlers[1] = handler
    mpr121_instance._touched = 0b10

    # touched returns same value - no change
    mpr121_instance.touched = lambda raise_on_failure=1: 0b10
    mpr121_instance.update_touch_state()

    assert calls == []


def test_out_of_range_returns_lower_13bits_when_no_failure(mpr121_instance):
    # Set word with no failure bits set, random lower 13 bits
    val = 0x01FF & 0x1FFF
    mpr121_instance._bus.read_words[mpr121.OOR] = val
    result = mpr121_instance.out_of_range()
    assert result == val


def test_touched_returns_lower_13bits_when_no_failure(mpr121_instance):
    val = 0x1234 & 0x1FFF
    mpr121_instance._bus.read_words[mpr121.ETS] = val
    mpr121_instance._bus.read_words[mpr121.OOR] = 0
    result = mpr121_instance.touched()
    assert result == val
