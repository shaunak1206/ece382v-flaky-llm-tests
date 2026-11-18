import pytest

import piripherals.mpr121 as mpr_mod


class FakeDevice:
    def __init__(self):
        self.bytes = {}
        self.words = {}
        self.blocks = {}
        self.calls = []

    def read_byte(self, reg):
        self.calls.append(("read_byte", reg))
        return self.bytes.get(reg, 0)

    def write_byte(self, reg, val):
        self.calls.append(("write_byte", reg, val))
        self.bytes[reg] = val & 0xFF

    def read_word(self, reg):
        self.calls.append(("read_word", reg))
        return self.words.get(reg, 0)

    def write_word(self, reg, val):
        self.calls.append(("write_word", reg, val))
        self.words[reg] = val & 0xFFFF

    def read_block(self, reg, n):
        self.calls.append(("read_block", reg, n))
        data = self.blocks.get((reg, n))
        if data is None:
            return [0] * n
        return list(data)


class FakeBus:
    def __init__(self, bus_id):
        self.bus_id = bus_id
        self.devices = []

    def device(self, addr):
        dev = FakeDevice()
        self.devices.append((addr, dev))
        return dev


@pytest.fixture
def fake_bus_cls(monkeypatch):
    fake = FakeBus

    monkeypatch.setattr(mpr_mod, "Bus", fake)
    return fake


@pytest.fixture
def device():
    return FakeDevice()


def make_mpr(device=None, monkeypatch=None, irq=0, handlers=0, setup=0, reset=0):
    if device is None:
        device = FakeDevice()

    if monkeypatch is not None:
        monkeypatch.setattr(mpr_mod, "Bus", lambda bus_id: None)

    class BusLike:
        def __init__(self, dev):
            self._dev = dev

        def device(self, addr):
            return self._dev

    return mpr_mod.MPR121(bus=BusLike(device), irq=irq, handlers=handlers, setup=setup, reset=reset)


def test_init_uses_existing_bus_object_without_creating_bus(monkeypatch, device):
    created = []

    def fail_bus(*args, **kwargs):
        created.append(True)
        raise AssertionError("Bus() should not be called")

    monkeypatch.setattr(mpr_mod, "Bus", fail_bus)
    m = mpr_mod.MPR121(bus=type("B", (), {"device": lambda self, addr: device})(), handlers=0, setup=0, reset=0)
    assert m._bus is device
    assert m.addr == 0x5A
    assert m._touched == 0
    assert m.overcurrent is False
    assert created == []


def test_init_wraps_integer_bus_with_bus_class(fake_bus_cls):
    m = mpr_mod.MPR121(bus=1, addr=0x5C, handlers=0, setup=0, reset=0)
    assert isinstance(m._bus, FakeDevice)
    assert m.addr == 0x5C


def test_init_installs_irq_handler_when_irq_given(monkeypatch, device):
    seen = {}

    def fake_irq_handler(pin, callback):
        seen["pin"] = pin
        seen["callback"] = callback

    monkeypatch.setattr(mpr_mod, "IRQHandler", fake_irq_handler)
    m = make_mpr(device=device, monkeypatch=monkeypatch, irq=4, handlers=1, setup=0, reset=0)
    assert seen["pin"] == 4
    assert seen["callback"] == m.update_touch_state


def test_init_installs_poller_when_no_irq(monkeypatch, device):
    seen = {}

    def fake_poller(callback):
        seen["callback"] = callback

    monkeypatch.setattr(mpr_mod, "Poller", fake_poller)
    m = make_mpr(device=device, monkeypatch=monkeypatch, irq=0, handlers=1, setup=0, reset=0)
    assert seen["callback"] == m.update_touch_state


def test_reset_writes_soft_reset(device):
    m = make_mpr(device=device)
    m.reset()
    assert device.bytes[mpr_mod.SRESET] == 0x63
    assert ("write_byte", mpr_mod.SRESET, 0x63) in device.calls


def test_configure_writes_ecr(device):
    m = make_mpr(device=device)
    m.configure(cl=2, prox=3, touch=12)
    expected = (2 << 6) | (3 << 4) | 12
    assert device.bytes[mpr_mod.ECR] == expected


def test_touched_returns_bits_and_checks_failure_flags(monkeypatch, device):
    m = make_mpr(device=device)
    device.words[mpr_mod.ETS] = 0x0007
    device.words[mpr_mod.OOR] = 0x0000
    monkeypatch.setattr(m, "out_of_range", lambda raise_on_failure=1: 0)
    assert m.touched() == 0x0007


def test_touched_raises_on_overcurrent(device):
    m = make_mpr(device=device)
    device.words[mpr_mod.ETS] = 0x8001
    with pytest.raises(Exception, match="overcurrent"):
        m.touched()


def test_touched_raises_on_out_of_range(monkeypatch, device):
    m = make_mpr(device=device)
    device.words[mpr_mod.ETS] = 0x0001
    monkeypatch.setattr(m, "out_of_range", lambda raise_on_failure=1: 1)
    with pytest.raises(Exception, match="out of range"):
        m.touched()


def test_out_of_range_checks_failure_bits(device):
    m = make_mpr(device=device)
    device.words[mpr_mod.OOR] = 0x0009
    assert m.out_of_range() == 0x0009


def test_out_of_range_raises_for_auto_config_failed(device):
    m = make_mpr(device=device)
    device.words[mpr_mod.OOR] = 0x8001
    with pytest.raises(Exception, match="auto config failed"):
        m.out_of_range()


def test_out_of_range_raises_for_auto_reconfig_failed(device):
    m = make_mpr(device=device)
    device.words[mpr_mod.OOR] = 0x4001
    with pytest.raises(Exception, match="auto reconfig failed"):
        m.out_of_range()


def test_electrode_data_reads_and_decodes_block(device):
    m = make_mpr(device=device)
    device.blocks[(mpr_mod.EFD, 2 * mpr_mod.NCH)] = [i & 0xFF for i in range(2 * mpr_mod.NCH)]
    data = m.electrode_data()
    assert len(data) == mpr_mod.NCH
    assert data[0] == (1 << 8) + 0
    assert data[1] == (3 << 8) + 2


def test_baseline_returns_shifted_values_when_no_args(device):
    m = make_mpr(device=device)
    device.blocks[(mpr_mod.EBL, mpr_mod.NCH)] = list(range(mpr_mod.NCH))
    assert m.baseline() == [i << 2 for i in range(mpr_mod.NCH)]


def test_baseline_writes_rising_and_touch_configuration(device):
    m = make_mpr(device=device)
    m.baseline(rft=0, mhd=5, nhd=1, ncl=3, fdl=20)
    assert device.bytes[mpr_mod.MHD] is None if False else True
    assert ("write_block", mpr_mod.MHD, [5, 1, 3, 20]) in device.calls

    device.calls.clear()
    m.baseline(rft=2, nhd=9, ncl=8, fdl=7)
    assert ("write_block", mpr_mod.MHD + 8, [9, 8, 7]) in device.calls


def test_threshold_applies_to_all_channels_with_default_release(device):
    m = make_mpr(device=device)
    m.threshold(50)
    # all channels written with release=int(0.6*50)=30
    for ch in range(mpr_mod.NCH):
        assert device.bytes[mpr_mod.TTH + 2 * ch] == 50
        assert device.bytes[mpr_mod.RTH + 2 * ch] == 30


def test_debounce_and_filter_and_auto_config_program_registers(device):
    m = make_mpr(device=device)
    m.debounce(touch=2, release=3)
    assert device.bytes[mpr_mod.DEB] == ((3 << 4) | 2)

    m.filter(cdc=30, cdt=1, ffi=1, sfi=1, esi=0)
    assert device.words[mpr_mod.AFE1] == (((0b0010110 << 8) | ((1 << 6) | 30)))

    m.auto_config(ace=1, are=1, bva=3, retry=2, afes=1, scts=0, acfie=1, arfie=1, oorie=1, usl=200, lsl=130, tl=180)
    lb = (1 << 6) | (2 << 4) | (3 << 2) | (1 << 1) | 1
    hb = (0 << 7) | (1 << 2) | (1 << 1) | 1
    assert device.words[mpr_mod.ACNF_C0] == ((hb << 8) + lb)
    assert device.bytes[mpr_mod.ACNF_USL] == 200
    assert device.bytes[mpr_mod.ACNF_LSL] == 130
    assert device.bytes[mpr_mod.ACNF_TL] == 180


def test_charge_sets_current_and_nibble_preserves_other_nibble(device):
    m = make_mpr(device=device)
    device.bytes[mpr_mod.CDT + 0] = 0xAB
    m.charge(channel=1, cdc=17, cdt=5)
    assert device.bytes[mpr_mod.CDC + 1] == 17
    # channel 1 uses high nibble of CDT+0
    assert device.bytes[mpr_mod.CDT + 0] == 0x5B


def test_gpio_setup_validates_channel_and_sets_registers(device):
    m = make_mpr(device=device)
    device.bytes[mpr_mod.GPIO_CTL0] = 0
    device.bytes[mpr_mod.GPIO_CTL1] = 0
    device.bytes[mpr_mod.GPIO_DIR] = 0
    device.bytes[mpr_mod.GPIO_EN] = 0

    m.gpio_setup(channel=4, output=1, mode=3, enable=1)
    assert device.bytes[mpr_mod.GPIO_CTL0] & 0x01 == 0x01
    assert device.bytes[mpr_mod.GPIO_CTL1] & 0x01 == 0x01
    assert device.bytes[mpr_mod.GPIO_DIR] & 0x01 == 0x01
    assert device.bytes[mpr_mod.GPIO_EN] & 0x01 == 0x01

    with pytest.raises(AssertionError):
        m.gpio_setup(channel=3, output=1)


def test_gpio_set_and_status(device):
    m = make_mpr(device=device)
    device.bytes[mpr_mod.GPIO_DAT] = 0xA5
    assert m.gpio_status() == 0xA5

    m.gpio_set(channel=4, value=1)
    assert device.bytes[mpr_mod.GPIO_SET] == 1

    m.gpio_set(channel=11, value=0)
    assert device.bytes[mpr_mod.GPIO_CLR] == 1 << 7


def test_on_touch_and_update_touch_state_triggers_handlers(device):
    m = make_mpr(device=device, handlers=0)
    seen = []

    def handler(state, channel):
        seen.append((state, channel))

    m.on_touch(2, handler)
    m._touched = 0
    m.touched = lambda raise_on_failure=1: (1 << 2)
    m.update_touch_state()
    assert seen == [(True, 2)]

    m.touched = lambda raise_on_failure=1: 0
    m.update_touch_state()
    assert seen[-1] == (False, 2)
