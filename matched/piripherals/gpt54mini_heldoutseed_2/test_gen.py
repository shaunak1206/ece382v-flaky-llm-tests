import importlib
import pytest

mpr121 = importlib.import_module("piripherals.mpr121")


class FakeDevice:
    def __init__(self):
        self.byte_reads = {}
        self.word_reads = {}
        self.block_reads = {}
        self.write_byte_calls = []
        self.write_word_calls = []
        self.write_block_calls = []

    def read_byte(self, reg):
        return self.byte_reads.get(reg, 0)

    def read_word(self, reg):
        return self.word_reads.get(reg, 0)

    def read_block(self, reg, count):
        return list(self.block_reads.get((reg, count), [0] * count))

    def write_byte(self, reg, value):
        self.write_byte_calls.append((reg, value))

    def write_word(self, reg, value):
        self.write_word_calls.append((reg, value))

    def write_block(self, reg, values):
        self.write_block_calls.append((reg, list(values)))


class FakeBusFactory:
    def __init__(self, device):
        self.device_obj = device
        self.calls = []

    def __call__(self, bus_num):
        self.calls.append(bus_num)
        return self

    def device(self, addr):
        self.calls.append(("device", addr))
        return self.device_obj


@pytest.fixture
def fake_device():
    return FakeDevice()


@pytest.fixture
def mpr(monkeypatch, fake_device):
    bus_factory = FakeBusFactory(fake_device)
    monkeypatch.setattr(mpr121, "Bus", bus_factory)
    monkeypatch.setattr(mpr121, "IRQHandler", lambda *a, **k: None)
    monkeypatch.setattr(mpr121, "Poller", lambda *a, **k: None)
    return mpr121.MPR121(bus=1, addr=0x5A, handlers=0, setup=0, reset=0)


def test_init_uses_bus_factory_and_sets_defaults(monkeypatch, fake_device):
    bus_factory = FakeBusFactory(fake_device)
    monkeypatch.setattr(mpr121, "Bus", bus_factory)
    monkeypatch.setattr(mpr121, "IRQHandler", lambda *a, **k: None)
    monkeypatch.setattr(mpr121, "Poller", lambda *a, **k: None)

    obj = mpr121.MPR121(bus=3, addr=0x5B, handlers=0, setup=0, reset=0)

    assert obj.addr == 0x5B
    assert obj._bus is fake_device
    assert bus_factory.calls == [3, ("device", 0x5B)]
    assert len(obj._handlers) == mpr121.NCH
    assert all(h == mpr121.noop for h in obj._handlers)
    assert obj._touched == 0
    assert obj.overcurrent is False


def test_init_with_existing_bus_object_skips_factory(monkeypatch, fake_device):
    monkeypatch.setattr(mpr121, "IRQHandler", lambda *a, **k: None)
    monkeypatch.setattr(mpr121, "Poller", lambda *a, **k: None)

    class BusLike:
        def __init__(self, device):
            self.device = device

    bus_like = BusLike(fake_device)
    obj = mpr121.MPR121(bus=bus_like, handlers=0, setup=0, reset=0)

    assert obj._bus is fake_device


def test_init_starts_irq_handler_when_irq_given(monkeypatch, fake_device):
    calls = []

    class BusFactory:
        def __call__(self, bus_num):
            return self

        def device(self, addr):
            return fake_device

    def fake_irq(irq, callback):
        calls.append((irq, callback))

    monkeypatch.setattr(mpr121, "Bus", BusFactory())
    monkeypatch.setattr(mpr121, "IRQHandler", fake_irq)
    monkeypatch.setattr(mpr121, "Poller", lambda *a, **k: pytest.fail("Poller should not be used"))

    obj = mpr121.MPR121(bus=1, irq=4, handlers=1, setup=0, reset=0)

    assert calls and calls[0][0] == 4
    assert calls[0][1] == obj.update_touch_state


def test_init_uses_poller_when_irq_disabled(monkeypatch, fake_device):
    calls = []

    class BusFactory:
        def __call__(self, bus_num):
            return self

        def device(self, addr):
            return fake_device

    monkeypatch.setattr(mpr121, "Bus", BusFactory())
    monkeypatch.setattr(mpr121, "IRQHandler", lambda *a, **k: pytest.fail("IRQHandler should not be used"))
    monkeypatch.setattr(mpr121, "Poller", lambda callback: calls.append(callback))

    obj = mpr121.MPR121(bus=1, irq=0, handlers=1, setup=0, reset=0)

    assert calls == [obj.update_touch_state]


def test_reset_writes_soft_reset(mpr, fake_device):
    mpr.reset()
    assert fake_device.write_byte_calls == [(mpr121.SRESET, 0x63)]


def test_configure_writes_ecr_bitfield(mpr, fake_device):
    mpr.configure(cl=2, prox=3, touch=12)
    assert fake_device.write_byte_calls == [(mpr121.ECR, (2 << 6) | (3 << 4) | 12)]


def test_touched_returns_low_bits_and_checks_failures(mpr, fake_device):
    fake_device.word_reads[mpr121.ETS] = 0x8005
    fake_device.word_reads[mpr121.OOR] = 0

    with pytest.raises(Exception, match="overcurrent"):
        mpr.touched()

    assert mpr.touched(raise_on_failure=0) == 0x0005


def test_touched_raises_on_out_of_range(mpr, fake_device):
    fake_device.word_reads[mpr121.ETS] = 0x0007
    fake_device.word_reads[mpr121.OOR] = 0x0001  # any non-zero touch status

    with pytest.raises(Exception, match="out of range"):
        mpr.touched()


def test_electrode_data_unpacks_13_words(mpr, fake_device):
    values = []
    for i in range(mpr121.NCH):
        values.extend([i, i + 1])
    fake_device.block_reads[(mpr121.EFD, 2 * mpr121.NCH)] = values

    assert mpr.electrode_data() == [i * 256 + i for i in range(mpr121.NCH)]


def test_baseline_read_shifts_values_left_two(mpr, fake_device):
    fake_device.block_reads[(mpr121.EBL, mpr121.NCH)] = list(range(mpr121.NCH))

    assert mpr.baseline() == [i << 2 for i in range(mpr121.NCH)]


def test_baseline_write_uses_rft_and_prox_registers(mpr, fake_device):
    mpr.baseline(rft=1, mhd=5, nhd=6, ncl=7, fdl=8, prox=0)
    mpr.baseline(rft=2, nhd=9, ncl=10, fdl=11, prox=1)

    assert fake_device.write_block_calls == [
        (mpr121.MHD + 4, [5, 6, 7, 8]),
        (mpr121.MHDX + 8, [9, 10, 11]),
    ]


def test_threshold_applies_to_all_channels_and_defaults_release(mpr, fake_device):
    mpr.threshold(50)

    assert len(fake_device.write_byte_calls) == mpr121.NCH * 2
    assert fake_device.write_byte_calls[0] == (mpr121.TTH, 50)
    assert fake_device.write_byte_calls[1] == (mpr121.RTH, int(0.6 * 50))
    assert fake_device.write_byte_calls[-2] == (mpr121.TTH + 2 * 12, 50)
    assert fake_device.write_byte_calls[-1] == (mpr121.RTH + 2 * 12, int(0.6 * 50))


def test_debounce_uses_touch_as_release_default(mpr, fake_device):
    mpr.debounce(touch=3)
    assert fake_device.write_byte_calls == [(mpr121.DEB, (3 << 4) | 3)]


def test_filter_writes_combined_word(mpr, fake_device):
    mpr.filter(cdc=30, cdt=2, ffi=1, sfi=3, esi=4)
    lb = (1 << 6) | 30
    hb = (2 << 5) | (3 << 3) | 4
    assert fake_device.write_word_calls == [(mpr121.AFE1, (hb << 8) | lb)]


def test_charge_writes_cdc_and_updates_cdt_bits(mpr, fake_device):
    fake_device.byte_reads[mpr121.CDT + 1] = 0b10110011
    mpr.charge(channel=3, cdc=17, cdt=5)

    assert fake_device.write_byte_calls[0] == (mpr121.CDC + 3, 17)
    assert fake_device.write_byte_calls[1] == (mpr121.CDT + 1, (0b10110011 & ~(0xF << 4)) | (5 << 4))


def test_auto_config_writes_word_and_limits(mpr, fake_device):
    mpr.auto_config(ace=1, are=0, bva=2, retry=3, afes=1, scts=1, acfie=0, arfie=1, oorie=0, usl=201, lsl=131, tl=181)

    lb = (1 << 6) | (3 << 4) | (2 << 2) | (0 << 1) | 1
    hb = (1 << 7) | (0 << 2) | (1 << 1) | 0
    assert fake_device.write_word_calls == [(mpr121.ACNF_C0, (hb << 8) + lb)]
    assert fake_device.write_byte_calls == [
        (mpr121.ACNF_USL, 201),
        (mpr121.ACNF_LSL, 131),
        (mpr121.ACNF_TL, 181),
    ]


def test_out_of_range_returns_low_bits_and_raises_for_failure_bits(mpr, fake_device):
    fake_device.word_reads[mpr121.OOR] = 0x000A
    assert mpr.out_of_range() == 0x000A

    fake_device.word_reads[mpr121.OOR] = 0x8001
    with pytest.raises(Exception, match="auto config failed"):
        mpr.out_of_range()

    fake_device.word_reads[mpr121.OOR] = 0x4001
    with pytest.raises(Exception, match="auto reconfig failed"):
        mpr.out_of_range()


def test_gpio_setup_asserts_channel_range(mpr):
    with pytest.raises(AssertionError):
        mpr.gpio_setup(channel=3, output=1)
    with pytest.raises(AssertionError):
        mpr.gpio_set(channel=12, value=1)


def test_gpio_set_writes_set_or_clear_register(mpr, fake_device):
    mpr.gpio_set(channel=4, value=1)
    mpr.gpio_set(channel=7, value=0)

    assert fake_device.write_byte_calls == [
        (mpr121.GPIO_SET, 1 << 0),
        (mpr121.GPIO_CLR, 1 << 3),
    ]


def test_on_touch_registers_wrapper_and_handle_touch_calls_handlers_with_channel(mpr):
    calls = []
    mpr.on_touch(2, lambda touched, ch: calls.append((touched, ch)))
    mpr._handle_touch(touched=(1 << 2), changed=(1 << 2))
    assert calls == [(True, 2)]


def test_on_touch_handler_fallback_without_channel_arg(mpr):
    calls = []
    mpr.on_touch(1, lambda touched: calls.append(touched))
    mpr._handle_touch(touched=0, changed=(1 << 1))
    assert calls == [False]


def test_update_touch_state_updates_cached_state_and_invokes_handlers(mpr, fake_device):
    calls = []
    mpr.on_touch(0, lambda touched, ch: calls.append((touched, ch)))
    fake_device.word_reads[mpr121.ETS] = 1
    fake_device.word_reads[mpr121.OOR] = 0

    mpr.update_touch_state()

    assert mpr._touched == 1
    assert calls == [(True, 0)]


def test_setup_calls_expected_configuration_sequence(mpr, monkeypatch):
    calls = []

    def rec(name):
        def inner(*args, **kwargs):
            calls.append((name, args, kwargs))
        return inner

    monkeypatch.setattr(mpr, "reset", rec("reset"))
    monkeypatch.setattr(mpr, "configure", rec("configure"))
    monkeypatch.setattr(mpr, "filter", rec("filter"))
    monkeypatch.setattr(mpr, "auto_config", rec("auto_config"))
    monkeypatch.setattr(mpr, "threshold", rec("threshold"))
    monkeypatch.setattr(mpr, "debounce", rec("debounce"))
    monkeypatch.setattr(mpr, "baseline", rec("baseline"))

    mpr.setup(reset=1, channels=6, prox=2, threshold=55, debounce=3, auto_config=1)

    assert calls[0][0] == "reset"
    assert calls[1] == ("configure", (), {"prox": 0, "touch": 0})
    assert calls[2][0] == "filter"
    assert ("auto_config", (), {}) in calls
    assert ("debounce", (3,), {}) in calls
    assert calls[-1] == ("configure", (), {"prox": 2, "touch": 6})
