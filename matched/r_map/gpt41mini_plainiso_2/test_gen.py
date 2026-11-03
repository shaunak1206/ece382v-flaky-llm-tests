import pytest
import types
import r_map
from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap
from r_map.ValidationError import ValidationError


# Helpers for mocks

def dummy_read_func(address):
    # Return a fixed pattern depending on address for read tests
    return 0xA5A5A5A5

def dummy_write_func(address, value):
    # Save written values for test inspection
    dummy_write_func.written = (address, value)
dummy_write_func.written = None


def dummy_block_read_func(address, size):
    # Return a byte array of incrementing values
    return bytes((i & 0xFF) for i in range(size))

def dummy_block_write_func(address, data):
    dummy_block_write_func.written = (address, bytes(data))
dummy_block_write_func.written = None


@pytest.fixture
def setup_bitfield():
    bf = BitField(name="bf1", width=8, reset_val=0x12, access="RW")
    return bf

@pytest.fixture
def setup_bitfieldref(setup_bitfield):
    bfr = BitFieldRef(name="bfr1", reg_offset=0, slice_width=8, field_offset=0)
    bfr._add(setup_bitfield)
    return bfr

@pytest.fixture
def setup_register(setup_bitfieldref):
    reg = Register(name="reg1", width=32, address=0x1000)
    reg._reg_read_func = dummy_read_func
    reg._reg_write_func = dummy_write_func
    reg._children = {setup_bitfieldref.name: setup_bitfieldref}  # emulate child bitfieldrefs
    setup_bitfieldref.parent = reg
    return reg

@pytest.fixture
def setup_registermap(setup_register):
    rm = RegisterMap(name="rm1")
    rm._block_read_func = dummy_block_read_func
    rm._block_write_func = dummy_block_write_func
    rm._children = {setup_register.name: setup_register}
    setup_register.parent = rm
    return rm


def test_bitfield_init_and_str(setup_bitfield):
    bf = setup_bitfield
    assert bf.width == 8
    assert bf.reset_val == 0x12
    assert bf.mask == 0xFF
    s = str(bf)
    assert "width: 8" in s
    assert "reset_val: 0x12" in s
    assert "value: 0x12" in s


def test_bitfield_value_setter_and_getter(setup_bitfield):
    bf = setup_bitfield
    bf.value = 0xAA
    assert bf.value == 0xAA & bf.mask
    # Setting as string enumeration that does not exist raises ValueError
    with pytest.raises(ValueError):
        bf.value = "nonexistent_enum"
    # Setting as integer works
    bf.value = 0x55
    assert bf.value == 0x55 & bf.mask
    # Setting as r_map.Enumeration instance uses its .value (mock one)
    class DummyEnum:
        value = 0x33
    bf.value = DummyEnum()
    assert bf.value == 0x33

def test_bitfield_reset(setup_bitfield):
    bf = setup_bitfield
    bf.value = 0xFF
    bf.reset()
    assert bf.value == bf.reset_val

def test_bitfield_validate_duplicate_enum(monkeypatch):
    bf = BitField(name="bfdup", width=2)
    # Add child Enumerations with same values to force validation warning
    enum1 = types.SimpleNamespace(name="e1", value=1)
    enum2 = types.SimpleNamespace(name="e2", value=1)
    bf._children = {"e1": enum1, "e2": enum2}
    errors = list(bf.validate())
    assert any("have the same value" in str(e) for e in errors)

def test_bitfieldref_init_and_properties(setup_bitfield):
    bfr = BitFieldRef(name="bfr", reg_offset=3, slice_width=5, field_offset=2)
    bfr._add(setup_bitfield)
    assert bfr.reg_offset == 3
    assert bfr.slice_width == 5
    assert bfr.field_offset == 2
    assert bfr.mask == (1 << 5) - 1
    assert bfr.bf is setup_bitfield
    # Reset_val returns the masked shifted reset val
    expected_reset_val = ((setup_bitfield.reset_val >> 2) & bfr.mask) << bfr.reg_offset
    assert bfr.reset_val == expected_reset_val

def test_bitfieldref_value_getter_setter(setup_bitfield):
    bfr = BitFieldRef(name="bfr", reg_offset=1, slice_width=3, field_offset=0)
    bfr._add(setup_bitfield)
    # Set bitfield value directly and check bfr.value
    setup_bitfield.value = 0b00001111
    # Expected value should be slice from bitfield value shifted to reg offset
    expected = ((setup_bitfield.value >> 0) & ((1 << 3) - 1)) << 1
    assert bfr.value == expected
    # Set bfr.value and check bitfield value is updated properly
    bfr.value = expected
    mask = ((1 << 3) - 1) << 0
    expected_bf_val = (setup_bitfield.value & ~mask) | ((expected >> 1) << 0)
    assert setup_bitfield.value == expected_bf_val

def test_bitfieldref_invalid_slice_width():
    with pytest.raises(ValueError):
        BitFieldRef(name="bfr_invalid", reg_offset=0, slice_width=-1, field_offset=0)
    with pytest.raises(ValueError):
        BitFieldRef(name="bfr_invalid", reg_offset=0, slice_width=1, field_offset=-5)

def test_bitfieldref_validate_errors():
    bfr = BitFieldRef(name="bfr", reg_offset=0, slice_width=10, field_offset=0)
    bfr._bf = None
    errors = list(bfr.validate())
    assert any("No child bitfield present" in str(e) for e in errors)
    # Add a bitfield with width smaller than slice_width to test error
    bf = BitField(name="bf", width=5)
    bfr._add(bf)
    bfr.slice_width = 10
    errors = list(bfr.validate())
    assert any("Slice width is larger than bitfield's width" in str(e) for e in errors)

def test_register_value_property_and_setter(setup_register):
    reg = setup_register
    bfr = next(iter(reg._children.values()))
    # Setup bitfield with value 0x12, check reg.value aggregates it properly
    bfr._bf.value = 0x12
    assert reg.value == 0x12
    # Set reg.value should set the child bitfieldrefs accordingly
    reg.value = 0x55
    assert bfr.value == (0x55 & bfr.mask) << bfr.reg_offset

def test_register_access_property(setup_register):
    reg = setup_register
    bfr = next(iter(reg._children.values()))
    bfr._bf.access = "RW"
    assert "RW" in reg.access

def test_register_reset(setup_register):
    reg = setup_register
    bfr = next(iter(reg._children.values()))
    bfr.value = 0xFF
    reg.reset()
    assert bfr.value == bfr._bf.reset_val << bfr.reg_offset

def test_register_str_method(setup_register):
    reg = setup_register
    # Just verify str returns something containing value string
    s = str(reg)
    assert "value" in s

def test_register_validate_errors():
    reg = Register(name="reg", address=0x2000)
    # No bitfieldrefs added yet, triggers "No bitfieldrefs present"
    errors = list(reg.validate())
    assert any("No bitfieldrefs present" in str(e) for e in errors)
    # Add non BitFieldRef child to test error
    class DummyChild:
        pass
    reg._children = {"dummy": DummyChild()}
    errors = list(reg.validate())
    assert any("is not of type BitFieldRef" in str(e) for e in errors)
    # Add two overlapping BitFieldRefs (simulate)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0, slice_width=4, field_offset=0)
    bf = BitField(name="bf", width=8)
    bfr1._add(bf)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=2, slice_width=4, field_offset=0)
    bfr2._add(bf)
    reg._children = {bfr1.name: bfr1, bfr2.name: bfr2}
    errors = list(reg.validate())
    assert any("overlaps" in str(e) for e in errors)

def test_register_read_and_write(setup_register):
    reg = setup_register
    val = reg.read()
    # Dummy read func returns 0xA5A5A5A5 - reg.value should be bitfield-reflected version of this value
    # But since only bitfield refs apply, value should update children accordingly
    # Just verify returned val is same as dummy read func returns
    assert val == 0xA5A5A5A5
    # Write with no arg writes current value, trigger dummy_write_func, check recorded write
    dummy_write_func.written = None
    reg.write()
    assert dummy_write_func.written is not None
    # Write with explicit value sets value and writes
    dummy_write_func.written = None
    reg.write(0x1234)
    assert dummy_write_func.written[1] == 0x1234

def test_register_iteration_order():
    reg = Register(name="reg_order", width=32)
    bf1 = BitFieldRef(name="bf1", reg_offset=4, slice_width=4, field_offset=0)
    bf2 = BitFieldRef(name="bf2", reg_offset=0, slice_width=4, field_offset=0)
    bf1._add(BitField(name="test", width=4))
    bf2._add(BitField(name="test2", width=4))
    reg._children = {bf1.name: bf1, bf2.name: bf2}
    reg._reg_write_func = dummy_write_func
    reg._reg_read_func = dummy_read_func
    # Iter should be sorted by reg_offset
    lst = list(iter(reg))
    assert lst[0] is bf2
    assert lst[1] is bf1

def test_registermap_block_dimensions_and_read_write(setup_registermap):
    rm = setup_registermap
    reg = next(iter(rm._children.values()))
    # _block_dimensions property (should cache on first access)
    start_addr, block_width = rm._block_dimensions
    assert start_addr == reg.address
    assert block_width >= 4
    # read loads register value based on dummy block read func
    data = rm.read()
    assert isinstance(data, bytes)
    # write without data should produce a block of the right size and triggers write func
    dummy_block_write_func.written = None
    rm.write()
    addr, written_data = dummy_block_write_func.written
    assert addr == start_addr
    assert len(written_data) == block_width
    # write with data triggers write func with passed bytes
    dummy_block_write_func.written = None
    dummy_data = bytearray(block_width)
    rm.write(dummy_data)
    assert dummy_block_write_func.written[1] == bytes(dummy_data)

def test_registermap_write_raises_with_no_registers():
    rm = RegisterMap(name="empty_rm")
    # no children, _block_dimensions raises ValueError
    rm._block_read_func = dummy_block_read_func
    rm._block_write_func = dummy_block_write_func
    with pytest.raises(ValueError):
        rm._block_dimensions
    # Add empty _children but no registers
    rm._children = {}
    with pytest.raises(ValueError):
        rm.write()

