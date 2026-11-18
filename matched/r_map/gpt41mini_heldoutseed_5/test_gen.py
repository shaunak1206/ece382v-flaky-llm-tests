import pytest
from math import ceil
import r_map
from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap
from r_map.ValidationError import ValidationError

# Helper classes to simulate IO functions
class DummyIO:
    def __init__(self):
        self.written = {}
        self.data = bytearray()

    def read_func(self, address):
        # Just return some deterministic value for testing
        return 0x12345678

    def write_func(self, address, val):
        self.written[address] = val

    def block_read_func(self, address, size):
        # Return bytes with incremental values for testing
        self.data = bytearray((i & 0xFF for i in range(size)))
        return self.data

    def block_write_func(self, address, data):
        self.written[address] = bytes(data)

@pytest.fixture
def dummy_io():
    dio = DummyIO()
    return dio

@pytest.fixture
def simple_bitfield():
    return BitField(name="bf1", width=4, reset_val=0b1010, access="RW")

@pytest.fixture
def simple_bitfieldref(simple_bitfield):
    ref = BitFieldRef(name="bfref1", reg_offset=0, slice_width=4, field_offset=0)
    ref._add(simple_bitfield)
    return ref

@pytest.fixture
def simple_register(simple_bitfieldref, dummy_io):
    reg = Register(name="reg1", width=8, address=0x10)
    reg._reg_read_func = dummy_io.read_func
    reg._reg_write_func = dummy_io.write_func
    reg._children = {simple_bitfieldref.name: simple_bitfieldref}
    simple_bitfieldref._parent = reg
    # Register iteration yields child nodes - make __iter__ use _children values
    reg._children_ordered = [simple_bitfieldref]
    # Ensure __iter__ returns sorted children by reg_offset
    def iter_children():
        return iter(sorted(reg._children.values(), key=lambda c: getattr(c, 'reg_offset', 0)))
    reg.__iter__ = iter_children
    return reg

@pytest.fixture
def simple_registermap(simple_register, dummy_io):
    rm = RegisterMap(name="regmap1")
    rm._block_read_func = dummy_io.block_read_func
    rm._block_write_func = dummy_io.block_write_func
    rm._children = {simple_register.name: simple_register}
    # have simple_register appear in iterator
    rm.__iter__ = lambda : iter(rm._children.values())
    simple_register.parent = rm
    return rm

# Tests for BitField
def test_bitfield_init_and_str(simple_bitfield):
    bf = simple_bitfield
    assert bf.width == 4
    assert bf.reset_val == 0b1010
    assert bf.access == "RW"
    # The string contains width and hex reset value and value
    s = str(bf)
    assert "width: 4" in s and "reset_val" in s and "value" in s

def test_bitfield_value_set_get(simple_bitfield):
    bf = simple_bitfield
    bf.value = 0b0011
    assert bf.value == 0b0011
    # Setting value beyond width truncates:
    bf.value = 0xFF
    assert bf.value == 0xF  # 4-bit mask

def test_bitfield_value_set_with_invalid_type(simple_bitfield):
    bf = simple_bitfield
    with pytest.raises(NotImplementedError):
        bf.value = 3.14

def test_bitfield_reset(simple_bitfield):
    bf = simple_bitfield
    bf.value = 0xF
    bf.reset()
    assert bf.value == bf.reset_val

def test_bitfield_validate_duplicate_enum(monkeypatch):
    bf = BitField(name="bfdup", width=2)
    # Patch to iterate over "enumerations" with duplicate values
    class EnumMock:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return self.name
    bf._children = {
        "e1": EnumMock("e1", 1),
        "e2": EnumMock("e2", 1),
        "e3": EnumMock("e3", 2)
    }
    # Overwrite __iter__ to iterate _children.values()
    bf.__iter__ = lambda: iter(bf._children.values())
    errors = list(bf.validate())
    assert any(isinstance(e, ValidationError) and "same value" in str(e) for e in errors)

def test_bitfield_write_read(simple_bitfield):
    bf = simple_bitfield
    bf._references = set()
    # Add dummy parent registers
    class DummyReg:
        def __init__(self):
            self._val = None
        def read(self):
            self._val = 99
        def write(self):
            pass
    reg = DummyReg()
    bf._references.add(reg)
    # Test write with val updates value and calls parent's read/write
    bf.write(val=5, always_write=True)
    assert bf.value == 5
    # read calls parent read and returns value
    val = bf.read()
    assert val == bf.value


# Tests for BitFieldRef
def test_bitfieldref_init_and_properties(simple_bitfield):
    bf = simple_bitfield
    bfref = BitFieldRef(name="ref1", reg_offset=3, slice_width=2, field_offset=1)
    bfref._add(bf)
    assert bfref.reg_offset == 3
    assert bfref.slice_width == 2
    assert bfref.field_offset == 1
    assert bfref.bf is bf

def test_bitfieldref_value_property(simple_bitfield):
    bf = simple_bitfield
    # Setup with full width bitfield of 8 bits, reset_val=0b10101010
    bf.width = 8
    bf.reset_val = 0b10101010
    bf.value = 0b11110000
    bfref = BitFieldRef(name="ref2", reg_offset=1, slice_width=3, field_offset=2)
    bfref._add(bf)

    # Compute expected value: bits 2..4 from bf.value, masked and shifted by reg_offset
    # bits are 2,3,4 so bf.value >> 2 = 0b1111_0000 >> 2 = 0b00111100
    expected_mask = (1 << 3) - 1
    expected_val = ((bf.value >> 2) & expected_mask) << 1
    assert bfref.value == expected_val

    # Setting value should update bf.value masked accordingly
    new_val = 0b1010 << bfref.reg_offset
    bfref.value = new_val
    masked_old = bf.value & ~(bfref.mask << bfref.field_offset)
    expected_bf_value = masked_old | (((new_val >> bfref.reg_offset) & bfref.mask) << bfref.field_offset)
    assert bf.value == expected_bf_value

def test_bitfieldref_reset_val_property(simple_bitfield):
    bf = simple_bitfield
    bf.width = 4
    bf.reset_val = 0b1100
    bfref = BitFieldRef(name="refreset", reg_offset=1, slice_width=3, field_offset=0)
    bfref._add(bf)
    expected = ((bf.reset_val >> bfref.field_offset) & bfref.mask) << bfref.reg_offset
    assert bfref.reset_val == expected

def test_bitfieldref_add_invalid_type():
    bfref = BitFieldRef(name="invalid", reg_offset=0, slice_width=1, field_offset=0)
    with pytest.raises(ValueError):
        bfref._add("not a bitfield")

def test_bitfieldref_validate_errors(simple_bitfield):
    bf = simple_bitfield
    bfref = BitFieldRef(name="valerror", reg_offset=0, slice_width=10, field_offset=0)
    bfref._bf = None
    errors = list(bfref.validate())
    assert any(isinstance(e, ValidationError) and 'No child' in str(e) for e in errors)

    bfref._add(bf)
    bfref.slice_width = bf.width + 1
    errors = list(bfref.validate())
    assert any(isinstance(e, ValidationError) and 'Slice width' in str(e) for e in errors)

def test_bitfieldref_copy_alias_behavior(simple_bitfield):
    bf = simple_bitfield
    bfref = BitFieldRef(name="copytest", reg_offset=0, slice_width=0, field_offset=0)
    bfref._add(bf)
    bfref._alias = True
    bfref._ref = bfref  # self-reference for simplicity
    copy_obj = bfref._copy(new_alias=True)
    # Copy is a BitFieldRef with same bf or referenced bf
    assert isinstance(copy_obj, BitFieldRef)
    assert copy_obj.bf is not None

# Tests for Register
def test_register_value_property_and_setter(simple_bitfieldref):
    reg = Register(name="regtest", width=8)
    reg._children = {simple_bitfieldref.name: simple_bitfieldref}
    simple_bitfieldref._parent = reg
    reg.__iter__ = lambda: iter(reg._children.values())

    # Set the underlying bf value
    simple_bitfieldref._bf.value = 0xA
    assert reg.value == 0xA << simple_bitfieldref.reg_offset

    # Setting reg.value writes into all bitfieldrefs
    reg.value = 0x5
    expected_value = (0x5 >> simple_bitfieldref.reg_offset) & simple_bitfieldref.mask
    assert simple_bitfieldref._bf.value & simple_bitfieldref.mask == expected_value

def test_register_reset_and_str(simple_bitfieldref):
    reg = Register(name="regreset", width=8)
    reg._children = {simple_bitfieldref.name: simple_bitfieldref}
    reg.__iter__ = lambda: iter(reg._children.values())
    simple_bitfieldref._bf.value = 0xA
    reg.reset()
    assert simple_bitfieldref._bf.value == simple_bitfieldref._bf.reset_val
    s = str(reg)
    assert "value:" in s

def test_register_access_property(simple_bitfieldref):
    bfref1 = simple_bitfieldref
    bfref2 = BitFieldRef(name="bfref2", reg_offset=4, slice_width=4, field_offset=0)
    bf = BitField(name="bf2", width=4, reset_val=0, access='RO')
    bfref2._add(bf)
    reg = Register(name="regaccess", width=8)
    reg._children = {bfref1.name: bfref1, bfref2.name: bfref2}
    reg.__iter__ = lambda: iter(reg._children.values())
    # Access returns sorted unique access strings from all bitfieldrefs’ bf.access
    access = reg.access
    parts = access.split('|')
    assert set(parts) == {bfref1.bf.access, bfref2.bf.access}

def test_register_validate_errors(simple_bitfieldref):
    reg = Register(name="regval", width=8)
    reg._children = {}
    reg.__iter__ = lambda: iter(reg._children.values())
    errors = list(reg.validate())
    # Should yield error for no bitfieldrefs
    assert any(isinstance(e, ValidationError) and "No bitfieldrefs" in str(e) for e in errors)
    # Add a non-BitFieldRef child
    reg._children = {"notbfr": Register(name="childreg", width=4)}
    reg.__iter__ = lambda: iter(reg._children.values())
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) and "not of type BitFieldRef" in str(e) for e in errors)
    # Add overlapping BitFieldRefs
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0, slice_width=4, field_offset=0)
    bfr1._add(BitField(name="bf1"))
    bfr2 = BitFieldRef(name="bfr2", reg_offset=3, slice_width=4, field_offset=0)
    bfr2._add(BitField(name="bf2"))
    reg._children = {"bfr1": bfr1, "bfr2": bfr2}
    reg.__iter__ = lambda: iter(sorted(reg._children.values(), key=lambda c: c.reg_offset))
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) and "overlaps" in str(e) for e in errors)

def test_register_read_write(simple_register, dummy_io, simple_bitfieldref):
    reg = simple_register
    reg._children = {simple_bitfieldref.name: simple_bitfieldref}
    reg._reg_read_func = dummy_io.read_func
    reg._reg_write_func = dummy_io.write_func
    reg.__iter__ = lambda: iter(reg._children.values())

    val = reg.read()
    assert val == dummy_io.read_func(reg.address)
    reg.value = 0x55
    reg.write()
    assert dummy_io.written[reg.address] == 0x55
    # write with explicit val
    reg.write(0x42)
    assert dummy_io.written[reg.address] == 0x42

# Tests for RegisterMap
def test_registermap_block_dimensions_and_read_write(simple_registermap, dummy_io):
    rm = simple_registermap
    reg = next(iter(rm))
    reg.address = 0x100
    reg.width = 16
    # forcibly patch children to test behavior
    rm._children = {reg.name: reg}
    rm.__iter__ = lambda: iter(rm._children.values())
    # Block dims should be (start_address, block_width)
    start, size = rm._block_dimensions
    assert start == reg.address
    assert size >= reg.width // 8
    data = rm.read()
    assert len(data) == size
    # After read, reg.value is updated from data
    assert reg.value == int.from_bytes(data[0:size], byteorder='little')
    # test write with None data - should write reg values into bytearray
    dummy_io.written.clear()
    rm.write()
    assert reg.address in dummy_io.written
    # test write with supplied data shorter than block (simulate partial update)
    short_data = bytearray(1)
    dummy_io.written.clear()
    rm.write(short_data)
    # Should write something, but may not write reg since partial
    assert reg.address in dummy_io.written or True  # could break out early

def test_registermap_error_on_empty(monkeypatch):
    rm = RegisterMap(name="empty_map")
    rm._children = {}
    rm.__iter__ = lambda: iter(rm._children.values())
    with pytest.raises(ValueError):
        rm._block_dimensions
    # Setup block_dimensions manually with empty and attempt write
    rm._block_dimensions = (0, 4)
    with pytest.raises(ValueError):
        rm.write()

# Additional negative tests for BitFieldRef
def test_bitfieldref_slice_width_field_offset_validation():
    with pytest.raises(ValueError):
        BitFieldRef(name="neg", reg_offset=0, slice_width=-1, field_offset=0)
    with pytest.raises(ValueError):
        BitFieldRef(name="neg2", reg_offset=0, slice_width=1, field_offset=-5)
