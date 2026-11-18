import pytest
import r_map
from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap
from r_map.ValidationError import ValidationError

class DummyIO:
    """Utility class to mock IO functions for tests"""
    def __init__(self):
        self.writes = []
        self.read_data = {}
        self.block_reads = {}
        self.block_writes = []

    def reg_read(self, address):
        return self.read_data.get(address, 0)

    def reg_write(self, address, val):
        self.writes.append((address, val))

    def block_read(self, address, size):
        return self.block_reads.get((address, size), bytes([0]*size))

    def block_write(self, address, data):
        self.block_writes.append((address, bytes(data))) 

@pytest.fixture
def bf():
    return BitField(name="bf_test", width=4, reset_val=0x5, access="RW")

@pytest.fixture
def bf_enum():
    bf = BitField(name="bf_enum", width=4)
    # add dummy enumerations (simulate enumeration children)
    class Enum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return f"Enum({self.name}={self.value})"
    e1 = Enum("E0", 0)
    e2 = Enum("E1", 1)
    e3 = Enum("E2", 1)  # duplicate value to trigger validation error
    bf._children = {e.name: e for e in (e1,e2,e3)}
    return bf

@pytest.fixture
def bf_ref(bf):
    bf_ref = BitFieldRef(name="bf_ref_test", reg_offset=1, slice_width=3, field_offset=1)
    bf_ref._children = {}
    bf_ref._bf = bf
    # establish parent-child relation manually for test purposes
    bf._references = {bf_ref}
    return bf_ref

@pytest.fixture
def reg(bf_ref):
    reg = Register(name="reg_test", width=8, address=0x10)
    reg._children = {bf_ref.name: bf_ref}
    reg._reg_read_func = lambda addr: 0xAA
    reg._reg_write_func = lambda addr, val: None
    bf_ref.parent = reg
    return reg

@pytest.fixture
def regmap(reg):
    regmap = RegisterMap(name="map_test")
    regmap._children = {reg.name: reg}
    regmap._block_read_func = lambda addr, size: (0xAA.to_bytes(size, 'little'))
    regmap._block_write_func = lambda addr, data: None
    return regmap

def test_bitfield_reset_and_value(bf):
    bf.value = 0b1010
    bf.reset()
    assert bf.value == bf.reset_val
    assert bf.value == (bf.reset_val & bf.mask)

def test_bitfield_value_set_with_int_masked(bf):
    bf.value = 0xFF  # wider than mask
    assert bf.value == (0xFF & bf.mask)

def test_bitfield_value_set_with_str_enum(bf_enum):
    # setting with matching name sets correct value
    bf_enum.value = "E0"
    assert bf_enum.value == 0
    bf_enum.value = "E1"
    assert bf_enum.value == 1
    with pytest.raises(ValueError):
        bf_enum.value = "Nonexistent"

def test_bitfield_value_set_with_enum_object(bf_enum):
    enum_obj = next(iter(bf_enum._children.values()))
    bf_enum.value = enum_obj
    assert bf_enum.value == enum_obj.value

def test_bitfield_annotation_returns_name_or_hex(bf_enum):
    # value matches E0
    bf_enum.value = 0
    assert bf_enum.annotation == "E0"
    # value not matching any enumeration
    bf_enum.value = 99
    assert bf_enum.annotation == hex(99)

def test_bitfield_validate_duplicate_enum_values_yields_error(bf_enum):
    errors = list(bf_enum.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    # specifically, duplicate values error should appear
    found_dup_error = any("same value" in str(e) for e in errors)
    assert found_dup_error

def test_bitfieldref_init_and_properties(bf):
    bf_ref = BitFieldRef(name="ref1", reg_offset=2, slice_width=3, field_offset=1)
    bf_ref._children = {}
    bf_ref._bf = bf
    bf._references = {bf_ref}
    assert bf_ref.reg_offset == 2
    assert bf_ref.slice_width == 3
    assert bf_ref.field_offset == 1
    assert bf_ref.mask == (1 << 3) -1
    # Reset and value properties shifted correctly
    reset_val_calc = ((bf.reset_val >> 1) & bf_ref.mask) << 2
    assert bf_ref.reset_val == reset_val_calc

def test_bitfieldref_value_set_get(bf, bf_ref):
    bf.value = 0b1111
    bf_ref.value = 0b1010 << bf_ref.reg_offset
    # The bitfield value should reflect the updated bits masked and shifted properly
    expected_value = ( ((bf.value & ~((bf_ref.mask) << bf_ref.field_offset)) ) | ((( ( (0b1010 << bf_ref.reg_offset) >> bf_ref.reg_offset) & bf_ref.mask) << bf_ref.field_offset)) )
    assert bf.value == expected_value
    # Getting the value from bf_ref is the shifted slice
    got_value = bf_ref.value
    expected_slice_val = ((bf.value >> bf_ref.field_offset) & bf_ref.mask) << bf_ref.reg_offset
    assert got_value == expected_slice_val

def test_bitfieldref_validate_slice_width_error(bf):
    bf_ref = BitFieldRef(name="ref2", reg_offset=0, slice_width=10, field_offset=0)
    bf_ref._children = {}
    bf_ref._bf = bf
    bf._references = {bf_ref}
    errors = list(bf_ref.validate())
    found_error = any(isinstance(e, ValidationError) and "Slice width" in str(e) for e in errors)
    assert found_error

def test_bitfieldref_validate_no_bf_error():
    bf_ref = BitFieldRef(name="ref3", reg_offset=1)
    bf_ref._children = {}
    # _bf not set
    errors = list(bf_ref.validate())
    found_error = any(isinstance(e, ValidationError) and "No child bitfield" in str(e) for e in errors)
    assert found_error

def test_register_value_property_set_get(bf_ref):
    reg = Register(name="register1", width=8)
    reg._children = {bf_ref.name: bf_ref}
    bf_ref._bf = BitField(name="bf1", width=4, reset_val=0)
    bf_ref._bf.value = 0x7
    bf_ref.value = 0xF << bf_ref.reg_offset
    # value property getter
    val = reg.value
    assert isinstance(val, int)
    # value property setter updates bitfield refs values
    reg.value = 0xA
    for c in reg:
        assert isinstance(c.value, int)

def test_register_reset(bf_ref):
    bf_val = BitField(name="bf_reset", width=4, reset_val=0x3)
    bf_ref = BitFieldRef(name="ref_reset", reg_offset=0, slice_width=4, field_offset=0)
    bf_ref._children = {}
    bf_ref._bf = bf_val
    bf_val._references = {bf_ref}
    reg = Register(name="reg_reset", width=8)
    reg._children = {bf_ref.name: bf_ref}
    bf_val.value = 0xA
    reg.reset()
    assert bf_val.value == bf_val.reset_val

def test_register_validate_errors():
    reg = Register(name="reg_val", width=8)
    # no bitfieldrefs -> yields error
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) and "No bitfieldrefs" in str(e) for e in errors)
    # add invalid child type
    reg._children = {'bad_child': Register(name="bad", width=8)}
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) and "not of type BitFieldRef" in str(e) for e in errors)
    # overlapping bitfieldrefs
    bf1 = BitFieldRef(name="bf1", reg_offset=0, slice_width=4, field_offset=0)
    bf2 = BitFieldRef(name="bf2", reg_offset=2, slice_width=4, field_offset=0)
    reg._children = {'bf1': bf1, 'bf2': bf2}
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) and "overlaps" in str(e) for e in errors)

def test_register_read_write_with_dummyIO():
    dio = DummyIO()
    reg = Register(name="reg_io", width=8, address=0x1)
    reg._children = {}
    # add one bitfieldref child and bitfield
    bf = BitField(name="bf", width=8)
    bf_ref = BitFieldRef(name="bf_ref_io", reg_offset=0, slice_width=8, field_offset=0)
    bf_ref._children = {}
    bf_ref._bf = bf
    bf._references = {bf_ref}
    reg._children = {bf_ref.name: bf_ref}
    reg._reg_read_func = dio.reg_read
    reg._reg_write_func = dio.reg_write
    dio.read_data[reg.address] = 0xAB

    val = reg.read()
    assert val == 0xAB
    assert reg.value == 0xAB

    reg.write(0x55)
    # The dummy IO should have recorded a write
    assert dio.writes[-1] == (reg.address, 0x55)

def test_register_write_without_val_writes_current_value():
    dio = DummyIO()
    reg = Register(name="reg_write", width=8, address=0x2)
    bf = BitField(name="bf_write", width=8)
    bf_ref = BitFieldRef(name="bf_ref_write", reg_offset=0, slice_width=8, field_offset=0)
    bf_ref._children = {}
    bf_ref._bf = bf
    bf._references = {bf_ref}
    reg._children = {bf_ref.name: bf_ref}
    reg._reg_write_func = dio.reg_write
    reg.value = 0x12
    reg.write()
    assert dio.writes[-1] == (reg.address, 0x12)

def test_register_iter_sorted_by_reg_offset():
    reg = Register(name="reg_iter", width=16)
    bf1 = BitFieldRef(name="bf1", reg_offset=4, slice_width=4, field_offset=0)
    bf2 = BitFieldRef(name="bf2", reg_offset=0, slice_width=4, field_offset=0)
    reg._children = {"bf1": bf1, "bf2": bf2}
    # Iterating returns sorted by reg_offset
    offsets = [c.reg_offset for c in reg]
    assert offsets == sorted(offsets)

def test_registermap_block_dimensions(regmap, reg):
    start, width = regmap._block_dimensions
    assert start == reg.address
    assert width == (reg.width // 8 or 1)  # block size in bytes

def test_registermap_read_write_cycle(regmap, reg):
    dio = DummyIO()
    addr = reg.address
    size = (reg.width + 7) // 8
    data_bytes = (0x55).to_bytes(size, 'little')
    dio.block_reads[(addr, size)] = data_bytes
    regmap._block_read_func = dio.block_read
    regmap._block_write_func = dio.block_write
    regmap._children = {reg.name: reg}

    data = regmap.read()
    assert data == data_bytes
    # After read, register value should equal value in data_bytes
    assert reg.value == int.from_bytes(data_bytes, 'little')

    # test write with no data triggers composing data from registers
    reg.value = 0xAA
    dio.block_writes = []
    regmap.write(None)
    # After write, block_writes should have last entry
    assert dio.block_writes
    written_addr, written_data = dio.block_writes[-1]
    assert written_addr == addr
    assert isinstance(written_data, (bytes, bytearray))
    assert len(written_data) == size

def test_registermap_write_with_user_data_partial(regmap, reg):
    dio = DummyIO()
    regmap._block_write_func = dio.block_write
    regmap._block_read_func = dio.block_read
    regmap._children = {reg.name: reg}
    addr = reg.address
    size = (reg.width + 7) // 8
    data_in = bytearray([0x11]*size)
    # partial data: smaller than block size to trigger break
    partial_data = data_in[:size-1]
    regmap.write(partial_data)
    # The block_write should be called with partial_data unchanged
    assert dio.block_writes[-1][1][:len(partial_data)] == partial_data

def test_bitfield_write_read_integration(bf, bf_ref):
    # Prepare registers and IO mocks
    dio = DummyIO()
    reg = Register(name="reg_int", width=8, address=0)
    reg._children = {bf_ref.name: bf_ref}
    bf_ref._bf = bf
    bf._references = {bf_ref}
    reg._reg_read_func = dio.reg_read
    reg._reg_write_func = dio.reg_write
    dio.read_data[reg.address] = 0xAA
    bf_ref.parent = reg

    # Use bitfield write (should trigger read-modify-write on parent reg)
    bf.write(0xF)
    # Check that register read and write have been triggered
    # Can't inspect reg._reg_read_func calls easily, but dio.writes should include one
    assert dio.writes, "BitField write should trigger register writes"

    # Use bitfield read (should trigger register read)
    val = bf.read()
    assert val == bf.value

