import pytest
import random
import r_map
from r_map import Register, BitField, BitFieldRef, RegisterMap
from r_map.ValidationError import ValidationError

# Dummy classes to support AddressedNode and Node iteration since original 
# structures are not provided here and Register inherits from AddressedNode.
# This setup assumes r_map.Node's __iter__ yields children and that r_map.AddressedNode behaves like a container of BitFieldRef or registers.

# Helper to set dummy reg_read and reg_write functions to support Register I/O methods
def dummy_reg_read(address):
    # Return fixed value dependent on address (int) but deterministic
    return (address * 0x55AA1234) & 0xFFFFFFFF

def dummy_reg_write(address, val):
    # For testing, just store writes in a dictionary keyed by address
    dummy_reg_write.store[address] = val
dummy_reg_write.store = {}

def dummy_block_read(address, size):
    # Return size bytes with value pattern to test RegisterMap.read()
    # pattern: bytes from 0 to size-1
    return bytes((i % 256 for i in range(size)))

def dummy_block_write(address, data):
    # Store data writes for verification
    dummy_block_write.store[address] = data
dummy_block_write.store = {}

# Basic BitField and BitFieldRef setup helper
def setup_bf_and_ref(name="bf", width=8, reset_val=0x12, access="RW",
                     reg_offset=0, slice_width=0, field_offset=0):
    bf = BitField(name=name, width=width, reset_val=reset_val, access=access)
    ref = BitFieldRef(name=f"{name}_ref", reg_offset=reg_offset, slice_width=slice_width, field_offset=field_offset)
    ref._add(bf)
    return bf, ref


def test_register_init_and_default_value():
    random.seed(0)
    reg = Register(name="test_reg", address=0x10)
    assert reg.width == 32
    # value is aggregate of value of bitfieldrefs, none present -> 0
    assert reg.value == 0  
    # Access on empty reg returns empty string
    assert reg.access == ''

def test_register_value_property_and_setter():
    random.seed(0)
    bf1, ref1 = setup_bf_and_ref(width=4, reset_val=0x3, reg_offset=0)
    bf2, ref2 = setup_bf_and_ref(width=4, reset_val=0x1, reg_offset=4)
    reg = Register(name="reg_val_test")
    reg._children = {
        ref1.name: ref1,
        ref2.name: ref2,
    }
    # Iteration returns refs sorted by .reg_offset
    refs_sorted = list(reg)
    assert refs_sorted == [ref1, ref2]

    # The combined value is bitwise or of bitfieldref values
    expected_value = ref1.value | ref2.value
    assert reg.value == expected_value

    # Set value - updates each ref.value properly and masked
    reg.value = 0xFF
    for r in reg:
        # mask test for each ref
        masked_val = ((0xFF >> r.reg_offset) & ((1 << r.slice_width) - 1 if r.slice_width != 0 else (1 << r.bf.width) - 1)) << r.reg_offset
        assert r.value == masked_val

def test_register_reset():
    random.seed(0)
    bf, ref = setup_bf_and_ref(width=8, reset_val=0x55, reg_offset=0)
    reg = Register()
    reg._children = {ref.name: ref}

    # Change bitfield value and reset
    for r in reg:
        r._bf.value = 0xAA
    reg.reset()
    for r in reg:
        assert r._bf.value == r._bf.reset_val == 0x55

def test_register_access_property():
    random.seed(0)
    # access strings will be joined sorted and unique
    bf1, ref1 = setup_bf_and_ref(access="RW", reg_offset=0)
    bf2, ref2 = setup_bf_and_ref(access="RO", reg_offset=4)
    bf3, ref3 = setup_bf_and_ref(access="RW", reg_offset=8)
    reg = Register()
    reg._children = {
        ref1.name: ref1,
        ref2.name: ref2,
        ref3.name: ref3,
    }
    expected_access_set = set([ref1.bf.access, ref2.bf.access, ref3.bf.access])
    expected_access_str = '|'.join(sorted(expected_access_set))
    assert reg.access == expected_access_str

def test_register_validate_no_bitfieldrefs():
    random.seed(0)
    reg = Register(name="reg_no_refs")
    reg._children = {}
    errors = list(reg.validate())
    assert any("No bitfieldrefs present" in str(e) for e in errors)
    # Also, if child is not BitFieldRef yield error
    reg._children = {"badchild": Register(name="a_register")}
    errors = list(reg.validate())
    assert any("is not of type BitFieldRef" in str(e) for e in errors)

def test_register_validate_bitfieldref_overlap():
    random.seed(0)
    bf1, ref1 = setup_bf_and_ref(name="bf1", reg_offset=0, slice_width=8)
    bf2, ref2 = setup_bf_and_ref(name="bf2", reg_offset=4, slice_width=8)
    reg = Register()
    reg._children = {ref1.name: ref1, ref2.name: ref2}
    errors = list(reg.validate())
    # There should be one overlap error since ref2 starts at 4 while ref1 ends at 8
    assert any("overlaps" in str(e) for e in errors)

def test_register_str_contains_value_hex_format():
    random.seed(0)
    bf, ref = setup_bf_and_ref(reset_val=0x01, reg_offset=0, slice_width=8)
    reg = Register(width=8, name="reg_str_test")
    reg._children = {ref.name: ref}
    reg.value = 0xAB
    s = str(reg)
    assert f"value: 0xab" in s.lower()

def test_register_read_and_write(monkeypatch):
    random.seed(0)
    bf, ref = setup_bf_and_ref(reg_offset=0, width=8)
    reg = Register(name="reg_io", address=0x10)
    reg._children = {ref.name: ref}
    monkeypatch.setattr(reg, "_reg_read_func", dummy_reg_read)
    monkeypatch.setattr(reg, "_reg_write_func", dummy_reg_write)
    # Clear dummy_reg_write storage
    dummy_reg_write.store.clear()

    val = reg.read()
    # The returned value must match dummy_reg_read return for address 0x10
    assert val == dummy_reg_read(0x10)
    # The register's value must be updated accordingly (masked)
    mask = (1 << reg.width) - 1
    assert (reg.value & mask) == (val & mask)

    reg.write(0x1234ABCD)
    # The dummy_reg_write must be called with (address, value)
    assert dummy_reg_write.store[reg.address] == 0x1234ABCD

    # Test write with val=None sets val from register.value and writes it
    dummy_reg_write.store.clear()
    reg.write()
    assert dummy_reg_write.store[reg.address] == reg.value

def test_bitfield_init_and_value_set_get():
    random.seed(0)
    bf = BitField(name="bf_test", width=8, reset_val=0xA5, access="RW")
    assert bf.width == 8
    assert bf.reset_val == 0xA5 & ((1 << 8) -1)
    assert bf.value == bf.reset_val
    bf.value = 0xAA
    assert bf.value == (0xAA & bf.mask)

    # Setting with string that matches enumeration should set value accordingly
    # Setup fake Enumeration child with name matching
    class Enum:
        def __init__(self):
            self.name = "ENUM1"
            self.value = 0x55
    e = Enum()
    bf._children = {e.name: e}
    bf.value = "ENUM1"
    assert bf.value == 0x55
    # Setting string with no match raises ValueError
    with pytest.raises(ValueError):
        bf.value = "NO_MATCH"

    # Setting value to r_map.Enumeration instance sets the .value
    bf.value = e
    assert bf.value == e.value

def test_bitfield_reset_and_str_and_annotation():
    random.seed(0)
    bf = BitField(name="bf_anno_str", width=4, reset_val=0x3, access="RW")
    bf.value = 0x7
    s = str(bf)
    assert "width: 4" in s
    bf.reset()
    assert bf.value == bf.reset_val == 0x3
    # annotation returns enumeration name if matching
    class Enum:
        def __init__(self):
            self.name = "ENUM"
            self.value = 0x3
    e = Enum()
    bf._children = {e.name: e}
    assert bf.annotation == "ENUM"
    bf.value = 0x4  # no matching enumeration
    # annotation falls back to hex string for unmatched value
    assert bf.annotation.startswith("0x")

def test_bitfield_validate_duplicate_enumerations():
    random.seed(0)
    bf = BitField(name="bf_val", width=4)
    class Enum:
        def __init__(self, name, val):
            self.name = name
            self.value = val
        def __str__(self):
            return self.name
    e1 = Enum("E1", 1)
    e2 = Enum("E2", 1)
    e3 = Enum("E3", 2)
    bf._children = {e1.name: e1, e2.name: e2, e3.name: e3}
    errors = list(bf.validate())
    found_dup_error = any("same value" in str(e) and "E1" in str(e) and "E2" in str(e) for e in errors)
    assert found_dup_error

def test_bitfield_write_read(monkeypatch):
    random.seed(0)
    bf = BitField(name="bf_io", width=8)
    bf._references = set()  # Will be added
    ref = BitFieldRef(name="ref_io", reg_offset=0)
    ref._add(bf)
    # Create a Register mock with read/write functions
    class DummyReg(Register):
        def __init__(self):
            super().__init__(width=8, address=0x20)
            self._value = 0
        def read(self):
            self._value = 0x34
            self.value = 0x34
            return self._value
        def write(self, val=None):
            if val is not None:
                self._value = val
            else:
                self._value = self.value
    reg = DummyReg()
    bf._references.add(reg)

    # Set initial value and write with always_write True
    bf.value = 0x12
    bf.write(val=0x12, always_write=True)
    assert reg._value == 0x12

    # Write with same val always_write False: should call write since same_value==True, but always_write False means no write
    bf.value = 0x12
    reg._value = 0x00
    bf.write(val=0x12, always_write=False)
    # Because same_value == True and always_write False, write() should NOT be called so DummyReg._value remains 0
    # But per code it writes if: always_write or not same_value, so no write expected here.
    assert reg._value == 0x00

    # Write with different val, always_write False, expect write-called (value updated)
    bf.value = 0x34
    bf.write(val=0x56, always_write=False)
    assert reg._value == 0x56

    # Read should call parent's read and return current value
    val = bf.read()
    assert val == bf.value == reg._value

def test_bitfieldref_init_invalid_and_properties():
    random.seed(0)
    # invalid slice_width < 0
    with pytest.raises(ValueError):
        BitFieldRef(name="badref", reg_offset=0, slice_width=-1)
    # invalid field_offset < 0
    with pytest.raises(ValueError):
        BitFieldRef(name="badref2", reg_offset=0, field_offset=-1)

    bf = BitField(width=8)
    ref = BitFieldRef(name="ref_test", reg_offset=2, slice_width=4, field_offset=1)
    ref._add(bf)
    # Check reset_val and value related to bitfield
    bf.reset_val = 0b10101010
    bf.value = 0b11001100
    expected_reset = ((bf.reset_val >> ref.field_offset) & ref.mask) << ref.reg_offset
    expected_value = ((bf.value >> ref.field_offset) & ref.mask) << ref.reg_offset
    assert ref.reset_val == expected_reset
    assert ref.value == expected_value

def test_bitfieldref_value_setter_corrects_parent_bitfield_value():
    random.seed(0)
    bf = BitField(width=8, reset_val=0xFF)
    ref = BitFieldRef(name="ref_valset", reg_offset=2, slice_width=3, field_offset=1)
    ref._add(bf)
    # Initially bf value is mask full 0xFF
    bf.value = 0b11111111
    # Set ref.value to a value, it should update bf.value bits correctly
    old_bf_value = bf.value
    # Set to low value 0x10 but shifted right by reg_offset so shifted back in setter
    new_val = 0x28  # binary 0010 1000
    ref.value = new_val
    # Confirm bits in bf.value changed at bit positions field_offset..field_offset+slice_width
    mask = ((1 << ref.slice_width) -1) << ref.field_offset
    expected_val = (old_bf_value & ~mask) | (((new_val >> ref.reg_offset) & ((1 << ref.slice_width)-1)) << ref.field_offset)
    assert bf.value == expected_val

def test_bitfieldref_copy_alias_behavior(monkeypatch):
    random.seed(0)
    bf = BitField(width=8)
    ref1 = BitFieldRef(name="ref1", reg_offset=0, slice_width=8)
    ref1._add(bf)
    ref2 = BitFieldRef(name="ref2", reg_offset=0, slice_width=8)
    ref2._add(bf)
    # ref2 triggers _alias True in ref1 or ref2 due to multiple references

    # To test copy behavior with alias, no deep copy of children
    copy_ref = ref1._copy(new_instance=True)
    # The copied object's bitfield should be same instance as original
    assert copy_ref.bf is bf

def test_bitfieldref_validate():
    random.seed(0)
    bf = BitField(width=8)
    ref = BitFieldRef(name="ref_val", reg_offset=0, slice_width=8)
    # No _bf set yields validation error
    ref._bf = None
    errors = list(ref.validate())
    assert any("No child bitfield present" in str(e) for e in errors)

    # slice_width larger than bitfield.width yields validation error
    ref._bf = bf
    ref.slice_width = 16
    errors = list(ref.validate())
    assert any("Slice width is larger than bitfield's width" in str(e) for e in errors)

def test_registermap_block_dimensions_and_read_write(monkeypatch):
    random.seed(0)
    # Setup RegisterMap with two registers at addresses 0x1000 and 0x1004
    regmap = RegisterMap(name="regmap_test")
    reg1 = Register(name="r1", address=0x1000, width=32)
    reg2 = Register(name="r2", address=0x1004, width=32)
    # Add dummy bitfieldref children to avoid "no bitfieldrefs present" error for iteration and value
    bf, ref = setup_bf_and_ref(width=32)
    reg1._children = {ref.name: ref}
    reg2._children = {ref.name: ref}

    regmap._children = {reg1.name: reg1, reg2.name: reg2}

    monkeypatch.setattr(regmap, "_block_read_func", dummy_block_read)
    monkeypatch.setattr(regmap, "_block_write_func", dummy_block_write)

    # Check _block_dimensions property returns correct block start and width
    start_address, block_width = regmap._block_dimensions
    expected_start = 0x1000
    expected_width = (0x1004 + 4) - 0x1000  # 8 bytes for two 32 bit regs
    assert start_address == expected_start
    assert block_width == expected_width

    # read populates registers correctly using dummy_block_read
    data = regmap.read()
    assert data == dummy_block_read(start_address, block_width)
    # Each register's value should be equal to its corresponding slice bytes interpreted as little endian
    for r in regmap:
        s, e, n = (r_map.RegisterMap.get_dimensions(start_address, r.address, r.width))
        expected_val = int.from_bytes(data[s:e], byteorder='little')
        assert r.value == expected_val

    # write with data=None generates bytearray correctly with register values and calls _block_write_func
    dummy_block_write.store.clear()
    regmap.write(data=None)
    stored = dummy_block_write.store[start_address]
    assert isinstance(stored, (bytes, bytearray))
    assert len(stored) == block_width

    # write with explicit data calls _block_write_func with that data
    sample_data = bytearray(b'\x01\x02\x03\x04\x05\x06\x07\x08')
    dummy_block_write.store.clear()
    regmap.write(data=sample_data)
    stored2 = dummy_block_write.store[start_address]
    assert stored2 == sample_data

def test_get_dimensions_function():
    # test get_dimensions utility function from RegisterMap.py
    from r_map.RegisterMap import get_dimensions
    start_address = 0x1000
    reg_address = 0x1004
    width = 32  # bits -> 4 bytes
    start_idx, end_idx, num_bytes = get_dimensions(start_address, reg_address, width)
    expected_start_idx = reg_address - start_address
    expected_end_idx = expected_start_idx + 4
    expected_num_bytes = expected_end_idx - expected_start_idx
    assert start_idx == expected_start_idx
    assert end_idx == expected_end_idx
    assert num_bytes == expected_num_bytes
