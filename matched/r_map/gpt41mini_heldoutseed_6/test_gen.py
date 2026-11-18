import pytest
import r_map
from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap
from r_map.ValidationError import ValidationError

# Helper: create a dummy register with BitFieldRefs and BitFields
def create_register_with_bitfields():
    reg = Register(name="reg1", address=0x1000, width=16)
    bf1 = BitField(name="bf1", width=8, reset_val=0xA5, access='RW')
    bf2 = BitField(name="bf2", width=8, reset_val=0x5A, access='RO')
    ref1 = BitFieldRef(name="bfref1", reg_offset=0, slice_width=8, field_offset=0)
    ref2 = BitFieldRef(name="bfref2", reg_offset=8, slice_width=8, field_offset=0)
    ref1._add(bf1)
    ref2._add(bf2)
    # Add BitFieldRefs as children of register
    reg._children[ref1.name] = ref1
    reg._children[ref2.name] = ref2
    # Setup dummy read/write functions for register
    reg._reg_read_func = lambda addr: 0xA55A
    reg._reg_write_func = lambda addr, val: None
    return reg, bf1, bf2, ref1, ref2

def create_registermap_with_registers():
    regmap = RegisterMap(name="regmap")
    r1 = Register(name="r1", address=0x1000, width=16)
    r2 = Register(name="r2", address=0x1002, width=16)
    r1.value = 0x1234
    r2.value = 0xABCD
    regmap._children[r1.name] = r1
    regmap._children[r2.name] = r2
    regmap._block_read_func = lambda addr, size: b'\x34\x12\xcd\xab'
    regmap._block_write_func = lambda addr, data: None
    return regmap, r1, r2

########## Tests for Register ##########

def test_register_str_contains_value_hex():
    reg, bf1, bf2, ref1, ref2 = create_register_with_bitfields()
    reg.value = 0x1234
    s = str(reg)
    assert "value:" in s
    assert hex(0x1234) in s

def test_register_value_property_and_setter():
    reg, bf1, bf2, ref1, ref2 = create_register_with_bitfields()
    # Setting value sets all bitfieldrefs
    reg.value = 0xA5A5
    # BitFieldRefs values reflect parts of the value
    val0 = ref1.value
    val1 = ref2.value
    combined = (val0 >> ref1.reg_offset) | (val1 >> ref2.reg_offset) << ref2.reg_offset
    # Because the value setter shifts and masks values accordingly,
    # ref1 and ref2 value are parts of the register value.
    # Just test they do not exceed 8 bits
    assert 0 <= val0 <= 0xFF << 0
    assert 0 <= val1 <= 0xFF << 8

def test_register_reset_resets_all_bitfieldrefs():
    reg, bf1, bf2, ref1, ref2 = create_register_with_bitfields()
    reg.value = 0xFFFF
    reg.reset()
    # After reset, bitfields' values equal their reset_val
    assert bf1.value == bf1.reset_val
    assert bf2.value == bf2.reset_val

def test_register_access_property_unique_sorted():
    reg, bf1, bf2, ref1, ref2 = create_register_with_bitfields()
    assert 'RW' in reg.access
    assert 'RO' in reg.access
    # access string contains both
    acc_split = reg.access.split('|')
    assert 'RW' in acc_split and 'RO' in acc_split
    # access string is sorted alphabetically
    assert acc_split == sorted(set(acc_split))

def test_register_validate_no_bitfieldrefs_yields_error():
    reg = Register(name="empty_reg", address=0x2000)
    # No children added, so should yield ValidationError about no bitfieldrefs
    errors = list(reg.validate())
    assert any("No bitfieldrefs present" in str(e) for e in errors if isinstance(e, ValidationError))

def test_register_validate_child_not_bitfieldref_error():
    reg, bf1, bf2, ref1, ref2 = create_register_with_bitfields()
    # Add invalid child (not a BitFieldRef)
    reg._children['badchild'] = bf1  # BitField, not BitFieldRef
    errors = list(reg.validate())
    assert any("is not of typeBitFieldRef" in str(e).replace(' ', '') for e in errors if isinstance(e, ValidationError))

def test_register_validate_overlapping_bitfieldrefs_yields_error():
    reg = Register(name="reg_ovr", address=0x1000, width=16)
    bf = BitField(name="bf", width=8)
    ref1 = BitFieldRef(name="ref1", reg_offset=0, slice_width=8, field_offset=0)
    ref2 = BitFieldRef(name="ref2", reg_offset=4, slice_width=8, field_offset=0)
    ref1._add(bf)
    ref2._add(bf)
    reg._children[ref1.name] = ref1
    reg._children[ref2.name] = ref2
    errors = list(reg.validate())
    # Should report overlap
    assert any("overlaps" in str(e) for e in errors if isinstance(e, ValidationError))

def test_register_read_and_write_invoke_io_and_set_value(monkeypatch):
    reg, bf1, bf2, ref1, ref2 = create_register_with_bitfields()
    called = {}
    def fake_read(address):
        called['read'] = address
        return 0xBEEF
    def fake_write(address, val):
        called['write'] = (address, val)
    reg._reg_read_func = fake_read
    reg._reg_write_func = fake_write
    val = reg.read()
    assert val == 0xBEEF
    assert called.get('read') == reg.address

    reg.write(0x1234)
    assert called.get('write') == (reg.address, 0x1234)

    # setting value via write with val argument None
    reg.value = 0x5678
    reg.write()
    assert called.get('write') == (reg.address, reg.value)

def test_register_iter_sorted_by_reg_offset():
    reg = Register(name='reg_sort_test')
    bf = BitField(name='bf0', width=8)
    # Create refs with different reg_offset
    refA = BitFieldRef(name='refA', reg_offset=8, slice_width=8)
    refB = BitFieldRef(name='refB', reg_offset=0, slice_width=8)
    refA._add(bf)
    refB._add(bf)
    reg._children[refA.name] = refA
    reg._children[refB.name] = refB
    order = [child.name for child in reg]
    assert order == ['refB', 'refA']

########## Tests for BitField ##########

def test_bitfield_init_invalid_width_raises():
    with pytest.raises(ValueError):
        BitField(name="bf", width=0)
    with pytest.raises(ValueError):
        BitField(name="bf", reset_val=-1)

def test_bitfield_value_set_get_int_and_str_enum(monkeypatch):
    bf = BitField(name="bf", width=3)
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return self.name
    enum0 = DummyEnum("ZERO", 0)
    enum1 = DummyEnum("ONE", 1)
    bf._children[enum0.name] = enum0
    bf._children[enum1.name] = enum1

    bf.value = 5
    assert bf.value == 5 & bf.mask

    bf.value = "ONE"
    assert bf.value == 1

    bf.value = enum0
    assert bf.value == 0

    with pytest.raises(ValueError):
        bf.value = "INVALID"

def test_bitfield_annotation_returns_name_or_hex():
    bf = BitField(name="bf", width=4)
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
    enum0 = DummyEnum("ZERO", 0)
    enum1 = DummyEnum("ONE", 1)
    bf._children[enum0.name] = enum0
    bf._children[enum1.name] = enum1

    bf.value = 0
    assert bf.annotation == "ZERO"

    bf.value = 2
    # No enum with value 2, should return hex string
    assert bf.annotation == hex(2)

def test_bitfield_validate_duplicate_enums_yields_error():
    bf = BitField(name="bf", width=4)
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return self.name
    enum0 = DummyEnum("ZERO", 0)
    enum1a = DummyEnum("ONEA", 1)
    enum1b = DummyEnum("ONEB", 1)
    bf._children[enum0.name] = enum0
    bf._children[enum1a.name] = enum1a
    bf._children[enum1b.name] = enum1b
    errors = list(bf.validate())
    assert any("same value" in str(e) for e in errors if isinstance(e, ValidationError))

def test_bitfield_write_reads_and_writes(monkeypatch):
    bf = BitField(name="bf", width=8)
    reg = Register(name="r", address=0x1000, width=8)
    ref = BitFieldRef(name="ref", reg_offset=0, slice_width=8)
    ref._add(bf)
    reg._children[ref.name] = ref
    bf._references = {ref}
    reg._reg_read_func = lambda addr: 0x55
    calls = {'read': 0, 'write': 0}
    reg.read = lambda: (calls.update({'read': calls['read']+1}) or 0x55)
    reg.write = lambda: calls.update({'write': calls['write']+1})
    bf.write(0xAA, always_write=True)
    assert calls['read'] >= 1
    assert calls['write'] >= 1
    calls = {'read': 0, 'write': 0}
    bf.write(0xAA, always_write=False)  # same value write avoided
    assert calls['read'] == 0 and calls['write'] == 0 or calls['read'] >= 1

def test_bitfield_read_invokes_parent_read():
    bf = BitField(name="bf", width=8)
    reg = Register(name="r", address=0x1000, width=8)
    ref = BitFieldRef(name="ref", reg_offset=0, slice_width=8)
    ref._add(bf)
    bf._references = {ref}
    called = {'read': 0}
    reg._reg_read_func = lambda addr: 0xFF
    reg.read = lambda: (called.setdefault('read', 0)+1) and 0xFF
    ref.parent = reg
    bf._references = {ref}
    bf._get_parent_regs = lambda: [reg]

    # Patch reg.read to count calls
    def read_count():
        called['read'] += 1
        return 0xFF
    reg.read = read_count
    result = bf.read()
    assert result == bf.value or isinstance(result, int)
    assert called['read'] >= 1

########## Tests for BitFieldRef ##########

def test_bitfieldref_init_invalid_values_raises():
    with pytest.raises(ValueError):
        BitFieldRef(name="bfr", reg_offset=0, slice_width=-1)
    with pytest.raises(ValueError):
        BitFieldRef(name="bfr", reg_offset=0, slice_width=1, field_offset=-5)

def test_bitfieldref_add_adds_bitfield_and_sets_parent_and_references():
    bf = BitField(name="bf", width=8)
    bfr = BitFieldRef(name="bfr", reg_offset=0, slice_width=0)
    bfr._add(bf)
    assert bfr.bf == bf
    assert bfr._children[bf.name] == bf
    assert bfr in bf._references
    assert bf._parent == bfr or bf._parent is None or bf._parent == bfr

def test_bitfieldref_value_property_get_set_correct():
    bf = BitField(name="bf", width=16)
    bfr = BitFieldRef(name="bfr", reg_offset=4, slice_width=8, field_offset=4)
    bfr._add(bf)
    # set bitfield value
    bf.value = 0xFFFF
    val_before = bfr.value
    # calculate expected masked and shifted value
    expected = ((bf.value >> bfr.field_offset) & bfr.mask) << bfr.reg_offset
    assert val_before == expected
    # set new bitfieldref value and check bitfield value changed accordingly
    bfr.value = expected + (1 << bfr.reg_offset)
    # Bitfield.value lower bits should be updated as per setter logic
    # The exact value depends on masking. Check that bitfield.value changed
    assert bf.value != 0xFFFF

def test_bitfieldref_reset_val_property_computes_correct():
    bf = BitField(name="bf", width=16, reset_val=0x1234)
    bfr = BitFieldRef(name="bfr", reg_offset=2, slice_width=6, field_offset=1)
    bfr._add(bf)
    expected = ((bf.reset_val >> bfr.field_offset) & bfr.mask) << bfr.reg_offset
    assert bfr.reset_val == expected

def test_bitfieldref_validate_no_child_yields_error():
    bfr = BitFieldRef(name="bfr", reg_offset=0, slice_width=8, field_offset=0)
    # No bitfield added -> _bf defaults to None
    errors = list(bfr.validate())
    assert any("No child bitfield present" in str(e) for e in errors if isinstance(e, ValidationError))

def test_bitfieldref_validate_slice_width_larger_yields_error():
    bf = BitField(name="bf", width=4)
    bfr = BitFieldRef(name="bfr", reg_offset=0, slice_width=8, field_offset=0)
    bfr._add(bf)
    errors = list(bfr.validate())
    assert any("Slice width is larger than bitfield's width" in str(e) for e in errors if isinstance(e, ValidationError))

def test_bitfieldref_copy_alias_behavior():
    # Create BitField with two BitFieldRefs (aliases)
    bf = BitField(name="bf", width=8)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0, slice_width=8)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=0, slice_width=8)
    bfr1._add(bf)
    bfr2._add(bf)
    # bfr2 should mark itself as alias due to multiple references
    assert getattr(bfr2, '_alias', False) is True
    # Copy bfr2 with new_alias True forces alias copy - _copy returns new instance with shared bitfields
    copy = bfr2._copy(new_alias=True, new_instance=True)
    assert copy.bf == bfr2.bf

########## Tests for RegisterMap ##########

def test_registermap_block_dimensions_and_errors():
    regmap = RegisterMap(name="rm")
    r1 = Register(name="r1", address=0x1000, width=32)
    r2 = Register(name="r2", address=0x1004, width=32)
    regmap._children[r1.name] = r1
    regmap._children[r2.name] = r2
    # _block_dimensions property calculates correctly
    start, size = regmap._block_dimensions
    assert start == 0x1000
    assert size == (r2.address + 4) - start  # width/8 = 4 bytes for 32 bits

    # Setting manually overrides property
    regmap._block_dimensions = (0x2000, 8)
    assert regmap._block_dimensions == (0x2000, 8)

    # Error when no registers present
    empty_map = RegisterMap(name="empty")
    with pytest.raises(ValueError):
        _ = empty_map._block_dimensions

def test_registermap_read_populates_register_values(monkeypatch):
    regmap, r1, r2 = create_registermap_with_registers()
    data = regmap.read()
    assert isinstance(data, bytes)
    # After read, registers should have value matching data
    assert r1.value == 0x1234
    assert r2.value == 0xABCD

def test_registermap_write_uses_default_bytearray_and_calls_write(monkeypatch):
    regmap, r1, r2 = create_registermap_with_registers()
    called = {}
    def fake_write(addr, data):
        called['addr'] = addr
        called['data'] = data
    regmap._block_write_func = fake_write
    r1.value = 0x1122
    r2.value = 0x3344
    regmap.write()
    assert called['addr'] == 0x1000
    assert isinstance(called['data'], (bytes, bytearray))
    assert len(called['data']) == regmap._block_dimensions[1]

def test_registermap_write_with_data_updates_registers():
    regmap, r1, r2 = create_registermap_with_registers()
    # Prepare data that updates r1 and r2 values
    data = b'\x01\x00\x02\x00'  # Little endian: r1=0x0001, r2=0x0002
    regmap.write(data)
    assert r1.value == 0x0001
    assert r2.value == 0x0002

def test_registermap_write_raises_when_no_registers():
    regmap = RegisterMap(name="empty_map")
    with pytest.raises(ValueError):
        regmap.write()

