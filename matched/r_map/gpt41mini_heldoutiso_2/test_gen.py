import pytest
import r_map
from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap
from r_map.ValidationError import ValidationError


class DummyRegIO:
    """Helper class to simulate reg read/write ops."""
    def __init__(self):
        self.written = {}
        self.read_values = {}

    def read_func(self, addr):
        return self.read_values.get(addr, 0)

    def write_func(self, addr, val):
        self.written[addr] = val


class DummyBlockIO:
    """Helper class to simulate block read/write ops."""
    def __init__(self, size):
        self.data = bytearray(size)
        self.written_data = None

    def block_read_func(self, addr, size):
        return self.data[:size]

    def block_write_func(self, addr, data):
        self.written_data = data


def test_register_init_and_defaults():
    reg = Register(name="reg1", width=16, address=0x10)
    assert reg.width == 16
    assert reg.name == "reg1"
    assert reg.address == 0x10
    assert reg.value == 0


def test_bitfield_init_rejects_invalid_width_and_reset_val():
    with pytest.raises(ValueError):
        BitField(name="bf", width=0)
    with pytest.raises(ValueError):
        BitField(name="bf", reset_val=-1)


def test_bitfield_value_setter_and_getter_int_and_enum():
    bf = BitField(name="bf", width=4)
    # Set int within mask
    bf.value = 0xF
    assert bf.value == 0xF
    # Set int outside mask
    bf.value = 0xFF
    assert bf.value == 0xF

    # Create dummy enumeration to test enum assignment
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    enum1 = DummyEnum("enum1", 3)
    bf._children = {"enum1": enum1}
    bf.value = enum1
    assert bf.value == 3


def test_bitfield_value_setter_string_matches_enum_and_raises():
    bf = BitField(name="bf", width=4)
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
    enum1 = DummyEnum("enum1", 5)
    bf._children = {"enum1": enum1}

    bf.value = "enum1"
    assert bf.value == 5

    with pytest.raises(ValueError):
        bf.value = "no_such_enum"

def test_bitfield_annotation_returns_enum_name_or_hex():
    bf = BitField(name="bf", width=3)
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
    enum1 = DummyEnum("enum1", 1)
    enum2 = DummyEnum("enum2", 3)
    bf._children = {"enum1": enum1, "enum2": enum2}

    bf.value = 1
    assert bf.annotation == "enum1"

    bf.value = 2
    assert bf.annotation == hex(2)


def test_bitfield_reset_sets_value_to_reset_val():
    bf = BitField(name="bf", width=8, reset_val=0xAA)
    bf.value = 0
    bf.reset()
    assert bf.value == 0xAA


def test_bitfield_validate_duplicate_enumerations():
    bf = BitField(name="bf", width=8)
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return self.name
    e1 = DummyEnum("e1", 1)
    e2 = DummyEnum("e2", 1)
    bf._children = {e1.name: e1, e2.name: e2}

    errors = list(bf.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert "have the same value" in errors[0].message


def test_bitfieldref_init_and_properties():
    bf = BitField(name="bf", width=8, reset_val=0xF0)
    bfr = BitFieldRef(name="bfr", reg_offset=2, slice_width=4, field_offset=1)
    bfr._add(bf)
    assert bfr.reg_offset == 2
    assert bfr.slice_width == 4
    assert bfr.field_offset == 1
    expected_mask = (1 << 4) - 1
    assert bfr.mask == expected_mask
    # Reset val property calculation
    expected = ((bf.reset_val >> 1) & expected_mask) << 2
    assert bfr.reset_val == expected


def test_bitfieldref_value_setter_getter():
    bf = BitField(name="bf", width=8)
    bf.value = 0xFF
    bfr = BitFieldRef(name="bfr", reg_offset=4, slice_width=4, field_offset=0)
    bfr._add(bf)

    # Value from BitFieldRef getter correct
    expected_val = ((bf.value >> 0) & ((1 << 4)-1)) << 4
    assert bfr.value == expected_val

    # Setting value affects underlying bitfield value properly
    old_val = bf.value
    new_val = 0xA0  # This is the value set at reg_offset shifted bits
    bfr.value = new_val
    # According to setter, new_val is shifted right by reg_offset, masked and placed at field_offset in bf.value
    bf_expected_val = (old_val & ~((1 << 4)-1)) | ((new_val >> 4) & ((1 << 4)-1))
    assert bf.value == bf_expected_val


def test_bitfieldref_validate_errors_and_alias():
    bf = BitField(name="bf", width=8)
    bfr = BitFieldRef(name="bfr", reg_offset=0, slice_width=16, field_offset=0)
    bfr._add(bf)

    # slice_width > bf.width triggers ValidationError
    errors = list(bfr.validate())
    assert any(isinstance(e, ValidationError) for e in errors)
    assert "Slice width is larger than bitfield's width" in errors[0].message

    # BitFieldRef with no _bf triggers error
    bfr2 = BitFieldRef(name="bfr2", reg_offset=0, slice_width=4, field_offset=0)
    bfr2._bf = None
    errors2 = list(bfr2.validate())
    assert any(isinstance(e, ValidationError) for e in errors2)
    assert "No child bitfield present" in errors2[0].message


def test_register_value_setter_and_getter_ordering_and_reset_val():
    bf1 = BitField(name="bf1", width=4, reset_val=0x3)
    bf2 = BitField(name="bf2", width=4, reset_val=0xC)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0, slice_width=4, field_offset=0)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=4, slice_width=4, field_offset=0)
    bfr1._add(bf1)
    bfr2._add(bf2)

    reg = Register(name="reg", width=8)
    # attach bitfieldrefs as children. They need reg_offset for sorting
    reg._children = {bfr1.name: bfr1, bfr2.name: bfr2}

    # The value of register is combined from bitfieldref values
    expected_val = bfr1.value | bfr2.value
    assert reg.value == expected_val

    # Setting register.value sets all children values appropriately
    reg.value = 0x5A
    # bfr1 is lower nibble, expects bits [0:4]
    expected_bfr1_val = ((0x5A >> bfr1.reg_offset) & bfr1.mask) << bfr1.reg_offset
    # Getting actual bfr1.value returns shifted and masked from bitfield value
    assert (bfr1.value & 0xF0) == expected_bfr1_val

    # reset_val is combination of reset_val of bitfield refs
    assert reg.reset_val == (bfr1.reset_val | bfr2.reset_val)


def test_register_validate_produces_errors_for_no_bitfieldrefs_and_wrong_type():
    reg = Register(name="reg")
    # No children => error "No bitfieldrefs present"
    errors = list(reg.validate())
    assert any("No bitfieldrefs present" in e.message for e in errors)

    # Add a child that is not BitFieldRef
    class DummyChild:
        pass
    dummy = DummyChild()
    reg._children = {"d": dummy}
    errors = list(reg.validate())
    assert any("is not of type BitFieldRef" in e.message for e in errors)


def test_register_validate_overlapping_bitfieldrefs_error():
    bf1 = BitField(name="bf1", width=4)
    bf2 = BitField(name="bf2", width=4)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0, slice_width=4, field_offset=0)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=3, slice_width=4, field_offset=0)  # Overlaps with bfr1 (3 < 0+4)
    bfr1._add(bf1)
    bfr2._add(bf2)

    reg = Register(name="reg", width=8)
    reg._children = {bfr1.name: bfr1, bfr2.name: bfr2}

    errors = list(reg.validate())
    assert any("overlaps" in e.message for e in errors)


def test_register_read_and_write_calls_underlying_funcs_and_value_set():
    bf1 = BitField(name="bf1", width=8)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0, slice_width=8, field_offset=0)
    bfr1._add(bf1)
    reg = Register(name="reg", width=8, address=0x10)
    reg._children = {bfr1.name: bfr1}

    dummy_io = DummyRegIO()
    dummy_io.read_values[reg.address] = 0xAB
    reg._reg_read_func = dummy_io.read_func
    reg._reg_write_func = dummy_io.write_func

    ret = reg.read()
    # The read() returns raw value read
    assert ret == 0xAB
    # The register value is also updated from read
    assert reg.value == 0xAB

    reg.write(0x12)
    # After write(value), underlying write_func is called with value passed
    assert dummy_io.written[reg.address] == 0x12

    # Write without parameter uses current register value
    dummy_io.written.clear()
    reg.write()
    assert dummy_io.written[reg.address] == reg.value


def test_register_reset_calls_reset_on_bitfieldrefs():
    bf1 = BitField(name="bf1", width=4, reset_val=0x3)
    bf2 = BitField(name="bf2", width=4, reset_val=0xC)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0, slice_width=4, field_offset=0)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=4, slice_width=4, field_offset=0)
    bfr1._add(bf1)
    bfr2._add(bf2)
    reg = Register(name="reg", width=8)
    reg._children = {bfr1.name: bfr1, bfr2.name: bfr2}

    bf1.value = 0x0
    bf2.value = 0x0
    reg.reset()
    assert bf1.value == bf1.reset_val
    assert bf2.value == bf2.reset_val


def test_registermap_block_dimensions_calculation_and_error():
    regmap = RegisterMap(name="rmap")
    bf = BitField(name="bf", width=8)
    bfr = BitFieldRef(name="bfr", reg_offset=0, slice_width=8, field_offset=0)
    bfr._add(bf)
    reg = Register(name="reg", address=0x20, width=8)
    reg._children = {bfr.name: bfr}
    regmap._children = {reg.name: reg}

    # Access _block_dimensions property calculates from children
    start_addr, block_width = regmap._block_dimensions
    assert start_addr == 0x20
    assert block_width == 1  # width=8 bits == 1 byte block

    # Setting explicitly overrides _block_dimensions
    regmap._block_dimensions = (0x10, 4)
    assert regmap._block_dimensions == (0x10, 4)

    # Empty regmap raises ValueError accessing _block_dimensions
    regmap_empty = RegisterMap(name="empty")
    with pytest.raises(ValueError):
        _ = regmap_empty._block_dimensions


def test_registermap_read_and_write_block_io():
    regmap = RegisterMap(name="rmap")
    # Two 8-bit registers, addresses 0x10 and 0x11
    bf1 = BitField(name="bf1", width=8)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0, slice_width=8, field_offset=0)
    bfr1._add(bf1)
    reg1 = Register(name="reg1", address=0x10, width=8)
    reg1._children = {bfr1.name: bfr1}

    bf2 = BitField(name="bf2", width=8)
    bfr2 = BitFieldRef(name="bfr2", reg_offset=0, slice_width=8, field_offset=0)
    bfr2._add(bf2)
    reg2 = Register(name="reg2", address=0x11, width=8)
    reg2._children = {bfr2.name: bfr2}

    regmap._children = {reg1.name: reg1, reg2.name: reg2}

    # block size is 2 bytes
    block_size = 2
    dummy_block = bytearray([0x11, 0x22])
    dummy_io = DummyBlockIO(block_size)
    dummy_io.data = dummy_block

    regmap._block_read_func = dummy_io.block_read_func
    regmap._block_write_func = dummy_io.block_write_func

    # Read uses block read func, updates registers' values
    data = regmap.read()
    assert data == dummy_block
    assert reg1.value == 0x11
    assert reg2.value == 0x22

    # Write with no data writes combined register values to underlying function
    reg1.value = 0x55
    reg2.value = 0xAA
    dummy_io.written_data = None
    regmap.write()
    assert dummy_io.written_data is not None
    # The data written reflects register values at correct positions
    assert dummy_io.written_data[0] == 0x55
    assert dummy_io.written_data[1] == 0xAA

    # Write with provided data passes through underlying function
    new_data = bytearray([0x33, 0x44])
    dummy_io.written_data = None
    regmap.write(new_data)
    # Registers' values updated from new_data
    assert reg1.value == 0x33
    assert reg2.value == 0x44
    # Underlying write called with new_data
    assert dummy_io.written_data == new_data


def test_registermap_write_raises_when_no_registers():
    regmap = RegisterMap(name="rmap_empty")
    dummy_io = DummyBlockIO(0)
    regmap._block_read_func = dummy_io.block_read_func
    regmap._block_write_func = dummy_io.block_write_func

    with pytest.raises(ValueError):
        regmap.write()


def test_bitfield_write_read_calls_parent_registers(monkeypatch):
    bf = BitField(name="bf", width=8)
    bfr = BitFieldRef(name="bfr", reg_offset=0, slice_width=8, field_offset=0)
    bfr._add(bf)

    reg = Register(name="reg", address=0x10, width=8)
    reg._children = {bfr.name: bfr}

    # Set _references for parent registers on bitfield (done in _add)
    # Already set by _add()

    dummy_io = DummyRegIO()
    dummy_io.read_values[reg.address] = 0x12
    reg._reg_read_func = dummy_io.read_func
    reg._reg_write_func = dummy_io.write_func

    # Write with different value triggers write calls on registers
    bf.value = 0
    bf.write(0x34, always_write=False)
    # Because prev value 0 != 0x34, write occurs on parent register
    assert dummy_io.written[reg.address] == 0x34

    # Write with same value and always_write True writes anyway
    dummy_io.written.clear()
    bf.write(0x34, always_write=True)
    assert dummy_io.written[reg.address] == 0x34

    # Read triggers read on parent registers and returns bitfield.value
    bf.value = 0x56
    v = bf.read()
    assert v == bf.value
    assert dummy_io.read_values[reg.address] == 0x12  # read_values unchanged


def test_bitfieldref_copy_keeps_alias_behavior():
    bf = BitField(name="bf", width=8)
    bfr1 = BitFieldRef(name="bfr1", reg_offset=0, slice_width=8, field_offset=0)
    bfr1._add(bf)

    # Add a second BitFieldRef referencing same BitField to trigger alias
    bfr2 = BitFieldRef(name="bfr2", reg_offset=0, slice_width=8, field_offset=0)
    bfr2._add(bf)

    # Mark bfr2 alias flag set by _add
    assert getattr(bfr2, "_alias", False) is True

    # Copy bfr2 with new_alias True disables deep copy
    bfr2._ref = bfr1
    copy_bfr = bfr2._copy(new_alias=True)
    # The copy has _ref set and children as per alias behavior
    assert copy_bfr.bf is not None

