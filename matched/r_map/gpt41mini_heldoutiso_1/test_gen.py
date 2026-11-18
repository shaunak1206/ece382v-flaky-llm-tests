import pytest
import builtins
from r_map.Register import Register
from r_map.BitField import BitField
from r_map.BitFieldRef import BitFieldRef
from r_map.RegisterMap import RegisterMap
from r_map.ValidationError import ValidationError


class DummyRegIO:
    def __init__(self):
        self.written = []
        self.to_read = 0

    def read(self, address):
        return self.to_read

    def write(self, address, val):
        self.written.append((address, val))


class DummyBlockIO:
    def __init__(self):
        self.written = []
        self.to_read = bytearray()

    def read(self, address, size):
        return self.to_read

    def write(self, address, data):
        self.written.append((address, data))


def make_bitfield(name='bf', width=8, reset_val=0xAA, access='RW'):
    bf = BitField(name=name, width=width, reset_val=reset_val, access=access)
    return bf


def make_bitfieldref(reg_offset=0, slice_width=0, field_offset=0, **kwargs):
    return BitFieldRef(reg_offset=reg_offset, slice_width=slice_width, field_offset=field_offset, **kwargs)


def make_register(name='reg', width=32, **kwargs):
    reg = Register(name=name, width=width, **kwargs)
    reg._reg_read_func = lambda addr: 0
    reg._reg_write_func = lambda addr, val: None
    return reg


def make_registermap(name='map'):
    rm = RegisterMap(name=name)
    # Provide dummy block read/write functions
    dummy = DummyBlockIO()
    rm._block_read_func = dummy.read
    rm._block_write_func = dummy.write
    rm._dummyio = dummy
    return rm


def test_register_str_and_value_and_access():
    # Setup bitfields and refs
    bf = make_bitfield(width=8, reset_val=0x0F, access='RW')
    bfr = make_bitfieldref(reg_offset=0, slice_width=8, field_offset=0)
    bfr._add(bf)
    reg = make_register(width=8)
    reg._reg_read_func = lambda addr: 0xAB
    reg._reg_write_func = lambda addr, val: None
    # Attach bitfieldref to reg as children
    reg._children = {bfr.name: bfr} if hasattr(bfr, 'name') else {'bf_ref': bfr}

    # Iterate on reg yields bitfieldref(s)
    # Patch __iter__ to yield our bfr
    reg.__iter__ = lambda self=None: iter([bfr])
    # Value should be the value of the bitfieldref, currently 0
    assert reg.value == 0
    # When we set value, bitfieldref value should change, considering shifting logic
    reg.value = 0xFF
    assert bfr.value >> bfr.reg_offset & ((1 << bfr.slice_width) - 1) == 0xFF
    # access property reads from bitfieldref's bitfield's access: bitfieldref.access is bitfield's access
    bfr._bf = bf
    assert 'RW' in reg.access
    # __str__ contains 'value:'
    s = str(reg)
    assert 'value:' in s


def test_register_reset_resets_bitfieldrefs():
    bf = make_bitfield(reset_val=0x55)
    bfr = make_bitfieldref(reg_offset=0, slice_width=8)
    bfr._add(bf)
    # Patch __iter__ to yield our one bitfieldref
    reg = make_register()
    reg.__iter__ = lambda self=None: iter([bfr])
    bf.value = 0xFF
    reg.reset()
    # After reset, bitfieldref's bitfield value must be reset_val
    assert bfr._bf.value == bfr._bf.reset_val == 0x55


def test_register_validate_behavior():
    reg = make_register()
    # empty register triggers no bitfieldrefs error
    reg._children = {}
    reg.__iter__ = lambda self=None: iter([])
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) and "No bitfieldrefs" in str(e) for e in errors)

    # With children that are not BitFieldRefs - cause error
    class DummyChild:
        pass

    dummy_child = DummyChild()
    reg.__iter__ = lambda self=None: iter([dummy_child])

    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) and "is not of type BitFieldRef" in str(e) for e in errors)

    # With overlapping bitfieldrefs
    bf = make_bitfield()
    bfr1 = make_bitfieldref(reg_offset=0, slice_width=4)
    bfr1._add(bf)
    bfr2 = make_bitfieldref(reg_offset=3, slice_width=4)
    bfr2._add(bf)

    reg.__iter__ = lambda self=None: iter([bfr1, bfr2])
    # patch _children map to support ordering
    reg._children = {bfr1.name if hasattr(bfr1, 'name') else 'bfr1': bfr1,
                     bfr2.name if hasattr(bfr2, 'name') else 'bfr2': bfr2}
    errors = list(reg.validate())
    assert any(isinstance(e, ValidationError) and "overlaps" in str(e) for e in errors)


def test_bitfield_init_and_str_and_reset_and_value_set_get():
    bf = BitField(name='bf', width=4, reset_val=2)
    assert bf.width == 4
    assert bf.reset_val == 2
    # mask is correct
    assert bf.mask == 0b1111
    # value defaults to reset_val
    assert bf.value == 2
    # __str__ contains width, reset_val and value in hex
    s = str(bf)
    assert 'width' in s and 'reset_val' in s and 'value' in s

    # reset sets value to reset_val
    bf.value = 5
    bf.reset()
    assert bf.value == bf.reset_val

    # value setter & getter with int masking
    bf.value = 31  # 31 & 0xf = 15
    assert bf.value == 15

    # Setting value as enumeration or string should raise NotImplementedError or ValueError (no enum yet)
    with pytest.raises(NotImplementedError):
        bf.value = object()


def test_bitfield_annotation_enum_and_validate():
    bf = BitField(name='bf', width=2)
    # Add dummy enum children with same value -> validation error
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return f'DummyEnum({self.name}:{self.value})'

    enum1 = DummyEnum('A', 1)
    enum2 = DummyEnum('B', 1)
    bf._children = {'A': enum1, 'B': enum2}
    # __iter__ gives enums
    bf.__iter__ = lambda self=None: iter(bf._children.values())
    errors = list(bf.validate())
    assert any("same value" in str(e) for e in errors)

    # annotation returns name if matches value
    bf.value = 1
    ann = bf.annotation
    assert ann == 'A' or ann == 'B'


def test_bitfield_write_and_read_io(monkeypatch):
    bf = make_bitfield()
    bfr = make_bitfieldref(reg_offset=0, slice_width=bf.width)
    bfr._add(bf)
    reg = make_register()
    reg.__iter__ = lambda self=None: iter([bfr])
    reg._reg_read_func = lambda addr: 0x12
    wrote = {}
    def fake_write(addr, val):
        wrote['val'] = val
    reg._reg_write_func = fake_write
    bf._references = {bfr}
    bfr._bf = bf

    bf._parent = bfr
    # patch _get_parent_regs for bf
    bf._references = set([bfr])
    bfr.parent = reg

    bf._get_parent_regs = lambda: [reg]

    bf.value = 10
    bf.write(10, always_write=True)
    # After write, the write function of reg called with val=10
    assert wrote['val'] == 10

    # read triggers reg.read and returns value
    val = bf.read()
    assert val == bf.value


def test_bitfieldref_init_and_properties_and_value_set_get():
    bf = make_bitfield(width=8, reset_val=0x12)
    bfr = BitFieldRef(reg_offset=2, slice_width=4, field_offset=1)
    bfr._add(bf)
    # mask should be 0b1111
    assert bfr.mask == 0b1111
    # reset_val property correct shifted and masked
    expected_reset = ((bf.reset_val >> bfr.field_offset) & bfr.mask) << bfr.reg_offset
    assert bfr.reset_val == expected_reset

    # value getter and setter: test that setting bfr.value updates bf.value correctly
    initial_bf_val = bf.value
    new_val = (0x5 << bfr.reg_offset)
    bfr.value = new_val
    expected_val = (initial_bf_val & ~(bfr.mask << bfr.field_offset)) | ((new_val >> bfr.reg_offset) << bfr.field_offset)
    assert bf.value == expected_val

    # __slice_width_setup raises no error and sets mask correctly
    bfr2 = BitFieldRef(reg_offset=0, slice_width=5)
    assert bfr2.mask == 0b11111


def test_bitfieldref_add_invalid_type():
    bfr = BitFieldRef(reg_offset=0, slice_width=1)
    with pytest.raises(ValueError):
        bfr._add("not a bitfield")  # expect error


def test_bitfieldref_validate_errors():
    bfr = BitFieldRef(reg_offset=0, slice_width=4)
    # no bitfield assigned should yield error
    bfr._bf = None
    errors = list(bfr.validate())
    assert any("No child bitfield" in str(e) for e in errors)

    # slice_width larger than bitfield width triggers error
    bf = make_bitfield(width=2)
    bfr._bf = bf
    bfr.slice_width = 5
    errors = list(bfr.validate())
    assert any("Slice width is larger" in str(e) for e in errors)


def test_registermap_block_dimensions_and_read_write():
    rm = make_registermap()
    # Prepare registers to add
    r1 = make_register(name='r1', width=16, address=0)
    r2 = make_register(name='r2', width=16, address=2)
    # patch __iter__ to yield regs in order
    rm.__iter__ = lambda self=None: iter([r1, r2])
    rm._children = {'r1': r1, 'r2': r2}

    # Confirm _block_dimensions property returns correct values
    start, size = rm._block_dimensions
    assert start == r1.address
    assert size == (r2.address + 2) - start

    # Provide values for registers
    r1.value = 0x1234
    r2.value = 0x5678
    # Setup dummy block_write_func and block_read_func to track calls
    called = {}
    def dummy_write(addr, data):
        called['addr'] = addr
        called['data'] = data
    rm._block_write_func = dummy_write

    dummy_data = r1.value.to_bytes(2, 'little') + r2.value.to_bytes(2, 'little')
    rm._block_read_func = lambda addr, size: dummy_data

    # Test read fills register values correctly
    data = rm.read()
    assert data == dummy_data
    assert r1.value == int.from_bytes(dummy_data[0:2], 'little')
    assert r2.value == int.from_bytes(dummy_data[2:4], 'little')
    # Test write with None data writes the register values to block
    rm._block_write_func = dummy_write
    rm.write()
    assert called['addr'] == start
    assert isinstance(called['data'], (bytes, bytearray))
    # Test write with data arg writes using that data
    data_to_write = bytearray(4)
    data_to_write[0:2] = (0x1111).to_bytes(2, 'little')
    data_to_write[2:4] = (0x2222).to_bytes(2, 'little')
    rm.write(data_to_write)
    # Registers updated accordingly
    assert r1.value == 0x1111
    assert r2.value == 0x2222


def test_registermap_write_empty_registers_raises():
    rm = RegisterMap(name='empty')
    rm._block_read_func = lambda a, s: bytes(s)
    rm._block_write_func = lambda a, d: None
    rm.__iter__ = lambda self=None: iter([])
    rm._children = {}

    with pytest.raises(ValueError, match="No registers found"):
        rm.write()


def test_bitfieldref_copy_deep_and_alias():
    bf = make_bitfield()
    bfr = BitFieldRef(reg_offset=0, slice_width=bf.width)
    bfr._add(bf)

    # Emulate reference presence on bitfield
    bf._references = {bfr}
    bfr._alias = False
    bfr._ref = None

    # Shallow copy with new_alias=True disables deep copy
    copy_obj = bfr._copy(new_alias=True)
    assert copy_obj._bf == bfr._bf

    # Deep copy with new_alias=False creates distinct copy (we test only type)
    deep_copy = bfr._copy(new_alias=False)
    # The copy should be same type but different instance
    assert isinstance(deep_copy, BitFieldRef)
    assert deep_copy is not bfr


def test_bitfieldref_value_setter_masks_and_shifts():
    bf = make_bitfield(width=8, reset_val=0)
    bfr = BitFieldRef(reg_offset=2, slice_width=4, field_offset=1)
    bfr._add(bf)

    bf.value = 0x00
    val_to_set = 0b10100 << bfr.reg_offset  # 0b10100 = 20
    bfr.value = val_to_set
    # Expect bf.value only 4 bits starting at field_offset replaced
    expected = (bf.value & ~(bfr.mask << bfr.field_offset)) | ((val_to_set >> bfr.reg_offset) << bfr.field_offset)
    assert bf.value == expected


def test_bitfield_value_setter_str_with_invalid_string_raises():
    bf = make_bitfield()
    with pytest.raises(ValueError):
        bf.value = 'invalidEnumName'


def test_bitfield_value_setter_str_with_valid_enum(monkeypatch):
    bf = make_bitfield()
    class DummyEnum:
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __str__(self):
            return f"DummyEnum({self.name})"
    enum = DummyEnum('valid', 1)
    bf._children = {'valid': enum}
    # Patch __iter__ to yield our dummy enum
    bf.__iter__ = lambda self=None: iter(bf._children.values())
    bf.value = 'valid'
    assert bf.value == 1


def test_register_write_read_interaction():
    reg = make_register(address=0x10)
    dummy = DummyRegIO()
    dummy.to_read = 0xDEADBEAF
    reg._reg_read_func = dummy.read
    reg._reg_write_func = dummy.write

    # Setup bitfieldref and bitfield so iteration works
    bf = make_bitfield(width=32)
    bfr = BitFieldRef(reg_offset=0, slice_width=32)
    bfr._add(bf)
    reg._children = {'bf_ref': bfr}
    reg.__iter__ = lambda self=None: iter([bfr])

    val = reg.read()
    assert val == dummy.to_read
    # Setting value and writing passes value through to reg_write_func
    reg.value = 0xABCD1234
    reg.write()
    # The dummy's written list should have 1 entry with addr and value matching our set value
    assert dummy.written[-1][0] == reg.address
    assert dummy.written[-1][1] == 0xABCD1234

