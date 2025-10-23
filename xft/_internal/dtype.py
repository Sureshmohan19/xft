# xft dtype system - Comprehensive data type management with type promotion.

import dataclasses
import functools
import warnings
from typing import Optional, Union, Tuple

import numpy as np
import ml_dtypes

__all__ = [
    'DType',                                # Main class
    'dtypes',
    'dtype',                                # Core functions
    'canonicalize_dtype',
    'check_and_canonicalize_user_dtype',
    'issubdtype',                           # Type checking
    'isdtype',
    'is_string_dtype',
    'is_scalar_type',
    'is_weakly_typed',                  
    'to_numeric',                           # Conversions
    'to_inexact',
    'to_complex',
    'as_numpy_dtype',
    'dtype_to_numpy',
    'numpy_to_dtype',
    'promote_types',                        # Promotion
    'result_type',
    'can_cast_safely',
    'TypePromotionError',
    'default_int_dtype',                    # Defaults
    'default_uint_dtype',
    'default_float_dtype',
    'default_complex_dtype',
    'scalar_type',                          # Utilities
    'short_dtype_name',
    'supports_inf',
    'bit_width',
    'ensure_array',
    'get_dtype_info',
    'compatible_dtypes',
    'set_promotion_mode',                   # Configuration
    'get_promotion_mode',
    'set_x64_enabled',
    'get_x64_enabled',
    'register_weak_type',                   # Weak types
]

# CONFIGURATION & GLOBALS

#TODO: Move these to a dedicated config.py file.
x64_enabled = True
_DTYPE_PROMOTION_MODE = "standard"  # "standard" or "strict"

# DTYPE INFO HELPERS (handle both NumPy and ml_dtypes)

int_info, float_info = ml_dtypes.iinfo, ml_dtypes.finfo

# DTYPE CLASS DEFINITION

@dataclasses.dataclass(frozen=True, eq=False)
class DType:
    """Core dtype class wrapping NumPy/ml_dtypes with additional metadata."""
    name: str
    priority: int
    native_type: object

    def __repr__(self) -> str:
        return f'dtypes.{self.name}'
    
    def __str__(self) -> str:
        return self.name
    
    def __lt__(self, other) -> bool:
        return self.priority < other.priority
    
    def __gt__(self, other) -> bool:
        return self.priority > other.priority
    
    def __hash__(self) -> int:
        return hash((self.name, self.priority))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, DType):
            return False
        return self.name == other.name and self.priority == other.priority
    
    @property
    def np_dtype(self):
        """Get the underlying NumPy dtype."""
        return self.native_type
    
    @property
    def itemsize(self) -> int:
        """Get size in bytes."""
        return self.np_dtype.itemsize

    @property
    def min(self):
        """Get minimum representable value."""
        if self is dtypes.bool:
            return False
        if dtypes.is_int(self):
            return int_info(self.native_type).min
        if dtypes.is_float(self):
            return -np.inf
        raise TypeError(f"min value is not defined for non-numeric dtype {self.name}")
    
    @property
    def max(self):
        """Get maximum representable value."""
        if self is dtypes.bool:
            return True
        if dtypes.is_int(self):
            return int_info(self.native_type).max
        if dtypes.is_float(self):
            return np.inf
        raise TypeError(f"max value is not defined for non-numeric dtype {self.name}")

    @property
    def bit_width(self) -> int:
        """Get bit width of the dtype."""
        if self is dtypes.bool:
            return 8
        if dtypes.is_int(self):
            return int_info(self.native_type).bits
        if dtypes.is_float(self):
            return float_info(self.native_type).bits
        if self in dtypes.complexes:
            return 2 * float_info(self.native_type).bits
        raise ValueError(f"bit_width is not defined for non-numeric dtype {self.name}")
    
    @property
    def supports_inf(self) -> bool:
        """Check if dtype supports infinity values."""
        if self in _no_inf_types:
            return False
        return dtypes.is_float(self) or self in dtypes.complexes
    
    @property
    def short_name(self) -> str:
        """Get abbreviated name (f32 instead of float32)."""
        return (self.name.replace('float', 'f')
                         .replace('uint', 'u')
                         .replace('int', 'i')
                         .replace('complex', 'c'))

# DTYPES NAMESPACE

class dtypes:
    """Namespace containing all supported dtypes and utility methods."""
    
    # Special dtypes
    void = DType("void", -2, np.dtype(np.void))
    float0 = DType("float0", -1, np.dtype([('float0', np.void, 0)]))
    bool = DType("bool", 0, np.dtype(np.bool_))
    
    # Integer dtypes (custom and standard)
    int2 = DType("int2", 4, np.dtype(ml_dtypes.int2))
    uint2 = DType("uint2", 5, np.dtype(ml_dtypes.uint2))
    int4 = DType("int4", 6, np.dtype(ml_dtypes.int4))
    uint4 = DType("uint4", 7, np.dtype(ml_dtypes.uint4))
    int8 = DType("int8", 8, np.dtype(np.int8))
    uint8 = DType("uint8", 9, np.dtype(np.uint8))
    int16 = DType("int16", 10, np.dtype(np.int16))
    uint16 = DType("uint16", 11, np.dtype(np.uint16))
    int32 = DType("int32", 12, np.dtype(np.int32))
    uint32 = DType("uint32", 13, np.dtype(np.uint32))
    int64 = DType("int64", 14, np.dtype(np.int64))
    uint64 = DType("uint64", 15, np.dtype(np.uint64))
    
    # Float dtypes (custom and standard)
    float4_e2m1fn = DType("float4_e2m1fn", 16, np.dtype(ml_dtypes.float4_e2m1fn))
    float8_e3m4 = DType("float8_e3m4", 17, np.dtype(ml_dtypes.float8_e3m4))
    float8_e4m3 = DType("float8_e4m3", 18, np.dtype(ml_dtypes.float8_e4m3))
    float8_e5m2 = DType("float8_e5m2", 19, np.dtype(ml_dtypes.float8_e5m2))
    float8_e4m3fn = DType("float8_e4m3fn", 20, np.dtype(ml_dtypes.float8_e4m3fn))
    float8_e8m0fnu = DType("float8_e8m0fnu", 21, np.dtype(ml_dtypes.float8_e8m0fnu))
    float8_e4m3b11fnuz = DType("float8_e4m3b11fnuz", 22, np.dtype(ml_dtypes.float8_e4m3b11fnuz))
    float8_e4m3fnuz = DType("float8_e4m3fnuz", 23, np.dtype(ml_dtypes.float8_e4m3fnuz))
    float8_e5m2fnuz = DType("float8_e5m2fnuz", 24, np.dtype(ml_dtypes.float8_e5m2fnuz))
    float16 = DType("float16", 25, np.dtype(np.float16))
    bfloat16 = DType("bfloat16", 26, np.dtype(ml_dtypes.bfloat16))
    float32 = DType("float32", 27, np.dtype(np.float32))
    float64 = DType("float64", 28, np.dtype(np.float64))
    
    # Complex dtypes
    complex64 = DType("complex64", 30, np.dtype(np.complex64))
    complex128 = DType("complex128", 32, np.dtype(np.complex128))
    
    # String dtype (if available in NumPy)
    string = DType("string", 40, np.dtypes.StringDType()) if hasattr(np.dtypes, 'StringDType') else None

    # Groupings
    sints = (int2, int4, int8, int16, int32, int64)
    uints = (uint2, uint4, uint8, uint16, uint32, uint64)
    ints = sints + uints
    
    fp4s = (float4_e2m1fn,)
    fp8s = (float8_e3m4, float8_e4m3, float8_e5m2, float8_e4m3fn, float8_e8m0fnu,
            float8_e4m3b11fnuz, float8_e4m3fnuz, float8_e5m2fnuz)
    floats = fp4s + fp8s + (float16, bfloat16, float32, float64)
    complexes = (complex64, complex128)
    
    string_types = (string,) if string is not None else ()
    custom_fps = fp4s + fp8s + (bfloat16,)
    custom_ints = (int2, int4, uint4, uint2)
   
    all = (bool,) + ints + floats + complexes
    
    # Default types (depend on x64_enabled)
    @staticmethod
    def _get_default_int():
        return dtypes.int64 if x64_enabled else dtypes.int32
    
    @staticmethod
    def _get_default_uint():
        return dtypes.uint64 if x64_enabled else dtypes.uint32
    
    @staticmethod
    def _get_default_float():
        return dtypes.float64 if x64_enabled else dtypes.float32
    
    @staticmethod
    def _get_default_complex():
        return dtypes.complex128 if x64_enabled else dtypes.complex64

    @classmethod
    def default_int(cls):
        return cls._get_default_int()
    
    @classmethod
    def default_uint(cls):
        return cls._get_default_uint()
    
    @classmethod
    def default_float(cls):
        return cls._get_default_float()
    
    @classmethod
    def default_complex(cls):
        return cls._get_default_complex()
    
    # Type checking methods
    @staticmethod
    def is_float(x: DType) -> bool:
        """Check if dtype is a floating-point type."""
        return 16 <= x.priority <= 28 and x in dtypes.floats
    
    @staticmethod
    def is_int(x: DType) -> bool:
        """Check if dtype is an integer type."""
        return 4 <= x.priority <= 15 and x in dtypes.ints
    
    @staticmethod
    def is_unsigned(x: DType) -> bool:
        """Check if dtype is unsigned."""
        return x in dtypes.uints
    
    @staticmethod
    def is_bool(x: DType) -> bool:
        """Check if dtype is boolean."""
        return x is dtypes.bool
    
    @staticmethod
    def is_string(x: DType) -> bool:
        """Check if dtype is a string type."""
        return x in dtypes.string_types

    @staticmethod
    def from_py(x):
        """Infer dtype from Python value."""
        if isinstance(x, bool):
            return dtypes.bool
        if isinstance(x, int):
            default_int = dtypes._get_default_int()
            info = int_info(default_int.native_type)
            if not (info.min <= x <= info.max):
                raise OverflowError(f"Python int {x} overflows default int type {default_int.name}")
            return default_int
        if isinstance(x, float):
            return dtypes._get_default_float()
        if isinstance(x, complex):
            return dtypes._get_default_complex()
        if isinstance(x, (list, tuple)):
            if not x:
                return dtypes._get_default_float()
            # Promote all element types
            return max((dtypes.from_py(e) for e in x), default=dtypes._get_default_float())
        raise TypeError(f"Could not infer dtype from value {x} of type {type(x)}")

# CONSTANT DECLARATIONS

# Dtypes that don't support infinity
_no_inf_types = {
    dtypes.float4_e2m1fn, dtypes.float8_e4m3fn, dtypes.float8_e4m3b11fnuz,
    dtypes.float8_e4m3fnuz, dtypes.float8_e5m2fnuz, dtypes.float8_e8m0fnu,
}

# 64-bit to 32-bit downcast map
_64_to_32_map = {
    dtypes.int64: dtypes.int32,
    dtypes.uint64: dtypes.uint32,
    dtypes.float64: dtypes.float32,
    dtypes.complex128: dtypes.complex64,
}

# Abstract type categories (for issubdtype)
_abstract_type_map = {
    np.generic: dtypes.all,
    np.number: dtypes.all,
    np.integer: dtypes.ints,
    np.signedinteger: dtypes.sints,
    np.unsignedinteger: dtypes.uints,
    np.floating: dtypes.floats,
    np.complexfloating: dtypes.complexes,
}

# Dtype kind strings for isdtype()
_dtype_kinds = {
    'bool': {dtypes.bool},
    'signed integer': set(dtypes.sints),
    'unsigned integer': set(dtypes.uints),
    'integral': set(dtypes.ints),
    'real floating': set(dtypes.floats),
    'complex floating': set(dtypes.complexes),
    'numeric': set(dtypes.ints + dtypes.floats + dtypes.complexes),
}

# Python scalar type to dtype mapping
_py_scalar_types = {
    bool: lambda: dtypes.bool,
    int: dtypes._get_default_int,
    float: dtypes._get_default_float,
    complex: dtypes._get_default_complex,
}

# Integer/bool to inexact dtype conversion
_change_to_inexact = {
    dtypes.bool: dtypes.float32,
    dtypes.uint8: dtypes.float32,
    dtypes.int8: dtypes.float32,
    dtypes.uint16: dtypes.float32,
    dtypes.int16: dtypes.float32,
    dtypes.uint32: dtypes.float32,
    dtypes.int32: dtypes.float32,
    dtypes.uint64: dtypes.float64,
    dtypes.int64: dtypes.float64,
    dtypes.int2: dtypes.float32,
    dtypes.int4: dtypes.float32,
    dtypes.uint2: dtypes.float32,
    dtypes.uint4: dtypes.float32,
}

# Build mapping from NumPy dtype to DType
_all_dtypes_for_map = [v for v in dtypes.__dict__.values() if isinstance(v, DType)]
_np_to_dtype_map = {
    v.native_type: v for v in _all_dtypes_for_map
    if hasattr(v.native_type, 'name')
}

# All valid xft dtypes
_xft_dtype_set = set(dtypes.all + dtypes.string_types + (dtypes.float0,))

# Weak type system
_weakTypes = [int, float, complex]
_registered_weakTypes = []

# CORE UTILITY FUNCTIONS

def _to_xft_dtype(x) -> DType:
    """Convert various inputs to a xft DType."""
    if isinstance(x, DType):
        return x
    try:
        np_dt = np.dtype(x)
        return _np_to_dtype_map[np_dt]
    except (TypeError, KeyError) as e:
        raise TypeError(f"Cannot convert value {x} to a valid xft DType.") from e

def _issubclass(a, b) -> bool:
    """Safe issubclass that returns False on TypeError."""
    try:
        return issubclass(a, b)
    except TypeError:
        return False

# DTYPE CONVERSIONS

def to_numeric(dtype_in) -> DType:
    """Convert bool to int, leave numeric types unchanged."""
    dt = _to_xft_dtype(dtype_in)
    return dtypes._get_default_int() if dt is dtypes.bool else dt

def to_inexact(dtype_in) -> DType:
    """Convert to floating-point or complex dtype."""
    dt = _to_xft_dtype(dtype_in)
    if dtypes.is_float(dt) or dt in dtypes.complexes:
        return dt
    return _change_to_inexact.get(dt, dt)

def to_complex(dtype_in) -> DType:
    """Convert to complex dtype."""
    float_equiv = to_inexact(dtype_in)
    if float_equiv is dtypes.float64:
        return dtypes.complex128
    return dtypes.complex64

# DTYPE INTROSPECTION

def supports_inf(dtype_in) -> bool:
    """Check if dtype supports infinity."""
    dt = _to_xft_dtype(dtype_in)
    return dt.supports_inf

def bit_width(dtype_in) -> int:
    """Get bit width of dtype."""
    dt = _to_xft_dtype(dtype_in)
    return dt.bit_width

def issubdtype(a, b) -> bool:
    """Check if 'a' is a subdtype of 'b' (NumPy-compatible API)."""
    return _issubtype_cached(a, b)

@functools.lru_cache(maxsize=512)
def _issubtype_cached(a, b) -> bool:
    """Cached subdtype checking."""
    if b in _abstract_type_map:
        try:
            a_dtype = _to_xft_dtype(a) if not isinstance(a, DType) else a
            return a_dtype in _abstract_type_map[b]
        except TypeError:
            return False
    
    # Both are concrete dtypes
    try:
        a_dtype = _to_xft_dtype(a) if not isinstance(a, DType) else a
        b_dtype = _to_xft_dtype(b) if not isinstance(b, DType) else b
        return a_dtype is b_dtype
    except TypeError:
        # Fallback to NumPy for edge cases
        return bool(np.issubdtype(a, b))

# DEFAULT DTYPE FUNCTIONS

def default_int_dtype() -> DType:
    """Get default integer dtype."""
    return dtypes._get_default_int()

def default_uint_dtype() -> DType:
    """Get default unsigned integer dtype."""
    return dtypes._get_default_uint()

def default_float_dtype() -> DType:
    """Get default floating-point dtype."""
    return dtypes._get_default_float()

def default_complex_dtype() -> DType:
    """Get default complex dtype."""
    return dtypes._get_default_complex()

def as_numpy_dtype(type_obj=None, align=False, copy=False):
    """Convert to NumPy dtype."""
    if type_obj is None:
        return dtypes._get_default_float().np_dtype
    if isinstance(type_obj, DType):
        return type_obj.np_dtype
    if isinstance(type_obj, type):
        if type_obj is bool:
            return dtypes.bool.np_dtype
        if type_obj is int:
            return dtypes._get_default_int().np_dtype
        if type_obj is float:
            return dtypes._get_default_float().np_dtype
        if type_obj is complex:
            return dtypes._get_default_complex().np_dtype
    return np.dtype(type_obj, align=align, copy=copy)

# DTYPE CANONICALIZATION & VALIDATION

@functools.lru_cache(maxsize=512)
def _canonicalize_dtype(x64_enabled_val, dtype_in) -> DType:
    """Canonicalize dtype based on x64 mode."""
    dt = _to_xft_dtype(dtype_in)
    if x64_enabled_val:
        return dt
    return _64_to_32_map.get(dt, dt)

def canonicalize_dtype(dtype_in) -> DType:
    """Canonicalize dtype based on current x64 setting."""
    return _canonicalize_dtype(x64_enabled, dtype_in)

def check_is_valid_dtype(dtype_in):
    """Verify dtype is valid for xft."""
    dt = _to_xft_dtype(dtype_in)
    if dt not in dtypes.all:
        raise TypeError(f"dtype '{dt.name}' is not a valid xft type")

def check_and_canonicalize_user_dtype(dtype_in, *, name: Optional[str] = None) -> DType:
    """Check and canonicalize user-provided dtype with warnings."""
    if dtype_in is None:
        msg = "dtype cannot be None."
        if name:
            msg += f" Please provide a value for '{name}'."
        raise ValueError(msg)
    
    try:
        original_dt = _to_xft_dtype(dtype_in)
    except TypeError:
        msg = f"'{dtype_in}' is not a valid xft dtype"
        if name:
            msg += f" for argument '{name}'"
        msg += "."
        raise TypeError(msg)
    
    canonical_dt = canonicalize_dtype(original_dt)
    
    if original_dt is not canonical_dt:
        name_str = f" for '{name}'" if name else ""
        msg = (f"User-provided dtype '{original_dt.name}'{name_str} is not available "
               f"in current x64 mode and will be truncated to '{canonical_dt.name}'.")
        warnings.warn(msg, stacklevel=2)
    
    return canonical_dt

# DTYPE INSPECTION & CLASSIFICATION

def isdtype(dtype_in, kind: Union[str, Tuple, DType]) -> bool:
    """Check if dtype matches kind(s) - array API compatible."""
    dt = _to_xft_dtype(dtype_in)
    kinds = (kind,) if not isinstance(kind, tuple) else kind
    
    for k in kinds:
        if isinstance(k, str):
            if k not in _dtype_kinds:
                raise ValueError(
                    f"Unrecognized kind: '{k}'. Valid kinds: {list(_dtype_kinds.keys())}"
                )
            if dt in _dtype_kinds[k]:
                return True
        else:
            try:
                k_dtype = _to_xft_dtype(k)
                if dt is k_dtype:
                    return True
            except TypeError:
                raise TypeError(f"Expected kind to be str or dtype, got '{k}'")
    
    return False

def is_string_dtype(dtype_in) -> bool:
    """Check if dtype is a string type."""
    dt = _to_xft_dtype(dtype_in)
    return dt in dtypes.string_types

def short_dtype_name(dtype_in) -> str:
    """Get short name for dtype (e.g., 'f32' for float32)."""
    dt = _to_xft_dtype(dtype_in)
    return dt.short_name

# MAIN DTYPE FUNCTION

def dtype(x, *, canonicalize: bool = False) -> DType:
    """Infer or convert to DType."""
    if x is None:
        raise ValueError("Cannot infer dtype from None.")

    dt = None
    
    # Handle DType directly
    if isinstance(x, DType):
        dt = x
    # Handle Python type classes
    elif isinstance(x, type) and x in _py_scalar_types:
        dt = _py_scalar_types[x]()
    # Handle Python scalar values
    elif type(x) in _py_scalar_types:
        dt = _py_scalar_types[type(x)]()
        # Overflow check for int literals
        if isinstance(x, int):
            info = int_info(dt.native_type)
            if not (info.min <= x <= info.max):
                raise OverflowError(
                    f"Python int {x} overflows default int type {dt.name}"
                )
    # Handle objects with dtype attribute
    elif hasattr(x, 'dtype'):
        dt = _to_xft_dtype(x.dtype)
    # Try NumPy's result_type as fallback
    else:
        try:
            np_dt = np.result_type(x)
            dt = _to_xft_dtype(np_dt)
        except (TypeError, KeyError) as e:
            raise TypeError(f"Cannot determine dtype of {x}") from e
    
    # Validate dtype is supported
    if dt not in _xft_dtype_set:
        raise TypeError(
            f"Value '{x}' with dtype '{dt.name}' is not a valid xft type."
        )
    
    return canonicalize_dtype(dt) if canonicalize else dt

# SCALAR TYPE HANDLING

def scalar_type(x) -> type:
    """Get Python scalar type for a value."""
    dt = dtype(x)

    if dtypes.is_int(dt):
        return int
    if dtypes.is_float(dt):
        return float
    if dtypes.is_bool(dt):
        return bool
    if dt in dtypes.complexes:
        return complex
    
    raise TypeError(
        f"Value '{x}' with dtype '{dt.name}' has no Python scalar type."
    )

def is_scalar_type(x) -> bool:
    """Check if x is a Python scalar type."""
    return type(x) in _py_scalar_types

def ensure_array(x, dtype_in=None):
    """Convert to NumPy array with optional dtype."""
    np_dtype = None
    if dtype_in is not None:
        dt = _to_xft_dtype(dtype_in)
        np_dtype = dt.np_dtype
    return np.asarray(x, dtype=np_dtype)

# WEAK TYPE SYSTEM

def _to_lattice_type(dtype_in, weak_type: bool):
    """Convert dtype to lattice node (weak types become Python types)."""
    dt = _to_xft_dtype(dtype_in) if not isinstance(dtype_in, DType) else dtype_in
    
    if weak_type:
        if dt is dtypes.bool:
            return bool
        if dtypes.is_int(dt):
            return int
        if dtypes.is_float(dt):
            return float
        if dt in dtypes.complexes:
            return complex
    
    return dt

def _infer_dtype_and_weak(value) -> Tuple[DType, bool]:
    """Infer dtype and weak status from a value."""
    dt = dtype(value)
    is_weak = (type(value) in _weakTypes or type(value) in _registered_weakTypes)
    return dt, is_weak

def register_weak_type(typ: type):
    """Register a custom weak type."""
    if typ not in _registered_weakTypes:
        _registered_weakTypes.append(typ)

def is_weakly_typed(x) -> bool:
    """Check if value is weakly typed."""
    return type(x) in _weakTypes or type(x) in _registered_weakTypes

# NUMPY DTYPE INTEROP

def dtype_to_numpy(dtype_in):
    """Convert xft DType to NumPy dtype."""
    dt = _to_xft_dtype(dtype_in)
    if dtypes.is_string(dt):
        raise TypeError("Modern string DType has no direct np.dtype equivalent")
    return dt.native_type

def numpy_to_dtype(np_dtype) -> DType:
    """Convert NumPy dtype to xft DType."""
    try:
        return _np_to_dtype_map[np_dtype]
    except KeyError as e:
        raise TypeError(
            f"NumPy dtype {np_dtype} is not a supported xft DType."
        ) from e

# TYPE PROMOTION LATTICE

class TypePromotionError(TypeError):
    """Raised when type promotion fails."""
    pass

def _type_promotion_lattice(method: str) -> dict:
    """Define promotion lattice as DAG.
    
    Standard mode: allows int→float promotion
    Strict mode: no implicit int→float
    """
    if method not in ('standard', 'strict'):
        raise ValueError(
            f"Invalid method '{method}'. Expected 'standard' or 'strict'."
        )
    
    b = dtypes.bool
    u2, u4, u8, u16, u32, u64 = (
        dtypes.uint2, dtypes.uint4, dtypes.uint8,
        dtypes.uint16, dtypes.uint32, dtypes.uint64
    )
    i2, i4, i8, i16, i32, i64 = (
        dtypes.int2, dtypes.int4, dtypes.int8,
        dtypes.int16, dtypes.int32, dtypes.int64
    )
    
    custom_floats = list(dtypes.custom_fps)
    bf, f16, f32, f64 = dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64
    c64, c128 = dtypes.complex64, dtypes.complex128
    
    # Weak types
    weak_int, weak_float, weak_complex = int, float, complex
    
    if method == "standard":
        # NumPy-compatible: generous promotion including int→float
        return {
            b: [weak_int],
            weak_int: [u2, u4, u8, i2, i4, i8],
            u2: [], u4: [],
            i2: [], i4: [],
            u8: [i16, u16],
            i8: [i16],
            u16: [i32, u32],
            i16: [i32],
            u32: [i64, u64],
            i32: [i64],
            u64: [weak_float],
            i64: [weak_float],
            weak_float: [*custom_floats, bf, f16, weak_complex],
            **{t: [] for t in custom_floats},
            bf: [f32],
            f16: [f32],
            f32: [f64, c64],
            f64: [c128],
            weak_complex: [c64],
            c64: [c128],
            c128: [],
        }
    else:  # strict
        all_ints = list(dtypes.ints)
        all_floats = list(dtypes.floats)
        all_complex = list(dtypes.complexes)
        
        return {
            dtypes.bool: [weak_int],
            weak_int: [weak_float] + all_ints,
            weak_float: [weak_complex] + all_floats,
            weak_complex: all_complex,
            **{t: [] for t in all_ints + all_floats + all_complex}
        }

def _make_lattice_upper_bounds(method: str) -> dict:
    """Precompute all reachable ancestors for each node."""
    lattice = _type_promotion_lattice(method)
    upper_bounds = {node: {node} for node in lattice}

    for node in lattice:
        while True:
            new_bounds = set()
            for ancestor in upper_bounds[node]:
                new_bounds.update(lattice[ancestor])
            
            if node in new_bounds:
                raise ValueError(f"Cycle detected in promotion lattice at node {node}")
            
            if new_bounds.issubset(upper_bounds[node]):
                break
            
            upper_bounds[node] |= new_bounds
    
    return upper_bounds

# Precompute upper bounds
_lattice_upper_bounds = {
    "standard": _make_lattice_upper_bounds("standard"),
    "strict": _make_lattice_upper_bounds("strict")
}

def _least_upper_bound(method: str, *nodes):
    """Find the least upper bound (smallest common ancestor)."""
    node_set = set(nodes)
    upper_bounds_map = _lattice_upper_bounds[method]
    
    try:
        all_bounds = [upper_bounds_map[n] for n in node_set]
    except KeyError as e:
        invalid = [n for n in node_set if n not in upper_bounds_map]
        raise ValueError(f"Invalid dtypes for promotion: {invalid}") from e
    
    common_upper_bounds = set.intersection(*all_bounds)
    
    least_upper = (common_upper_bounds & node_set) or {
        c for c in common_upper_bounds
        if common_upper_bounds.issubset(upper_bounds_map[c])
    }
    
    if len(least_upper) == 1:
        return least_upper.pop()
    elif len(least_upper) == 0:
        node_names = tuple(
            n.name if isinstance(n, DType) else str(n) for n in nodes
        )
        
        if method == "strict":
            msg = (
                f"No implicit promotion for {node_names} in strict mode. "
                f"Use explicit casting or switch to 'standard' mode."
            )
        else:
            msg = f"No implicit promotion path for {node_names}. Use explicit casting."
        
        raise TypePromotionError(msg)
    else:
        raise TypePromotionError(
            f"Ambiguous promotion for {nodes}: multiple least upper bounds {least_upper}"
        )

def promote_types(a, b) -> DType:
    """Promote two dtypes to their common type (binary promotion).
    
    JAX-compatible API for pairwise type promotion.
    """
    method = _DTYPE_PROMOTION_MODE
    
    # Convert to lattice nodes
    a_node = a if a in _weakTypes else _to_xft_dtype(a)
    b_node = b if b in _weakTypes else _to_xft_dtype(b)
    
    result = _least_upper_bound(method, a_node, b_node)
    
    # Convert weak types to default dtypes
    if result is int:
        return dtypes._get_default_int()
    elif result is float:
        return dtypes._get_default_float()
    elif result is complex:
        return dtypes._get_default_complex()
    
    return result

def _lattice_result_type(*args) -> Tuple[DType, bool]:
    """Core promotion logic with weak type handling.
    
    Returns (promoted_dtype, is_weak) tuple.
    """
    method = _DTYPE_PROMOTION_MODE
    
    dtypes_and_weak = [_infer_dtype_and_weak(arg) for arg in args]
    arg_dtypes, arg_weaks = zip(*dtypes_and_weak)
    
    # Single argument
    if len(arg_dtypes) == 1:
        return arg_dtypes[0], arg_weaks[0]
    
    # All same dtype with at least one strong type
    if len(set(arg_dtypes)) == 1 and not all(arg_weaks):
        return arg_dtypes[0], False
    
    # All weak types in standard mode
    if all(arg_weaks) and method != "strict":
        lattice_nodes = [_to_lattice_type(dt, weak=False) for dt in arg_dtypes]
        result_node = _least_upper_bound(method, *lattice_nodes)
        
        if result_node is int:
            result_dtype = dtypes._get_default_int()
        elif result_node is float:
            result_dtype = dtypes._get_default_float()
        elif result_node is complex:
            result_dtype = dtypes._get_default_complex()
        elif result_node is bool:
            result_dtype = dtypes.bool
        else:
            result_dtype = result_node
        
        return result_dtype, True
    
    # Mixed weak/strong or general case
    lattice_nodes = [
        _to_lattice_type(dt, w) for dt, w in zip(arg_dtypes, arg_weaks)
    ]
    result_node = _least_upper_bound(method, *lattice_nodes)
    
    result_is_weak = result_node in _weakTypes
    
    # Convert result node to DType
    if result_node is int:
        result_dtype = dtypes._get_default_int()
    elif result_node is float:
        result_dtype = dtypes._get_default_float()
    elif result_node is complex:
        result_dtype = dtypes._get_default_complex()
    elif result_node is bool:
        result_dtype = dtypes.bool
    else:
        result_dtype = result_node
    
    # Don't mark bool results as weak
    if result_dtype is dtypes.bool:
        result_is_weak = False
    
    return result_dtype, result_is_weak

def result_type(*args, return_weak_type: bool = False):
    """Determine result dtype for n-ary operations.
    
    This is the main public promotion API.
    """
    if len(args) == 0:
        raise ValueError("result_type() requires at least one argument")
    
    result_dtype, is_weak = _lattice_result_type(*args)
    
    # Canonicalize the result
    result_dtype = canonicalize_dtype(result_dtype)
    
    return (result_dtype, is_weak) if return_weak_type else result_dtype

def can_cast_safely(from_dtype, to_dtype) -> bool:
    """Check if casting is safe (no precision/range loss)."""
    from_dt = canonicalize_dtype(from_dtype)
    to_dt = canonicalize_dtype(to_dtype)
    
    if from_dt is to_dt:
        return True
    
    # Check if promotion yields target type
    promoted = promote_types(from_dt, to_dt)
    return promoted is to_dt

# CONFIGURATION FUNCTIONS

def set_promotion_mode(mode: str):
    """Set the type promotion mode.
    
    Args:
        mode: Either 'standard' (NumPy-compatible, allows int→float) or
              'strict' (no implicit int→float promotion)
    """
    global _DTYPE_PROMOTION_MODE
    
    if mode not in ("standard", "strict"):
        raise ValueError(
            f"Invalid promotion mode '{mode}'. Expected 'standard' or 'strict'."
        )
    
    _DTYPE_PROMOTION_MODE = mode

def get_promotion_mode() -> str:
    """Get the current type promotion mode."""
    return _DTYPE_PROMOTION_MODE

def set_x64_enabled(enabled: bool):
    """Enable or disable 64-bit mode.
    
    When disabled, 64-bit types are automatically downcast to 32-bit.
    """
    global x64_enabled
    x64_enabled = enabled

def get_x64_enabled() -> bool:
    """Check if 64-bit mode is enabled."""
    return x64_enabled

# CONVENIENCE FUNCTIONS

def get_dtype_info(dtype_in) -> dict:
    """Get comprehensive information about a dtype."""
    dt = _to_xft_dtype(dtype_in)
    
    info = {
        'name': dt.name,
        'short_name': dt.short_name,
        'priority': dt.priority,
        'itemsize': dt.itemsize,
    }
    
    if dt is not dtypes.bool and dtypes.is_int(dt):
        info['min'] = dt.min
        info['max'] = dt.max
        info['bit_width'] = dt.bit_width
        info['is_signed'] = not dtypes.is_unsigned(dt)
    elif dtypes.is_float(dt):
        info['bit_width'] = dt.bit_width
        info['supports_inf'] = dt.supports_inf
    elif dt in dtypes.complexes:
        info['bit_width'] = dt.bit_width
        info['supports_inf'] = dt.supports_inf
    
    return info

def compatible_dtypes(*dtypes_in) -> bool:
    """Check if multiple dtypes can be promoted together."""
    try:
        result_type(*dtypes_in)
        return True
    except TypePromotionError:
        return False