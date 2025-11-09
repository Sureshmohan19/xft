"""
XFT Array - Python wrapper over C++ Array class.
Provides NumPy-like interface for deep learning operations.
"""

from typing import Union, Tuple, List, Optional

import numpy as np

from xft._core import (
    Array as _ArrayCore,
    DType,
    Order,
    to_numpy,
    dtype_name,
)

class ArrayDispatch:
    """
    Centralized dispatch table for element-wise access.

    This class maps combinations of data type (`DType`) and dimensionality (`ndim`)
    to the correct low-level C++ getter/setter functions in the `_core` module.

    Why this exists:
    ----------------
    Instead of branching logic all over `Array.__getitem__` and `Array.__setitem__`,
    we centralize the mapping once in this dispatch layer.
    That makes the array indexing logic simple, scalable, and type-safe.

    Example
    -------
    >>> dispatch = ArrayDispatch(core)
    >>> getter = dispatch.get_fn(DType.Float32, 2)
    >>> value = getter(0, 1)
    >>> setter = dispatch.set_fn(DType.Float32, 2)
    >>> setter(0, 1, 3.14)

    Design
    ------
    - Uses a lookup table: (dtype, ndim) → (get_fn, set_fn)
    - Currently supports:
        * Float32, Float64
        * 1D and 2D arrays
    - Each function directly calls into the C++ binding defined in `_core.cpp`.

    TODO:
    -----
    1. Add support for higher-dimensional arrays (3D, ND)
       → Extend the C++ side with `get_float32_3d`, `set_float32_3d`, etc.
    2. Generalize with dynamic indexing:
       → Possibly replace hardcoded table with a generic C++ dispatcher or
         a `core.at<float>(indices)` interface.
    3. Extend to support other data types (e.g., Int32, Bool)
       once new dtypes are added to the backend.

    """

    def __init__(self, core):
        """Initialize with reference to underlying C++ core module."""
        self._core = core

        # ----------------------------------------------------------------------
        # (DType, ndim) → (getter, setter)
        # ----------------------------------------------------------------------
        # Each entry pairs a dtype and dimensionality with its corresponding
        # low-level element access functions from the `_core` binding.
        # For example:
        #   (DType.Float32, 1) → (core.get_float32, core.set_float32)
        #   (DType.Float64, 2) → (core.get_float64_2d, core.set_float64_2d)
        #
        # Note:
        # These are function *references*, not function calls.
        # The dispatch layer simply returns the correct callable to the caller.
        #
        self._table = {
            (DType.Float32, 1): (core.get_float32, core.set_float32),
            (DType.Float64, 1): (core.get_float64, core.set_float64),
            (DType.Float32, 2): (core.get_float32_2d, core.set_float32_2d),
            (DType.Float64, 2): (core.get_float64_2d, core.set_float64_2d),
        }

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def get_fn(self, dtype, ndim):
        """
        Retrieve the getter function for the given (dtype, ndim).

        Parameters
        ----------
        dtype : DType
            Data type of the array (e.g., DType.Float32)
        ndim : int
            Number of dimensions (1 or 2 currently supported)

        Returns
        -------
        Callable
            The appropriate getter function bound from `_core`.

        Raises
        ------
        TypeError
            If the dtype-ndim combination is unsupported.
        """
        try:
            return self._table[(dtype, ndim)][0]
        except KeyError:
            raise TypeError(f"Unsupported combination: dtype={dtype}, ndim={ndim}")

    def set_fn(self, dtype, ndim):
        """
        Retrieve the setter function for the given (dtype, ndim).

        Parameters
        ----------
        dtype : DType
            Data type of the array (e.g., DType.Float64)
        ndim : int
            Number of dimensions (1 or 2 currently supported)

        Returns
        -------
        Callable
            The appropriate setter function bound from `_core`.

        Raises
        ------
        TypeError
            If the dtype–ndim combination is unsupported.
        """
        try:
            return self._table[(dtype, ndim)][1]
        except KeyError:
            raise TypeError(f"Unsupported combination: dtype={dtype}, ndim={ndim}")


class Array:
    """
    Multi-dimensional array for deep learning operations.

    Wraps the C++ `xft::Array` class with a Pythonic interface.

    Key features:
    --------------
    • NumPy-like semantics (shape, strides, dtype, etc.)
    • Zero-copy interoperability with NumPy via Python's buffer protocol
    • Fast element access bridged through C++ bindings (`_core`)

    Examples
    --------
    >>> import xft
    >>> # Create a new empty array
    >>> arr = xft.Array([3, 4], dtype=xft.DType.Float32)
    >>> print(arr.shape)  # (3, 4)

    >>> # Wrap an existing NumPy array (zero-copy)
    >>> np_arr = np.random.randn(10, 20).astype(np.float32)
    >>> xft_arr = xft.Array(np_arr)

    >>> # Convert back to NumPy (shares memory)
    >>> back = xft_arr.numpy()
    >>> back is np_arr  # False, but shares memory
    True
    """

    def __init__(self, data, dtype: DType = DType.Float32, order: Order = Order.C):
        """
        Initialize an XFT Array.

        Parameters
        ----------
        data : Union[np.ndarray, list, tuple, int, _ArrayCore]
            Input defining the array:
              • NumPy ndarray → wraps existing data (zero-copy)
              • list / tuple / int → allocates new array of given shape
              • _ArrayCore → wraps an existing C++ Array (internal use only)
        dtype : DType, optional
            Element type. Defaults to `Float32`.
        order : Order, optional
            Memory layout (C=row-major, F=column-major). Defaults to `Order.C`.

        Raises
        ------
        TypeError
            If the input type is unsupported.
        ValueError
            If the shape or dtype is invalid.
        """

        # ----------------------------------------------------------------------
        # Case 1: Wrap existing NumPy array (zero-copy)
        # ----------------------------------------------------------------------
        if isinstance(data, np.ndarray):
            # The C++ constructor uses NumPy's buffer info to create
            # an `xft::Array` that shares memory with the NumPy array.
            # No data is copied; both views point to the same memory block.
            self.__core = _ArrayCore(data)

        # ----------------------------------------------------------------------
        # Case 2: Wrap an existing C++ core Array
        # ----------------------------------------------------------------------
        elif isinstance(data, _ArrayCore):
            # Internal use case — for converting back from C++ arrays
            # without allocating new memory.
            self.__core = data

        # ----------------------------------------------------------------------
        # Case 3: Allocate new array from shape
        # ----------------------------------------------------------------------
        elif isinstance(data, (list, tuple, int)):
            # Allocates a new array of given shape (filled with undefined data)
            # The actual memory allocation happens on the C++ side.
            self.__core = _ArrayCore(data, dtype, order)

        # ----------------------------------------------------------------------
        # Invalid input type
        # ----------------------------------------------------------------------
        else:
            raise TypeError(f"Cannot create Array from type {type(data)}")

        # TODO:
        # 1. Add support for Python scalars (e.g., xft.Array(3.14))
        #    → would create a 0-D array
        # 2. Validate shape contents (no negative dimensions)
        # 3. Possibly infer dtype automatically from NumPy arrays in the future
    
    # ----------------------------------------------------------------------
    # Array Metadata and Layout Accessors
    # ----------------------------------------------------------------------
    # These properties provide NumPy-like introspection and memory layout
    # information. All values are retrieved from the underlying C++ object.
    # ----------------------------------------------------------------------

    @property
    def _core(self) -> _ArrayCore:
        """Internal: underlying C++ Array object (not part of public API)."""
        return self.__core

    # ----------------------------------------------------------------------
    # Basic shape and dimensionality
    # ----------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the array as a tuple of dimension sizes.
        
        Example:
            >>> arr.shape
            (3, 4)
        """
        return tuple(self._core.shape)

    @property
    def strides(self) -> Tuple[int, ...]:
        """
        Memory strides (in bytes) for each dimension.
        
        Defines how many bytes to move in memory when advancing one step
        along each axis. For a C-contiguous (row-major) Float32 array
        of shape (3, 4), strides = (16, 4).
        """
        return tuple(self._core.strides)

    @property
    def ndim(self) -> int:
        """
        Number of array dimensions (rank).
        
        Example:
            >>> arr.ndim
            2
        """
        return self._core.ndim

    @property
    def size(self) -> int:
        """
        Total number of elements in the array.
        
        Equal to the product of all shape dimensions.
        """
        return self._core.size


    # ----------------------------------------------------------------------
    # Memory and data type information
    # ----------------------------------------------------------------------

    @property
    def nbytes(self) -> int:
        """
        Total number of bytes occupied by the array data.
        
        Computed as: size * itemsize
        """
        return self._core.nbytes

    @property
    def dtype(self) -> DType:
        """
        Data type of array elements (e.g., Float32, Float64).
        
        Example:
            >>> arr.dtype
            DType.Float32
        """
        return self._core.dtype

    @property
    def itemsize(self) -> int:
        """
        Size in bytes of a single element.
        
        Example:
            >>> arr.itemsize
            4  # for Float32
        """
        return self._core.itemsize


    # ----------------------------------------------------------------------
    # Memory layout checks
    # ----------------------------------------------------------------------

    def is_contiguous(self) -> bool:
        """
        Check if array is C-contiguous (row-major layout).
        
        Returns True if the last dimension varies fastest in memory.
        """
        return self._core.is_contiguous()

    def is_f_contiguous(self) -> bool:
        """
        Check if array is Fortran-contiguous (column-major layout).
        
        Returns True if the first dimension varies fastest in memory.
        """
        return self._core.is_f_contiguous()


    # ----------------------------------------------------------------------
    # NumPy Interoperability
    # ----------------------------------------------------------------------

    def numpy(self) -> np.ndarray:
        """
        Convert to a NumPy array (zero-copy view).
        
        The returned NumPy array shares the same underlying memory buffer
        with this XFT Array — no data is copied. Modifications to either
        reflect in the other.
        
        Example:
            >>> np_view = arr.numpy()
            >>> np_view[0, 0] = 42
            >>> arr[0, 0]  # 42.0
        """
        return to_numpy(self._core)

    
    # ----------------------------------------------------------------------
    # Element Access and Assignment
    # ----------------------------------------------------------------------
    # Provides Pythonic indexing (arr[i], arr[i, j]) by dispatching to the
    # correct C++ getter/setter based on dtype and number of dimensions.
    #
    # Supports only 1D and 2D arrays at this stage.
    # TODO: Extend to support slicing and N-D access in future.
    # ----------------------------------------------------------------------

    def __getitem__(self, key):
        """
        Retrieve element(s) from the array (1D or 2D only).

        Dispatches to the appropriate C++ getter function based on
        data type (`dtype`) and dimensionality (`ndim`).

        Args:
            key (int | tuple): Index or index tuple, e.g. `0` or `(1, 2)`.

        Returns:
            float | double: The value at the specified position.

        Raises:
            TypeError: If the key is not an int or tuple.
            IndexError: If the number of indices does not match `ndim`.

        Example:
            >>> arr = xft.Array([3, 4], dtype=xft.DType.Float32)
            >>> arr[0, 1] = 2.5
            >>> print(arr[0, 1])
            2.5
        """
        # Normalize key to tuple form for uniform handling.
        if isinstance(key, int):
            key = (key,)
        elif not isinstance(key, tuple):
            raise TypeError(f"Unsupported index type: {type(key)}")

        # Ensure index dimensionality matches array dimensionality.
        if len(key) != self.ndim:
            raise IndexError(f"Expected {self.ndim} indices, got {len(key)}")

        # Resolve the correct C++ getter via dtype/ndim dispatch table.
        dispatch = ArrayDispatch(self._core)
        get_fn = dispatch.get_fn(self.dtype, self.ndim)
        return get_fn(*key)


    def __setitem__(self, key, value):
        """
        Assign a value to an element (1D or 2D only).

        Dispatches to the correct C++ setter function based on dtype
        and dimensionality. Automatically converts Python scalars to float.

        Args:
            key (int | tuple): Index or index tuple, e.g. `0` or `(1, 2)`.
            value (float | int): Value to assign.

        Raises:
            TypeError: If the key is not an int or tuple.
            IndexError: If the number of indices does not match `ndim`.

        Example:
            >>> arr = xft.Array([3], dtype=xft.DType.Float64)
            >>> arr[0] = 3.1415
            >>> print(arr[0])
            3.1415
        """
        # Normalize key to tuple form.
        if isinstance(key, int):
            key = (key,)
        elif not isinstance(key, tuple):
            raise TypeError(f"Unsupported index type: {type(key)}")

        # Dimensionality check.
        if len(key) != self.ndim:
            raise IndexError(f"Expected {self.ndim} indices, got {len(key)}")

        # Resolve setter and perform type-safe write.
        dispatch = ArrayDispatch(self._core)
        set_fn = dispatch.set_fn(self.dtype, self.ndim)
        set_fn(*key, float(value))

    
    def __repr__(self) -> str:
        """
        Developer-friendly representation.
        
        Returns a compact string including class name, shape, dtype, and ndim.
        
        Example:
            >>> xft.Array([2, 3], dtype=xft.DType.Float32)
            xft.Array(shape=(2, 3), dtype=float32, ndim=2)
        """
        return f"xft.Array(shape={self.shape}, dtype={dtype_name(self.dtype)}, ndim={self.ndim})"


    def __str__(self) -> str:
        """
        Human-readable array printout (NumPy-like formatting).

        Displays contents and metadata for 1D and 2D arrays.
        For large arrays, truncates middle elements for readability.

        Example:
            >>> arr = xft.Array([2, 3], dtype=xft.DType.Float32)
            >>> arr[0, 0] = 1.0
            >>> arr[0, 1] = 2.0
            >>> arr[0, 2] = 3.0
            >>> arr[1, 0] = 4.0
            >>> arr[1, 1] = 5.0
            >>> arr[1, 2] = 6.0
            >>> print(arr)
            xft.Array(
            [[1.00, 2.00, 3.00],
            [4.00, 5.00, 6.00]],
            dtype=float32, shape=(2, 3)
            )
        """
        dispatch = ArrayDispatch(self._core)
        get_fn = dispatch.get_fn(self.dtype, self.ndim)

        # Handle 1D arrays
        if self.ndim == 1:
            elems = [f"{get_fn(i):.2f}" for i in range(self.shape[0])]
            s = _truncate(elems)
            return f"xft.Array([{s}], dtype={dtype_name(self.dtype)}, shape={self.shape})"

        # Handle 2D arrays
        elif self.ndim == 2:
            rows = []
            for i in range(self.shape[0]):
                row_vals = [f"{get_fn(i, j):.2f}" for j in range(self.shape[1])]
                rows.append(f"[{_truncate(row_vals)}]")
            inner = ",\n   ".join(rows)
            return (
                f"xft.Array(\n"
                f"  [{inner}],\n"
                f"  dtype={dtype_name(self.dtype)}, shape={self.shape}\n"
                f")"
            )

        # Fallback for higher dimensions (not yet supported)
        else:
            return f"xft.Array(ndim={self.ndim}, dtype={dtype_name(self.dtype)}, shape={self.shape})"


# Internal: Truncation helper
# ----------------------------------------------------------------------
def _truncate(items, max_elems: int = 6) -> str:
    """
    Helper to truncate long lists for pretty printing.
    Example:
        _truncate(['1', '2', '3', '4', '5', '6', '7']) -> '1, 2, 3, ..., 6, 7'
    """
    n = len(items)
    if n <= max_elems:
        return ", ".join(items)
    head = ", ".join(items[:3])
    tail = ", ".join(items[-2:])
    return f"{head}, ..., {tail}"


def array(data, dtype: DType = DType.Float32, order: Order = Order.C) -> Array:
    """
    Create an XFT Array from Python data.

    Convenience function similar to `numpy.array()`.
    Automatically converts Python lists or tuples into NumPy arrays
    (to obtain contiguous, typed memory) before wrapping as an XFT array.

    Args:
        data: NumPy array, shape tuple, or nested Python lists.
        dtype: Element data type (Float32 or Float64).
        order: Memory layout order (C=row-major, F=column-major).

    Returns:
        XFT Array instance.

    Examples:
        >>> arr = xft.array([1, 2, 3], dtype=xft.DType.Float32)
        >>> arr2 = xft.array([[1, 2], [3, 4]])
    """
    # ----------------------------------------------------------------------
    # NOTE:
    # Python lists and tuples are not contiguous numeric arrays —
    # they are lists of Python objects in scattered memory.
    #
    # To create an XFT Array from nested lists (e.g., [[1,2],[3,4]]),
    # we first convert them to a NumPy ndarray. NumPy provides:
    #   - Contiguous memory layout (C/F order)
    #   - Known dtype and element size
    #   - Shape and stride information
    #
    # This ensures that our C++ backend receives a valid data buffer.
    #
    # TODO (Future):
    # Implement a custom Python list → C++ array converter
    # to remove this NumPy dependency while preserving
    # zero-copy and layout correctness.
    # ----------------------------------------------------------------------
    if isinstance(data, (list, tuple)) and not isinstance(data[0], (int, float)):
        # Nested list or tuple → convert to NumPy array
        np_arr = np.array(data, dtype='float32' if dtype == DType.Float32 else 'float64')
        return Array(np_arr)

    # Directly pass shape tuple, NumPy array, or int to constructor
    return Array(data, dtype, order)


def from_numpy(np_array: np.ndarray) -> Array:
    """
    Wrap an existing NumPy array as an XFT Array (zero-copy).

    The returned XFT Array shares memory with the NumPy array.
    Any modifications in one reflect in the other.

    Args:
        np_array: NumPy ndarray (float32 or float64).

    Returns:
        XFT Array sharing memory with the given NumPy array.

    Examples:
        >>> np_arr = np.random.randn(10, 20).astype(np.float32)
        >>> xft_arr = xft.from_numpy(np_arr)
        >>> xft_arr[0, 0] = 42  # Also updates np_arr[0, 0]
    """
    # Directly wrap NumPy ndarray using zero-copy buffer protocol.
    # The C++ layer extracts the pointer, shape, and strides,
    # and creates an xft::Array view over the same memory.
    return Array(np_array)
