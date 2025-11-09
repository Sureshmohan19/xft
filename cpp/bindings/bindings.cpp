#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "xft/array.h"
#include "xft/scalar_types.h"

namespace py = pybind11;

// ============================================================================
// Helper Functions
// ============================================================================

// Convert Python list/tuple/int to std::vector<size_t> for shape
// Why?
// Because the shape is a vector of size_t, and we need to convert the Python object to a vector of size_t.
// Possible Python objects: int, tuple, list. (No other objects are allowed as it does not make sense.)
//
std::vector<size_t> to_shape_vector(py::object obj) {
    // we are initializing the result vector to an empty vector of size_t.
    std::vector<size_t> result;
    // we are checking if the object is a tuple.
    // we are iterating over the tuple and converting each item to a size_t.
    // we are pushing the item to the result vector.
    if (py::isinstance<py::tuple>(obj)) {
        auto tup = obj.cast<py::tuple>();
        for (auto item : tup) {
            result.push_back(item.cast<size_t>());
        }
    // we are checking if the object is a list.
    // we are iterating over the list and converting each item to a size_t.
    // we are pushing the item to the result vector.
    } else if (py::isinstance<py::list>(obj)) {
        auto list = obj.cast<py::list>();
        for (auto item : list) {
            result.push_back(item.cast<size_t>());
        }
    // we are checking if the object is a single integer.
    // we are pushing the integer to the result vector. No loop needed.
    } else if (py::isinstance<py::int_>(obj)) {
        result.push_back(obj.cast<size_t>());
    // we are checking if the object is not a tuple, list, or integer.
    // we are throwing an invalid argument error.
    } else {
        throw std::invalid_argument("Shape must be int, tuple, or list");
    }
    // we are returning the result vector.
    return result;
}

// Convert NumPy dtype to XFT ScalarType
// Currently, we only support Float32 and Float64.
//
xft::ScalarType numpy_dtype_to_xft(const py::dtype& dt) {
    // we are checking if the dtype is a float.
    // we are returning the Float32 enum value.
    if (dt.is(py::dtype::of<float>())) {
        return xft::ScalarType::Float32;
    // we are checking if the dtype is a double.
    // we are returning the Float64 enum value.
    } else if (dt.is(py::dtype::of<double>())) {
        return xft::ScalarType::Float64;
    }
    // we are checking if the dtype is not a float or double.
    // we are throwing an invalid argument error.
    throw std::invalid_argument("Unsupported NumPy dtype: " + std::string(py::str(py::handle(dt))));
}

// ============================================================================
// XFT Core Module - Python Bindings
// ============================================================================

PYBIND11_MODULE(_core, m) {
    m.doc() = "XFT - simple deep learning framework ❤️ (Core Module)";
    
    // Version Info
    m.attr("__version__") = "0.0.1";
    m.attr("__author__") = "Suresh Neethimohan";

    // DType Enum (Python name for ScalarType)
    // In future (next version), we will add more data types and make separate DType enum.
    //
    // Why?
    // Because ScalarType is internal and users never interact with it directly,
    // it makes sense to keep it internal and have a separate DType enum.
    //
    py::enum_<xft::ScalarType>(m, "DType", "Data type enumeration")
        .value("Float32", xft::ScalarType::Float32, "32-bit floating point")
        .value("Float64", xft::ScalarType::Float64, "64-bit floating point")
        .export_values();

    // Array::Order Enum
    //
    // Why?
    // Because we need to specify the memory layout of the array.
    // Possible values: C (row-major) and F (column-major).
    //
    py::enum_<xft::Array::Order>(m, "Order", "Memory layout order")
        .value("C", xft::Array::Order::C, "Row-major (C-style)")
        .value("F", xft::Array::Order::F, "Column-major (Fortran-style)")
        .export_values();

    // Array Class
    // It is the main class that users will interact with to create and manipulate arrays 
    // (or tensors if you want to think of it that way).
    //
    // Why?
    // Because we need to create a multi-dimensional array with NumPy-like interface.
    // Supports zero-copy interop with NumPy via buffer protocol.
    //
    py::class_<xft::Array, std::shared_ptr<xft::Array>>(m, "Array", py::buffer_protocol(), 
        "Multi-dimensional array with NumPy-compatible interface")
        
        // Constructor: Wrap NumPy array (zero-copy view)
        //
        // Why is this first?
        // pybind11 checks constructors in order. Since py::array is more specific than
        // py::object, we need this before the shape-based constructor to avoid ambiguity.
        // Otherwise, NumPy arrays would be incorrectly interpreted as shape tuples.
        //
        // What does this do?
        // Creates an XFT array that shares memory with a NumPy array (zero-copy).
        // This enables seamless interoperability between XFT and the NumPy ecosystem
        // (PyTorch, TensorFlow, scikit-learn, etc.) without data duplication.
        //
        // How does it work?
        // 1. Extract buffer info from NumPy via Python buffer protocol (ptr, shape, strides)
        // 2. Convert NumPy's dtype to XFT's ScalarType (Float32/Float64)
        // 3. Wrap the raw memory pointer using Storage::wrap (no ownership transfer)
        // 4. Create XFT Array using the wrapped storage
        //
        // Memory lifetime:
        // The NumPy array must stay alive as long as the XFT array exists.
        // Python's garbage collector handles this automatically via reference counting.
        // Storage::wrap uses a no-op deleter since we don't own the memory (NumPy owns it).
        //
        // Example:
        //   np_arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        //   xft_arr = xft.Array(np_arr)  # Zero-copy wrap
        //   np_arr[0, 0] = 999           # Modifies xft_arr too!
        //
        .def(py::init([](py::array arr) {
            // Request buffer info from NumPy array (via Python buffer protocol)
            // This gives us: data pointer, shape, strides, dtype, itemsize
            py::buffer_info info = arr.request();
            
            // Convert NumPy dtype to XFT ScalarType
            // Currently supports: np.float32 -> Float32, np.float64 -> Float64
            xft::ScalarType dtype = numpy_dtype_to_xft(arr.dtype());
            
            // Extract shape from buffer info
            // NumPy stores shape as vector<ssize_t>, we convert to vector<size_t>
            std::vector<size_t> shape(info.shape.begin(), info.shape.end());
            
            // Extract strides from buffer info
            // NumPy strides are in bytes (distance between elements in memory)
            // Example: For shape (3, 4) with float32, strides = (16, 4)
            std::vector<size_t> strides(info.strides.begin(), info.strides.end());
            
            // Wrap the NumPy buffer without taking ownership
            // Storage::wrap creates a reference-counted storage that points to
            // NumPy's memory but doesn't free it (uses no-op deleter)
            auto storage = xft::Storage::wrap(
                info.ptr,                  // Raw data pointer from NumPy
                info.size * info.itemsize, // Total bytes (num_elements * bytes_per_element)
                dtype                      // Element type
            );
            
            // Create XFT Array using the wrapped storage
            // The Array now shares memory with the NumPy array
            // Modifications through either interface affect both
            // The one thing we cant do is delete the NumPy array. 
            // We can only delete the XFT array beause we own it.
            return std::make_shared<xft::Array>(storage, shape, strides, dtype);
        }),
        py::arg("numpy_array"),
        "Wrap NumPy array as XFT array (zero-copy view)")
        
        // Constructor: Create new array from shape
        //
        // What does this do?
        // Allocates a new array with the specified shape, dtype, and memory layout.
        // The array is zero-initialized (uninitialized memory, contents undefined).
        //
        // Parameters:
        // - shape: int, tuple, or list specifying dimensions (e.g., (3, 4) for 3x4 array)
        // - dtype: ScalarType (Float32 or Float64), defaults to Float32
        // - order: Memory layout (C=row-major, F=column-major), defaults to C
        //
        // Example:
        //   arr = xft.Array([3, 4], dtype=xft.DType.Float32, order=xft.Order.C)
        //
        .def(py::init([](py::object shape, xft::ScalarType dtype, xft::Array::Order order) {
            // Convert Python shape (int/tuple/list) to vector<size_t>
            // Then create new Array with allocated storage
            return std::make_shared<xft::Array>(to_shape_vector(shape), dtype, order);
        }), 
        py::arg("shape"), 
        py::arg("dtype") = xft::ScalarType::Float32,    // Defaults to Float32
        py::arg("order") = xft::Array::Order::C,        // Defaults to C (row-major)
        "Create array with given shape and dtype")
        
        // Properties
        //
        // Why?
        // Because we need to get the properties of the array.
        // Possible properties: ndim, shape, strides, size, nbytes, dtype, itemsize, data, offset.
        //
        // ndim: Number of dimensions
        // Example usage: print(arr.ndim)  # Output: 2
        .def_property_readonly("ndim", &xft::Array::ndim, "Number of dimensions")
        
        // shape: Tuple of dimension sizes
        // Example usage: print(arr.shape)  # Output: (3, 4)
        .def_property_readonly("shape", 
            static_cast<const std::vector<size_t>& (xft::Array::*)() const>(&xft::Array::shape),
            "Tuple of dimension sizes")
        
        // strides: Tuple of strides in bytes
        // NumPy strides are in bytes (distance between elements in memory)
        // So do we.
        // Example: For shape (3, 4) with float32, strides = (16, 4)
        // How does it work?
        //
        //   shape[0] = 3   → number of rows
        //   shape[1] = 4   → number of columns
        //   itemsize = 4   → each float32 element occupies 4 bytes
        //
        // In row-major (C-style) order, array elements are stored row by row in memory.
        // That means all elements of the first row come first, followed by all elements of
        // the second row, and so on.
        //
        // Memory layout:
        //
        //   (0,0) (0,1) (0,2) (0,3)
        //   (1,0) (1,1) (1,2) (1,3)
        //   (2,0) (2,1) (2,2) (2,3)
        //
        // Each step along the last axis (columns) moves forward by 4 bytes
        //   → stride for axis 1 = 4
        //
        // Each step along the first axis (rows) skips over one entire row of 4 elements,
        // each 4 bytes long → 4 * 4 = 16 bytes
        //   → stride for axis 0 = 16
        //
        // Therefore, strides = (16, 4)
        // meaning:
        //   - Move 16 bytes to go from one row to the next
        //   - Move 4 bytes to go from one column to the next
        //
        // General rule for row-major layout:
        //   strides[i] = itemsize * product(shape[j] for j > i)
        //
        // Example verification:
        //   strides[1] = 4
        //   strides[0] = 4 * shape[1] = 4 * 4 = 16
        //
        .def_property_readonly("strides", 
            static_cast<const std::vector<size_t>& (xft::Array::*)() const>(&xft::Array::strides),
            "Tuple of strides in bytes")
            
        // size: Total number of elements
        // Example usage: print(arr.size)  # Output: 12
        .def_property_readonly("size", &xft::Array::size, "Total number of elements")
        
        // nbytes: Total bytes consumed by array elements
        // Example usage: print(arr.nbytes)  # Output: 48
        .def_property_readonly("nbytes", &xft::Array::nbytes, "Total bytes consumed by array elements")
        
        // dtype: Data type of array elements
        // Example usage: print(arr.dtype)  # Output: xft.DType.Float32
        .def_property_readonly("dtype", &xft::Array::dtype, "Data type of array elements")
        
        // itemsize: Size in bytes of each element
        // Example usage: print(arr.itemsize)  # Output: 4
        .def_property_readonly("itemsize", &xft::Array::itemsize, "Size in bytes of each element")
        
        // data: Memory address of array data (as integer)
        // Example usage: print(arr.data)  # Output: 0x10000000
        .def_property_readonly("data", [](xft::Array& self) {
            return reinterpret_cast<uintptr_t>(self.data());
        }, "Memory address of array data (as integer)")

        // offset: Byte offset from storage base pointer
        // Example usage: print(arr.offset)  # Output: 0
        .def_property_readonly("offset", &xft::Array::offset, "Byte offset from storage base pointer")
        

        // Methods
        //
        // is_contiguous: Check if array is C-contiguous (row-major)
        // Example usage: print(arr.is_contiguous())  # Output: True
        .def("is_contiguous", &xft::Array::is_contiguous, "Check if array is C-contiguous (row-major)")
        
        // is_f_contiguous: Check if array is Fortran-contiguous (column-major)
        // Example usage: print(arr.is_f_contiguous())  # Output: False
        .def("is_f_contiguous", &xft::Array::is_f_contiguous, "Check if array is Fortran-contiguous (column-major)")
        
        // ============================================================================
        // Element Access Bindings
        // ============================================================================
        //
        // These bindings expose typed element access methods for the `xft::Array` class.
        // Each function wraps the templated `Array::at<T>()` method for specific dtypes.
        //
        // Supported data types:
        //   - Float32  → float
        //   - Float64  → double
        //
        // Supported dimensions:
        //   - 1D : arr[i]
        //   - 2D : arr[i, j]
        //
        // Notes:
        //   - These functions are used internally by the Python dispatch layer
        //     (ArrayDispatch in xft/dispatch.py).
        //   - The Python wrapper exposes them via `__getitem__` / `__setitem__`.
        //   - No slicing or higher-dimensional access is implemented yet.
        //
        // Example (Python):
        //   >>> arr = xft.Array([3, 4], dtype=xft.DType.Float32)
        //   >>> arr[0, 1] = 1.23
        //   >>> print(arr[0, 1])   # 1.23
        // ============================================================================

        // -----------------------------------------------------------------------------
        // Float32 element access
        // -----------------------------------------------------------------------------
        .def("get_float32", [](xft::Array& self, size_t i0) {
            return self.at<float>(i0);
        }, py::arg("i0"), "Get element at 1D index (float32)")

        .def("set_float32", [](xft::Array& self, size_t i0, float value) {
            self.at<float>(i0) = value;
        }, py::arg("i0"), py::arg("value"), "Set element at 1D index (float32)")

        .def("get_float32_2d", [](xft::Array& self, size_t i0, size_t i1) {
            return self.at<float>(i0, i1);
        }, py::arg("i0"), py::arg("i1"), "Get element at 2D index (float32)")

        .def("set_float32_2d", [](xft::Array& self, size_t i0, size_t i1, float value) {
            self.at<float>(i0, i1) = value;
        }, py::arg("i0"), py::arg("i1"), py::arg("value"), "Set element at 2D index (float32)")

        // -----------------------------------------------------------------------------
        // Float64 element access
        // -----------------------------------------------------------------------------
        .def("get_float64", [](xft::Array& self, size_t i0) {
            return self.at<double>(i0);
        }, py::arg("i0"), "Get element at 1D index (float64)")

        .def("set_float64", [](xft::Array& self, size_t i0, double value) {
            self.at<double>(i0) = value;
        }, py::arg("i0"), py::arg("value"), "Set element at 1D index (float64)")

        .def("get_float64_2d", [](xft::Array& self, size_t i0, size_t i1) {
            return self.at<double>(i0, i1);
        }, py::arg("i0"), py::arg("i1"), "Get element at 2D index (float64)")

        .def("set_float64_2d", [](xft::Array& self, size_t i0, size_t i1, double value) {
            self.at<double>(i0, i1) = value;
        }, py::arg("i0"), py::arg("i1"), py::arg("value"), "Set element at 2D index (float64)")

        // to_string: Return internal string representation of the array
        // Example usage: print(arr.to_string())  # Output: Array(shape=[3, 4], dtype=float32, strides=[16, 4], contiguous=true)
        .def("to_string", &xft::Array::to_string, "Return internal string representation of the array")
        
        // ============================================================================
        // Python Buffer Protocol
        // ============================================================================
        //
        // Enables zero-copy interoperability between xft::Array and NumPy.
        //
        // This binding allows users to create NumPy views directly from an xft::Array
        // instance via:
        //     >>> np.asarray(xft_array)
        // or
        //     >>> np.array(xft_array, copy=False)
        //
        // The buffer protocol exposes the array's underlying memory layout — including
        // pointer, element size, shape, strides, and dtype — to Python.
        //
        // Implementation details:
        //   • Uses pybind11’s `def_buffer()` to register a `py::buffer_info` struct.
        //   • Translates the internal `xft::ScalarType` enum into a Python
        //     struct-style format string (e.g., 'f' for float32, 'd' for float64).
        //   • Converts C++ vectors for shape and strides into `std::vector<ssize_t>`
        //     as required by the buffer protocol.
        //   • Provides **zero-copy** access: NumPy views share the same memory, so
        //     modifications in either side reflect immediately in the other.
        //
        // Supported dtypes:
        //   - Float32  → `py::format_descriptor<float>`
        //   - Float64  → `py::format_descriptor<double>`
        //
        // Raises:
        //   std::runtime_error if dtype is unsupported.
        //
        // Example (Python):
        //   >>> import numpy as np, xft
        //   >>> arr = xft.Array([3, 4], dtype=xft.DType.Float32)
        //   >>> np_view = np.asarray(arr)   # shares memory with arr
        //   >>> np_view[:] = 42
        //   >>> print(arr[0, 0])            # 42.0
        // ============================================================================
        .def_buffer([](xft::Array& self) -> py::buffer_info {
            // Convert XFT dtype → Python format string
            std::string format;
            switch (self.dtype()) {
                // Float32
                case xft::ScalarType::Float32:
                    format = py::format_descriptor<float>::format();
                    break;
                // Float64
                case xft::ScalarType::Float64:
                    format = py::format_descriptor<double>::format();
                    break;
                // TODO: Add more dtypes here when needed,
                // Unsupported dtype
                default:
                    throw std::runtime_error("Unsupported dtype for buffer protocol");
            }

            // Convert shape and strides to ssize_t vectors
            std::vector<ssize_t> shape(self.shape().begin(), self.shape().end());
            std::vector<ssize_t> strides(self.strides().begin(), self.strides().end());

            // Construct and return the pybind11 buffer descriptor
            return py::buffer_info(
                self.data(),        // Pointer to underlying data
                self.itemsize(),    // Size (in bytes) of one element
                format,             // Python struct-style format string
                self.ndim(),        // Number of dimensions
                shape,              // Shape of each dimension
                strides             // Strides (in bytes) for each dimension
            );
        });


    // ============================================================================
    // Module-Level Utilities
    // ============================================================================
    // Lightweight helpers that complement xft::Array.
    // They’re accessible directly under the Python module (e.g., xft.to_numpy()).
    //
    // Convert XFT Array to NumPy array (zero-copy via buffer protocol)
    m.def("to_numpy", [](xft::Array& arr) -> py::array {
        // Use buffer protocol - np.asarray will call __buffer__
        return py::array(py::cast(arr));
    }, py::arg("array"), "Convert XFT array to NumPy array (zero-copy view)");
    
    // Get dtype name as string
    m.def("dtype_name", [](xft::ScalarType dtype) {
        return std::string(xft::scalarTypeName(dtype));
    }, py::arg("dtype"), "Get string name of data type");
    
    // Get dtype size in bytes
    m.def("dtype_size", [](xft::ScalarType dtype) {
        return xft::scalarTypeSize(dtype);
    }, py::arg("dtype"), "Get size in bytes of data type");

    // Internal note: 
    // Scalar class (cpp/xft/scalar.h and cpp/xft/scalar_types.h) is kept internal.
    // Users never interact with it directly - they use Array instead.
}