#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// ============================================================================
// XFT Core Module - Python Bindings
// ============================================================================
// This module will expose Array operations to Python users.
// Scalar is kept internal - users never interact with it directly.
//
// Build output: xft_core.so (or xft_core.pyd on Windows)
// Usage: import xft_core

PYBIND11_MODULE(xft_core, m) {
    m.doc() = "XFT - Deep Learning Framework (Core Module)";
    
    // ========================================================================
    // Version Info
    // ========================================================================
    m.attr("__version__") = "0.1.0";
    m.attr("__author__") = "XFT Team";
    
    // ========================================================================
    // Future: Array class will be exposed here
    // ========================================================================
    // py::class_<xft::Array>(m, "Array")
    //     .def(py::init<std::vector<float>, std::vector<size_t>>())
    //     .def("__add__", ...)
    //     ...
}

// ============================================================================
// Notes for Future Development
// ============================================================================
// When we add Array class, we'll:
//   1. Keep ScalarType enum exposed (users need it for dtype queries)
//   2. Remove Scalar class exposure (internal only)
//   3. Add Array class with these bindings:
//
// py::class_<xft::Array>(m, "Array")
//     .def(py::init<std::vector<float>, std::vector<size_t>>())
//     .def("__add__", [](const Array& self, float value) {
//         return self.add(xft::Scalar(value));  // Scalar used internally
//     })
//     .def("__add__", [](const Array& self, const Array& other) {
//         return self.add(other);
//     });
//
// Users will write: arr + 5.0  (Python float)
// Internally: Converted to Scalar(5.0f) automatically
// Users never see Scalar class!