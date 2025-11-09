// ============================================================================
// XFT Array Module - C++ Unit Tests
// ============================================================================
// Tests multi-dimensional arrays, strides, element access, and layout.

#include "xft/array.h"
#include "xft/scalar_types.h"
#include <iostream>
#include <cmath>

// ============================================================================
// Test Macros
// ============================================================================

#define TEST_ASSERT(condition, message)                                        \
    if (!(condition)) {                                                        \
        std::cerr << "    ✗ FAILED: " << message << std::endl;                \
        std::cerr << "      at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false;                                                          \
    }

// ============================================================================
// Test Functions
// ============================================================================

// Test basic 1D array creation
bool test_1d_array_creation() {
    xft::Array arr({10}, xft::ScalarType::Float32);
    
    TEST_ASSERT(arr.ndim() == 1, "ndim should be 1");
    TEST_ASSERT(arr.size() == 10, "size should be 10");
    TEST_ASSERT(arr.shape()[0] == 10, "shape[0] should be 10");
    TEST_ASSERT(arr.dtype() == xft::ScalarType::Float32, "dtype should be Float32");
    TEST_ASSERT(arr.itemsize() == 4, "itemsize should be 4 for Float32");
    TEST_ASSERT(arr.nbytes() == 40, "nbytes should be 40");
    TEST_ASSERT(arr.is_contiguous(), "1D array should be contiguous");
    TEST_ASSERT(arr.data() != nullptr, "data pointer should be non-null");
    
    return true;
}

// Test 2D array creation (C-order)
bool test_2d_array_c_order() {
    xft::Array arr({3, 4}, xft::ScalarType::Float32, xft::Array::Order::C);
    
    TEST_ASSERT(arr.ndim() == 2, "ndim should be 2");
    TEST_ASSERT(arr.size() == 12, "size should be 12");
    TEST_ASSERT(arr.shape()[0] == 3, "shape[0] should be 3");
    TEST_ASSERT(arr.shape()[1] == 4, "shape[1] should be 4");
    TEST_ASSERT(arr.is_contiguous(), "C-order should be contiguous");
    TEST_ASSERT(!arr.is_f_contiguous(), "C-order should not be F-contiguous");
    
    // Check strides (C-order: rightmost varies fastest)
    // stride[1] = 4 bytes (one float)
    // stride[0] = 16 bytes (4 floats)
    TEST_ASSERT(arr.strides()[1] == 4, "stride[1] should be 4");
    TEST_ASSERT(arr.strides()[0] == 16, "stride[0] should be 16");
    
    return true;
}

// Test 2D array creation (F-order)
bool test_2d_array_f_order() {
    xft::Array arr({3, 4}, xft::ScalarType::Float32, xft::Array::Order::F);
    
    TEST_ASSERT(arr.ndim() == 2, "ndim should be 2");
    TEST_ASSERT(arr.size() == 12, "size should be 12");
    TEST_ASSERT(!arr.is_contiguous(), "F-order should not be C-contiguous");
    TEST_ASSERT(arr.is_f_contiguous(), "F-order should be F-contiguous");
    
    // Check strides (F-order: leftmost varies fastest)
    // stride[0] = 4 bytes (one float)
    // stride[1] = 12 bytes (3 floats)
    TEST_ASSERT(arr.strides()[0] == 4, "stride[0] should be 4");
    TEST_ASSERT(arr.strides()[1] == 12, "stride[1] should be 12");
    
    return true;
}

// Test 3D array creation
bool test_3d_array_creation() {
    xft::Array arr({2, 3, 4}, xft::ScalarType::Float32);
    
    TEST_ASSERT(arr.ndim() == 3, "ndim should be 3");
    TEST_ASSERT(arr.size() == 24, "size should be 24");
    TEST_ASSERT(arr.shape()[0] == 2, "shape[0] should be 2");
    TEST_ASSERT(arr.shape()[1] == 3, "shape[1] should be 3");
    TEST_ASSERT(arr.shape()[2] == 4, "shape[2] should be 4");
    TEST_ASSERT(arr.is_contiguous(), "3D C-order should be contiguous");
    
    // C-order strides for shape (2, 3, 4)
    // stride[2] = 4 bytes
    // stride[1] = 16 bytes (4 floats)
    // stride[0] = 48 bytes (12 floats)
    TEST_ASSERT(arr.strides()[2] == 4, "stride[2] should be 4");
    TEST_ASSERT(arr.strides()[1] == 16, "stride[1] should be 16");
    TEST_ASSERT(arr.strides()[0] == 48, "stride[0] should be 48");
    
    return true;
}

// Test 1D element access
bool test_1d_element_access() {
    xft::Array arr({10}, xft::ScalarType::Float32);
    
    // Write
    arr.at<float>(0) = 1.0f;
    arr.at<float>(5) = 42.0f;
    arr.at<float>(9) = 99.0f;
    
    // Read
    TEST_ASSERT(arr.at<float>(0) == 1.0f, "arr[0] should be 1.0");
    TEST_ASSERT(arr.at<float>(5) == 42.0f, "arr[5] should be 42.0");
    TEST_ASSERT(arr.at<float>(9) == 99.0f, "arr[9] should be 99.0");
    
    return true;
}

// Test 2D element access (C-order)
bool test_2d_element_access_c_order() {
    xft::Array arr({3, 4}, xft::ScalarType::Float32, xft::Array::Order::C);
    
    // Write in C-order pattern
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            arr.at<float>(i, j) = static_cast<float>(i * 10 + j);
        }
    }
    
    // Read and verify
    TEST_ASSERT(arr.at<float>(0, 0) == 0.0f, "arr[0,0] == 0");
    TEST_ASSERT(arr.at<float>(0, 3) == 3.0f, "arr[0,3] == 3");
    TEST_ASSERT(arr.at<float>(1, 0) == 10.0f, "arr[1,0] == 10");
    TEST_ASSERT(arr.at<float>(1, 2) == 12.0f, "arr[1,2] == 12");
    TEST_ASSERT(arr.at<float>(2, 3) == 23.0f, "arr[2,3] == 23");
    
    return true;
}

// Test 2D element access (F-order)
bool test_2d_element_access_f_order() {
    xft::Array arr({3, 4}, xft::ScalarType::Float32, xft::Array::Order::F);
    
    // Write
    arr.at<float>(0, 0) = 100.0f;
    arr.at<float>(1, 2) = 200.0f;
    arr.at<float>(2, 3) = 300.0f;
    
    // Read
    TEST_ASSERT(arr.at<float>(0, 0) == 100.0f, "arr[0,0] == 100");
    TEST_ASSERT(arr.at<float>(1, 2) == 200.0f, "arr[1,2] == 200");
    TEST_ASSERT(arr.at<float>(2, 3) == 300.0f, "arr[2,3] == 300");
    
    return true;
}

// Test 3D element access
bool test_3d_element_access() {
    xft::Array arr({2, 3, 4}, xft::ScalarType::Float32);
    
    // Write
    arr.at<float>(0, 0, 0) = 1.0f;
    arr.at<float>(0, 1, 2) = 2.0f;
    arr.at<float>(1, 2, 3) = 3.0f;
    
    // Read
    TEST_ASSERT(arr.at<float>(0, 0, 0) == 1.0f, "arr[0,0,0] == 1");
    TEST_ASSERT(arr.at<float>(0, 1, 2) == 2.0f, "arr[0,1,2] == 2");
    TEST_ASSERT(arr.at<float>(1, 2, 3) == 3.0f, "arr[1,2,3] == 3");
    
    return true;
}

// Test N-D element access (general)
bool test_nd_element_access() {
    xft::Array arr({2, 3, 4}, xft::ScalarType::Float32);
    
    // Write using vector indices
    arr.at<float>({0, 0, 0}) = 10.0f;
    arr.at<float>({1, 1, 1}) = 20.0f;
    arr.at<float>({1, 2, 3}) = 30.0f;
    
    // Read
    TEST_ASSERT(arr.at<float>({0, 0, 0}) == 10.0f, "arr[0,0,0] == 10");
    TEST_ASSERT(arr.at<float>({1, 1, 1}) == 20.0f, "arr[1,1,1] == 20");
    TEST_ASSERT(arr.at<float>({1, 2, 3}) == 30.0f, "arr[1,2,3] == 30");
    
    return true;
}

// Test Float64 (double) dtype
bool test_float64_dtype() {
    xft::Array arr({5}, xft::ScalarType::Float64);
    
    TEST_ASSERT(arr.dtype() == xft::ScalarType::Float64, "dtype should be Float64");
    TEST_ASSERT(arr.itemsize() == 8, "itemsize should be 8 for Float64");
    TEST_ASSERT(arr.nbytes() == 40, "nbytes should be 40");
    
    // Write and read doubles
    arr.at<double>(0) = 3.141592653589793;
    arr.at<double>(4) = 2.718281828459045;
    
    TEST_ASSERT(arr.at<double>(0) == 3.141592653589793, "double precision write/read");
    TEST_ASSERT(arr.at<double>(4) == 2.718281828459045, "double precision write/read");
    
    return true;
}

// Test out of bounds access should throw
bool test_out_of_bounds() {
    xft::Array arr({3, 4}, xft::ScalarType::Float32);
    
    // Valid access should work
    arr.at<float>(2, 3) = 1.0f;
    
    // Out of bounds should throw
    try {
        arr.at<float>(3, 0); // Row index out of bounds
        TEST_ASSERT(false, "should throw out_of_range");
    } catch (const std::out_of_range&) {
        // Expected
    }
    
    try {
        arr.at<float>(0, 4); // Column index out of bounds
        TEST_ASSERT(false, "should throw out_of_range");
    } catch (const std::out_of_range&) {
        // Expected
    }
    
    return true;
}

// Test wrong dtype access should throw
bool test_wrong_dtype_access() {
    xft::Array arr({10}, xft::ScalarType::Float32);
    
    // Correct type should work
    arr.at<float>(0) = 1.0f;
    
    // Wrong type should throw
    try {
        arr.at<double>(0) = 1.0; // Array is Float32, not Float64
        TEST_ASSERT(false, "should throw invalid_argument for type mismatch");
    } catch (const std::invalid_argument&) {
        // Expected
    }
    
    return true;
}

// Test wrong dimension access should throw
bool test_wrong_dimension_access() {
    xft::Array arr({3, 4}, xft::ScalarType::Float32);
    
    // 2D access should work
    arr.at<float>(1, 2) = 1.0f;
    
    // 1D access on 2D array should throw
    try {
        arr.at<float>(5);
        TEST_ASSERT(false, "1D access on 2D array should throw");
    } catch (const std::invalid_argument&) {
        // Expected
    }
    
    // 3D access on 2D array should throw
    try {
        arr.at<float>(1, 2, 3);
        TEST_ASSERT(false, "3D access on 2D array should throw");
    } catch (const std::invalid_argument&) {
        // Expected
    }
    
    return true;
}

// Test calculate_offset
bool test_calculate_offset() {
    xft::Array arr({3, 4}, xft::ScalarType::Float32, xft::Array::Order::C);
    
    // C-order strides: [16, 4]
    // offset(0, 0) = 0*16 + 0*4 = 0
    TEST_ASSERT(arr.calculate_offset({0, 0}) == 0, "offset(0,0) == 0");
    
    // offset(0, 1) = 0*16 + 1*4 = 4
    TEST_ASSERT(arr.calculate_offset({0, 1}) == 4, "offset(0,1) == 4");
    
    // offset(1, 0) = 1*16 + 0*4 = 16
    TEST_ASSERT(arr.calculate_offset({1, 0}) == 16, "offset(1,0) == 16");
    
    // offset(2, 3) = 2*16 + 3*4 = 44
    TEST_ASSERT(arr.calculate_offset({2, 3}) == 44, "offset(2,3) == 44");
    
    return true;
}

// Test wrapping external memory
bool test_wrap_external_memory() {
    float external_data[12];
    for (int i = 0; i < 12; ++i) {
        external_data[i] = static_cast<float>(i);
    }
    
    // Wrap as 3x4 array
    xft::Array arr(external_data, {3, 4}, xft::ScalarType::Float32);
    
    TEST_ASSERT(arr.ndim() == 2, "ndim should be 2");
    TEST_ASSERT(arr.size() == 12, "size should be 12");
    TEST_ASSERT(arr.data() == external_data, "should point to external data");
    
    // Read through array
    TEST_ASSERT(arr.at<float>(0, 0) == 0.0f, "arr[0,0] == 0");
    TEST_ASSERT(arr.at<float>(1, 2) == 6.0f, "arr[1,2] == 6");
    TEST_ASSERT(arr.at<float>(2, 3) == 11.0f, "arr[2,3] == 11");
    
    // Modify through array
    arr.at<float>(0, 0) = 99.0f;
    TEST_ASSERT(external_data[0] == 99.0f, "should modify external data");
    
    return true;
}

// Test wrapping with custom strides
bool test_wrap_with_custom_strides() {
    float external_data[12];
    for (int i = 0; i < 12; ++i) {
        external_data[i] = static_cast<float>(i);
    }
    
    // Wrap with F-order strides
    // F-order for (3, 4): strides = [4, 12]
    xft::Array arr(external_data, {3, 4}, xft::ScalarType::Float32, {4, 12});
    
    TEST_ASSERT(arr.strides()[0] == 4, "stride[0] should be 4");
    TEST_ASSERT(arr.strides()[1] == 12, "stride[1] should be 12");
    TEST_ASSERT(arr.is_f_contiguous(), "should be F-contiguous");
    
    return true;
}

// Test empty shape should throw
bool test_empty_shape() {
    try {
        xft::Array arr({}, xft::ScalarType::Float32);
        TEST_ASSERT(false, "empty shape should throw");
    } catch (const std::invalid_argument&) {
        // Expected
    }
    return true;
}

// Test zero dimension should throw
bool test_zero_dimension() {
    try {
        xft::Array arr({3, 0, 4}, xft::ScalarType::Float32);
        TEST_ASSERT(false, "zero dimension should throw");
    } catch (const std::invalid_argument&) {
        // Expected
    }
    return true;
}

// Test to_string representation
bool test_to_string() {
    xft::Array arr({2, 3, 4}, xft::ScalarType::Float32);
    std::string repr = arr.to_string();
    
    // Should contain key information
    TEST_ASSERT(repr.find("Array") != std::string::npos, "should contain 'Array'");
    TEST_ASSERT(repr.find("shape") != std::string::npos, "should contain 'shape'");
    TEST_ASSERT(repr.find("dtype") != std::string::npos, "should contain 'dtype'");
    TEST_ASSERT(repr.find("float32") != std::string::npos, "should contain 'float32'");
    TEST_ASSERT(repr.find("contiguous") != std::string::npos, "should contain 'contiguous'");
    
    return true;
}

// Test storage sharing between arrays
// Note: use_count() tracks Storage objects sharing data, not shared_ptr<Storage> instances
bool test_storage_sharing() {
    xft::Array arr1({10}, xft::ScalarType::Float32);
    
    // Get storage (creates a local shared_ptr to the SAME Storage object)
    auto storage = arr1.storage();
    // use_count is still 1 because all shared_ptrs point to the SAME Storage object
    TEST_ASSERT(storage->use_count() == 1, "use_count == 1 (same Storage object)");
    
    // Create another array sharing same storage
    xft::Array arr2(storage, {10}, {4}, xft::ScalarType::Float32, 0);
    // use_count is still 1: arr1, arr2, and storage all reference the SAME Storage object
    TEST_ASSERT(storage->use_count() == 1, "use_count still 1 (same Storage object)");
    
    // Both arrays should point to same data
    TEST_ASSERT(arr1.data() == arr2.data(), "should share data");
    TEST_ASSERT(arr1.storage().get() == arr2.storage().get(), "same Storage object");
    
    // Modifications through one should affect the other
    arr1.at<float>(0) = 42.0f;
    TEST_ASSERT(arr2.at<float>(0) == 42.0f, "should see same data");
    
    return true;
}

// Test large array
bool test_large_array() {
    // Create 1M element array (4 MB)
    xft::Array arr({1000000}, xft::ScalarType::Float32);
    
    TEST_ASSERT(arr.size() == 1000000, "size should be 1M");
    TEST_ASSERT(arr.nbytes() == 4000000, "nbytes should be 4M");
    
    // Test boundary access
    arr.at<float>(0) = 1.0f;
    arr.at<float>(999999) = 2.0f;
    
    TEST_ASSERT(arr.at<float>(0) == 1.0f, "should access first element");
    TEST_ASSERT(arr.at<float>(999999) == 2.0f, "should access last element");
    
    return true;
}

// Test contiguity detection for various layouts
bool test_contiguity_detection() {
    // C-order should be C-contiguous
    xft::Array arr_c({3, 4}, xft::ScalarType::Float32, xft::Array::Order::C);
    TEST_ASSERT(arr_c.is_contiguous(), "C-order should be C-contiguous");
    TEST_ASSERT(!arr_c.is_f_contiguous(), "C-order should not be F-contiguous");
    
    // F-order should be F-contiguous
    xft::Array arr_f({3, 4}, xft::ScalarType::Float32, xft::Array::Order::F);
    TEST_ASSERT(!arr_f.is_contiguous(), "F-order should not be C-contiguous");
    TEST_ASSERT(arr_f.is_f_contiguous(), "F-order should be F-contiguous");
    
    // 1D arrays are both C and F contiguous
    xft::Array arr_1d({10}, xft::ScalarType::Float32);
    TEST_ASSERT(arr_1d.is_contiguous(), "1D should be C-contiguous");
    TEST_ASSERT(arr_1d.is_f_contiguous(), "1D should be F-contiguous");
    
    return true;
}

// Test offset parameter in storage constructor
bool test_array_with_offset() {
    auto storage = xft::Storage::create(1024, xft::ScalarType::Float32);
    
    // Fill storage with pattern
    float* data = static_cast<float*>(storage->data());
    for (int i = 0; i < 256; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Create array with offset (skip first 10 elements = 40 bytes)
    xft::Array arr(storage, {5}, {4}, xft::ScalarType::Float32, 40);
    
    TEST_ASSERT(arr.offset() == 40, "offset should be 40");
    TEST_ASSERT(arr.size() == 5, "size should be 5");
    
    // arr[0] should point to storage[10]
    TEST_ASSERT(arr.at<float>(0) == 10.0f, "arr[0] should be 10");
    TEST_ASSERT(arr.at<float>(4) == 14.0f, "arr[4] should be 14");
    
    return true;
}

// Test shape access by axis
bool test_shape_access_by_axis() {
    xft::Array arr({2, 3, 4}, xft::ScalarType::Float32);
    
    TEST_ASSERT(arr.shape(0) == 2, "shape(0) == 2");
    TEST_ASSERT(arr.shape(1) == 3, "shape(1) == 3");
    TEST_ASSERT(arr.shape(2) == 4, "shape(2) == 4");
    
    // Out of bounds should throw
    try {
        arr.shape(3);
        TEST_ASSERT(false, "shape(3) should throw");
    } catch (const std::out_of_range&) {
        // Expected
    }
    
    return true;
}

// ============================================================================
// Test Runner
// ============================================================================

struct TestCase {
    const char* name;
    bool (*func)();
};

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "XFT Array C++ Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    TestCase tests[] = {
        {"test_1d_array_creation", test_1d_array_creation},
        {"test_2d_array_c_order", test_2d_array_c_order},
        {"test_2d_array_f_order", test_2d_array_f_order},
        {"test_3d_array_creation", test_3d_array_creation},
        {"test_1d_element_access", test_1d_element_access},
        {"test_2d_element_access_c_order", test_2d_element_access_c_order},
        {"test_2d_element_access_f_order", test_2d_element_access_f_order},
        {"test_3d_element_access", test_3d_element_access},
        {"test_nd_element_access", test_nd_element_access},
        {"test_float64_dtype", test_float64_dtype},
        {"test_out_of_bounds", test_out_of_bounds},
        {"test_wrong_dtype_access", test_wrong_dtype_access},
        {"test_wrong_dimension_access", test_wrong_dimension_access},
        {"test_calculate_offset", test_calculate_offset},
        {"test_wrap_external_memory", test_wrap_external_memory},
        {"test_wrap_with_custom_strides", test_wrap_with_custom_strides},
        {"test_empty_shape", test_empty_shape},
        {"test_zero_dimension", test_zero_dimension},
        {"test_to_string", test_to_string},
        {"test_storage_sharing", test_storage_sharing},
        {"test_large_array", test_large_array},
        {"test_contiguity_detection", test_contiguity_detection},
        {"test_array_with_offset", test_array_with_offset},
        {"test_shape_access_by_axis", test_shape_access_by_axis},
    };
    
    int passed = 0;
    int failed = 0;
    
    for (const auto& test : tests) {
        std::cout << "Running " << test.name << "..." << std::endl;
        if (test.func()) {
            std::cout << "    ✓ PASSED" << std::endl;
            passed++;
        } else {
            failed++;
        }
    }
    
    std::cout << "========================================" << std::endl;
    if (failed == 0) {
        std::cout << "✓ ALL TESTS PASSED (" << passed << " tests)" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } else {
        std::cout << "✗ SOME TESTS FAILED" << std::endl;
        std::cout << "  Passed: " << passed << std::endl;
        std::cout << "  Failed: " << failed << std::endl;
        std::cout << "========================================" << std::endl;
        return 1;
    }
}

