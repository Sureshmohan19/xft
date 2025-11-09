// ============================================================================
// XFT Integration Tests
// ============================================================================
// Tests the complete system: memory + storage + array working together.

#include "xft/memory.h"
#include "xft/storage.h"
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
// Integration Test Functions
// ============================================================================

// Test: Memory → Storage → Array flow
bool test_full_stack_integration() {
    // 1. Allocate aligned memory directly
    size_t size_bytes = 1024;
    void* raw_ptr = xft::memory::alloc_aligned(size_bytes, 32);
    TEST_ASSERT(raw_ptr != nullptr, "memory allocation should succeed");
    TEST_ASSERT(xft::memory::is_aligned(raw_ptr, 32), "memory should be aligned");
    
    // 2. Wrap in Storage
    auto storage = xft::Storage::wrap(raw_ptr, size_bytes, xft::ScalarType::Float32);
    TEST_ASSERT(storage->data() == raw_ptr, "storage should wrap memory");
    TEST_ASSERT(storage->size_bytes() == size_bytes, "storage size should match");
    
    // 3. Create Array from Storage
    xft::Array arr(storage, {256}, {4}, xft::ScalarType::Float32, 0);
    TEST_ASSERT(arr.size() == 256, "array should have 256 elements");
    TEST_ASSERT(arr.data() == raw_ptr, "array should point to same memory");
    
    // 4. Use the array
    arr.at<float>(0) = 3.14f;
    arr.at<float>(255) = 2.71f;
    
    // 5. Verify data through different views
    float* direct_view = static_cast<float*>(raw_ptr);
    TEST_ASSERT(direct_view[0] == 3.14f, "should see data through raw pointer");
    TEST_ASSERT(direct_view[255] == 2.71f, "should see data through raw pointer");
    
    float* storage_view = static_cast<float*>(storage->data());
    TEST_ASSERT(storage_view[0] == 3.14f, "should see data through storage");
    TEST_ASSERT(storage_view[255] == 2.71f, "should see data through storage");
    
    // 6. Clean up (storage doesn't own, so we must free manually)
    storage.reset();
    xft::memory::free_aligned(raw_ptr);
    
    return true;
}

// Test: Multiple arrays sharing one storage
// Note: use_count() tracks Storage objects sharing data, not shared_ptr<Storage> instances
bool test_shared_storage_multiple_arrays() {
    // Create storage
    auto storage = xft::Storage::create(1024, xft::ScalarType::Float32);
    TEST_ASSERT(storage->use_count() == 1, "initial use_count == 1");
    
    // Create first array (full storage)
    xft::Array arr1(storage, {256}, {4}, xft::ScalarType::Float32, 0);
    // use_count still 1: storage and arr1.storage() point to SAME Storage object
    TEST_ASSERT(storage->use_count() == 1, "use_count still 1 (same Storage object)");
    
    // Create second array (offset view)
    xft::Array arr2(storage, {64}, {4}, xft::ScalarType::Float32, 512);
    // use_count still 1: all shared_ptrs point to SAME Storage object
    TEST_ASSERT(storage->use_count() == 1, "use_count still 1 (same Storage object)");
    
    // Write through arr1
    arr1.at<float>(128) = 99.0f;
    
    // Read through arr2 (offset 512 bytes = 128 floats, so arr2[0] == arr1[128])
    TEST_ASSERT(arr2.at<float>(0) == 99.0f, "arr2 should see arr1's write");
    
    // Verify storage pointer persistence - all point to same Storage object
    TEST_ASSERT(arr1.storage().get() == storage.get(), "arr1 references same Storage");
    TEST_ASSERT(arr2.storage().get() == storage.get(), "arr2 references same Storage");
    
    return true;
}

// Test: 2D matrix operations with proper stride calculations
bool test_matrix_layout_integration() {
    // Create 3x4 matrix in C-order
    xft::Array mat_c({3, 4}, xft::ScalarType::Float32, xft::Array::Order::C);
    
    // Fill with pattern: mat[i][j] = i*10 + j
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            mat_c.at<float>(i, j) = static_cast<float>(i * 10 + j);
        }
    }
    
    // Verify memory layout (C-order: row-major)
    float* data = static_cast<float*>(mat_c.data());
    TEST_ASSERT(data[0] == 0.0f, "data[0] == mat[0][0]");
    TEST_ASSERT(data[1] == 1.0f, "data[1] == mat[0][1]");
    TEST_ASSERT(data[4] == 10.0f, "data[4] == mat[1][0]");
    TEST_ASSERT(data[11] == 23.0f, "data[11] == mat[2][3]");
    
    // Create 3x4 matrix in F-order
    xft::Array mat_f({3, 4}, xft::ScalarType::Float32, xft::Array::Order::F);
    
    // Fill with same pattern
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            mat_f.at<float>(i, j) = static_cast<float>(i * 10 + j);
        }
    }
    
    // Verify memory layout (F-order: column-major)
    float* data_f = static_cast<float*>(mat_f.data());
    TEST_ASSERT(data_f[0] == 0.0f, "data[0] == mat[0][0]");
    TEST_ASSERT(data_f[1] == 10.0f, "data[1] == mat[1][0]");  // Next in column
    TEST_ASSERT(data_f[3] == 1.0f, "data[3] == mat[0][1]");   // Next column
    
    // Both representations should give same logical result
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            TEST_ASSERT(mat_c.at<float>(i, j) == mat_f.at<float>(i, j),
                       "C and F layouts should match logically");
        }
    }
    
    return true;
}

// Test: Type safety across the stack
bool test_type_safety_integration() {
    // Float32 storage
    auto storage_f32 = xft::Storage::create(1024, xft::ScalarType::Float32);
    xft::Array arr_f32(storage_f32, {256}, {4}, xft::ScalarType::Float32, 0);
    
    // Should work with float
    arr_f32.at<float>(0) = 1.0f;
    TEST_ASSERT(arr_f32.at<float>(0) == 1.0f, "float access should work");
    
    // Should fail with double
    try {
        arr_f32.at<double>(0) = 1.0;
        TEST_ASSERT(false, "double access on float32 array should throw");
    } catch (const std::invalid_argument&) {
        // Expected
    }
    
    // Float64 storage
    auto storage_f64 = xft::Storage::create(2048, xft::ScalarType::Float64);
    xft::Array arr_f64(storage_f64, {256}, {8}, xft::ScalarType::Float64, 0);
    
    // Should work with double
    arr_f64.at<double>(0) = 1.0;
    TEST_ASSERT(arr_f64.at<double>(0) == 1.0, "double access should work");
    
    // Should fail with float
    try {
        arr_f64.at<float>(0) = 1.0f;
        TEST_ASSERT(false, "float access on float64 array should throw");
    } catch (const std::invalid_argument&) {
        // Expected
    }
    
    return true;
}

// Test: Large multidimensional array
bool test_large_multidim_array() {
    // Create 100x100x100 array (1M elements, 4MB)
    xft::Array arr({100, 100, 100}, xft::ScalarType::Float32);
    
    TEST_ASSERT(arr.size() == 1000000, "size should be 1M");
    TEST_ASSERT(arr.nbytes() == 4000000, "nbytes should be 4M");
    TEST_ASSERT(arr.is_contiguous(), "should be contiguous");
    
    // Test corner access
    arr.at<float>(0, 0, 0) = 1.0f;
    arr.at<float>(99, 99, 99) = 2.0f;
    arr.at<float>(50, 50, 50) = 3.0f;
    
    TEST_ASSERT(arr.at<float>(0, 0, 0) == 1.0f, "corner access");
    TEST_ASSERT(arr.at<float>(99, 99, 99) == 2.0f, "corner access");
    TEST_ASSERT(arr.at<float>(50, 50, 50) == 3.0f, "middle access");
    
    return true;
}

// Test: Memory alignment propagation
bool test_alignment_propagation() {
    // Create storage with 64-byte alignment
    auto storage = xft::Storage::create(1024, xft::ScalarType::Float32, 64);
    
    // Check storage alignment
    TEST_ASSERT(xft::memory::is_aligned(storage->data(), 64), 
               "storage should be 64-byte aligned");
    
    // Create array from storage
    xft::Array arr(storage, {256}, {4}, xft::ScalarType::Float32, 0);
    
    // Array should also be aligned
    TEST_ASSERT(xft::memory::is_aligned(arr.data(), 64),
               "array data should inherit alignment");
    
    // Test with offset that maintains alignment
    xft::Array arr_offset(storage, {64}, {4}, xft::ScalarType::Float32, 64);
    TEST_ASSERT(xft::memory::is_aligned(arr_offset.data(), 64),
               "offset array should maintain alignment");
    
    return true;
}

// Test: Scalar type registry integration
bool test_scalar_type_integration() {
    // Test Float32
    auto arr_f32 = xft::Array({10}, xft::ScalarType::Float32);
    TEST_ASSERT(xft::scalarTypeName(arr_f32.dtype()) == "float32",
               "dtype name should be float32");
    TEST_ASSERT(xft::scalarTypeSize(arr_f32.dtype()) == 4,
               "dtype size should be 4");
    
    // Test Float64
    auto arr_f64 = xft::Array({10}, xft::ScalarType::Float64);
    TEST_ASSERT(xft::scalarTypeName(arr_f64.dtype()) == "float64",
               "dtype name should be float64");
    TEST_ASSERT(xft::scalarTypeSize(arr_f64.dtype()) == 8,
               "dtype size should be 8");
    
    // Test string conversions
    TEST_ASSERT(xft::stringToScalarType("float32") == xft::ScalarType::Float32,
               "string to Float32");
    TEST_ASSERT(xft::stringToScalarType("float64") == xft::ScalarType::Float64,
               "string to Float64");
    TEST_ASSERT(!xft::stringToScalarType("invalid").has_value(),
               "invalid string should return nullopt");
    
    return true;
}

// Test: Complex data flow scenario
bool test_complex_data_flow() {
    // Scenario: Create matrix, fill with data, create view, modify through view
    
    // 1. Create 4x5 matrix
    xft::Array original({4, 5}, xft::ScalarType::Float32);
    
    // 2. Fill with sequential values
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            original.at<float>(i, j) = static_cast<float>(i * 5 + j);
        }
    }
    
    // 3. Create view of second row (starting at offset 5*4 = 20 bytes)
    auto storage = original.storage();
    xft::Array row_view(storage, {5}, {4}, xft::ScalarType::Float32, 20);
    
    // 4. Verify view sees correct data (row 1: values 5-9)
    TEST_ASSERT(row_view.at<float>(0) == 5.0f, "row_view[0] == 5");
    TEST_ASSERT(row_view.at<float>(4) == 9.0f, "row_view[4] == 9");
    
    // 5. Modify through view
    row_view.at<float>(2) = 999.0f;
    
    // 6. Verify original sees modification
    TEST_ASSERT(original.at<float>(1, 2) == 999.0f, 
               "original should see view modification");
    
    // 7. Verify storage use count (still 1 - all shared_ptrs point to same Storage)
    TEST_ASSERT(storage->use_count() == 1, 
               "use_count still 1 (storage + original + row_view all reference same Storage object)");
    
    return true;
}

// Test: Error handling across components
bool test_error_handling_integration() {
    // Test 1: Invalid array shape propagates
    try {
        xft::Array arr({0}, xft::ScalarType::Float32);
        TEST_ASSERT(false, "zero dimension should throw");
    } catch (const std::invalid_argument&) {
        // Expected
    }
    
    // Test 2: Zero storage size throws
    try {
        auto storage = xft::Storage::create(0, xft::ScalarType::Float32);
        TEST_ASSERT(false, "zero storage should throw");
    } catch (const std::invalid_argument&) {
        // Expected
    }
    
    // Test 3: Invalid alignment throws
    try {
        xft::memory::alloc_aligned(1024, 17); // Not power of 2
        TEST_ASSERT(false, "invalid alignment should throw");
    } catch (const std::invalid_argument&) {
        // Expected
    }
    
    // Test 4: Out of bounds access throws
    xft::Array arr({10}, xft::ScalarType::Float32);
    try {
        arr.at<float>(10);
        TEST_ASSERT(false, "out of bounds should throw");
    } catch (const std::out_of_range&) {
        // Expected
    }
    
    return true;
}

// Test: Memory reuse pattern
// Note: use_count() always 1 since all shared_ptrs point to same Storage object
bool test_memory_reuse_pattern() {
    // Common pattern: allocate once, reuse for multiple operations
    
    // 1. Allocate storage
    auto storage = xft::Storage::create(4096, xft::ScalarType::Float32);
    
    // 2. Use as 1024-element 1D array
    {
        xft::Array arr1d(storage, {1024}, {4}, xft::ScalarType::Float32, 0);
        arr1d.at<float>(0) = 1.0f;
        // use_count still 1: storage and arr1d.storage() point to SAME Storage object
        TEST_ASSERT(storage->use_count() == 1, "use_count still 1 with arr1d");
    }
    TEST_ASSERT(storage->use_count() == 1, "use_count still 1 after arr1d destroyed");
    
    // 3. Reuse same storage as 16x16 2D array
    {
        xft::Array arr2d(storage, {16, 16}, {64, 4}, xft::ScalarType::Float32, 0);
        TEST_ASSERT(arr2d.at<float>(0, 0) == 1.0f, "should see previous data");
        arr2d.at<float>(15, 15) = 2.0f;
        TEST_ASSERT(storage->use_count() == 1, "use_count still 1 with arr2d");
    }
    TEST_ASSERT(storage->use_count() == 1, "use_count still 1 after arr2d destroyed");
    
    // 4. Reuse as 4x4x4 3D array
    {
        xft::Array arr3d(storage, {4, 4, 4}, {64, 16, 4}, 
                        xft::ScalarType::Float32, 0);
        TEST_ASSERT(arr3d.at<float>(0, 0, 0) == 1.0f, "should see original data");
        TEST_ASSERT(storage->use_count() == 1, "use_count still 1 with arr3d");
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
    std::cout << "XFT Integration Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    TestCase tests[] = {
        {"test_full_stack_integration", test_full_stack_integration},
        {"test_shared_storage_multiple_arrays", test_shared_storage_multiple_arrays},
        {"test_matrix_layout_integration", test_matrix_layout_integration},
        {"test_type_safety_integration", test_type_safety_integration},
        {"test_large_multidim_array", test_large_multidim_array},
        {"test_alignment_propagation", test_alignment_propagation},
        {"test_scalar_type_integration", test_scalar_type_integration},
        {"test_complex_data_flow", test_complex_data_flow},
        {"test_error_handling_integration", test_error_handling_integration},
        {"test_memory_reuse_pattern", test_memory_reuse_pattern},
    };
    
    int passed = 0;
    int failed = 0;
    
    for (const auto& test : tests) {
        std::cout << "\nRunning " << test.name << "..." << std::endl;
        if (test.func()) {
            std::cout << "    ✓ PASSED" << std::endl;
            passed++;
        } else {
            failed++;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    if (failed == 0) {
        std::cout << "✓ ALL INTEGRATION TESTS PASSED (" << passed << " tests)" << std::endl;
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

