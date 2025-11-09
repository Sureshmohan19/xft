// ============================================================================
// XFT Storage Module - C++ Unit Tests
// ============================================================================
// Tests reference-counted storage, wrapping external memory, and use_count.

#include "xft/storage.h"
#include "xft/scalar_types.h"
#include <iostream>
#include <cstring>

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

// Test basic storage creation
bool test_basic_creation() {
    auto storage = xft::Storage::create(1024, xft::ScalarType::Float32);
    
    TEST_ASSERT(storage != nullptr, "storage should be created");
    TEST_ASSERT(storage->data() != nullptr, "storage data should be non-null");
    TEST_ASSERT(storage->size_bytes() == 1024, "size should match");
    TEST_ASSERT(storage->dtype() == xft::ScalarType::Float32, "dtype should match");
    TEST_ASSERT(storage->owns_data() == true, "storage should own its data");
    TEST_ASSERT(storage->use_count() == 1, "initial use_count should be 1");
    
    return true;
}

// Test zero-size allocation should throw
bool test_zero_size_creation() {
    try {
        auto storage = xft::Storage::create(0, xft::ScalarType::Float32);
        TEST_ASSERT(false, "zero-size allocation should throw");
    } catch (const std::invalid_argument& e) {
        // Expected
    }
    return true;
}

// Test data read/write
bool test_data_readwrite() {
    auto storage = xft::Storage::create(1024, xft::ScalarType::Float32);
    
    // Write data
    float* data = static_cast<float*>(storage->data());
    data[0] = 3.14f;
    data[1] = 2.71f;
    data[255] = 42.0f;
    
    // Read back
    TEST_ASSERT(data[0] == 3.14f, "data[0] should be readable");
    TEST_ASSERT(data[1] == 2.71f, "data[1] should be readable");
    TEST_ASSERT(data[255] == 42.0f, "data[255] should be readable");
    
    // Test const access
    const auto* const_storage = storage.get();
    const float* const_data = static_cast<const float*>(const_storage->data());
    TEST_ASSERT(const_data[0] == 3.14f, "const data access should work");
    
    return true;
}

// Test shared ownership (reference counting)
// Note: use_count() tracks how many Storage objects share the same data buffer,
// NOT how many shared_ptr<Storage> instances exist.
// Since Storage copy is deleted, each Storage has its own data_shared_.
bool test_shared_ownership() {
    auto storage1 = xft::Storage::create(1024, xft::ScalarType::Float32);
    TEST_ASSERT(storage1->use_count() == 1, "initial use_count == 1");
    
    // Create second reference to SAME Storage object
    auto storage2 = storage1;
    // use_count is still 1 because both shared_ptrs point to the SAME Storage object
    // which has ONE data_shared_ member
    TEST_ASSERT(storage1->use_count() == 1, "use_count still 1 (same Storage object)");
    TEST_ASSERT(storage2->use_count() == 1, "use_count still 1 (same Storage object)");
    TEST_ASSERT(storage1->data() == storage2->data(), "both should point to same data");
    TEST_ASSERT(storage1.get() == storage2.get(), "both point to same Storage object");
    
    // Create third reference
    auto storage3 = storage2;
    TEST_ASSERT(storage1->use_count() == 1, "use_count still 1");
    TEST_ASSERT(storage2->use_count() == 1, "use_count still 1");
    TEST_ASSERT(storage3->use_count() == 1, "use_count still 1");
    
    // All three shared_ptrs point to the SAME Storage object
    TEST_ASSERT(storage1.get() == storage2.get(), "same Storage");
    TEST_ASSERT(storage2.get() == storage3.get(), "same Storage");
    
    return true;
}

// Test wrapping external memory
bool test_wrap_external_memory() {
    // Allocate external memory
    float external_data[256];
    external_data[0] = 1.0f;
    external_data[1] = 2.0f;
    external_data[255] = 256.0f;
    
    // Wrap it
    auto storage = xft::Storage::wrap(external_data, sizeof(external_data), 
                                      xft::ScalarType::Float32);
    
    TEST_ASSERT(storage != nullptr, "wrapped storage should be created");
    TEST_ASSERT(storage->data() == external_data, "data pointer should match");
    TEST_ASSERT(storage->size_bytes() == sizeof(external_data), "size should match");
    TEST_ASSERT(storage->dtype() == xft::ScalarType::Float32, "dtype should match");
    TEST_ASSERT(storage->owns_data() == false, "wrapped storage should not own data");
    
    // Read through storage
    float* data = static_cast<float*>(storage->data());
    TEST_ASSERT(data[0] == 1.0f, "should read external data");
    TEST_ASSERT(data[1] == 2.0f, "should read external data");
    TEST_ASSERT(data[255] == 256.0f, "should read external data");
    
    // Modify through storage
    data[0] = 99.0f;
    TEST_ASSERT(external_data[0] == 99.0f, "modifications should affect external data");
    
    return true;
}

// Test wrapping null pointer should throw
bool test_wrap_null_pointer() {
    try {
        auto storage = xft::Storage::wrap(nullptr, 1024, xft::ScalarType::Float32);
        TEST_ASSERT(false, "wrapping null should throw");
    } catch (const std::invalid_argument& e) {
        // Expected
    }
    return true;
}

// Test different dtypes
bool test_different_dtypes() {
    // Float32
    auto storage_f32 = xft::Storage::create(1024, xft::ScalarType::Float32);
    TEST_ASSERT(storage_f32->dtype() == xft::ScalarType::Float32, "Float32 dtype");
    
    float* data_f32 = static_cast<float*>(storage_f32->data());
    data_f32[0] = 3.14f;
    TEST_ASSERT(data_f32[0] == 3.14f, "Float32 read/write");
    
    // Float64
    auto storage_f64 = xft::Storage::create(1024, xft::ScalarType::Float64);
    TEST_ASSERT(storage_f64->dtype() == xft::ScalarType::Float64, "Float64 dtype");
    
    double* data_f64 = static_cast<double*>(storage_f64->data());
    data_f64[0] = 3.141592653589793;
    TEST_ASSERT(data_f64[0] == 3.141592653589793, "Float64 read/write");
    
    return true;
}

// Test memory alignment
bool test_memory_alignment() {
    auto storage = xft::Storage::create(1024, xft::ScalarType::Float32, 64);
    
    TEST_ASSERT(storage->data() != nullptr, "storage should be created");
    
    // Check alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(storage->data());
    TEST_ASSERT((addr & 63) == 0, "data should be 64-byte aligned");
    
    return true;
}

// Test large storage
bool test_large_storage() {
    // 100 MB
    size_t size = 100 * 1024 * 1024;
    auto storage = xft::Storage::create(size, xft::ScalarType::Float32);
    
    TEST_ASSERT(storage != nullptr, "large storage should be created");
    TEST_ASSERT(storage->size_bytes() == size, "size should match");
    
    // Test write at boundaries
    char* data = static_cast<char*>(storage->data());
    data[0] = 'A';
    data[size - 1] = 'Z';
    
    TEST_ASSERT(data[0] == 'A', "should write at start");
    TEST_ASSERT(data[size - 1] == 'Z', "should write at end");
    
    return true;
}

// Test use_count with multiple scopes
bool test_use_count_scopes() {
    std::shared_ptr<xft::Storage> storage1;
    
    {
        auto storage_inner = xft::Storage::create(1024, xft::ScalarType::Float32);
        TEST_ASSERT(storage_inner->use_count() == 1, "use_count == 1 in inner scope");
        
        storage1 = storage_inner;
        // Still use_count == 1 because both shared_ptrs point to same Storage object
        TEST_ASSERT(storage_inner->use_count() == 1, "use_count still 1");
        TEST_ASSERT(storage1->use_count() == 1, "both see use_count == 1");
        TEST_ASSERT(storage1.get() == storage_inner.get(), "same Storage object");
    }
    
    // Inner scope exited, storage_inner destroyed but Storage object still alive
    TEST_ASSERT(storage1->use_count() == 1, "use_count == 1 after scope exit");
    
    return true;
}

// Test storage with pattern fill
bool test_pattern_fill() {
    auto storage = xft::Storage::create(1024, xft::ScalarType::Float32);
    
    // Fill with pattern
    int* data = static_cast<int*>(storage->data());
    for (int i = 0; i < 256; ++i) {
        data[i] = i * 7;
    }
    
    // Verify pattern
    for (int i = 0; i < 256; ++i) {
        TEST_ASSERT(data[i] == i * 7, "pattern should match");
    }
    
    return true;
}

// Test wrapped storage doesn't free memory
bool test_wrapped_storage_lifetime() {
    float external_data[100];
    external_data[0] = 42.0f;
    
    {
        auto storage = xft::Storage::wrap(external_data, sizeof(external_data),
                                         xft::ScalarType::Float32);
        TEST_ASSERT(storage->owns_data() == false, "should not own data");
        
        float* data = static_cast<float*>(storage->data());
        data[0] = 99.0f;
        TEST_ASSERT(external_data[0] == 99.0f, "should modify external data");
    }
    
    // Storage destroyed, external data should still be valid
    TEST_ASSERT(external_data[0] == 99.0f, "external data should persist");
    
    return true;
}

// Test multiple wraps of same memory
bool test_multiple_wraps() {
    float external_data[100];
    
    auto storage1 = xft::Storage::wrap(external_data, sizeof(external_data),
                                      xft::ScalarType::Float32);
    auto storage2 = xft::Storage::wrap(external_data, sizeof(external_data),
                                      xft::ScalarType::Float32);
    
    TEST_ASSERT(storage1->data() == external_data, "storage1 should wrap data");
    TEST_ASSERT(storage2->data() == external_data, "storage2 should wrap data");
    TEST_ASSERT(storage1->data() == storage2->data(), "both should point to same data");
    
    // Note: These are independent wrappers, so use_count is independent
    TEST_ASSERT(storage1->use_count() == 1, "storage1 use_count == 1");
    TEST_ASSERT(storage2->use_count() == 1, "storage2 use_count == 1");
    
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
    std::cout << "XFT Storage C++ Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    TestCase tests[] = {
        {"test_basic_creation", test_basic_creation},
        {"test_zero_size_creation", test_zero_size_creation},
        {"test_data_readwrite", test_data_readwrite},
        {"test_shared_ownership", test_shared_ownership},
        {"test_wrap_external_memory", test_wrap_external_memory},
        {"test_wrap_null_pointer", test_wrap_null_pointer},
        {"test_different_dtypes", test_different_dtypes},
        {"test_memory_alignment", test_memory_alignment},
        {"test_large_storage", test_large_storage},
        {"test_use_count_scopes", test_use_count_scopes},
        {"test_pattern_fill", test_pattern_fill},
        {"test_wrapped_storage_lifetime", test_wrapped_storage_lifetime},
        {"test_multiple_wraps", test_multiple_wraps},
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

