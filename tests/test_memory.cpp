// ============================================================================
// XFT Memory Module - C++ Unit Tests
// ============================================================================
// Tests aligned memory allocation, deallocation, and alignment validation.

#include "xft/memory.h"
#include <iostream>
#include <cstdlib>
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

#define TEST_ASSERT_NEAR(a, b, tolerance, message)                             \
    if (std::abs((a) - (b)) > (tolerance)) {                                   \
        std::cerr << "    ✗ FAILED: " << message << std::endl;                \
        std::cerr << "      Expected: " << (b) << std::endl;                  \
        std::cerr << "      Got: " << (a) << std::endl;                       \
        std::cerr << "      at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false;                                                          \
    }

// ============================================================================
// Test Functions
// ============================================================================

// Test basic aligned allocation and deallocation
bool test_basic_allocation() {
    void* ptr = xft::memory::alloc_aligned(1024, 32);
    TEST_ASSERT(ptr != nullptr, "alloc_aligned should return non-null");
    TEST_ASSERT(xft::memory::is_aligned(ptr, 32), "pointer should be 32-byte aligned");
    
    // Verify we can write to the memory
    std::memset(ptr, 0xAB, 1024);
    unsigned char* bytes = static_cast<unsigned char*>(ptr);
    TEST_ASSERT(bytes[0] == 0xAB, "memory should be writable");
    TEST_ASSERT(bytes[1023] == 0xAB, "all allocated memory should be accessible");
    
    xft::memory::free_aligned(ptr);
    return true;
}

// Test zero-size allocation
bool test_zero_size_allocation() {
    void* ptr = xft::memory::alloc_aligned(0, 32);
    TEST_ASSERT(ptr == nullptr, "zero-size allocation should return nullptr");
    
    // Should be safe to free nullptr
    xft::memory::free_aligned(nullptr);
    return true;
}

// Test various alignment values
bool test_alignment_values() {
    // Test common alignments
    size_t alignments[] = {8, 16, 32, 64, 128, 256};
    for (size_t alignment : alignments) {
        void* ptr = xft::memory::alloc_aligned(1024, alignment);
        TEST_ASSERT(ptr != nullptr, "allocation should succeed for valid alignment");
        TEST_ASSERT(xft::memory::is_aligned(ptr, alignment), 
                   "pointer should have requested alignment");
        xft::memory::free_aligned(ptr);
    }
    return true;
}

// Test invalid alignment (not power of 2)
bool test_invalid_alignment() {
    try {
        xft::memory::alloc_aligned(1024, 17); // Not power of 2
        TEST_ASSERT(false, "should throw exception for invalid alignment");
    } catch (const std::invalid_argument& e) {
        // Expected
    }
    
    try {
        xft::memory::alloc_aligned(1024, 3); // Not power of 2
        TEST_ASSERT(false, "should throw exception for invalid alignment");
    } catch (const std::invalid_argument& e) {
        // Expected
    }
    
    return true;
}

// Test alignment too small (< sizeof(void*))
bool test_alignment_too_small() {
    try {
        xft::memory::alloc_aligned(1024, 4); // Less than sizeof(void*)
        TEST_ASSERT(false, "should throw exception for alignment < sizeof(void*)");
    } catch (const std::invalid_argument& e) {
        // Expected
    }
    return true;
}

// Test large allocations
bool test_large_allocation() {
    // Allocate 100 MB
    size_t size = 100 * 1024 * 1024;
    void* ptr = xft::memory::alloc_aligned(size, 32);
    TEST_ASSERT(ptr != nullptr, "large allocation should succeed");
    TEST_ASSERT(xft::memory::is_aligned(ptr, 32), "large allocation should be aligned");
    
    // Verify we can write to start and end
    char* bytes = static_cast<char*>(ptr);
    bytes[0] = 'A';
    bytes[size - 1] = 'Z';
    TEST_ASSERT(bytes[0] == 'A', "should be able to write to start");
    TEST_ASSERT(bytes[size - 1] == 'Z', "should be able to write to end");
    
    xft::memory::free_aligned(ptr);
    return true;
}

// Test align_size helper function
bool test_align_size() {
    TEST_ASSERT(xft::memory::align_size(0, 32) == 0, "align_size(0, 32) == 0");
    TEST_ASSERT(xft::memory::align_size(1, 32) == 32, "align_size(1, 32) == 32");
    TEST_ASSERT(xft::memory::align_size(32, 32) == 32, "align_size(32, 32) == 32");
    TEST_ASSERT(xft::memory::align_size(33, 32) == 64, "align_size(33, 32) == 64");
    TEST_ASSERT(xft::memory::align_size(64, 32) == 64, "align_size(64, 32) == 64");
    TEST_ASSERT(xft::memory::align_size(65, 32) == 96, "align_size(65, 32) == 96");
    
    // Test with different alignments
    TEST_ASSERT(xft::memory::align_size(50, 16) == 64, "align_size(50, 16) == 64");
    TEST_ASSERT(xft::memory::align_size(100, 64) == 128, "align_size(100, 64) == 128");
    
    return true;
}

// Test calculate_aligned_size helper
bool test_calculate_aligned_size() {
    // 10 elements of 4 bytes each = 40 bytes, aligned to 32 = 64 bytes
    TEST_ASSERT(xft::memory::calculate_aligned_size(10, 4, 32) == 64,
               "10 * 4 bytes aligned to 32 == 64");
    
    // 100 elements of 4 bytes each = 400 bytes, aligned to 32 = 416 bytes
    TEST_ASSERT(xft::memory::calculate_aligned_size(100, 4, 32) == 416,
               "100 * 4 bytes aligned to 32 == 416");
    
    // 1000 elements of 8 bytes each = 8000 bytes, aligned to 32 = 8000 bytes (already aligned)
    TEST_ASSERT(xft::memory::calculate_aligned_size(1000, 8, 32) == 8000,
               "1000 * 8 bytes aligned to 32 == 8000");
    
    return true;
}

// Test is_aligned for various pointers
bool test_is_aligned() {
    // Allocate with different alignments and test
    void* ptr32 = xft::memory::alloc_aligned(1024, 32);
    TEST_ASSERT(xft::memory::is_aligned(ptr32, 32), "32-aligned ptr should be 32-aligned");
    TEST_ASSERT(xft::memory::is_aligned(ptr32, 16), "32-aligned ptr should be 16-aligned");
    TEST_ASSERT(xft::memory::is_aligned(ptr32, 8), "32-aligned ptr should be 8-aligned");
    xft::memory::free_aligned(ptr32);
    
    void* ptr64 = xft::memory::alloc_aligned(1024, 64);
    TEST_ASSERT(xft::memory::is_aligned(ptr64, 64), "64-aligned ptr should be 64-aligned");
    TEST_ASSERT(xft::memory::is_aligned(ptr64, 32), "64-aligned ptr should be 32-aligned");
    xft::memory::free_aligned(ptr64);
    
    // Test nullptr
    TEST_ASSERT(xft::memory::is_aligned(nullptr, 32), "nullptr should be considered aligned");
    
    return true;
}

// Test multiple allocations and frees
bool test_multiple_allocations() {
    const int count = 100;
    void* ptrs[count];
    
    // Allocate many blocks
    for (int i = 0; i < count; ++i) {
        ptrs[i] = xft::memory::alloc_aligned(1024, 32);
        TEST_ASSERT(ptrs[i] != nullptr, "allocation should succeed");
        TEST_ASSERT(xft::memory::is_aligned(ptrs[i], 32), "each allocation should be aligned");
    }
    
    // Free all blocks
    for (int i = 0; i < count; ++i) {
        xft::memory::free_aligned(ptrs[i]);
    }
    
    return true;
}

// Test memory persistence (write and read)
bool test_memory_persistence() {
    void* ptr = xft::memory::alloc_aligned(1024, 32);
    TEST_ASSERT(ptr != nullptr, "allocation should succeed");
    
    // Write pattern
    int* data = static_cast<int*>(ptr);
    for (int i = 0; i < 256; ++i) {
        data[i] = i * i;
    }
    
    // Read back and verify
    for (int i = 0; i < 256; ++i) {
        TEST_ASSERT(data[i] == i * i, "data should persist");
    }
    
    xft::memory::free_aligned(ptr);
    return true;
}

// Test default alignment constant
bool test_default_alignment() {
    TEST_ASSERT(xft::memory::DEFAULT_ALIGNMENT == 32, 
               "DEFAULT_ALIGNMENT should be 32 for AVX2");
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
    std::cout << "XFT Memory C++ Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    TestCase tests[] = {
        {"test_basic_allocation", test_basic_allocation},
        {"test_zero_size_allocation", test_zero_size_allocation},
        {"test_alignment_values", test_alignment_values},
        {"test_invalid_alignment", test_invalid_alignment},
        {"test_alignment_too_small", test_alignment_too_small},
        {"test_large_allocation", test_large_allocation},
        {"test_align_size", test_align_size},
        {"test_calculate_aligned_size", test_calculate_aligned_size},
        {"test_is_aligned", test_is_aligned},
        {"test_multiple_allocations", test_multiple_allocations},
        {"test_memory_persistence", test_memory_persistence},
        {"test_default_alignment", test_default_alignment},
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

