// ============================================================================
// XFT Scalar - C++ Unit Tests
// This file tests our C++ Scalar class directly, without Python bindings.
// ============================================================================

#include "xft/scalar.h"
#include "xft/scalar_types.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <string>

// Simple test framework macros
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running test_" #name "..."; \
    test_##name(); \
    std::cout << " ✓ PASSED\n"; \
} while(0)

#define ASSERT(condition) do { \
    if (!(condition)) { \
        std::cerr << "\n  ✗ FAILED: " #condition "\n"; \
        std::cerr << "  at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_NEAR(a, b, epsilon) ASSERT(std::abs((a) - (b)) < (epsilon))

using namespace xft;

// Test 1: Basic Construction
TEST(construction) {
    // Create a Float32 scalar
    Scalar s(3.14f);
    
    // Verify type is correct
    ASSERT_EQ(s.dtype(), ScalarType::Float32);
    ASSERT_EQ(s.dtypeString(), "float32");
    ASSERT(s.isFloat32());
    
    // Verify value is correct
    float value = s.toFloat();
    ASSERT_NEAR(value, 3.14f, 1e-6f);
}

// Test 2: Value Extraction
TEST(value_extraction) {
    Scalar s(42.5f);
    
    // Test template method
    float val1 = s.to<float>();
    ASSERT_NEAR(val1, 42.5f, 1e-6f);
    
    // Test convenience method
    float val2 = s.toFloat();
    ASSERT_NEAR(val2, 42.5f, 1e-6f);
    
    // Both methods should give same result
    ASSERT_EQ(val1, val2);
}

// Test 3: Type Checking
TEST(type_checking) {
    Scalar s(1.0f);
    
    // Should be Float32
    ASSERT(s.isFloat32());
    ASSERT_EQ(s.dtype(), ScalarType::Float32);
    
    // Verify string representation
    std::string dtype_str = s.dtypeString();
    ASSERT_EQ(dtype_str, "float32");
}

// Test 4: Equality Comparison
TEST(equality) {
    Scalar s1(3.14f);
    Scalar s2(3.14f);
    Scalar s3(2.71f);
    
    // Same value -> equal
    ASSERT(s1 == s2);
    ASSERT_EQ(s1, s2);
    
    // Different value -> not equal
    ASSERT(s1 != s3);
    ASSERT_NE(s1, s3);
    
    // Self-equality
    ASSERT(s1 == s1);
}

// Test 5: Boolean Conversion (Truthiness)
TEST(boolean_conversion) {
    Scalar zero(0.0f);
    Scalar nonzero(3.14f);
    Scalar negative(-1.0f);
    
    // Zero is falsy
    ASSERT(!static_cast<bool>(zero));
    
    // Non-zero is truthy
    ASSERT(static_cast<bool>(nonzero));
    ASSERT(static_cast<bool>(negative));
}

// Test 6: String Representation
TEST(string_representation) {
    Scalar s(3.14f);
    
    std::string repr = s.repr();
    
    // Should contain the value
    ASSERT(repr.find("3.14") != std::string::npos);
    
    // Should contain the dtype
    ASSERT(repr.find("float32") != std::string::npos);
    
    // Should look like a function call
    ASSERT(repr.find("Scalar") != std::string::npos);
    
    std::cout << "\n    repr: " << repr;
}

// Test 7: Edge Cases - Special Float Values
TEST(special_float_values) {
    // Zero
    Scalar s_zero(0.0f);
    ASSERT_EQ(s_zero.toFloat(), 0.0f);
    
    // Negative
    Scalar s_neg(-123.456f);
    ASSERT_NEAR(s_neg.toFloat(), -123.456f, 1e-5f);
    
    // Very small
    Scalar s_small(1e-10f);
    ASSERT_NEAR(s_small.toFloat(), 1e-10f, 1e-15f);
    
    // Very large
    Scalar s_large(1e10f);
    ASSERT_NEAR(s_large.toFloat(), 1e10f, 1e5f);
    
    // Infinity
    Scalar s_inf(INFINITY);
    ASSERT(std::isinf(s_inf.toFloat()));
    
    // NaN
    Scalar s_nan(NAN);
    ASSERT(std::isnan(s_nan.toFloat()));
}

// Test 8: Type System - Enum Conversions
TEST(type_system) {
    // ScalarType to string
    std::string name = scalarTypeToString(ScalarType::Float32);
    ASSERT_EQ(name, "float32");
    
    // String to ScalarType
    auto dtype = stringToScalarType("float32");
    ASSERT(dtype.has_value());
    ASSERT_EQ(dtype.value(), ScalarType::Float32);
    
    // Invalid string
    auto invalid = stringToScalarType("invalid_type");
    ASSERT(!invalid.has_value());
}

// Test 9: Copy Semantics
TEST(copy_semantics) {
    Scalar s1(3.14f);
    
    // Copy construction
    Scalar s2 = s1;
    ASSERT_EQ(s1, s2);
    ASSERT_NEAR(s2.toFloat(), 3.14f, 1e-6f);
    
    // Copy assignment
    Scalar s3(0.0f);
    s3 = s1;
    ASSERT_EQ(s1, s3);
    ASSERT_NEAR(s3.toFloat(), 3.14f, 1e-6f);
    
    // Copies are independent
    Scalar s4 = s1;
    Scalar s5(99.9f);
    s4 = s5;
    ASSERT_NE(s1, s4);  // s1 unchanged
}

// Test 10: Move Semantics
TEST(move_semantics) {
    Scalar s1(3.14f);
    
    // Move construction
    Scalar s2 = std::move(s1);
    ASSERT_NEAR(s2.toFloat(), 3.14f, 1e-6f);
    
    // Move assignment
    Scalar s3(0.0f);
    Scalar s4(2.71f);
    s3 = std::move(s4);
    ASSERT_NEAR(s3.toFloat(), 2.71f, 1e-6f);
}

// Test 11: Performance - Construction and Extraction
TEST(performance_basic) {
    const int iterations = 1'000'000;
    
    std::cout << "\n    Timing " << iterations << " scalar operations...";
    
    // Benchmark construction + extraction
    float sum = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        Scalar s(static_cast<float>(i));
        sum += s.toFloat();
    }
    
    // Verify correctness (sum of 0..N-1)
    // Cast to avoid integer overflow for large N
    float expected = static_cast<float>(iterations - 1) * iterations / 2.0f;
    // Use larger tolerance due to accumulation of floating point errors
    ASSERT_NEAR(sum, expected, expected * 1e-3f);
    
    std::cout << " done";
}

// Test 12: Error Handling - Type Mismatch (Future)
TEST(error_handling) {
    Scalar s(3.14f);
    
    // Currently only Float32 exists, so this test just verifies
    // that extraction works correctly.
    // When we add Float64, we can test type mismatch errors.
    
    try {
        float val = s.to<float>();  // Should succeed
        ASSERT_NEAR(val, 3.14f, 1e-6f);
    } catch (...) {
        ASSERT(false);  // Should not throw for correct type
    }
}

// Main Test Runner
int main() {
    std::cout << "XFT Scalar C++ Unit Tests\n";
    std::cout << "----------------------------------------\n";
    
    try {
        RUN_TEST(construction);
        RUN_TEST(value_extraction);
        RUN_TEST(type_checking);
        RUN_TEST(equality);
        RUN_TEST(boolean_conversion);
        RUN_TEST(string_representation);
        RUN_TEST(special_float_values);
        RUN_TEST(type_system);
        RUN_TEST(copy_semantics);
        RUN_TEST(move_semantics);
        RUN_TEST(performance_basic);
        RUN_TEST(error_handling);
        
        std::cout << "\n✓ ALL TESTS PASSED (" << 12 << " tests)\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ TEST FAILED with exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "\n✗ TEST FAILED with unknown exception\n";
        return 1;
    }
}