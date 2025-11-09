#ifndef XFT_SCALAR_TYPES_H
#define XFT_SCALAR_TYPES_H

#include <string>
#include <optional>

namespace xft {

// ============================================================================
// x86_64 Linux - Fixed Size Types
// ============================================================================
// On x86_64 Linux with GCC/Clang:
//   - float is ALWAYS 32 bits (IEEE 754 single precision)
//   - double is ALWAYS 64 bits (IEEE 754 double precision)

using float32 = float;    // 32-bit IEEE 754
using float64 = double;   // 64-bit IEEE 754

// ============================================================================
// ScalarType Enum: Runtime Type Identification
// ============================================================================
// This is actually a runtime type identification.
// We need to store "what type is this?" at runtime.

enum class ScalarType : char {
    Float32 = 0,
    Float64 = 1,
};

// ============================================================================
// Type Registry: Single Source of Truth
// ============================================================================
// This macro is NOT abstraction - it's CODE GENERATION.
// It prevents us from writing the same code 10 times.
//
// One entry here generates:
//   - Enum value
//   - String conversion
//   - Type dispatch
//   - Python binding
//
// Without this macro, you'd write duplicate code everywhere.
// With this macro, you write it ONCE.

#define XFT_FORALL_SCALAR_TYPES(_) \
    _(Float32, float, "float32") \
    _(Float64, double, "float64")

// ============================================================================
// Helper: Type to String
// ============================================================================
// Convert a ScalarType enum value to a string.

inline std::string scalarTypeToString(ScalarType type) {
    switch (type) {
        #define CASE_RETURN_STRING(enum_name, cpp_type, str_name) \
            case ScalarType::enum_name: return str_name;
        
        XFT_FORALL_SCALAR_TYPES(CASE_RETURN_STRING)
        
        #undef CASE_RETURN_STRING
    }
    
    // Unreachable (all enum values covered)
    return "unknown";
}

// ============================================================================
// Helper: String to Type
// ============================================================================
// Convert a string to a ScalarType enum value.

inline std::optional<ScalarType> stringToScalarType(const std::string& str) {
    #define CHECK_STRING_MATCH(enum_name, cpp_type, str_name) \
        if (str == str_name) return ScalarType::enum_name;
    
    XFT_FORALL_SCALAR_TYPES(CHECK_STRING_MATCH)
    
    #undef CHECK_STRING_MATCH
    
    // If the string does not match any of the enum values, return nullopt.
    // Why? It will cause a runtime error if we try to use the type. 
    //So we need to handle this.
    return std::nullopt;
}

// ============================================================================
// Helper: Get size in bytes for a ScalarType
// ============================================================================

inline size_t scalarTypeSize(ScalarType type) {
    switch (type) {
        case ScalarType::Float32: return sizeof(float);
        case ScalarType::Float64: return sizeof(double);
    }
    // Unreachable (all enum values covered)
    return 0;
}

// ============================================================================
// Helper: Get string name for a ScalarType (alias for scalarTypeToString)
// ============================================================================

inline std::string scalarTypeName(ScalarType type) {
    return scalarTypeToString(type);
}

}  // namespace xft

#endif  // XFT_SCALAR_TYPES_H