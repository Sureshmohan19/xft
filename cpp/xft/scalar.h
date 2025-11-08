#ifndef XFT_SCALAR_H
#define XFT_SCALAR_H

#include "scalar_types.h"
#include <variant>
#include <stdexcept>

namespace xft {

// ============================================================================
// Scalar: Internal C++ Type for Scalar Operations
// ============================================================================
// This class is NOT exposed to Python users. It's purely for internal C++
// use to handle scalar values in a type-safe way.
//
// Purpose:
//   - Represent a single numeric value with runtime type information
//   - Used in Array operations: array.add(Scalar(5.0f))
//   - Converts Python scalars (int/float) to C++ scalars
//   - Enables type-safe operations without template explosion
//
// Users will never see this - they just pass Python floats/ints.

class Scalar {
private:
    ScalarType type_;
    
    // ========================================================================
    // Storage: Type-safe union of all scalar types
    // ========================================================================
    // std::variant is like a union but type-safe:
    //   - Stores ONE value at a time
    //   - Always knows which type it's currently holding
    //   - No undefined behavior (unlike C unions)
    //
    // x86_64 Linux: sizeof(std::variant<float, double>) == 16 bytes
    //   - 8 bytes for double value (largest type)
    //   - 1 byte for type index (which type is active?)
    //   - 7 bytes padding (alignment)
    
    // Storage for scalar values
    // IMPORTANT: Keep this in sync with XFT_FORALL_SCALAR_TYPES in scalar_types.h
    // TODO: Generate from macro without trailing comma (requires macro redesign)
    using ScalarVariant = std::variant<
        float,   // Float32
        double   // Float64
    >;
    
    ScalarVariant data_;

public:
    // ========================================================================
    // Constructors: One for each scalar type
    // ========================================================================
    // Explicit: Prevents accidental conversions
    // Example: Scalar s = 3.14f;  // Error! Must be explicit: Scalar s(3.14f);
    //
    // Why explicit? Prevents bugs like:
    //   void process(Scalar s);
    //   process(5);  // Should this be int or float? Explicit forces clarity.
    
    explicit Scalar(float value)
        : type_(ScalarType::Float32), data_(value) {
        } // float32

    explicit Scalar(double value)
        : type_(ScalarType::Float64), data_(value) {
        } // float64

    // TODO: Add constructors for other types:
    
    // ========================================================================
    // Type Query: What type is this scalar?
    // ========================================================================
    
    ScalarType dtype() const {return type_;} // scalar type
    
    std::string dtypeString() const {return scalarTypeToString(type_);} // scalar type string
    
    // ========================================================================
    // Value Extraction: Get the raw value back
    // ========================================================================
    // Template method that extracts the value as the requested type.
    // Uses std::get<T> which throws std::bad_variant_access if type mismatch.
    //
    // Example:
    //   Scalar s(3.14f);
    //   float val = s.to<float>();  // OK
    //   double val = s.to<double>(); // Throws! (wrong type)
    //
    // This is intentionally strict at the moment - no automatic conversions.
    // If you want float->double conversion, do it explicitly.
    // Future: Add automatic conversions if thats a good idea.
    
    template<typename T>
    T to() const {
        try {
            return std::get<T>(data_);
            // The above is all we needed. 
            // The below is just a good error message.
        } catch (const std::bad_variant_access&) {
            throw std::runtime_error(
                "Type mismatch: attempted to extract " + 
                std::string(typeid(T).name()) + 
                " from scalar of type " + dtypeString()
            );
        }
    }
    
    // ========================================================================
    // Convenience Methods: Type-specific extraction
    // These are shortcuts for common cases. Prefer these over to<T>() 
    // for readability.
    // ========================================================================

    // Note: Since we only target ARM64 Linux, we can hardcode the types 
    // as float and double without any abstraction.
    
    float toFloat() const {return to<float>();}     // float32
    double toDouble() const {return to<double>();}  // float64

    // TODO: Add convenience methods for other types:
    
    // ========================================================================
    // Type Checking: Is this scalar a specific type?
    // Useful for conditional logic without try/catch.
    // ========================================================================
    
    bool isFloat32() const {return type_ == ScalarType::Float32;} // float32
    bool isFloat64() const {return type_ == ScalarType::Float64;} // float64
    
    // TODO: Add type checking for other types:

    // ========================================================================
    // Equality Comparison
    // Two scalars are equal if they have the same type AND same value.
    // ========================================================================
    
    bool operator==(const Scalar& other) const {
        // Different types? Not equal.
        if (type_ != other.type_) {return false;}
        
        // Same type, compare values
        // std::visit calls the lambda with both stored values
        return std::visit([](auto&& lhs, auto&& rhs) -> bool {
            using LhsType = std::decay_t<decltype(lhs)>;
            using RhsType = std::decay_t<decltype(rhs)>;
            
            // Both should be the same type (we checked above)
            if constexpr (std::is_same_v<LhsType, RhsType>) {
                return lhs == rhs;
            } else {
                return false;  // Shouldn't happen
            }
        }, data_, other.data_);
    }
    
    // ========================================================================
    // Inequality Comparison
    // Two scalars are not equal if they have different types or different values.
    // ========================================================================
    
    bool operator!=(const Scalar& other) const {return !(*this == other);} // != is the inverse of ==
    
    // ========================================================================
    // Conversion to bool (for conditionals)
    // ========================================================================
    // Allows: if (scalar) { ... }
    // Follows NumPy convention: 0 is false, everything else is true.
    //
    // Example:
    //   Scalar zero(0.0f);
    //   Scalar nonzero(3.14f);
    //   assert(!zero);     // false (zero is falsy)
    //   assert(nonzero);   // true (non-zero is truthy)
    
    explicit operator bool() const {
        return std::visit([](auto&& value) -> bool {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, void>) {
                return false;  // Should never happen, but we handle it anyway.
            } else {
                return value != static_cast<T>(0);
            }
        }, data_);
    }
    
    // ========================================================================
    // Debugging: String representation
    // For internal debugging, not exposed to Python 
    // (Python will have its own if we expose it to Python)
    // ========================================================================
    
    std::string repr() const {
        std::string result = "Scalar(";
        
        std::visit([&result](auto&& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, float>) {
                result += std::to_string(value) + "f";
            } else if constexpr (std::is_same_v<T, double>) {
                result += std::to_string(value) + "d";
            } else if constexpr (!std::is_same_v<T, void>) {
                result += std::to_string(value);
            }
            // TODO: Add other types here.
        }, data_);
        
        result += ", dtype=" + dtypeString() + ")";
        return result;
    }
};

// ============================================================================
// Helper Functions: Create scalars from C++ types
// These factory functions make construction cleaner in some contexts.
// ============================================================================
//
// Instead of: array.add(Scalar(5.0f))
// You can:    array.add(scalar_float32(5.0f))

inline Scalar scalar_float32(float value) { return Scalar(value);}  // float32
inline Scalar scalar_float64(double value) { return Scalar(value);} // float64

// TODO: Add constructors for other types:

}  // namespace xft

#endif  // XFT_SCALAR_H