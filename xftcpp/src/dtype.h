#ifndef XFTCPP_DTYPE_H_
#define XFTCPP_DTYPE_H_

#include <optional>
#include <ostream>
#include <string>

#include "absl/status/statusor.h"
#include "xla/xla_data.pb.h"

namespace xftcpp {

// Data type of array elements.
//
// The Kind enum is intentionally designed to have the same values as
// xla::PrimitiveType to enable efficient zero-cost conversion between the two.
class DType {
 public: 
  // CRITICAL: These values MUST match xla::PrimitiveType exactly.
  // The conversion functions use static_assert to verify this at compile time.
  // If XLA adds new types, we need to add them here with matching values.
  enum Kind {
    // Invalid/uninitialized type
    kInvalid = 0,

    // Boolean (two-state: true/false)
    kPred = 1,

    // Signed integers (fixed width)
    kS2 = 26,   // 2-bit signed (sub-byte)
    kS4 = 21,   // 4-bit signed (sub-byte)
    kS8 = 2,    // 8-bit signed (int8)
    kS16 = 3,   // 16-bit signed (int16)
    kS32 = 4,   // 32-bit signed (int32)
    kS64 = 5,   // 64-bit signed (int64)

    // Unsigned integers (fixed width)
    kU2 = 27,   // 2-bit unsigned (sub-byte)
    kU4 = 22,   // 4-bit unsigned (sub-byte)
    kU8 = 6,    // 8-bit unsigned (uint8)
    kU16 = 7,   // 16-bit unsigned (uint16)
    kU32 = 8,   // 32-bit unsigned (uint32)
    kU64 = 9,   // 64-bit unsigned (uint64)

    // Standard floating-point formats
    kF16 = 10,  // IEEE 754 half precision (16-bit)
    kF32 = 11,  // IEEE 754 single precision (32-bit)
    kF64 = 12,  // IEEE 754 double precision (64-bit)

    // Special floating-point formats
    kBF16 = 16, // Brain Float 16 (truncated F32: 1 sign, 8 exp, 7 mantissa)

    // Complex numbers (pairs of floats)
    kC64 = 15,   // Complex64: pair of F32 (real, imaginary)
    kC128 = 18,  // Complex128: pair of F64 (real, imaginary)

    // Special types
    kToken = 17,  // Token for sequencing side-effecting operations (zero size)
    kOpaque = 14, // Opaque type for backend-specific data

    // Low-precision floating-point formats (for ML/AI)
    kF4E2M1FN = 32,      // 4-bit float (2-bit exponent, 1-bit mantissa)
    kF8E3M4 = 29,        // 8-bit float (3-bit exponent, 4-bit mantissa)
    kF8E4M3 = 28,        // 8-bit float (4-bit exponent, 3-bit mantissa)
    kF8E4M3FN = 20,      // F8 E4M3 with finite-only normalization
    kF8E4M3B11FNUZ = 23, // F8 E4M3 with bias 11, finite, no negative zero
    kF8E4M3FNUZ = 25,    // F8 E4M3 finite, no negative zero
    kF8E5M2 = 19,        // 8-bit float (5-bit exponent, 2-bit mantissa)
    kF8E5M2FNUZ = 24,    // F8 E5M2 finite, no negative zero
    kF8E8M0FNU = 33,     // 8-bit float (8-bit exponent, 0-bit mantissa)

    // Future: If XLA adds types with enum value 34+, add them here

    // String type (variable-length, raw bytes)
    // Not supported by XLA PJRT, but kept for compatibility with some IFRT uses.
    // Uses high enum value (99) to avoid collision with future XLA types.
    kString = 99,
  };

  // ============================================================================
  // Constructors and Assignment
  // ============================================================================

  // Construct a DType from a Kind enum value.
  explicit DType(Kind kind) : kind_(kind) {}

  // Default copy/move semantics (DType is a simple value type)
  DType(const DType&) = default;
  DType(DType&&) = default;
  DType& operator=(const DType&) = default;
  DType& operator=(DType&&) = default;

  // Get the underlying Kind enum value.
  Kind kind() const { return kind_; }

  // ============================================================================
  // Utilities
  // ============================================================================

  // Returns the size of a single element in bytes.
  // Returns std::nullopt if:
  // - Type is not byte-aligned (kS2, kU2, kS4, kU4, kF4E2M1FN)
  // - Type has no fixed size (kString, kToken, kOpaque, kInvalid)
  std::optional<int> byte_size() const;

  // Returns the size of a single element in bits.
  // Returns std::nullopt if the type has no fixed size.
  std::optional<int> bit_size() const;

  // Returns a human-readable string representation.
  // Examples: "F32", "S32", "BF16", "INVALID"
  std::string DebugString() const;

  // ============================================================================
  // Comparison Operators
  // ============================================================================

  bool operator==(const DType& other) const { return kind_ == other.kind_; }
  bool operator!=(const DType& other) const { return kind_ != other.kind_; }

  // Hash support for use in absl containers
  template <typename H>
  friend H AbslHashValue(H h, const DType& value) {
    return H::combine(std::move(h), value.kind());
  }

  // Support for absl::StrCat and absl::StrFormat
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const DType& dtype) {
    sink.Append(dtype.DebugString());
  }

 private:
  Kind kind_;
};

// Stream output operator
std::ostream& operator<<(std::ostream& os, const DType& dtype);

// XLA Conversion Functions (non-member functions in xftcpp namespace)
// Convert DType to xla::PrimitiveType.
// Returns error if the type is not supported by XLA (e.g., kString).
absl::StatusOr<xla::PrimitiveType> ToPrimitiveType(DType dtype);

// Convert xla::PrimitiveType to DType.
// Returns error if the primitive type is not recognized.
absl::StatusOr<DType> FromPrimitiveType(xla::PrimitiveType primitive_type);

}  // namespace xftcpp

#endif  // XFTCPP_DTYPE_H_