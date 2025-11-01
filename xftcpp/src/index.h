/* Multi-dimensional Index for xftcpp
 * Adapted from XLA's IFRT Index
 * 
 * Represents a position in a multi-dimensional array/tensor.
 * Think of it like coordinates: Index([2, 3, 4]) means row 2, column 3, depth 4.
 * 
 * Key features:
 * - Inline storage for up to 6 dimensions (avoids heap allocation for common cases)
 * - Element-wise arithmetic operations
 * - Zero-cost in release builds (all inline)
 */

#ifndef XFTCPP_INDEX_H_
#define XFTCPP_INDEX_H_

#include <cstdint>
#include <ostream>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"

namespace xftcpp {

// ============================================================================
// Index - Multi-dimensional array index
// ============================================================================

// Represents a multi-dimensional index into an array.
//
// An Index is essentially a vector of non-negative integers where each element
// represents a position along one dimension.
//
// Examples:
//   Index([0, 0])       -> Top-left corner of a 2D array
//   Index([5, 10, 3])   -> Position at row 5, col 10, depth 3 in 3D array
//   Index([])           -> Scalar (0-dimensional)
//
// Design choices:
// 1. Uses InlinedVector to avoid heap allocation for <= 6 dimensions
//    Most ML tensors have 2-4 dimensions (batch, height, width, channels)
// 2. All operations are inline for zero-cost abstraction
// 3. Every element must be >= 0 (indices can't be negative)
//
// Common operations:
//   - Addition: index + offset (element-wise)
//   - Subtraction: index - offset (element-wise)
//   - Scaling: index * multiplier (element-wise)
class Index {
 public:
  // Maximum elements to inline before allocating on the heap.
  // 6 is chosen to cover most common cases:
  // - 2D: [H, W]
  // - 3D: [D, H, W]
  // - 4D: [N, H, W, C] (batch, height, width, channels)
  // - 5D: [N, T, H, W, C] (batch, time, height, width, channels)
  // - 6D: [N, T, D, H, W, C] (batch, time, depth, height, width, channels)
  static constexpr int kInlineElementSize = 6;

  // Storage type: inlined vector that avoids heap allocation for small sizes
  using Elements = absl::InlinedVector<int64_t, kInlineElementSize>;

  // ============================================================================
  // Constructors
  // ============================================================================

  // Construct from a span of elements.
  // Example: Index({2, 3, 4}) creates a 3D index
  //
  // Note: absl::InlinedVector can implicitly convert to Span, so you can
  // also pass an Elements vector directly to this constructor.
  explicit Index(absl::Span<const int64_t> elements)
      : elements_(Elements(elements.begin(), elements.end())) {}

  // Create a zero index with the given number of elements.
  // Example: Index::Zeros(3) -> Index([0, 0, 0])
  // 
  // Use case: Starting point for iteration or as a default origin
  //
  // Implementation note: Creates an Elements vector and implicitly converts
  // it to Span for the constructor. This works because absl::InlinedVector
  // can be implicitly converted to absl::Span.
  static Index Zeros(int num_elements) {
    return Index(Elements(/*n=*/num_elements, /*value=*/0));
  }

  // Default copy/move - compiler-generated is fine for value semantics
  Index(const Index&) = default;
  Index(Index&&) = default;
  Index& operator=(const Index&) = default;
  Index& operator=(Index&&) = default;

  // ============================================================================
  // Accessors
  // ============================================================================

  // Get the elements as a read-only span.
  // Returns a view into the underlying storage - no copy.
  absl::Span<const int64_t> elements() const { return elements_; }

  // ============================================================================
  // Comparison Operators
  // ============================================================================

  bool operator==(const Index& other) const {
    return elements_ == other.elements_;
  }

  bool operator!=(const Index& other) const {
    return elements_ != other.elements_;
  }

  // ============================================================================
  // Arithmetic Operators
  // ============================================================================

  // Element-wise addition.
  //
  // Example: Index([1, 2, 3]) + Index([10, 20, 30]) -> Index([11, 22, 33])
  //
  // Use case: Computing positions after applying an offset
  //   origin + offset = absolute_position
  //
  // REQUIRES: this->elements().size() == offset.elements().size()
  Index operator+(const Index& offset) const {
    CHECK_EQ(elements_.size(), offset.elements_.size());
    Index result = *this;
    for (size_t i = 0; i < elements_.size(); ++i) {
      result.elements_[i] += offset.elements_[i];
    }
    return result;
  }

  // Element-wise subtraction.
  //
  // Example: Index([11, 22, 33]) - Index([10, 20, 30]) -> Index([1, 2, 3])
  //
  // Use case: Computing relative offset between two positions
  //   absolute_position - origin = offset
  //
  // REQUIRES: this->elements().size() == offset.elements().size()
  Index operator-(const Index& offset) const {
    CHECK_EQ(elements_.size(), offset.elements_.size());
    Index result = *this;
    for (size_t i = 0; i < elements_.size(); ++i) {
      result.elements_[i] -= offset.elements_[i];
    }
    return result;
  }

  // Element-wise multiplication by a span.
  //
  // Example: Index([2, 3, 4]) * {10, 100, 1000} -> Index([20, 300, 4000])
  //
  // Use case: Converting logical indices to byte offsets or stride calculations
  //   logical_index * strides = linear_offset
  //
  // Note: Takes a span instead of another Index because multipliers are often
  // shape dimensions or strides, which are just spans of int64_t.
  //
  // REQUIRES: this->elements().size() == multiplier.size()
  Index operator*(absl::Span<const int64_t> multiplier) const {
    CHECK_EQ(elements_.size(), multiplier.size());
    Index result = *this;
    for (size_t i = 0; i < elements_.size(); ++i) {
      result.elements_[i] *= multiplier[i];
    }
    return result;
  }

  // Compound assignment operators (modify this index in-place)
  Index& operator+=(const Index& offset) { 
    return *this = *this + offset; 
  }

  Index& operator-=(const Index& offset) { 
    return *this = *this - offset; 
  }

  Index& operator*=(absl::Span<const int64_t> multiplier) {
    return *this = *this * multiplier;
  }

  // ============================================================================
  // String Representation
  // ============================================================================

  // Returns a human-readable string representation.
  // Format: "[e0,e1,e2,...]"
  // 
  // Examples:
  //   Index([2, 3, 4]).DebugString() -> "[2,3,4]"
  //   Index([]).DebugString() -> "[]"
  std::string DebugString() const {
    return absl::StrCat("[", absl::StrJoin(elements_, ","), "]");
  }

  // Support for absl::StrCat and absl::StrFormat
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Index& index) {
    sink.Append(index.DebugString());
  }

  // ============================================================================
  // Hashing Support
  // ============================================================================

  // Support for absl hash containers (flat_hash_map, flat_hash_set)
  //
  // Allows using Index as a key in hash maps/sets:
  //   absl::flat_hash_map<Index, Value> index_map;
  template <typename H>
  friend H AbslHashValue(H h, const Index& index) {
    return H::combine(std::move(h), index.elements_);
  }

 private:
  // The actual index elements.
  // InlinedVector stores up to kInlineElementSize elements inline,
  // avoiding heap allocation for the common case.
  Elements elements_;
};

// ============================================================================
// Stream Output Operator
// ============================================================================

// Allows printing Index to output streams.
// Example: std::cout << index << std::endl;
inline std::ostream& operator<<(std::ostream& os, const Index& index) {
  return os << index.DebugString();
}

}  // namespace xftcpp

#endif  // XFTCPP_INDEX_H_

