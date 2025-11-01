/* Index Domain for xftcpp
 * Adapted from XLA's IFRT IndexDomain
 * 
 * Represents a rectangular region (slice) in a multi-dimensional array.
 * Think of it as describing "from where to where" a piece of data spans.
 * 
 * Key concept: IndexDomain = origin (starting position) + shape (size)
 * 
 * Example for a 2D array:
 *   IndexDomain(origin=[2,3], shape=[4,5])
 *   means: starts at row 2, col 3 and spans 4 rows and 5 cols
 *   covers: rows 2-5 (inclusive-exclusive), cols 3-7 (inclusive-exclusive)
 * 
 * Why this matters for sharding:
 * When a tensor is split across devices, each device owns a rectangular slice.
 * IndexDomain describes which slice each device owns.
 */

#ifndef XFTCPP_INDEX_DOMAIN_H_
#define XFTCPP_INDEX_DOMAIN_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"

#include "xftcpp/src/index.h"
#include "xftcpp/src/shape.h"

namespace xftcpp {

// ============================================================================
// IndexDomain - A rectangular region in multi-dimensional space
// ============================================================================

// Represents a slice of a multi-dimensional array.
//
// An IndexDomain is defined by two components:
// 1. origin: The starting position (lower inclusive bound)
// 2. shape: The size of the region
//
// Mathematical interpretation:
//   For dimension i, the domain spans [origin[i], origin[i] + shape[i])
//   (inclusive lower bound, exclusive upper bound)
//
// Examples:
//
//   1D case:
//     IndexDomain(origin=[5], shape=[10])
//     -> covers indices 5, 6, 7, ..., 14 (10 elements starting at 5)
//
//   2D case:
//     IndexDomain(origin=[2,3], shape=[4,5])
//     -> covers rows [2,6) and cols [3,8)
//     -> a 4x5 rectangle starting at position (2,3)
//
//   3D case:
//     IndexDomain(origin=[0,10,20], shape=[8,16,32])
//     -> covers depth [0,8), rows [10,26), cols [20,52)
//
// Use cases in sharding:
//
//   When a [100, 200] array is split across 4 devices in a 2x2 grid:
//     Device 0: IndexDomain([0,0],   [50,100])  - top-left
//     Device 1: IndexDomain([0,100], [50,100])  - top-right
//     Device 2: IndexDomain([50,0],  [50,100])  - bottom-left
//     Device 3: IndexDomain([50,100],[50,100])  - bottom-right
//
// Why header-only?
//   - Simple value type with trivial operations
//   - All methods are one-liners or inline-worthy
//   - Eliminates one .cpp file to maintain
//   - Allows better compiler optimization
class IndexDomain {
 public:
  // ============================================================================
  // Constructors
  // ============================================================================

  // General IndexDomain construction with explicit origin and shape.
  //
  // Parameters:
  //   origin: Starting position of the domain in each dimension
  //   shape: Size of the domain in each dimension
  //
  // REQUIRES: origin.elements().size() == shape.dims().size()
  //
  // Example:
  //   IndexDomain(Index([2,3]), Shape([4,5]))
  //   -> 4x5 region starting at (2,3)
  IndexDomain(Index origin, Shape shape)
      : origin_(std::move(origin)), shape_(std::move(shape)) {}

  // IndexDomain construction with a zero origin (starts at [0,0,...,0]).
  //
  // This is a convenience constructor for domains that start at the beginning.
  // Common use case: describing the full array shape as a domain.
  //
  // Example:
  //   IndexDomain(Shape([100,200]))
  //   -> same as IndexDomain(Index([0,0]), Shape([100,200]))
  //   -> covers the entire 100x200 array
  explicit IndexDomain(Shape shape)
      : origin_(Index::Zeros(shape.dims().size())), shape_(std::move(shape)) {}

  // Default copy/move - compiler-generated is fine for value semantics
  IndexDomain(const IndexDomain&) = default;
  IndexDomain(IndexDomain&&) = default;
  IndexDomain& operator=(const IndexDomain&) = default;
  IndexDomain& operator=(IndexDomain&&) noexcept = default;

  // ============================================================================
  // Accessors
  // ============================================================================

  // Get the starting position of this domain.
  const Index& origin() const { return origin_; }

  // Get the size of this domain.
  const Shape& shape() const { return shape_; }

  // ============================================================================
  // Comparison Operators
  // ============================================================================

  bool operator==(const IndexDomain& other) const {
    return origin_ == other.origin_ && shape_ == other.shape_;
  }

  bool operator!=(const IndexDomain& other) const {
    return origin_ != other.origin_ || shape_ != other.shape_;
  }

  // ============================================================================
  // Arithmetic Operators
  // ============================================================================

  // Shift the domain by an offset (moves the origin, keeps the shape).
  //
  // Example:
  //   IndexDomain([2,3], [4,5]) + Index([10,20])
  //   -> IndexDomain([12,23], [4,5])
  //   The domain moves but keeps the same size.
  //
  // Use case: Converting between coordinate systems or adjusting slices.
  IndexDomain operator+(const Index& offset) const {
    return IndexDomain(origin_ + offset, shape_);
  }

  // Shift the domain by a negative offset.
  //
  // Example:
  //   IndexDomain([12,23], [4,5]) - Index([10,20])
  //   -> IndexDomain([2,3], [4,5])
  IndexDomain operator-(const Index& offset) const {
    return IndexDomain(origin_ - offset, shape_);
  }

  // In-place shift operators (modify this domain's origin)
  IndexDomain& operator+=(const Index& offset) {
    origin_ += offset;
    return *this;
  }

  IndexDomain& operator-=(const Index& offset) {
    origin_ -= offset;
    return *this;
  }

  // ============================================================================
  // String Representation
  // ============================================================================

  // Returns a human-readable string representation.
  // Format: "IndexDomain(origin=[...],shape=[...])"
  //
  // Example:
  //   IndexDomain([2,3], [4,5]).DebugString()
  //   -> "IndexDomain(origin=[2,3],shape=[4,5])"
  std::string DebugString() const {
    return absl::StrCat("IndexDomain(origin=", origin_.DebugString(),
                        ",shape=", shape_.DebugString(), ")");
  }

  // Support for absl::StrCat and absl::StrFormat
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const IndexDomain& index_domain) {
    sink.Append(index_domain.DebugString());
  }

  // ============================================================================
  // Hashing Support
  // ============================================================================

  // Support for absl hash containers (flat_hash_map, flat_hash_set)
  //
  // Allows using IndexDomain as a key in hash maps/sets:
  //   absl::flat_hash_map<IndexDomain, Value> domain_map;
  template <typename H>
  friend H AbslHashValue(H h, const IndexDomain& index_domain) {
    return H::combine(std::move(h), index_domain.origin_, index_domain.shape_);
  }

 private:
  // Starting position of this domain in each dimension
  Index origin_;
  
  // Size of this domain in each dimension
  Shape shape_;
};

// ============================================================================
// Stream Output Operator
// ============================================================================

// Allows printing IndexDomain to output streams.
// Example: std::cout << index_domain << std::endl;
inline std::ostream& operator<<(std::ostream& os, 
                                const IndexDomain& index_domain) {
  return os << index_domain.DebugString();
}

}  // namespace xftcpp

#endif  // XFTCPP_INDEX_DOMAIN_H_

