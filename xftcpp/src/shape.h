/* XFT-CPP Shape
 *
 * Defines shape types for arrays:
 * - Shape: Static shape with fixed dimensions
 * - DynamicShape: Shape with runtime-variable dimensions
 */

#ifndef XFTCPP_SHAPE_H_
#define XFTCPP_SHAPE_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <variant>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/types/span.h"

namespace xftcpp {

// ============================================================================
// Shape - Static shape with fixed dimensions
// ============================================================================

// Shape of an array with statically known dimensions.
// Every dimension size must be >= 0.
//
// Uses InlinedVector to avoid heap allocation for small shapes (up to 6 dims).
// Most ML models have 2-4 dimensions, so this optimization helps performance.
class Shape {
 public:
  // Inline up to 6 dimensions to avoid heap allocation for common cases.
  // Examples: [batch, height, width, channels] = 4 dims (fits inline)
  static constexpr int kInlineDimensionSize = 6;

  using Dimensions = absl::InlinedVector<int64_t, kInlineDimensionSize>;

  // Construct from a span of dimensions
  explicit Shape(absl::Span<const int64_t> dims)
      : dims_(Dimensions(dims.begin(), dims.end())) {}

  // Default copy/move semantics
  Shape(const Shape&) = default;
  Shape(Shape&&) = default;
  Shape& operator=(const Shape&) = default;
  Shape& operator=(Shape&&) = default;

  // Get dimensions as a span (read-only view)
  absl::Span<const int64_t> dims() const { return dims_; }

  // Total number of elements in this shape (product of all dimensions)
  // Example: Shape([2, 3, 4]) has 2*3*4 = 24 elements
  int64_t num_elements() const;

  // Human-readable string representation
  // Example: Shape([2, 3, 4]) -> "2,3,4"
  std::string DebugString() const;

  // ============================================================================
  // Comparison and Hashing
  // ============================================================================

  bool operator==(const Shape& other) const { return dims_ == other.dims_; }
  bool operator!=(const Shape& other) const { return dims_ != other.dims_; }

  template <typename H>
  friend H AbslHashValue(H h, const Shape& shape);

  // Support for absl::StrCat and absl::StrFormat
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Shape& shape) {
    sink.Append(shape.DebugString());
  }

  // TODO(future): Add FromProto/ToProto when we need serialization

 private:
  Dimensions dims_;
};

// Hash support for using Shape in absl containers
template <typename H>
H AbslHashValue(H h, const Shape& shape) {
  return H::combine(std::move(h), shape.dims_);
}

// ============================================================================
// BoundedDynamicShapeTag - Marks which dimensions are dynamic
// ============================================================================

// A tag indicating which dimensions in a Shape are dynamically sized.
// Used together with Shape to create a DynamicShape.
//
// Example: Shape([10, -1, 256]) with tag [false, true, false]
// means the second dimension can vary at runtime (up to bound of -1).
//
// This is for "bounded dynamism" - dimensions can vary but have an upper bound.
class BoundedDynamicShapeTag {
 public:
  static constexpr int kInlineDimensionSize = 6;

  using DynamicDimensions = absl::InlinedVector<bool, kInlineDimensionSize>;

  // Construct from a span of bools indicating which dims are dynamic
  // At least one dimension must be dynamic (otherwise use static Shape)
  explicit BoundedDynamicShapeTag(absl::Span<const bool> dynamic_dims)
      : dynamic_dims_(
            DynamicDimensions(dynamic_dims.begin(), dynamic_dims.end())) {
    CHECK(absl::c_any_of(dynamic_dims_, [](bool b) { return b; }))
        << "At least one dimension needs to be dynamically sized.";
  }

  // Default copy/move semantics
  BoundedDynamicShapeTag(const BoundedDynamicShapeTag&) = default;
  BoundedDynamicShapeTag(BoundedDynamicShapeTag&&) = default;
  BoundedDynamicShapeTag& operator=(const BoundedDynamicShapeTag&) = default;
  BoundedDynamicShapeTag& operator=(BoundedDynamicShapeTag&&) = default;

  // Get the dynamic dimension flags
  absl::Span<const bool> DynamicDims() const { return dynamic_dims_; }

  bool operator==(const BoundedDynamicShapeTag& other) const {
    return dynamic_dims_ == other.dynamic_dims_;
  }

  bool operator!=(const BoundedDynamicShapeTag& other) const {
    return !(*this == other);
  }

  template <typename H>
  friend H AbslHashValue(H h, const BoundedDynamicShapeTag& value) {
    return H::combine(std::move(h), value.dynamic_dims_);
  }

  // TODO(future): Add FromProto/ToProto when we need serialization

 private:
  // Same length as Shape's dims(), indicates which dimensions are dynamic
  DynamicDimensions dynamic_dims_;
};

// ============================================================================
// DynamicShapeTag - Polymorphic tag for different dynamism types
// ============================================================================

// Static polymorphism for different types of dynamism.
// Currently only supports bounded dynamism, but designed for extensibility.
// Future: Could add UnboundedDynamicShapeTag, SparseDynamicShapeTag, etc.
using DynamicShapeTag = std::variant<BoundedDynamicShapeTag>;

// ============================================================================
// DynamicShape - Shape with runtime-variable dimensions
// ============================================================================

// Shape with dynamism in dimension sizes.
//
// For bounded dynamic shapes:
// - The Shape stores the upper bounds for dynamic dimensions
// - The Tag indicates which dimensions are dynamic
// - Actual size is determined at runtime (up to the bound)
class DynamicShape {
 public:
  // Constructs DynamicShape from Shape and DynamicShapeTag.
  // Returns error if dimensions mismatch between shape and tag.
  static absl::StatusOr<DynamicShape> Create(Shape shape, DynamicShapeTag tag);

  // Default copy/move semantics
  DynamicShape(const DynamicShape&) = default;
  DynamicShape(DynamicShape&&) = default;
  DynamicShape& operator=(const DynamicShape&) = default;
  DynamicShape& operator=(DynamicShape&&) = default;

  // Get the dynamism tag
  const DynamicShapeTag& GetTag() const { return tag_; }

  // Get the shape after padding to maximum bounds
  // For bounded dynamic shapes, returns the Shape with bounds
  absl::StatusOr<Shape> GetPaddedShape() const;

  // Check if a specific dimension is dynamic
  // dimension: index into the shape's dimensions
  bool IsDynamicDim(int dimension) const;

  // ============================================================================
  // Comparison and Hashing
  // ============================================================================

  bool operator==(const DynamicShape& other) const {
    return tag_ == other.tag_ && shape_ == other.shape_;
  }
  bool operator!=(const DynamicShape& other) const { return !(*this == other); }

  template <typename H>
  friend H AbslHashValue(H h, const DynamicShape& value) {
    return H::combine(std::move(h), value.shape_,
                      std::get<BoundedDynamicShapeTag>(value.tag_));
  }

  // Human-readable string representation
  std::string DebugString() const;

  // TODO(future): Add FromProto/ToProto when we need serialization

 private:
  // Private constructor - use Create() factory method
  DynamicShape(Shape shape, DynamicShapeTag tag)
      : shape_(std::move(shape)), tag_(std::move(tag)) {}

  Shape shape_;          // Upper bounds for dynamic dimensions
  DynamicShapeTag tag_;  // Indicates which dimensions are dynamic
};

// ============================================================================
// Stream Output Operators
// ============================================================================

std::ostream& operator<<(std::ostream& os, const Shape& shape);
std::ostream& operator<<(std::ostream& os, const DynamicShape& dynamic_shape);

}  // namespace xftcpp

#endif  // XFTCPP_SHAPE_H_