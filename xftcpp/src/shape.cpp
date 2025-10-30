/* XFT-CPP Shape - Implementation
 *
 * Implements Shape and DynamicShape functionality.
 * Proto serialization methods removed - marked with TODO for future addition.
 */

#include "xftcpp/src/shape.h"

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <variant>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

namespace xftcpp {

namespace {

// Helper for std::visit with multiple lambdas (C++17 pattern)
// Allows us to handle different variant types cleanly
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

// Explicit deduction guide (required for C++17)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

}  // namespace

// ============================================================================
// Shape Implementation
// ============================================================================

// TODO(future): Implement Shape::FromProto when serialization is needed
// absl::StatusOr<Shape> Shape::FromProto(const ShapeProto& proto);

// TODO(future): Implement Shape::ToProto when serialization is needed
// ShapeProto Shape::ToProto() const;

int64_t Shape::num_elements() const {
  int64_t count = 1;
  for (int64_t d : dims_) {
    count *= d;
  }
  return count;
}

std::string Shape::DebugString() const {
  return absl::StrCat("[", absl::StrJoin(dims_, ","), "]");
}

// ============================================================================
// BoundedDynamicShapeTag Implementation
// ============================================================================

// TODO(future): Implement BoundedDynamicShapeTag::FromProto when needed
// absl::StatusOr<BoundedDynamicShapeTag> BoundedDynamicShapeTag::FromProto(
//     const BoundedDynamicShapeTagProto& proto);

// TODO(future): Implement BoundedDynamicShapeTag::ToProto when needed
// BoundedDynamicShapeTagProto BoundedDynamicShapeTag::ToProto() const;

// ============================================================================
// DynamicShape Implementation
// ============================================================================

absl::StatusOr<DynamicShape> DynamicShape::Create(Shape shape,
                                                   DynamicShapeTag tag) {
  // Validate that shape and tag have matching dimensions
  // Use std::visit to handle different tag types (currently only BoundedDynamicShapeTag)
  absl::Status validation_status = std::visit(
      overloaded{
          [&](const BoundedDynamicShapeTag& tag) -> absl::Status {
            if (tag.DynamicDims().size() != shape.dims().size()) {
              return absl::InvalidArgumentError(
                  "Shape and tag must have the same number of dimensions.");
            }
            return absl::OkStatus();
          },
      },
      tag);

  if (!validation_status.ok()) {
    return validation_status;
  }

  return DynamicShape(std::move(shape), std::move(tag));
}

absl::StatusOr<Shape> DynamicShape::GetPaddedShape() const {
  // For bounded dynamic shapes, return the shape with upper bounds
  // The shape_ already contains the maximum bounds for dynamic dimensions
  return std::visit(
      overloaded{
          [this](const BoundedDynamicShapeTag& tag) { return shape_; },
      },
      tag_);
}

bool DynamicShape::IsDynamicDim(int dimension) const {
  // Check if a specific dimension is dynamically sized
  return std::visit(
      overloaded{
          [dimension](const BoundedDynamicShapeTag& tag) {
            return tag.DynamicDims().at(dimension);
          },
      },
      tag_);
}

// TODO(future): Implement DynamicShape::FromProto when serialization is needed
// absl::StatusOr<DynamicShape> DynamicShape::FromProto(
//     const DynamicShapeProto& proto);

// TODO(future): Implement DynamicShape::ToProto when serialization is needed
// DynamicShapeProto DynamicShape::ToProto() const;

std::string DynamicShape::DebugString() const {
  // Format: [<=10, 5, <=256] where "<=" prefix indicates dynamic dimensions
  return std::visit(
      overloaded{
          [this](const BoundedDynamicShapeTag& tag) {
            absl::InlinedVector<std::string, Shape::kInlineDimensionSize>
                dim_reps;
            dim_reps.reserve(shape_.dims().size());
            for (int i = 0; i < shape_.dims().size(); ++i) {
              // Add "<=" prefix for dynamic dimensions
              absl::string_view prefix = tag.DynamicDims()[i] ? "<=" : "";
              dim_reps.push_back(absl::StrCat(prefix, shape_.dims()[i]));
            }
            return absl::StrCat("[", absl::StrJoin(dim_reps, ","), "]");
          }},
      tag_);
}

// ============================================================================
// Stream Output Operators
// ============================================================================

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
  return os << shape.DebugString();
}

std::ostream& operator<<(std::ostream& os, const DynamicShape& dynamic_shape) {
  return os << dynamic_shape.DebugString();
}

}  // namespace xftcpp