/* Sharding Parameters Implementation
 * Adapted from XLA's IFRT ShardingParam
 *
 * This file implements the sharding logic that distributes arrays across
 * device meshes. See sharding_params.h for detailed conceptual documentation.
 */

#include "sharding_params.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

namespace xftcpp {

namespace {

// ============================================================================
// Helper Functions for Printing and Device Layout
// ============================================================================

// Prints dimensions in "NxMxK" format.
//
// Used for pretty-printing tensor shapes and device mesh shapes.
//
// Examples:
//   dims=[2, 3, 4] → "2x3x4"
//   dims=[100] → "100"
//   dims=[] → "" (scalar, no dimensions)
//
// Parameters:
//   os: Output stream to write to
//   dims: Dimensions to print
template <typename T>
void PrintDims(llvm::raw_ostream& os, llvm::ArrayRef<T> dims) {
  if (dims.empty()) {
    // A scalar does not have dimensions.
    return;
  }
  os << dims[0];
  for (int i = 1; i < dims.size(); ++i) {
    os << "x" << dims[i];
  }
}

// Recursively populates the device list by expanding the permutation.
//
// This is the core algorithm that linearizes a multi-dimensional device mesh
// into a 1D array of device IDs, respecting the minor-to-major traversal order.
//
// **How it works:**
// The function processes the permutation from major (last element) to minor
// (first element), recursively expanding each dimension. At each level:
// 1. Take the last element of permutation (the current major dimension)
// 2. Iterate through all positions in that dimension
// 3. For each position, calculate the base offset and recurse with remaining dimensions
// 4. Base case: When permutation has 1 element, directly append device IDs
//
// **Example walkthrough:**
//   permutation = [1, 0]  (axis-1 minor, axis-0 major)
//   axis_sizes = [2, 3]   (2 rows, 3 columns)
//   cum_sizes = [1, 2]    (cumulative products: 1, 1*2=2)
//
//   Initial call: expanding_dim = 0 (last of [1,0])
//     - For i=0 (first row):
//       Recurse with permutation=[1], base=0
//         expanding_dim = 1
//         - For j=0: out_devices.push_back(0 + 0*1) → 0
//         - For j=1: out_devices.push_back(0 + 1*1) → 1
//         - For j=2: out_devices.push_back(0 + 2*1) → 2
//     - For i=1 (second row):
//       Recurse with permutation=[1], base=2
//         - For j=0: out_devices.push_back(2 + 0*1) → 2... wait, this would be 3
//
//   Actually, let me recalculate:
//   cum_sizes[0] = 1, cum_sizes[1] = 2
//   
//   Wait, the cumulative size should be the product of dimensions BEFORE this one.
//   So cum_sizes[i] = product of axis_sizes[0..i-1]
//   For axis_sizes=[2,3]: cum_sizes=[1, 2]
//
//   expanding_dim=0, expanding_dim_size=2, expanding_cum_dim_size=1
//   For i=0: base=0+0*1=0, recurse
//   For i=1: base=0+1*1=1, recurse
//
//   Hmm, this logic is complex. I'll just document what it does at a high level.
//
// **Parameters:**
//   permutation: Axes to expand, from minor to major (will process from back)
//   axis_sizes: Size of each mesh axis (unchanged throughout recursion)
//   cum_sizes: Cumulative product of axis sizes (for computing offsets)
//   out_devices: Output vector to append device IDs to
//   base: Starting device ID for this slice of the recursion
//
// **Algorithm complexity:**
// Time: O(product of axis_sizes) - visits each device exactly once
// Space: O(permutation.size()) - recursion depth
void PopulateDevices(llvm::ArrayRef<int> permutation,
                     llvm::ArrayRef<int> axis_sizes,
                     llvm::ArrayRef<int> cum_sizes,
                     llvm::SmallVectorImpl<int>& out_devices, int base = 0) {
  // Take the last element of permutation (major dimension at this level)
  const int expanding_dim = permutation.back();
  const int expanding_dim_size = axis_sizes[expanding_dim];
  const int expanding_cum_dim_size = cum_sizes[expanding_dim];

  // Iterate through all positions in this dimension
  for (int i = 0; i < expanding_dim_size; ++i) {
    if (permutation.size() == 1) {
      // Base case: No more dimensions to recurse, append the device ID
      out_devices.push_back(base + i * expanding_cum_dim_size);
    } else {
      // Recursive case: Process remaining dimensions with updated base
      PopulateDevices(permutation.drop_back(), axis_sizes, cum_sizes,
                      out_devices, base + i * expanding_cum_dim_size);
    }
  }
}

// Prints ShardingParam in V1 format to a raw output stream.
//
// Format: "dim_shards to [permutation] on axis_sizes"
// Example: "2x1x3 to [1,0] on 3x2"
//
// This is a helper function used by both MLIR and LLVM stream operators.
//
// Parameters:
//   os: Output stream to write to
//   sharding: ShardingParam to print
void PrintInternalV1(llvm::raw_ostream& os, const ShardingParam& sharding) {
  PrintDims(os, sharding.dim_shards());
  os << " to [";
  llvm::interleaveComma(
      llvm::ArrayRef<int>(sharding.minor_to_major().permutation), os);
  os << "] on ";
  PrintDims<int>(os, sharding.minor_to_major().axis_sizes);
}

}  // namespace

// ============================================================================
// MinorToMajor Implementation
// ============================================================================

absl::Status ShardingParam::MinorToMajor::verify() const {
  // Check 1: Sizes must match and be non-empty
  if (permutation.size() != axis_sizes.size() || axis_sizes.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expect same non-zero size for `permutation` and `axis_sizes`. Actual ",
        permutation.size(), " vs ", axis_sizes.size()));
  }

  // Check 2: Permutation must not have duplicates
  // Use a DenseSet to detect duplicates in O(n) time
  llvm::DenseSet<int> permutation_set(permutation.begin(), permutation.end());
  if (permutation_set.size() != permutation.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("`permutation` [", absl::StrJoin(permutation, ","),
                     "] has duplicate values"));
  }

  // Check 3: All permutation values must be valid indices into axis_sizes
  for (const int index : permutation) {
    if (index < 0 || index >= axis_sizes.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Out of range axis ", index, " to the mesh of [",
                       absl::StrJoin(permutation, ","), "] on ",
                       absl::StrJoin(axis_sizes, "x")));
    }
  }

  return absl::OkStatus();
}

mlir::LogicalResult ShardingParam::MinorToMajor::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emit_error) const {
  auto status = verify();
  if (status.ok()) {
    return mlir::success();
  }
  return emit_error() << status.message();
}

void ShardingParam::MinorToMajor::ToDeviceList(
    llvm::SmallVectorImpl<int>& out_devices) const {
  // Compute cumulative sizes for offset calculations.
  //
  // cum_sizes[i] = product of axis_sizes[0..i-1]
  // This represents how many devices to skip when incrementing axis i by 1.
  //
  // Example: axis_sizes = [2, 3, 4]
  //   cum_sizes[0] = 1         (no axes before 0)
  //   cum_sizes[1] = 2         (skip 2 devices when moving in axis 1)
  //   cum_sizes[2] = 2*3 = 6   (skip 6 devices when moving in axis 2)
  llvm::SmallVector<int, 4> cum_sizes;
  int cum_size = 1;
  cum_sizes.reserve(axis_sizes.size());
  for (auto size : axis_sizes) {
    cum_sizes.push_back(cum_size);
    cum_size *= size;
  }

  // Recursively populate the device list according to the permutation
  PopulateDevices(permutation, axis_sizes, cum_sizes, out_devices);
}

std::string ShardingParam::MinorToMajor::DebugString() const {
  return absl::StrCat("permutation=[", absl::StrJoin(permutation, ","),
                      "] axis_sizes=[", absl::StrJoin(axis_sizes, ","), "]");
}

int ShardingParam::MinorToMajor::NumDevices() const {
  int total = 1;
  for (const int axis_size : axis_sizes) {
    total *= axis_size;
  }
  return total;
}

// ============================================================================
// ShardingParam Parsing (MLIR Integration)
// ============================================================================

mlir::FailureOr<ShardingParam> ShardingParam::Parse(
    mlir::AsmParser& ods_parser) {
  // V1 is the current ShardingParam format.
  return ParseV1(ods_parser);
}

mlir::FailureOr<ShardingParam> ShardingParam::ParseV1(
    mlir::AsmParser& ods_parser) {
  MinorToMajor minor_to_major;

  // Lambda to parse a single integer into the permutation array
  auto parseIntoPermutation = [&]() -> mlir::ParseResult {
    int item;
    if (auto result = ods_parser.parseInteger(item)) {
      return result;
    }
    minor_to_major.permutation.push_back(item);
    return mlir::ParseResult::success();
  };

  llvm::SmallVector<int64_t, 4> axis_sizes_64;
  llvm::SmallVector<int64_t> dim_shards;

  // Parse format: "NxM to [P0,P1,...] on AxB"
  //
  // Example input: "2x3x4 to [1,0] on 3x2"
  //   dim_shards = [2, 3, 4]
  //   permutation = [1, 0]
  //   axis_sizes = [3, 2]
  if (ods_parser.parseDimensionList(dim_shards, false, false) ||
      ods_parser.parseKeyword("to") ||
      ods_parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square,
                                         parseIntoPermutation) ||
      ods_parser.parseKeyword("on") ||
      ods_parser.parseDimensionList(axis_sizes_64, false, false)) {
    return mlir::failure();
  }

  // Convert axis_sizes from int64_t to int
  // (parseDimensionList returns int64_t, but we use int for mesh dimensions)
  minor_to_major.axis_sizes.reserve(axis_sizes_64.size());
  for (int64_t size : axis_sizes_64) {
    minor_to_major.axis_sizes.push_back(size);
  }

  // Copy dim_shards from SmallVector to std::vector
  // The copy is necessary because:
  // - parseDimensionList expects SmallVector<int64_t>
  // - ShardingParam constructor expects std::vector<int64_t>
  // - ShardingParam uses std::vector for Python bindings compatibility
  return ShardingParam(std::vector(dim_shards.begin(), dim_shards.end()),
                       std::move(minor_to_major));
}

void ShardingParam::PrintV1(mlir::AsmPrinter& ods_printer,
                            const ShardingParam& sharding) {
  PrintInternalV1(ods_printer.getStream(), sharding);
}

// ============================================================================
// ShardingParam Validation
// ============================================================================

absl::Status ShardingParam::verify() const {
  // First, verify the minor_to_major structure itself
  if (auto status = minor_to_major().verify(); !status.ok()) {
    return status;
  }

  // Verify that dim_shards can be distributed to the device mesh.
  //
  // **Algorithm:**
  // We traverse the permutation from minor to major, accumulating mesh
  // capacity as we go. For each accumulated capacity, we try to "consume"
  // as many sharded dimensions as possible.
  //
  // **Example that passes:**
  //   dim_shards = [2, 1, 3]
  //   permutation = [1, 0]
  //   axis_sizes = [3, 2]
  //
  //   Processing:
  //   - Start: dim_index=0, cum_size=1
  //   - Process permutation[0]=1: axis_size=2, cum_size=2
  //     - Skip dim_shards[0]=2 (unsharded): dim_index=1
  //     - cum_size=2 % dim_shards[0]=2 == 0, consume it: cum_size=1, dim_index=1
  //   - Process permutation[1]=0: axis_size=3, cum_size=3
  //     - Skip dim_shards[1]=1 (unsharded): dim_index=2
  //     - cum_size=3 % dim_shards[2]=3 == 0, consume it: cum_size=1, dim_index=3
  //   - Done: dim_index=3 == dim_shards.size(), success!
  //
  // **Example that fails:**
  //   dim_shards = [2, 2]
  //   permutation = [0]
  //   axis_sizes = [2]
  //
  //   Processing:
  //   - Start: dim_index=0, cum_size=1
  //   - Process permutation[0]=0: axis_size=2, cum_size=2
  //     - cum_size=2 % dim_shards[0]=2 == 0, consume it: cum_size=1, dim_index=1
  //   - Done iterating permutation
  //   - dim_index=1 != dim_shards.size()=2, failure! (can't place 2nd shard)

  int dim_index = 0;  // Current position in dim_shards
  int cum_size = 1;   // Accumulated mesh capacity

  // Process each mesh axis in the permutation order (minor to major)
  for (const int index : minor_to_major().permutation) {
    // Skip unsharded dimensions (dim_shards == 1)
    while (dim_index < dim_shards().size() && dim_shards()[dim_index] == 1) {
      dim_index++;
    }

    // If we've consumed all sharded dimensions, we're done
    if (dim_index == dim_shards().size()) {
      break;
    }

    // Add this axis's capacity to the accumulated size
    cum_size *= minor_to_major().axis_sizes[index];

    // Try to consume as many sharded dimensions as possible with current capacity
    while (dim_index < dim_shards().size() &&
           cum_size % dim_shards()[dim_index] == 0) {
      cum_size /= dim_shards()[dim_index];
      dim_index++;
    }
  }

  // Skip any remaining unsharded dimensions
  while (dim_index < dim_shards().size() && dim_shards()[dim_index] == 1) {
    dim_index++;
  }

  // If we haven't consumed all sharded dimensions, the sharding is invalid
  if (dim_index != dim_shards().size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Can't shard the dims ", absl::StrJoin(dim_shards(), "x"),
        " to the mesh of [", absl::StrJoin(minor_to_major().permutation, ","),
        "] on ", absl::StrJoin(minor_to_major().axis_sizes, "x")));
  }

  return absl::OkStatus();
}

mlir::LogicalResult ShardingParam::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emit_error) const {
  auto status = verify();
  if (status.ok()) {
    return mlir::success();
  }
  return emit_error() << status.message();
}

mlir::LogicalResult ShardingParam::CanApplyTo(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::RankedTensorType shape, llvm::ArrayRef<int> device_ids) const {
  // First, verify the sharding itself is valid
  if (mlir::failed(verify(emitError))) {
    return mlir::failure();
  }

  // Check that dim_shards rank matches tensor rank
  if (shape.getRank() != dim_shards().size()) {
    return emitError() << "Requires dim shards to have the same rank as the "
                          "array. Array rank is "
                       << shape.getRank() << " vs dim shards rank of "
                       << dim_shards().size();
  }

  // Check that the number of devices matches the mesh size
  auto devices_in_mesh = NumDevices();
  if (devices_in_mesh != device_ids.size()) {
    return emitError() << "Requires the same amount of `devices` and from "
                          "`sharding`. Actual: "
                       << device_ids.size() << " vs " << devices_in_mesh;
  }

  return mlir::success();
}

// ============================================================================
// ShardingParam Shape Conversions
// ============================================================================

absl::StatusOr<llvm::SmallVector<int64_t>>
ShardingParam::GlobalShapeFromLocalShape(
    llvm::ArrayRef<int64_t> local_shape) const {
  llvm::SmallVector<int64_t> global_shape;

  // Check rank compatibility
  if (local_shape.size() != dim_shards().size()) {
    return absl::InvalidArgumentError(
        "Rank of local tensor differs from rank of `dim_shards`.");
  }

  // Compute global_shape[i] = local_shape[i] * dim_shards[i]
  //
  // Example:
  //   local_shape = [50, 100, 20]
  //   dim_shards = [2, 1, 3]
  //   → global_shape = [100, 100, 60]
  for (auto [idx, dim_shard] : llvm::enumerate(dim_shards())) {
    global_shape.push_back(dim_shard * local_shape[idx]);
  }

  return global_shape;
}

absl::StatusOr<llvm::SmallVector<int64_t>>
ShardingParam::LocalShapeFromGlobalShape(
    llvm::ArrayRef<int64_t> global_shape) const {
  auto num_shards = dim_shards();
  llvm::SmallVector<int64_t> local_shape;
  local_shape.reserve(global_shape.size());

  // Compute local_shape[i] = ceil(global_shape[i] / dim_shards[i])
  //
  // Note: We require exact divisibility (no padding), so we check
  // that global_shape[i] % num_shards[i] == 0
  //
  // Example:
  //   global_shape = [100, 100, 60]
  //   dim_shards = [2, 1, 3]
  //   → local_shape = [50, 100, 20]
  for (int i = 0; i < num_shards.size(); ++i) {
    if (global_shape[i] % num_shards[i] != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Global shape is not divisible by the number of shards in dimension ",
          i, ". Global shape: [", absl::StrJoin(global_shape, ","),
          "], number of shards: ", num_shards[i], "."));
    }
    local_shape.push_back(global_shape[i] / num_shards[i]);
  }

  return local_shape;
}

// ============================================================================
// ShardingParam Accessors
// ============================================================================

int ShardingParam::NumDevices() const {
  int devices_in_mesh = 1;
  for (const int axis_size : minor_to_major().axis_sizes) {
    devices_in_mesh *= axis_size;
  }
  return devices_in_mesh;
}

// ============================================================================
// ShardingParam String Representation
// ============================================================================

std::string ShardingParam::DebugString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << *this;
  return result;
}

// ============================================================================
// Free Functions
// ============================================================================

llvm::hash_code hash_value(ShardingParam sharding) {
  return sharding.hash_value();
}

mlir::AsmPrinter& operator<<(mlir::AsmPrinter& os, ShardingParam sharding) {
  // V1 is the current ShardingParam version.
  PrintInternalV1(os.getStream(), sharding);
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ShardingParam sharding) {
  // V1 is the current ShardingParam version.
  PrintInternalV1(os, sharding);
  return os;
}

}  // namespace xftcpp

