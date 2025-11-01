/* Sharding Parameters for Distributed Arrays
 * Adapted from XLA's IFRT ShardingParam
 *
 * Defines how a multi-dimensional array/tensor is partitioned and distributed
 * across a mesh of devices (GPUs, TPUs, etc.).
 *
 * Key Concepts:
 * - dim_shards: How many slices to make along each tensor dimension
 * - minor_to_major: How to map those slices to physical devices
 * - device mesh: A multi-dimensional grid of devices (e.g., 2x3 = 6 devices)
 *
 * Example Use Cases:
 * - Data parallelism: Replicate model across devices, shard data
 * - Model parallelism: Shard model weights across devices
 * - Hybrid: Both data and model parallelism combined
 */

#ifndef XFTCPP_SHARDING_PARAMS_H_
#define XFTCPP_SHARDING_PARAMS_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

namespace xftcpp {

// ============================================================================
// ShardingParam - Defines how to distribute an array across devices
// ============================================================================

// Represents the sharding (partitioning) of an array across a device mesh.
//
// A ShardingParam tells you:
// 1. How to slice a tensor into pieces (dim_shards)
// 2. How to arrange devices in memory layout order (minor_to_major)
// 3. Which device gets which piece
//
// === THE THREE COMPONENTS ===
//
// **dim_shards** (tensor slicing):
//   - One value per tensor dimension
//   - Each value = number of slices for that dimension
//   - Example: [2, 1, 3] for a 3D tensor means:
//     * Dimension 0: Cut into 2 pieces
//     * Dimension 1: Don't cut (1 piece = replicated)
//     * Dimension 2: Cut into 3 pieces
//     * Total: 2 × 1 × 3 = 6 slices
//
// **permutation** (device layout order):
//   - Defines traversal order of the device mesh
//   - Values are mesh axis indices (0, 1, 2, ...)
//   - Order: minor (fastest-changing) to major (slowest-changing)
//   - Like array memory layout: [1,0] means axis-1 changes faster than axis-0
//
// **axis_sizes** (device mesh shape):
//   - One value per mesh axis
//   - Each value = number of devices along that axis
//   - Example: [3, 2] means a 3×2 grid = 6 devices
//   - Total devices = product of all axis sizes
//
// === DETAILED EXAMPLES ===
//
// **Example 1: Simple 2D sharding**
//   Sharding: dim_shards=[2,1,3], permutation=[1,0], axis_sizes=[3,2]
//   
//   Meaning:
//   - Tensor is 3D, cut into 2×1×3 = 6 slices
//   - Device mesh is 3×2 = 6 devices
//   - Permutation [1,0] means: axis-1 changes faster (minor-to-major order)
//   
//   Device layout (mesh axis-0 = rows, axis-1 = columns):
//     Axis-1 →
//   A [0  1]
//   x [2  3]
//   i [4  5]
//   s
//   0
//   ↓
//   
//   With permutation [1,0], we traverse columns first, then rows:
//     Device order: 0, 1, 2, 3, 4, 5
//   
//   Slice assignment (dim_shards=[2,1,3]):
//     - 2 slices in dim-0
//     - 3 slices in dim-2
//     - Linearized slice indices: 0→(0,0), 1→(0,1), 2→(0,2), 3→(1,0), 4→(1,1), 5→(1,2)
//       where (i,j) means i-th slice of dim-0, j-th slice of dim-2
//   
//   Final mapping:
//     Device 0 gets slice (0,0): dim-0[0], dim-2[0]
//     Device 1 gets slice (0,1): dim-0[0], dim-2[1]
//     Device 2 gets slice (0,2): dim-0[0], dim-2[2]
//     Device 3 gets slice (1,0): dim-0[1], dim-2[0]
//     Device 4 gets slice (1,1): dim-0[1], dim-2[1]
//     Device 5 gets slice (1,2): dim-0[1], dim-2[2]
//
// **Example 2: Replication**
//   Sharding: dim_shards=[2,1], permutation=[0,1], axis_sizes=[2,3]
//   
//   Meaning:
//   - Tensor is 2D, cut into 2×1 = 2 slices (only dim-0 is sharded)
//   - Device mesh is 2×3 = 6 devices
//   - Permutation [0,1] means: axis-0 changes faster (rows iterate first)
//   
//   Since we have 2 slices but 6 devices, replication occurs:
//     - Slice 0 replicated on devices: 0, 1, 2 (first row of mesh)
//     - Slice 1 replicated on devices: 3, 4, 5 (second row of mesh)
//   
//   This is data parallelism: Each slice is replicated across multiple devices.
//
// **Example 3: Different traversal order**
//   Sharding: dim_shards=[4], permutation=[1,0], axis_sizes=[2,2]
//   
//   Meaning:
//   - Tensor is 1D, cut into 4 slices
//   - Device mesh is 2×2 = 4 devices
//   - Permutation [1,0] changes traversal order
//   
//   Device layout:
//     [0  1]
//     [2  3]
//   
//   With permutation [1,0] (axis-1 is minor, changes faster):
//     Linearization: 0, 2, 1, 3
//     (traverse down columns first, then across rows)
//   
//   Slice assignment:
//     Device 0 gets slice 0
//     Device 2 gets slice 1
//     Device 1 gets slice 2
//     Device 3 gets slice 3
//
// === INVALID EXAMPLES (What NOT to do) ===
//
// **Invalid 1: Mismatched dimensions**
//   dim_shards=[1,1], permutation=[0,1], axis_sizes=[2]
//   ❌ permutation has 2 elements, but axis_sizes has 1
//   ✅ Both must have the same length (number of mesh axes)
//
// **Invalid 2: Not enough devices**
//   dim_shards=[2,2], permutation=[0], axis_sizes=[2]
//   ❌ 2×2=4 slices, but only 2 devices
//   ✅ Product of dim_shards must divide evenly into product of axis_sizes
//
// **Invalid 3: Uneven distribution**
//   dim_shards=[1,2], permutation=[0,1], axis_sizes=[3,2]
//   ❌ 2 slices in dim-1, but axis-0 has 3 devices (2 doesn't divide 3)
//   ✅ Each sharded dimension must align with device mesh axes
//
// === MATHEMATICAL FORMULATION ===
//
// Given:
//   - T: tensor with shape (d₀, d₁, ..., dₙ₋₁)
//   - S: dim_shards = [s₀, s₁, ..., sₙ₋₁]
//   - M: device mesh with axis sizes [a₀, a₁, ..., aₘ₋₁]
//   - P: permutation = [p₀, p₁, ..., pₘ₋₁]
//
// Requirements:
//   1. ∏sᵢ divides ∏aⱼ (total slices divides total devices)
//   2. Each sᵢ > 1 must align with corresponding mesh axes
//   3. P is a valid permutation of {0, 1, ..., m-1}
//
// Result:
//   - Each device gets ∏aⱼ / ∏sᵢ replicas (for replication)
//   - Each slice has shape (⌈d₀/s₀⌉, ⌈d₁/s₁⌉, ..., ⌈dₙ₋₁/sₙ₋₁⌉)
//
// === FURTHER READING ===
// For conversions to/from other sharding formats (XLA HloSharding, TensorFlow
// sharding, etc.), see the support directory in the original XLA codebase.
class ShardingParam {
 public:
  // ==========================================================================
  // MinorToMajor - Device mesh layout and traversal order
  // ==========================================================================
  
  // Represents how to traverse a multi-dimensional device mesh.
  //
  // The combination of `permutation` and `axis_sizes` defines:
  // 1. The shape of the device mesh (axis_sizes)
  // 2. The order to iterate through devices (permutation)
  //
  // Sizes of `permutation` and `axis_sizes` must be equal.
  //
  // **Why "Minor to Major"?**
  // This terminology comes from memory layout:
  // - Minor dimension = changes fastest (like columns in row-major order)
  // - Major dimension = changes slowest (like rows in row-major order)
  //
  // **Permutation Semantics:**
  // - permutation[0] = minor axis (changes fastest)
  // - permutation[n-1] = major axis (changes slowest)
  // - Values in permutation are indices into axis_sizes
  //
  // **Example 1: Row-major traversal**
  //   axis_sizes = [2, 3]  (2 rows, 3 columns)
  //   permutation = [1, 0]  (axis-1 changes faster than axis-0)
  //   
  //   Device layout:
  //     [0  1  2]
  //     [3  4  5]
  //   
  //   Traversal: 0, 1, 2, 3, 4, 5 (across columns first, then down rows)
  //
  // **Example 2: Column-major traversal**
  //   axis_sizes = [2, 3]  (2 rows, 3 columns)
  //   permutation = [0, 1]  (axis-0 changes faster than axis-1)
  //   
  //   Device layout:
  //     [0  2  4]
  //     [1  3  5]
  //   
  //   Traversal: 0, 1, 2, 3, 4, 5 (down rows first, then across columns)
  //
  // **Example 3: 3D mesh**
  //   axis_sizes = [2, 2, 2]  (2x2x2 cube)
  //   permutation = [2, 1, 0]  (axis-2 fastest, axis-0 slowest)
  //   
  //   This defines a specific traversal pattern through the 3D cube.
  struct MinorToMajor {
    // A permutation of the range [0, 1, ..., n-1] where n = axis_sizes.size()
    //
    // Must contain each value in [0, n-1] exactly once.
    // Defines the order to traverse mesh axes, from minor (fastest) to major (slowest).
    //
    // Example: [1, 0] means axis-1 is minor, axis-0 is major
    llvm::SmallVector<int, 4> permutation;

    // The size of each mesh axis, before applying the permutation.
    //
    // axis_sizes[i] = number of devices along mesh axis i
    // Product of all sizes = total number of devices
    //
    // Example: [3, 2] means 3 devices along axis-0, 2 along axis-1 (6 total)
    llvm::SmallVector<int, 4> axis_sizes;

    // Validates the structure of this MinorToMajor.
    //
    // Checks:
    // 1. permutation.size() == axis_sizes.size()
    // 2. permutation is a valid permutation of [0, 1, ..., n-1]
    // 3. All axis_sizes are positive
    //
    // Returns:
    //   - absl::OkStatus() if valid
    //   - Error status describing what's wrong if invalid
    absl::Status verify() const;

    // MLIR-style verification with diagnostics.
    // Same checks as verify(), but emits detailed error messages.
    //
    // Parameters:
    //   emit_error: Function to create diagnostic error messages
    //
    // Returns:
    //   - mlir::success() if valid
    //   - mlir::failure() if invalid (with diagnostics emitted)
    mlir::LogicalResult verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emit_error) const;

    // Equality comparison
    bool operator==(const MinorToMajor& other) const {
      return permutation == other.permutation && axis_sizes == other.axis_sizes;
    }

    bool operator!=(const MinorToMajor& other) const {
      return !(*this == other);
    }

    // Produces a flat list of device IDs according to the permutation.
    //
    // This linearizes the multi-dimensional device mesh into a 1D array,
    // traversing in the order specified by the permutation.
    //
    // Example:
    //   axis_sizes = [2, 3]
    //   permutation = [1, 0]  (axis-1 changes faster)
    //   
    //   Conceptual 2D layout:
    //     [0  1  2]
    //     [3  4  5]
    //   
    //   Output: [0, 1, 2, 3, 4, 5]
    //
    // Parameters:
    //   out_devices: Output vector to populate (will be cleared and resized)
    //
    // Postconditions:
    //   out_devices.size() == product of all axis_sizes
    void ToDeviceList(llvm::SmallVectorImpl<int>& out_devices) const;
  };

  // ==========================================================================
  // Constructors
  // ==========================================================================

  // Constructs a ShardingParam from dimension shards and mesh layout.
  //
  // Parameters:
  //   dim_shards: How many slices for each tensor dimension
  //   minor_to_major: Device mesh shape and traversal order
  //
  // Note: This does NOT validate the parameters. Call verify() after construction
  // to ensure the sharding is valid.
  //
  // Example:
  //   ShardingParam sharding(
  //       {2, 1, 3},  // Shard dim-0 into 2, dim-2 into 3
  //       {{1, 0}, {3, 2}}  // 3×2 device mesh, row-major order
  //   );
  //   if (auto status = sharding.verify(); !status.ok()) {
  //     // Handle error
  //   }
  ShardingParam(std::vector<int64_t> dim_shards, MinorToMajor minor_to_major)
      : dim_shards_(std::move(dim_shards)),
        minor_to_major_(std::move(minor_to_major)) {}

  // Default copy/move semantics
  ShardingParam(const ShardingParam&) = default;
  ShardingParam(ShardingParam&&) = default;
  ShardingParam& operator=(const ShardingParam&) = default;
  ShardingParam& operator=(ShardingParam&&) = default;

  // ==========================================================================
  // Parsing and Printing (MLIR Integration)
  // ==========================================================================

  // Parses a ShardingParam from MLIR assembly format.
  //
  // Assembly format: $dim_shards to $permutation on $axis_sizes
  //
  // Example: "2x1x3 to [1,0] on 3x2"
  //
  // Parameters:
  //   ods_parser: MLIR assembly parser
  //
  // Returns:
  //   Parsed ShardingParam on success, failure otherwise
  static mlir::FailureOr<ShardingParam> Parse(mlir::AsmParser& ods_parser);

  // Parses V1 of ShardingParam.
  // This method is meant to be used in versioned MLIR dialects.
  //
  // Parameters:
  //   ods_parser: MLIR assembly parser
  //
  // Returns:
  //   Parsed ShardingParam on success, failure otherwise
  static mlir::FailureOr<ShardingParam> ParseV1(mlir::AsmParser& ods_parser);

  // Prints V1 of ShardingParam.
  // This method is meant to be used in versioned MLIR dialects.
  //
  // Parameters:
  //   ods_printer: MLIR assembly printer
  //   sharding: The ShardingParam to print
  static void PrintV1(mlir::AsmPrinter& ods_printer,
                      const ShardingParam& sharding);

  // ==========================================================================
  // Validation
  // ==========================================================================

  // Validates the internal consistency of this ShardingParam.
  //
  // Checks:
  // 1. minor_to_major is valid (via MinorToMajor::verify())
  // 2. All dim_shards are positive
  // 3. Product of dim_shards divides evenly into product of axis_sizes
  // 4. Each sharded dimension aligns properly with the device mesh
  //
  // This does NOT check compatibility with a specific tensor shape.
  // For that, use CanApplyTo().
  //
  // Returns:
  //   - absl::OkStatus() if valid
  //   - Error status describing what's wrong if invalid
  //
  // Example:
  //   ShardingParam sharding(...);
  //   if (auto status = sharding.verify(); !status.ok()) {
  //     LOG(ERROR) << "Invalid sharding: " << status;
  //   }
  absl::Status verify() const;

  // MLIR-style verification with diagnostics.
  // Same checks as verify(), but emits detailed error messages.
  //
  // Parameters:
  //   emit_error: Function to create diagnostic error messages
  //
  // Returns:
  //   - mlir::success() if valid
  //   - mlir::failure() if invalid (with diagnostics emitted)
  mlir::LogicalResult verify(
      llvm::function_ref<mlir::InFlightDiagnostic()> emit_error) const;

  // Validates if this sharding can be applied to a specific tensor.
  //
  // Checks all requirements from verify(), plus:
  // 1. dim_shards.size() matches tensor rank
  // 2. Each tensor dimension is evenly divisible by its corresponding shard count
  // 3. The number of device_ids matches NumDevices()
  //
  // Parameters:
  //   emitError: Function to create diagnostic error messages
  //   shape: MLIR tensor type with shape information
  //   device_ids: IDs of physical devices to use
  //
  // Returns:
  //   - mlir::success() if compatible
  //   - mlir::failure() if incompatible (with diagnostics emitted)
  //
  // Example:
  //   ShardingParam sharding({2, 3}, ...);
  //   RankedTensorType tensor_type = ...;
  //   ArrayRef<int> devices = {0, 1, 2, 3, 4, 5};
  //   
  //   // This would fail if dimensions don't align
  //   auto result = sharding.CanApplyTo(emitError, tensor_type, devices);
  mlir::LogicalResult CanApplyTo(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
      mlir::RankedTensorType shape,
      llvm::ArrayRef<int> device_ids) const;

  // ==========================================================================
  // Shape Conversions
  // ==========================================================================

  // Computes the global (full) tensor shape from a local (per-device) shape.
  //
  // This is the inverse of LocalShapeFromGlobalShape().
  //
  // Formula:
  //   global_shape[i] = local_shape[i] * dim_shards[i]
  //
  // Example:
  //   dim_shards = [2, 1, 3]
  //   local_shape = [50, 100, 20]
  //   → global_shape = [100, 100, 60]
  //
  // Use case: Given a shard on one device, compute the full tensor size.
  //
  // Parameters:
  //   local_shape: Shape of data on a single device
  //
  // Returns:
  //   Global shape if successful, error if dimensions mismatch
  absl::StatusOr<llvm::SmallVector<int64_t>> GlobalShapeFromLocalShape(
      llvm::ArrayRef<int64_t> local_shape) const;

  // Computes the local (per-device) shape from a global (full) tensor shape.
  //
  // This is the inverse of GlobalShapeFromLocalShape().
  //
  // Formula:
  //   local_shape[i] = ceil(global_shape[i] / dim_shards[i])
  //
  // Example:
  //   dim_shards = [2, 1, 3]
  //   global_shape = [100, 100, 61]
  //   → local_shape = [50, 100, 21]  (note: 61/3 rounds up to 21)
  //
  // Use case: Given a full tensor, compute the size of each shard.
  //
  // Note: Uses ceiling division to handle shapes that don't divide evenly.
  // The last shard along each dimension may be padded.
  //
  // Parameters:
  //   global_shape: Full tensor dimensions
  //
  // Returns:
  //   Local shape if successful, error if dimensions mismatch
  absl::StatusOr<llvm::SmallVector<int64_t>> LocalShapeFromGlobalShape(
      llvm::ArrayRef<int64_t> global_shape) const;

  // ==========================================================================
  // Accessors
  // ==========================================================================

  // Returns the number of devices the array is sharded over.
  //
  // This equals the product of all axis sizes in the device mesh.
  // It may be larger than the number of shards if replication is used.
  //
  // Example 1 (no replication):
  //   dim_shards = [2, 3], axis_sizes = [2, 3]
  //   → 6 shards distributed to 6 devices
  //
  // Example 2 (with replication):
  //   dim_shards = [2], axis_sizes = [2, 3]
  //   → 2 shards, each replicated on 3 devices = 6 total devices
  int NumDevices() const;

  // Returns the dimension sharding specification.
  // One value per tensor dimension indicating the number of slices.
  llvm::ArrayRef<int64_t> dim_shards() const { return dim_shards_; }

  // Returns the device mesh layout and traversal order.
  const MinorToMajor& minor_to_major() const { return minor_to_major_; }

  // ==========================================================================
  // Comparison and Hashing
  // ==========================================================================

  bool operator==(const ShardingParam& other) const {
    return dim_shards_ == other.dim_shards_ &&
           minor_to_major_ == other.minor_to_major_;
  }

  bool operator!=(const ShardingParam& other) const {
    return !(*this == other);
  }

  // LLVM-style hashing support.
  //
  // Computes a hash code for this ShardingParam, allowing it to be used
  // in LLVM hash containers like DenseMap and DenseSet.
  //
  // Returns:
  //   Hash code combining dim_shards and minor_to_major
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(dim_shards(),
                              llvm::ArrayRef<int>(minor_to_major_.permutation),
                              llvm::ArrayRef<int>(minor_to_major_.axis_sizes));
  }

  // Support for absl hash containers (flat_hash_map, flat_hash_set, etc.)
  //
  // Allows using ShardingParam as a key:
  //   absl::flat_hash_map<ShardingParam, Value> sharding_map;
  template <typename H>
  friend H AbslHashValue(H h, const ShardingParam& value) {
    h = H::combine(std::move(h), value.dim_shards_);
    h = H::combine_contiguous(std::move(h),
                              value.minor_to_major_.permutation.data(),
                              value.minor_to_major_.permutation.size());
    return H::combine_contiguous(std::move(h),
                                 value.minor_to_major_.axis_sizes.data(),
                                 value.minor_to_major_.axis_sizes.size());
  }

  // Support for absl::StrCat and absl::StrFormat
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ShardingParam& sharding) {
    sink.Append(sharding.DebugString());
  }

  // ==========================================================================
  // String Representation
  // ==========================================================================

  // Returns a human-readable debug string.
  //
  // Format: "dim_shards=[s₀,s₁,...] permutation=[p₀,p₁,...] axis_sizes=[a₀,a₁,...]"
  //
  // Example output:
  //   "dim_shards=[2,1,3] permutation=[1,0] axis_sizes=[3,2]"
  //
  // This is useful for logging, debugging, and error messages.
  std::string DebugString() const;

 private:
  // How many slices to make along each tensor dimension.
  // dim_shards_[i] = number of slices for dimension i
  // A value of 1 means no sharding (full replication) on that dimension
  std::vector<int64_t> dim_shards_;

  // Device mesh layout and traversal order
  MinorToMajor minor_to_major_;
};

// ============================================================================
// Free Functions for Hashing
// ============================================================================

// LLVM-style hash_value free function.
// Allows ShardingParam to be used in LLVM hash containers.
//
// Example:
//   llvm::DenseMap<ShardingParam, Value> sharding_map;
inline llvm::hash_code hash_value(ShardingParam sharding) {
  return sharding.hash_value();
}

// ============================================================================
// Stream Output Operators
// ============================================================================

// Allows printing ShardingParam to MLIR assembly printers.
//
// Example (in MLIR printing code):
//   ods_printer << sharding;
mlir::AsmPrinter& operator<<(mlir::AsmPrinter& os, ShardingParam sharding);

// Allows printing ShardingParam to LLVM raw output streams.
//
// Example:
//   llvm::outs() << sharding << "\n";
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ShardingParam sharding);

// Allows printing ShardingParam to standard C++ output streams.
//
// Example:
//   std::cout << sharding << std::endl;
inline std::ostream& operator<<(std::ostream& os,
                                 const ShardingParam& sharding) {
  return os << sharding.DebugString();
}

// Allows printing MinorToMajor to standard C++ output streams.
inline std::ostream& operator<<(std::ostream& os,
                                 const ShardingParam::MinorToMajor& mtm) {
  return os << mtm.DebugString();
}

}  // namespace xftcpp

#endif  // XFTCPP_SHARDING_PARAMS_H_

