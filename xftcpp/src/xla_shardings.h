/* XLA Sharding Wrappers for xftcpp
 * Adapted from XLA's IFRT XLA Sharding
 * 
 * This file provides sharding implementations that wrap XLA's native
 * HloSharding representation. HloSharding is XLA's main sharding format
 * used throughout the compiler and runtime.
 * 
 * Key concept: HloSharding describes how to partition tensors across devices
 * in XLA computations. This wrapper makes it compatible with xftcpp's sharding
 * interface while preserving the underlying XLA sharding information.
 */

#ifndef XFTCPP_XLA_SHARDING_H_
#define XFTCPP_XLA_SHARDING_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"

#include "xla/hlo/ir/hlo_sharding.h"

#include "xftcpp/src/device_list.h"
#include "xftcpp/src/sharding.h"
#include "xftcpp/src/shape.h"
#include "xftcpp/src/memory.h"

namespace xftcpp {

// Forward declarations
class IndexDomain;

// ============================================================================
// HloSharding - XLA HloSharding Wrapper
// ============================================================================

// Wrapper around XLA's HloSharding for use with xftcpp.
//
// HloSharding is XLA's native sharding representation used throughout the
// XLA compiler stack. It encodes:
// - How tensors are partitioned across devices (tiled, replicated, etc.)
// - Device mesh topology and assignment
// - Subgroup replication patterns
//
// This wrapper class:
// 1. Holds an xla::HloSharding instance
// 2. Implements the xftcpp Sharding interface
// 3. Provides lazy validation (checked when used, not at construction)
// 4. Caches expensive hash computation for performance
//
// Design rationale for lazy validation:
// Creating HloSharding is often on the hot path when:
// - Receiving shardings from user code
// - Passing shardings to lower-level runtimes
// We validate when the sharding is actually used (e.g., in Disassemble())
// rather than at construction time to optimize the common path.
//
// Example usage:
//   xla::HloSharding xla_sharding = xla::HloSharding::Tile(...);
//   auto sharding = HloSharding::Create(devices, memory_kind, xla_sharding);
//   // Later, when actually partitioning data:
//   auto shards = sharding->Disassemble(shape);  // Validates here
class HloSharding final : public Sharding {
 public:
  // Creates an HloSharding wrapper.
  //
  // Parameters:
  //   devices: Devices that participate in this sharding
  //   memory_kind: Memory type to use for all shards
  //   xla_hlo_sharding: The XLA HloSharding specification
  //
  // Note: This bypasses upfront consistency checks between the HloSharding
  // and the device list to optimize the common path. Validation happens
  // when the sharding is actually used (e.g., in Disassemble()).
  //
  // REQUIRES: devices is not null and not empty
  static std::unique_ptr<HloSharding> Create(
      DeviceListRef devices,
      MemoryKind memory_kind,
      xla::HloSharding xla_hlo_sharding);

  ~HloSharding() override = default;

  // ============================================================================
  // XLA-Specific Methods
  // ============================================================================

  // Returns the wrapped XLA HloSharding.
  //
  // This allows code that needs to interface directly with XLA to access
  // the native HloSharding representation. Use this when:
  // - Passing sharding info to XLA compiler
  // - Converting between IFRT and XLA representations
  // - Debugging sharding configurations
  const xla::HloSharding& xla_hlo_sharding() const { 
    return xla_hlo_sharding_; 
  }

  // ============================================================================
  // Sharding Interface Implementation
  // ============================================================================

  // Get the shape of a single shard for the given array shape.
  //
  // For tiled shardings, computes how the shape is divided.
  // For replicated shardings, returns the full shape.
  //
  // Example: For a [8, 16] array tiled on 2 devices in dimension 0:
  //   GetShardShape([8, 16]) -> [4, 16]
  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  // Check if this sharding uses the same logical partitioning as another.
  //
  // Two HloShardings have the same partitioning if their underlying
  // xla::HloSharding objects are equal. This compares the partitioning
  // scheme but NOT device assignment or memory kind.
  //
  // Use case: Checking if two arrays are partitioned the same way
  // (useful for determining if resharding is needed)
  bool HasSamePartitioning(const Sharding& other) const override;

  // Create a new sharding with different devices/memory but same partitioning.
  //
  // This is used to move data between devices or memory types while
  // maintaining the same sharding pattern.
  //
  // Parameters:
  //   devices: New device list (must match current size if provided)
  //   memory_kind: New memory type (if provided)
  //
  // Returns: New HloSharding with updated assignment but same partitioning
  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const override;

  // Break a full array shape into per-device (shape, sharding) pairs.
  //
  // This is where validation happens - we check that the HloSharding is
  // consistent with the provided shape and device list.
  //
  // Returns: vector of (shard_shape, shard_sharding) pairs, one per device
  // Each pair describes:
  // - shard_shape: The shape of data on this device
  // - shard_sharding: A SingleDeviceSharding for this device
  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  // Disassemble variant for dynamic shapes (shapes with runtime-variable dims)
  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  // Map each shard to its region (IndexDomain) in the full array.
  //
  // IndexDomain describes which slice of the original array each shard contains.
  // This is computed from the HloSharding's tile assignment.
  //
  // Example: [8, 16] array tiled on 2 devices in dim 0 ->
  //   [IndexDomain(origin=[0,0], shape=[4,16]),
  //    IndexDomain(origin=[4,0], shape=[4,16])]
  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  // Human-readable string representation for debugging.
  //
  // Includes the XLA HloSharding string representation, devices, and memory kind.
  std::string DebugString() const override;

 private:
  // Private constructor - use Create() factory method instead.
  //
  // This enforces the creation pattern and ensures proper initialization.
  HloSharding(DeviceListRef devices, 
              MemoryKind memory_kind,
              xla::HloSharding xla_hlo_sharding);

  // Hash computation for absl hash containers.
  //
  // Uses a cached atomic hash for performance - HloSharding hashing can be
  // expensive for complex tiled shardings with large device meshes.
  //
  // The hash is computed once on first access and cached in hash_.
  // Multiple threads may compute and write the same value (benign race).
  void Hash(absl::HashState state) const override;

  // ============================================================================
  // Member Variables
  // ============================================================================

  // The wrapped XLA HloSharding specification.
  // This is the source of truth for the partitioning scheme.
  xla::HloSharding xla_hlo_sharding_;

  // Cached hash value for performance.
  // 
  // Why caching? HloSharding hashing can be expensive because it may involve:
  // - Hashing large tile assignment arrays (thousands of devices)
  // - Complex metadata structures
  // - Recursion for tuple shardings
  //
  // Caching strategy:
  // - kUnsetHash (0) indicates the hash hasn't been computed yet
  // - Once computed, the hash is stored atomically
  // - Multiple threads may race to compute, but they'll compute the same value
  // - This is a benign data race - worst case we compute twice
  //
  // Thread safety: std::atomic ensures memory ordering but doesn't prevent
  // redundant computation. This is intentional for performance.
  static constexpr uint64_t kUnsetHash = 0;
  mutable std::atomic<uint64_t> hash_ = kUnsetHash;
};

// ============================================================================
// Testing Utilities
// ============================================================================

// Test-only function: Compute IndexDomains using XLA HloSharding APIs directly.
//
// This is a reference implementation that uses xla::HloSharding's internal
// methods for computing index domains. It's useful for:
// - Testing the fast path against a known-correct slow path
// - Validating optimizations
// - Debugging complex sharding configurations
//
// Why "slow path"? This uses XLA's full machinery which may:
// - Perform additional validation
// - Use less optimized code paths
// - Allocate more temporary data structures
//
// Production code should use HloSharding::IndexDomains() instead.
//
// Parameters:
//   sharding: The HloSharding to compute domains for
//   shape: The full array shape
//   single_device_shard_semantics: Whether to include non-addressable devices
//
// Returns: Vector of IndexDomain, one per device
std::vector<IndexDomain> TEST_HloShardingIndexDomainsSlowPath(
    const HloSharding& sharding, 
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics);

}  // namespace xftcpp

#endif  // XFTCPP_XLA_SHARDING_H_

