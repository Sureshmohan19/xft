/* Concrete Sharding implementation for xftcpp
 * Adapted from XLA's IFRT Sharding
 * 
 * Sharding describes how array data is distributed across devices.
 * Think of it like slicing a pizza - sharding tells you which device gets which piece.
 */

#ifndef XFTCPP_SHARDING_H_
#define XFTCPP_SHARDING_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"

#include "xftcpp/src/device.h"
#include "xftcpp/src/device_list.h"
#include "xftcpp/src/memory.h"
#include "xftcpp/src/shape.h"

namespace xftcpp {

// Forward declarations
class Client;
class IndexDomain;
class Sharding;
class ShardingParam;

// ============================================================================
// Type Aliases
// ============================================================================

// Shared pointer to const Sharding - used throughout the codebase
// Using const because shardings are immutable once created
using ShardingRef = std::shared_ptr<const Sharding>;

// ============================================================================
// Enums
// ============================================================================

// Controls whether to include non-addressable devices in operations.
// 
// In multi-process/multi-host settings, some devices may be visible but not
// addressable by the current process. This enum controls whether to include them.
//
// Example: In a 2-host setup with 4 GPUs each:
// - Host 0 can address GPUs 0-3
// - Host 1 can address GPUs 4-7
// - But all hosts can SEE all 8 GPUs
enum class SingleDeviceShardSemantics : int {
  // Only process addressable devices (default, most common)
  // Disassemble() returns shards only for devices this process can use
  kAddressableShards = 0,

  // Process ALL devices (including non-addressable ones)
  // Disassemble() returns shards for all devices, even ones we can't touch
  // Useful for getting a global view of the sharding
  kAllShards,
};

// ============================================================================
// Base Sharding Class
// ============================================================================

// Abstract base class for all sharding types.
//
// A Sharding describes:
// 1. HOW data is partitioned (the logical partitioning scheme)
// 2. WHERE the pieces go (device assignment)
// 3. WHAT kind of memory to use (memory_kind)
//
// Key operations:
// - GetShardShape(): Given a full array shape, what's the shape of each shard?
// - Disassemble(): Break a logical array into per-device pieces
// - IndexDomains(): Map each shard to its slice of the original array
class Sharding {
 public:
  // Virtual destructor for proper cleanup of derived classes
  virtual ~Sharding() = default;

  // Not copyable or movable - shardings are meant to be shared via ShardingRef
  Sharding(const Sharding&) = delete;
  Sharding(Sharding&&) = delete;
  Sharding& operator=(const Sharding&) = delete;
  Sharding& operator=(Sharding&&) = delete;

  // ============================================================================
  // Basic Properties
  // ============================================================================

  // Get all devices involved in this sharding
  // Devices may appear multiple times (e.g., in replicated sharding)
  const DeviceListRef& devices() const { return devices_; }

  // Get the memory kind (e.g., HBM, host memory) for all shards
  MemoryKind memory_kind() const { return memory_kind_; }

  // Check if this is a fully replicated sharding
  // Fully replicated means: every device has a complete copy of the data
  // (shard_shape == full_shape for all devices)
  bool IsFullyReplicated() const { return is_fully_replicated_; }

  // ============================================================================
  // Comparison
  // ============================================================================

  // Two shardings are equal if they have the same partitioning scheme,
  // same devices, and same memory kind
  bool operator==(const Sharding& other) const;
  bool operator!=(const Sharding& other) const { return !(*this == other); }

  // ============================================================================
  // Core Virtual Methods (must be implemented by subclasses)
  // ============================================================================

  // Get the shape of a single shard for the given array shape.
  // Returns error if:
  // - The sharding doesn't have a uniform shard shape, OR
  // - The given shape is incompatible with this sharding
  //
  // Example: For a [8, 16] array sharded across 2 devices in dim 0:
  //   GetShardShape([8, 16]) -> [4, 16]
  virtual absl::StatusOr<Shape> GetShardShape(const Shape& shape) const = 0;

  // Check if two shardings use the same logical partitioning scheme.
  // This compares HOW data is split, not WHERE it goes.
  //
  // Example: Two ConcreteEvenShardings with the same shard shape but
  // different devices have the same partitioning.
  virtual bool HasSamePartitioning(const Sharding& other) const = 0;

  // Create a new sharding with the same partitioning but different devices/memory.
  // The number of devices must match (if provided).
  //
  // Use case: Moving data to different devices while keeping the same distribution.
  virtual absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const = 0;

  // Break a full array shape into per-device (shape, sharding) pairs.
  // Each pair describes one device's shard of the data.
  //
  // Returns: vector of (shard_shape, shard_sharding) pairs
  // - shard_shape: the shape of data on this device
  // - shard_sharding: a SingleDeviceSharding for this device
  //
  // Example: [8, 16] array on 2 devices ->
  //   [([4, 16], SingleDeviceSharding(dev0)),
  //    ([4, 16], SingleDeviceSharding(dev1))]
  virtual absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
  Disassemble(const Shape& shape) const = 0;

  virtual absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
  Disassemble(const Shape& shape,
              SingleDeviceShardSemantics semantics) const = 0;

  // Disassemble variant for dynamic shapes (shapes with unknown dimensions)
  virtual absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
  Disassemble(const DynamicShape& dynamic_shape) const = 0;

  virtual absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
  Disassemble(const DynamicShape& dynamic_shape,
              SingleDeviceShardSemantics semantics) const = 0;

  // Map each shard to its region (IndexDomain) in the full array.
  // IndexDomain describes which slice of the original array this shard contains.
  //
  // Example: [8, 16] array sharded on 2 devices in dim 0 ->
  //   [IndexDomain(origin=[0,0], shape=[4,16]),
  //    IndexDomain(origin=[4,0], shape=[4,16])]
  //
  // Note: For replicated sharding, all domains are the same (entire array)
  virtual absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const = 0;

  virtual absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics semantics) const = 0;

  // ============================================================================
  // Hashing and String Conversion
  // ============================================================================

  // Support for absl hash containers (flat_hash_map, flat_hash_set)
  template <typename H>
  friend H AbslHashValue(H h, const Sharding& sharding) {
    sharding.Hash(absl::HashState::Create(&h));
    return std::move(h);
  }

  // Support for absl::StrCat, absl::StrFormat
  template <typename Sink>
  friend void AbslStringify(Sink& sink,
                            std::shared_ptr<const Sharding>& sharding) {
    if (sharding == nullptr) {
      sink.Append("<nullptr>");
    } else {
      sink.Append(sharding->DebugString());
    }
  }

  // String representation for debugging
  virtual std::string DebugString() const = 0;

 protected:
  // Constructor for base class - only callable by subclasses
  Sharding(DeviceListRef devices, MemoryKind memory_kind,
           bool is_fully_replicated);

  // Hash computation - must be implemented by subclasses
  virtual void Hash(absl::HashState state) const = 0;

  // Member variables - shared by all sharding types
  DeviceListRef devices_;              // Devices in this sharding
  MemoryKind memory_kind_;             // Memory type for all shards
  bool is_fully_replicated_;           // True if fully replicated
};

std::ostream& operator<<(std::ostream& os, const Sharding& sharding);

// ============================================================================
// SingleDeviceSharding - Simplest Case
// ============================================================================

// Sharding where all data lives on a single device.
// No splitting, no distribution - just one device has everything.
//
// This is the "I don't need distribution" case.
// Technically "fully replicated" with replication factor of 1.
//
// Example: A small array that fits on one GPU.
class SingleDeviceSharding final : public Sharding {
 public:
  // Factory method to create a single-device sharding
  // Parameters:
  //   device: The one device that will hold all the data
  //   memory_kind: Type of memory to use on that device
  static std::unique_ptr<SingleDeviceSharding> Create(
      Device* device, MemoryKind memory_kind);

  ~SingleDeviceSharding() override = default;

  // Sharding interface implementation

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape,
      SingleDeviceShardSemantics semantics) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape,
      SingleDeviceShardSemantics semantics) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics semantics) const override;

  std::string DebugString() const override;

 private:
  // Private constructor - use Create() instead
  explicit SingleDeviceSharding(DeviceListRef device_list,
                               MemoryKind memory_kind);

  void Hash(absl::HashState state) const override;
};

// ============================================================================
// OpaqueSharding - Unknown Distribution
// ============================================================================

// Sharding where we know WHICH devices are involved, but not HOW data is split.
// This is the "I don't know the partitioning scheme" case.
//
// Operations like Disassemble() will fail because we don't have enough info.
//
// Use case: Interfacing with external systems where sharding is opaque,
// or when you only care about device placement, not data layout.
class OpaqueSharding final : public Sharding {
 public:
  // Factory method to create an opaque sharding
  // Parameters:
  //   devices: The devices involved (but we don't know how data is split)
  //   memory_kind: Type of memory to use
  //
  // REQUIRES: !devices->empty()
  static std::unique_ptr<OpaqueSharding> Create(
      DeviceListRef devices, MemoryKind memory_kind);

  ~OpaqueSharding() override = default;

  // Sharding interface implementation
  // Note: Most operations that require knowing the partitioning scheme
  // (like Disassemble, GetShardShape) will return errors

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape,
      SingleDeviceShardSemantics semantics) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape,
      SingleDeviceShardSemantics semantics) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics semantics) const override;

  std::string DebugString() const override;

 private:
  // Private constructor - use Create() instead
  explicit OpaqueSharding(DeviceListRef devices, MemoryKind memory_kind);

  void Hash(absl::HashState state) const override;
};


// ============================================================================
// ConcreteSharding - Non-Uniform Shard Shapes
// ============================================================================

// Sharding where we explicitly know:
// 1. The full array shape
// 2. Each shard's shape (can be DIFFERENT per device)
// 3. Which devices get which shards
//
// This is the most flexible but heavyweight sharding type.
// Use ConcreteEvenSharding if all shards are the same size.
//
// Supports both static and dynamic shapes via std::variant.
//
// Example: [9, 16] array on 2 devices, split unevenly in dim 0
//   -> shape=[9,16], shard_shapes=[[5,16], [4,16]]
class ConcreteSharding final : public Sharding {
 public:
  // Factory method for static shapes
  // Parameters:
  //   devices: Devices to distribute shards across
  //   memory_kind: Memory type to use
  //   shape: Full logical array shape
  //   shard_shapes: Vector of shapes, one per addressable device
  //   index_domains: Optional per-shard index domains (slices of original array)
  //
  // REQUIRES: devices->AddressableDeviceList()->size() == shard_shapes.size()
  // REQUIRES: !devices->empty()
  static std::unique_ptr<ConcreteSharding> Create(
      DeviceListRef devices, 
      MemoryKind memory_kind, 
      Shape shape,
      std::vector<Shape> shard_shapes,
      std::optional<std::vector<IndexDomain>> index_domains = std::nullopt);

  // Factory method for dynamic shapes
  // REQUIRES: devices->AddressableDeviceList()->size() == shard_dynamic_shapes.size()
  // REQUIRES: !devices->empty()
  static std::unique_ptr<ConcreteSharding> Create(
      DeviceListRef devices, 
      MemoryKind memory_kind,
      DynamicShape dynamic_shape,
      std::vector<DynamicShape> shard_dynamic_shapes);

  ~ConcreteSharding() override = default;

  /// Check which type of shape this sharding holds
  bool has_dynamic_shape() const {
    return std::holds_alternative<DynamicShape>(shape_) &&
          std::holds_alternative<std::vector<DynamicShape>>(shard_shapes_);
  }

  bool has_static_shape() const {
    return std::holds_alternative<Shape>(shape_) &&
          std::holds_alternative<std::vector<Shape>>(shard_shapes_);
  }

  // Accessors for shapes (check has_static_shape() / has_dynamic_shape() first!)
  const Shape& shape() const {
    DCHECK(has_static_shape());
    return std::get<Shape>(shape_);
  }

  const DynamicShape& dynamic_shape() const {
    DCHECK(has_dynamic_shape());
    return std::get<DynamicShape>(shape_);
  }

  const std::vector<Shape>& shard_shapes() const {
    DCHECK(std::holds_alternative<std::vector<Shape>>(shard_shapes_));
    return std::get<std::vector<Shape>>(shard_shapes_);
  }

  const std::vector<DynamicShape>& shard_dynamic_shapes() const {
    DCHECK(std::holds_alternative<std::vector<DynamicShape>>(shard_shapes_));
    return std::get<std::vector<DynamicShape>>(shard_shapes_);
  }

  // Sharding interface implementation

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape,
      SingleDeviceShardSemantics semantics) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape,
      SingleDeviceShardSemantics semantics) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics semantics) const override;

  std::string DebugString() const override;

 private:
  // Private constructors - use Create() instead
  ConcreteSharding(DeviceListRef devices, 
                    MemoryKind memory_kind,
                    Shape shape, std::vector<Shape> shard_shapes,
                    std::optional<std::vector<IndexDomain>> index_domains);

  ConcreteSharding(DeviceListRef devices, 
                    MemoryKind memory_kind,
                    DynamicShape dynamic_shape,
                    std::vector<DynamicShape> shard_dynamic_shapes);

  void Hash(absl::HashState state) const override;

  // Either static or dynamic shape (std::variant to save space)
  std::variant<Shape, DynamicShape> shape_;
  
  // Either static or dynamic shard shapes (matches shape_ type)
  std::variant<std::vector<Shape>, std::vector<DynamicShape>> shard_shapes_;
  
  // Cached single shard shape if all shards are identical
  // (optimization: avoid iterating through shard_shapes_)
  std::optional<Shape> shard_shape_;
  
  // Optional index domains describing where each shard comes from in the original array
  std::optional<std::vector<IndexDomain>> index_domains_;
};

// ============================================================================
// ConcreteEvenSharding - All Shards Same Size
// ============================================================================

// Sharding where we explicitly know:
// 1. The full array shape
// 2. Each shard has the SAME shape (uniform sharding)
// 3. Which devices get which shards
//
// This is more efficient than ConcreteSharding when all shards are identical.
// Common case: evenly splitting an array across N devices.
//
// Example: [8, 16] array on 2 devices, split in dim 0
//   -> shape=[8,16], shard_shape=[4,16], 2 devices
class ConcreteEvenSharding final : public Sharding {
 public:
  // Factory method to create a concrete even sharding
  // Parameters:
  //   devices: Devices to distribute shards across
  //   memory_kind: Memory type to use
  //   shape: Full logical array shape
  //   shard_shape: Shape of each individual shard (must be same for all)
  //   is_fully_replicated: True if this is full replication (shard_shape == shape)
  //
  // REQUIRES: !devices->empty()
  static std::unique_ptr<ConcreteEvenSharding> Create(
      DeviceListRef devices, 
      MemoryKind memory_kind, 
      Shape shape,
      Shape shard_shape, 
      bool is_fully_replicated = false);

  ~ConcreteEvenSharding() override = default;

  // Accessors for the stored shapes
  const Shape& shape() const { return shape_; }
  const Shape& shard_shape() const { return shard_shape_; }

  // Sharding interface implementation

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape,
      SingleDeviceShardSemantics semantics) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape,
      SingleDeviceShardSemantics semantics) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics semantics) const override;

  std::string DebugString() const override;

 private:
  // Private constructor - use Create() instead
  ConcreteEvenSharding(DeviceListRef devices, 
                        MemoryKind memory_kind,
                        Shape shape, 
                        Shape shard_shape,
                        bool is_fully_replicated);

  void Hash(absl::HashState state) const override;

  // The full logical array shape
  Shape shape_;
  
  // The shape of each shard (same for all devices)
  Shape shard_shape_;
};

// ============================================================================
// ShardingParamSharding - Computed from IR ShardingParam
// ============================================================================

// Sharding derived from an IR-level ShardingParam specification.
// This is the most compact representation - it algorithmically computes
// shard shapes and placements from a parameter specification.
//
// ShardingParam describes:
// - How many shards per dimension (dim_shards)
// - Device assignment layout (minor_to_major ordering)
// - Replication factor
//
// This is used when interfacing with compiler IR or for efficient
// representation of regular sharding patterns.
//
// Example: [8, 16] array with dim_shards=[2, 1] on 2 devices
//   -> Splits dimension 0 into 2 shards, dimension 1 not split
//   -> Each device gets [4, 16]
class ShardingParamSharding final : public Sharding {
 public:
  // Factory method to create a sharding from ShardingParam
  // Parameters:
  //   sharding_param: The IR sharding specification
  //   devices: Devices to use (count must match param's device count)
  //   memory_kind: Memory type to use
  //
  // REQUIRES: !devices->empty()
  // REQUIRES: Product of sharding_param's axis sizes == devices->size()
  static absl::StatusOr<std::unique_ptr<ShardingParamSharding>> Create(
      ShardingParam sharding_param, 
      DeviceListRef devices,
      MemoryKind memory_kind);

  ~ShardingParamSharding() override = default;

  // Get the underlying ShardingParam
  const ShardingParam& sharding_param() const { return sharding_param_; }

  // Sharding interface implementation

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape,
      SingleDeviceShardSemantics semantics) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape,
      SingleDeviceShardSemantics semantics) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics semantics) const override;

  std::string DebugString() const override;

 private:
  // Private constructor - use Create() instead
  ShardingParamSharding(ShardingParam sharding_param, 
                        DeviceListRef devices,
                        MemoryKind memory_kind);

  void Hash(absl::HashState state) const override;

  // The IR-level sharding specification
  ShardingParam sharding_param_;
};

}  // namespace xftcpp

#endif  // XFTCPP_SHARDING_H_