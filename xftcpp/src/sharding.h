// Copyright 2025 XFT Authors.
// Direct sharding implementations without IFRT abstraction layer.
//
// Sharding describes how array data is distributed across devices.
// This file contains concrete sharding types for common partitioning patterns.

#ifndef XFTCPP_SHARDING_H_
#define XFTCPP_SHARDING_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "device.h"
#include "device_list.h"
#include "index_domain.h"
#include "memory.h"
#include "shape.h"
#include "sharding_param.h"

namespace xftcpp {

// Forward declarations
class Client;

// Shared pointer to sharding (used throughout the codebase for reference counting)
class Sharding;
using ShardingRef = std::shared_ptr<const Sharding>;

// Controls whether single-device shard operations include only addressable devices
// or all devices (including non-addressable ones in multi-host scenarios).
enum class SingleDeviceShardSemantics : int {
  // Process only shards on addressable devices (devices this process can control).
  // Assembly requires single-device arrays only for addressable shards.
  // Disassembly returns single-device arrays only for addressable shards.
  kAddressableShards = 0,

  // Process shards on all devices (addressable and non-addressable).
  // Assembly requires single-device arrays for all shards.
  // Disassembly returns single-device arrays for all shards.
  // Not supported by all runtimes (requires expressing arrays on non-addressable devices).
  kAllShards,
};

// Base sharding functionality shared by all concrete sharding types.
// This is NOT an abstract base class - it's a concrete class that holds common data.
// Specific sharding types (SingleDevice, Opaque, etc.) contain an instance of this.
class Sharding {
 public:

  // Not copyable or movable
  Sharding(const Sharding&) = delete;
  Sharding(Sharding&&) = delete;
  Sharding& operator=(const Sharding&) = delete;
  Sharding& operator=(Sharding&&) = delete;

  virtual ~Sharding() = default;

  // All devices in this sharding. Devices may appear more than once.
  const DeviceListRef& devices() const { return devices_; }

  // Memory kind for all shards in this sharding.
  MemoryKind memory_kind() const { return memory_kind_; }

  // Whether this sharding is fully replicated.
  // Fully replicated means logical shape == shard shape, and every shard
  // contains the entire data of the logical array.
  bool IsFullyReplicated() const { return is_fully_replicated_; }

  // Returns if this sharding is equal to `other`.
  bool operator==(const Sharding& other) const;
  bool operator!=(const Sharding& other) const { return !(*this == other); }

  // Returns shard shape if the sharding has a uniform shape for all shards.
  // Returns error if sharding may not have a single shard shape,
  // or if shape is not valid for this sharding.
  virtual absl::StatusOr<Shape> GetShardShape(const Shape& shape) const = 0;

  // Returns whether this sharding has the same logical partitioning as other.
  // Same partitioning means same sharding type and equivalent partitioning scheme.
  // Does NOT check if devices or memory kinds are the same.
  virtual bool HasSamePartitioning(const Sharding& other) const = 0;

  // Returns a new sharding with same logical partitioning but different devices/memory.
  // If devices provided, count must match current device count.
  // If memory_kind provided, must be valid for the devices.
  virtual absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const = 0;

  // Breaks a shape into per-device shapes and shardings.
  // May return error if disassembly is unsupported for this sharding type.
  virtual absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
  Disassemble(const Shape& shape) const = 0;
  
  virtual absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
  Disassemble(const Shape& shape,
              SingleDeviceShardSemantics single_device_shard_semantics) const = 0;

  // Variant of Disassemble for dynamic shapes.
  virtual absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
  Disassemble(const DynamicShape& dynamic_shape) const = 0;
  
  virtual absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
  Disassemble(const DynamicShape& dynamic_shape,
              SingleDeviceShardSemantics single_device_shard_semantics) const = 0;

  // Maps each shard to an IndexDomain over shape.
  // Result is a list where array[index_domain_i] = disassembled_array_i.
  // Multiple shards may map to equal IndexDomain (e.g., replicated sharding).
  virtual absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const = 0;
  
  virtual absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const = 0;

  // Debug string for logging and error messages.
  virtual std::string DebugString() const = 0;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Sharding& sharding) {
    sink.Append(sharding.DebugString());
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const std::shared_ptr<const Sharding>& sharding) {
    if (sharding == nullptr) {
      sink.Append("<nullptr>");
    } else {
      sink.Append(sharding->DebugString());
    }
  }

 protected:
  Sharding(DeviceListRef devices, MemoryKind memory_kind,
           bool is_fully_replicated);
  DeviceListRef devices_;
  MemoryKind memory_kind_;
  bool is_fully_replicated_;
};

std::ostream& operator<<(std::ostream& os, const Sharding& sharding);

// Single-device sharding - data lives entirely on one device.
// The simplest sharding: no partitioning, just placement on a single device.
class SingleDeviceSharding final : public Sharding {
 public:
  // Creates a single-device sharding.
  static std::unique_ptr<SingleDeviceSharding> Create(Device* device,
                                                      MemoryKind memory_kind);

  ~SingleDeviceSharding() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape) const override;
  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape) const override;
  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;
  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  std::string DebugString() const override;

  template <typename H>
  friend H AbslHashValue(H h, const SingleDeviceSharding& sharding) {
    return H::combine(std::move(h), sharding.devices_, sharding.memory_kind_);
  }

 private:
  explicit SingleDeviceSharding(DeviceListRef device_list,
                                MemoryKind memory_kind);
};

// Opaque sharding - device assignment is known but partitioning semantics are not.
// Cannot disassemble into per-device shapes because partitioning is undefined.
// Used when you know WHERE data lives but not HOW it's split.
class OpaqueSharding final : public Sharding {
 public:
  // Creates an opaque sharding. Disassemble() will fail.
  static std::unique_ptr<OpaqueSharding> Create(DeviceListRef devices,
                                                MemoryKind memory_kind);

  ~OpaqueSharding() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape) const override;
  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape) const override;
  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;
  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  std::string DebugString() const override;

  template <typename H>
  friend H AbslHashValue(H h, const OpaqueSharding& sharding) {
    return H::combine(std::move(h), sharding.devices_, sharding.memory_kind_);
  }

 private:
  explicit OpaqueSharding(DeviceListRef devices, MemoryKind memory_kind);
};

// Concrete sharding with explicit per-shard shapes that may differ.
// Stores the exact shape for each addressable device shard.
// More flexible than ConcreteEvenSharding but requires storing all shard shapes.
class ConcreteSharding final : public Sharding {
 public:
  // Creates a concrete sharding with potentially non-identical shard shapes.
  // Requires: devices->AddressableDeviceList()->size() == shard_shapes.size()
  static std::unique_ptr<ConcreteSharding> Create(
      DeviceListRef devices, MemoryKind memory_kind, Shape shape,
      std::vector<Shape> shard_shapes,
      std::optional<std::vector<IndexDomain>> index_domains = std::nullopt);

  // Creates a concrete sharding with dynamic shapes.
  static std::unique_ptr<ConcreteSharding> Create(
      DeviceListRef devices, MemoryKind memory_kind, DynamicShape dynamic_shape,
      std::vector<DynamicShape> shard_dynamic_shapes);

  bool has_dynamic_shape() const {
    return std::holds_alternative<DynamicShape>(shape_) &&
           std::holds_alternative<std::vector<DynamicShape>>(shard_shapes_);
  }

  bool has_static_shape() const {
    return std::holds_alternative<Shape>(shape_) &&
           std::holds_alternative<std::vector<Shape>>(shard_shapes_);
  }

  const Shape& shape() const { return std::get<Shape>(shape_); }
  const DynamicShape& dynamic_shape() const { return std::get<DynamicShape>(shape_); }
  const std::vector<Shape>& shard_shapes() const {
    return std::get<std::vector<Shape>>(shard_shapes_);
  }
  const std::vector<DynamicShape>& shard_dynamic_shapes() const {
    return std::get<std::vector<DynamicShape>>(shard_shapes_);
  }

  ~ConcreteSharding() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape) const override;
  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape) const override;
  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;
  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  std::string DebugString() const override;

  template <typename H>
  friend H AbslHashValue(H h, const ConcreteSharding& sharding) {
    return H::combine(std::move(h), sharding.devices_, sharding.memory_kind_,
                     sharding.shape_, sharding.shard_shapes_);
  }

 private:
  ConcreteSharding(DeviceListRef devices, MemoryKind memory_kind, Shape shape,
                   std::vector<Shape> shard_shapes,
                   std::optional<std::vector<IndexDomain>> index_domains);

  ConcreteSharding(DeviceListRef devices, MemoryKind memory_kind,
                   DynamicShape dynamic_shape,
                   std::vector<DynamicShape> shard_dynamic_shapes);

  std::variant<Shape, DynamicShape> shape_;
  std::variant<std::vector<Shape>, std::vector<DynamicShape>> shard_shapes_;
  std::optional<Shape> shard_shape_;  // Cached if all shards have same shape
  std::optional<std::vector<IndexDomain>> index_domains_;
};

// Concrete even sharding - all shards have identical shapes.
// More efficient than ConcreteSharding when all shards are uniform.
// Stores only one shard shape instead of per-device shapes.
class ConcreteEvenSharding final : public Sharding {
 public:
  // Creates a concrete even sharding where all shards have the same shape.
  static std::unique_ptr<ConcreteEvenSharding> Create(
      DeviceListRef devices, MemoryKind memory_kind, Shape shape,
      Shape shard_shape, bool is_fully_replicated = false);

  Shape shape() const { return shape_; }
  const Shape& shard_shape() const { return shard_shape_; }

  ~ConcreteEvenSharding() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape) const override;
  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape) const override;
  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;
  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  std::string DebugString() const override;

  template <typename H>
  friend H AbslHashValue(H h, const ConcreteEvenSharding& sharding) {
    return H::combine(std::move(h), sharding.devices_, sharding.memory_kind_,
                     sharding.is_fully_replicated_, sharding.shape_,
                     sharding.shard_shape_);
  }

 private:
  ConcreteEvenSharding(DeviceListRef devices, MemoryKind memory_kind,
                       Shape shape, Shape shard_shape,
                       bool is_fully_replicated);

  Shape shape_;
  Shape shard_shape_;
};

// Sharding derived from a ShardingParam (describes tiled partitioning).
// ShardingParam specifies how to tile an array across devices using
// per-dimension shard counts and device ordering.
class ShardingParamSharding final : public Sharding {
 public:
  // Creates a sharding from a ShardingParam.
  // The device count in sharding_param must match devices->size().
  static absl::StatusOr<std::unique_ptr<ShardingParamSharding>> Create(
      ShardingParam sharding_param, DeviceListRef devices,
      MemoryKind memory_kind);

  const ShardingParam& sharding_param() const { return sharding_param_; }

  ~ShardingParamSharding() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape) const override;
  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape) const override;
  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;
  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  std::string DebugString() const override;

  template <typename H>
  friend H AbslHashValue(H h, const ShardingParamSharding& sharding) {
    return H::combine(std::move(h), sharding.devices_, sharding.memory_kind_,
                     sharding.is_fully_replicated_, sharding.sharding_param_);
  }

 private:
  ShardingParamSharding(ShardingParam sharding_param, DeviceListRef devices,
                        MemoryKind memory_kind);

  ShardingParam sharding_param_;
};

}  // namespace xftcpp

#endif  // XFTCPP_SHARDING_H_