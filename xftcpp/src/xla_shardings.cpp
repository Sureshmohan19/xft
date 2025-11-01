/* XLA Sharding Implementation for xftcpp
 * Adapted from XLA's IFRT XLA Sharding
 * 
 * This file implements HloSharding, which wraps XLA's native HloSharding
 * representation for use with xftcpp's sharding interface.
 */

#include "xftcpp/src/xla_shardings.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"

#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

#include "xftcpp/src/device.h"
#include "xftcpp/src/device_list.h"
#include "xftcpp/src/memory.h"
#include "xftcpp/src/shape.h"
#include "xftcpp/src/sharding.h"

namespace xftcpp {

// Forward declaration for Index and IndexDomain
// These need to be defined elsewhere in the codebase
class Index {
 public:
  using Elements = std::vector<int64_t>;
  explicit Index(Elements elements) : elements_(std::move(elements)) {}
  explicit Index(absl::Span<const int64_t> elements)
      : elements_(elements.begin(), elements.end()) {}
  
  const Elements& elements() const { return elements_; }
  
  // Element-wise multiplication with a span (for computing origins)
  Index operator*(absl::Span<const int64_t> dims) const {
    Elements result(elements_.size());
    for (size_t i = 0; i < elements_.size(); ++i) {
      result[i] = elements_[i] * dims[i];
    }
    return Index(std::move(result));
  }
  
 private:
  Elements elements_;
};

class IndexDomain {
 public:
  IndexDomain(const Shape& shape) : origin_(), shape_(shape) {}
  
  IndexDomain(Index origin, Shape shape) 
      : origin_(std::move(origin)), shape_(std::move(shape)) {}
  
  const Shape& shape() const { return shape_; }
  const Index& origin() const { return origin_; }
  
 private:
  Index origin_;
  Shape shape_;
};

namespace {

// ============================================================================
// Helper Functions
// ============================================================================

// Generates IndexDomains for an HloSharding using XLA HloSharding APIs.
//
// This is the "slow path" implementation that uses XLA's TileOffsetForDevice
// and TileLimitForDevice methods. It's O(N^2) where N is the number of devices.
//
// Why O(N^2)? Each call to TileOffsetForDevice/TileLimitForDevice may need to:
// 1. Traverse the tile assignment array
// 2. Compute device positions in the logical tile mesh
// 3. Handle subgroup replication
//
// This function is used as a fallback when:
// - The sharding has complex subgroup types (not just REPLICATED)
// - The sharding is not a simple tiled pattern
// - We need a reference implementation for testing
//
// Parameters:
//   hlo_sharding: The XLA HloSharding specification
//   devices: Device list (size must match tile assignment)
//   shape: The full array shape
//   single_device_shard_semantics: Whether to include non-addressable devices
//
// Returns: Vector of IndexDomain, one per device
std::vector<IndexDomain> IndexDomainsSlowPath(
    const xla::HloSharding& hlo_sharding, const DeviceListRef& devices,
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  // Create an XLA Shape - only dimensions matter, not the element type.
  // We use S32 as a dummy type since XLA shape utilities require it.
  // MakeShapeWithDescendingLayout ensures dimensions are in row-major order.
  auto xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      xla::PrimitiveType::S32, shape.dims());
  
  // Warn if using slow path with many devices - this can be a perf issue
  if (devices->size() > 8) {
    LOG_FIRST_N(WARNING, 1)
        << "Taking a slow path for HloSharding::IndexDomains(). This will not "
           "scale for a large number of devices.";
  }

  std::vector<IndexDomain> result;
  result.reserve(devices->size());
  
  // Pre-allocate temporary buffers to avoid repeated allocations
  Index::Elements origin(shape.dims().size());
  Shape::Dimensions shard_shape(shape.dims().size());
  
  const absl::Span<Device* const> device_ptrs = devices->devices();
  
  // Iterate through each device and compute its tile bounds
  for (int device_idx = 0; device_idx < device_ptrs.size(); ++device_idx) {
    // Check if we should include this device based on semantics
    if (single_device_shard_semantics ==
            SingleDeviceShardSemantics::kAllShards ||
        device_ptrs[device_idx]->IsAddressable()) {
      // Get the offset (starting position) of this device's tile
      auto tile_offset =
          hlo_sharding.TileOffsetForDevice(xla_shape, device_idx);
      // Get the limit (ending position + 1) of this device's tile
      auto tile_limit = hlo_sharding.TileLimitForDevice(xla_shape, device_idx);
      
      // Compute origin and shape for this tile
      for (int i = 0; i < shape.dims().size(); ++i) {
        origin[i] = tile_offset[i];
        shard_shape[i] = tile_limit[i] - tile_offset[i];
      }
      
      result.push_back(IndexDomain(Index(origin), Shape(shard_shape)));
    }
  }
  
  return result;
}

// Returns a canonicalized memory kind for the given devices.
//
// Memory kind canonicalization ensures:
// - Device-specific memory kinds are resolved to concrete types
// - Default memory kinds are mapped to appropriate device memory
// - Invalid memory kinds are detected early
//
// REQUIRES: !devices->devices().empty()
MemoryKind CanonicalizeMemoryKindWithDevices(const MemoryKind& memory_kind,
                                             const DeviceListRef& devices) {
  CHECK(devices != nullptr);
  CHECK(!devices->devices().empty());
  return CanonicalizeMemoryKind(memory_kind, devices->devices().front());
}

}  // namespace

// ============================================================================
// HloSharding Implementation
// ============================================================================

std::unique_ptr<HloSharding> HloSharding::Create(
    DeviceListRef devices, MemoryKind memory_kind,
    xla::HloSharding xla_hlo_sharding) {
  // Canonicalize the memory kind to ensure it's valid for these devices
  memory_kind = CanonicalizeMemoryKindWithDevices(memory_kind, devices);
  
  // Use private constructor via unique_ptr
  return std::unique_ptr<HloSharding>(new HloSharding(
      std::move(devices), memory_kind, std::move(xla_hlo_sharding)));
}

HloSharding::HloSharding(DeviceListRef devices, MemoryKind memory_kind,
                         xla::HloSharding xla_hlo_sharding)
    : Sharding(std::move(devices), memory_kind,
               // is_fully_replicated is computed here because it needs to
               // access devices_ after it's been moved into the base class.
               // Computing it before the move would be unsafe.
               /*is_fully_replicated=*/false),
      xla_hlo_sharding_(std::move(xla_hlo_sharding)) {
  // Compute whether this sharding is fully replicated.
  //
  // A sharding is fully replicated when:
  // 1. XLA marks it as replicated (IsReplicated()), OR
  // 2. It's tiled/tile-maximal but only has 1 device (trivially replicated)
  //
  // Why check device count? A single-device tiled sharding is functionally
  // equivalent to replication - there's only one copy of the data.
  is_fully_replicated_ =
      xla_hlo_sharding_.IsReplicated() ||
      ((xla_hlo_sharding_.IsTiled() || xla_hlo_sharding_.IsTileMaximal()) &&
       devices_->size() == 1);
}

absl::StatusOr<Shape> HloSharding::GetShardShape(const Shape& shape) const {
  // Handle special sharding types that don't partition the data:
  // - TileMaximal: All data on one device (or replicated)
  // - Manual: User-managed sharding (assumes full shape)
  // - Unreduced: Special marker for unreduced collectives
  // - Unknown: Sharding info not available
  if (xla_hlo_sharding_.IsTileMaximal() || xla_hlo_sharding_.IsManual() ||
      xla_hlo_sharding_.IsUnreduced() || xla_hlo_sharding_.IsUnknown()) {
    return shape;
  }

  // Validate: tile count must match device count
  // TotalNumTiles() includes replication - each tile may be replicated
  if (xla_hlo_sharding_.TotalNumTiles() != devices_->size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("sharding's tile count and device count does not "
                        "match: %d vs. %d; shape=%s, sharding=%s",
                        xla_hlo_sharding_.TotalNumTiles(), devices_->size(),
                        shape.DebugString(), xla_hlo_sharding_.ToString()));
  }

  // Validate: shape rank must match sharding rank
  // TiledDataRank() is the number of dimensions being tiled
  if (shape.dims().size() != xla_hlo_sharding_.TiledDataRank()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Numbers of dimensions don't match. From Shape %d vs from "
        "HloSharding %d",
        shape.dims().size(), xla_hlo_sharding_.TiledDataRank()));
  }

  // Compute the shard shape by dividing each dimension by its tile count
  //
  // Example: shape=[8, 16], tile_assignment=[2, 4] (8 devices)
  //   -> shard_shape=[ceil(8/2), ceil(16/4)] = [4, 4]
  //
  // Uses CeilOfRatio to handle uneven division:
  //   shape=[9, 16], tile_assignment=[2, 4]
  //   -> shard_shape=[ceil(9/2), ceil(16/4)] = [5, 4]
  //   Device 0 gets [5,4], Device 1 gets [4,4] (last device smaller)
  const absl::Span<const int64_t> tile_assignment_dims =
      xla_hlo_sharding_.tile_assignment().dimensions();
  
  Shape::Dimensions tile_shape;
  tile_shape.reserve(shape.dims().size());
  for (int64_t i = 0; i < shape.dims().size(); ++i) {
    tile_shape.push_back(
        xla::CeilOfRatio(shape.dims()[i], tile_assignment_dims[i]));
  }
  
  return Shape(std::move(tile_shape));
}

bool HloSharding::HasSamePartitioning(const Sharding& other) const {
  // Fast path: same object
  if (this == &other) {
    return true;
  }
  
  // Quick check: device count must match for same partitioning
  if (devices()->size() != other.devices()->size()) {
    return false;
  }
  
  // Check if other is also an HloSharding
  const auto* other_hlo_sharding = dynamic_cast<const HloSharding*>(&other);
  if (!other_hlo_sharding) {
    return false;
  }
  
  // Compare the underlying XLA HloSharding objects
  // This compares the tile assignment, replication groups, etc.
  return xla_hlo_sharding_ == other_hlo_sharding->xla_hlo_sharding_;
}

absl::StatusOr<std::unique_ptr<Sharding>> HloSharding::WithDeviceAssignment(
    std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind) const {
  // Validate: device count must match if changing devices
  // We can't change the partitioning, only the device assignment
  if (devices.has_value() && (*devices)->size() != devices_->size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "HloSharding should have the same number of devices as the current "
        "sharding, but was asked to have %d devices",
        (*devices)->size()));
  }
  
  // Create new HloSharding with updated assignment but same partitioning
  return Create(devices.value_or(devices_), memory_kind.value_or(memory_kind_),
                xla_hlo_sharding_);
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
HloSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  return Disassemble(shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
HloSharding::Disassemble(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  
  // Determine if this is an "even" sharding where all shards have the same size.
  // Even shardings can use a fast path that avoids computing individual IndexDomains.
  //
  // A sharding is even if:
  // 1. It's replicated (all shards = full array)
  // 2. It's tile maximal (single tile, possibly replicated)
  // 3. It's unreduced (special marker, treated as full shape)
  // 4. It's tiled AND each dimension divides evenly by tile count
  // 5. It's manual (by convention, same global/shard shapes)
  bool is_even_sharding = false;
  
  if (xla_hlo_sharding_.IsReplicated() || xla_hlo_sharding_.IsTileMaximal() ||
      xla_hlo_sharding_.IsUnreduced()) {
    is_even_sharding = true;
  } else if (xla_hlo_sharding_.IsTiled()) {
    const int64_t tiled_data_rank = xla_hlo_sharding_.TiledDataRank();
    
    // Validate rank matches
    if (shape.dims().size() != tiled_data_rank) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "shape must have %d dimensions, but has %d dimensions: "
          "shape=%s, sharding=%s",
          tiled_data_rank, shape.dims().size(), shape.DebugString(),
          xla_hlo_sharding_.ToString()));
    }
    
    // Check if all dimensions divide evenly
    is_even_sharding = true;
    for (int i = 0; i < tiled_data_rank; ++i) {
      if (shape.dims()[i] % xla_hlo_sharding_.tile_assignment().dim(i) != 0) {
        is_even_sharding = false;
        break;
      }
    }
  } else if (xla_hlo_sharding_.IsManual()) {
    // By convention, MANUAL sharding has the same global/shard shapes.
    // This is used for cases where the user manually manages data distribution.
    is_even_sharding = true;
  }

  const absl::Span<Device* const> devices = devices_->devices();
  
  if (is_even_sharding) {
    // ========================================================================
    // Fast path for even sharding
    // ========================================================================
    // All shards have the same shape, so compute once and reuse.
    // This avoids expensive per-device IndexDomain computation.
    
    auto shard_shape_result = GetShardShape(shape);
    if (!shard_shape_result.ok()) {
      return shard_shape_result.status();
    }
    Shape shard_shape = std::move(shard_shape_result).value();
    
    std::vector<std::pair<Shape, ShardingRef>> result;
    if (single_device_shard_semantics ==
        SingleDeviceShardSemantics::kAllShards) {
      result.reserve(devices_->size());
    } else {
      result.reserve(devices_->AddressableDeviceList()->size());
    }
    
    for (int i = 0; i < devices_->size(); ++i) {
      if (single_device_shard_semantics ==
              SingleDeviceShardSemantics::kAllShards ||
          devices[i]->IsAddressable()) {
        result.push_back({
            shard_shape,
            SingleDeviceSharding::Create(devices[i], memory_kind_),
        });
      }
    }
    
    return result;
  }

  // ==========================================================================
  // Slow path for uneven sharding
  // ==========================================================================
  // Use IndexDomains() to get per-device shard shapes.
  // This is needed when different devices get different-sized shards.
  //
  // Example: shape=[9, 16] tiled [2, 4] (8 devices)
  //   Device 0-3: [5, 4] each
  //   Device 4-7: [4, 4] each
  
  auto index_domains_result = IndexDomains(shape);
  if (!index_domains_result.ok()) {
    return index_domains_result.status();
  }
  std::vector<IndexDomain> index_domains = 
      std::move(index_domains_result).value();
  
  CHECK_EQ(index_domains.size(), devices_->size());
  
  std::vector<std::pair<Shape, ShardingRef>> result;
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards) {
    result.reserve(devices_->size());
  } else {
    result.reserve(devices_->AddressableDeviceList()->size());
  }
  
  for (int i = 0; i < index_domains.size(); ++i) {
    if (single_device_shard_semantics ==
            SingleDeviceShardSemantics::kAllShards ||
        devices[i]->IsAddressable()) {
      result.push_back({
          index_domains[i].shape(),
          SingleDeviceSharding::Create(devices[i], memory_kind_),
      });
    }
  }
  
  return result;
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
HloSharding::Disassemble(const DynamicShape& dynamic_shape) const {
  DCHECK(this);
  return Disassemble(dynamic_shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
HloSharding::Disassemble(
    const DynamicShape& dynamic_shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  // HloSharding currently only supports static shapes.
  // Dynamic shapes (with runtime-variable dimensions) would require
  // XLA to support dynamic shapes in its sharding representation.
  return absl::InvalidArgumentError(absl::StrFormat(
      "HloSharding can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %s",
      dynamic_shape.DebugString()));
}

absl::StatusOr<std::vector<IndexDomain>> HloSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  return IndexDomains(shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<IndexDomain>> HloSharding::IndexDomains(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  std::vector<IndexDomain> result;
  const int num_devices = devices_->size();
  
  // ========================================================================
  // Handle Manual Sharding
  // ========================================================================
  // Manual sharding means the user is managing data distribution themselves.
  // XLA doesn't know how to compute index domains in this case.
  if (xla_hlo_sharding_.IsManual()) {
    return absl::InvalidArgumentError(
        "Manual sharding does not support IndexDomains");
  }
  
  // ========================================================================
  // Fast path: Replicated or TileMaximal
  // ========================================================================
  // All devices have the full array, so all IndexDomains are identical.
  if (xla_hlo_sharding_.IsReplicated() || xla_hlo_sharding_.IsTileMaximal()) {
    IndexDomain element(shape);
    if (single_device_shard_semantics ==
        SingleDeviceShardSemantics::kAllShards) {
      result.resize(/*count=*/num_devices, /*value=*/element);
    } else {
      result.resize(/*count=*/devices_->AddressableDeviceList()->size(),
                    /*value=*/element);
    }
    return result;
  }
  
  // ========================================================================
  // Use slow path for non-tiled shardings
  // ========================================================================
  // IsTiled() checks if this is a regular tiled sharding.
  // If not, we need to use the slow XLA API path.
  if (!xla_hlo_sharding_.IsTiled()) {
    return IndexDomainsSlowPath(xla_hlo_sharding_, devices_, shape,
                                single_device_shard_semantics);
  }
  
  // ========================================================================
  // Check subgroup types
  // ========================================================================
  // Subgroups represent nested replication patterns.
  // Example: A 4x2 device mesh where each 2-device group is replicated.
  //
  // If any subgroup type is not REPLICATED, we need the slow path
  // because the index domain computation is more complex.
  for (const xla::OpSharding::Type subgroup_type :
       xla_hlo_sharding_.subgroup_types()) {
    if (subgroup_type != xla::OpSharding::REPLICATED) {
      return IndexDomainsSlowPath(xla_hlo_sharding_, devices_, shape,
                                  single_device_shard_semantics);
    }
  }
  
  // ========================================================================
  // Validate tile assignment matches device count
  // ========================================================================
  if (xla_hlo_sharding_.tile_assignment().num_elements() != num_devices) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "sharding's tile_assignment_devices and device count does not "
        "match: %d vs. %d; shape=%s, sharding=%s",
        xla_hlo_sharding_.tile_assignment().num_elements(), num_devices,
        shape.DebugString(), DebugString()));
  }
  
  // ========================================================================
  // Validate shape rank
  // ========================================================================
  const int64_t tiled_data_rank = xla_hlo_sharding_.TiledDataRank();
  if (shape.dims().size() != tiled_data_rank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("shape must have %d dimensions, but has %d dimensions: "
                        "shape=%s, sharding=%s",
                        tiled_data_rank, shape.dims().size(),
                        shape.DebugString(), xla_hlo_sharding_.ToString()));
  }
  
  // ========================================================================
  // Fast path: Simple tiled sharding with only replicated subgroups
  // ========================================================================
  // We can use XLA's EachTile() API which efficiently iterates through tiles.
  
  auto tile_shape_result = GetShardShape(shape);
  if (!tile_shape_result.ok()) {
    return tile_shape_result.status();
  }
  Shape tile_shape = std::move(tile_shape_result).value();
  
  const absl::Span<const int64_t> shape_dims = shape.dims();
  
  // Pre-allocate array to hold all IndexDomains
  // We compute for all devices first, then filter based on semantics
  std::vector<std::optional<IndexDomain>> all(num_devices);
  
  // EachTile() calls the lambda for each tile in the tile assignment.
  // The lambda receives:
  // - device_index: which device this tile is assigned to
  // - tile_offset: starting position of this tile in the full array
  // - tile_limit: ending position (exclusive) of this tile
  //
  // Example: shape=[8,16], tile_assignment=[2,4]
  //   Tile (0,0) -> device 0, offset=[0,0], limit=[4,4]
  //   Tile (0,1) -> device 1, offset=[0,4], limit=[4,8]
  //   Tile (1,0) -> device 4, offset=[4,0], limit=[8,4]
  //   ... etc
  auto status = xla_hlo_sharding_.EachTile(
      shape_dims, [shape_dims, &all](int device_index,
                                     absl::Span<const int64_t> tile_offset,
                                     absl::Span<const int64_t> tile_limit) {
        // Compute the tile shape from offset and limit
        Shape::Dimensions tile_shape;
        tile_shape.reserve(shape_dims.size());
        for (int i = 0; i < shape_dims.size(); ++i) {
          tile_shape.push_back(tile_limit[i] - tile_offset[i]);
        }
        
        all[device_index] =
            IndexDomain(Index(tile_offset), Shape(std::move(tile_shape)));
      });
  
  if (!status.ok()) {
    return status;
  }
  
  // ========================================================================
  // Filter based on device addressability
  // ========================================================================
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards) {
    result.reserve(num_devices);
  } else {
    result.reserve(devices_->AddressableDeviceList()->size());
  }
  
  const absl::Span<Device* const> devices = devices_->devices();
  for (int device_idx = 0; device_idx < num_devices; ++device_idx) {
    if (single_device_shard_semantics ==
            SingleDeviceShardSemantics::kAllShards ||
        devices[device_idx]->IsAddressable()) {
      result.push_back(*std::move(all[device_idx]));
    }
  }
  
  return result;
}

std::string HloSharding::DebugString() const {
  return absl::StrFormat("HloSharding(memory_kind: %v, hlo_sharding: %s)",
                         memory_kind_, xla_hlo_sharding_.ToString());
}

void HloSharding::Hash(absl::HashState state) const {
  // Load the cached hash (relaxed memory order is fine - see explanation below)
  uint64_t hash = hash_.load(std::memory_order_relaxed);
  
  if (hash == kUnsetHash) {
    // Hash not computed yet - compute it now
    //
    // Why hash these three components?
    // 1. devices_: Different device assignment = different sharding
    // 2. memory_kind_: Different memory type = different sharding
    // 3. xla_hlo_sharding_: Different partitioning = different sharding
    hash = absl::HashOf(devices_, memory_kind_, xla_hlo_sharding_);
    
    // Extremely unlikely: if the hash happens to be kUnsetHash (0),
    // increment it to avoid confusion with "not yet computed"
    if (ABSL_PREDICT_FALSE(hash == kUnsetHash)) {
      ++hash;
    }
    
    // Store the computed hash for future use
    //
    // Thread safety note: We use relaxed memory order because:
    // 1. Multiple threads may race to compute and write the hash
    // 2. They'll all compute the SAME value (hash function is deterministic)
    // 3. Writing the same value multiple times is benign
    // 4. We don't need synchronization - worst case we compute twice
    //
    // This is a performance optimization to avoid expensive synchronization
    // (mutex, atomic with stricter ordering) for a benign data race.
    hash_.store(hash, std::memory_order_relaxed);
  }
  
  // Combine the (possibly cached) hash into the state
  absl::HashState::combine(std::move(state), hash);
}

// ============================================================================
// Test Utilities
// ============================================================================

std::vector<IndexDomain> TEST_HloShardingIndexDomainsSlowPath(
    const HloSharding& hlo_sharding, const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  // Expose the internal slow path for testing.
  // This allows tests to validate that the fast path produces the same
  // results as the reference implementation.
  return IndexDomainsSlowPath(hlo_sharding.xla_hlo_sharding(),
                              hlo_sharding.devices(), shape,
                              single_device_shard_semantics);
}

}  // namespace xftcpp

