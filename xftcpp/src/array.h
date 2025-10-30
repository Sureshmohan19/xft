/* XFT-CPP Array - Core Array Implementation
 *
 * This is the connecting layer between Python arrays (PyArray) and the actual
 * XLA PJRT implementation. It directly wraps PJRT buffers and provides array
 * operations without touching the underlying PJRT implementation.
 *
 * Structure:
 * ==========
 * PyArray (Python-facing wrapper)
 *     ↓
 * xftcpp::Array (this file - C++ array logic)
 *     ↓
 * xla::PjRtBuffer (XLA's device memory - not modified by us)
 *
 * Responsibilities:
 * ================
 * - Hold references to one or more PjRtBuffers (sharded arrays)
 * - Manage array metadata: dtype, shape, sharding, layout
 * - Provide operations: copy, transfer, disassemble
 * - Validate buffer and sharding consistency
 *
 * This is pure C++ with no Python dependencies.
 */

#ifndef XFTCPP_ARRAY_H_
#define XFTCPP_ARRAY_H_

#include <cstdint>
#include <memory>
#include <vector>

// XLA PJRT includes (the only external dependency we use)
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"

#include "absl/status/statusor.h"

namespace xftcpp {

// Forward declarations for our own types (we'll implement these later)
class DType;
class Shape;
class Sharding;

// Our core array class
class XFTArray {
 public:
    // Type alias for holding PJRT buffers. Currently uses std::vector for simplicity.
    // Future optimization: Use absl::InlinedVector<std::shared_ptr<xla::PjRtBuffer>, 1>    
    // to avoid heap allocation for single-shard arrays (the common case).    
    using PjRtBuffers = std::vector<std::shared_ptr<xla::PjRtBuffer>>;

	// ============================================================================
	// Factory Methods (Create)
	// ============================================================================

	// Full array construction with static shape. Provides complete control.
	// - client: PJRT client managing devices (must not be nullptr)
	// - dtype: Data type of array elements
	// - shape: Dimensions of the full array (static, known at creation)
	// - sharding: Describes how array is distributed across devices
	// - pjrt_buffers: Device buffers (may be empty for placeholder arrays)
	// - layout: Memory layout (nullptr = default layout)
	// Returns error if validation fails (buffer count mismatch, nullptr client, etc.)
	static absl::StatusOr<std::shared_ptr<XFTArray>> Create(
		xla::PjRtClient* client,
		DType dtype,
		Shape shape,
		Sharding sharding,
		PjRtBuffers pjrt_buffers,
		std::shared_ptr<const xla::PjRtLayout> layout);

	// Array construction with dynamic shape (dimensions can change at runtime).
	// Used for advanced cases like variable-length sequences in RNNs.
	// Parameters same as static shape version, but shape can have dynamic dimensions.
	// Note: Dynamic shapes are complex - implement only when needed for specific models.
	static absl::StatusOr<std::shared_ptr<XFTArray>> Create(
		xla::PjRtClient* client,
		DType dtype,
		Shape dynamic_shape,  // Will support dynamic dimensions later
		Sharding sharding,
		PjRtBuffers pjrt_buffers,
		std::shared_ptr<const xla::PjRtLayout> layout);

	// Shorthand for single-device arrays (most common case).
	// Auto-infers dtype, shape, and sharding from the buffer itself.
	// - client: PJRT client managing devices
	// - pjrt_buffer: Single device buffer containing the data
	// Returns fully-formed array without needing to specify metadata explicitly.
	static absl::StatusOr<std::shared_ptr<XFTArray>> Create(
		xla::PjRtClient* client,
		std::shared_ptr<xla::PjRtBuffer> pjrt_buffer);

	// ============================================================================
	// NOT YET IMPLEMENTED (will add when needed):
	// ============================================================================
	// - Multi-shard shorthand: Create(client, shape, pjrt_buffers)
	//   Reason: Can use full Create() method instead, adds complexity without much benefit
	//   Will implement if we find repeated patterns that need this convenience method
	//
	// - Dynamic shape multi-shard: Create(client, dynamic_shape, pjrt_buffers)
	//   Reason: Dynamic shapes are advanced, wait until we have use cases requiring them
	//   Will implement after basic dynamic shape support is stable

	// ============================================================================
	// Accessors (Getters)
	// ============================================================================
	// All getters return by value (not reference) for safety. While returning
	// references is more efficient, it can cause dangling reference bugs if the
	// array is deleted while caller holds the reference. Return-by-value is safer
	// and the performance cost is negligible for small metadata types.

	// Get read-only access to the underlying PJRT buffers
	absl::Span<const std::shared_ptr<xla::PjRtBuffer>> pjrt_buffers() const {
		return absl::MakeConstSpan(pjrt_buffers_);
	}

	// Get mutable access to the underlying PJRT buffers
	// Returns error if array is in invalid state (e.g., deleted)
	absl::StatusOr<absl::Span<std::shared_ptr<xla::PjRtBuffer>>> mutable_pjrt_buffers() {
		return absl::MakeSpan(pjrt_buffers_);
	}

	// Get the memory layout (nullptr means default layout)
 	std::shared_ptr<const xla::PjRtLayout> layout() const { return layout_; }

	// Get the PJRT client that manages devices for this array
	xla::PjRtClient* client() const { return client_; }

	// Get the data type of array elements (float32, int32, etc.)
	DType dtype() const { return dtype_; }

	// Get the shape (dimensions) of the full array
	Shape shape() const { return shape_; }

	// Get the sharding (how data is distributed across devices)
	Sharding sharding() const { return sharding_; }
	
	// Destructor
	~XFTArray() = default;

	// Explicitly disable copy (arrays shouldn't be copied)
	XFTArray(const XFTArray&) = delete;
	XFTArray& operator=(const XFTArray&) = delete;

	// Explicitly enable move (arrays can be moved efficiently)
	XFTArray(XFTArray&&) = default;
	XFTArray& operator=(XFTArray&&) = default;

	// ============================================================================
	// NOT YET IMPLEMENTED (will add when needed):
	// ============================================================================
	// - FullyReplicatedShard(): Get one shard from replicated array
	// - has_dynamic_shape() / has_static_shape(): Check shape type
	// - DynamicShape: Get dynamic shapes
	// - DisassembleIntoSingleDeviceArrays(): Break into per-device arrays
	// - CopyToHostBuffer(): Transfer data to host memory
	// - Copy(): Copy array to different devices/memory
	// - GetReadyFuture(): Async readiness check
	// - Delete(): Explicitly free device memory
	// - IsDeleted(): Check if array was deleted
	// - DebugString(): Human-readable representation
	// - GetPjRtBuffer(semantics, index): Get single buffer with copy control

 private:
	// Constructor for static shape
	XFTArray(xla::PjRtClient* client,
			DType dtype,
			Shape shape,
			Sharding sharding,
			PjRtBuffers pjrt_buffers,
			std::shared_ptr<const xla::PjRtLayout> layout);

	// (Optional) Constructor for dynamic shape (add later if needed)
	XFTArray(xla::PjRtClient* client,
			DType dtype,
			DynamicShape dynamic_shape,
			Sharding sharding,
			PjRtBuffers pjrt_buffers,
			std::shared_ptr<const xla::PjRtLayout> layout);

	// Member fields
	xla::PjRtClient* client_;
	DType dtype_;
	Shape shape_; 
	Sharding sharding_;
	PjRtBuffers pjrt_buffers_;
	std::shared_ptr<const xla::PjRtLayout> layout_;
};

}  // namespace xftcpp

#endif  // XFTCPP_ARRAY_H_
