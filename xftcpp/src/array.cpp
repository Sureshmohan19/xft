/* XFT-CPP Array - Implementation
 *
 * Implementation of the core Array class that wraps PJRT buffers.
 * This connects PyArray (Python) to xla::PjRtBuffer (XLA implementation).
 *
 * Key Points:
 * ===========
 * - Holds std::shared_ptr<PjRtBuffer> for each shard
 * - Immutable after construction (thread-safe)
 * - All validation happens at construction time
 * - We don't modify PJRT implementation, only use its API
 */

#include "xftcpp/src/array.h"

namespace xftcpp {
namespace xftarray {
		
	// ============================================================================
	// Constructor (Static Shape)
	// ============================================================================
	// Creates an XFTArray with a statically known shape. This is the most common
	// constructor path used when the array dimensions are fixed and known at
	// creation time.
	//
	// Parameters:
	// - client: The PJRT client managing devices.
	// - dtype: Data type of the array elements (e.g., float32, int32).
	// - shape: Static shape of the array.
	// - sharding: Describes how the array is partitioned across devices.
	// - pjrt_buffers: One or more PJRT buffers representing device memory.
	// - layout: Optional layout information (nullptr = default layout).
	XFTArray::XFTArray(xla::PjRtClient* client,
						DType dtype,
						Shape shape,
						Sharding sharding,
						PjRtBuffers pjrt_buffers,
						std::shared_ptr<const xla::PjRtLayout> layout)
		: 	client_(client),
			dtype_(std::move(dtype)),
			shape_(std::move(shape)),
			sharding_(std::move(sharding)),
			pjrt_buffers_(std::move(pjrt_buffers)),
			layout_(std::move(layout)) {}

	// ============================================================================
	// Constructor (Dynamic Shape)
	// ============================================================================
	// Creates an XFTArray with a dynamic shape. This constructor is intended for
	// use cases where one or more dimensions of the array can vary at runtime,
	// such as variable-length sequences in RNNs or batch-dependent models.
	//
	// Parameters:
	// - client: The PJRT client managing devices.
	// - dtype: Data type of the array elements.
	// - dynamic_shape: Shape object that supports dynamic dimensions.
	// - sharding: Describes how the array is partitioned across devices.
	// - pjrt_buffers: One or more PJRT buffers representing device memory.
	// - layout: Optional layout information (nullptr = default layout).
	XFTArray::XFTArray(xla::PjRtClient* client,
						DType dtype,
						DynamicShape dynamic_shape,
						Sharding sharding,
						PjRtBuffers pjrt_buffers,
						std::shared_ptr<const xla::PjRtLayout> layout)
		: 	client_(client),
			dtype_(std::move(dtype)),
			shape_(std::move(dynamic_shape)), 
			sharding_(std::move(sharding)),
			pjrt_buffers_(std::move(pjrt_buffers)),
			layout_(std::move(layout)) {}


}  // namespace xftarray
}  // namespace xftcpp