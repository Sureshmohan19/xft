/* XFT-CPP Device
 *
 * Wraps xla::PjRtDevice to provide device information and capabilities.
 * This is a simplified version of xla/python/pjrt_ifrt/pjrt_device.h
 * without the IFRT abstraction layer.
 *
 * A Device represents a single accelerator (GPU, TPU, CPU) that can
 * execute computations. Devices are managed by a Client and have associated
 * memory spaces.
 */

#ifndef XFTCPP_DEVICE_H_
#define XFTCPP_DEVICE_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"

namespace xftcpp {

// Forward declarations
class Client;
class Memory;

// Represents a single device (GPU, TPU, CPU core) that can run computations.
// Wraps xla::PjRtDevice and caches commonly-accessed properties for performance.
//
// Devices are created and owned by Client. Users should not construct Device
// objects directly.
class Device {
 public:
  // Parameters:
  // - client: The Client that owns this device
  // - id: Globally unique device ID across all processes
  // - kind: Device type string (e.g., "GPU", "TPU", "CPU")
  // - to_string: Short human-readable description
  // - debug_string: Verbose debug description
  // - process_index: Which process this device belongs to
  // - pjrt_device: The underlying PJRT device (nullptr for non-addressable)
  Device(Client* client,
         int id,
         std::string kind,
         std::string to_string,
         std::string debug_string,
         int process_index,
         xla::PjRtDevice* pjrt_device);

  // Devices are not copyable or movable (managed by Client)
  Device(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(const Device&) = delete;
  Device& operator=(Device&&) = delete;

  // ============================================================================
  // Basic Device Information
  // ============================================================================

  // Get the Client that owns this device
  Client* client() const { return client_; }

  // Get the globally unique device ID (across all processes)
  int id() const { return id_; }

  // Get device kind string: "GPU", "TPU", "CPU", etc.
  // This identifies the type of accelerator.
  absl::string_view Kind() const { return kind_; }

  // Short human-readable string: "GPU:0", "TPU:1", etc.
  absl::string_view ToString() const { return to_string_; }

  // Verbose debug string with all device details
  absl::string_view DebugString() const { return debug_string_; }

  // ============================================================================
  // Device Addressability
  // ============================================================================

  // Whether this process can issue commands to this device.
  // In multi-process settings, a device may be visible but not addressable.
  // Only addressable devices have a non-null pjrt_device().
  bool IsAddressable() const { return pjrt_device_ != nullptr; }

  // The index of the process that can address this device.
  // In single-process: always 0
  // In multi-process: identifies which process owns this device
  int ProcessIndex() const { return process_index_; }

  // ============================================================================
  // Memory Spaces
  // ============================================================================

  // Get the default memory space for this device.
  // Most operations use default memory unless otherwise specified.
  // Returns error if device has no memory spaces.
  absl::StatusOr<Memory*> DefaultMemory() const { return default_memory_; }
  
  // Get all memory spaces accessible from this device.
  // Example: GPU might have device memory, pinned host memory, etc.
  // The order is unspecified.
  absl::Span<Memory* const> Memories() const { return memories_; }

  // ============================================================================
  // PJRT Interop
  // ============================================================================

  // Get the underlying xla::PjRtDevice.
  // Returns nullptr for non-addressable devices.
  // Use this when you need to call PJRT APIs directly.
  xla::PjRtDevice* pjrt_device() const { return pjrt_device_; }

  // ============================================================================
  // String Formatting
  // ============================================================================

  // Support for absl::StrCat and absl::StrFormat
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Device& device) {
    sink.Append(device.ToString());
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Device* device) {
    if (device == nullptr) {
      sink.Append("<nullptr>");
    } else {
      sink.Append(device->ToString());
    }
  }

 private:
  friend class Client;  // Client sets up memory spaces after construction

  // ============================================================================
  // Member Variables (cached for performance)
  // ============================================================================

  Client* client_;                            // Owning client
  int id_;                                    // Globally unique ID
  std::string kind_;                          // "GPU", "TPU", "CPU"
  std::string to_string_;                     // Short description
  std::string debug_string_;                  // Verbose description
  int process_index_;                         // Which process owns this device
  xla::PjRtDevice* pjrt_device_;              // Underlying PJRT device (may be nullptr)

  // Memory spaces (set by Client after construction)
  absl::StatusOr<Memory*> default_memory_;    // Default memory space
  std::vector<Memory*> memories_;             // All memory spaces
};

}  // namespace xftcpp

#endif  // XFTCPP_DEVICE_H_