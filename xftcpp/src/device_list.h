/* Concrete DeviceList implementation for xftcpp
 * Adapted from XLA's IFRT DeviceList
 * This is a simplified version with all abstractions removed
 */

#ifndef XFTCPP_DEVICE_LIST_H_
#define XFTCPP_DEVICE_LIST_H_

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

#include "xftcpp.src/device.h"

namespace xftcpp {

// Forward declaration for Client (you may need to adjust based on your actual Client class)
class Client;

// Concrete implementation of an ordered immutable list of devices.
// This replaces XLA's abstract DeviceList base class with a concrete version.
// Reference counting is simplified - using shared_ptr instead of custom ReferenceCounted.
class DeviceList {
 public:
  // Constructor takes a vector of Device pointers
  // Devices are stored as raw pointers (owned by Client, not by DeviceList)
  explicit DeviceList(const std::vector<Device*>& devices);
  
  // Not copyable or movable - DeviceList represents a specific runtime configuration
  DeviceList(const DeviceList&) = delete;
  DeviceList(DeviceList&&) = delete;
  DeviceList& operator=(const DeviceList&) = delete;
  DeviceList& operator=(DeviceList&&) = delete;

  ~DeviceList() = default;

  // Returns the number of devices in this list
  int size() const { return devices_.size(); }

  // Returns true if the device list contains no devices
  bool empty() const { return devices_.empty(); }

  // Returns the internal vector of device pointers
  // Devices are not owned by DeviceList - they're owned by the Client
  const std::vector<Device*>& devices() const { return devices_; }

  // Returns a DeviceList containing only addressable devices from this list.
  // If all devices are addressable, returns this same DeviceList.
  // Otherwise, creates and caches a filtered DeviceList with only addressable devices.
  // The returned pointer is valid as long as this DeviceList exists.
  DeviceList* AddressableDeviceList() const;

  // Returns true if all devices in this list are addressable
  bool IsFullyAddressable() const { return AddressableDeviceList() == this; }

  // Equality comparison - checks if both lists contain the same devices in the same order
  bool operator==(const DeviceList& other) const;
  bool operator!=(const DeviceList& other) const { return !(*this == other); }

  // Abseil integration: allows DeviceList to work with absl::StrCat, absl::StrFormat, etc.
  // Usage: absl::StrCat("Devices: ", device_list) will automatically call ToString()
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const DeviceList& device_list) {
    sink.Append(device_list.ToString());
  }

  // Abseil integration: allows DeviceList to be used as a key in absl hash containers
  // (absl::flat_hash_set, absl::flat_hash_map) by providing hash computation
  template <typename H>
  friend H AbslHashValue(H h, const DeviceList& device_list) {
    return H::combine(std::move(h), device_list.hash());
  }

  // Returns a hash value computed from device IDs
  // This hash is only stable within the current process (not across processes)
  uint64_t hash() const;

  // Returns a fingerprint computed from device IDs
  // This fingerprint is stable both within and across processes
  // Uses HighwayHash for consistent hashing
  uint64_t fingerprint() const;

  // Returns a human-readable string representation of this device list
  std::string ToString() const;
  std::string DebugString() const { return ToString(); }

 private:
  // The actual list of device pointers
  // These are non-owning pointers - devices are owned by the Client
  std::vector<Device*> devices_;

  // Cached filtered list containing only addressable devices
  // This is computed lazily on first call to AddressableDeviceList()
  // mutable because it's a cache that can be set in const methods
  // unique_ptr because we need stable pointer identity for the "returns this" optimization
  mutable std::unique_ptr<DeviceList> addressable_device_list_;
};

// Convenience function to extract device IDs from a DeviceList
// Returns a vector containing the ID of each device in the list
std::vector<DeviceId> GetDeviceIds(const DeviceList& device_list);

// Type alias for shared_ptr to DeviceList (replaces XLA's RCReferenceWrapper)
using DeviceListRef = std::shared_ptr<DeviceList>;

}  // namespace xftcpp

#endif  // XFTCPP_DEVICE_LIST_H_