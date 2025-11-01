/* Concrete DeviceList implementation for xftcpp
 * Adapted from XLA's IFRT DeviceList
 */

#include "xftcpp/src/device_list.h"

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

// HighwayHash includes for fingerprint computation
#include "highwayhash/arch_specific.h"
#include "highwayhash/hh_types.h"
#include "highwayhash/highwayhash.h"

namespace xftcpp {

namespace {

// Helper class for computing stable fingerprints across processes
// Uses HighwayHash which is a fast, high-quality hash function
class FingerprintPrinter {
 public:
  FingerprintPrinter() : hash_(kDefaultKey64) {}
  
  // Append data to the hash computation
  void Append(const absl::AlphaNum& a) { 
    hash_.Append(a.data(), a.size()); 
  }
  
  // Finalize and return the 64-bit fingerprint
  // This is a consuming operation (note the &&)
  uint64_t Fingerprint() && {
    highwayhash::HHResult64 result;
    hash_.Finalize(&result);
    return result;
  }

 private:
  // Fixed key for consistent hashing across runs and processes
  // These magic numbers are from the original XLA implementation
  static constexpr highwayhash::HHKey kDefaultKey64 = {
      0x4ea9929a25d561c6,
      0x98470d187b523e8f,
      0x592040a2da3c4b53,
      0xbff8b246e3c587a2,
  };
  
  // HighwayHash state object
  // HH_TARGET is a macro that selects the best SIMD implementation for the CPU
  highwayhash::HighwayHashCatT<HH_TARGET> hash_;
};

}  // namespace

// Constructor: simply stores the device pointers
DeviceList::DeviceList(const std::vector<Device*>& devices) 
    : devices_(devices) {}

// Returns a filtered list containing only addressable devices
// This implements lazy caching: compute once, reuse the result
DeviceList* DeviceList::AddressableDeviceList() const {
  // Check if all devices are addressable
  bool all_addressable = true;
  for (Device* device : devices_) {
    if (!device->IsAddressable()) {
      all_addressable = false;
      break;
    }
  }
  
  // If all devices are addressable, return this same DeviceList
  // This is an optimization to avoid creating a duplicate list
  if (all_addressable) {
    return const_cast<DeviceList*>(this);
  }
  
  // If we already computed the filtered list, return the cached version
  if (addressable_device_list_) {
    return addressable_device_list_.get();
  }
  
  // Filter to only addressable devices
  std::vector<Device*> addressable_devices;
  for (Device* device : devices_) {
    if (device->IsAddressable()) {
      addressable_devices.push_back(device);
    }
  }
  
  // Create and cache the filtered DeviceList
  // We use unique_ptr because we need stable pointer identity
  addressable_device_list_ = std::make_unique<DeviceList>(addressable_devices);
  return addressable_device_list_.get();
}

// Equality check: two DeviceLists are equal if they contain the same devices
// in the same order. We compare device pointers directly.
bool DeviceList::operator==(const DeviceList& other) const {
  if (devices_.size() != other.devices_.size()) {
    return false;
  }
  
  for (size_t i = 0; i < devices_.size(); ++i) {
    if (devices_[i] != other.devices_[i]) {
      return false;
    }
  }
  
  return true;
}

// Compute hash from device IDs
// This is a simple hash that's fast but only stable within the process
// It uses std::hash which may vary between runs
uint64_t DeviceList::hash() const {
  uint64_t h = 0;
  
  // Simple hash combination
  // We hash the device IDs, not the pointers, for better stability
  for (Device* device : devices_) {
    // Mix in each device ID using XOR and bit rotation
    uint64_t device_hash = std::hash<int>{}(device->id());
    h ^= device_hash + 0x9e3779b9 + (h << 6) + (h >> 2);
  }
  
  return h;
}

// Compute fingerprint from device IDs
// This uses HighwayHash for a stable hash across processes
// The fingerprint will be the same even if you restart the program
uint64_t DeviceList::fingerprint() const {
  FingerprintPrinter printer;
  
  // Hash each device ID in order
  for (Device* device : devices_) {
    printer.Append(device->id());
  }
  
  return std::move(printer).Fingerprint();
}

// Convert to human-readable string
// Format: "DeviceList([device0, device1, ...])"
std::string DeviceList::ToString() const {
  if (devices_.empty()) {
    return "DeviceList([])";
  }
  
  // Use absl::StrJoin for clean comma-separated output
  // We create a vector of device strings first
  std::vector<std::string> device_strs;
  device_strs.reserve(devices_.size());
  
  for (Device* device : devices_) {
    // Your Device::ToString() returns string_view, convert to string
    device_strs.push_back(std::string(device->ToString()));
  }
  
  return absl::StrCat("DeviceList([", absl::StrJoin(device_strs, ", "), "])");
}

// Utility function to extract all device IDs from a DeviceList
std::vector<int> GetDeviceIds(const DeviceList& device_list) {
  std::vector<int> ids;
  ids.reserve(device_list.devices().size());
  
  for (const Device* device : device_list.devices()) {
    ids.push_back(device->id());
  }
  
  return ids;
}

}  // namespace xftcpp