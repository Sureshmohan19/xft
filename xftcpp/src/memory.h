// Copyright 2025 XFT Authors.
// Direct PJRT memory implementation without IFRT abstraction layer.

#ifndef XFTCPP_MEMORY_H_
#define XFTCPP_MEMORY_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"

namespace xftcpp {

// Forward declarations
class Client;
class Device;

// Unique identifier for memory spaces.
using MemoryId = int32_t;

// Platform-dependent identifier for memory kinds (e.g., "device", "pinned", "host").
// When unspecified, the platform uses its default memory kind.
//
// MemoryKind instances are lightweight and use string deduplication internally,
// so copies are cheap and pointer-stable comparisons are used for equality.
class MemoryKind {
 public:
  // Creates a MemoryKind with no specific kind (uses platform default).
  MemoryKind() = default;

  // Creates a MemoryKind from a platform-dependent identifier.
  // The string is deduplicated internally and remains stable.
  explicit MemoryKind(std::optional<absl::string_view> memory_kind);

  bool operator==(const MemoryKind& other) const {
    // Use pointer comparison. memory_kind_ always points to a deduplicated string.
    if (!memory_kind_.has_value() && !other.memory_kind_.has_value()) {
      return true;
    }
    if (memory_kind_.has_value() && other.memory_kind_.has_value() &&
        memory_kind_->data() == other.memory_kind_->data()) {
      return true;
    }
    return false;
  }
  bool operator!=(const MemoryKind& other) const { return !(*this == other); }

  template <typename H>
  friend H AbslHashValue(H h, const MemoryKind& memory_kind) {
    return H::combine(std::move(h), memory_kind.memory_kind_);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const MemoryKind& memory_kind) {
    sink.Append(memory_kind.ToString());
  }

  // Returns the platform-dependent identifier, or nullopt if using default.
  std::optional<absl::string_view> memory_kind() const { return memory_kind_; }

 private:
  std::string ToString() const;

  std::optional<absl::string_view> memory_kind_;
};

// Represents a PJRT memory space attached to one or more devices.
// Each memory space has a unique ID, kind, and set of devices that can access it.
class PjRtMemory {
 public:
  // Creates a PjRtMemory wrapping an addressable PJRT memory space.
  PjRtMemory(Client* client, xla::PjRtMemorySpace* pjrt_memory);

  // Creates a PjRtMemory for non-addressable devices without a backing PjRtMemorySpace.
  PjRtMemory(Client* client, const MemoryKind& kind, Device* device);

  // Not copyable or movable.
  PjRtMemory(const PjRtMemory&) = delete;
  PjRtMemory(PjRtMemory&&) = delete;
  PjRtMemory& operator=(const PjRtMemory&) = delete;
  PjRtMemory& operator=(PjRtMemory&&) = delete;

  Client* client() const { return client_; }
  xla::PjRtMemorySpace* pjrt_memory() const { return pjrt_memory_; }

  // Returns unique identifier for this memory space. Returns -1 for unaddressable memory.
  MemoryId Id() const;

  // Returns the platform-dependent memory kind.
  const MemoryKind& Kind() const { return kind_; }

  // Returns a concise user-facing debug string.
  absl::string_view ToString() const;

  // Returns a verbose debug string for error logging.
  absl::string_view DebugString() const;

  // Returns devices attached to this memory space.
  absl::Span<Device* const> Devices() const { return devices_; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const PjRtMemory& memory) {
    sink.Append(memory.DebugString());
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const PjRtMemory* memory) {
    if (memory == nullptr) {
      sink.Append("<nullptr>");
    } else {
      sink.Append(memory->DebugString());
    }
  }

 private:
  Client* client_;
  xla::PjRtMemorySpace* pjrt_memory_;  // nullptr for unaddressable memory
  MemoryKind kind_;
  std::vector<Device*> devices_;
};

// Resolves a MemoryKind to a concrete kind for the given device.
// If memory_kind is already specified, returns it unchanged.
// Otherwise, returns the device's default memory kind, or an empty MemoryKind if unavailable.
MemoryKind CanonicalizeMemoryKind(MemoryKind memory_kind, Device* device);

}  // namespace xftcpp

#endif  // XFTCPP_MEMORY_H_