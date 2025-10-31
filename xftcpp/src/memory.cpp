// Copyright 2025 XFT Authors.
// Direct PJRT memory implementation without IFRT abstraction layer.


#include <optional>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_client.h"
#include "xftcpp/src/client.h"
#include "xftcpp/src/device.h"
#include "xftcpp/src/memory.h"

namespace xftcpp {

namespace {

// Global state that keeps stable copies of memory kind strings.
// Uses node_hash_set to ensure pointer stability for string_view comparisons.
struct MemoryKindsSet {
  absl::Mutex mu;
  absl::node_hash_set<std::string> memory_kinds_set ABSL_GUARDED_BY(mu);
};

}  // namespace

// ============================================================================
// MemoryKind Implementation
// ============================================================================

MemoryKind::MemoryKind(std::optional<absl::string_view> memory_kind) {
  static auto* const global_set = new MemoryKindsSet();
  
  if (!memory_kind.has_value()) {
    return;
  }
  
  absl::MutexLock lock(&global_set->mu);
  auto it = global_set->memory_kinds_set.find(*memory_kind);
  if (it == global_set->memory_kinds_set.end()) {
    // Insert new string and use its stable pointer
    memory_kind_ = 
        *global_set->memory_kinds_set.insert(std::string(*memory_kind)).first;
  } else {
    // Reuse existing deduplicated string
    memory_kind_ = *it;
  }
}

std::string MemoryKind::ToString() const {
  if (memory_kind_.has_value()) {
    return std::string(*memory_kind_);
  }
  return "(default)";
}

// ============================================================================
// PjRtMemory Implementation
// ============================================================================

PjRtMemory::PjRtMemory(Client* client, xla::PjRtMemorySpace* pjrt_memory)
    : client_(client), pjrt_memory_(pjrt_memory), kind_(pjrt_memory->kind()) {
  // Map PJRT devices to our Device wrappers
  for (xla::PjRtDevice* pjrt_device : pjrt_memory->devices()) {
    devices_.push_back(client->LookupDevice(pjrt_device));
  }
}

PjRtMemory::PjRtMemory(Client* client, const MemoryKind& kind, Device* device)
    : client_(client), pjrt_memory_(nullptr), kind_(kind) {
  devices_.push_back(device);
}

MemoryId PjRtMemory::Id() const {
  if (pjrt_memory_ == nullptr) {
    return -1;
  }
  return static_cast<MemoryId>(pjrt_memory_->id());
}

absl::string_view PjRtMemory::ToString() const {
  if (pjrt_memory_ == nullptr) {
    return "UNADDRESSABLE_MEMORY_SPACE";
  }
  return pjrt_memory_->ToString();
}

absl::string_view PjRtMemory::DebugString() const {
  if (pjrt_memory_ == nullptr) {
    return "Unaddressable PjRtMemory";
  }
  return pjrt_memory_->DebugString();
}

// ============================================================================
// MemoryKind Canonicalization
// ============================================================================

MemoryKind CanonicalizeMemoryKind(MemoryKind memory_kind, Device* device) {
  // If memory kind is already specified, use it as-is
  if (memory_kind.memory_kind().has_value()) {
    return memory_kind;
  }
  
  // Try to get device's default memory kind
  auto default_memory = device->DefaultMemory();
  if (default_memory.ok()) {
    return (*default_memory)->Kind();
  }
  
  // Fall back to unspecified memory kind
  return MemoryKind();
}

}  // namespace xftcpp