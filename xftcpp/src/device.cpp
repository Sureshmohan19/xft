/* XFT-CPP Device - Implementation
 *
 * Simple implementation - mostly just constructor and DefaultMemory() getter.
 * Memory spaces are populated by Client after construction via friend access.
 */

#include "xftcpp/src/device.h"

#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"

namespace xftcpp {

// ============================================================================
// Constructor
// ============================================================================

Device::Device(Client* client,
               int id,
               std::string kind,
               std::string to_string,
               std::string debug_string,
               int process_index,
               xla::PjRtDevice* pjrt_device)
    : client_(client),
      id_(id),
      kind_(std::move(kind)),
      to_string_(std::move(to_string)),
      debug_string_(std::move(debug_string)),
      process_index_(process_index),
      pjrt_device_(pjrt_device) {
  // Note: default_memory_ and memories_ are left uninitialized.
  // They will be set by Client after all devices and memories are created,
  // since there's a circular dependency between Device and Memory.
}

}  // namespace xftcpp