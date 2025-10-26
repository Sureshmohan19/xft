#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <dlfcn.h>  // for dlopen

#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "absl/status/statusor.h"

int main() {
  std::cout << "=== Testing PjRT CPU Buffer Creation (JAX-style plugin load) ===\n";

  // 🔹 Allow dynamic plugin path (environment variable like JAX)
  const char* plugin_path = std::getenv("PJRT_CPU_PLUGIN_PATH");
  if (!plugin_path) {
    plugin_path =
        "/Users/aakritisuresh/Desktop/xla/bazel-out/darwin_arm64-opt/bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so";
  }

  void* handle = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    std::cerr << "Failed to load PJRT CPU plugin: " << dlerror() << "\n";
    return 1;
  }
  std::cout << "✅ Loaded PJRT plugin: " << plugin_path << "\n";

  // Create PJRT client
  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_or =
      xla::GetPjRtCpuClient(/*asynchronous=*/false);
  if (!client_or.ok()) {
    std::cerr << "Failed to create client: " << client_or.status() << "\n";
    return 1;
  }

  auto client = std::move(client_or.value());
  std::cout << "Client created: " << client->platform_name()
            << " | Devices: " << client->device_count() << "\n";

  // Create host buffer
  xla::PjRtDevice* device = client->addressable_devices()[0];
  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {4});
  xla::Literal literal(shape);
  std::memcpy(literal.untyped_data(), host_data.data(),
              host_data.size() * sizeof(float));

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> buffer_or =
      client->BufferFromHostLiteral(literal, device->memory_spaces()[0]);
  if (!buffer_or.ok()) {
    std::cerr << "Failed to create buffer: " << buffer_or.status() << "\n";
    return 1;
  }

  std::unique_ptr<xla::PjRtBuffer> buffer = std::move(buffer_or.value());
  absl::StatusOr<std::shared_ptr<xla::Literal>> result_literal =
      buffer->ToLiteralSync();
  if (!result_literal.ok()) {
    std::cerr << "Failed to copy back: " << result_literal.status() << "\n";
    return 1;
  }

  absl::Span<const float> result_data = (*result_literal)->data<float>();
  bool ok = std::equal(result_data.begin(), result_data.end(), host_data.begin());
  std::cout << (ok ? "✓ SUCCESS\n" : "✗ FAILURE\n");
}
