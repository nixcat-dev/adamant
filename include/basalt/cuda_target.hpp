#pragma once

namespace basalt::cuda_target {

// The compute capabilities the engine is built for (8.9 and 8.6); the only gate for accepting a device.
constexpr bool supports(int device_major, int device_minor) {
    return device_major == 8 && (device_minor == 9 || device_minor == 6);
}

}  // namespace basalt::cuda_target
