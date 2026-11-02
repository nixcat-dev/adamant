#pragma once

namespace basalt::cuda_target {

inline constexpr int major = 8;
inline constexpr int minor = 9;

constexpr bool supports(int device_major, int device_minor) {
    return device_major == major && device_minor == minor;
}

}  // namespace basalt::cuda_target
