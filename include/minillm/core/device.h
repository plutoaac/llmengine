#pragma once

#include <string>

namespace minillm {

enum class DeviceType {
    CPU,
    CUDA,
    Unknown,
};

struct Device {
    DeviceType type{DeviceType::CPU};
    int index{0};

    static Device cpu() { return {DeviceType::CPU, 0}; }
    static Device cuda(int idx) { return {DeviceType::CUDA, idx}; }

    std::string to_string() const;
    bool operator==(const Device& other) const = default;
};

} // namespace minillm
