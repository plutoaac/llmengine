#include "minillm/core/device.h"

namespace minillm {

std::string Device::to_string() const {
    switch (type) {
    case DeviceType::CPU:    return "cpu:" + std::to_string(index);
    case DeviceType::CUDA:   return "cuda:" + std::to_string(index);
    default:                 return "unknown:" + std::to_string(index);
    }
}

} // namespace minillm
