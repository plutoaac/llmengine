#pragma once

#include <expected>
#include <functional>
#include <map>
#include <string>

#include "minillm/core/device.h"
#include "minillm/core/status.h"
#include "minillm/graph/node.h"
#include "minillm/graph/op_type.h"

namespace minillm {

class RuntimeContext;

using KernelFn = std::function<Status(const Node& node, RuntimeContext& ctx)>;

class KernelRegistry {
public:
    void register_kernel(DeviceType device, OpType op, KernelFn fn);
    std::expected<KernelFn, Status> find(DeviceType device, OpType op) const;
    bool has(DeviceType device, OpType op) const;

private:
    using Key = std::pair<DeviceType, OpType>;
    struct KeyComp {
        bool operator()(const Key& a, const Key& b) const {
            if (a.first != b.first) return a.first < b.first;
            return static_cast<int>(a.second) < static_cast<int>(b.second);
        }
    };
    std::map<Key, KernelFn, KeyComp> kernels_;
};

} // namespace minillm
