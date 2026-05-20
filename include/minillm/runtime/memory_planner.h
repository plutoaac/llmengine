#pragma once

#include <cstddef>
#include <expected>
#include <string>
#include <vector>

#include "minillm/core/device.h"
#include "minillm/core/dtype.h"
#include "minillm/core/status.h"
#include "minillm/graph/value.h"

namespace minillm {

class Graph;

struct MemoryPlanOptions {
    size_t alignment = 64;
    bool include_outputs = false;
};

struct MemoryLiveRange {
    ValueId value;
    std::string name;
    size_t bytes = 0;
    size_t aligned_bytes = 0;
    DType dtype = DType::Unknown;
    Device device = Device::cpu();
    ValueKind kind = ValueKind::Intermediate;
    size_t first_node = 0;
    size_t last_node = 0;
    bool eligible = false;
    std::string skip_reason;
    size_t buffer_id = static_cast<size_t>(-1);
};

struct MemoryBuffer {
    size_t id = 0;
    size_t bytes = 0;
    DType dtype = DType::Unknown;
    Device device = Device::cpu();
    size_t last_use = 0;
    std::vector<ValueId> values;
};

struct MemoryPlan {
    std::vector<MemoryLiveRange> ranges;
    std::vector<MemoryBuffer> buffers;
    size_t eligible_value_count = 0;
    size_t skipped_value_count = 0;
    size_t naive_bytes = 0;
    size_t planned_bytes = 0;

    double savings_ratio() const;
    const MemoryLiveRange* range_for(ValueId id) const;
    std::string report() const;
};

class MemoryPlanner {
public:
    static std::expected<MemoryPlan, Status> plan(
        const Graph& graph,
        MemoryPlanOptions options = {});
};

} // namespace minillm
