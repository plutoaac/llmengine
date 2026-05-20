#include "minillm/runtime/memory_planner.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

#include "minillm/core/shape.h"
#include "minillm/graph/graph.h"

namespace minillm {

namespace {

constexpr size_t kNoBuffer = static_cast<size_t>(-1);

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0) return value;
    const size_t rem = value % alignment;
    return rem == 0 ? value : value + (alignment - rem);
}

std::string format_bytes(size_t bytes) {
    constexpr double KiB = 1024.0;
    constexpr double MiB = 1024.0 * KiB;
    constexpr double GiB = 1024.0 * MiB;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    if (bytes >= static_cast<size_t>(GiB)) {
        oss << (static_cast<double>(bytes) / GiB) << " GiB";
    } else if (bytes >= static_cast<size_t>(MiB)) {
        oss << (static_cast<double>(bytes) / MiB) << " MiB";
    } else if (bytes >= static_cast<size_t>(KiB)) {
        oss << (static_cast<double>(bytes) / KiB) << " KiB";
    } else {
        oss.str("");
        oss.clear();
        oss << bytes << " B";
    }
    return oss.str();
}

std::string kind_name(ValueKind kind) {
    switch (kind) {
    case ValueKind::Input:        return "Input";
    case ValueKind::Constant:     return "Constant";
    case ValueKind::Intermediate: return "Intermediate";
    case ValueKind::Output:       return "Output";
    default:                      return "Unknown";
    }
}

bool same_pool(const MemoryBuffer& buffer, const MemoryLiveRange& range) {
    return buffer.dtype == range.dtype && buffer.device == range.device;
}

} // namespace

double MemoryPlan::savings_ratio() const {
    if (naive_bytes == 0) return 0.0;
    return 1.0 - static_cast<double>(planned_bytes) /
                     static_cast<double>(naive_bytes);
}

const MemoryLiveRange* MemoryPlan::range_for(ValueId id) const {
    for (const auto& range : ranges) {
        if (range.value == id) return &range;
    }
    return nullptr;
}

std::string MemoryPlan::report() const {
    std::ostringstream oss;
    oss << "Graph memory plan:\n";
    oss << "  values: " << ranges.size() << "\n";
    oss << "  eligible intermediates: " << eligible_value_count << "\n";
    oss << "  skipped values: " << skipped_value_count << "\n";
    oss << "  buffers: " << buffers.size() << "\n";
    oss << "  naive bytes: " << naive_bytes << " (" << format_bytes(naive_bytes) << ")\n";
    oss << "  planned peak: " << planned_bytes << " (" << format_bytes(planned_bytes) << ")\n";
    oss << "  reuse saving: " << std::fixed << std::setprecision(1)
        << (savings_ratio() * 100.0) << "%\n";

    if (!buffers.empty()) {
        oss << "  buffer assignments:\n";
        for (const auto& buffer : buffers) {
            oss << "    #" << buffer.id
                << " bytes=" << buffer.bytes
                << " dtype=" << dtype_name(buffer.dtype)
                << " device=" << buffer.device.to_string()
                << " values=[";
            for (size_t i = 0; i < buffer.values.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << "%" << buffer.values[i].value;
            }
            oss << "]\n";
        }
    }
    return oss.str();
}

std::expected<MemoryPlan, Status> MemoryPlanner::plan(
    const Graph& graph,
    MemoryPlanOptions options) {
    if (options.alignment == 0) {
        return std::unexpected(Status::invalid_argument(
            "memory planner alignment must be greater than zero"));
    }

    auto st = graph.validate();
    if (!st.ok()) return std::unexpected(st);

    auto order = graph.topological_sort();
    if (!order) return std::unexpected(order.error());

    const auto& values = graph.values();
    const size_t value_count = values.size();
    const size_t node_count = order->size();
    const size_t graph_end = node_count == 0 ? 0 : node_count - 1;

    std::vector<size_t> first_node(value_count, 0);
    std::vector<size_t> last_node(value_count, 0);
    std::vector<bool> has_producer(value_count, false);

    for (size_t order_idx = 0; order_idx < node_count; ++order_idx) {
        auto nd = graph.node((*order)[order_idx]);
        if (!nd) return std::unexpected(nd.error());

        for (const auto& input : (*nd)->inputs()) {
            if (input.value < value_count) {
                last_node[input.value] = std::max(last_node[input.value], order_idx);
            }
        }
        for (const auto& output : (*nd)->outputs()) {
            if (output.value < value_count) {
                first_node[output.value] = order_idx;
                last_node[output.value] = std::max(last_node[output.value], order_idx);
                has_producer[output.value] = true;
            }
        }
    }

    MemoryPlan plan;
    plan.ranges.reserve(value_count);

    for (const auto& value : values) {
        MemoryLiveRange range;
        range.value = value.id;
        range.name = value.name;
        range.dtype = value.dtype;
        range.device = value.device;
        range.kind = value.kind;
        range.buffer_id = kNoBuffer;

        if (value.id.value < first_node.size()) {
            range.first_node = has_producer[value.id.value] ? first_node[value.id.value] : 0;
            range.last_node = last_node[value.id.value];
        }

        if (value.kind == ValueKind::Output) {
            range.last_node = graph_end;
        }
        if (range.last_node < range.first_node) {
            range.last_node = range.first_node;
        }

        const bool kind_ok =
            value.kind == ValueKind::Intermediate ||
            (options.include_outputs && value.kind == ValueKind::Output);
        if (!kind_ok) {
            range.skip_reason = "kind=" + kind_name(value.kind);
            ++plan.skipped_value_count;
            plan.ranges.push_back(std::move(range));
            continue;
        }
        if (value.device.type != DeviceType::CPU) {
            range.skip_reason = "non-CPU device";
            ++plan.skipped_value_count;
            plan.ranges.push_back(std::move(range));
            continue;
        }
        if (value.shape.has_dynamic_dim()) {
            range.skip_reason = "dynamic shape";
            ++plan.skipped_value_count;
            plan.ranges.push_back(std::move(range));
            continue;
        }

        auto elem_count = value.shape.numel();
        if (!elem_count) {
            range.skip_reason = elem_count.error().to_string();
            ++plan.skipped_value_count;
            plan.ranges.push_back(std::move(range));
            continue;
        }
        auto elem_size = dtype_size(value.dtype);
        if (!elem_size) {
            range.skip_reason = elem_size.error().to_string();
            ++plan.skipped_value_count;
            plan.ranges.push_back(std::move(range));
            continue;
        }
        if (*elem_count > std::numeric_limits<size_t>::max() / *elem_size) {
            return std::unexpected(Status::out_of_range(
                "memory planner tensor byte size overflow for " + value.name));
        }

        range.bytes = *elem_count * *elem_size;
        range.aligned_bytes = align_up(range.bytes, options.alignment);
        range.eligible = true;
        ++plan.eligible_value_count;
        plan.naive_bytes += range.aligned_bytes;
        plan.ranges.push_back(std::move(range));
    }

    std::vector<size_t> eligible_indices;
    eligible_indices.reserve(plan.ranges.size());
    for (size_t i = 0; i < plan.ranges.size(); ++i) {
        if (plan.ranges[i].eligible) {
            eligible_indices.push_back(i);
        }
    }

    std::sort(eligible_indices.begin(), eligible_indices.end(),
              [&](size_t lhs, size_t rhs) {
                  const auto& a = plan.ranges[lhs];
                  const auto& b = plan.ranges[rhs];
                  if (a.first_node != b.first_node) return a.first_node < b.first_node;
                  if (a.last_node != b.last_node) return a.last_node < b.last_node;
                  return a.aligned_bytes > b.aligned_bytes;
              });

    for (size_t range_idx : eligible_indices) {
        auto& range = plan.ranges[range_idx];
        size_t best = kNoBuffer;
        size_t best_waste = std::numeric_limits<size_t>::max();

        for (size_t i = 0; i < plan.buffers.size(); ++i) {
            auto& buffer = plan.buffers[i];
            if (!same_pool(buffer, range)) continue;
            if (buffer.last_use >= range.first_node) continue;
            if (buffer.bytes < range.aligned_bytes) continue;

            const size_t waste = buffer.bytes - range.aligned_bytes;
            if (waste < best_waste) {
                best = i;
                best_waste = waste;
            }
        }

        if (best == kNoBuffer) {
            MemoryBuffer buffer;
            buffer.id = plan.buffers.size();
            buffer.bytes = range.aligned_bytes;
            buffer.dtype = range.dtype;
            buffer.device = range.device;
            buffer.last_use = range.last_node;
            buffer.values.push_back(range.value);
            range.buffer_id = buffer.id;
            plan.buffers.push_back(std::move(buffer));
        } else {
            auto& buffer = plan.buffers[best];
            buffer.last_use = range.last_node;
            buffer.values.push_back(range.value);
            range.buffer_id = buffer.id;
        }
    }

    for (const auto& buffer : plan.buffers) {
        plan.planned_bytes += buffer.bytes;
    }

    return plan;
}

} // namespace minillm
