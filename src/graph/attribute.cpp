#include "minillm/graph/attribute.h"

namespace minillm {

std::string attribute_to_string(const AttributeValue& value) {
    return std::visit([](auto&& arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int64_t>) {
            return std::to_string(arg);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::to_string(arg);
        } else if constexpr (std::is_same_v<T, bool>) {
            return arg ? "true" : "false";
        } else if constexpr (std::is_same_v<T, std::string>) {
            return arg;
        } else if constexpr (std::is_same_v<T, Shape>) {
            return arg.to_string();
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            std::string s = "[";
            for (size_t i = 0; i < arg.size(); ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(arg[i]);
            }
            s += "]";
            return s;
        }
    }, value);
}

} // namespace minillm
