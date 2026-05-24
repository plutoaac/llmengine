#pragma once

#include <string>
#include <string_view>
#include <utility>

#define TRY(expr) \
    do { auto _st_ = (expr); if (!_st_.ok()) return _st_; } while(0)

#define CONCAT_(a, b) a ## b
#define CONCAT(a, b) CONCAT_(a, b)

#define TRY_TENSOR(var, expr) \
    auto CONCAT(_t_, __LINE__) = (expr); \
    if (!CONCAT(_t_, __LINE__)) return CONCAT(_t_, __LINE__).error(); \
    auto var = std::move(CONCAT(_t_, __LINE__)).value()

namespace minillm {

enum class ErrorCode {
    Ok,
    InvalidArgument,
    OutOfRange,
    ShapeMismatch,
    Unsupported,
    NotImplemented,
    RuntimeError,
    InternalError,
    IOError,
    NotFound,
    TypeError,
};

class Status {
public:
    Status() = default;
    Status(ErrorCode code, std::string message);

    static Status make_ok();
    static Status invalid_argument(std::string message);
    static Status out_of_range(std::string message);
    static Status shape_mismatch(std::string message);
    static Status unsupported(std::string message);
    static Status not_implemented(std::string message);
    static Status runtime_error(std::string message);
    static Status internal_error(std::string message);
    static Status io_error(std::string message);
    static Status not_found(std::string message);
    static Status type_error(std::string message);

    bool ok() const;
    ErrorCode code() const;
    std::string_view message() const;
    std::string to_string() const;

private:
    ErrorCode code_{ErrorCode::Ok};
    std::string message_;
};

} // namespace minillm
