#include "minillm/core/status.h"

namespace minillm {

Status::Status(ErrorCode code, std::string message)
    : code_(code), message_(std::move(message)) {}

Status Status::make_ok() { return {}; }
Status Status::invalid_argument(std::string msg) { return {ErrorCode::InvalidArgument, std::move(msg)}; }
Status Status::out_of_range(std::string msg) { return {ErrorCode::OutOfRange, std::move(msg)}; }
Status Status::shape_mismatch(std::string msg) { return {ErrorCode::ShapeMismatch, std::move(msg)}; }
Status Status::unsupported(std::string msg) { return {ErrorCode::Unsupported, std::move(msg)}; }
Status Status::not_implemented(std::string msg) { return {ErrorCode::NotImplemented, std::move(msg)}; }
Status Status::runtime_error(std::string msg) { return {ErrorCode::RuntimeError, std::move(msg)}; }
Status Status::internal_error(std::string msg) { return {ErrorCode::InternalError, std::move(msg)}; }
Status Status::io_error(std::string msg) { return {ErrorCode::IOError, std::move(msg)}; }
Status Status::not_found(std::string msg) { return {ErrorCode::NotFound, std::move(msg)}; }
Status Status::type_error(std::string msg) { return {ErrorCode::TypeError, std::move(msg)}; }

bool Status::ok() const { return code_ == ErrorCode::Ok; }
ErrorCode Status::code() const { return code_; }
std::string_view Status::message() const { return message_; }

std::string Status::to_string() const {
    static constexpr std::string_view names[] = {
        "Ok", "InvalidArgument", "OutOfRange", "ShapeMismatch",
        "Unsupported", "NotImplemented", "RuntimeError", "InternalError",
        "IOError", "NotFound", "TypeError"};
    auto idx = static_cast<size_t>(code_);
    auto name = idx < std::size(names) ? names[idx] : "Unknown";
    if (message_.empty()) return std::string(name);
    return std::string(name) + ": " + message_;
}

} // namespace minillm
