#pragma once

#include <bit>
#include <cstdint>
#include <cmath>
#include <limits>

namespace minillm {

class bfloat16_t {
private:
    uint16_t bits_;

    struct from_bits_t {};
    constexpr bfloat16_t(uint16_t b, from_bits_t) : bits_(b) {}

    // IEEE 754 round-to-nearest-even for float → BF16
    static constexpr uint16_t float_to_bf16_bits(float f) {
        uint32_t bits = std::bit_cast<uint32_t>(f);
        uint32_t sign = bits >> 31;
        uint32_t exp  = (bits >> 23) & 0xFF;
        uint32_t mant = bits & 0x7FFFFF;

        // Inf / NaN: preserve sign, set BF16 Inf mantissa
        if (exp == 0xFF) {
            return static_cast<uint16_t>((sign << 15) | 0x7F80 | (mant ? 1 : 0));
        }
        // Zero / subnormal
        if (exp == 0) {
            return static_cast<uint16_t>(sign << 15);
        }
        // Normal: RNE on the dropped 16 bits
        uint32_t dropped = bits & 0xFFFF;
        uint32_t round_up = (dropped > 0x8000) || (dropped == 0x8000 && (mant & 0x10000));
        uint32_t bf16_mant = (mant >> 16) + round_up;
        uint32_t bf16_exp  = exp;
        if (bf16_mant >= 0x80) {
            bf16_exp++;
            bf16_mant = 0;
            if (bf16_exp >= 0xFF) {
                return static_cast<uint16_t>((sign << 15) | 0x7F80);
            }
        }
        return static_cast<uint16_t>((sign << 15) | (bf16_exp << 7) | bf16_mant);
    }

    static constexpr float bf16_bits_to_float(uint16_t bits) {
        return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
    }

public:
    constexpr bfloat16_t() : bits_(0) {}
    constexpr bfloat16_t(float f) : bits_(float_to_bf16_bits(f)) {}
    constexpr bfloat16_t(double d) : bfloat16_t(static_cast<float>(d)) {}
    constexpr bfloat16_t(int v) : bfloat16_t(static_cast<float>(v)) {}

    explicit operator float() const { return bf16_bits_to_float(bits_); }
    explicit operator double() const { return static_cast<double>(static_cast<float>(*this)); }

    constexpr uint16_t bits() const { return bits_; }

    static constexpr bfloat16_t from_bits(uint16_t b) { return {b, from_bits_t{}}; }

    bool operator==(const bfloat16_t& rhs) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(rhs);
        if (std::isnan(a) || std::isnan(b)) return false;
        return a == b;
    }
    bool operator!=(const bfloat16_t& rhs) const { return !(*this == rhs); }
    bool operator<(const bfloat16_t& rhs) const  { return static_cast<float>(*this) < static_cast<float>(rhs); }
    bool operator<=(const bfloat16_t& rhs) const { return !(rhs < *this); }
    bool operator>(const bfloat16_t& rhs) const  { return rhs < *this; }
    bool operator>=(const bfloat16_t& rhs) const { return !(*this < rhs); }

    bfloat16_t operator-() const { return from_bits(bits_ ^ 0x8000); }

    bfloat16_t& operator+=(const bfloat16_t& rhs) { *this = bfloat16_t(static_cast<float>(*this) + static_cast<float>(rhs)); return *this; }
    bfloat16_t& operator-=(const bfloat16_t& rhs) { *this = bfloat16_t(static_cast<float>(*this) - static_cast<float>(rhs)); return *this; }
    bfloat16_t& operator*=(const bfloat16_t& rhs) { *this = bfloat16_t(static_cast<float>(*this) * static_cast<float>(rhs)); return *this; }
    bfloat16_t& operator/=(const bfloat16_t& rhs) { *this = bfloat16_t(static_cast<float>(*this) / static_cast<float>(rhs)); return *this; }

    static constexpr bfloat16_t zero()     { return from_bits(0x0000); }
    static constexpr bfloat16_t neg_zero() { return from_bits(0x8000); }
    static constexpr bfloat16_t infinity() { return from_bits(0x7F80); }
    static constexpr bfloat16_t nan()      { return from_bits(0x7FC0); }
};

static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 2 bytes");
static_assert(std::is_trivially_copyable_v<bfloat16_t>, "bfloat16_t must be trivially copyable");

inline bfloat16_t operator+(bfloat16_t a, bfloat16_t b) { a += b; return a; }
inline bfloat16_t operator-(bfloat16_t a, bfloat16_t b) { a -= b; return a; }
inline bfloat16_t operator*(bfloat16_t a, bfloat16_t b) { a *= b; return a; }
inline bfloat16_t operator/(bfloat16_t a, bfloat16_t b) { a /= b; return a; }

} // namespace minillm

namespace std {

template<>
class numeric_limits<minillm::bfloat16_t> {
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr bool has_denorm = denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr float_round_style round_style = round_to_nearest;
    static constexpr int digits = 8;
    static constexpr int digits10 = 2;
    static constexpr int max_digits10 = 4;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -125;
    static constexpr int min_exponent10 = -37;
    static constexpr int max_exponent = 128;
    static constexpr int max_exponent10 = 38;

    static constexpr minillm::bfloat16_t min()          { return minillm::bfloat16_t::from_bits(0x0080); }
    static constexpr minillm::bfloat16_t lowest()       { return minillm::bfloat16_t::from_bits(0xFF7F); }
    static constexpr minillm::bfloat16_t max()          { return minillm::bfloat16_t::from_bits(0x7F7F); }
    static constexpr minillm::bfloat16_t epsilon()      { return minillm::bfloat16_t::from_bits(0x3C00); }
    static constexpr minillm::bfloat16_t round_error()  { return minillm::bfloat16_t(0.5f); }
    static constexpr minillm::bfloat16_t infinity()     { return minillm::bfloat16_t::infinity(); }
    static constexpr minillm::bfloat16_t quiet_NaN()    { return minillm::bfloat16_t::nan(); }
};

} // namespace std
