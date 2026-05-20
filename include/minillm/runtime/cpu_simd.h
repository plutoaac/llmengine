#pragma once

// CPU SIMD (AVX2/AVX-512/SSE2) infrastructure for MiniLLMEngine kernels.
// Provides portable SIMD abstractions that compile to the best available
// instruction set at build time.

#include <cmath>
#include <cstring>

// Detect SIMD level
#if defined(__AVX512F__)
    #include <immintrin.h>
    #define MINILLM_SIMD_AVX512 1
    #define MINILLM_SIMD_AVX2 1
    #define MINILLM_SIMD_WIDTH 16
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define MINILLM_SIMD_AVX2 1
    #define MINILLM_SIMD_WIDTH 8
#elif defined(__SSE2__)
    #include <emmintrin.h>
    #define MINILLM_SIMD_SSE2 1
    #define MINILLM_SIMD_WIDTH 4
#endif

#ifndef MINILLM_SIMD_WIDTH
#define MINILLM_SIMD_WIDTH 1
#endif

// SIMD vector type and portable operations
#if defined(MINILLM_SIMD_AVX512)
using vfloat = __m512;
#define VF_LOAD(p)    _mm512_loadu_ps(p)
#define VF_STORE(p,v) _mm512_storeu_ps(p,v)
#define VF_ADD(a,b)   _mm512_add_ps(a,b)
#define VF_SUB(a,b)   _mm512_sub_ps(a,b)
#define VF_MUL(a,b)   _mm512_mul_ps(a,b)
#define VF_DIV(a,b)   _mm512_div_ps(a,b)
#define VF_MAX(a,b)   _mm512_max_ps(a,b)
#define VF_MIN(a,b)   _mm512_min_ps(a,b)
#define VF_SET1(s)    _mm512_set1_ps(s)
#define VF_SETZERO()  _mm512_setzero_ps()
#define VF_FMADD(a,b,c) _mm512_fmadd_ps(a,b,c)
#elif defined(MINILLM_SIMD_AVX2)
using vfloat = __m256;
#define VF_LOAD(p)    _mm256_loadu_ps(p)
#define VF_STORE(p,v) _mm256_storeu_ps(p,v)
#define VF_ADD(a,b)   _mm256_add_ps(a,b)
#define VF_SUB(a,b)   _mm256_sub_ps(a,b)
#define VF_MUL(a,b)   _mm256_mul_ps(a,b)
#define VF_DIV(a,b)   _mm256_div_ps(a,b)
#define VF_MAX(a,b)   _mm256_max_ps(a,b)
#define VF_MIN(a,b)   _mm256_min_ps(a,b)
#define VF_SET1(s)    _mm256_set1_ps(s)
#define VF_SETZERO()  _mm256_setzero_ps()
#if defined(__FMA__)
    #define VF_FMADD(a,b,c) _mm256_fmadd_ps(a,b,c)
#else
    #define VF_FMADD(a,b,c) _mm256_add_ps(_mm256_mul_ps(a,b),c)
#endif
#elif defined(MINILLM_SIMD_SSE2)
using vfloat = __m128;
#define VF_LOAD(p)    _mm_loadu_ps(p)
#define VF_STORE(p,v) _mm_storeu_ps(p,v)
#define VF_ADD(a,b)   _mm_add_ps(a,b)
#define VF_SUB(a,b)   _mm_sub_ps(a,b)
#define VF_MUL(a,b)   _mm_mul_ps(a,b)
#define VF_DIV(a,b)   _mm_div_ps(a,b)
#define VF_MAX(a,b)   _mm_max_ps(a,b)
#define VF_MIN(a,b)   _mm_min_ps(a,b)
#define VF_SET1(s)    _mm_set1_ps(s)
#define VF_SETZERO()  _mm_setzero_ps()
#define VF_FMADD(a,b,c) _mm_add_ps(_mm_mul_ps(a,b),c)
#else
using vfloat = float;
#define VF_LOAD(p)    (*(p))
#define VF_STORE(p,v) (*(p) = (v))
#define VF_ADD(a,b)   ((a) + (b))
#define VF_SUB(a,b)   ((a) - (b))
#define VF_MUL(a,b)   ((a) * (b))
#define VF_DIV(a,b)   ((a) / (b))
#define VF_MAX(a,b)   ((a) > (b) ? (a) : (b))
#define VF_MIN(a,b)   ((a) < (b) ? (a) : (b))
#define VF_SET1(s)    (s)
#define VF_SETZERO()  (0.0f)
#define VF_FMADD(a,b,c) ((a) * (b) + (c))
#endif

namespace minillm::simd {

// Horizontal sum of vector register to scalar
inline float hsum(vfloat v) {
#if defined(MINILLM_SIMD_AVX512)
    return _mm512_reduce_add_ps(v);
#elif defined(MINILLM_SIMD_AVX2)
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(hi, lo);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
#elif defined(MINILLM_SIMD_SSE2)
    __m128 s = _mm_add_ps(v, _mm_movehl_ps(v, v));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
#else
    return v; // scalar fallback
#endif
}

// Horizontal max of vector register to scalar
inline float hmax(vfloat v) {
#if defined(MINILLM_SIMD_AVX512)
    return _mm512_reduce_max_ps(v);
#elif defined(MINILLM_SIMD_AVX2)
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m = _mm_max_ps(hi, lo);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
#elif defined(MINILLM_SIMD_SSE2)
    __m128 m = _mm_max_ps(v, _mm_movehl_ps(v, v));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
#else
    return v;
#endif
}

// Vectorized exp approximation: store to temp, call scalar expf, reload
inline vfloat v_exp(vfloat x) {
    alignas(64) float tmp[MINILLM_SIMD_WIDTH];
    VF_STORE(tmp, x);
    for (int i = 0; i < MINILLM_SIMD_WIDTH; ++i) tmp[i] = expf(tmp[i]);
    return VF_LOAD(tmp);
}

// Vectorized sigmoid: 1 / (1 + exp(-x))
inline vfloat v_sigmoid(vfloat x) {
    vfloat ones = VF_SET1(1.0f);
    vfloat neg_x = VF_MUL(VF_SET1(-1.0f), x);
    return VF_DIV(ones, VF_ADD(ones, v_exp(neg_x)));
}

} // namespace minillm::simd
