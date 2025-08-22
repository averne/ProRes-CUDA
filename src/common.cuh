/*
 * Copyright (c) 2025 averne <averne381@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License,
 * or (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include <cstdint>
#include <concepts>

#include <cuda.h>

#include "util.hpp"

#define U8(x)  uint8_t (x)
#define U16(x) uint16_t(x)

__device__ __forceinline__
static uint findLSB(uint v) {
    return __ffs(v) - 1;
}

__device__ __forceinline__
static uint findMSB(uint v) {
    return 31 - __clz(v);
}

template <typename T> __device__ __forceinline__
static T clamp(T v, auto lo, auto hi) {
    return min(max(v, lo), hi);
}

template <std::unsigned_integral T> __device__ __forceinline__
static int bitCount(T v) {
    if constexpr (std::is_same_v<T, std::uint64_t>)
        return __popcll(v);
    else
        return __popc(v);
}

// CUDA doesn't have a bitfield extract intrinsic for some reason
template <std::unsigned_integral T> __device__ __forceinline__
static T bitfieldExtract(T v, int off, int bits) {
    if constexpr (std::is_same_v<T, std::uint64_t>) {
        std::uint64_t tmp = v;
        asm volatile("bfe.u64 %0, %1, %2, %3;" : "=l"(tmp) : "l"(tmp), "r"(off), "r"(bits));
        return static_cast<T>(tmp);
    } else {
        std::uint32_t tmp = v;
        asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(tmp) : "r"(tmp), "r"(off), "r"(bits));
        return static_cast<T>(tmp);
    }
}

template <typename T> __device__ __forceinline__
static T sign_extend(T v, int bits) {
    if constexpr (sizeof(T) == sizeof(std::uint64_t)) {
        std::int64_t tmp = v;
        asm volatile("bfe.s64 %0, %1, 0, %2;" : "=l"(tmp) : "l"(tmp), "r"(bits));
        return static_cast<T>(tmp);
    } else {
        std::int32_t tmp = v;
        asm volatile("bfe.s32 %0, %1, 0, %2;" : "=r"(tmp) : "r"(tmp), "r"(bits));
        return static_cast<T>(tmp);
    }
}

#define DEFINE_VEC1_UNARY_OP_RET(type, op)                                          \
    __host__ __device__ __forceinline__                                             \
    static type operator op(type a) {                                               \
        return CAT(make_, type)(op a.x);                                            \
    }

#define DEFINE_VEC1_BINARY_OP_RET(type, op)                                         \
    template <typename T> __host__ __device__ __forceinline__                       \
    static type operator op(type a, T b) {                                          \
        return CAT(make_, type)(a.x op b.x);                                        \
    }

#define DEFINE_VEC1_IBINARY_OP_RET(type, op)                                        \
    template <std::integral T> __host__ __device__ __forceinline__                  \
    static type operator op(type a, T b) {                                          \
        return CAT(make_, type)(a.x op b);                                          \
    }

#define DEFINE_VEC2_UNARY_OP_RET(type, op)                                          \
    __host__ __device__ __forceinline__                                             \
    static type operator op(type a) {                                               \
        return CAT(make_, type)(op a.x, op a.y);                                    \
    }

#define DEFINE_VEC2_BINARY_OP_RET(type, op)                                         \
    template <typename T> __host__ __device__ __forceinline__                       \
    static type operator op(type a, T b) {                                          \
        return CAT(make_, type)(a.x op b.x, a.y op b.y);                            \
    }

#define DEFINE_VEC2_IBINARY_OP_RET(type, op)                                        \
    template <std::integral T> __host__ __device__ __forceinline__                  \
    static type operator op(type a, T b) {                                          \
        return CAT(make_, type)(a.x op b, a.y op b);                                \
    }

#define DEFINE_VEC3_UNARY_OP_RET(type, op)                                          \
    __host__ __device__ __forceinline__                                             \
    static type operator op(type a) {                                               \
        return CAT(make_, type)(op a.x, op a.y, op a.z);                            \
    }

#define DEFINE_VEC3_BINARY_OP_RET(type, op)                                         \
    template <typename T> __host__ __device__ __forceinline__                       \
    static type operator op(type a, T b) {                                          \
        return CAT(make_, type)(a.x op b.x, a.y op b.y, a.z op b.z);                \
    }

#define DEFINE_VEC3_IBINARY_OP_RET(type, op)                                        \
    template <std::integral T> __host__ __device__ __forceinline__                  \
    static type operator op(type a, T b) {                                          \
        return CAT(make_, type)(a.x op b, a.y op b, a.z op b);                      \
    }

#define DEFINE_VEC4_UNARY_OP_RET(type, op)                                          \
    __host__ __device__ __forceinline__                                             \
    static type operator op(type a) {                                               \
        return CAT(make_, type)(op a.x, op a.y, op a.z, op a.w);                    \
    }

#define DEFINE_VEC4_BINARY_OP_RET(type, op)                                         \
    template <typename T> __host__ __device__ __forceinline__                       \
    static type operator op(type a, T b) {                                          \
        return CAT(make_, type)(a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w);    \
    }

#define DEFINE_VEC4_IBINARY_OP_RET(type, op)                                        \
    template <std::integral T> __host__ __device__ __forceinline__                  \
    static type operator op(type a, T b) {                                          \
        return CAT(make_, type)(a.x op b, a.y op b, a.z op b, a.w op b);            \
    }

#define DEFINE_VEC_OPS(type, un, bin, ibin)                                         \
    un  (type, ~);                                                                  \
    un  (type, !);                                                                  \
    bin (type, +);                                                                  \
    bin (type, -);                                                                  \
    bin (type, *);                                                                  \
    bin (type, /);                                                                  \
    bin (type, &);                                                                  \
    bin (type, ^);                                                                  \
    ibin(type, <<);                                                                 \
    ibin(type, >>);


#define DEFINE_VEC1_OPS(type) DEFINE_VEC_OPS(type, DEFINE_VEC1_UNARY_OP_RET, DEFINE_VEC1_BINARY_OP_RET, DEFINE_VEC1_IBINARY_OP_RET)
#define DEFINE_VEC2_OPS(type) DEFINE_VEC_OPS(type, DEFINE_VEC2_UNARY_OP_RET, DEFINE_VEC2_BINARY_OP_RET, DEFINE_VEC2_IBINARY_OP_RET)
#define DEFINE_VEC3_OPS(type) DEFINE_VEC_OPS(type, DEFINE_VEC3_UNARY_OP_RET, DEFINE_VEC3_BINARY_OP_RET, DEFINE_VEC3_IBINARY_OP_RET)
#define DEFINE_VEC4_OPS(type) DEFINE_VEC_OPS(type, DEFINE_VEC4_UNARY_OP_RET, DEFINE_VEC4_BINARY_OP_RET, DEFINE_VEC4_IBINARY_OP_RET)

DEFINE_VEC2_OPS(short2);
DEFINE_VEC2_OPS(ushort2);
DEFINE_VEC2_OPS(int2);
DEFINE_VEC2_OPS(uint2);
DEFINE_VEC3_OPS(short3);
DEFINE_VEC3_OPS(ushort3);
DEFINE_VEC3_OPS(int3);
DEFINE_VEC3_OPS(uint3);
DEFINE_VEC4_OPS(short4);
DEFINE_VEC4_OPS(ushort4);
DEFINE_VEC4_OPS(int4);
DEFINE_VEC4_OPS(uint4);
