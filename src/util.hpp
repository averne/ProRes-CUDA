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

#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <array>
#include <algorithm>

namespace util {

#ifndef __CUDACC__
#   define __host__
#   define __device__
#   define __global__
#   define __shared__
#endif

#define _CAT(x, y) x ## y
#define  CAT(x, y) _CAT(x, y)
#define _STRING(x) #x
#define  STRING(x) _STRING(x)

#define VAR_ANONYMOUS CAT(var, __COUNTER__)

#define SCOPEGUARD(f) auto VAR_ANONYMOUS = ::util::ScopeGuard(f)

#define UNUSED(...) ::util::unused(__VA_ARGS__)

#define CUDA_CHECK(expr) ({                                             \
    cudaError_t _err_ = (expr);                                         \
    if (_err_ != cudaSuccess)                                           \
        std::fprintf(stderr, STRING(expr) ": failed with %s (%d)\n",    \
            cudaGetErrorString(_err_), _err_);                          \
})

template <typename F>
struct ScopeGuard {
    [[nodiscard]] ScopeGuard(F &&f): f(std::move(f)) { }

    ScopeGuard(const ScopeGuard &) = delete;
    ScopeGuard &operator =(const ScopeGuard &) = delete;

    ~ScopeGuard() {
        if (this->want_run)
            this->f();
    }

    void cancel() {
        this->want_run = false;
    }

    private:
        bool want_run = true;
        F f;
};

__host__ __device__
void unused(auto &&...args) {
    (static_cast<void>(args), ...);
}

__host__ __device__
auto align_down(auto v, auto a) {
    return v & ~(a - 1);
}

__host__ __device__
auto align_up(auto v, auto a) {
    return align_down(v + a - 1, a);
}

__host__ __device__
auto bit(auto bit) {
    return static_cast<decltype(bit)>(1) << bit;
}

__host__ __device__
auto mask(auto bit) {
    return (static_cast<decltype(bit)>(1) << bit) - 1;
}

__host__ __device__
auto ceil_rshift(auto a, auto b) {
    return (a + mask(b)) >> b;
}

template <std::integral T> __host__
T byteswap(T v) {
    if (std::is_same_v<T, std::uint8_t>) {
        return v;
    } else if (std::is_same_v<T, std::uint16_t>) {
        return __builtin_bswap16(v);
    } else if (std::is_same_v<T, std::uint32_t>) {
        return __builtin_bswap32(v);
    } else if (std::is_same_v<T, std::uint64_t>) {
        return __builtin_bswap64(v);
    } else {
        static_assert(std::is_unsigned_v<T>, "Unsupported type for byteswap");
        return 0;
    }
}

struct FourCC {
    consteval FourCC(char a, char b, char c, char d) {
        this->code = (d << 0x18) | (c << 0x10) | (b << 0x8) | a;
    }

    operator std::uint32_t() const {
        return this->code;
    }

    private:
        std::uint32_t code = 0;
};

class Bytestream {
    public:
        template <typename T>
        Bytestream(const T *data, std::size_t size):
            data(reinterpret_cast<std::uintptr_t>(data)), size(size) { }

        template <typename T>
        T get() {
            assert(this->pos + sizeof(T) < this->size);

            SCOPEGUARD([this] { this->pos += sizeof(T); });
            return *reinterpret_cast<const T *>(this->data + this->pos);
        }

        template <typename T>
        T get_be() {
            assert(this->pos + sizeof(T) < this->size);

            SCOPEGUARD([this] { this->pos += sizeof(T); });
            return byteswap(*reinterpret_cast<const T *>(this->data + this->pos));
        }

        template <typename T>
        void read_into(T *dst, std::size_t size) {
            assert(this->pos + size < this->size);
            std::copy_n(reinterpret_cast<const T *>(this->data + this->pos), size, dst);
            this->pos += size;
        }

        std::size_t skip(std::size_t n) {
            assert(this->pos + n < this->size);
            return this->pos += n;
        }

        std::size_t tell() const {
            return this->pos;
        }

        std::size_t left() const {
            return this->size - this->pos;
        }

    private:
        const std::uintptr_t data;
        std::size_t size, pos = 0;
};

class Bitstream {
    public:
        Bitstream() = default;

        template <typename T> __host__ __device__
        Bitstream(const T *data, std::size_t size) {
            this->init(data, size);
        }

        template <typename T> __host__ __device__
        void init(const T *data, std::size_t size) {
            this->buf_start = this->buf = reinterpret_cast<std::uintptr_t>(data);
            this->buf_end = this->buf + size;
            this->size_in_bits = size * 8;

            this->load64();
        }

        __host__ __device__
        bool get_bit() {
            if (this->bits_valid == 0)
                this->load64();

            auto val = !!(this->bits >> (sizeof(this->bits) * 8 - 1));
            this->bits <<= 1;
            this->bits_valid--;
            return val;
        }

        __host__ __device__
        std::uint32_t get_bits(std::size_t n) {
            if (n == 0)
                return 0;

            if (n > this->bits_valid)
                this->reload32();

            auto val = static_cast<std::uint32_t>(this->bits >> (sizeof(this->bits) * 8 - n));
            this->bits <<= n;
            this->bits_valid -= n;
            return val;
        }

        __host__ __device__
        std::uint32_t show_bits(std::size_t n) {
            if (n > this->bits_valid)
                this->reload32();

            return static_cast<std::uint32_t>(this->bits >> (sizeof(this->bits) * 8 - n));
        }

        __host__ __device__
        void skip_bits(std::size_t n) {
            if (n > this->bits_valid)
                this->reload32();

            this->bits <<= n;
            this->bits_valid -= n;
        }

        __host__ __device__
        std::uint32_t tell_bits() const {
            return (this->buf - this->buf_start) * 8 - this->bits_valid;
        }

        __host__ __device__
        std::uint32_t left_bits() const {
            return this->size_in_bits - (this->buf - this->buf_start) * 8 + this->bits_valid;
        }

    private:
        __host__ __device__
        void load64() {
#ifndef __CUDACC__
            auto hi = byteswap(*reinterpret_cast<const std::uint32_t *>(this->buf));
            auto lo = byteswap(*reinterpret_cast<const std::uint32_t *>(this->buf + 4));
            this->bits = (static_cast<std::uint64_t>(hi) << 32) | lo;
#else
            this->reverse_copy(reinterpret_cast<std::uint8_t *>(&this->bits), sizeof(this->bits));
#endif

            this->buf       += sizeof(std::uint64_t);
            this->bits_valid = sizeof(std::uint64_t) * 8;
        }

        __host__ __device__
        void reload32() {
#ifndef __CUDACC__
            auto v = byteswap(*reinterpret_cast<const std::uint32_t *>(this->buf));
#else
            std::uint32_t v;
            this->reverse_copy(reinterpret_cast<std::uint8_t *>(&v), sizeof(v));
#endif
            this->bits |= static_cast<std::uint64_t>(v) << (sizeof(std::uint32_t) * 8 - this->bits_valid);

            this->buf        += sizeof(std::uint32_t);
            this->bits_valid += sizeof(std::uint32_t) * 8;
        }

        __host__ __device__
        void reverse_copy(std::uint8_t *dst, std::size_t size) const {
            for (auto src = reinterpret_cast<std::uint8_t *>(this->buf); size > 0; --size, ++dst)
                *dst = *(src + size - 1);
        }

    private:
        std::uintptr_t buf_start, buf, buf_end;

        std::uint64_t bits = 0;
        uint32_t bits_valid, size_in_bits = 0;
};

} // namespace util
