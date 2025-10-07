// -*- C++ -*-
//===-- esimd_radix_sort_utils.h -----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------===//

#ifndef _ONEDPL_KT_ESIMD_RADIX_SORT_UTILS_H
#define _ONEDPL_KT_ESIMD_RADIX_SORT_UTILS_H

#include <limits>
#include <cstdint>
#include <type_traits>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"

#include "oneapi/dpl/pstl/onedpl_config.h"
#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"

#include "esimd_defs.h"

namespace oneapi::dpl::experimental::kt::gpu::esimd::__impl
{

template <::std::uint8_t __radix_bits, ::std::uint16_t __data_per_workitem, ::std::uint16_t __workgroup_size>
constexpr void
__check_esimd_sort_params()
{
    static_assert(__radix_bits == 8);
    static_assert(__data_per_workitem % 32 == 0);
    static_assert(__workgroup_size == 32 || __workgroup_size == 64);
}

template <typename _RT, typename _T>
inline __dpl_esimd::__ns::simd<_RT, 32>
__scan(__dpl_esimd::__ns::simd<_T, 32> __src, const _T __init = 0)
{
    __dpl_esimd::__ns::simd<_RT, 32> __res;
    __res.template select<8, 4>(0) = __src.template select<8, 4>(0);
    __res[0] += __init;
    __res.template select<8, 4>(1) = __src.template select<8, 4>(1) + __res.template select<8, 4>(0);
    __res.template select<8, 4>(2) = __src.template select<8, 4>(2) + __res.template select<8, 4>(1);
    __res.template select<8, 4>(3) = __src.template select<8, 4>(3) + __res.template select<8, 4>(2);
    __res.template select<4, 1>(4) = __res.template select<4, 1>(4) + __res[3];
    __res.template select<4, 1>(8) = __res.template select<4, 1>(8) + __res[7];
    __res.template select<4, 1>(12) = __res.template select<4, 1>(12) + __res[11];
    __res.template select<4, 1>(16) = __res.template select<4, 1>(16) + __res[15];
    __res.template select<4, 1>(20) = __res.template select<4, 1>(20) + __res[19];
    __res.template select<4, 1>(24) = __res.template select<4, 1>(24) + __res[23];
    __res.template select<4, 1>(28) = __res.template select<4, 1>(28) + __res[27];
    return __res;
}

template <typename _RT, typename _T>
inline __dpl_esimd::__ns::simd<_RT, 16>
__scan(__dpl_esimd::__ns::simd<_T, 16> __src, const _T __init = 0)
{
    __dpl_esimd::__ns::simd<_RT, 16> __res;
    __res.template select<4, 4>(0) = __src.template select<4, 4>(0);
    __res[0] += __init;
    __res.template select<4, 4>(1) = __src.template select<4, 4>(1) + __res.template select<4, 4>(0);
    __res.template select<4, 4>(2) = __src.template select<4, 4>(2) + __res.template select<4, 4>(1);
    __res.template select<4, 4>(3) = __src.template select<4, 4>(3) + __res.template select<4, 4>(2);
    __res.template select<4, 1>(4) = __res.template select<4, 1>(4) + __res[3];
    __res.template select<4, 1>(8) = __res.template select<4, 1>(8) + __res[7];
    __res.template select<4, 1>(12) = __res.template select<4, 1>(12) + __res[11];
    return __res;
}

template <typename _RT, typename _T>
inline __dpl_esimd::__ns::simd<_RT, 64>
__scan(__dpl_esimd::__ns::simd<_T, 64> __src, const _T __init = 0)
{
    __dpl_esimd::__ns::simd<_RT, 64> __res;
    __res.template select<32, 1>(0) = __scan<_RT, _T>(__src.template select<32, 1>(0), __init);
    __res.template select<32, 1>(32) = __scan<_RT, _T>(__src.template select<32, 1>(32), __res[31]);
    return __res;
}

// get bits value (bucket) in a certain radix position
template <::std::uint16_t __radix_mask, typename _T, int _N, std::enable_if_t<::std::is_unsigned_v<_T>, int> = 0>
__dpl_esimd::__ns::simd<::std::uint16_t, _N>
__get_bucket(__dpl_esimd::__ns::simd<_T, _N> __value, ::std::uint32_t __radix_offset)
{
    return __dpl_esimd::__ns::simd<::std::uint16_t, _N>(__value >> __radix_offset) & __radix_mask;
}

template <typename _T, bool __is_ascending, std::enable_if_t<::std::is_integral_v<_T>, int> = 0>
constexpr _T
__sort_identity()
{
    if constexpr (__is_ascending)
        return ::std::numeric_limits<_T>::max();
    else
        return ::std::numeric_limits<_T>::lowest();
}

// std::numeric_limits<_T>::max and std::numeric_limits<_T>::lowest cannot be used as an idenentity for
// performing radix sort of floating point numbers.
// They do not set the smallest exponent bit (i.e. the max is 7F7FFFFF for 32bit float),
// thus such an identity is not guaranteed to be put at the end of the sorted sequence after each radix sort stage,
// e.g. 00FF0000 numbers will be pushed out by 7F7FFFFF identities when sorting 16-23 bits.
template <typename _T, bool __is_ascending,
          std::enable_if_t<::std::is_floating_point_v<_T> && sizeof(_T) == sizeof(::std::uint32_t), int> = 0>
constexpr _T
__sort_identity()
{
    if constexpr (__is_ascending)
        return sycl::bit_cast<_T>(0x7FFF'FFFFu);
    else
        return sycl::bit_cast<_T>(0xFFFF'FFFFu);
}

template <typename _T, bool __is_ascending,
          std::enable_if_t<::std::is_floating_point_v<_T> && sizeof(_T) == sizeof(::std::uint64_t), int> = 0>
constexpr _T
__sort_identity()
{
    if constexpr (__is_ascending)
        return sycl::bit_cast<_T>(0x7FFF'FFFF'FFFF'FFFFu);
    else
        return sycl::bit_cast<_T>(0xFFFF'FFFF'FFFF'FFFFu);
}

template <bool __is_ascending, int _N>
__dpl_esimd::__ns::simd<bool, _N>
__order_preserving_cast(__dpl_esimd::__ns::simd<bool, _N> __src)
{
    if constexpr (__is_ascending)
        return __src;
    else
        return !__src;
}

template <bool __is_ascending, typename _UInt, int _N, std::enable_if_t<::std::is_unsigned_v<_UInt>, int> = 0>
__dpl_esimd::__ns::simd<_UInt, _N>
__order_preserving_cast(__dpl_esimd::__ns::simd<_UInt, _N> __src)
{
    if constexpr (__is_ascending)
        return __src;
    else
        return ~__src; //bitwise inversion
}

template <bool __is_ascending, typename _Int, int _N,
          std::enable_if_t<::std::is_integral_v<_Int>&& ::std::is_signed_v<_Int>, int> = 0>
__dpl_esimd::__ns::simd<::std::make_unsigned_t<_Int>, _N>
__order_preserving_cast(__dpl_esimd::__ns::simd<_Int, _N> __src)
{
    using _UInt = ::std::make_unsigned_t<_Int>;
    // __mask: 100..0 for ascending, 011..1 for descending
    constexpr _UInt __mask =
        (__is_ascending) ? _UInt(1) << ::std::numeric_limits<_Int>::digits : ::std::numeric_limits<_UInt>::max() >> 1;
    return __src.template bit_cast_view<_UInt>() ^ __mask;
}

template <bool __is_ascending, typename _Float, int _N,
          std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint32_t), int> = 0>
__dpl_esimd::__ns::simd<::std::uint32_t, _N>
__order_preserving_cast(__dpl_esimd::__ns::simd<_Float, _N> __src)
{
    __dpl_esimd::__ns::simd<::std::uint32_t, _N> __uint32_src = __src.template bit_cast_view<::std::uint32_t>();
    __dpl_esimd::__ns::simd<::std::uint32_t, _N> __mask;
    __dpl_esimd::__ns::simd_mask<_N> __sign_bit_m = (__uint32_src >> 31 == 0);
    if constexpr (__is_ascending)
    {
        __mask = __dpl_esimd::__ns::merge(__dpl_esimd::__ns::simd<::std::uint32_t, _N>(0x80000000u),
                                          __dpl_esimd::__ns::simd<::std::uint32_t, _N>(0xFFFFFFFFu), __sign_bit_m);
    }
    else
    {
        __mask =
            __dpl_esimd::__ns::merge(__dpl_esimd::__ns::simd<::std::uint32_t, _N>(0x7FFFFFFFu),
                                     __dpl_esimd::__ns::simd<::std::uint32_t, _N>(::std::uint32_t(0)), __sign_bit_m);
    }
    return __uint32_src ^ __mask;
}

template <bool __is_ascending, typename _Float, int _N,
          std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint64_t), int> = 0>
__dpl_esimd::__ns::simd<::std::uint64_t, _N>
__order_preserving_cast(__dpl_esimd::__ns::simd<_Float, _N> __src)
{
    __dpl_esimd::__ns::simd<::std::uint64_t, _N> __uint64_src = __src.template bit_cast_view<::std::uint64_t>();
    __dpl_esimd::__ns::simd<::std::uint64_t, _N> __mask;
    __dpl_esimd::__ns::simd_mask<_N> __sign_bit_m = (__uint64_src >> 63 == 0);
    if constexpr (__is_ascending)
    {
        __mask =
            __dpl_esimd::__ns::merge(__dpl_esimd::__ns::simd<::std::uint64_t, _N>(0x8000000000000000u),
                                     __dpl_esimd::__ns::simd<::std::uint64_t, _N>(0xFFFFFFFFFFFFFFFFu), __sign_bit_m);
    }
    else
    {
        __mask =
            __dpl_esimd::__ns::merge(__dpl_esimd::__ns::simd<::std::uint64_t, _N>(0x7FFFFFFFFFFFFFFFu),
                                     __dpl_esimd::__ns::simd<::std::uint64_t, _N>(::std::uint64_t(0)), __sign_bit_m);
    }
    return __uint64_src ^ __mask;
}

template <typename _T, int _N>
inline std::enable_if_t<(_N > 16) && (_N % 16 == 0), __dpl_esimd::__ns::simd<_T, _N>>
__create_simd(_T initial, _T step)
{
    __dpl_esimd::__ns::simd<_T, _N> ret;
    ret.template select<16, 1>(0) = __dpl_esimd::__ns::simd<_T, 16>(0, 1) * step + initial;
    _ONEDPL_PRAGMA_UNROLL
    for (int pos = 16; pos < _N; pos += 16)
    {
        ret.template select<16, 1>(pos) = ret.template select<16, 1>(0) + pos * step;
    }
    return ret;
}

template <typename _T>
struct __slm_lookup
{
    ::std::uint32_t __slm;
    inline __slm_lookup(::std::uint32_t __slm) : __slm(__slm) {}

    template <int __table_size>
    inline void
    __setup(__dpl_esimd::__ns::simd<_T, __table_size> __source) SYCL_ESIMD_FUNCTION
    {
        __dpl_esimd::__block_store_slm<_T, __table_size>(__slm, __source);
    }

    template <int _N, typename _Idx>
    inline auto
    __lookup(_Idx __idx) SYCL_ESIMD_FUNCTION
    {
        return __dpl_esimd::__vector_load<_T, 1, _N>(__slm +
                                                     __dpl_esimd::__ns::simd<::std::uint32_t, _N>(__idx) * sizeof(_T));
    }

    template <int _N, int __table_size, typename _Idx>
    inline auto
    __lookup(__dpl_esimd::__ns::simd<_T, __table_size> __source, _Idx __idx) SYCL_ESIMD_FUNCTION
    {
        __setup(__source);
        return __lookup<_N>(__idx);
    }
};

template <typename _Rng>
auto
__rng_data(const _Rng& __rng)
{
    return __rng.begin();
}

// ESIMD functionality requires using an accessor directly due to the restriction:
//      sycl::accessor::operator[] are supported only with -fsycl-esimd-force-stateless-mem.
//      Otherwise, all memory accesses through an accessor are done via explicit APIs
// TODO: rely on begin() once -fsycl-esimd-force-stateless-mem has been enabled by default
template <typename _T, sycl::access::mode _M>
auto
__rng_data(const oneapi::dpl::__ranges::all_view<_T, _M>& __view)
{
    return __view.accessor();
}

struct __rng_dummy
{
};

template <typename _Rng>
struct __rng_value_type_deducer
{
    using __value_t = oneapi::dpl::__internal::__value_t<_Rng>;
};

template <>
struct __rng_value_type_deducer<__rng_dummy>
{
    using __value_t = void;
};

template <typename _Rng1, typename _Rng2 = __rng_dummy>
struct __rng_pack
{
    using _KeyT = typename __rng_value_type_deducer<_Rng1>::__value_t;
    using _ValT = typename __rng_value_type_deducer<_Rng2>::__value_t;
    static constexpr bool __has_values = !std::is_void_v<_ValT>;

    const auto&
    __keys_rng() const
    {
        return __m_keys_rng;
    }
    const auto&
    __vals_rng() const
    {
        static_assert(__has_values);
        return __m_vals_rng;
    }

    __rng_pack(const _Rng1& __rng1, const _Rng2& __rng2 = __rng_dummy{}) : __m_keys_rng(__rng1), __m_vals_rng(__rng2) {}
    __rng_pack(_Rng1&& __rng1, _Rng2&& __rng2 = __rng_dummy{})
        : __m_keys_rng(::std::move(__rng1)), __m_vals_rng(::std::move(__rng2))
    {
    }

  private:
    _Rng1 __m_keys_rng;
    _Rng2 __m_vals_rng;
};

template <::std::uint16_t _N, typename _KeyT>
struct __keys_simd_pack
{
    __dpl_esimd::__ns::simd<_KeyT, _N> __keys;
};

template <::std::uint16_t _N, typename _KeyT, typename _ValT>
struct __pairs_simd_pack
{
    __dpl_esimd::__ns::simd<_KeyT, _N> __keys;
    __dpl_esimd::__ns::simd<_ValT, _N> __vals;
};

template <::std::uint16_t _N, typename _T1, typename _T2 = void>
auto
__make_simd_pack()
{
    if constexpr (::std::is_void_v<_T2>)
    {
        return __keys_simd_pack<_N, _T1>{};
    }
    else
    {
        return __pairs_simd_pack<_N, _T1, _T2>{};
    }
}

} // namespace oneapi::dpl::experimental::kt::gpu::esimd::__impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_UTILS_H
