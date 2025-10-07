// -*- C++ -*-
//===-- esimd_defs.h -----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------===//

#ifndef _ONEDPL_KT_ESIMD_DEFS_H
#define _ONEDPL_KT_ESIMD_DEFS_H

#include <limits>
#include <cstdint>
#include <type_traits>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include <sycl/ext/intel/esimd.hpp>

// The macro is created to guarantee inlining of functions which contain slm_init. See ESIMD spec for more details:
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md#static-allocation-of-slm-using-slm_init-function
#define _ONEDPL_ESIMD_INLINE inline __attribute__((always_inline))

#define _ONEDPL_ESIMD_LSC_FENCE_PRESENT (_ONEDPL_LIBSYCL_VERSION >= 70200)

namespace oneapi::dpl::experimental::kt::gpu::esimd::__impl
{

// TODO: rename to show the meaning clearly: default vectorization factor
constexpr int __data_per_step = 16;

} // namespace oneapi::dpl::experimental::kt::gpu::esimd::__impl

// This namespace mostly consists of abstractions on the top of regular ESIMD functions.
// The purpose is:
//  - Provide unified API for sycl::buffer accessors and USM pointers.
//  - Provide better performance.
// TODO: contribute into ESIMD or form a feature request.
namespace __dpl_esimd
{

namespace __ns = sycl::ext::intel::esimd;
namespace __ens = sycl::ext::intel::experimental::esimd;

// converts sizeof(_T) to 32 bits, so that it could be used in operations with 32-bit SIMD without changing the type
template <typename _T>
inline constexpr ::std::uint32_t __size32 = sizeof(_T);

template <typename _T, int _N>
void
__copy_from(const _T* __input, ::std::uint32_t __base_offset, __ns::simd<_T, _N>& __values)
{
    __values.copy_from(__input + __base_offset);
}

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P>
void
__copy_from(sycl::accessor<_T, 1, _Mode, sycl::target::device, _P> __input, ::std::uint32_t __base_offset,
            __ns::simd<_T, _N>& __values)
{
    __values.copy_from(__input, __base_offset * __size32<_T>);
}

template <typename _T, int _N>
void
__copy_to(_T* __output, ::std::uint32_t __base_offset, const __ns::simd<_T, _N>& __values)
{
    __values.copy_to(__output + __base_offset);
}

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P>
void
__copy_to(sycl::accessor<_T, 1, _Mode, sycl::target::device, _P> __output, ::std::uint32_t __base_offset,
          const __ns::simd<_T, _N>& __values)
{
    __values.copy_to(__output, __base_offset * __size32<_T>);
}

template <typename _T, int _N>
__ns::simd<_T, _N>
__gather(const _T* __input, __ns::simd<::std::uint32_t, _N> __offsets, ::std::uint32_t __base_offset,
         __ns::simd_mask<_N> __mask = 1)
{
    return __ns::gather(__input + __base_offset, __offsets * __size32<_T>, __mask);
}

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P,
          ::std::enable_if_t<sizeof(_T) <= sizeof(::std::uint32_t), int> = 0>
__ns::simd<_T, _N>
__gather(sycl::accessor<_T, 1, _Mode, sycl::target::device, _P> __input, __ns::simd<::std::uint32_t, _N> __offsets,
         ::std::uint32_t __base_offset, __ns::simd_mask<_N> __mask = 1)
{
    return __ns::gather<_T>(__input, __offsets * __size32<_T>, __base_offset * __size32<_T>, __mask);
}

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P,
          ::std::enable_if_t<sizeof(_T) == sizeof(::std::uint64_t), int> = 0>
__ns::simd<_T, _N>
__gather(sycl::accessor<_T, 1, _Mode, sycl::target::device, _P> __input, __ns::simd<::std::uint32_t, _N> __offsets,
         ::std::uint32_t __base_offset, __ns::simd_mask<_N> __mask = 1)
{
    return __ens::lsc_gather<_T>(__input, __offsets * __size32<_T> + __base_offset * __size32<_T>, __mask);
}

template <typename _T, int _N>
void
__scatter(_T* __output, __ns::simd<::std::uint32_t, _N> __offsets, __ns::simd<_T, _N> __values,
          __ns::simd_mask<_N> __mask = 1)
{
    __ns::scatter(__output, __offsets * __size32<_T>, __values, __mask);
}

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P,
          ::std::enable_if_t<sizeof(_T) <= sizeof(::std::uint32_t), int> = 0>
void
__scatter(sycl::accessor<_T, 1, _Mode, sycl::target::device, _P> __output, __ns::simd<::std::uint32_t, _N> __offsets,
          __ns::simd<_T, _N> __values, __ns::simd_mask<_N> __mask = 1)
{
    __ns::scatter(__output, __offsets * __size32<_T>, __values, /*global_offset*/ 0, __mask);
}

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P,
          ::std::enable_if_t<sizeof(_T) == sizeof(::std::uint64_t), int> = 0>
void
__scatter(sycl::accessor<_T, 1, _Mode, sycl::target::device, _P> __output, __ns::simd<::std::uint32_t, _N> __offsets,
          __ns::simd<_T, _N> __values, __ns::simd_mask<_N> __mask = 1)
{
    __ens::lsc_scatter<_T>(__output, __offsets * __size32<_T>, __values, __mask);
}

template <typename _T, int _VSize, int _LANES, __ens::cache_hint _H1 = __ens::cache_hint::none,
          __ens::cache_hint _H3 = __ens::cache_hint::none>
inline std::enable_if_t<(_VSize <= 4), __ns::simd<_T, _VSize * _LANES>>
__vector_load(const _T* __src, const __ns::simd<::std::uint32_t, _LANES> __offset, __ns::simd_mask<_LANES> __mask = 1)
{
    return __ens::lsc_gather<_T, _VSize, __ens::lsc_data_size::default_size, _H1, _H3, _LANES>(__src, __offset, __mask);
}

template <typename _T, int _VSize, int _LANES, __ens::cache_hint _H1 = __ens::cache_hint::none,
          __ens::cache_hint _H3 = __ens::cache_hint::none>
inline std::enable_if_t<(_VSize > 4), __ns::simd<_T, _VSize * _LANES>>
__vector_load(const _T* __src, const __ns::simd<::std::uint32_t, _LANES> __offset, __ns::simd_mask<_LANES> __mask = 1)
{
    __ns::simd<_T, _VSize * _LANES> __res;
    __res.template select<4 * _LANES, 1>(0) = __vector_load<_T, 4, _LANES, _H1, _H3>(__src, __offset, __mask);
    __res.template select<(_VSize - 4) * _LANES, 1>(4 * _LANES) =
        __vector_load<_T, _VSize - 4, _LANES, _H1, _H3>(__src, __offset + 4 * sizeof(_T), __mask);
    return __res;
}

template <typename _T, int _VSize, int _LANES, int LaneStride = _VSize, __ens::cache_hint _H1 = __ens::cache_hint::none,
          __ens::cache_hint _H3 = __ens::cache_hint::none>
inline __ns::simd<_T, _VSize * _LANES>
__vector_load(const _T* __src, ::std::uint32_t __offset, __ns::simd_mask<_LANES> __mask = 1)
{
    return __vector_load<_T, _VSize, _LANES, _H1, _H3>(__src, {__offset, LaneStride * sizeof(_T)}, __mask);
}

template <typename _T, int _VSize, int _LANES>
inline std::enable_if_t<(_VSize <= 4), __ns::simd<_T, _VSize * _LANES>>
__vector_load(const __ns::simd<::std::uint32_t, _LANES> __offset, __ns::simd_mask<_LANES> __mask = 1)
{
    return __ens::lsc_slm_gather<_T, _VSize, __ens::lsc_data_size::default_size, _LANES>(__offset, __mask);
}

template <typename _T, int _VSize, int _LANES>
inline std::enable_if_t<(_VSize > 4), __ns::simd<_T, _VSize * _LANES>>
__vector_load(const __ns::simd<::std::uint32_t, _LANES> __offset, __ns::simd_mask<_LANES> __mask = 1)
{
    __ns::simd<_T, _VSize * _LANES> __res;
    __res.template select<4 * _LANES, 1>(0) = __vector_load<_T, 4, _LANES>(__offset, __mask);
    __res.template select<(_VSize - 4) * _LANES, 1>(4 * _LANES) =
        __vector_load<_T, _VSize - 4, _LANES>(__offset + 4 * sizeof(_T), __mask);
    return __res;
}

template <typename _T, int _VSize, int _LANES, int LaneStride = _VSize>
inline __ns::simd<_T, _VSize * _LANES>
__vector_load(::std::uint32_t __offset, __ns::simd_mask<_LANES> __mask = 1)
{
    return __vector_load<_T, _VSize, _LANES>({__offset, LaneStride * sizeof(_T)}, __mask);
}

template <typename TWhatEver>
struct __is_sycl_accessor : ::std::false_type
{
};

template <typename _T, int _N, sycl::access_mode _Mode, sycl::access::placeholder _P>
struct __is_sycl_accessor<sycl::accessor<_T, _N, _Mode, sycl::target::device, _P>> : ::std::true_type
{
};

template <typename... _Tp>
inline constexpr bool is_sycl_accessor_v = __is_sycl_accessor<_Tp...>::value;

template <typename _T, int _VSize, int _LANES, __ens::cache_hint _H1 = __ens::cache_hint::none,
          __ens::cache_hint _H3 = __ens::cache_hint::none>
inline std::enable_if_t<(_VSize <= 4 && _LANES <= 32), void>
__vector_store(_T* __dest, __ns::simd<::std::uint32_t, _LANES> __offset, __ns::simd<_T, _VSize * _LANES> __data,
               __ns::simd_mask<_LANES> __mask = 1)
{
    __ens::lsc_scatter<_T, _VSize, __ens::lsc_data_size::default_size, _H1, _H3, _LANES>(__dest, __offset, __data,
                                                                                         __mask);
}

template <typename _T, int _VSize, int _LANES, typename _AccessorTy, __ens::cache_hint _H1 = __ens::cache_hint::none,
          __ens::cache_hint _H3 = __ens::cache_hint::none>
inline std::enable_if_t<(is_sycl_accessor_v<_AccessorTy> && _VSize <= 4 && _LANES <= 32), void>
__vector_store(_AccessorTy __acc, __ns::simd<::std::uint32_t, _LANES> __offset, __ns::simd<_T, _VSize * _LANES> __data,
               __ns::simd_mask<_LANES> __mask = 1)
{
    __ens::lsc_scatter<_T, _VSize, __ens::lsc_data_size::default_size, _H1, _H3, _LANES>(__acc, __offset, __data,
                                                                                         __mask);
}

template <typename _T, int _VSize, int _LANES, __ens::cache_hint _H1 = __ens::cache_hint::none,
          __ens::cache_hint _H3 = __ens::cache_hint::none>
inline std::enable_if_t<(_LANES > 32), void>
__vector_store(_T* __dest, __ns::simd<::std::uint32_t, _LANES> __offset, __ns::simd<_T, _VSize * _LANES> __data,
               __ns::simd_mask<_LANES> __mask = 1)
{
    __vector_store<_T, _VSize, 32>(__dest, __offset.template select<32, 1>(0),
                                   __data.template select<_VSize * 32, 1>(0), __mask.template select<32, 1>(0));
    __vector_store<_T, _VSize, _LANES - 32>(__dest, __offset.template select<_LANES - 32, 1>(32),
                                            __data.template select<_VSize*(_LANES - 32), 1>(32),
                                            __mask.template select<_LANES - 32, 1>(32));
}

template <typename _T, int _VSize, int _LANES, typename _AccessorTy, __ens::cache_hint _H1 = __ens::cache_hint::none,
          __ens::cache_hint _H3 = __ens::cache_hint::none>
inline std::enable_if_t<(is_sycl_accessor_v<_AccessorTy> && _LANES > 32), void>
__vector_store(_AccessorTy __acc, __ns::simd<::std::uint32_t, _LANES> __offset, __ns::simd<_T, _VSize * _LANES> __data,
               __ns::simd_mask<_LANES> __mask = 1)
{
    __vector_store<_T, _VSize, 32>(__acc, __offset.template select<32, 1>(0), __data.template select<_VSize * 32, 1>(0),
                                   __mask.template select<32, 1>(0));
    __vector_store<_T, _VSize, _LANES - 32>(__acc, __offset.template select<_LANES - 32, 1>(32),
                                            __data.template select<_VSize*(_LANES - 32), 1>(32),
                                            __mask.template select<_LANES - 32, 1>(32));
}

template <typename _T, int _VSize, int _LANES, __ens::cache_hint _H1 = __ens::cache_hint::none,
          __ens::cache_hint _H3 = __ens::cache_hint::none>
inline std::enable_if_t<(_VSize > 4 && _LANES <= 32), void>
__vector_store(_T* __dest, __ns::simd<::std::uint32_t, _LANES> __offset, __ns::simd<_T, _VSize * _LANES> __data,
               __ns::simd_mask<_LANES> __mask = 1)
{
    __vector_store<_T, 4, _LANES>(__dest, __offset, __data.template select<4 * _LANES, 1>(0), __mask);
    __vector_store<_T, _VSize - 4, _LANES>(__dest, __offset + 4 * sizeof(_T),
                                           __data.template select<(_VSize - 4) * _LANES, 1>(4 * _LANES), __mask);
}

template <typename _T, int _VSize, int _LANES, typename _AccessorTy, __ens::cache_hint _H1 = __ens::cache_hint::none,
          __ens::cache_hint _H3 = __ens::cache_hint::none>
inline std::enable_if_t<(is_sycl_accessor_v<_AccessorTy> && _VSize > 4 && _LANES <= 32), void>
__vector_store(_AccessorTy __acc, __ns::simd<::std::uint32_t, _LANES> __offset, __ns::simd<_T, _VSize * _LANES> __data,
               __ns::simd_mask<_LANES> __mask = 1)
{
    __vector_store<_T, 4, _LANES, _AccessorTy>(__acc, __offset, __data.template select<4 * _LANES, 1>(0), __mask);
    __vector_store<_T, _VSize - 4, _LANES, _AccessorTy>(
        __acc, __offset + 4 * sizeof(_T), __data.template select<(_VSize - 4) * _LANES, 1>(4 * _LANES), __mask);
}

template <typename _T, int _VSize, int _LANES, int LaneStride = _VSize, __ens::cache_hint _H1 = __ens::cache_hint::none,
          __ens::cache_hint _H3 = __ens::cache_hint::none>
inline void
__vector_store(_T* __dest, ::std::uint32_t __offset, __ns::simd<_T, _VSize * _LANES> __data,
               __ns::simd_mask<_LANES> __mask = 1)
{
    // optimization needed here, hard for compiler to optimize the __offset vector calculation
    __vector_store<_T, _VSize, _LANES, _H1, _H3>(__dest, {__offset, LaneStride * sizeof(_T)}, __data, __mask);
}

template <typename _T, int _VSize, int _LANES, typename _AccessorTy, int LaneStride = _VSize,
          __ens::cache_hint _H1 = __ens::cache_hint::none, __ens::cache_hint _H3 = __ens::cache_hint::none>
inline std::enable_if_t<(is_sycl_accessor_v<_AccessorTy>), void>
__vector_store(_AccessorTy __acc, ::std::uint32_t __offset, __ns::simd<_T, _VSize * _LANES> __data,
               __ns::simd_mask<_LANES> __mask = 1)
{
    // optimization needed here, hard for compiler to optimize the __offset vector calculation
    __vector_store<_T, _VSize, _LANES, _H1, _H3>(__acc, {__offset, LaneStride * sizeof(_T)}, __data, __mask);
}

template <typename _T, int _VSize, int _LANES>
inline std::enable_if_t<(_VSize <= 4 && _LANES <= 32), void>
__vector_store(__ns::simd<::std::uint32_t, _LANES> __offset, __ns::simd<_T, _VSize * _LANES> __data,
               __ns::simd_mask<_LANES> __mask = 1)
{
    __ens::lsc_slm_scatter<_T, _VSize>(__offset, __data, __mask);
}

template <typename _T, int _VSize, int _LANES>
inline std::enable_if_t<(_LANES > 32), void>
__vector_store(__ns::simd<::std::uint32_t, _LANES> __offset, __ns::simd<_T, _VSize * _LANES> __data,
               __ns::simd_mask<_LANES> __mask = 1)
{
    __vector_store<_T, _VSize, 32>(__offset.template select<32, 1>(0), __data.template select<_VSize * 32, 1>(0),
                                   __mask.template select<32, 1>(0));
    __vector_store<_T, _VSize, _LANES - 32>(__offset.template select<_LANES - 32, 1>(32),
                                            __data.template select<_VSize*(_LANES - 32), 1>(32),
                                            __mask.template select<_LANES - 32, 1>(32));
}

template <typename _T, int _VSize, int _LANES>
inline std::enable_if_t<(_VSize > 4 && _LANES <= 32), void>
__vector_store(__ns::simd<::std::uint32_t, _LANES> __offset, __ns::simd<_T, _VSize * _LANES> __data,
               __ns::simd_mask<_LANES> __mask = 1)
{
    __vector_store<_T, 4, _LANES>(__offset, __data.template select<4 * _LANES, 1>(0), __mask);
    __vector_store<_T, _VSize - 4, _LANES>(__offset + 4 * sizeof(_T),
                                           __data.template select<(_VSize - 4) * _LANES, 1>(4 * _LANES), __mask);
}

template <typename _T, int _VSize, int _LANES, int LaneStride = _VSize>
inline void
__vector_store(::std::uint32_t __offset, __ns::simd<_T, _VSize * _LANES> __data, __ns::simd_mask<_LANES> __mask = 1)
{
    return __vector_store<_T, _VSize, _LANES>({__offset, LaneStride * sizeof(_T)}, __data, __mask);
}

template <int _NElts>
constexpr int
__lsc_slm_block_size_rounding()
{
    static_assert(_NElts >= 1);

    if constexpr (_NElts < 2)
        return 1;

    if constexpr (_NElts < 3)
        return 2;

    if constexpr (_NElts < 4)
        return 3;

    if constexpr (_NElts < 8)
        return 4;

    if constexpr (_NElts < 16)
        return 8;

    if constexpr (_NElts < 32)
        return 16;

    if constexpr (_NElts < 64)
        return 32;

    return 64;
}

template <typename _SrcType, int _N, typename _DestType>
constexpr int
__lsc_op_block_size()
{
    return _N * sizeof(_SrcType) / sizeof(_DestType);
}

template <typename _T>
using _LscOpAlignedT = ::std::conditional_t<sizeof(_T) <= sizeof(::std::uint32_t), ::std::uint32_t, ::std::uint64_t>;

template <typename _T, int _N, typename _OpAlignedT = _LscOpAlignedT<_T>,
          int _NElts = __lsc_op_block_size<_T, _N, _OpAlignedT>(),
          ::std::enable_if_t<_NElts == __lsc_slm_block_size_rounding<_NElts>(), int> = 0>
inline __ns::simd<_T, _N>
__block_load_slm(::std::uint32_t __slm_offset)
{
    __ns::simd<_T, _N> __res;
    __res.template bit_cast_view<_OpAlignedT>() = __ens::lsc_slm_block_load<_OpAlignedT, _NElts>(__slm_offset);
    return __res;
}

template <typename _T, int _N, typename _OpAlignedT = _LscOpAlignedT<_T>,
          int _NElts = __lsc_op_block_size<_T, _N, _OpAlignedT>(),
          ::std::enable_if_t<_NElts != __lsc_slm_block_size_rounding<_NElts>(), int> = 0>
inline __ns::simd<_T, _N>
__block_load_slm(::std::uint32_t __slm_offset)
{
    constexpr int __block_size_rounded = __lsc_slm_block_size_rounding<_NElts>();

    __ns::simd<_T, _N> __res;
    constexpr int __block_size = __lsc_op_block_size<_OpAlignedT, __block_size_rounded, _T>();
    __res.template select<__block_size, 1>(0) = __block_load_slm<_T, __block_size>(__slm_offset);
    __res.template select<_N - __block_size, 1>(__block_size) =
        __block_load_slm<_T, _N - __block_size>(__slm_offset + __block_size * sizeof(_T));
    return __res;
}

template <typename _T, int _N, typename _OpAlignedT = _LscOpAlignedT<_T>,
          int _NElts = __lsc_op_block_size<_T, _N, _OpAlignedT>(),
          ::std::enable_if_t<_NElts == __lsc_slm_block_size_rounding<_NElts>(), int> = 0>
void
__block_store_slm(::std::uint32_t __slm_offset, __ns::simd<_T, _N> __data)
{
    __ens::lsc_slm_block_store<_OpAlignedT, _NElts>(__slm_offset, __data.template bit_cast_view<::std::uint32_t>());
}

template <typename _T, int _N, typename _OpAlignedT = _LscOpAlignedT<_T>,
          int _NElts = __lsc_op_block_size<_T, _N, _OpAlignedT>(),
          ::std::enable_if_t<_NElts != __lsc_slm_block_size_rounding<_NElts>(), int> = 0>
void
__block_store_slm(::std::uint32_t __slm_offset, __ns::simd<_T, _N> __data)
{
    constexpr int __block_size_rounded = __lsc_slm_block_size_rounding<_NElts>();

    constexpr int __block_size = __lsc_op_block_size<_OpAlignedT, __block_size_rounded, _T>();
    __block_store_slm<_T, __block_size>(__slm_offset, __data.template select<__block_size, 1>(0));
    __block_store_slm<_T, _N - __block_size>(__slm_offset + __block_size * sizeof(_T),
                                             __data.template select<_N - __block_size, 1>(__block_size));
}

template <typename _T, int _NElts, ::std::enable_if_t<sizeof(_T) == sizeof(::std::uint8_t), int> = 0>
constexpr int
__lsc_block_store_size_rounding()
{
    // https://github.com/intel/llvm/blob/3dbc2c00c26e599e8a10d44e3168a45d3c496eeb/sycl/include/sycl/ext/intel/experimental/esimd/memory.hpp#L2067
    // Allowed _NElts values for 8 bit data are 4, 8, 12, 16, 32, 64, 128, 256, 512.

    static_assert(_NElts >= 1);

    if constexpr (_NElts < 8)
        return 4;

    if constexpr (_NElts < 12)
        return 8;

    if constexpr (_NElts < 16)
        return 12;

    if constexpr (_NElts < 32)
        return 16;

    if constexpr (_NElts < 64)
        return 32;

    if constexpr (_NElts < 128)
        return 64;

    if constexpr (_NElts < 256)
        return 128;

    if constexpr (_NElts < 512)
        return 256;

    return 512;
}

template <typename _T, int _NElts, ::std::enable_if_t<sizeof(_T) == sizeof(::std::uint16_t), int> = 0>
constexpr int
__lsc_block_store_size_rounding()
{
    // https://github.com/intel/llvm/blob/3dbc2c00c26e599e8a10d44e3168a45d3c496eeb/sycl/include/sycl/ext/intel/experimental/esimd/memory.hpp#L2067
    // Allowed _NElts values for 16 bit data are 2, 4, 8, 16, 32, 64, 128, 256.

    static_assert(_NElts >= 1);

    if constexpr (_NElts < 2)
        return 1;

    if constexpr (_NElts < 4)
        return 2;

    if constexpr (_NElts < 8)
        return 4;

    if constexpr (_NElts < 16)
        return 8;

    if constexpr (_NElts < 32)
        return 16;

    if constexpr (_NElts < 64)
        return 32;

    if constexpr (_NElts < 128)
        return 64;

    if constexpr (_NElts < 256)
        return 128;

    return 256;
}

template <typename _T, int _NElts, ::std::enable_if_t<sizeof(_T) == sizeof(::std::uint32_t), int> = 0>
constexpr int
__lsc_block_store_size_rounding()
{
    // https://github.com/intel/llvm/blob/3dbc2c00c26e599e8a10d44e3168a45d3c496eeb/sycl/include/sycl/ext/intel/experimental/esimd/memory.hpp#L2067
    // Allowed _NElts values for 32 bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128.

    static_assert(_NElts >= 1);

    if constexpr (_NElts < 2)
        return 1;

    if constexpr (_NElts < 3)
        return 2;

    if constexpr (_NElts < 4)
        return 3;

    if constexpr (_NElts < 8)
        return 4;

    if constexpr (_NElts < 16)
        return 8;

    if constexpr (_NElts < 32)
        return 16;

    if constexpr (_NElts < 64)
        return 32;

    if constexpr (_NElts < 128)
        return 64;

    return 128;
}

template <typename _T, int _NElts, ::std::enable_if_t<sizeof(_T) == sizeof(::std::uint64_t), int> = 0>
constexpr int
__lsc_block_store_size_rounding()
{
    // https://github.com/intel/llvm/blob/3dbc2c00c26e599e8a10d44e3168a45d3c496eeb/sycl/include/sycl/ext/intel/experimental/esimd/memory.hpp#L2067
    // Allowed _NElts values for 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64.

    static_assert(_NElts >= 1);

    if constexpr (_NElts < 2)
        return 1;

    if constexpr (_NElts < 3)
        return 2;

    if constexpr (_NElts < 4)
        return 3;

    if constexpr (_NElts < 8)
        return 4;

    if constexpr (_NElts < 16)
        return 8;

    if constexpr (_NElts < 32)
        return 16;

    if constexpr (_NElts < 64)
        return 32;

    return 64;
}

template <typename _T, int _N, __ens::cache_hint _H1 = __ens::cache_hint::none,
          __ens::cache_hint _H3 = __ens::cache_hint::none,
          ::std::enable_if_t<_N == __lsc_block_store_size_rounding<_T, _N>(), int> = 0>
inline void
__block_store(_T* __dst, __ns::simd<_T, _N> __data)
{
    // https://github.com/intel/llvm/blob/3dbc2c00c26e599e8a10d44e3168a45d3c496eeb/sycl/include/sycl/ext/intel/experimental/esimd/memory.hpp#L2067
    // Allowed _NElts values for  8 bit data are          4, 8, 12, 16, 32, 64, 128, 256, 512.
    // Allowed _NElts values for 16 bit data are    2,    4, 8,     16, 32, 64, 128, 256.
    // Allowed _NElts values for 32 bit data are 1, 2, 3, 4, 8,     16, 32, 64, 128.
    // Allowed _NElts values for 64 bit data are 1, 2, 3, 4, 8,     16, 32, 64.
    __ens::lsc_block_store<::std::uint32_t, _N, __ens::lsc_data_size::default_size, _H1, _H3>(
        __dst, __data.template bit_cast_view<::std::uint32_t>(), 1);
}

template <typename _T, int _N, __ens::cache_hint _H1 = __ens::cache_hint::none,
          __ens::cache_hint _H3 = __ens::cache_hint::none,
          ::std::enable_if_t<_N != __lsc_block_store_size_rounding<_T, _N>(), int> = 0>
inline void
__block_store(_T* __dst, __ns::simd<_T, _N> __data)
{
    constexpr ::std::uint32_t __block_size = 64 * sizeof(::std::uint32_t) / sizeof(_T);

    constexpr int __block_size_rounded = __lsc_block_store_size_rounding<_T, _N>();
    static_assert(__block_size == __block_size_rounded);

    __block_store<_T, __block_size>(__dst, __data.template select<__block_size, 1>(0));
    __block_store<_T, _N - __block_size>(__dst + __block_size,
                                         __data.template select<_N - __block_size, 1>(__block_size));
}

} // namespace __dpl_esimd

#endif // _ONEDPL_KT_ESIMD_DEFS_H
