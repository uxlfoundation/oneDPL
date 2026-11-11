// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_SEGMENTED_REDUCE_UTILS_H
#define _ONEDPL_SEGMENTED_REDUCE_UTILS_H

#include <algorithm>   // std::max, std::min
#include <functional>  // std::invoke
#include <iterator>    // std::iterator_traits
#include <type_traits> // std::decay_t, std::invoke_result_t, std::common_type_t, std::is_integral_v
#include <utility>     // std::move

#include "onedpl_config.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{

// The helpers of reduce_by_segment with segments defined by offsets, shared by the host and the device implementations

template <typename _InitType, typename _InputIterator, typename _BinaryOperation>
using __segmented_reduce_acc_t = std::decay_t<
    std::invoke_result_t<_BinaryOperation&, _InitType, typename std::iterator_traits<_InputIterator>::reference>>;

// Clamps an integral value to [0, n], also for a signed value and an unsigned n or the other way round
template <typename _Tp, typename _Size>
constexpr _Size
__clamp_to_size(_Tp __value, _Size __n)
{
    static_assert(std::is_integral_v<_Tp> && std::is_integral_v<_Size>, "The value and the bound must be integral");

    using _Common = std::common_type_t<_Tp, _Size>;
    return static_cast<_Size>(std::min<_Common>(std::max<_Tp>(__value, 0), static_cast<_Common>(__n)));
}

// Folds the elements [__begin, __end) of a random access iterator or a range from left to right into an accumulator
// made from the initial value, as std::accumulate does; an empty range gives the initial value converted to the
// accumulator type
template <typename _Acc, typename _Source, typename _Index, typename _InitType, typename _BinaryOperation>
_Acc
__serial_accumulate(const _Source& __src, _Index __begin, _Index __end, _InitType __init, _BinaryOperation& __binary_op)
{
    _Acc __acc = std::move(__init);

    // Unrolled on a device, where a work-item reducing alone would wait for every load
#if defined(__SYCL_DEVICE_ONLY__)
    _ONEDPL_PRAGMA(unroll 16)
#endif
    for (; __begin < __end; ++__begin)
        __acc = std::invoke(__binary_op, std::move(__acc), __src[__begin]);

    return __acc;
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_SEGMENTED_REDUCE_UTILS_H
