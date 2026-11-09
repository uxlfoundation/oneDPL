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
#include <type_traits> // std::decay_t, std::invoke_result_t, std::common_type_t, std::is_integral_v, std::is_same_v
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

// Reduces the elements [__begin, __end) of a random access iterator or a range into an accumulator made from the
// initial value; an empty range gives the initial value converted to the accumulator type
template <typename _Acc, typename _Source, typename _Index, typename _InitType, typename _BinaryOperation>
_Acc
__segmented_reduce_serial(const _Source& __src, _Index __begin, _Index __end, _InitType __init,
                          _BinaryOperation& __binary_op)
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

// Ensure the offset is within [0, n]
template <typename _Offset, typename _Size>
constexpr _Size
__segmented_reduce_clamp_offset(_Offset __offset, _Size __n)
{
    static_assert(std::is_integral_v<_Offset> && !std::is_same_v<_Offset, bool>,
                  "The offsets of reduce_by_segment must be of an integral type other than bool");

    using _Common = std::common_type_t<_Offset, _Size>;
    return static_cast<_Size>(std::min<_Common>(std::max<_Offset>(__offset, 0), __n));
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_SEGMENTED_REDUCE_UTILS_H
