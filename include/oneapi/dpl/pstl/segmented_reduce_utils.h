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

#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

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

// Reduces the elements [__begin, __end) of a random access iterator or a range as std::ranges::fold_left does: the
// initial value itself is combined with the first element, so that the result is a generalized sum of the initial
// value and the elements; an empty range gives the initial value converted to the accumulator type
template <typename _Acc, typename _Source, typename _Index, typename _InitType, typename _BinaryOperation>
_Acc
__segmented_reduce_serial(const _Source& __src, _Index __begin, _Index __end, _InitType __init,
                          _BinaryOperation& __binary_op)
{
    if (__begin >= __end)
        return std::move(__init);

    _Acc __acc = std::invoke(__binary_op, std::move(__init), __src[__begin]);

    // Unrolled on a device, where a work-item reducing alone would wait for every load
#if defined(__SYCL_DEVICE_ONLY__)
    _ONEDPL_PRAGMA(unroll 16)
#endif
    for (++__begin; __begin < __end; ++__begin)
        __acc = std::invoke(__binary_op, std::move(__acc), __src[__begin]);

    return __acc;
}

template <typename _Offset, typename _Size>
constexpr _Size
__segmented_reduce_clamp_offset(_Offset __offset, _Size __n)
{
    constexpr bool __is_offset = std::is_integral_v<_Offset> && !std::is_same_v<_Offset, bool>;
    static_assert(__is_offset, "The offsets of reduce_by_segment must be of an integral type other than bool");

    // An int in place of an offset of another type keeps the static_assert the only error
    using _Unsigned = std::common_type_t<std::make_unsigned_t<std::conditional_t<__is_offset, _Offset, int>>,
                                         std::make_unsigned_t<_Size>>;

    if (__offset <= 0)
        return 0;

    return static_cast<_Unsigned>(__offset) < static_cast<_Unsigned>(__n) ? static_cast<_Size>(__offset) : __n;
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_SEGMENTED_REDUCE_UTILS_H
