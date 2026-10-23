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

// The accumulator made from the initial value of an empty segment or from a result of the operation: by an implicit
// conversion, as the RFC requires both to be convertible to the accumulator type
template <typename _Acc, typename _Value>
_Acc
__segmented_reduce_to_acc(_Value&& __value)
{
    return std::forward<_Value>(__value);
}

// Reduces the elements [__begin, __end) of a random access iterator or a range as std::ranges::fold_left does: the
// initial value itself is combined with the first element, so that the result is a generalized sum of the initial
// value and the elements; an empty range gives the initial value converted to the accumulator type
template <typename _Acc, typename _Source, typename _Index, typename _InitType, typename _BinaryOperation>
_Acc
__segmented_reduce_serial(const _Source& __src, _Index __begin, _Index __end, _InitType __init,
                          _BinaryOperation& __binary_op)
{
    if (__begin >= __end)
        return __segmented_reduce_to_acc<_Acc>(std::move(__init));
    _Acc __acc = __segmented_reduce_to_acc<_Acc>(std::invoke(__binary_op, std::move(__init), __src[__begin]));
    // Unrolled on a device, where a work-item reducing alone would wait for every load
#if defined(__SYCL_DEVICE_ONLY__)
    _ONEDPL_PRAGMA(unroll 16)
#endif
    for (++__begin; __begin < __end; ++__begin)
        __acc = std::invoke(__binary_op, std::move(__acc), __src[__begin]);
    return __acc;
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_SEGMENTED_REDUCE_UTILS_H
