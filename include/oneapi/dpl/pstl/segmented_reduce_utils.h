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

#include <iterator>
#include <type_traits>

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

template <typename _Offset, typename _Size>
constexpr _Size
__segmented_reduce_clamp_offset(_Offset __offset, _Size __n)
{
    static_assert(std::is_integral_v<_Offset>, "The offsets of reduce_by_segment must be of an integral type");

    using _Unsigned = std::common_type_t<std::make_unsigned_t<_Offset>, std::make_unsigned_t<_Size>>;

    if (__offset <= 0)
        return 0;
    return static_cast<_Unsigned>(__offset) < static_cast<_Unsigned>(__n) ? static_cast<_Size>(__offset) : __n;
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_SEGMENTED_REDUCE_UTILS_H
