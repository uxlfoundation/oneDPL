// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_REDUCE_BY_SEGMENT_IMPL_HETERO_H
#define _ONEDPL_REDUCE_BY_SEGMENT_IMPL_HETERO_H

#include <utility> // std::pair, std::make_pair, std::forward

#include "numeric_impl_hetero.h" // __pattern_reduce_by_segment, __pattern_reduce_by_segment_offsets

namespace oneapi
{
namespace dpl
{
namespace internal
{

// The device overloads of reduce_by_segment: the key-based one and the one of the segments defined by offsets. The
// overloads of the host policies are in internal/reduce_by_segment_impl.h, which calls them all by the qualified name
// and includes this header first.
template <typename _BackendTag, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator1, typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
std::pair<OutputIterator1, OutputIterator2>
reduce_by_segment_impl(__internal::__hetero_tag<_BackendTag> __tag, Policy&& policy, InputIterator1 first1,
                       InputIterator1 last1, InputIterator2 first2, OutputIterator1 result1, OutputIterator2 result2,
                       BinaryPred binary_pred, BinaryOperator binary_op)
{
    // The algorithm reduces values in [first2, first2 + (last1-first1)) where the associated
    // keys for the values are equal to the adjacent key.
    //
    // Example: keys          = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first1, last1)
    //          values        = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first2, first2+n)
    //
    //          keys_result   = { 1, 2, 3, 4, 1, 3, 1, 3, 0 } -- result1
    //          values_result = { 1, 2, 3, 4, 2, 6, 2, 6, 0 } -- result2

    if (first1 == last1)
        return std::make_pair(result1, result2);

    // number of unique keys
    const auto __n = oneapi::dpl::__internal::__pattern_reduce_by_segment(
        __tag, std::forward<Policy>(policy), first1, last1, first2, result1, result2, binary_pred, binary_op);

    return std::make_pair(result1 + __n, result2 + __n);
}

template <typename _BackendTag, typename _Policy, typename _RandomAccessInputIt, typename _SegmentNum,
          typename _RandomAccessOffsetIt, typename _RandomAccessOutputIt, typename _InitType, typename _BinaryOperation>
void
__reduce_by_segment_offsets_impl(__internal::__hetero_tag<_BackendTag> __tag, _Policy&& __policy,
                                 _RandomAccessInputIt __first_value, _RandomAccessInputIt __last_value,
                                 _RandomAccessOutputIt __result_value, _SegmentNum __num_segments,
                                 _RandomAccessOffsetIt __offsets, _InitType __init_value, _BinaryOperation __binary_op)
{
    oneapi::dpl::__internal::__pattern_reduce_by_segment_offsets(__tag, std::forward<_Policy>(__policy), __first_value,
                                                                 __last_value, __result_value, __num_segments,
                                                                 __offsets, __init_value, __binary_op);
}

} // namespace internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_REDUCE_BY_SEGMENT_IMPL_HETERO_H
