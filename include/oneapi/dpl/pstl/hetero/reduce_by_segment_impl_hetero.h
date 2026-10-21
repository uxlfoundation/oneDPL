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

#include <utility> // std::forward

#include "numeric_impl_hetero.h" // __pattern_reduce_by_segment_offsets

namespace oneapi
{
namespace dpl
{
namespace internal
{

// The device overload of reduce_by_segment with segments defined by offsets. The overloads of the host policies
// are in internal/reduce_by_segment_impl.h, which calls it by the qualified name and includes this header first.
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
