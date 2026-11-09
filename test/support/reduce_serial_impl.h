// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _REDUCE_SERIAL_IMPL_H
#define _REDUCE_SERIAL_IMPL_H

#include <algorithm>   // std::max, std::min
#include <iterator>    // std::iterator_traits
#include <type_traits> // std::common_type_t, std::decay_t, std::invoke_result_t
#include <utility>     // std::move

template <typename RandAccessItKeysIn, typename RandAccessItValsIn, typename RandAccessItKeysOut,
          typename RandAccessItValsOut, typename Size, typename BinaryPredicate, typename BinaryOperation>
Size
reduce_by_segment_serial(RandAccessItKeysIn keys, RandAccessItValsIn vals,
                         RandAccessItKeysOut res_keys, RandAccessItValsOut res_vals, Size n,
                         BinaryPredicate binary_pred, BinaryOperation binary_op)
{
    if (n < 1)
        return 0;
    
    using ValT = ::std::decay_t<decltype(vals[0])>;
    using KeyT = ::std::decay_t<decltype(keys[0])>;
    KeyT tmp_key = keys[0];
    ValT tmp_val = vals[0];
    Size segment_count = 0;
    for (Size i = 1; i < n; ++i)
    {
        if (binary_pred(keys[i - 1], keys[i]))
        {
            tmp_val = binary_op(tmp_val, vals[i]);
        }
        else
        {
            res_keys[segment_count] = tmp_key;
            res_vals[segment_count] = tmp_val;
            segment_count++;
            tmp_key = keys[i];
            tmp_val = vals[i];
        }
    }
    res_keys[segment_count] = tmp_key;
    res_vals[segment_count] = tmp_val;

    return segment_count + 1;
}

// Reference of reduce_by_segment with segments defined by offsets: the segment s is [offsets[s], offsets[s + 1])
// clamped to the input [0, last - first), and a non-positive num_segments writes nothing.
template <typename RandAccessItIn, typename RandAccessItOffsets, typename RandAccessItOut, typename SegmentNum,
          typename Init, typename BinaryOperation>
RandAccessItOut
reduce_by_segment_offsets_serial(RandAccessItIn first, RandAccessItIn last, RandAccessItOffsets offsets,
                                 SegmentNum num_segments, Init init_value, BinaryOperation binary_op,
                                 RandAccessItOut result)
{
    using AccT = std::decay_t<
        std::invoke_result_t<BinaryOperation&, Init, typename std::iterator_traits<RandAccessItIn>::reference>>;
    using OffsetType = typename std::iterator_traits<RandAccessItOffsets>::value_type;
    using SizeT = typename std::iterator_traits<RandAccessItIn>::difference_type;
    using CommonIndexT = std::common_type_t<OffsetType, SizeT>;

    const SizeT n = last - first;
    SegmentNum s = 0;
    for (; s < num_segments; ++s)
    {
        const OffsetType s_first_idx = std::max<OffsetType>(offsets[s], OffsetType(0));
        const OffsetType s_last_idx = std::max<OffsetType>(std::min<CommonIndexT>(offsets[s + 1], n), OffsetType(0));

        AccT acc = init_value;
        for (OffsetType i = s_first_idx; i < s_last_idx; ++i)
            acc = binary_op(std::move(acc), first[i]);
        result[s] = std::move(acc);
    }
    return result + s;
}

#endif //_REDUCE_SERIAL_IMPL_H
