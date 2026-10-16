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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

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
// clamped to the input [0, last - first), reduced as std::ranges::fold_left from init_value; an empty segment gives
// init_value converted to the accumulator type, and a non-positive num_segments writes nothing.
template <typename RandAccessItIn, typename RandAccessItOffsets, typename RandAccessItOut, typename SegmentNum,
          typename Init, typename BinaryOperation>
void
reduce_by_segment_offsets_serial(RandAccessItIn first, RandAccessItIn last, RandAccessItOffsets offsets,
                                 SegmentNum num_segments, Init init_value, BinaryOperation binary_op,
                                 RandAccessItOut result)
{
    // The accumulator type of the algorithm
    using AccT = std::decay_t<
        std::invoke_result_t<BinaryOperation&, Init, typename std::iterator_traits<RandAccessItIn>::reference>>;
    using Size = typename std::iterator_traits<RandAccessItIn>::difference_type;

    const Size n = last - first;
    auto clamp = [n](auto offset) -> Size {
        if constexpr (std::is_signed_v<decltype(offset)>)
        {
            if (offset < 0)
                return 0;
        }
        return std::uint64_t(offset) < std::uint64_t(n) ? Size(offset) : n;
    };
    for (SegmentNum s = 0; s < num_segments; ++s)
    {
        const Size begin = clamp(offsets[s]);
        const Size end = std::max(begin, clamp(offsets[s + 1]));
        // A left fold with an AccT accumulator, which combines init_value itself with the first element
        if (begin == end)
        {
            result[s] = AccT(init_value);
            continue;
        }
        AccT acc(binary_op(init_value, first[begin]));
        for (Size i = begin + 1; i < end; ++i)
            acc = binary_op(std::move(acc), first[i]);
        result[s] = std::move(acc);
    }
}

#endif //_REDUCE_SERIAL_IMPL_H
