// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>

#include "sycl_defs.h"
#include "parallel_backend_sycl.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

#include "../../segmented_reduce_utils.h"
#include "../../utils.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// reduce_by_segment with segments defined by offsets

// Writes the initial value into every result: all the segments are empty
template <typename _Acc, typename _InitType>
struct __segmented_reduce_init_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _InitType __init;

    // __result is a zip view of the result only, so that the segments are not vectorized over
    template <typename _IsFull, typename _Params, typename _Result>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Result __result) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");
        using std::get;
        get<0>(__result[__idx]) = _Acc(__init);
    }
};

template <typename _Name>
class __segmented_reduce_fill_wrapper;

// One work-group reduces one segment, as with __work_group_reduce_kernel of the reduction, whose bricks
// (unseq_backend::transform_reduce and unseq_backend::reduce_over_group) it uses: the operation is required to be
// commutative, so the work-items load the segment coalesced. The bricks address the input by the global position of
// a work-item; the work-group of a segment is presented to them as the only one of the kernel, and its segment as the
// whole input.
class __segmented_reduce_group_item
{
    sycl::nd_item<1> __item;

  public:
    explicit __segmented_reduce_group_item(const sycl::nd_item<1>& __nd_item) : __item(__nd_item) {}

    std::size_t
    get_global_id(int) const
    {
        return __item.get_local_id(0);
    }
    std::size_t
    get_local_id(int) const
    {
        return __item.get_local_id(0);
    }
    std::size_t
    get_local_range(int) const
    {
        return __item.get_local_range(0);
    }
    sycl::range<1>
    get_local_range() const
    {
        return __item.get_local_range();
    }
    std::size_t
    get_group(int) const
    {
        return 0;
    }
    std::size_t
    get_group_linear_id() const
    {
        return 0;
    }
    sycl::group<1>
    get_group() const
    {
        return __item.get_group();
    }
};

// The elements of a segment of at least two elements, taken in pairs by the bricks of the reduction. The first pair
// starts from the operation on the initial value and the first element, as in std::ranges::fold_left; every other
// pair from the operation on its two elements, as a partial result of std::reduce does, so that the accumulator is
// never made from an element alone, which may mean something else than the element to the operation. With an odd
// length the last pair also takes the last element.
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_pairs
{
    _Size __begin;           // the first element of the segment
    std::size_t __num_pairs; // the number of pairs
    bool __is_odd;           // whether the length of the segment is odd
    _InitType __init;
    _BinaryOperation __binary_op;

    template <typename _Input>
    _Acc
    operator()(const std::size_t __idx, const _Input& __input) const
    {
        const _Size __first = __begin + 2 * _Size(__idx);
        _Acc __acc = __idx == 0 ? _Acc(std::invoke(__binary_op, __init, __input[__first]))
                                : _Acc(std::invoke(__binary_op, __input[__first], __input[__first + 1]));
        if (__idx == 0)
            __acc = std::invoke(__binary_op, std::move(__acc), __input[__first + 1]);
        if (__is_odd && __idx + 1 == __num_pairs)
            __acc = std::invoke(__binary_op, std::move(__acc), __input[__first + 2]);
        return __acc;
    }
};

template <typename... _Name>
class __segmented_reduce_group_kernel;

template <typename _Acc, typename _KernelName>
struct __segmented_reduce_group_submitter;

template <typename _Acc, typename... _Name>
struct __segmented_reduce_group_submitter<_Acc, __internal::__optional_kernel_name<_Name...>>
{
    // As in __parallel_transform_reduce
    static constexpr std::uint8_t __vector_size = 4;

    template <typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size, typename _InitType,
              typename _BinaryOperation>
    __future<sycl::event>
    operator()(sycl::queue& __q, _InRange&& __input, _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n,
               const _Size __m, const std::uint16_t __work_group_size, _InitType __init,
               _BinaryOperation __binary_op) const
    {
        using _Pairs = __segmented_reduce_pairs<_Acc, _Size, _InitType, _BinaryOperation>;
        using _TransformPattern = unseq_backend::transform_reduce<_BinaryOperation, _Pairs, _Acc,
                                                                  /*_Commutative=*/std::true_type, __vector_size>;
        using _ReducePattern = unseq_backend::reduce_over_group<_BinaryOperation, _Acc>;

        const std::size_t __local_mem_size = _ReducePattern{__binary_op}.local_mem_req(__work_group_size);
        // One work-group per segment. A launch that would not fit the int range of the work-item indices, which a SYCL
        // implementation may assume (as DPC++ does by default), has fewer work-groups, each reducing every
        // __num_groups-th segment.
        const std::size_t __num_groups =
            std::min<std::size_t>(__m, std::size_t(std::numeric_limits<int>::max()) / __work_group_size);
        auto __event = __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
            __dpl_sycl::__local_accessor<_Acc> __local_mem(sycl::range<1>(__local_mem_size), __cgh);
            __cgh.parallel_for<_Name...>(
                sycl::nd_range<1>(sycl::range<1>(__num_groups * __work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __nd_item) {
                    const std::size_t __local_idx = __nd_item.get_local_id(0);
                    const __segmented_reduce_group_item __item(__nd_item);
                    for (std::size_t __group = __nd_item.get_group(0); __group < std::size_t(__m);
                         __group += __num_groups)
                    {
                        // The local memory is no longer read for the previous segment
                        if (__group != __nd_item.get_group(0))
                            sycl::group_barrier(__nd_item.get_group());

                        const _Size __segment = __group;
                        const _Size __begin =
                            oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment], __n);
                        const _Size __end =
                            oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment + 1], __n);
                        const _Size __length = __begin < __end ? __end - __begin : _Size{0};

                        // The same for every work-item of the work-group, so none of them reaches a barrier below
                        if (__length < 2)
                        {
                            if (__local_idx == 0)
                                __result[__segment] = __length == 0
                                                          ? _Acc(__init)
                                                          : _Acc(std::invoke(__binary_op, __init, __input[__begin]));
                            continue;
                        }

                        // The bricks of the reduction take unsigned sizes
                        const std::size_t __num_pairs = __length / 2;
                        const _TransformPattern __transform_pattern{
                            __binary_op, _Pairs{__begin, __num_pairs, __length % 2 != 0, __init, __binary_op}};
                        const _ReducePattern __reduce_pattern{__binary_op};

                        // The iterations of a work-item as in __parallel_transform_reduce
                        const std::size_t __iters_per_work_item = __adjust_iters_per_work_item<__vector_size>(
                            oneapi::dpl::__internal::__dpl_ceiling_div(__num_pairs, std::size_t(__work_group_size)));
                        const bool __is_full = __num_pairs == __work_group_size * __iters_per_work_item;

                        // 1. The partial results of the work-items
                        oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __partial;
                        __transform_pattern(__item, __num_pairs, __iters_per_work_item,
                                            /*__global_offset*/ std::size_t{0}, __is_full,
                                            /*__n_groups*/ std::size_t{1}, __partial, __input);
                        const std::size_t __n_items = __transform_pattern.output_size(
                            __num_pairs, std::size_t(__work_group_size), __iters_per_work_item);

                        // 2. Their reduction within the work-group
                        _Acc __reduced = __reduce_pattern(__item, __n_items, __partial, __local_mem);
                        if (__local_idx < __n_items)
                        {
                            if (__local_idx == 0)
                                __result[__segment] = std::move(__reduced);
                            __partial.__destroy();
                        }
                    }
                });
        });
        return __future{std::move(__event)};
    }
};

// Writes the initial value into every result: the input is empty
template <typename _Acc, typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Size,
          typename _InitType>
void
__parallel_reduce_by_segment_offsets_fill(_BackendTag, _ExecutionPolicy&& __exec, _Range&& __result, _Size __m,
                                          const _InitType& __init)
{
    // A separate kernel name: the reduction kernel of the same call is instantiated too
    __parallel_for(_BackendTag{},
                   make_wrapped_policy<__segmented_reduce_fill_wrapper>(std::forward<_ExecutionPolicy>(__exec)),
                   __segmented_reduce_init_brick<_Acc, _InitType>{__init}, __m,
                   oneapi::dpl::__ranges::make_zip_view(std::forward<_Range>(__result)))
        .__checked_deferrable_wait();
}

// The segments of the input [0, __n) defined by the __m + 1 offsets, reduced into the __m results, one work-group per
// segment
template <typename _Acc, typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange,
          typename _OutRange, typename _Size, typename _InitType, typename _BinaryOperation>
void
__parallel_reduce_by_segment_offsets(_BackendTag, _ExecutionPolicy&& __exec, _InRange&& __input,
                                     _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n, const _Size __m,
                                     _InitType __init, _BinaryOperation __binary_op)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    using _KernelName = __internal::__kernel_name_provider<__segmented_reduce_group_kernel<_CustomName>>;

    sycl::queue __q = __exec.queue();

    // The work-group size as in __parallel_transform_reduce: adjusted to the local memory limit, which is doubled to
    // take into account the memory used by the compiled kernel, and at most 256
    const std::size_t __work_group_size = std::min<std::size_t>(
        oneapi::dpl::__internal::__slm_adjusted_work_group_size(__q, std::size_t(sizeof(_Acc) * 2)), 256);

    __segmented_reduce_group_submitter<_Acc, _KernelName>()(
        __q, std::forward<_InRange>(__input), std::forward<_OffsetRange>(__offsets), std::forward<_OutRange>(__result),
        __n, __m, static_cast<std::uint16_t>(__work_group_size), __init, __binary_op)
        .__checked_deferrable_wait();
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
