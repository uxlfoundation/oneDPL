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
#include <cmath>
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

#include "../../functional_impl.h" // oneapi::dpl::maximum
#include "../../segmented_reduce_utils.h"
#include "../../utils.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// reduce_by_segment with segments defined by offsets

// One work-item per short segment. A longer segment is left to the reduction in chunks and only reports its length,
// capped as only its square root matters, through an atomic maximum.
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_variable_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _Size __n;         // the size of the input
    _Size __threshold; // the length of the longest short segment
    _InitType __init;
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _Offsets, typename _Result, typename _Input,
              typename _MaxLength>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Offsets __offsets, _Result __result, _Input __input,
               _MaxLength __max_length) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");

        const _Size __begin = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__idx], __n);
        const _Size __end = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__idx + 1], __n);
        if (__end - __begin > __threshold)
        {
            sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                __ref(__max_length[0]);
            __ref.fetch_max(static_cast<std::uint32_t>(
                std::min<_Size>(__end - __begin, std::numeric_limits<std::uint32_t>::max())));
            return;
        }
        std::get<0>(__result[__idx]) =
            oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(__input, __begin, __end, __init, __binary_op);
    }
};

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
        get<0>(__result[__idx]) = oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init);
    }
};

template <typename _Name>
class __segmented_reduce_fill_wrapper;

// A short segment is reduced by one work-item. A long segment is split into chunks, each reduced by one work-item, and
// the partial results of the segment are combined by one work-item. The values below were measured on Intel GPUs
// (Data Center GPU Max 1100, integrated graphics) and CPUs (Xeon Platinum 8480+, Xeon 696X, Core Ultra 7 165U):
// - a segment up to this length is short;
inline constexpr std::size_t __segmented_reduce_gpu_threshold = 512;
inline constexpr std::size_t __segmented_reduce_cpu_threshold = 32768;
// - on GPUs the best chunk length is close to the square root of the longest segment, which balances the serial work
//   of the two levels; on CPUs the chunk length hardly matters above 1024.
inline constexpr std::size_t __segmented_reduce_cpu_chunk_size = 4096;

inline std::size_t
__segmented_reduce_chunk_size(bool __is_gpu, std::size_t __max_length)
{
    if (!__is_gpu)
        return __segmented_reduce_cpu_chunk_size;
    const auto __root = static_cast<std::size_t>(std::sqrt(static_cast<double>(__max_length)));
    return std::clamp<std::size_t>(oneapi::dpl::__internal::__dpl_bit_ceil(__root), 64, 16384);
}

// The length of a segment within the input; a negatively sized segment is empty
template <typename _Size>
struct __segmented_reduce_length
{
    _Size __n; // the size of the input

    template <typename _OffsetEnd, typename _OffsetBegin>
    _Size
    operator()(_OffsetEnd __end, _OffsetBegin __begin) const
    {
        const _Size __b = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__begin, __n);
        const _Size __e = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__end, __n);
        return __b < __e ? __e - __b : _Size{0};
    }
};

// The number of chunks of a segment given by the pair of its offsets
template <typename _Size>
struct __segmented_reduce_chunk_count
{
    _Size __n;          // the size of the input
    _Size __chunk_size; // the length of a chunk
    _Size __min_length; // the segments not longer than this have no chunks

    template <typename _Bounds>
    _Size
    operator()(const _Bounds& __bounds) const
    {
        const _Size __length = __segmented_reduce_length<_Size>{__n}(std::get<1>(__bounds), std::get<0>(__bounds));
        return __length > __min_length ? (__length + __chunk_size - 1) / __chunk_size : _Size{0};
    }
};

// Reduces one chunk of a long segment. The first chunk of a segment takes the remainder of its length and starts
// from the operation on the initial value and its first element; every other chunk has exactly __chunk_size elements
// and starts from the operation on its first two elements.
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_chunk_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _Size __n;            // the size of the input
    _Size __num_segments; // the number of segments
    _Size __chunk_size;   // the length of a chunk
    _InitType __init;
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _ChunkEnds, typename _Offsets, typename _Partials,
              typename _Input>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _ChunkEnds __chunk_ends, _Offsets __offsets,
               _Partials __partials, _Input __input) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");

        const _Size __chunk = __idx;
        if (__chunk >= __chunk_ends[__num_segments - 1])
            return;

        // The segment of the chunk: the first one whose chunks end after it
        _Size __lo = 0, __hi = __num_segments - 1;
        while (__lo < __hi)
        {
            const _Size __mid = __lo + (__hi - __lo) / 2;
            if (__chunk_ends[__mid] > __chunk)
                __hi = __mid;
            else
                __lo = __mid + 1;
        }
        const _Size __first_chunk = __lo == 0 ? _Size{0} : _Size(__chunk_ends[__lo - 1]);
        const _Size __begin = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__lo], __n);
        const _Size __end = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__lo + 1], __n);
        const _Size __head = (__end - __begin) - (_Size(__chunk_ends[__lo]) - __first_chunk - 1) * __chunk_size;

        // One loop for every chunk, so that the work-items of a sub-group do not diverge into two long loops
        const _Size __k = __chunk - __first_chunk;
        const _Size __chunk_begin = __k == 0 ? __begin : __begin + __head + (__k - 1) * __chunk_size;
        const _Size __chunk_end = __k == 0 ? __begin + __head : __chunk_begin + __chunk_size;
        // A chunk other than the first one starts from the operation on its first two elements, as a partial result
        // of std::reduce does: the accumulator is never made from an element alone, which may mean something else than
        // the element to the operation (such a chunk has at least 64 elements)
        _Acc __start = __k == 0 ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                      std::invoke(__binary_op, __init, __input[__chunk_begin]))
                                : oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                      std::invoke(__binary_op, __input[__chunk_begin], __input[__chunk_begin + 1]));
        std::get<0>(__partials[__idx]) = oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(
            __input, __chunk_begin + (__k == 0 ? 1 : 2), __chunk_end, std::move(__start), __binary_op);
    }
};

// Combines the partial results of the chunks of every segment
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_combine_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _Size __num_partials; // the number of the computed partial results
    bool __write_empty;   // whether a segment without chunks gets the initial value; otherwise it is left as is
    _InitType __init;
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _ChunkEnds, typename _Partials, typename _Result>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _ChunkEnds __chunk_ends, _Partials __partials,
               _Result __result) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");

        const _Size __first = __idx == 0 ? _Size{0} : _Size(__chunk_ends[__idx - 1]);
        // A segment without chunks is empty, or short and already reduced. Unordered offsets may give more chunks than
        // computed: their results are unspecified.
        const _Size __last = std::min<_Size>(__chunk_ends[__idx], __num_partials);
        if (__first >= __last)
        {
            if (__write_empty)
                std::get<0>(__result[__idx]) = oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init);
            return;
        }
        std::get<0>(__result[__idx]) = oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(
            __partials, __first + 1, __last, _Acc(__partials[__first]), __binary_op);
    }
};

template <typename _Name>
class __segmented_reduce_length_wrapper;
template <typename _Name>
class __segmented_reduce_scan_wrapper;
template <typename _Name>
class __segmented_reduce_chunk_wrapper;
template <typename _Name>
class __segmented_reduce_combine_wrapper;

enum class __segmented_reduce_dispatch_strategy
{
    __thread_per_segment,
    __sub_group,
    __work_group,
    __two_work_groups,
    __sixteen_work_groups,
    __grid
};

// UNTUNED: provisional inclusive upper bounds for the invocation-wide average N / M, not individual lengths.
inline constexpr std::size_t __segmented_reduce_dispatch_thread_threshold = 16;
inline constexpr std::size_t __segmented_reduce_dispatch_sub_group_threshold = 256;
inline constexpr std::size_t __segmented_reduce_dispatch_work_group_threshold = 8192;
inline constexpr std::size_t __segmented_reduce_dispatch_two_work_groups_threshold = 65536;
inline constexpr std::size_t __segmented_reduce_dispatch_sixteen_work_groups_threshold = 1048576;

constexpr __segmented_reduce_dispatch_strategy
__segmented_reduce_dispatch(std::size_t __n, std::size_t __m)
{
    const std::size_t __average = __m == 0 ? 0 : __n / __m;
    return __average <= __segmented_reduce_dispatch_thread_threshold
               ? __segmented_reduce_dispatch_strategy::__thread_per_segment
           : __average <= __segmented_reduce_dispatch_sub_group_threshold
               ? __segmented_reduce_dispatch_strategy::__sub_group
           : __average <= __segmented_reduce_dispatch_work_group_threshold
               ? __segmented_reduce_dispatch_strategy::__work_group
           : __average <= __segmented_reduce_dispatch_two_work_groups_threshold
               ? __segmented_reduce_dispatch_strategy::__two_work_groups
           : __average <= __segmented_reduce_dispatch_sixteen_work_groups_threshold
               ? __segmented_reduce_dispatch_strategy::__sixteen_work_groups
               : __segmented_reduce_dispatch_strategy::__grid;
}

#if _ONEDPL_USE_GROUP_ALGOS && _ONEDPL_USE_SUB_GROUPS
template <typename... _Names>
class __segmented_reduce_dispatch_name;
template <typename... _Names>
class __segmented_reduce_dispatch_thread_kernel;
template <typename... _Names>
class __segmented_reduce_dispatch_sub_group_kernel;
template <typename... _Names>
class __segmented_reduce_dispatch_group_kernel;
template <typename... _Names>
class __segmented_reduce_dispatch_combine_kernel;

// One work-group writes the final result; multiple work-groups write disjoint, init-free partials.
template <typename _Name, typename _Acc, typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size,
          typename _InitType, typename _BinaryOperation, typename _WritePartials>
sycl::event
__segmented_reduce_submit_groups(sycl::queue& __q, _InRange __input, _OffsetRange __offsets, _OutRange __destination,
                                 _Size __n, std::size_t __groups, std::size_t __wg_size,
                                 std::size_t __groups_per_segment, _InitType __init, _BinaryOperation __binary_op,
                                 _WritePartials)
{
    using _GroupName = __segmented_reduce_dispatch_name<_Name, _OutRange, _WritePartials>;
    using _GroupKernel = __internal::__kernel_name_generator<__segmented_reduce_dispatch_group_kernel, _GroupName>;
    const _Acc __identity = sycl::known_identity<_BinaryOperation, _Acc>::value;
    return __q.submit([&](sycl::handler& __cgh) {
        oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __destination);
        __cgh.parallel_for<_GroupKernel>(
            sycl::nd_range<1>{__groups * __wg_size, __wg_size}, [=](sycl::nd_item<1> __item) {
                const std::size_t __group = __item.get_group(0);
                const std::size_t __segment = __group / __groups_per_segment;
                const std::uint32_t __part = __group % __groups_per_segment;
                const std::uint32_t __lane = __item.get_local_id(0);
                const std::uint32_t __lanes = __item.get_local_range(0);
                const std::size_t __begin =
                    oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment], __n);
                const std::size_t __end =
                    oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment + 1], __n);
                const std::size_t __length = __begin < __end ? __end - __begin : 0;
                // Quotient/remainder partitioning avoids length * part and overflowing round-up expressions.
                const std::size_t __quotient = __length / __groups_per_segment;
                const std::uint32_t __remainder = __length % __groups_per_segment;
                const std::size_t __start = __begin + __quotient * __part + std::min(__part, __remainder);
                const std::size_t __count = __quotient + (__part < __remainder);
                _Acc __value = __identity;
                // The operation is required to be associative and commutative, so four independent accumulators
                // keep more loads in flight and break the dependency chain. Indices stay below __count + 4 * __lanes.
                auto __accumulate = [&](auto __limit) {
                    using _Index = decltype(__limit);
                    const _Index __stride = __lanes;
                    _Index __i = __lane;
                    _Acc __value1 = __identity;
                    _Acc __value2 = __identity;
                    _Acc __value3 = __identity;
                    for (; __i + 3 * __stride < __limit; __i += 4 * __stride)
                    {
                        __value = __binary_op(__value, __input[__start + __i]);
                        __value1 = __binary_op(__value1, __input[__start + __i + __stride]);
                        __value2 = __binary_op(__value2, __input[__start + __i + 2 * __stride]);
                        __value3 = __binary_op(__value3, __input[__start + __i + 3 * __stride]);
                    }
                    for (; __i < __limit; __i += __stride)
                        __value = __binary_op(__value, __input[__start + __i]);
                    __value = __binary_op(__binary_op(__value, __value1), __binary_op(__value2, __value3));
                };
                if (__count <= std::numeric_limits<std::uint32_t>::max() - 4 * std::size_t{__lanes})
                    __accumulate(static_cast<std::uint32_t>(__count));
                else
                    __accumulate(__count);
                __value = __dpl_sycl::__reduce_over_group(__item.get_group(), __value, __binary_op);
                if (__lane == 0)
                {
                    if constexpr (_WritePartials::value)
                        __destination[__group] = __value; // Also write the identity for empty partitions.
                    else
                        __destination[__segment] = __length == 0 ? __init : __binary_op(__init, __value);
                }
            });
    });
}

template <typename _Name, typename _Acc, typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size,
          typename _InitType, typename _BinaryOperation>
bool
__parallel_reduce_by_segment_offsets_gpu_impl_thread_per_segment(sycl::queue& __q, _InRange __input,
                                                                 _OffsetRange __offsets, _OutRange __result, _Size __n,
                                                                 std::size_t __segments, _InitType __init,
                                                                 _BinaryOperation __binary_op)
{
    using _ThreadKernel = __internal::__kernel_name_generator<__segmented_reduce_dispatch_thread_kernel, _Name>;

    sycl::event __event = __q.submit([&](sycl::handler& __cgh) {
        oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
        __cgh.parallel_for<_ThreadKernel>(sycl::range<1>(__segments), [=](sycl::id<1> __id) {
            const std::size_t __segment = __id[0];
            const _Size __begin = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment], __n);
            const _Size __end = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment + 1], __n);
            auto __op = __binary_op;
            // The average does not bound this segment: finish even an arbitrarily long outlier here.
            __result[__segment] =
                oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(__input, __begin, __end, __init, __op);
        });
    });
    __event.wait_and_throw();

    return true;
}

template <typename _Acc>
std::size_t
__segmented_reduce_work_group_size(sycl::queue& __q)
{
    const auto __device = __q.get_device();
    std::size_t __wg_size = oneapi::dpl::__internal::__max_work_group_size(__q, 256);
    __wg_size = std::min(__wg_size, __device.template get_info<sycl::info::device::max_work_item_sizes<1>>()[0]);
    // Reserve headroom for the implementation's group collective, as in the existing segmented backend.
    return oneapi::dpl::__internal::__slm_adjusted_work_group_size(__q, 2 * sizeof(_Acc), __wg_size);
}

template <typename _Name, typename _Acc, typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size,
          typename _InitType, typename _BinaryOperation>
bool
__parallel_reduce_by_segment_offsets_gpu_impl_sub_group(sycl::queue& __q, _InRange __input, _OffsetRange __offsets,
                                                        _OutRange __result, _Size __n, std::size_t __segments,
                                                        _InitType __init, _BinaryOperation __binary_op)
{
    const std::size_t __wg_size = __segmented_reduce_work_group_size<_Acc>(__q);
    if (__wg_size == 0)
        return false;

    const auto __sg_sizes = __q.get_device().template get_info<sycl::info::device::sub_group_sizes>();
    if (__sg_sizes.empty())
        return false;

    const std::size_t __min_sg_size = *std::min_element(__sg_sizes.begin(), __sg_sizes.end());
    if (__min_sg_size == 0)
        return false;

    const std::size_t __segments_per_wg = __wg_size / __min_sg_size + (__wg_size % __min_sg_size != 0);
    const std::size_t __groups = __segments / __segments_per_wg + (__segments % __segments_per_wg != 0);
    if (__groups > std::numeric_limits<std::size_t>::max() / __wg_size)
        return false;

    const _Acc __identity = sycl::known_identity<_BinaryOperation, _Acc>::value;
    using _SubGroupKernel = __internal::__kernel_name_generator<__segmented_reduce_dispatch_sub_group_kernel, _Name>;
    sycl::event __event = __q.submit([&](sycl::handler& __cgh) {
        oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
        __cgh.parallel_for<_SubGroupKernel>(
            sycl::nd_range<1>{__groups * __wg_size, __wg_size}, [=](sycl::nd_item<1> __item) {
                const auto __sg = __item.get_sub_group();
                const std::uint32_t __lane = __sg.get_local_linear_id();
                const std::uint32_t __lanes = __sg.get_local_linear_range();
                const std::uint32_t __sg_count = __sg.get_group_linear_range();
                const std::size_t __base = __item.get_group(0) * __segments_per_wg;
                const std::uint32_t __count = std::min(__segments_per_wg, __segments - __base);
                // Fixed work-group tiles, but actual subgroup IDs/counts/widths, including a partial subgroup.
                for (std::uint32_t __s = __sg.get_group_linear_id(); __s < __count; __s += __sg_count)
                {
                    const std::size_t __segment = __base + __s;
                    const std::size_t __begin =
                        oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment], __n);
                    const std::size_t __end =
                        oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment + 1], __n);
                    const std::size_t __length = __begin < __end ? __end - __begin : 0;
                    _Acc __value = __identity;
                    for (std::size_t __i = __lane; __i < __length;)
                    {
                        __value = __binary_op(__value, __input[__begin + __i]);
                        if (__length - __i <= __lanes)
                            break;
                        __i += __lanes;
                    }
                    __value = __dpl_sycl::__reduce_over_group(__sg, __value, __binary_op);
                    if (__lane == 0)
                        __result[__segment] = __length == 0 ? __init : __binary_op(__init, __value);
                }
            });
    });
    __event.wait_and_throw();

    return true;
}

template <typename _Name, typename _Acc, typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size,
          typename _InitType, typename _BinaryOperation>
bool
__parallel_reduce_by_segment_offsets_gpu_impl_work_group(sycl::queue& __q, _InRange __input, _OffsetRange __offsets,
                                                         _OutRange __result, _Size __n, std::size_t __segments,
                                                         _InitType __init, _BinaryOperation __binary_op)
{
    const std::size_t __wg_size = __segmented_reduce_work_group_size<_Acc>(__q);
    if (__wg_size == 0 || __segments > std::numeric_limits<std::size_t>::max() / __wg_size)
        return false;

    sycl::event __event = __segmented_reduce_submit_groups<_Name, _Acc>(
        __q, __input, __offsets, __result, __n, __segments, __wg_size, 1, __init, __binary_op, std::false_type{});
    __event.wait_and_throw();

    return true;
}

template <typename _Name, typename _Acc, typename _InRange, typename _OffsetRange, typename _OutRange,
          typename _Size, typename _InitType, typename _BinaryOperation>
bool
__segmented_reduce_multiple_work_groups(sycl::queue& __q, _InRange __input, _OffsetRange __offsets,
                                         _OutRange __result, _Size __n, std::size_t __segments, _InitType __init,
                                         _BinaryOperation __binary_op, std::size_t __groups_per_segment)
{
    const std::size_t __wg_size = __segmented_reduce_work_group_size<_Acc>(__q);
    if (__wg_size == 0)
        return false;
    const std::size_t __size_limit = std::numeric_limits<std::size_t>::max();
    if (__segments > __size_limit / __groups_per_segment)
        return false;
    const std::size_t __groups = __segments * __groups_per_segment;
    if (__groups > __size_limit / __wg_size)
        return false;
    if (__groups > __size_limit / sizeof(_Acc) ||
        __groups > __q.get_device().template get_info<sycl::info::device::max_mem_alloc_size>() / sizeof(_Acc))
        return false;

    using _CombineKernel = __internal::__kernel_name_generator<__segmented_reduce_dispatch_combine_kernel, _Name>;
    const _Acc __identity = sycl::known_identity<_BinaryOperation, _Acc>::value;
    auto __partials = __buffer<_Acc>(__groups).get_buffer();
    sycl::event __partial_event = __segmented_reduce_submit_groups<_Name, _Acc>(
        __q, __input, __offsets, oneapi::dpl::__ranges::views::all_write(__partials), __n, __groups, __wg_size,
        __groups_per_segment, __init, __binary_op, std::true_type{});
    sycl::event __event = __q.submit([&](sycl::handler& __cgh) {
        // The buffer dependency and explicit event also order the two submissions for USM on out-of-order queues.
        __cgh.depends_on(__partial_event);
        auto __partial_acc = __partials.template get_access<sycl::access_mode::read>(__cgh);
        oneapi::dpl::__ranges::__require_access(__cgh, __offsets, __result);
        __cgh.parallel_for<_CombineKernel>(sycl::nd_range<1>{__segments * __wg_size, __wg_size},
                                           [=](sycl::nd_item<1> __item) {
            const std::size_t __segment = __item.get_group(0);
            const std::uint32_t __lane = __item.get_local_id(0);
            const std::uint32_t __lanes = __item.get_local_range(0);
            _Acc __value = __identity;
            for (std::uint32_t __p = __lane; __p < __groups_per_segment; __p += __lanes)
                __value = __binary_op(__value, __partial_acc[__segment * __groups_per_segment + __p]);
            __value = __dpl_sycl::__reduce_over_group(__item.get_group(), __value, __binary_op);
            if (__lane == 0)
            {
                const _Size __begin = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment], __n);
                const _Size __end = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment + 1], __n);
                __result[__segment] = __begin >= __end ? __init : __binary_op(__init, __value);
            }
        });
    });
    // The temporary buffer must stay alive until both kernels finish, even in deferred-wait mode.
    __event.wait_and_throw();
    return true;
}

template <typename _Name, typename _Acc, typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size,
          typename _InitType, typename _BinaryOperation>
bool
__parallel_reduce_by_segment_offsets_gpu_impl_two_work_groups(sycl::queue& __q, _InRange __input,
                                                              _OffsetRange __offsets, _OutRange __result, _Size __n,
                                                              std::size_t __segments, _InitType __init,
                                                              _BinaryOperation __binary_op)
{
    return __segmented_reduce_multiple_work_groups<_Name, _Acc>(__q, __input, __offsets, __result, __n, __segments,
                                                                __init, __binary_op, 2);
}

template <typename _Name, typename _Acc, typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size,
          typename _InitType, typename _BinaryOperation>
bool
__parallel_reduce_by_segment_offsets_gpu_impl_sixteen_work_groups(sycl::queue& __q, _InRange __input,
                                                                  _OffsetRange __offsets, _OutRange __result, _Size __n,
                                                                  std::size_t __segments, _InitType __init,
                                                                  _BinaryOperation __binary_op)
{
    return __segmented_reduce_multiple_work_groups<_Name, _Acc>(__q, __input, __offsets, __result, __n, __segments,
                                                                __init, __binary_op, 16);
}

template <typename _Name, typename _Acc, typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size,
          typename _InitType, typename _BinaryOperation>
bool
__parallel_reduce_by_segment_offsets_gpu_impl_grid(sycl::queue& __q, _InRange __input, _OffsetRange __offsets,
                                                   _OutRange __result, _Size __n, std::size_t __segments,
                                                   _InitType __init, _BinaryOperation __binary_op)
{
    // UNTUNED: four groups per compute unit, bounded to keep both scratch and the final reduction small.
    const std::size_t __compute_units = __q.get_device().template get_info<sycl::info::device::max_compute_units>();
    const std::size_t __grid_groups = 4 * std::min<std::size_t>(__compute_units, 256);
    const std::size_t __groups_per_segment =
        std::clamp<std::size_t>(__grid_groups / __segments + (__grid_groups % __segments != 0), 32, 1024);

    return __segmented_reduce_multiple_work_groups<_Name, _Acc>(__q, __input, __offsets, __result, __n, __segments,
                                                                __init, __binary_op, __groups_per_segment);
}

// Returns false before any submission if the optimized launch geometry cannot be represented safely.
template <typename _Acc, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange, typename _OutRange,
          typename _Size, typename _InitType, typename _BinaryOperation>
bool
__parallel_reduce_by_segment_offsets_gpu(_ExecutionPolicy&& __exec, _InRange __input, _OffsetRange __offsets,
                                         _OutRange __result, _Size __n, _Size __m, _InitType __init,
                                         _BinaryOperation __binary_op)
{
    const std::size_t __size_limit = std::numeric_limits<std::size_t>::max();
    if (__n <= 0 || __m <= 0 || static_cast<std::uintmax_t>(__n) > __size_limit ||
        static_cast<std::uintmax_t>(__m) >= __size_limit)
        return false;

    // Include the complete submission specialization even for explicitly named policies: the generator may otherwise
    // discard its additional type arguments. The group kernel also distinguishes final and partial destinations.
    using _Name = __segmented_reduce_dispatch_name<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>,
                                                   _Acc, _ExecutionPolicy, _InRange, _OffsetRange, _OutRange, _Size,
                                                   _InitType, _BinaryOperation>;

    sycl::queue __q = __exec.queue();
    const std::size_t __segments = static_cast<std::size_t>(__m);

    switch (__segmented_reduce_dispatch(static_cast<std::size_t>(__n), __segments))
    {
    case __segmented_reduce_dispatch_strategy::__thread_per_segment:
        return __parallel_reduce_by_segment_offsets_gpu_impl_thread_per_segment<_Name, _Acc>(
            __q, __input, __offsets, __result, __n, __segments, __init, __binary_op);

    case __segmented_reduce_dispatch_strategy::__sub_group:
        return __parallel_reduce_by_segment_offsets_gpu_impl_sub_group<_Name, _Acc>(
            __q, __input, __offsets, __result, __n, __segments, __init, __binary_op);

    case __segmented_reduce_dispatch_strategy::__work_group:
        return __parallel_reduce_by_segment_offsets_gpu_impl_work_group<_Name, _Acc>(
            __q, __input, __offsets, __result, __n, __segments, __init, __binary_op);

    case __segmented_reduce_dispatch_strategy::__two_work_groups:
        return __parallel_reduce_by_segment_offsets_gpu_impl_two_work_groups<_Name, _Acc>(
            __q, __input, __offsets, __result, __n, __segments, __init, __binary_op);

    case __segmented_reduce_dispatch_strategy::__sixteen_work_groups:
        return __parallel_reduce_by_segment_offsets_gpu_impl_sixteen_work_groups<_Name, _Acc>(
            __q, __input, __offsets, __result, __n, __segments, __init, __binary_op);

    case __segmented_reduce_dispatch_strategy::__grid:
        return __parallel_reduce_by_segment_offsets_gpu_impl_grid<_Name, _Acc>(
            __q, __input, __offsets, __result, __n, __segments, __init, __binary_op);
    }

    assert(false && "Unexpected segmented reduce dispatch strategy");
    return false;
}
#endif // _ONEDPL_USE_GROUP_ALGOS && _ONEDPL_USE_SUB_GROUPS

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

// Reduces the segments longer than __min_length in chunks; __max_segments bounds the number of such segments
template <typename _Acc, typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange,
          typename _OutRange, typename _Size, typename _InitType, typename _BinaryOperation>
void
__segmented_reduce_in_chunks(_BackendTag, _ExecutionPolicy&& __exec, _InRange&& __input, _OffsetRange&& __offsets,
                              _OutRange&& __result, _Size __n, _Size __m, const _InitType& __init,
                              const _BinaryOperation& __binary_op, bool __is_gpu, _Size __min_length,
                              bool __write_empty, _Size __max_length, _Size __max_segments)
{
    const _Size __chunk_size = __segmented_reduce_chunk_size(__is_gpu, __max_length);

    // The end of the chunks of every segment: an inclusive scan of their numbers, computed from the pairs of
    // the adjacent offsets. The chunks depend on it through __chunk_ends, so the host does not wait for it; its
    // future keeps the temporary data of the scan until the partial results are combined.
    auto __chunk_ends = __buffer<_Size>(__m).get_buffer();
    auto __scan = __parallel_transform_scan(
        _BackendTag{}, make_wrapped_policy<__segmented_reduce_scan_wrapper>(__exec),
        oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::take_view_simple(__offsets, __m),
                                             oneapi::dpl::__ranges::drop_view_simple(__offsets, _Size{1})),
        oneapi::dpl::__ranges::views::all_write(__chunk_ends), __m,
        __segmented_reduce_chunk_count<_Size>{__n, __chunk_size, __min_length},
        unseq_backend::__no_init_value<_Size>{}, std::plus<_Size>{}, /*_Inclusive=*/std::true_type{});

    // At most n / __chunk_size + 1 chunks per segment, unless the offsets are not ordered
    const _Size __num_partials = __n / __chunk_size + __max_segments + 1;
    auto __partials = __buffer<_Acc>(__num_partials).get_buffer();
    __parallel_for(_BackendTag{}, make_wrapped_policy<__segmented_reduce_chunk_wrapper>(__exec),
                   __segmented_reduce_chunk_brick<_Acc, _Size, _InitType, _BinaryOperation>{__n, __m, __chunk_size,
                                                                                            __init, __binary_op},
                   __num_partials, oneapi::dpl::__ranges::views::all_read(__chunk_ends), __offsets,
                   oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::views::all_write(__partials)), __input);
    __parallel_for(_BackendTag{}, make_wrapped_policy<__segmented_reduce_combine_wrapper>(__exec),
                   __segmented_reduce_combine_brick<_Acc, _Size, _InitType, _BinaryOperation>{
                       __num_partials, __write_empty, __init, __binary_op},
                   __m, oneapi::dpl::__ranges::views::all_read(__chunk_ends),
                   oneapi::dpl::__ranges::views::all_read(__partials), oneapi::dpl::__ranges::make_zip_view(__result))
        .__checked_deferrable_wait();
    __scan.__checked_deferrable_wait();
}

// The segments of the input [0, __n) defined by the __m + 1 offsets, reduced into the __m results
template <typename _Acc, typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange,
          typename _OutRange, typename _Size, typename _InitType, typename _BinaryOperation>
void
__parallel_reduce_by_segment_offsets(_BackendTag, _ExecutionPolicy&& __exec, _InRange&& __input,
                                     _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n, const _Size __m,
                                     _InitType __init, _BinaryOperation __binary_op)
{
    const bool __is_gpu = __exec.queue().get_device().is_gpu();
#if _ONEDPL_USE_GROUP_ALGOS && _ONEDPL_USE_SUB_GROUPS
    using _InputValue = std::decay_t<oneapi::dpl::__internal::__value_t<_InRange>>;
    using _InputReference = std::decay_t<decltype(__input[_Size{0}])>;
    if constexpr (unseq_backend::__can_use_group_reduce_scan<_BinaryOperation, _Acc>::value &&
                  std::is_same_v<_InputValue, _Acc> && std::is_same_v<_InputReference, _Acc> &&
                  std::is_same_v<std::decay_t<_InitType>, _Acc> &&
                  (std::is_integral_v<_Acc> || std::is_same_v<_Acc, float> || std::is_same_v<_Acc, double> ||
                   std::is_same_v<_Acc, sycl::half>))
    {
        if (__is_gpu && __parallel_reduce_by_segment_offsets_gpu<_Acc>(__exec, __input, __offsets, __result, __n, __m,
                                                                       __init, __binary_op))
            return;
    }
#endif // _ONEDPL_USE_GROUP_ALGOS && _ONEDPL_USE_SUB_GROUPS
    const _Size __threshold = __is_gpu ? __segmented_reduce_gpu_threshold : __segmented_reduce_cpu_threshold;

    if (__n / __m > __threshold)
    {
        // The average segment is long: all the segments in chunks. The longest one selects the chunk length on GPUs.
        const _Size __max_length =
            __is_gpu
                ? __parallel_transform_reduce<_Size, std::true_type>(
                      _BackendTag{}, make_wrapped_policy<__segmented_reduce_length_wrapper>(__exec),
                      oneapi::dpl::maximum<_Size>{},
                      unseq_backend::walk_n<__segmented_reduce_length<_Size>>{__segmented_reduce_length<_Size>{__n}},
                      unseq_backend::__init_value<_Size>{_Size{0}},
                      oneapi::dpl::__ranges::drop_view_simple(__offsets, _Size{1}),
                      oneapi::dpl::__ranges::take_view_simple(__offsets, __m))
                      .get()
                : __n;
        __segmented_reduce_in_chunks<_Acc>(_BackendTag{}, __exec, __input, __offsets, __result, __n, __m, __init,
                                           __binary_op, __is_gpu, _Size{0}, /*__write_empty=*/true, __max_length, __m);
        return;
    }

    // The short segments, one work-item each; the long ones only report the longest length. The report is in a buffer,
    // which costs less on GPUs than copying it from device USM (0.08 ms vs 0.33 ms on integrated graphics).
    auto __report_buf = __buffer<std::uint32_t>(1).get_buffer();
    // The buffer has no host data: the atomic maximum starts from the zero written here
    __report_buf.get_host_access(sycl::write_only)[0] = 0;
    __parallel_for(_BackendTag{}, __exec,
                   __segmented_reduce_variable_brick<_Acc, _Size, _InitType, _BinaryOperation>{__n, __threshold, __init,
                                                                                               __binary_op},
                   __m, __offsets, oneapi::dpl::__ranges::make_zip_view(__result), __input,
                   oneapi::dpl::__ranges::views::all(__report_buf))
        .__checked_deferrable_wait();
    const _Size __max_length = __report_buf.get_host_access(sycl::read_only)[0];

    // The long segments in chunks: at most n / __threshold of them
    if (__max_length > 0)
        __segmented_reduce_in_chunks<_Acc>(_BackendTag{}, __exec, __input, __offsets, __result, __n, __m, __init,
                                           __binary_op, __is_gpu, __threshold, /*__write_empty=*/false, __max_length,
                                           __n / __threshold);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
