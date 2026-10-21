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

// One work-item per short segment. A longer segment is left to the reduction in chunks; with a report, it reports its
// length, capped to 32 bits, through an atomic maximum.
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

    // Without a report: a long segment is only skipped
    template <typename _IsFull, typename _Params, typename _Offsets, typename _Result, typename _Input>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Offsets __offsets, _Result __result, _Input __input) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");

        const _Size __begin = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__idx], __n);
        const _Size __end = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__idx + 1], __n);
        if (__end - __begin <= __threshold)
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
        get<0>(__result[__idx]) = _Acc(__init);
    }
};

template <typename _Name>
class __segmented_reduce_fill_wrapper;

// A short segment is reduced by one work-item. A long segment is split into chunks, each reduced by one work-item, and
// the partial results of the segment are combined by one work-item. The values below were measured on Intel GPUs
// and CPUs:
// - a segment up to this length is short;
inline constexpr std::size_t __segmented_reduce_gpu_threshold = 512;
inline constexpr std::size_t __segmented_reduce_cpu_threshold = 32768;
// - all the segments are reduced in chunks once their average length exceeds this on GPUs, and the threshold above on
//   CPUs: on GPUs the short segments beside the long ones in chunks cost more than all the segments in chunks;
inline constexpr std::size_t __segmented_reduce_gpu_chunks_average = 256;
// - on GPUs an input up to this size is reduced by one kernel, one work-item per segment of any length;
inline constexpr std::size_t __segmented_reduce_gpu_small_input = 1024;
// - on GPUs the best chunk length of a long segment is close to the square root of its length, which balances the
//   serial work of its chunks and of their combination; on CPUs the chunk length hardly matters above 1024.
inline constexpr std::size_t __segmented_reduce_gpu_min_chunk_size = 64;
inline constexpr std::size_t __segmented_reduce_gpu_max_chunk_size = 16384;
inline constexpr std::size_t __segmented_reduce_cpu_chunk_size = 4096;
// - on GPUs the chunks of arithmetic types are reduced by sub-groups of this size, each reducing a chunk this many times
//   faster: the chunk length is close to the square root of the length times the sub-group size, with at least 32
//   elements per work-item, as the search for the segment of a chunk costs more than a few elements.
inline constexpr std::uint8_t __segmented_reduce_gpu_sub_group_size = 32;
inline constexpr std::size_t __segmented_reduce_gpu_min_group_chunk_size = 1024;
inline constexpr std::size_t __segmented_reduce_gpu_max_group_chunk_size = 65536;
// - only when the typical segment in chunks, the average one if all the segments are and the longest one otherwise, is
//   at least this long: shorter segments leave a sub-group too few elements per work-item.
inline constexpr std::size_t __segmented_reduce_gpu_min_group_length = 1024;

// The chunk length of a long segment, computed on the device for every segment: on GPUs the least power of two whose
// square is not less than the length, times the sub-group size for a chunk reduced by a sub-group, without the floating
// point that a device may lack in double precision
template <typename _Size>
_Size
__segmented_reduce_chunk_size(bool __is_gpu, bool __by_groups, _Size __length)
{
    if (!__is_gpu)
        return _Size(__segmented_reduce_cpu_chunk_size);
    const _Size __lanes = __by_groups ? _Size(__segmented_reduce_gpu_sub_group_size) : _Size{1};
    const _Size __max_chunk_size =
        _Size(__by_groups ? __segmented_reduce_gpu_max_group_chunk_size : __segmented_reduce_gpu_max_chunk_size);
    _Size __chunk_size =
        _Size(__by_groups ? __segmented_reduce_gpu_min_group_chunk_size : __segmented_reduce_gpu_min_chunk_size);
    while (__chunk_size < __max_chunk_size && __chunk_size * (__chunk_size / __lanes) < __length)
        __chunk_size *= 2;
    return __chunk_size;
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

// The length of a long segment within the input, and zero for a short one
template <typename _Size>
struct __segmented_reduce_long_length
{
    _Size __n;         // the size of the input
    _Size __threshold; // the length of the longest short segment

    template <typename _OffsetEnd, typename _OffsetBegin>
    _Size
    operator()(_OffsetEnd __end, _OffsetBegin __begin) const
    {
        const _Size __length = __segmented_reduce_length<_Size>{__n}(__end, __begin);
        return __length > __threshold ? __length : _Size{0};
    }
};

// The number of chunks of a segment given by the pair of its offsets
template <typename _Size>
struct __segmented_reduce_chunk_count
{
    _Size __n;          // the size of the input
    _Size __min_length; // the segments not longer than this have no chunks
    bool __is_gpu;      // selects the chunk length
    bool __by_groups;   // the chunks are reduced by sub-groups

    template <typename _Bounds>
    _Size
    operator()(const _Bounds& __bounds) const
    {
        const _Size __length = __segmented_reduce_length<_Size>{__n}(std::get<1>(__bounds), std::get<0>(__bounds));
        if (__length <= __min_length)
            return _Size{0};
        const _Size __chunk_size = __segmented_reduce_chunk_size(__is_gpu, __by_groups, __length);
        return (__length + __chunk_size - 1) / __chunk_size;
    }
};

// The elements [__begin, __end) of a chunk of a long segment; the first chunk of a segment takes the remainder of its
// length, every other chunk has exactly the chunk length of the segment
template <typename _Size>
struct __segmented_reduce_chunk_bounds
{
    _Size __begin;
    _Size __end;
    bool __is_first; // the first chunk of its segment, which also takes the initial value
};

template <typename _Size, typename _ChunkEnds, typename _Offsets>
__segmented_reduce_chunk_bounds<_Size>
__segmented_reduce_find_chunk(const _Size __chunk, _ChunkEnds __chunk_ends, _Offsets __offsets, const _Size __n,
                              const _Size __num_segments, const bool __is_gpu, const bool __by_groups)
{
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
    const _Size __chunk_size = __segmented_reduce_chunk_size(__is_gpu, __by_groups, _Size(__end - __begin));
    const _Size __head = (__end - __begin) - (_Size(__chunk_ends[__lo]) - __first_chunk - 1) * __chunk_size;

    const _Size __k = __chunk - __first_chunk;
    const _Size __chunk_begin = __k == 0 ? __begin : __begin + __head + (__k - 1) * __chunk_size;
    return {__chunk_begin, __k == 0 ? __begin + __head : __chunk_begin + __chunk_size, __k == 0};
}

// Reduces one chunk of a long segment. The first chunk of a segment starts from the operation on the initial value
// and its first element; every other chunk starts from the operation on its first two elements.
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_chunk_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _Size __n;            // the size of the input
    _Size __num_segments; // the number of segments
    bool __is_gpu;        // selects the chunk length
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

        // One loop for every chunk, so that the work-items of a sub-group do not diverge into two long loops
        const auto [__chunk_begin, __chunk_end, __is_first] =
            __segmented_reduce_find_chunk(__chunk, __chunk_ends, __offsets, __n, __num_segments, __is_gpu,
                                          /*__by_groups=*/false);
        // A chunk other than the first one starts from the operation on its first two elements, as a partial result
        // of std::reduce does: the accumulator is never made from an element alone, which may mean something else than
        // the element to the operation (such a chunk has at least 64 elements)
        _Acc __start = __is_first ? _Acc(std::invoke(__binary_op, __init, __input[__chunk_begin]))
                                  : _Acc(std::invoke(__binary_op, __input[__chunk_begin], __input[__chunk_begin + 1]));
        std::get<0>(__partials[__idx]) = oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(
            __input, __chunk_begin + (__is_first ? 1 : 2), __chunk_end, std::move(__start), __binary_op);
    }
};

// Reduces the chunks of the long segments on GPUs, one sub-group per chunk: the lanes take the elements of the chunk
// in turn, so that the loads of the sub-group are coalesced, and a tree combines the results of the lanes. Both
// reorder the operands, which the required commutativity of the operation allows. Only for arithmetic types, which a
// sub-group shifts between its lanes.
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_chunk_group_brick
{
    _Size __n;            // the size of the input
    _Size __num_segments; // the number of segments
    _Size __num_partials; // the number of the partial results computed
    _InitType __init;
    mutable _BinaryOperation __binary_op;

    template <typename _ChunkEnds, typename _Offsets, typename _Partials, typename _Input>
    void
    operator()(const sycl::nd_item<1>& __item, _ChunkEnds __chunk_ends, _Offsets __offsets, _Partials __partials,
               _Input __input) const
    {
        const __dpl_sycl::__sub_group __sg = __item.get_sub_group();
        const _Size __lane = __sg.get_local_linear_id();
        const _Size __lanes = __sg.get_local_linear_range();
        const _Size __groups_per_item = __sg.get_group_linear_range();
        // Unordered offsets may give more chunks than computed: their results are unspecified
        const _Size __num_chunks = std::min<_Size>(__chunk_ends[__num_segments - 1], __num_partials);
        // Every sub-group reduces every (number of the sub-groups)-th chunk
        for (_Size __chunk = __item.get_group_linear_id() * __groups_per_item + __sg.get_group_linear_id();
             __chunk < __num_chunks; __chunk += __item.get_group_range(0) * __groups_per_item)
        {
            auto [__first, __last, __is_first] =
                __segmented_reduce_find_chunk(__chunk, __chunk_ends, __offsets, __n, __num_segments, /*__is_gpu=*/true,
                                              /*__by_groups=*/true);
            // The first chunk of a segment starts from the operation on the initial value and its first element
            _Acc __head{};
            if (__is_first)
                __head = std::invoke(__binary_op, __init, __input[__first++]);

            // Every lane that takes part gets at least two elements: its accumulator starts from the operation on two
            // of them, never from an element alone, which may mean something else than the element to the operation
            const _Size __active = std::min<_Size>(__lanes, (__last - __first) / 2);
            _Acc __acc{};
            if (__lane < __active)
            {
                _Size __i = __first + __lane;
                __acc = std::invoke(__binary_op, __input[__i], __input[__i + __active]);
                // Four loads in flight
                for (__i += 2 * __active; __i + 3 * __active < __last; __i += 4 * __active)
                    __acc = std::invoke(
                        __binary_op, __acc,
                        std::invoke(
                            __binary_op, _Acc(std::invoke(__binary_op, __input[__i], __input[__i + __active])),
                            _Acc(std::invoke(__binary_op, __input[__i + 2 * __active], __input[__i + 3 * __active]))));
                for (; __i < __last; __i += __active)
                    __acc = std::invoke(__binary_op, __acc, __input[__i]);
            }
            for (_Size __shift = 1; __shift < __active; __shift *= 2)
            {
                const _Acc __other = sycl::shift_group_left(__sg, __acc, __shift);
                if (__lane % (2 * __shift) == 0 && __lane + __shift < __active)
                    __acc = std::invoke(__binary_op, __acc, __other);
            }

            if (__lane == 0)
            {
                // A first chunk of one element more than the initial value takes it alone
                if (__is_first)
                    __acc = __active > 0       ? _Acc(std::invoke(__binary_op, __head, __acc))
                            : __first < __last ? _Acc(std::invoke(__binary_op, __head, __input[__first]))
                                               : __head;
                std::get<0>(__partials[__chunk]) = __acc;
            }
        }
    }
};

template <typename... _Name>
class __segmented_reduce_chunk_group_kernel;

template <typename _KernelName>
struct __segmented_reduce_chunk_group_submitter;

template <typename... _Name>
struct __segmented_reduce_chunk_group_submitter<__internal::__optional_kernel_name<_Name...>>
{
    template <typename _Brick, typename... _Ranges>
    sycl::event
    operator()(sycl::queue& __q, const _Brick& __brick, const std::size_t __num_groups,
               const std::size_t __work_group_size, _Ranges&&... __rngs) const
    {
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...);
            __cgh.parallel_for<_Name...>(
                sycl::nd_range<1>(sycl::range<1>(__num_groups * __work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item)
                    [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED(__segmented_reduce_gpu_sub_group_size)]] {
                        __brick(__item, __rngs...);
                    });
        });
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
                std::get<0>(__result[__idx]) = _Acc(__init);
            return;
        }
        std::get<0>(__result[__idx]) = oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(
            __partials, __first + 1, __last, _Acc(__partials[__first]), __binary_op);
    }
};

template <typename _Name>
class __segmented_reduce_short_wrapper;
template <typename _Name>
class __segmented_reduce_length_wrapper;
template <typename _Name>
class __segmented_reduce_scan_wrapper;
template <typename _Name>
class __segmented_reduce_chunk_wrapper;
template <typename _Name>
class __segmented_reduce_combine_wrapper;

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

// The segments of the input [0, __n) defined by the __m + 1 offsets, reduced into the __m results
template <typename _Acc, typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange,
          typename _OutRange, typename _Size, typename _InitType, typename _BinaryOperation>
void
__parallel_reduce_by_segment_offsets(_BackendTag, _ExecutionPolicy&& __exec, _InRange&& __input,
                                     _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n, const _Size __m,
                                     _InitType __init, _BinaryOperation __binary_op)
{
    const bool __is_gpu = __exec.queue().get_device().is_gpu();
    const _Size __threshold = __is_gpu ? __segmented_reduce_gpu_threshold : __segmented_reduce_cpu_threshold;

    // Reduces the segments longer than __min_length in chunks; __max_segments bounds the number of such segments, and
    // __typical_length is the length of a typical one
    auto __reduce_in_chunks = [&](_Size __min_length, bool __write_empty, _Size __max_segments,
                                  _Size __typical_length) {
        // On GPUs a work-item reducing a chunk alone gathers its elements from as many cache lines as the sub-group
        // has lanes, so the chunks of arithmetic types are reduced by sub-groups there
        bool __by_groups = false;
        if constexpr (std::is_arithmetic_v<_Acc> &&
                      std::is_arithmetic_v<oneapi::dpl::__internal::__value_t<std::decay_t<_InRange>>>)
            __by_groups = __is_gpu && __typical_length >= _Size(__segmented_reduce_gpu_min_group_length) &&
                          oneapi::dpl::__internal::__supports_sub_group_size(__exec.queue(),
                                                                             __segmented_reduce_gpu_sub_group_size);

        // The end of the chunks of every segment: an inclusive scan of their numbers, computed from the pairs of
        // the adjacent offsets. The chunks depend on it through __chunk_ends, so the host does not wait for it; its
        // future keeps the temporary data of the scan until the partial results are combined.
        auto __chunk_ends = __buffer<_Size>(__m).get_buffer();
        auto __scan = __parallel_transform_scan(
            _BackendTag{}, make_wrapped_policy<__segmented_reduce_scan_wrapper>(__exec),
            oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::take_view_simple(__offsets, __m),
                                                 oneapi::dpl::__ranges::drop_view_simple(__offsets, _Size{1})),
            oneapi::dpl::__ranges::views::all_write(__chunk_ends), __m,
            __segmented_reduce_chunk_count<_Size>{__n, __min_length, __is_gpu, __by_groups},
            unseq_backend::__no_init_value<_Size>{}, std::plus<_Size>{}, /*_Inclusive=*/std::true_type{});

        // At most n / (the least chunk length) + 1 chunks per segment, unless the offsets are not ordered. On GPUs a
        // segment of length L also has at most sqrt(L / S) + L / (the greatest chunk length) + 1 chunks, with S the
        // sub-group size for the chunks reduced by sub-groups and 1 otherwise, so all of them at most
        // sqrt(__max_segments * n / S) + n / (the greatest chunk length) + __max_segments: far fewer for long
        // segments, which would otherwise get many idle work-items and an oversized buffer of partial results.
        const _Size __min_chunk_size = _Size(!__is_gpu     ? __segmented_reduce_cpu_chunk_size
                                             : __by_groups ? __segmented_reduce_gpu_min_group_chunk_size
                                                           : __segmented_reduce_gpu_min_chunk_size);
        _Size __num_partials = __n / __min_chunk_size + __max_segments + 1;
        if (__is_gpu)
        {
            const double __lanes = __by_groups ? double(__segmented_reduce_gpu_sub_group_size) : 1.0;
            const _Size __max_chunk_size = _Size(__by_groups ? __segmented_reduce_gpu_max_group_chunk_size
                                                             : __segmented_reduce_gpu_max_chunk_size);
            const _Size __root = _Size(std::sqrt(double(__max_segments) * double(__n) / __lanes)) + 1;
            __num_partials = std::min<_Size>(__num_partials, __root + __n / __max_chunk_size + __max_segments + 1);
        }
        auto __partials = __buffer<_Acc>(__num_partials).get_buffer();
        auto __reduce_by_items = [&]() {
            __parallel_for(_BackendTag{}, make_wrapped_policy<__segmented_reduce_chunk_wrapper>(__exec),
                           __segmented_reduce_chunk_brick<_Acc, _Size, _InitType, _BinaryOperation>{
                               __n, __m, __is_gpu, __init, __binary_op},
                           __num_partials, oneapi::dpl::__ranges::views::all_read(__chunk_ends), __offsets,
                           oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::views::all_write(__partials)),
                           __input);
        };
        if constexpr (std::is_arithmetic_v<_Acc> &&
                      std::is_arithmetic_v<oneapi::dpl::__internal::__value_t<std::decay_t<_InRange>>>)
        {
            if (__by_groups)
            {
                using _GroupKernel = __internal::__kernel_name_provider<__segmented_reduce_chunk_group_kernel<
                    oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;

                sycl::queue __q = __exec.queue();
                const std::size_t __work_group_size =
                    oneapi::dpl::__internal::__max_work_group_size(__q, std::size_t(256));
                const std::size_t __chunks_per_group = __work_group_size / __segmented_reduce_gpu_sub_group_size;
                // One sub-group per chunk, at most 32 per compute unit: the bound of the number of chunks may be far
                // above their actual number, and a sub-group idles as many work-items as it has lanes
                const std::size_t __num_sub_groups = std::min<std::size_t>(
                    __num_partials, std::size_t(oneapi::dpl::__internal::__max_compute_units(__q)) * 32);
                __segmented_reduce_chunk_group_submitter<_GroupKernel>{}(
                    __q,
                    __segmented_reduce_chunk_group_brick<_Acc, _Size, _InitType, _BinaryOperation>{
                        __n, __m, __num_partials, __init, __binary_op},
                    oneapi::dpl::__internal::__dpl_ceiling_div(__num_sub_groups, __chunks_per_group), __work_group_size,
                    oneapi::dpl::__ranges::views::all_read(__chunk_ends), __offsets,
                    oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::views::all_write(__partials)), __input);
            }
            else
                __reduce_by_items();
        }
        else
            __reduce_by_items();
        __parallel_for(_BackendTag{}, make_wrapped_policy<__segmented_reduce_combine_wrapper>(__exec),
                       __segmented_reduce_combine_brick<_Acc, _Size, _InitType, _BinaryOperation>{
                           __num_partials, __write_empty, __init, __binary_op},
                       __m, oneapi::dpl::__ranges::views::all_read(__chunk_ends),
                       oneapi::dpl::__ranges::views::all_read(__partials),
                       oneapi::dpl::__ranges::make_zip_view(__result))
            .__checked_deferrable_wait();
        __scan.__checked_deferrable_wait();
    };

    // No segment of a small input is longer than the input, which a work-item reduces alone faster on GPUs than the
    // scan, the chunks and their combination take; a larger input may hold a segment too long for one work-item
    if (__is_gpu && __n <= _Size(__segmented_reduce_gpu_small_input))
    {
        __parallel_for(
            _BackendTag{}, make_wrapped_policy<__segmented_reduce_short_wrapper>(__exec),
            __segmented_reduce_variable_brick<_Acc, _Size, _InitType, _BinaryOperation>{__n, __n, __init, __binary_op},
            __m, __offsets, oneapi::dpl::__ranges::make_zip_view(__result), __input)
            .__checked_deferrable_wait();
        return;
    }

    if (__n / __m > (__is_gpu ? _Size(__segmented_reduce_gpu_chunks_average) : __threshold))
    {
        // The average segment is long enough: all the segments in chunks
        __reduce_in_chunks(_Size{0}, /*__write_empty=*/true, __m, __n / __m);
        return;
    }

    const __segmented_reduce_variable_brick<_Acc, _Size, _InitType, _BinaryOperation> __short_brick{
        __n, __threshold, __init, __binary_op};
    if (__is_gpu)
    {
        // The short segments, one work-item each, beside a reduction over the offsets for the longest long segment,
        // whose result reaches the host as for the other reductions: reading a buffer through a host accessor costs
        // much more on some GPUs.
        auto __short =
            __parallel_for(_BackendTag{}, make_wrapped_policy<__segmented_reduce_short_wrapper>(__exec), __short_brick,
                           __m, __offsets, oneapi::dpl::__ranges::make_zip_view(__result), __input);
        const _Size __max_length = __parallel_transform_reduce<_Size, std::true_type>(
                                       _BackendTag{}, make_wrapped_policy<__segmented_reduce_length_wrapper>(__exec),
                                       oneapi::dpl::maximum<_Size>{},
                                       unseq_backend::walk_n<__segmented_reduce_long_length<_Size>>{
                                           __segmented_reduce_long_length<_Size>{__n, __threshold}},
                                       unseq_backend::__init_value<_Size>{_Size{0}},
                                       oneapi::dpl::__ranges::drop_view_simple(__offsets, _Size{1}),
                                       oneapi::dpl::__ranges::take_view_simple(__offsets, __m))
                                       .get();
        // The long segments in chunks: at most n / __threshold of them
        if (__max_length > 0)
            __reduce_in_chunks(__threshold, /*__write_empty=*/false, __n / __threshold, __max_length);
        __short.__checked_deferrable_wait();
        return;
    }

    // On CPU devices the short segments, one work-item each, and the long ones only report the longest length, into a
    // buffer zeroed and read through host accessors, which costs less there than a reduction of its own.
    auto __report_buf = __buffer<std::uint32_t>(1).get_buffer();
    __report_buf.get_host_access(sycl::write_only)[0] = 0;
    __parallel_for(_BackendTag{}, __exec, __short_brick, __m, __offsets, oneapi::dpl::__ranges::make_zip_view(__result),
                   __input, oneapi::dpl::__ranges::views::all(__report_buf))
        .__checked_deferrable_wait();
    const _Size __max_length = __report_buf.get_host_access(sycl::read_only)[0];

    // The long segments in chunks: at most n / __threshold of them
    if (__max_length > 0)
        __reduce_in_chunks(__threshold, /*__write_empty=*/false, __n / __threshold, __max_length);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
