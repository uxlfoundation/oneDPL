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

// One work-item per short segment. A longer segment is left to the reduction in chunks and only reports its length,
// capped to 32 bits, through an atomic maximum.
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
        get<0>(__result[__idx]) = _Acc(__init);
    }
};

// Zeroes the report of the longest length on the device
struct __segmented_reduce_report_init_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    // __report is a zip view of the report only, so that it is not vectorized over
    template <typename _IsFull, typename _Params, typename _Report>
    void
    operator()(_IsFull, const std::size_t, _Params, _Report __report) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");
        using std::get;
        get<0>(__report[0]) = 0;
    }
};

template <typename _Name>
class __segmented_reduce_fill_wrapper;
template <typename _Name>
class __segmented_reduce_report_wrapper;

// A short segment is reduced by one work-item. A long segment is split into chunks, each reduced by one work-item, and
// the partial results of the segment are combined by one work-item. The values below were measured on Intel GPUs
// and CPUs:
// - a segment up to this length is short;
inline constexpr std::size_t __segmented_reduce_gpu_threshold = 512;
inline constexpr std::size_t __segmented_reduce_cpu_threshold = 32768;
// - on GPUs the best chunk length of a long segment is close to the square root of its length, which balances the
//   serial work of its chunks and of their combination; on CPUs the chunk length hardly matters above 1024.
inline constexpr std::size_t __segmented_reduce_gpu_min_chunk_size = 64;
inline constexpr std::size_t __segmented_reduce_gpu_max_chunk_size = 16384;
inline constexpr std::size_t __segmented_reduce_cpu_chunk_size = 4096;

// The chunk length of a long segment, computed on the device for every segment: on GPUs the least power of two whose
// square is not less than the length, without the floating point that a device may lack in double precision
template <typename _Size>
_Size
__segmented_reduce_chunk_size(bool __is_gpu, _Size __length)
{
    if (!__is_gpu)
        return _Size(__segmented_reduce_cpu_chunk_size);
    _Size __chunk_size = _Size(__segmented_reduce_gpu_min_chunk_size);
    while (__chunk_size < _Size(__segmented_reduce_gpu_max_chunk_size) && __chunk_size * __chunk_size < __length)
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

// The number of chunks of a segment given by the pair of its offsets
template <typename _Size>
struct __segmented_reduce_chunk_count
{
    _Size __n;          // the size of the input
    _Size __min_length; // the segments not longer than this have no chunks
    bool __is_gpu;      // selects the chunk length

    template <typename _Bounds>
    _Size
    operator()(const _Bounds& __bounds) const
    {
        const _Size __length = __segmented_reduce_length<_Size>{__n}(std::get<1>(__bounds), std::get<0>(__bounds));
        if (__length <= __min_length)
            return _Size{0};
        const _Size __chunk_size = __segmented_reduce_chunk_size(__is_gpu, __length);
        return (__length + __chunk_size - 1) / __chunk_size;
    }
};

// Reduces one chunk of a long segment. The first chunk of a segment takes the remainder of its length and starts
// from the operation on the initial value and its first element; every other chunk has exactly the chunk length of
// the segment and starts from the operation on its first two elements.
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
        const _Size __chunk_size = __segmented_reduce_chunk_size(__is_gpu, _Size(__end - __begin));
        const _Size __head = (__end - __begin) - (_Size(__chunk_ends[__lo]) - __first_chunk - 1) * __chunk_size;

        // One loop for every chunk, so that the work-items of a sub-group do not diverge into two long loops
        const _Size __k = __chunk - __first_chunk;
        const _Size __chunk_begin = __k == 0 ? __begin : __begin + __head + (__k - 1) * __chunk_size;
        const _Size __chunk_end = __k == 0 ? __begin + __head : __chunk_begin + __chunk_size;
        // A chunk other than the first one starts from the operation on its first two elements, as a partial result
        // of std::reduce does: the accumulator is never made from an element alone, which may mean something else than
        // the element to the operation (such a chunk has at least 64 elements)
        _Acc __start = __k == 0 ? _Acc(std::invoke(__binary_op, __init, __input[__chunk_begin]))
                                : _Acc(std::invoke(__binary_op, __input[__chunk_begin], __input[__chunk_begin + 1]));
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
                std::get<0>(__result[__idx]) = _Acc(__init);
            return;
        }
        std::get<0>(__result[__idx]) = oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(
            __partials, __first + 1, __last, _Acc(__partials[__first]), __binary_op);
    }
};

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

    // Reduces the segments longer than __min_length in chunks; __max_segments bounds the number of such segments
    auto __reduce_in_chunks = [&](_Size __min_length, bool __write_empty, _Size __max_segments) {
        // The end of the chunks of every segment: an inclusive scan of their numbers, computed from the pairs of
        // the adjacent offsets. The chunks depend on it through __chunk_ends, so the host does not wait for it; its
        // future keeps the temporary data of the scan until the partial results are combined.
        auto __chunk_ends = __buffer<_Size>(__m).get_buffer();
        auto __scan = __parallel_transform_scan(
            _BackendTag{}, make_wrapped_policy<__segmented_reduce_scan_wrapper>(__exec),
            oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::take_view_simple(__offsets, __m),
                                                 oneapi::dpl::__ranges::drop_view_simple(__offsets, _Size{1})),
            oneapi::dpl::__ranges::views::all_write(__chunk_ends), __m,
            __segmented_reduce_chunk_count<_Size>{__n, __min_length, __is_gpu}, unseq_backend::__no_init_value<_Size>{},
            std::plus<_Size>{}, /*_Inclusive=*/std::true_type{});

        // At most n / (the least chunk length) + 1 chunks per segment, unless the offsets are not ordered
        const _Size __min_chunk_size =
            _Size(__is_gpu ? __segmented_reduce_gpu_min_chunk_size : __segmented_reduce_cpu_chunk_size);
        const _Size __num_partials = __n / __min_chunk_size + __max_segments + 1;
        auto __partials = __buffer<_Acc>(__num_partials).get_buffer();
        __parallel_for(_BackendTag{}, make_wrapped_policy<__segmented_reduce_chunk_wrapper>(__exec),
                       __segmented_reduce_chunk_brick<_Acc, _Size, _InitType, _BinaryOperation>{__n, __m, __is_gpu,
                                                                                                __init, __binary_op},
                       __num_partials, oneapi::dpl::__ranges::views::all_read(__chunk_ends), __offsets,
                       oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::views::all_write(__partials)),
                       __input);
        __parallel_for(_BackendTag{}, make_wrapped_policy<__segmented_reduce_combine_wrapper>(__exec),
                       __segmented_reduce_combine_brick<_Acc, _Size, _InitType, _BinaryOperation>{
                           __num_partials, __write_empty, __init, __binary_op},
                       __m, oneapi::dpl::__ranges::views::all_read(__chunk_ends),
                       oneapi::dpl::__ranges::views::all_read(__partials),
                       oneapi::dpl::__ranges::make_zip_view(__result))
            .__checked_deferrable_wait();
        __scan.__checked_deferrable_wait();
    };

    if (__n / __m > __threshold)
    {
        // The average segment is long: all the segments in chunks
        __reduce_in_chunks(_Size{0}, /*__write_empty=*/true, __m);
        return;
    }

    // The short segments, one work-item each; the long ones only report the longest length. The report is in a buffer,
    // which costs less on GPUs than copying it from device USM. On GPUs the atomic maximum starts from a zero written by
    // a kernel, so that the host only reads the buffer, as the key-based reduce_by_segment does; on CPU devices from a
    // zero written through a host accessor, which costs less there than one more kernel.
    auto __report_buf = __buffer<std::uint32_t>(1).get_buffer();
    if (__is_gpu)
    {
        __parallel_for(_BackendTag{}, make_wrapped_policy<__segmented_reduce_report_wrapper>(__exec),
                       __segmented_reduce_report_init_brick{}, _Size{1},
                       oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::views::all_write(__report_buf)));
    }
    else
    {
        __report_buf.get_host_access(sycl::write_only)[0] = 0;
    }
    __parallel_for(_BackendTag{}, __exec,
                   __segmented_reduce_variable_brick<_Acc, _Size, _InitType, _BinaryOperation>{__n, __threshold, __init,
                                                                                               __binary_op},
                   __m, __offsets, oneapi::dpl::__ranges::make_zip_view(__result), __input,
                   oneapi::dpl::__ranges::views::all(__report_buf))
        .__checked_deferrable_wait();
    const _Size __max_length = __report_buf.get_host_access(sycl::read_only)[0];

    // The long segments in chunks: at most n / __threshold of them
    if (__max_length > 0)
        __reduce_in_chunks(__threshold, /*__write_empty=*/false, __n / __threshold);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
