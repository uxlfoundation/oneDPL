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

#include <algorithm>   // std::min, std::clamp, std::find
#include <cmath>       // std::sqrt
#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint8_t, std::uint32_t
#include <functional>  // std::invoke, std::less, std::plus
#include <limits>      // std::numeric_limits
#include <type_traits> // std::true_type
#include <utility>     // std::move, std::forward
#include <vector>      // std::vector

#include "sycl_defs.h"
#include "parallel_backend_sycl.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

#include "../../functional_impl.h" // oneapi::dpl::identity, oneapi::dpl::maximum
#include "../../utils.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// reduce_by_segment with segments defined by offsets

// One work-item per short segment. A longer segment is left to the reduction in chunks and only reports its length,
// capped as only its square root matters, through an atomic maximum of the device scope.
template <typename _Acc, typename _Size, typename _BinaryOperation>
struct __segmented_reduce_variable_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;
    static constexpr std::uint8_t __vector_size = 1; // one segment or chunk per work-item, in a loop of its length

    _Size __n = {};         // the size of the input
    _Size __threshold = {}; // the length of the longest short segment
    _Acc __init;            // the initial value converted to the accumulator type
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _Offsets, typename _Result, typename _Input,
              typename _MaxLongSegmentLengthBuf>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Offsets __offsets, _Result __result, _Input __input,
               _MaxLongSegmentLengthBuf __max_long_segment_length_buf) const
    {
        using _MaxLongSegmentLength = oneapi::dpl::__internal::__value_t<_MaxLongSegmentLengthBuf>;
        using __atomic_ref_t = sycl::atomic_ref<_MaxLongSegmentLength, sycl::memory_order::relaxed,
                                                sycl::memory_scope::device, sycl::access::address_space::global_space>;

        const _Size __begin = oneapi::dpl::__internal::__clamp_to_size_and_cast<_Size>(__offsets[__idx], __n);
        const _Size __end = oneapi::dpl::__internal::__clamp_to_size_and_cast<_Size>(__offsets[__idx + 1], __n);

        if (__end - __begin > __threshold)
        {
            __atomic_ref_t __ref(__max_long_segment_length_buf[0]);

            __ref.fetch_max(oneapi::dpl::__internal::__clamp_to_size_and_cast<_MaxLongSegmentLength>(
                __end - __begin, std::numeric_limits<_MaxLongSegmentLength>::max()));
            return;
        }

        __result[__idx] =
            oneapi::dpl::__internal::__serial_accumulate<_Acc>(__input, __begin, __end, __init, __binary_op);
    }
};

template <typename _Name>
class __segmented_reduce_fill_wrapper;

// Whether a device supports the atomics of the device scope, which SYCL does not require
inline bool
__has_device_scope_atomics(sycl::device __device)
{
    const std::vector<sycl::memory_scope> __scopes =
        __device.get_info<sycl::info::device::atomic_memory_scope_capabilities>();
    return std::find(__scopes.begin(), __scopes.end(), sycl::memory_scope::device) != __scopes.end();
}

// A short segment is reduced by one work-item. A long segment is split into chunks, each reduced by one work-item, and
// the partial results of the segment are combined by one work-item. The values below were measured on GPUs and CPUs
// as SYCL devices:
// - a segment up to this length is short;
inline constexpr std::size_t __segmented_reduce_gpu_short_segment_max = 512;
inline constexpr std::size_t __segmented_reduce_cpu_short_segment_max = 32768;
// - on GPUs the best chunk length is close to the square root of the longest segment, which balances the serial work
//   of the two levels; on CPUs the chunk length hardly matters above 1024.
inline constexpr std::size_t __segmented_reduce_cpu_chunk_size = 4096;

inline std::size_t
__segmented_reduce_chunk_size(bool __is_gpu, std::size_t __max_length)
{
    if (__is_gpu)
    {
        const auto __root = static_cast<std::size_t>(std::sqrt(static_cast<double>(__max_length)));
        return std::clamp<std::size_t>(oneapi::dpl::__internal::__dpl_bit_ceil(__root), 64, 16384);
    }
    return __segmented_reduce_cpu_chunk_size;
}

// The length of a segment within the input; a negatively sized segment is empty
template <typename _Size>
struct __segmented_reduce_length
{
    _Size __n = {}; // the size of the input

    template <typename _Offset>
    _Size
    operator()(_Offset __begin, _Offset __end) const
    {
        const _Size __b = oneapi::dpl::__internal::__clamp_to_size_and_cast<_Size>(__begin, __n);
        const _Size __e = oneapi::dpl::__internal::__clamp_to_size_and_cast<_Size>(__end, __n);
        return __b < __e ? __e - __b : _Size{0};
    }
};

// The number of chunks of a segment given by the pair of its offsets
template <typename _Size>
struct __segmented_reduce_chunk_count
{
    _Size __n = {};          // the size of the input
    _Size __chunk_size = {}; // the length of a chunk
    _Size __min_length = {}; // the segments not longer than this have no chunks

    template <typename _Bounds>
    _Size
    operator()(const _Bounds& __bounds) const
    {
        const _Size __length = __segmented_reduce_length<_Size>{__n}(std::get<0>(__bounds), std::get<1>(__bounds));
        return __length > __min_length ? oneapi::dpl::__internal::__dpl_ceiling_div(__length, __chunk_size) : _Size{0};
    }
};

// Reduces one chunk of a long segment. The first chunk of a segment, its head, takes the remainder of its length and
// starts from the initial value converted to the accumulator type; every other chunk has exactly __chunk_size elements
// and starts from the operation on its first two elements.
template <typename _Acc, typename _Size, typename _BinaryOperation>
struct __segmented_reduce_chunk_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;
    static constexpr std::uint8_t __vector_size = 1; // one segment or chunk per work-item, in a loop of its length

    _Size __n = {};            // the size of the input
    _Size __num_segments = {}; // the number of segments
    _Size __chunk_size = {};   // the length of a chunk
    _Acc __init;               // the initial value converted to the accumulator type
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _ChunkEnds, typename _Offsets, typename _Partials,
              typename _Input>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _ChunkEnds __chunk_ends, _Offsets __offsets,
               _Partials __partials, _Input __input) const
    {
        const _Size __chunk = __idx;
        if (__chunk >= __chunk_ends[__num_segments - 1])
            return;

        // The segment of the chunk: the first one whose chunks end after it, the last one if no other does
        const _Size __segment_idx = oneapi::dpl::__internal::__pstl_upper_bound(
            __chunk_ends, _Size{0}, __num_segments - 1, &__chunk, std::less<>{}, oneapi::dpl::identity{},
            oneapi::dpl::identity{});
        const _Size __segment_first_chunk = __segment_idx == 0 ? _Size{0} : _Size(__chunk_ends[__segment_idx - 1]);
        const _Size __segment_begin =
            oneapi::dpl::__internal::__clamp_to_size_and_cast<_Size>(__offsets[__segment_idx], __n);
        const _Size __segment_end =
            oneapi::dpl::__internal::__clamp_to_size_and_cast<_Size>(__offsets[__segment_idx + 1], __n);
        const _Size __head_length = (__segment_end - __segment_begin) -
                                    (_Size(__chunk_ends[__segment_idx]) - __segment_first_chunk - 1) * __chunk_size;

        // One loop for every chunk, so that the work-items of a sub-group do not diverge into two long loops
        const _Size __chunk_idx_in_segment = __chunk - __segment_first_chunk;
        const bool __is_head = __chunk_idx_in_segment == 0;
        const _Size __chunk_begin =
            __is_head ? __segment_begin : __segment_begin + __head_length + (__chunk_idx_in_segment - 1) * __chunk_size;
        const _Size __chunk_end = __is_head ? __segment_begin + __head_length : __chunk_begin + __chunk_size;
        // The head starts from the initial value; any other chunk starts from the operation on its first two
        // elements, as a partial result of std::reduce does: the accumulator is never made from an element alone,
        // which may mean something else than the element to the operation (such a chunk has at least 64 elements)
        _Acc __start = __init;
        if (!__is_head)
            __start = std::invoke(__binary_op, __input[__chunk_begin], __input[__chunk_begin + 1]);
        __partials[__idx] = oneapi::dpl::__internal::__serial_accumulate<_Acc>(
            __input, __chunk_begin + (__is_head ? 0 : 2), __chunk_end, std::move(__start), __binary_op);
    }
};

// Combines the partial results of the chunks of every segment
template <typename _Acc, typename _Size, typename _BinaryOperation>
struct __segmented_reduce_combine_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;
    static constexpr std::uint8_t __vector_size = 1; // one segment or chunk per work-item, in a loop of its length

    _Size __num_partials = {};  // the number of the computed partial results
    bool __write_empty = false; // whether a segment without chunks gets the initial value; otherwise it is left as is
    _Acc __init;                // the initial value converted to the accumulator type
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _ChunkEnds, typename _Partials, typename _Result>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _ChunkEnds __chunk_ends, _Partials __partials,
               _Result __result) const
    {
        const _Size __first = __idx == 0 ? _Size{0} : _Size(__chunk_ends[__idx - 1]);
        // A segment without chunks is empty, or short and already reduced. Unordered offsets may give more chunks than
        // computed: their results are unspecified.
        const _Size __last = std::min<_Size>(__chunk_ends[__idx], __num_partials);
        if (__first >= __last)
        {
            if (__write_empty)
                __result[__idx] = __init;
            return;
        }
        __result[__idx] = oneapi::dpl::__internal::__serial_accumulate<_Acc>(__partials, __first + 1, __last,
                                                                             _Acc(__partials[__first]), __binary_op);
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

// Writes the initial value into every result: every segment is empty, because the input is or because the offsets
// give no element to any segment
template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Size, typename _Acc>
void
__parallel_reduce_by_segment_offsets_fill(_BackendTag, _ExecutionPolicy&& __exec, _Range&& __result, _Size __m,
                                          const _Acc& __init)
{
    using _Fill = oneapi::dpl::__internal::__brick_fill<oneapi::dpl::__internal::__hetero_tag<_BackendTag>, _Acc>;
    // A separate kernel name: the reduction kernel of the same call is instantiated too
    __parallel_for(_BackendTag{},
                   make_wrapped_policy<__segmented_reduce_fill_wrapper>(std::forward<_ExecutionPolicy>(__exec)),
                   unseq_backend::walk_n_vectors_or_scalars<_Fill>{_Fill{__init}, static_cast<std::size_t>(__m)}, __m,
                   std::forward<_Range>(__result))
        .event()
        .wait_and_throw();
}

// The segments of the input [0, __n) defined by the __m + 1 offsets, reduced into the __m results
template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange, typename _OutRange,
          typename _Size, typename _Acc, typename _BinaryOperation>
void
__parallel_reduce_by_segment_offsets(_BackendTag, _ExecutionPolicy&& __exec, _InRange&& __input,
                                     _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n, const _Size __m,
                                     _Acc __init, _BinaryOperation __binary_op)
{
    const sycl::device __device = __exec.queue().get_device();
    const bool __is_gpu = __device.is_gpu();
    const _Size __threshold =
        __is_gpu ? __segmented_reduce_gpu_short_segment_max : __segmented_reduce_cpu_short_segment_max;

    // Reduces the segments longer than __min_length in chunks; __max_segments bounds the number of such segments
    auto __reduce_in_chunks = [&](_Size __min_length, bool __write_empty, _Size __max_length, _Size __max_segments) {
        const _Size __chunk_size = __segmented_reduce_chunk_size(__is_gpu, __max_length);

        // The end of the chunks of every segment: an inclusive scan of their numbers, computed from the pairs of
        // the adjacent offsets. Its future keeps the temporary data of the scan until the partial results are
        // combined, which waits for all three steps: each one starts on the device after the previous one. The
        // temporary data are device USM or, without it, SYCL buffers.
        sycl::queue __q = __exec.queue();
        __block_storage<_Size, 0> __chunk_ends{__q, static_cast<std::size_t>(__m)};
        auto __scan = __parallel_transform_scan(
            _BackendTag{}, make_wrapped_policy<__segmented_reduce_scan_wrapper>(__exec),
            oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::take_view_simple(__offsets, __m),
                                                 oneapi::dpl::__ranges::drop_view_simple(__offsets, _Size{1})),
            __chunk_ends.__all_view(), __m, __segmented_reduce_chunk_count<_Size>{__n, __chunk_size, __min_length},
            unseq_backend::__no_init_value<_Size>{}, std::plus<_Size>{}, /*_Inclusive=*/std::true_type{});

        // At most n / __chunk_size + 1 chunks per segment, unless the offsets are not ordered
        const _Size __num_partials = __n / __chunk_size + __max_segments + 1;
        __block_storage<_Acc, 0> __partials{__q, static_cast<std::size_t>(__num_partials)};

        auto __chunks = __parallel_for(
            _BackendTag{}, make_wrapped_policy<__segmented_reduce_chunk_wrapper>(__exec), __scan.event(),
            __segmented_reduce_chunk_brick<_Acc, _Size, _BinaryOperation>{__n, __m, __chunk_size, __init, __binary_op},
            __num_partials, __chunk_ends.__all_view(), __offsets, __partials.__all_view(), __input);
        __parallel_for(_BackendTag{}, make_wrapped_policy<__segmented_reduce_combine_wrapper>(__exec), __chunks.event(),
                       __segmented_reduce_combine_brick<_Acc, _Size, _BinaryOperation>{__num_partials, __write_empty,
                                                                                       __init, __binary_op},
                       __m, __chunk_ends.__all_view(), __partials.__all_view(), __result)
            .event()
            .wait_and_throw();
    };

    // The average segment is long, or the device cannot report the longest length of the short segments' pass:
    // all the segments in chunks, which needs no atomics. The longest one selects the chunk length on GPUs.
    if (__n / __m > __threshold || !__has_device_scope_atomics(__device))
    {
        const _Size __max_length =
            __is_gpu
                ? __parallel_transform_reduce<_Size, std::true_type>(
                      _BackendTag{}, make_wrapped_policy<__segmented_reduce_length_wrapper>(__exec),
                      oneapi::dpl::maximum<_Size>{},
                      unseq_backend::walk_n<__segmented_reduce_length<_Size>>{__segmented_reduce_length<_Size>{__n}},
                      unseq_backend::__init_value<_Size>{_Size{0}},
                      oneapi::dpl::__ranges::take_view_simple(__offsets, __m),
                      oneapi::dpl::__ranges::drop_view_simple(__offsets, _Size{1}))
                : __n;
        if (__max_length == 0)
        {
            __parallel_reduce_by_segment_offsets_fill(_BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
                                                      std::forward<_OutRange>(__result), __m, __init);
            return;
        }
        __reduce_in_chunks(_Size{0}, /*__write_empty=*/true, __max_length, __m);
        return;
    }

    // The short segments, one work-item each; the long ones only report the longest length. The report is in a buffer,
    // which costs less on GPUs than copying it from device USM.
    oneapi::dpl::__par_backend_hetero::__buffer<std::uint32_t> __max_long_segment_length_storage(1);
    auto __max_long_segment_length_buf = __max_long_segment_length_storage.get_buffer();

    // The buffer has no host data: the atomic maximum starts from the zero written here
    __max_long_segment_length_buf.get_host_access(sycl::write_only)[0] = 0;
    __parallel_for(
        _BackendTag{}, __exec,
        __segmented_reduce_variable_brick<_Acc, _Size, _BinaryOperation>{__n, __threshold, __init, __binary_op}, __m,
        __offsets, __result, __input, oneapi::dpl::__ranges::views::all(__max_long_segment_length_buf))
        .event()
        .wait_and_throw();

    // The long segments in chunks: at most n / __threshold of them
    const _Size __max_length = __max_long_segment_length_buf.get_host_access(sycl::read_only)[0];
    if (__max_length > 0)
        __reduce_in_chunks(__threshold, /*__write_empty=*/false, __max_length, __n / __threshold);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
