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

#include <algorithm>   // std::min, std::clamp
#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint8_t, std::uint32_t, std::uint64_t
#include <functional>  // std::invoke, std::less, std::plus
#include <type_traits> // std::true_type
#include <utility>     // std::move, std::forward

#include "sycl_defs.h"
#include "parallel_backend_sycl.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

#include "../../functional_impl.h" // oneapi::dpl::identity
#include "../../utils.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// reduce_by_segment with segments defined by offsets

// A short segment is reduced by one work-item. A long segment is split into chunks, each reduced by one work-item, and
// the partial results of the segment are combined by one work-item. The values below were measured on GPUs and CPUs
// as SYCL devices:
// - a segment up to this length is short;
inline constexpr std::size_t __segmented_reduce_gpu_short_segment_max = 512;
inline constexpr std::size_t __segmented_reduce_cpu_short_segment_max = 32768;
// - on GPUs the best chunk length is close to the square root of the length of the segment, which balances the serial
//   work of the two levels, within these bounds; on CPUs the chunk length hardly matters above 1024.
inline constexpr std::size_t __segmented_reduce_gpu_min_chunk_size = 64;
inline constexpr std::size_t __segmented_reduce_gpu_max_chunk_size = 16384;
inline constexpr std::size_t __segmented_reduce_cpu_chunk_size = 4096;

// The chunk length of a long segment, derived from its own length only: a power of two close to the square root of the
// length, within the bounds (both equal on CPUs)
template <typename _Size>
struct __segmented_reduce_chunk_length
{
    _Size __min_chunk_size = {};
    _Size __max_chunk_size = {};

    _Size
    operator()(const _Size __length) const
    {
        // The floor of the binary logarithm of the length, which is positive
        const std::uint32_t __log2 =
            std::uint32_t{63} - static_cast<std::uint32_t>(sycl::clz(static_cast<std::uint64_t>(__length)));
        return std::clamp<_Size>(_Size{2} << (__log2 / 2), __min_chunk_size, __max_chunk_size);
    }
};

// One work-item per short segment, which also writes the empty ones. A longer segment is left to the reduction in
// chunks and only sets the flag of the long segments, by an atomic store of the work-group scope that every device
// supports: every writer stores the same value and the host reads it only after the kernel, so the atomics on host USM
// need not be atomic with respect to the host (sycl::aspect::usm_atomic_host_allocations).
template <typename _Acc, typename _Size, typename _BinaryOperation>
struct __segmented_reduce_short_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;
    static constexpr std::uint8_t __vector_size = 1; // one segment or chunk per work-item, in a loop of its length

    _Size __n = {};         // the size of the input
    _Size __threshold = {}; // the length of the longest short segment
    _Acc __init;            // the initial value converted to the accumulator type
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _Offsets, typename _Result, typename _Input,
              typename _HasLongBuf>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Offsets __offsets, _Result __result, _Input __input,
               _HasLongBuf __has_long_buf) const
    {
        const _Size __begin = oneapi::dpl::__internal::__clamp_to_size(__offsets[__idx], __n);
        const _Size __end = oneapi::dpl::__internal::__clamp_to_size(__offsets[__idx + 1], __n);

        if (__end - __begin > __threshold)
        {
            __dpl_sycl::__atomic_ref<std::uint32_t, sycl::access::address_space::global_space> __ref(__has_long_buf[0]);
            __ref.store(1);
            return;
        }

        __result[__idx] =
            oneapi::dpl::__internal::__serial_accumulate<_Acc>(__input, __begin, __end, __init, __binary_op);
    }
};

// The flag of the long segments, set by the short pass and read by the host: host USM where a device writes it (Level
// Zero GPUs), which the host reads without a copy from the device, or else device USM or, without USM, a SYCL buffer,
// as a __result_storage is
struct __segmented_reduce_flag : public __result_storage<std::uint32_t>
{
    // The view a brick of __parallel_for takes as a range
    struct __view
    {
        using value_type = std::uint32_t;
        using __acc_t = sycl::accessor<std::uint32_t, 1, sycl::access_mode::read_write, __dpl_sycl::__target_device>;
        std::uint32_t* __data = nullptr;
        __acc_t __acc;

        // Element access is only valid in device code
        std::uint32_t&
        operator[](std::size_t __idx) const
        {
            return __data ? __data[__idx] : __acc[__idx];
        }

        std::size_t
        size() const
        {
            return 1;
        }

        // ADL-discoverable call used by __ranges::__require_access in utils_ranges_sycl.h
        friend void
        __require_access_range(sycl::handler& __cgh, __view& __v)
        {
            if (__v.__acc.size() != 0)
                __cgh.require(__v.__acc);
        }
    };

    // The flag starts unset
    explicit __segmented_reduce_flag(const sycl::queue& __q) : __result_storage<std::uint32_t>(__q, 1)
    {
        const std::uint32_t __unset = 0;
        if (this->__kind == sycl::usm::alloc::host)
            *this->__usm_buf.get() = __unset;
        else if (this->__usm_buf)
            this->__usm_buf.get_deleter().__q->memcpy(this->__usm_buf.get(), &__unset, sizeof(__unset)).wait();
        else
            this->__sycl_buf.get_host_access(sycl::write_only)[0] = __unset;
    }

    __view
    __all_view()
    {
        // As __block_storage: checking the buffer size covers both "no device USM" and
        // _ONEDPL_SYCL2020_DEFAULT_ACCESSOR_CONSTRUCTOR_BROKEN
        if (this->__sycl_buf.size() != 0)
            return __view{this->__usm_buf.get(), {this->__sycl_buf}};
        return __view{this->__usm_buf.get(), {}};
    }

    // Valid after the kernel that sets it has completed
    bool
    __is_set()
    {
        std::uint32_t __value = 0;
        this->__copy_result(&__value, 1);
        return __value != 0;
    }
};

// The number of chunks of a long segment given by the pair of its offsets, 0 for a short one
template <typename _Size>
struct __segmented_reduce_chunk_count
{
    _Size __n = {};         // the size of the input
    _Size __threshold = {}; // the length of the longest short segment
    __segmented_reduce_chunk_length<_Size> __chunk_length;

    template <typename _Offset>
    _Size
    operator()(_Offset __first, _Offset __last) const
    {
        const _Size __begin = oneapi::dpl::__internal::__clamp_to_size(__first, __n);
        const _Size __end = oneapi::dpl::__internal::__clamp_to_size(__last, __n);
        const _Size __length = __end - __begin;
        return __length > __threshold ? oneapi::dpl::__internal::__dpl_ceiling_div(__length, __chunk_length(__length))
                                      : _Size{0};
    }

    // The same for the pair of the offsets as a tuple, the input of the scan
    template <typename _Bounds>
    _Size
    operator()(const _Bounds& __bounds) const
    {
        return (*this)(std::get<0>(__bounds), std::get<1>(__bounds));
    }
};

// Reduces one chunk of a long segment. The first chunk of a segment, its head, takes the remainder of its length and
// starts from the initial value converted to the accumulator type; every other chunk has exactly the chunk length of
// the segment and starts from the operation on its first two elements.
template <typename _Acc, typename _Size, typename _BinaryOperation>
struct __segmented_reduce_chunk_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;
    static constexpr std::uint8_t __vector_size = 1; // one segment or chunk per work-item, in a loop of its length

    _Size __n = {};            // the size of the input
    _Size __num_segments = {}; // the number of segments
    __segmented_reduce_chunk_length<_Size> __chunk_length;
    _Acc __init; // the initial value converted to the accumulator type
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _ChunkEnds, typename _Offsets, typename _Partials,
              typename _Input>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _ChunkEnds __chunk_ends, _Offsets __offsets,
               _Partials __partials, _Input __input) const
    {
        const _Size __chunk = __idx;

        // The segment of the chunk: the first one whose chunks end after it, the last one if no other does. The chunk
        // ends are a block storage, indexed by std::size_t.
        const std::size_t __segment_idx = oneapi::dpl::__internal::__pstl_upper_bound(
            __chunk_ends, std::size_t{0}, std::size_t(__num_segments - 1), &__chunk, std::less<>{},
            oneapi::dpl::identity{}, oneapi::dpl::identity{});
        const _Size __segment_first_chunk = __segment_idx == 0 ? _Size{0} : _Size(__chunk_ends[__segment_idx - 1]);
        const _Size __segment_begin = oneapi::dpl::__internal::__clamp_to_size(__offsets[__segment_idx], __n);
        const _Size __segment_end = oneapi::dpl::__internal::__clamp_to_size(__offsets[__segment_idx + 1], __n);
        const _Size __chunk_size = __chunk_length(__segment_end - __segment_begin);
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

// Combines the partial results of the chunks of every long segment
template <typename _Acc, typename _Size, typename _BinaryOperation>
struct __segmented_reduce_combine_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;
    static constexpr std::uint8_t __vector_size = 1; // one segment or chunk per work-item, in a loop of its length

    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _ChunkEnds, typename _Partials, typename _Result>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _ChunkEnds __chunk_ends, _Partials __partials,
               _Result __result) const
    {
        const _Size __first = __idx == 0 ? _Size{0} : _Size(__chunk_ends[__idx - 1]);
        const _Size __last = __chunk_ends[__idx];

        // A segment without chunks is short and already reduced
        if (__first >= __last)
            return;

        // The partials are a block storage, indexed by std::size_t
        __result[__idx] = oneapi::dpl::__internal::__serial_accumulate<_Acc>(
            __partials, std::size_t(__first) + 1, std::size_t(__last), _Acc(__partials[std::size_t(__first)]),
            __binary_op);
    }
};

template <typename _Name>
class __segmented_reduce_fill_wrapper;
template <typename _Name>
class __segmented_reduce_scan_wrapper;
template <typename _Name>
class __segmented_reduce_chunk_wrapper;
template <typename _Name>
class __segmented_reduce_combine_wrapper;

// Writes the initial value into every result: every segment is empty, because the input is
template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Size, typename _Acc>
void
__parallel_reduce_by_segment_offsets_fill(_BackendTag, _ExecutionPolicy&& __exec, _Range&& __result, _Size __m,
                                          const _Acc& __init)
{
    using _Fill = oneapi::dpl::__internal::__brick_fill<oneapi::dpl::__internal::__hetero_tag<_BackendTag>, _Acc>;
    // A separate kernel name: the reduction kernels of the same call are instantiated too
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
    const bool __is_gpu = __exec.queue().get_device().is_gpu();
    const _Size __threshold =
        __is_gpu ? __segmented_reduce_gpu_short_segment_max : __segmented_reduce_cpu_short_segment_max;
    const __segmented_reduce_chunk_length<_Size> __chunk_length{
        _Size(__is_gpu ? __segmented_reduce_gpu_min_chunk_size : __segmented_reduce_cpu_chunk_size),
        _Size(__is_gpu ? __segmented_reduce_gpu_max_chunk_size : __segmented_reduce_cpu_chunk_size)};

    // The short segments, one work-item each, which also writes the empty ones; the long ones only set the flag
    __segmented_reduce_flag __has_long{__exec.queue()};
    __parallel_for(_BackendTag{}, __exec,
                   __segmented_reduce_short_brick<_Acc, _Size, _BinaryOperation>{__n, __threshold, __init, __binary_op},
                   __m, __offsets, __result, __input, __has_long.__all_view())
        .event()
        .wait_and_throw();

    // The long segments in chunks, if any: none if no segment is longer than the input
    if (__n <= __threshold || !__has_long.__is_set())
        return;

    sycl::queue __q = __exec.queue();
    // The end of the chunks of every long segment: an inclusive scan of their numbers, computed from the pairs of the
    // adjacent offsets, independently of the other segments; its future keeps the temporary data of the scan until the
    // end. The temporary data are device USM or, without it, SYCL buffers.
    __block_storage<_Size, 0> __chunk_ends{__q, static_cast<std::size_t>(__m)};

    auto __scan_f = __parallel_transform_scan(
        _BackendTag{}, make_wrapped_policy<__segmented_reduce_scan_wrapper>(__exec),
        oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::take_view_simple(__offsets, __m),
                                             oneapi::dpl::__ranges::drop_view_simple(__offsets, _Size{1})),
        __chunk_ends.__all_view(), __m, __segmented_reduce_chunk_count<_Size>{__n, __threshold, __chunk_length},
        unseq_backend::__no_init_value<_Size>{}, std::plus<_Size>{}, /*_Inclusive=*/std::true_type{});

    // The number of the chunks: the last chunk end, the only element copied to the host
    __scan_f.event().wait_and_throw();
    const _Size __num_partials = __chunk_ends.__get_data_at(static_cast<std::size_t>(__m - 1));

    // The combination starts on the device after the chunks
    __block_storage<_Acc, 0> __partials{__q, static_cast<std::size_t>(__num_partials)};

    auto __chunks_f = __parallel_for(
        _BackendTag{}, make_wrapped_policy<__segmented_reduce_chunk_wrapper>(__exec),
        __segmented_reduce_chunk_brick<_Acc, _Size, _BinaryOperation>{__n, __m, __chunk_length, __init, __binary_op},
        __num_partials, __chunk_ends.__all_view(), __offsets, __partials.__all_view(), __input);
    __parallel_for(_BackendTag{}, make_wrapped_policy<__segmented_reduce_combine_wrapper>(__exec), __chunks_f.event(),
                   __segmented_reduce_combine_brick<_Acc, _Size, _BinaryOperation>{__binary_op}, __m,
                   __chunk_ends.__all_view(), __partials.__all_view(), __result)
        .event()
        .wait_and_throw();
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
