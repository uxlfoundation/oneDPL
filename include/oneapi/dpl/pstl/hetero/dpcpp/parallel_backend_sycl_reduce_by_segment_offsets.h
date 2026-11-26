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
#include <tuple>       // std::get
#include <type_traits> // std::is_signed_v, std::true_type
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

// One work-item per short segment. A longer segment is left to the reduction in chunks and only reports its length
// through an atomic maximum of the device scope; a capped length still selects the same chunk size.
template <typename _Acc, typename _Size, typename _BinaryOperation>
struct __segmented_reduce_short_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;
    static constexpr std::uint8_t __vector_size = 1; // one segment or chunk per work-item, in a loop of its length

    _Size __n = {};                 // the size of the input
    _Size __short_segment_max = {}; // the length of the longest short segment
    _Acc __init;                    // the initial value converted to the accumulator type
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _Offsets, typename _Result, typename _Input,
              typename _MaxLengthView>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Offsets __offsets, _Result __result, _Input __input,
               _MaxLengthView __max_length_view) const
    {
        using _MaxLength = oneapi::dpl::__internal::__value_t<_MaxLengthView>;
        using __atomic_ref_t = sycl::atomic_ref<_MaxLength, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>;

        const _Size __begin = oneapi::dpl::__internal::__clamp_to_size_and_cast<_Size>(__offsets[__idx], __n);
        const _Size __end = oneapi::dpl::__internal::__clamp_to_size_and_cast<_Size>(__offsets[__idx + 1], __n);

        if (__end - __begin > __short_segment_max)
        {
            __atomic_ref_t __ref(__max_length_view[0]);

            __ref.fetch_max(oneapi::dpl::__internal::__clamp_to_size_and_cast<_MaxLength>(
                __end - __begin, std::numeric_limits<_MaxLength>::max()));
            return;
        }

        __result[__idx] =
            oneapi::dpl::__internal::__serial_accumulate<_Acc>(__input, __begin, __end, __init, __binary_op);
    }
};

template <typename... _Name>
class __copy_scratch_to_result_kernel;

// Copies the value of a __combined_storage from its scratch into its result, by one work-item after an event
template <typename _KernelName>
struct __copy_scratch_to_result_submitter;

template <typename... _Name>
struct __copy_scratch_to_result_submitter<__internal::__optional_kernel_name<_Name...>>
{
    template <typename _ValueType>
    sycl::event
    operator()(sycl::queue& __q, const sycl::event& __e, __combined_storage<_ValueType>& __storage) const
    {
        return __q.submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__e);
            auto __scratch_acc = __get_accessor(sycl::read_only, __storage, __cgh);
            auto __result_acc = __get_result_accessor(sycl::write_only, __storage, __cgh, __dpl_sycl::__no_init{});
            __cgh.single_task<_Name...>([=]() { __result_acc.__data()[0] = __scratch_acc.__data()[0]; });
        });
    }
};

// A single zero-initialized value, raised by atomics of a kernel in the scratch, in device memory.
// Where the result is host USM, the value is copied into it on the device, as loading device memory on the host costs
// much on some GPUs; otherwise the host loads the scratch itself.
// Without USM, a SYCL buffer holds both.
template <typename _ValueType>
struct __atomic_result_storage : public __combined_storage<_ValueType>
{
    sycl::event __zero_done; // the completion of the initial zero on the device, if it is not written by the host

    // The view of the scratch a brick of __parallel_for takes as a range
    struct __view
    {
        using value_type = _ValueType;
        using __acc_t = sycl::accessor<_ValueType, 1, sycl::access_mode::read_write, __dpl_sycl::__target_device>;
        _ValueType* __data = nullptr;
        __acc_t __acc;

        // Element access is only valid in device code
        _ValueType&
        operator[](std::size_t __idx) const
        {
            return __data ? __data[__idx] : __acc[__idx];
        }

        // The elements addressable through the view: __parallel_for checks its ranges are not empty in debug builds
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

    // The scratch comes first in device USM and in the SYCL buffer
    explicit __atomic_result_storage(sycl::queue& __q) : __combined_storage<_ValueType>(__q, 1, 1)
    {
        if (this->__usm_buf)
            __zero_done = __q.fill(this->__usm_buf.get(), _ValueType{}, 1);
        else
            this->__sycl_buf->get_host_access(sycl::write_only)[0] = _ValueType{};
    }

    __view
    __all_view()
    {
        // checking the buffer size is the simplest way to cover both "no device USM"
        // and _ONEDPL_SYCL2020_DEFAULT_ACCESSOR_CONSTRUCTOR_BROKEN
        if (this->__sycl_buf)
            return __view{this->__usm_buf.get(), {*this->__sycl_buf}};
        else
            return __view{this->__usm_buf.get(), {}};
    }

    // Waits for the kernel that raised the value, the event __e, copies it into a result in host USM, and loads it
    template <typename _ExecutionPolicy>
    _ValueType
    __load_result(_ExecutionPolicy&& __exec, sycl::event __e)
    {
        _ValueType __result = {};

        // A result in host USM gets the value copied into it; otherwise the host loads the scratch itself
        if (this->__kind == sycl::usm::alloc::host)
        {
            using _CopyScratchToResultKernel = __internal::__kernel_name_provider<
                __copy_scratch_to_result_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;
            sycl::queue __q = __exec.queue();
            __copy_scratch_to_result_submitter<_CopyScratchToResultKernel>{}(__q, __e, *this).wait_and_throw();
            this->__copy_result(&__result, 1);
        }
        else
        {
            __e.wait_and_throw();
            this->__copy_n(&__result, nullptr, 1, /*offset*/ 0); // the scratch, first in device USM and in the buffer
        }

        return __result;
    }
};

template <typename _Name>
class __segmented_reduce_fill_wrapper;

// Whether a device supports the atomics of the device scope, which SYCL does not require
inline bool
__has_device_scope_atomics([[maybe_unused]] sycl::device __device)
{
#if _ONEDPL_SYCL2020_ATOMIC_MEMORY_SCOPE_CAPABILITIES_BROKEN
    // The capabilities cannot be queried, so conservatively assume no support
    return false;
#else
    const std::vector<sycl::memory_scope> __scopes =
        __device.get_info<sycl::info::device::atomic_memory_scope_capabilities>();
    return std::find(__scopes.begin(), __scopes.end(), sycl::memory_scope::device) != __scopes.end();
#endif
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
        const _Size __chunk_idx = __idx;
        if (__chunk_idx >= __chunk_ends[__num_segments - 1])
            return;

        // The segment of the chunk: the first one whose chunks end after it, the last one if no other does
        const _Size __segment_idx = oneapi::dpl::__internal::__pstl_upper_bound(
            __chunk_ends, _Size{0}, __num_segments - 1, &__chunk_idx, std::less<>{}, oneapi::dpl::identity{},
            oneapi::dpl::identity{});
        const _Size __segment_first_chunk = __segment_idx == 0 ? _Size{0} : __chunk_ends[__segment_idx - 1];
        const _Size __segment_begin =
            oneapi::dpl::__internal::__clamp_to_size_and_cast<_Size>(__offsets[__segment_idx], __n);
        const _Size __segment_end =
            oneapi::dpl::__internal::__clamp_to_size_and_cast<_Size>(__offsets[__segment_idx + 1], __n);
        const _Size __head_length = (__segment_end - __segment_begin) -
                                    (__chunk_ends[__segment_idx] - __segment_first_chunk - 1) * __chunk_size;

        // One loop for every chunk, so that the work-items of a sub-group do not diverge into two long loops
        const _Size __chunk_idx_in_segment = __chunk_idx - __segment_first_chunk;
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
        const _Size __first = __idx == 0 ? _Size{0} : __chunk_ends[__idx - 1];
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
                                                                             __partials[__first], __binary_op);
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
    __parallel_for(
        _BackendTag{}, make_wrapped_policy<__segmented_reduce_fill_wrapper>(std::forward<_ExecutionPolicy>(__exec)),
        unseq_backend::walk_n_vectors_or_scalars<_Fill>(_Fill{__init}, __m), __m, std::forward<_Range>(__result))
        .event()
        .wait_and_throw();
}

// A short segment is reduced by one work-item. A long segment is split into chunks, each reduced by one work-item, and
// the partial results of the segment are combined by one work-item. The values below were measured on GPUs and CPUs
// as SYCL devices.
class __segmented_reduce_chunk_tuning
{
    // A segment up to this length is short
    static constexpr std::size_t __gpu_short_segment_max = 512;
    static constexpr std::size_t __cpu_short_segment_max = 32768;

    // On GPUs the best chunk length is close to the square root of the longest segment, which balances the serial work
    // of the two levels, within these bounds; on CPUs the chunk length hardly matters above 1024
    static constexpr std::size_t __gpu_min_chunk_size = 64;
    static constexpr std::size_t __gpu_max_chunk_size = 16384;
    static constexpr std::size_t __cpu_chunk_size = 4096;

  public:
    static std::size_t
    __short_segment_max(bool __is_gpu)
    {
        return __is_gpu ? __gpu_short_segment_max : __cpu_short_segment_max;
    }

    static std::size_t
    __chunk_size(bool __is_gpu, std::size_t __max_length)
    {
        // The serial work for the longest segment of length L is chunk + L / chunk: one work-item reduces a chunk,
        // another combines the L / chunk partial results; the chunk length sqrt(L) makes the sum the smallest
        return __is_gpu ? std::clamp<std::size_t>(
                              oneapi::dpl::__internal::__dpl_bit_ceil(std::size_t(std::sqrt(__max_length))),
                              __gpu_min_chunk_size, __gpu_max_chunk_size)
                        : __cpu_chunk_size;
    }
};

// The segments of the input [0, __n) defined by the __m + 1 offsets, reduced into the __m results
template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange, typename _OutRange,
          typename _Size, typename _Acc, typename _BinaryOperation>
void
__parallel_reduce_by_segment_offsets(_BackendTag, _ExecutionPolicy&& __exec, _InRange&& __input,
                                     _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n, const _Size __m,
                                     _Acc __init, _BinaryOperation __binary_op)
{
    // Signed, so that a reversed segment (end < begin) is short and gets the initial value
    static_assert(std::is_signed_v<_Size>, "The size type must be signed");

    sycl::queue __q = __exec.queue();
    const sycl::device __device = __q.get_device();
    const bool __is_gpu = __device.is_gpu();

    // Reduces the segments longer than __min_length in chunks; __max_segments bounds the number of such segments
    auto __reduce_in_chunks = [&](_Size __min_length, bool __write_empty, _Size __max_length, _Size __max_segments) {
        const _Size __chunk_size = __segmented_reduce_chunk_tuning::__chunk_size(__is_gpu, __max_length);

        // The end of the chunks of every segment: an inclusive scan of their numbers, computed from the pairs of
        // the adjacent offsets. Its future keeps the temporary data of the scan until the partial results are
        // combined, which waits for all three steps: each one starts on the device after the previous one. The
        // temporary data are device USM or, without it, SYCL buffers.
        __block_storage<_Size, 0> __chunk_ends(__q, __m);
        auto __scan = __parallel_transform_scan(
            _BackendTag{}, make_wrapped_policy<__segmented_reduce_scan_wrapper>(__exec),
            oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::take_view_simple(__offsets, __m),
                                                 oneapi::dpl::__ranges::drop_view_simple(__offsets, _Size{1})),
            __chunk_ends.__all_view(), __m, __segmented_reduce_chunk_count<_Size>{__n, __chunk_size, __min_length},
            unseq_backend::__no_init_value<_Size>{}, std::plus<_Size>{}, /*_Inclusive=*/std::true_type{});

        // At most n / __chunk_size + 1 chunks per segment, unless the offsets are not ordered
        const _Size __num_partials = __n / __chunk_size + __max_segments + 1;
        __block_storage<_Acc, 0> __partials(__q, __num_partials);

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

    const _Size __short_segment_max = __segmented_reduce_chunk_tuning::__short_segment_max(__is_gpu);

    // Long segments on average, or no atomics for the short-segment pass: all the segments in chunks
    if (__n / __m > __short_segment_max || !__has_device_scope_atomics(__device))
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
            __parallel_reduce_by_segment_offsets_fill(_BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
                                                      std::forward<_OutRange>(__result), __m, __init);
        else
            __reduce_in_chunks(_Size{0}, /*__write_empty=*/true, __max_length, __m);

        return;
    }

    // The short segments, one work-item each; the long ones only report the longest length
    // 32 bits: the atomics of every device; the cap of the length by the brick does not change the chunk size
    __atomic_result_storage<std::uint32_t> __max_length_storage(__q);
    sycl::event __short_done = __parallel_for(_BackendTag{}, __exec, __max_length_storage.__zero_done,
                                              __segmented_reduce_short_brick<_Acc, _Size, _BinaryOperation>{
                                                  __n, __short_segment_max, __init, __binary_op},
                                              __m, __offsets, __result, __input, __max_length_storage.__all_view())
                                   .event();

    // The long segments in chunks: at most n / __short_segment_max of them
    const _Size __max_length = __max_length_storage.__load_result(__exec, __short_done);
    if (__max_length > 0)
        __reduce_in_chunks(__short_segment_max, /*__write_empty=*/false, __max_length, __n / __short_segment_max);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
