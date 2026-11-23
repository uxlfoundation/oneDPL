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

#include <algorithm>   // std::min, std::max, std::clamp, std::find
#include <cassert>     // assert
#include <cmath>       // std::sqrt
#include <cstddef>     // std::size_t
#include <cstdint>     // std::int32_t, std::uint8_t, std::uint32_t, std::uintptr_t
#include <functional>  // std::invoke, std::less, std::plus
#include <limits>      // std::numeric_limits
#include <tuple>       // std::get
#include <type_traits> // std::integral_constant, std::conditional_t, std::enable_if_t, std::is_*_v, ...
#include <utility>     // std::move, std::forward, std::declval, std::pair
#include <vector>      // std::vector

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

// A work-item per short segment; a long one only reports its length (atomic max, capped to 32 bits)
template <typename _Acc, typename _Size, typename _BinaryOperation>
struct __segmented_reduce_short_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;
    static constexpr std::uint8_t __vector_size = 1; // one segment or chunk per work-item, in a loop of its length

    _Size __n = {};                 // input size
    _Size __short_segment_max = {}; // longest short segment length
    _Acc __init;                    // init converted to _Acc
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _Offsets, typename _Result, typename _Input,
              typename _MaxLengthView>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Offsets __offsets, _Result __result, _Input __input,
               _MaxLengthView __max_length_view) const
    {
        using oneapi::dpl::__internal::__clamp_to_size_and_cast;
        using oneapi::dpl::__internal::__serial_accumulate;
        using _MaxLength = oneapi::dpl::__internal::__value_t<_MaxLengthView>;
        using __atomic_ref_t = sycl::atomic_ref<_MaxLength, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>;

        const _Size __begin = __clamp_to_size_and_cast<_Size>(__offsets[__idx], __n);
        const _Size __end = __clamp_to_size_and_cast<_Size>(__offsets[__idx + 1], __n);

        // A reversed segment has a negative length: short, its reduce gives the init
        static_assert(std::is_signed_v<_Size>, "The size type must be signed");
        const _Size __length = __end - __begin;
        if (__length > __short_segment_max)
        {
            __atomic_ref_t __ref(__max_length_view[0]);

            __ref.fetch_max(__clamp_to_size_and_cast<_MaxLength>(__length, std::numeric_limits<_MaxLength>::max()));
        }
        else
        {
            __result[__idx] = __serial_accumulate<_Acc>(__input, __begin, __end, __init, __binary_op);
        }
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
            this->__sycl_buf.get_host_access(sycl::write_only)[0] = _ValueType{};
    }

    __view
    __all_view()
    {
        // checking the buffer size is the simplest way to cover both "no device USM"
        // and _ONEDPL_SYCL2020_DEFAULT_ACCESSOR_CONSTRUCTOR_BROKEN
        if (this->__sycl_buf.size() != 0)
            return __view{this->__usm_buf.get(), {this->__sycl_buf}};
        else
            return __view{this->__usm_buf.get(), {}};
    }

    // Waits for the kernel that raised the value, the event __e, copies it into a result in host USM, and loads it
    template <typename _ExecutionPolicy>
    _ValueType
    __load_result(_ExecutionPolicy&&, sycl::queue& __q, sycl::event __e)
    {
        _ValueType __result = {};

        // A result in host USM gets the value copied into it; otherwise the host loads the scratch itself
        if (this->__kind == sycl::usm::alloc::host)
        {
            using _CopyScratchToResultKernel = __internal::__kernel_name_provider<
                __copy_scratch_to_result_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;

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

// Short segment: one work-item; long: chunks per work-item, then combined by one work-item. Tuned on GPUs and CPUs
class __segmented_reduce_chunk_tuning
{
    // A segment up to this length is short
    static constexpr std::size_t __gpu_short_segment_max = 512;
    static constexpr std::size_t __cpu_short_segment_max = 32768;

    // GPU: best chunk ~ sqrt(longest segment), balancing both levels, within these bounds; CPU: hardly matters >1024
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

// Chunks of a segment (0 if not longer than __min_length), scanned into chunk ends; __length - 1 >= 0 when used
template <typename _Size>
struct __segmented_reduce_chunk_count
{
    _Size __n = {};          // input size
    _Size __chunk_size = {}; // chunk length
    _Size __min_length = {}; // segments not longer than this (>= 0) have no chunks

    template <typename _Bounds>
    _Size
    operator()(const _Bounds& __bounds) const
    {
        using oneapi::dpl::__internal::__clamp_to_size_and_cast;
        const _Size __begin = __clamp_to_size_and_cast<_Size>(std::get<0>(__bounds), __n);
        const _Size __end = __clamp_to_size_and_cast<_Size>(std::get<1>(__bounds), __n);

        // A reversed segment has a negative length, hence no chunks
        static_assert(std::is_signed_v<_Size>, "The size type must be signed");
        const _Size __length = __end - __begin;
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

    _Size __n = {};            // input size
    _Size __num_segments = {}; // number of segments
    _Size __chunk_size = {};   // chunk length
    _Acc __init;               // init converted to _Acc
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _ChunkEnds, typename _Offsets, typename _Partials,
              typename _Input>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _ChunkEnds __chunk_ends, _Offsets __offsets,
               _Partials __partials, _Input __input) const
    {
        using oneapi::dpl::__internal::__clamp_to_size_and_cast;
        using oneapi::dpl::__internal::__serial_accumulate;

        const _Size __chunk_idx = __idx;
        if (__chunk_idx >= __chunk_ends[__num_segments - 1])
            return;

        // The segment of the chunk: the first one whose chunks end after it, the last one if no other does
        const _Size __segment_idx = oneapi::dpl::__internal::__pstl_upper_bound(
            __chunk_ends, _Size{0}, __num_segments - 1, &__chunk_idx, std::less<>{}, oneapi::dpl::identity{},
            oneapi::dpl::identity{});
        const _Size __segment_first_chunk = __segment_idx == 0 ? _Size{0} : __chunk_ends[__segment_idx - 1];
        const _Size __segment_begin = __clamp_to_size_and_cast<_Size>(__offsets[__segment_idx], __n);
        const _Size __segment_end = __clamp_to_size_and_cast<_Size>(__offsets[__segment_idx + 1], __n);

        // Head length: the remainder after the full chunks, in [1, __chunk_size]
        const _Size __head_length = (__segment_end - __segment_begin) -
                                    (__chunk_ends[__segment_idx] - __segment_first_chunk - 1) * __chunk_size;

        // One loop for every chunk, so sub-group work-items do not diverge into two long loops
        const _Size __chunk_idx_in_segment = __chunk_idx - __segment_first_chunk;
        const bool __is_head = __chunk_idx_in_segment == 0;
        const _Size __chunk_begin =
            __is_head ? __segment_begin : __segment_begin + __head_length + (__chunk_idx_in_segment - 1) * __chunk_size;
        const _Size __chunk_end = __is_head ? __segment_begin + __head_length : __chunk_begin + __chunk_size;

        // Chunks other than the head start from op(first two elements), as std::reduce does: an accumulator made from a
        // lone element may mean something else to the op. Unlike __reduce_pair, the elements are not converted to the
        // accumulator type: this path also takes any types, whose accumulator may not be constructible from an element
        _Acc __start = __init;
        if (!__is_head)
            __start = std::invoke(__binary_op, __input[__chunk_begin], __input[__chunk_begin + 1]);

        __partials[__idx] = __serial_accumulate<_Acc>(__input, __chunk_begin + (__is_head ? 0 : 2), __chunk_end,
                                                      std::move(__start), __binary_op);
    }
};

// A work-item per segment combines its chunk partials in chunk order
template <typename _Acc, typename _Size, typename _BinaryOperation>
struct __segmented_reduce_combine_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;
    static constexpr std::uint8_t __vector_size = 1; // one segment or chunk per work-item, in a loop of its length

    _Size __num_partials = {}; // number of computed partial results
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _ChunkEnds, typename _Partials, typename _Result>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _ChunkEnds __chunk_ends, _Partials __partials,
               _Result __result) const
    {
        using oneapi::dpl::__internal::__serial_accumulate;

        const _Size __first = __idx == 0 ? _Size{0} : __chunk_ends[__idx - 1];

        // No chunks: short and already reduced. Unordered offsets may exceed the computed chunks: results unspecified
        const _Size __last = std::min(__chunk_ends[__idx], __num_partials);
        if (__first < __last)
            __result[__idx] = __serial_accumulate<_Acc>(__partials, std::size_t(__first) + 1, std::size_t(__last),
                                                        __partials[__first], __binary_op);
    }
};

// Whether a range is pointer-based, so consecutive elements load as one vector; most views are not
template <typename _Range, typename = void>
inline constexpr bool __has_pointer_begin_v = false;
template <typename _Range>
inline constexpr bool __has_pointer_begin_v<
    _Range, std::enable_if_t<std::is_pointer_v<decltype(std::declval<const _Range&>().begin())>>> = true;

struct __segmented_reduce_flat_geometry
{
    std::size_t __work_group_size = {}; // multiple of the sub-group size
    std::size_t __num_groups = {};
    std::size_t __inputs_per_item = {};
};

// GPU, arithmetic types: each sub-group reduces a piece of consecutive elements, its first segment found by search,
// writing whole segments; a second kernel combines the partial ones. With unsorted offsets (results unspecified)
// a segment may stay unwritten
struct __segmented_reduce_flat_tuning
{
    static constexpr std::size_t __sub_group_size = 32;

    // Min elements per work-item for small inputs, to amortize the search and combination; large inputs take 4x
    static constexpr std::size_t __min_inputs_per_item = 16;
    static constexpr std::size_t __max_inputs_per_item = 4 * __min_inputs_per_item;

    // Small and tiny inputs, up to these elements per hardware thread, take fewer elements per work-item
    static constexpr std::size_t __small_input_max_per_thread = 1024;
    static constexpr std::size_t __tiny_input_max_per_thread = 256;
    static constexpr std::size_t __small_min_inputs_per_item = 8;
    static constexpr std::size_t __tiny_min_inputs_per_item = 4;

    // Work-group size limit of both kernels, also of the work-group per segment path
    static constexpr std::size_t __work_group_size_limit = 256;

    // The largest work-group of whole sub-groups within that limit and the one of the device
    static std::size_t
    __max_work_group(const sycl::queue& __q)
    {
        return std::max(__sub_group_size, oneapi::dpl::__internal::__max_work_group_size(__q, __work_group_size_limit) /
                                              __sub_group_size * __sub_group_size);
    }

    // Whole segments up to this length get a lane each: cheaper than a sub-group per segment
    static constexpr std::size_t __tiny_segment_max = 128;

    // Min whole vectors for sub-group vector loads; shorter ranges go faster per element (alignment/head/tail cost)
    static constexpr std::size_t __min_vectors = 8;

    // Above this input a work-group combines its pieces in local memory before the second kernel
    static constexpr std::size_t __piece_store_input_max = std::size_t(1) << 24;

    // A piece per hardware thread, elements per work-item within the bounds, at least two work-groups per GPU core
    static __segmented_reduce_flat_geometry
    __geometry(const sycl::queue& __q, const std::size_t __n)
    {
        std::size_t __work_group_size = __max_work_group(__q);
        const oneapi::dpl::__internal::__device_compute_units __compute_units(__q);
        const std::size_t __min_groups = 2 * __compute_units.__num_xe_cores;

        const std::size_t __min_inputs =
            __n <= __tiny_input_max_per_thread * __compute_units.__hw_threads    ? __tiny_min_inputs_per_item
            : __n <= __small_input_max_per_thread * __compute_units.__hw_threads ? __small_min_inputs_per_item
                                                                                 : __min_inputs_per_item;
        const std::size_t __inputs_per_item = std::clamp<std::size_t>(
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __compute_units.__hw_threads * __sub_group_size),
            __min_inputs, __max_inputs_per_item);

        auto __groups = [__n, __inputs_per_item](std::size_t __size) {
            return oneapi::dpl::__internal::__dpl_ceiling_div(__n, __inputs_per_item * __size);
        };

        while (__work_group_size > __sub_group_size && __groups(__work_group_size) < __min_groups)
            __work_group_size = std::max(__sub_group_size, __work_group_size / 2 / __sub_group_size * __sub_group_size);

        return {__work_group_size, __groups(__work_group_size), __inputs_per_item};
    }
};

template <typename... _Name>
class __segmented_reduce_flat_kernel;
template <typename... _Name>
class __segmented_reduce_flat_group_kernel;
template <typename... _Name>
class __segmented_reduce_flat_tail_kernel;

// First-kernel local memory to combine pieces in a work-group: heads, tails, tail segments, head states
template <bool _InGroup, typename _Acc, typename _Size>
struct __segmented_reduce_flat_local
{
    __segmented_reduce_flat_local(sycl::handler&, std::size_t) {}
};

// The head of a piece of the flat path: the part of the segment begun before the piece
enum class __segmented_reduce_head_state : std::uint8_t
{
    __none = 0,      // the piece begins with a segment of its own
    __ends_in_piece, // the segment ends in the piece
    __goes_on        // the segment goes on past the piece
};

template <typename _Acc, typename _Size>
struct __segmented_reduce_flat_local<true, _Acc, _Size>
{
    __dpl_sycl::__local_accessor<_Acc> __loc_heads_tails_acc;
    __dpl_sycl::__local_accessor<_Size> __loc_tail_segments_acc;
    __dpl_sycl::__local_accessor<__segmented_reduce_head_state> __loc_head_states_acc;

    __segmented_reduce_flat_local(sycl::handler& __cgh, std::size_t __sub_groups)
        : __loc_heads_tails_acc(sycl::range<1>(2 * __sub_groups), __cgh),
          __loc_tail_segments_acc(sycl::range<1>(__sub_groups), __cgh),
          __loc_head_states_acc(sycl::range<1>(__sub_groups), __cgh)
    {
    }
};

// Op on two values converted to the accumulator type, so that it does not run in a narrower input type
template <typename _Acc, typename _T1, typename _T2, typename _BinaryOperation>
_Acc
__reduce_pair(_T1 __a, _T2 __b, _BinaryOperation __binary_op)
{
    return _Acc(std::invoke(__binary_op, _Acc(__a), _Acc(__b)));
}

// Combines the first __active lanes onto lane 0: a group reduce with a known identity, else compile-time shifts, as
// run-time ones reach other lanes' registers indirectly. Every lane passes a constructed value, unused off lane 0
template <typename _Acc, typename _SubGroup, typename _BinaryOperation>
_Acc
__reduce_active_lanes(const _SubGroup __sg, _Acc __acc, const std::uint32_t __active, _BinaryOperation __binary_op)
{
    const std::uint32_t __lane = __sg.get_local_linear_id();

    if constexpr (oneapi::dpl::unseq_backend::__can_use_group_reduce_scan<_BinaryOperation, _Acc>::value)
    {
        return __dpl_sycl::__reduce_over_group(
            __sg, __lane < __active ? __acc : sycl::known_identity_v<_BinaryOperation, _Acc>, __binary_op);
    }
    else
    {
        auto __step = [&](auto __shift) {
            if (__shift < __active)
            {
                const _Acc __other = sycl::shift_group_left(__sg, __acc, __shift);
                if (__lane + __shift < __active)
                    __acc = __reduce_pair<_Acc>(__acc, __other, __binary_op);
            }
        };

        // Shifts up to 16 cover the at most 32 lanes; __shift < __active is uniform, as the shift needs all lanes
        __step(std::integral_constant<std::uint32_t, 1>{});
        __step(std::integral_constant<std::uint32_t, 2>{});
        __step(std::integral_constant<std::uint32_t, 4>{});
        __step(std::integral_constant<std::uint32_t, 8>{});
        __step(std::integral_constant<std::uint32_t, 16>{});

        return __acc;
    }
}

// Op on four values as two accumulator pairs; by value so all four load before the op waits on any
template <typename _Acc, typename _T, typename _BinaryOperation>
_Acc
__reduce_four(_T __a, _T __b, _T __c, _T __d, _BinaryOperation __binary_op)
{
    return __reduce_pair<_Acc>(__reduce_pair<_Acc>(__a, __b, __binary_op), __reduce_pair<_Acc>(__c, __d, __binary_op),
                               __binary_op);
}

// Folds __load(__index) for __index below __end by __stride: by eight, four, two and one, the loads of a step at once
template <typename _Acc, typename _Load, typename _BinaryOperation>
_Acc
__fold_strided(_Acc __acc, std::uint32_t __index, const std::uint32_t __end, const std::uint32_t __stride, _Load __load,
               _BinaryOperation __binary_op)
{
    auto __four = [&](std::uint32_t __i) {
        return __reduce_four<_Acc>(__load(__i), __load(__i + __stride), __load(__i + 2 * __stride),
                                   __load(__i + 3 * __stride), __binary_op);
    };
    for (; __index + 7 * __stride < __end; __index += 8 * __stride)
        __acc = __reduce_pair<_Acc>(
            __acc, __reduce_pair<_Acc>(__four(__index), __four(__index + 4 * __stride), __binary_op), __binary_op);
    if (__index + 3 * __stride < __end)
    {
        __acc = __reduce_pair<_Acc>(__acc, __four(__index), __binary_op);
        __index += 4 * __stride;
    }
    if (__index + __stride < __end)
    {
        __acc = __reduce_pair<_Acc>(
            __acc, __reduce_pair<_Acc>(__load(__index), __load(__index + __stride), __binary_op), __binary_op);
        __index += 2 * __stride;
    }
    if (__index < __end)
        __acc = __reduce_pair<_Acc>(__acc, __load(__index), __binary_op);
    return __acc;
}

template <typename _Acc, typename _KernelName, typename _GroupKernelName, typename _TailKernelName,
          typename _BinaryOperation, typename _Tuning>
struct __segmented_reduce_flat_submitter;

template <typename _Acc, typename... _Name, typename... _GroupName, typename... _TailName, typename _BinaryOperation,
          typename _Tuning>
struct __segmented_reduce_flat_submitter<_Acc, __internal::__optional_kernel_name<_Name...>,
                                         __internal::__optional_kernel_name<_GroupName...>,
                                         __internal::__optional_kernel_name<_TailName...>, _BinaryOperation, _Tuning>
{
    _Acc __init; // init converted to _Acc
    mutable _BinaryOperation __binary_op;

    template <typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size>
    void
    operator()(sycl::queue& __q, _InRange&& __input, _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n,
               const _Size __m) const
    {
        // Accumulators in __lazy_ctor_storage are never destroyed: arithmetic types only
        static_assert(std::is_trivially_destructible_v<_Acc>);
        // A reversed segment has a negative length, so it is tiny
        static_assert(std::is_signed_v<_Size>, "The size type must be signed");
        constexpr std::size_t __sub_group_size = _Tuning::__sub_group_size;
        const __segmented_reduce_flat_geometry __geometry = _Tuning::__geometry(__q, __n);
        const std::size_t __work_group_size = __geometry.__work_group_size;
        // Kernel values in 32 bits (counted in pieces and work-groups) so kernel arguments stay in registers
        const std::uint32_t __sub_groups = __work_group_size / __sub_group_size;
        const std::size_t __num_groups = __geometry.__num_groups;
        const std::size_t __inputs_per_item = __geometry.__inputs_per_item;
        const std::uint32_t __piece_size = __inputs_per_item * __sub_group_size;
        const std::uint32_t __num_pieces = __num_groups * __sub_groups;

        // Second-kernel units: work-groups when pieces are combined in local memory, else pieces
        const bool __combine_in_group = std::size_t(__n) > _Tuning::__piece_store_input_max;
        const std::uint32_t __num_units = __combine_in_group ? __num_groups : __num_groups * __sub_groups;
        const std::uint32_t __unit_size = __combine_in_group ? __piece_size * __sub_groups : __piece_size;

        // Unit tail segments, then head and tail partials, in one storage (measured faster) of the more aligned type
        using _StorageUnit = std::conditional_t<(alignof(_Acc) > alignof(_Size)), _Acc, _Size>;
        const std::uint32_t __segment_slots =
            oneapi::dpl::__internal::__dpl_ceiling_div(__num_units * sizeof(_Size), sizeof(_StorageUnit));
        const std::size_t __partial_slots =
            oneapi::dpl::__internal::__dpl_ceiling_div(2 * __num_units * sizeof(_Acc), sizeof(_StorageUnit));
        __device_storage<_StorageUnit> __units_storage(__q, __segment_slots + __partial_slots);

        // First kernel, with or without in-group combining: separate so the latter sets up no local memory
        auto __pieces = [&](auto __in_group_tag) {
            constexpr bool __in_group_v = decltype(__in_group_tag)::value;

            return __q.submit([&](sycl::handler& __cgh) {
                oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
                auto __units_acc = __get_accessor(sycl::write_only, __units_storage, __cgh, __dpl_sycl::__no_init{});

                const __segmented_reduce_flat_local<__in_group_v, _Acc, _Size> __local(__cgh, __sub_groups);

                __cgh.parallel_for<std::conditional_t<__in_group_v, _GroupName, _Name>...>(
                    sycl::nd_range<1>(sycl::range<1>(__num_groups * __work_group_size),
                                      sycl::range<1>(__work_group_size)),
                    [=, *this](sycl::nd_item<1> __item) [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(__sub_group_size)]] {
                        _Size* const __tail_segment_of = reinterpret_cast<_Size*>(__units_acc.__data());
                        _Acc* const __heads = reinterpret_cast<_Acc*>(__units_acc.__data() + __segment_slots);
                        _Acc* const __tails = __heads + __num_units;

                        const auto __sg = __item.get_sub_group();
                        const std::uint32_t __lane = __sg.get_local_linear_id();
                        const std::uint32_t __sg_id = __sg.get_group_linear_id();
                        const std::uint32_t __group_id = __item.get_group(0);
                        const std::uint32_t __piece = __group_id * std::uint32_t(__sub_groups) + __sg_id;
                        // In 64 bits: the product may exceed 32 bits before clamping to __n
                        const _Size __piece_begin = std::min(__n, _Size(std::size_t(__piece) * __piece_size));
                        const _Size __piece_end = std::min(__n, __piece_begin + _Size(__piece_size));

                        __piece_parts<_Size> __parts(__m);

                        __write_empty_segments(__sg, __offsets, __result, __n, __m, __piece, __num_pieces);
                        if (__piece_begin < __piece_end)
                            __reduce_piece(__sg, __input, __offsets, __result, __n, __m, __piece_begin, __piece_end,
                                           __parts);

                        if constexpr (__in_group_v)
                            __combine_pieces_in_group(__item, __sg_id, __lane, __group_id, __sub_groups, __m, __parts,
                                                      __local, __result, __heads, __tails, __tail_segment_of);
                        else
                            __store_piece_parts(__lane, __piece, __m, __parts, __heads, __tails, __tail_segment_of);
                    });
            });
        };

        sycl::event __event = __combine_in_group ? __pieces(/*__in_group_tag = */ std::true_type{})
                                                 : __pieces(/*__in_group_tag = */ std::false_type{});

        // Each unit's tail with the following heads its segment covers, a sub-group per unit; none for a single unit
        if (__num_units > 1)
        {
            const std::size_t __tail_work_group_size = _Tuning::__max_work_group(__q);
            const std::uint32_t __tail_sub_groups = __tail_work_group_size / __sub_group_size;
            const std::size_t __tail_groups =
                oneapi::dpl::__internal::__dpl_ceiling_div(__num_units, __tail_sub_groups);

            __event = __q.submit([&](sycl::handler& __cgh) {
                oneapi::dpl::__ranges::__require_access(__cgh, __offsets, __result);
                auto __units_acc = __get_accessor(sycl::read_only, __units_storage, __cgh);

                __cgh.depends_on(__event);

                __cgh.parallel_for<_TailName...>(
                    sycl::nd_range<1>(sycl::range<1>(__tail_groups * __tail_work_group_size),
                                      sycl::range<1>(__tail_work_group_size)),
                    [=, *this](sycl::nd_item<1> __item) [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(__sub_group_size)]] {
                        const auto __sg = __item.get_sub_group();
                        const std::uint32_t __unit =
                            __item.get_group(0) * __tail_sub_groups + __sg.get_group_linear_id();

                        if (__unit < __num_units)
                            __combine_tail_with_heads(
                                __sg, __offsets, __result, __n, __m, __unit, __num_units, __unit_size,
                                reinterpret_cast<const _Size*>(__units_acc.__data()),
                                reinterpret_cast<const _Acc*>(__units_acc.__data() + __segment_slots));
                    });
            });
        }

        // Wait: the temporary data is released on return
        __event.wait_and_throw();
    }

  private:
    // A piece's output on lane 0: head partial with its state, tail partial with its segment (__m for none)
    template <typename _Size>
    struct __piece_parts
    {
        _Size __tail_segment = {};
        __segmented_reduce_head_state __head_state = __segmented_reduce_head_state::__none;
        oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __head, __tail;

        explicit __piece_parts(const _Size __m) : __tail_segment(__m) {}
    };

    // A very short segment held whole by the piece
    template <typename _Size>
    static bool
    __is_tiny(const _Size __begin, const _Size __end, const _Size __piece_begin, const _Size __piece_end)
    {
        // A reversed segment in the piece has a negative length, so it is tiny
        return __begin >= __piece_begin && __end <= __piece_end &&
               __end - __begin <= _Size(_Tuning::__tiny_segment_max);
    }

    // Element at an offset from the piece start, via a pointer where the input has one
    template <typename _InRange, typename _Size>
    static decltype(auto)
    __element(const _InRange __input, const _Size __piece_begin, const std::uint32_t __index_in_piece)
    {
        if constexpr (__has_pointer_begin_v<_InRange>)
            return (__input.begin() + __piece_begin)[__index_in_piece];
        else
            return __input[__piece_begin + _Size(__index_in_piece)];
    }

    // Sub-group reduces [__first, __last) of the piece (non-empty) onto lane 0, lanes interleaved for coalescing;
    // a segment start adds the init
    template <typename _SubGroup, typename _Size, typename _Input>
    _Acc
    __reduce_by_sub_group(const _SubGroup __sg, const _Size __piece_begin, const std::uint32_t __first,
                          const std::uint32_t __last, const bool __is_first, const _Input __input) const
    {
        const std::uint32_t __lane = __sg.get_local_linear_id();
        const std::uint32_t __sub_group_size = __sg.get_local_linear_range();

        // At least two values a lane, so that an accumulator never comes from a lone element; idle lanes hold the init
        std::uint32_t __active = std::min(__sub_group_size, (__last - __first) / 2);
        oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __acc;

        // With a pointer input, lanes load aligned vectors; the unaligned head and tail go to lane 0
        bool __by_vectors = false;
        if constexpr (__has_pointer_begin_v<_Input>)
        {
            using _T = std::remove_cv_t<std::remove_pointer_t<decltype(__input.begin())>>;
            if constexpr (std::is_arithmetic_v<_T> && !std::is_same_v<_T, bool> &&
                          (sizeof(_T) == 1 || sizeof(_T) == 2 || sizeof(_T) == 4 || sizeof(_T) == 8))
            {
                const _T* const __piece_data = __input.begin() + __piece_begin;
                // Four elements a vector, two for 8-byte types, to stay within 16 bytes and be aligned as often
                constexpr std::uint32_t __vector_width = std::min<std::uint32_t>(4, 16 / sizeof(_T));
                using _V = sycl::vec<_T, __vector_width>;
                constexpr std::uint32_t __vector_bytes = sizeof(_V);
                const std::uint32_t __misalignment =
                    std::uint32_t(reinterpret_cast<std::uintptr_t>(__piece_data + __first) % __vector_bytes);
                // First vector-aligned index, at most __last
                const std::uint32_t __aligned_first = std::min<std::uint32_t>(
                    __last, __first + (__vector_bytes - __misalignment) % __vector_bytes / sizeof(_T));
                const std::uint32_t __num_vectors = (__last - __aligned_first) / __vector_width;
                if (__num_vectors >= _Tuning::__min_vectors)
                {
                    __by_vectors = true;
                    __active = std::min(__sub_group_size, __num_vectors);
                    auto __load_vector = [&](std::uint32_t __vector_index) -> _Acc {
                        _V __values;
                        __values.load(0, sycl::address_space_cast<sycl::access::address_space::generic_space,
                                                                  sycl::access::decorated::no>(
                                             __piece_data + __aligned_first + __vector_width * __vector_index));
                        if constexpr (__vector_width == 2)
                            return __reduce_pair<_Acc>(_T(__values[0]), _T(__values[1]), __binary_op);
                        else
                            return __reduce_four<_Acc>(_T(__values[0]), _T(__values[1]), _T(__values[2]),
                                                       _T(__values[3]), __binary_op);
                    };
                    if (__lane < __active)
                    {
                        __acc.__setup(__fold_strided(__load_vector(__lane), __lane + __active, __num_vectors, __active,
                                                     __load_vector, __binary_op));
                    }

                    // Unaligned head and tail, at most three elements each (one for 8 bytes), to lane 0
                    if (__lane == 0)
                    {
                        for (std::uint32_t __i = __first; __i < __aligned_first; ++__i)
                            __acc.__v = __reduce_pair<_Acc>(__acc.__v, __piece_data[__i], __binary_op);
                        for (std::uint32_t __i = __aligned_first + __vector_width * __num_vectors; __i < __last; ++__i)
                            __acc.__v = __reduce_pair<_Acc>(__acc.__v, __piece_data[__i], __binary_op);
                    }
                }
            }
        }

        if (!__by_vectors && __lane < __active)
        {
            const std::uint32_t __i = __first + __lane;

            __acc.__setup(__fold_strided(
                __reduce_pair<_Acc>(__element(__input, __piece_begin, __i),
                                    __element(__input, __piece_begin, __i + __active), __binary_op),
                __i + 2 * __active, __last, __active,
                [&](std::uint32_t __j) { return __element(__input, __piece_begin, __j); }, __binary_op));
        }
        if (__lane >= __active)
            __acc.__setup(__init);

        if (__active > 0)
            __acc.__v = __reduce_active_lanes<_Acc>(__sg, __acc.__v, __active, __binary_op);

        // A range of one element (no active lane) is that element; a segment start adds the init
        const _Acc __total = __active > 0 ? __acc.__v : _Acc(__element(__input, __piece_begin, __first));
        return __is_first ? __reduce_pair<_Acc>(__init, __total, __binary_op) : __total;
    }

    // Run of very short segments from __first_segment, a lane each, up to a longer one; empty and reversed ones are
    // left to __write_empty_segments. Returns the run's segment count and the end of its last segment
    template <typename _SubGroup, typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size>
    std::pair<std::uint32_t, _Size>
    __reduce_tiny_run(const _SubGroup __sg, const _InRange __input, const _OffsetRange __offsets,
                      const _OutRange __result, const _Size __n, const _Size __m, const _Size __first_segment,
                      const _Size __first_begin, const _Size __piece_begin, const _Size __piece_end) const
    {
        using oneapi::dpl::__internal::__clamp_to_size_and_cast;

        const std::uint32_t __lane = __sg.get_local_linear_id();
        const _Size __lane_segment = __first_segment + _Size(__lane);
        _Size __lane_begin = __first_begin, __lane_end = __first_begin;
        bool __in_run = false;

        if (__lane_segment < __m)
        {
            __lane_begin = __clamp_to_size_and_cast<_Size>(__offsets[__lane_segment], __n);
            __lane_end = __clamp_to_size_and_cast<_Size>(__offsets[__lane_segment + 1], __n);
            __in_run = __is_tiny(__lane_begin, __lane_end, __piece_begin, __piece_end);
        }
        // Run length: the first lane not in the run, or the sub-group size if all are
        const std::uint32_t __count = __dpl_sycl::__reduce_over_group(
            __sg, __in_run ? std::uint32_t(_Tuning::__sub_group_size) : __lane, sycl::minimum<std::uint32_t>());

        if (__lane < __count && __lane_begin < __lane_end)
        {
            // Within the piece, whose size fits in 32 bits
            const std::uint32_t __begin_in_piece = std::uint32_t(__lane_begin - __piece_begin);
            const std::uint32_t __end_in_piece = std::uint32_t(__lane_end - __piece_begin);

            // One lane per tiny segment, hence stride 1
            __result[__lane_segment] = __fold_strided(
                __reduce_pair<_Acc>(__init, __element(__input, __piece_begin, __begin_in_piece), __binary_op),
                __begin_in_piece + 1, __end_in_piece, 1,
                [&](std::uint32_t __j) { return __element(__input, __piece_begin, __j); }, __binary_op);
        }

        return {__count, __dpl_sycl::__select_from_group(__sg, __lane_end, __count - 1)};
    }

    // Skips the empty segments from __first_segment, all at __begin, which __write_empty_segments writes; returns the
    // last of them
    template <typename _SubGroup, typename _OffsetRange, typename _Size>
    static _Size
    __skip_empty_run(const _SubGroup __sg, const _OffsetRange __offsets, const _Size __n, const _Size __m,
                     const _Size __first_segment, const _Size __begin)
    {
        using oneapi::dpl::__internal::__clamp_to_size_and_cast;

        // The segment before the first offset after __begin is not empty, or the run reaches the last segment
        const _Size __after_run =
            __group_lower_bound<_Tuning::__sub_group_size>(__sg, __first_segment, __m + 1, [&](const _Size __i) {
                return __clamp_to_size_and_cast<_Size>(__offsets[__i], __n) <= __begin;
            });
        // Unsorted offsets (results unspecified) may end the search before __first_segment
        return std::max(__first_segment - 1, __after_run - 2);
    }

    // Empty and reversed segments get the init from the piece of their index, so that a long run of empty segments at
    // one offset spreads over all the sub-groups
    template <typename _SubGroup, typename _OffsetRange, typename _OutRange, typename _Size>
    void
    __write_empty_segments(const _SubGroup __sg, const _OffsetRange __offsets, const _OutRange __result,
                           const _Size __n, const _Size __m, const std::uint32_t __piece,
                           const std::uint32_t __num_pieces) const
    {
        using oneapi::dpl::__internal::__clamp_to_size_and_cast;

        const _Size __slice = oneapi::dpl::__internal::__dpl_ceiling_div(__m, _Size(__num_pieces));
        // At most __m + __num_pieces: no overflow
        const _Size __first = std::min(__m, _Size(__piece) * __slice);
        const _Size __last = std::min(__m, __first + __slice);
        for (_Size __segment = __first + _Size(__sg.get_local_linear_id()); __segment < __last;
             __segment += _Size(_Tuning::__sub_group_size))
            if (__clamp_to_size_and_cast<_Size>(__offsets[__segment], __n) >=
                __clamp_to_size_and_cast<_Size>(__offsets[__segment + 1], __n))
                __result[__segment] = __init;
    }

    // Part [__begin, __end) of __segment in the piece: whole segments are written, empty ones are left to
    // __write_empty_segments; a part begun before the piece is its head, one going on past its end its tail
    template <typename _SubGroup, typename _InRange, typename _OutRange, typename _Size>
    void
    __reduce_segment_part(const _SubGroup __sg, const _InRange __input, const _OutRange __result, const _Size __segment,
                          const _Size __begin, const _Size __end, const _Size __piece_begin, const _Size __piece_end,
                          __piece_parts<_Size>& __parts) const
    {
        const std::uint32_t __lane = __sg.get_local_linear_id();
        const std::uint32_t __part_first = __begin > __piece_begin ? std::uint32_t(__begin - __piece_begin) : 0;
        const std::uint32_t __part_last =
            std::uint32_t(std::max(__piece_begin, std::min(__end, __piece_end)) - __piece_begin);

        if (__part_first < __part_last)
        {
            const bool __is_first = __begin >= __piece_begin;
            const _Acc __acc =
                __reduce_by_sub_group(__sg, __piece_begin, __part_first, __part_last, __is_first, __input);

            const bool __ends_in_piece = __end <= __piece_end;

            if (__is_first)
            {
                if (__ends_in_piece && __lane == 0)
                    __result[__segment] = __acc;

                if (!__ends_in_piece)
                {
                    __parts.__tail.__setup(__acc);
                    __parts.__tail_segment = __segment;
                }
            }
            else
            {
                __parts.__head.__setup(__acc);
                __parts.__head_state = __ends_in_piece ? __segmented_reduce_head_state::__ends_in_piece
                                                       : __segmented_reduce_head_state::__goes_on;
            }
        }
    }

    // Walks the piece's segments, uniform across lanes, from the last one beginning before it (its head)
    template <typename _SubGroup, typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size>
    void
    __reduce_piece(const _SubGroup __sg, const _InRange __input, const _OffsetRange __offsets, const _OutRange __result,
                   const _Size __n, const _Size __m, const _Size __piece_begin, const _Size __piece_end,
                   __piece_parts<_Size>& __parts) const
    {
        using oneapi::dpl::__internal::__clamp_to_size_and_cast;

        // The last segment beginning before the piece, 0 for the first piece
        _Size __segment = std::max<_Size>(
            0, __group_lower_bound<_Tuning::__sub_group_size>(__sg, _Size{0}, __m, [&](const _Size __i) {
                   return __clamp_to_size_and_cast<_Size>(__offsets[__i], __n) < __piece_begin;
               }) - 1);

        // Each offset is loaded once, as a segment begins where the previous one ends; the walk ends at the piece end
        _Size __begin = __clamp_to_size_and_cast<_Size>(__offsets[__segment], __n);
        for (; __segment < __m && __begin < __piece_end; ++__segment)
        {
            const _Size __end = __clamp_to_size_and_cast<_Size>(__offsets[__segment + 1], __n);
            if (__is_tiny(__begin, __end, __piece_begin, __piece_end))
            {
                const auto [__count, __run_end] = __reduce_tiny_run(__sg, __input, __offsets, __result, __n, __m,
                                                                    __segment, __begin, __piece_begin, __piece_end);
                // Continue after the run; its last segment may end the piece. __count >= 1, as __segment is tiny
                __segment += _Size(__count) - 1;
                // A sub-group of empty segments at __begin: their run may go on far, so the rest of it is skipped
                if (__count == _Tuning::__sub_group_size && __run_end == __begin)
                    __segment = __skip_empty_run(__sg, __offsets, __n, __m, __segment + 1, __begin);
                __begin = __run_end;
                continue;
            }

            __reduce_segment_part(__sg, __input, __result, __segment, __begin, __end, __piece_begin, __piece_end,
                                  __parts);
            __begin = __end;
        }
    }

    // Piece parts for the second kernel, from lane 0
    template <typename _Size>
    static void
    __store_piece_parts(const std::uint32_t __lane, const std::uint32_t __piece, const _Size __m,
                        const __piece_parts<_Size>& __parts, _Acc* const __heads, _Acc* const __tails,
                        _Size* const __tail_segment_of)
    {
        if (__lane == 0 && __parts.__head_state != __segmented_reduce_head_state::__none)
            __heads[__piece] = __parts.__head.__v;
        if (__lane == 0 && __parts.__tail_segment < __m)
            __tails[__piece] = __parts.__tail.__v;
        if (__lane == 0)
            __tail_segment_of[__piece] = __parts.__tail_segment;
    }

    // Combine a work-group's pieces in local memory: lane j of sub-group 0 joins piece j's tail with following heads,
    // lane 0 the group head; a segment past the work-group leaves its tail and the group head to the second kernel
    template <typename _Size, typename _Local, typename _OutRange>
    void
    __combine_pieces_in_group(const sycl::nd_item<1> __item, const std::uint32_t __sg_id, const std::uint32_t __lane,
                              const std::uint32_t __group_id, const std::size_t __sub_groups, const _Size __m,
                              const __piece_parts<_Size>& __parts, const _Local __local, const _OutRange __result,
                              _Acc* const __heads, _Acc* const __tails, _Size* const __tail_segment_of) const
    {
        if (__lane == 0)
        {
            if (__parts.__head_state != __segmented_reduce_head_state::__none)
                __local.__loc_heads_tails_acc[__sg_id] = __parts.__head.__v;
            if (__parts.__tail_segment < __m)
                __local.__loc_heads_tails_acc[__sub_groups + __sg_id] = __parts.__tail.__v;
            __local.__loc_tail_segments_acc[__sg_id] = __parts.__tail_segment;
            __local.__loc_head_states_acc[__sg_id] = __parts.__head_state;
        }

        // No tail unless found below; the barrier orders this before the writes after it
        if (__sg_id == 0 && __lane == 0)
            __tail_segment_of[__group_id] = __m;

        sycl::group_barrier(__item.get_group());
        if (__sg_id != 0 || __lane >= __sub_groups)
            return;

        // Folds the heads from __piece on into __acc while the segment goes on; returns the state it stops at
        auto __chain = [&](_Acc& __acc, __segmented_reduce_head_state __state, std::size_t __piece) {
            for (; __state == __segmented_reduce_head_state::__goes_on && __piece < __sub_groups; ++__piece)
            {
                __state = __local.__loc_head_states_acc[__piece];
                if (__state != __segmented_reduce_head_state::__none)
                    __acc = __reduce_pair<_Acc>(__acc, __local.__loc_heads_tails_acc[__piece], __binary_op);
            }
            return __state;
        };

        const _Size __tail_segment = __local.__loc_tail_segments_acc[__lane];
        if (__tail_segment < __m)
        {
            _Acc __acc = __local.__loc_heads_tails_acc[__sub_groups + __lane];
            const __segmented_reduce_head_state __state =
                __chain(__acc, __segmented_reduce_head_state::__goes_on, __lane + 1);
            if (__state == __segmented_reduce_head_state::__ends_in_piece)
                __result[__tail_segment] = __acc;
            if (__state != __segmented_reduce_head_state::__ends_in_piece)
            {
                __tails[__group_id] = __acc;
                __tail_segment_of[__group_id] = __tail_segment;
            }
        }

        if (__lane == 0 && __local.__loc_head_states_acc[0] != __segmented_reduce_head_state::__none)
        {
            _Acc __acc = __local.__loc_heads_tails_acc[0];
            __chain(__acc, __local.__loc_head_states_acc[0], 1);
            __heads[__group_id] = __acc;
        }
    }

    // Second kernel for __unit, by its sub-group: the unit's tail with the following heads its segment covers
    template <typename _SubGroup, typename _OffsetRange, typename _OutRange, typename _Size>
    void
    __combine_tail_with_heads(const _SubGroup __sg, const _OffsetRange __offsets, const _OutRange __result,
                              const _Size __n, const _Size __m, const std::uint32_t __unit,
                              const std::size_t __num_units, const std::size_t __unit_size,
                              const _Size* const __tail_segment_of, const _Acc* const __heads) const
    {
        using oneapi::dpl::__internal::__clamp_to_size_and_cast;

        constexpr std::uint32_t __sub_group_size = _Tuning::__sub_group_size;
        const std::uint32_t __lane = __sg.get_local_linear_id();

        const _Size __tail_segment = __tail_segment_of[__unit];
        if (__tail_segment >= __m)
            return;

        // Tails follow the heads in the storage; the tail loads with its segment end, so neither waits for the other
        const _Acc __tail = (__heads + __num_units)[__unit];
        const _Size __segment_end = __clamp_to_size_and_cast<_Size>(__offsets[__tail_segment + 1], __n);
        // Units after this one up to the one holding the segment's last element
        const std::uint32_t __heads_count = std::uint32_t(std::size_t(__segment_end - 1) / __unit_size) - __unit;
        if (__heads_count == 0 && __lane == 0)
            __result[__tail_segment] = __tail;
        if (__heads_count == 0)
            return;

        // Following heads, one per lane, then by four; heads are accumulators, so a lane may start from one alone; idle
        // lanes hold the tail, unused
        const _Acc* const __first_head = __heads + (__unit + 1);
        oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __acc;

        if (__lane < __heads_count)
            __acc.__setup(__first_head[__lane]);
        if (__lane >= __heads_count)
            __acc.__setup(__tail);

        __acc.__v = __fold_strided(
            __acc.__v, __lane + __sub_group_size, __heads_count, __sub_group_size,
            [&](std::uint32_t __head_index) { return __first_head[__head_index]; }, __binary_op);
        __acc.__v =
            __reduce_active_lanes<_Acc>(__sg, __acc.__v, std::min(__heads_count, __sub_group_size), __binary_op);
        if (__lane == 0)
            __result[__tail_segment] = __reduce_pair<_Acc>(__tail, __acc.__v, __binary_op);
    }
};

// GPU, arithmetic types, small input: one kernel, a work-group per segment; empty, out-of-input or reversed
// segments get the init
struct __segmented_reduce_group_tuning
{
    // As on the flat path, whose sub-group size the dispatch checks for both
    static constexpr std::size_t __sub_group_size = __segmented_reduce_flat_tuning::__sub_group_size;

    // Up to this many elements and segments, a work-group per segment always beats the flat path; positions within a
    // segment then fit in 32 bits
    static constexpr std::size_t __max_input = std::size_t(1) << 12;
    static_assert(__max_input <= std::numeric_limits<std::uint32_t>::max());

    // With fewer segments than __min_segments_beyond, a work-group per segment up to this many elements
    static constexpr std::size_t __max_few_segments_input = std::size_t(1) << 14;

    // With at least this many segments, a work-group per segment up to this many elements and segments per hardware
    // thread
    static constexpr std::size_t __min_segments_beyond = 16;
    static constexpr std::size_t __max_inputs_per_thread = 4;

    // A few segments per GPU core on a mid-size input: a work-group per segment saves the flat path's search and
    // second kernel
    static constexpr std::size_t __max_segments_per_core = 40;
    static constexpr std::size_t __max_mid_inputs_per_thread = 512;
    static constexpr std::size_t __max_mid_average_length = 4096;

    // At least __min_segments_beyond segments of a few thousand elements on average: a work-group per segment beats
    // the flat path up to this input; the lower bound doubles above a few thousand segments
    static constexpr std::size_t __min_long_average_length = 1024;
    static constexpr std::size_t __max_long_average_length = 16384;
    static constexpr std::size_t __max_short_bound_segments = 4096;
    static constexpr std::size_t __max_long_average_input = std::size_t(1) << 25;

    static bool
    __applies(const sycl::queue& __q, const std::size_t __n, const std::size_t __m)
    {
        if (__n <= __max_input && __m <= __max_input)
            return true;
        if (__m < __min_segments_beyond)
            return __n <= __max_few_segments_input;

        const std::size_t __min_average =
            __m <= __max_short_bound_segments ? __min_long_average_length : 2 * __min_long_average_length;
        if (__n >= __min_average * __m && __n <= __max_long_average_length * __m && __n <= __max_long_average_input)
            return true;

        const oneapi::dpl::__internal::__device_compute_units __compute_units(__q);
        const std::size_t __max_input_beyond = std::min<std::size_t>(
            std::numeric_limits<std::uint32_t>::max(), __max_inputs_per_thread * __compute_units.__hw_threads);
        if (__n <= __max_input_beyond && __m <= __max_input_beyond)
            return true;

        const std::size_t __max_mid_input = std::min<std::size_t>(
            std::numeric_limits<std::uint32_t>::max(), __max_mid_inputs_per_thread * __compute_units.__hw_threads);
        return __n <= __max_mid_input && __n <= __max_mid_average_length * __m &&
               __m <= __max_segments_per_core * __compute_units.__num_xe_cores;
    }

    // A power of two about the average segment length within the bounds; the largest beyond small inputs
    static std::size_t
    __work_group_size(const sycl::queue& __q, const std::size_t __n, const std::size_t __m)
    {
        const std::size_t __max_size = __segmented_reduce_flat_tuning::__max_work_group(__q);
        if (__n > __max_input)
            return __max_size;

        const std::size_t __average = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __m);
        return std::min(__max_size, std::max(__sub_group_size, oneapi::dpl::__internal::__dpl_bit_ceil(__average)));
    }
};

template <typename... _Name>
class __segmented_reduce_group_kernel;

template <typename _Acc, typename _KernelName, typename _BinaryOperation, typename _Tuning>
struct __segmented_reduce_group_submitter;

template <typename _Acc, typename... _Name, typename _BinaryOperation, typename _Tuning>
struct __segmented_reduce_group_submitter<_Acc, __internal::__optional_kernel_name<_Name...>, _BinaryOperation, _Tuning>
{
    static_assert(std::is_arithmetic_v<_Acc>, "the work-group per segment path takes arithmetic accumulators only");

    _Acc __init; // init converted to _Acc
    mutable _BinaryOperation __binary_op;

    template <typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size>
    void
    operator()(sycl::queue& __q, _InRange&& __input, _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n,
               const _Size __m) const
    {
        using oneapi::dpl::__internal::__clamp_to_size_and_cast;

        constexpr std::uint32_t __sub_group_size = _Tuning::__sub_group_size;
        const std::uint32_t __work_group_size = _Tuning::__work_group_size(__q, __n, __m);
        const std::uint32_t __sub_groups = __work_group_size / __sub_group_size;

        sycl::event __event = __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
            __dpl_sycl::__local_accessor<_Acc> __loc_partials_acc(sycl::range<1>(__sub_groups), __cgh);

            __cgh.parallel_for<_Name...>(
                sycl::nd_range<1>(sycl::range<1>(std::size_t(__m) * __work_group_size),
                                  sycl::range<1>(__work_group_size)),
                [=, *this](sycl::nd_item<1> __item) [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(__sub_group_size)]] {
                    const std::uint32_t __segment = __item.get_group(0);
                    const std::uint32_t __local_id = __item.get_local_id(0);

                    const _Size __begin = __clamp_to_size_and_cast<_Size>(__offsets[__segment], __n);
                    const _Size __end = __clamp_to_size_and_cast<_Size>(__offsets[__segment + 1], __n);

                    // Uniform across the work-group, so it leaves before any barrier
                    if (__begin >= __end)
                    {
                        if (__local_id == 0)
                            __result[__segment] = __init;
                        return;
                    }

                    // Work-item 0 starts from op(init, first); those past the length start from the first,
                    // skipped below
                    const std::uint32_t __length = __end - __begin;
                    const std::uint32_t __first = __local_id < __length ? __local_id : 0;
                    _Acc __acc = __local_id == 0 ? __reduce_pair<_Acc>(__init, __input[__begin], __binary_op)
                                                 : _Acc(__input[__begin + __first]);
                    // Up to two work-groups long, at most one more element a work-item: a plain step is
                    // cheaper than the fold
                    if (__length > 2 * __work_group_size)
                        __acc = __fold_strided(
                            __acc, __local_id + __work_group_size, __length, __work_group_size,
                            [&](std::uint32_t __i) { return __input[__begin + __i]; }, __binary_op);
                    else if (__local_id + __work_group_size < __length)
                        __acc =
                            __reduce_pair<_Acc>(__acc, __input[__begin + __local_id + __work_group_size], __binary_op);

                    const auto __sg = __item.get_sub_group();
                    const std::uint32_t __sg_id = __sg.get_group_linear_id();
                    const std::uint32_t __items = std::min(__length, __work_group_size);
                    const std::uint32_t __sg_first = __sg_id * __sub_group_size;
                    const std::uint32_t __active =
                        __items > __sg_first ? std::min(__sub_group_size, __items - __sg_first) : 0;
                    __acc = __reduce_active_lanes<_Acc>(__sg, __acc, __active, __binary_op);

                    // Uniform across the work-group, as the barrier needs
                    const std::uint32_t __active_sub_groups =
                        oneapi::dpl::__internal::__dpl_ceiling_div(__items, __sub_group_size);
                    if (__active_sub_groups > 1)
                    {
                        if (__active > 0 && __sg.get_local_linear_id() == 0)
                            __loc_partials_acc[__sg_id] = __acc;

                        sycl::group_barrier(__item.get_group());

                        if (__sg_id == 0)
                        {
                            const std::uint32_t __lane = __sg.get_local_linear_id();
                            // Inactive lanes read slot 0 to stay in bounds; their values are not used
                            __acc = __reduce_active_lanes<_Acc>(
                                __sg, __loc_partials_acc[__lane < __active_sub_groups ? __lane : 0],
                                __active_sub_groups, __binary_op);
                        }
                    }

                    if (__local_id == 0)
                        __result[__segment] = __acc;
                });
        });

        // Wait: the algorithm has no asynchronous interface
        __event.wait_and_throw();
    }
};

template <typename _Name>
class __segmented_reduce_scan_wrapper;
template <typename _Name>
class __segmented_reduce_chunk_wrapper;
template <typename _Name>
class __segmented_reduce_combine_wrapper;
template <typename _Name>
class __segmented_reduce_init_wrapper;

// Writes the init into every result: the input is empty, or the offsets give no element to any segment
template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Size, typename _Acc>
void
__parallel_reduce_by_segment_offsets_fill(_BackendTag, _ExecutionPolicy&& __exec, _Range&& __result, _Size __m,
                                          const _Acc& __init)
{
    using _Fill = oneapi::dpl::__internal::__brick_fill<oneapi::dpl::__internal::__hetero_tag<_BackendTag>, _Acc>;
    // Separate kernel name, as the same call also instantiates the reduction kernel
    __parallel_for(
        _BackendTag{}, make_wrapped_policy<__segmented_reduce_fill_wrapper>(std::forward<_ExecutionPolicy>(__exec)),
        unseq_backend::walk_n_vectors_or_scalars<_Fill>(_Fill{__init}, __m), __m, std::forward<_Range>(__result))
        .event()
        .wait_and_throw();
}

// Reduces the segments of [0, __n) given by __m + 1 offsets into __m results
template <typename _Acc, typename _ChunkTuning = __segmented_reduce_chunk_tuning,
          typename _FlatTuning = __segmented_reduce_flat_tuning,
          typename _GroupTuning = __segmented_reduce_group_tuning>
struct __parallel_reduce_by_segment_offsets
{
    template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange,
              typename _OutRange, typename _Size, typename _BinaryOperation>
    void
    operator()(_BackendTag, _ExecutionPolicy&& __exec, _InRange&& __input, _OffsetRange&& __offsets,
               _OutRange&& __result, const _Size __n, const _Size __m, const _Acc __init,
               _BinaryOperation __binary_op) const
    {
        // Segment lengths and index steps may go negative before they are clamped
        static_assert(std::is_signed_v<_Size>, "The size type must be signed");

        // No segments and an empty input are handled by the caller; the divisions below need both sizes positive
        assert(__m > 0 && __n > 0);

        sycl::queue __q = __exec.queue();
        const sycl::device __device = __q.get_device();
        const bool __is_gpu = __device.is_gpu();

        if (__try_reduce_on_gpu(_BackendTag{}, __exec, __q, __is_gpu, __input, __offsets, __result, __n, __m, __init,
                                __binary_op))
            return;

        if (__try_reduce_without_device_scope_atomics(_BackendTag{}, __exec, __q, __device, __is_gpu, __input,
                                                      __offsets, __result, __n, __m, __init, __binary_op))
            return;

        __reduce_short_and_long_segments(_BackendTag{}, __exec, __q, __is_gpu, __input, __offsets, __result, __n, __m,
                                         __init, __binary_op);
    }

  private:
    // Arithmetic input and accumulator on a supported GPU use the group/flat path; use 32-bit indices when safe.
    template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange,
              typename _OutRange, typename _Size, typename _BinaryOperation>
    static bool
    __try_reduce_on_gpu(_BackendTag, _ExecutionPolicy&&, sycl::queue& __q, const bool __is_gpu, _InRange& __input,
                        _OffsetRange& __offsets, _OutRange& __result, const _Size __n, const _Size __m,
                        const _Acc& __init, _BinaryOperation __binary_op)
    {
        if constexpr (std::is_arithmetic_v<_Acc> &&
                      std::is_arithmetic_v<oneapi::dpl::__internal::__value_t<std::decay_t<_InRange>>>)
        {
            if (__is_gpu && oneapi::dpl::__internal::__supports_sub_group_size(__q, _FlatTuning::__sub_group_size))
            {
                using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

                // 32-bit index arithmetic when the sizes allow it, with room for sums past __n and __m:
                // GPUs emulate 64-bit ones
                using _Index32 = std::int32_t;
                if constexpr (sizeof(_Size) > sizeof(_Index32))
                {
                    constexpr _Size __index32_max = std::numeric_limits<_Index32>::max() / 2;
                    if (__n <= __index32_max && __m <= __index32_max)
                    {
                        __reduce_on_gpu<_CustomName>(__q, __input, __offsets, __result, _Index32(__n), _Index32(__m),
                                                     __init, __binary_op);
                        return true;
                    }
                }

                __reduce_on_gpu<_CustomName>(__q, __input, __offsets, __result, __n, __m, __init, __binary_op);
                return true;
            }
        }

        return false;
    }

    // Without device-scope atomics, initialize all outputs and reduce the input in chunks.
    template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange,
              typename _OutRange, typename _Size, typename _BinaryOperation>
    static bool
    __try_reduce_without_device_scope_atomics(_BackendTag, _ExecutionPolicy&& __exec, sycl::queue& __q,
                                              const sycl::device& __device, const bool __is_gpu, _InRange& __input,
                                              _OffsetRange& __offsets, _OutRange& __result, const _Size __n,
                                              const _Size __m, const _Acc& __init, _BinaryOperation __binary_op)
    {
        if (__has_device_scope_atomics(__device))
            return false;

        // Then the longest length is not reported: init for the empty segments, all the others in chunks
        __parallel_reduce_by_segment_offsets_fill(_BackendTag{},
                                                  make_wrapped_policy<__segmented_reduce_init_wrapper>(__exec),
                                                  __result, std::size_t(__m), __init);
        __reduce_in_chunks(_BackendTag{}, __exec, __q, __is_gpu, __input, __offsets, __result, __n, __m, __init,
                           __binary_op, _Size{0}, __n, __m);

        return true;
    }

    // Reduce short segments per work-item; split longer segments into chunks and combine their partial results.
    template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange,
              typename _OutRange, typename _Size, typename _BinaryOperation>
    static void
    __reduce_short_and_long_segments(_BackendTag, _ExecutionPolicy&& __exec, sycl::queue& __q, const bool __is_gpu,
                                     _InRange& __input, _OffsetRange& __offsets, _OutRange& __result, const _Size __n,
                                     const _Size __m, const _Acc& __init, _BinaryOperation __binary_op)
    {
        // Short segments by a work-item each, which also writes the empty ones; the longer ones in chunks
        const _Size __short_segment_max = _ChunkTuning::__short_segment_max(__is_gpu);
        const _Size __max_length = __reduce_short_segments(_BackendTag{}, __exec, __q, __input, __offsets, __result,
                                                           __n, __m, __short_segment_max, __init, __binary_op);
        // At most __n / __short_segment_max segments are longer than __short_segment_max
        if (__max_length > 0)
            __reduce_in_chunks(_BackendTag{}, __exec, __q, __is_gpu, __input, __offsets, __result, __n, __m, __init,
                               __binary_op, __short_segment_max, __max_length, __n / __short_segment_max);
    }

    // GPU, arithmetic types: a work-group per segment for a small input, else pieces of the input independent of the
    // segments. The kernel names include the index type, which the caller chooses by the sizes
    template <typename _CustomName, typename _InRange, typename _OffsetRange, typename _OutRange, typename _Index,
              typename _BinaryOperation>
    static void
    __reduce_on_gpu(sycl::queue& __q, _InRange& __input, _OffsetRange& __offsets, _OutRange& __result, const _Index __n,
                    const _Index __m, const _Acc& __init, _BinaryOperation __binary_op)
    {
        if (_GroupTuning::__applies(__q, __n, __m))
        {
            using _GroupKernel =
                __internal::__kernel_name_provider<__segmented_reduce_group_kernel<_CustomName, _Index>>;
            __segmented_reduce_group_submitter<_Acc, _GroupKernel, _BinaryOperation, _GroupTuning>{__init, __binary_op}(
                __q, __input, __offsets, __result, __n, __m);
        }
        else
        {
            using _FlatKernel = __internal::__kernel_name_provider<__segmented_reduce_flat_kernel<_CustomName, _Index>>;
            using _FlatGroupKernel =
                __internal::__kernel_name_provider<__segmented_reduce_flat_group_kernel<_CustomName, _Index>>;
            using _FlatTailKernel =
                __internal::__kernel_name_provider<__segmented_reduce_flat_tail_kernel<_CustomName, _Index>>;
            __segmented_reduce_flat_submitter<_Acc, _FlatKernel, _FlatGroupKernel, _FlatTailKernel, _BinaryOperation,
                                              _FlatTuning>{__init, __binary_op}(__q, __input, __offsets, __result, __n,
                                                                                __m);
        }
    }

    // Whether a device supports the atomics of the device scope, which SYCL does not require
    static bool
    __has_device_scope_atomics(const sycl::device& __device)
    {
        const std::vector<sycl::memory_scope> __scopes =
            __device.get_info<sycl::info::device::atomic_memory_scope_capabilities>();
        return std::find(__scopes.begin(), __scopes.end(), sycl::memory_scope::device) != __scopes.end();
    }

    // Short segments, a work-item each; returns the longest long-segment length, 0 if none
    template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange,
              typename _OutRange, typename _Size, typename _BinaryOperation>
    static _Size
    __reduce_short_segments(_BackendTag, _ExecutionPolicy&& __exec, sycl::queue& __q, _InRange& __input,
                            _OffsetRange& __offsets, _OutRange& __result, const _Size __n, const _Size __m,
                            const _Size __short_segment_max, const _Acc& __init, _BinaryOperation __binary_op)
    {
        // 32 bits: the atomics of every device; the cap of the length by the brick does not change the chunk size
        __atomic_result_storage<std::uint32_t> __max_length_storage(__q);
        sycl::event __short_done = __parallel_for(_BackendTag{}, __exec, __max_length_storage.__zero_done,
                                                  __segmented_reduce_short_brick<_Acc, _Size, _BinaryOperation>{
                                                      __n, __short_segment_max, __init, __binary_op},
                                                  __m, __offsets, __result, __input, __max_length_storage.__all_view())
                                       .event();

        return __max_length_storage.__load_result(__exec, __q, __short_done);
    }

    // Reduces segments longer than __min_length in chunks; __max_segments bounds their number
    template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange,
              typename _OutRange, typename _Size, typename _BinaryOperation>
    static void
    __reduce_in_chunks(_BackendTag, _ExecutionPolicy&& __exec, sycl::queue& __q, const bool __is_gpu, _InRange& __input,
                       _OffsetRange& __offsets, _OutRange& __result, const _Size __n, const _Size __m,
                       const _Acc& __init, _BinaryOperation __binary_op, const _Size __min_length,
                       const _Size __max_length, const _Size __max_segments)
    {
        const _Size __chunk_size = _ChunkTuning::__chunk_size(__is_gpu, __max_length);

        // Chunk ends per segment: inclusive scan of chunk counts; its future keeps the temporaries
        // alive until combining
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
                       __segmented_reduce_combine_brick<_Acc, _Size, _BinaryOperation>{__num_partials, __binary_op},
                       __m, __chunk_ends.__all_view(), __partials.__all_view(), __result)
            .event()
            .wait_and_throw();
    }
};

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
