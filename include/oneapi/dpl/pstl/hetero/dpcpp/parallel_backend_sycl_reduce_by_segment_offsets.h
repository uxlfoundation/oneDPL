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
#include <mutex>
#include <optional>
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

// Whether a range reaches its elements through a pointer returned by its begin(), as the views of USM memory and of
// buffers do, so that consecutive elements can be loaded as one vector
template <typename _Range, typename = void>
struct __segmented_reduce_pointer_input : std::false_type
{
};
template <typename _Range>
struct __segmented_reduce_pointer_input<
    _Range, std::enable_if_t<std::is_pointer_v<decltype(std::declval<const _Range&>().begin())>>> : std::true_type
{
};

// Reduces the elements [__first, __last), at least one, by a sub-group: the lanes take the elements in turn, so that the
// loads of the sub-group are coalesced, and a tree combines the results of the lanes. Both reorder the operands, which
// the required commutativity of the operation allows. A range that begins its segment starts from the operation on the
// initial value and its first element. The result is on lane 0. Only for arithmetic types, which a sub-group shifts
// between its lanes.
template <typename _Acc, typename _SubGroup, typename _Size, typename _InitType, typename _BinaryOperation,
          typename _Input>
_Acc
__segmented_reduce_by_sub_group(const _SubGroup& __sg, _Size __first, const _Size __last, const bool __is_first,
                                const _InitType& __init, _BinaryOperation& __binary_op, const _Input& __input)
{
    const _Size __lane = __sg.get_local_linear_id();
    const _Size __lanes = __sg.get_local_linear_range();
    _Acc __head{};
    if (__is_first)
        __head = std::invoke(__binary_op, __init, __input[__first++]);

    // Every lane that takes part gets at least two elements: its accumulator starts from the operation on two of them,
    // never from an element alone, which may mean something else than the element to the operation
    _Size __active = std::min<_Size>(__lanes, (__last - __first) / 2);
    _Acc __acc{};
    // Where the input is reached through a pointer, every lane loads four consecutive elements at once, from the first
    // address aligned for such a vector on; the elements before it and after the last whole vector go to lane 0. The
    // pairs of a vector keep the accumulator from being made from an element alone.
    bool __by_vectors = false;
    if constexpr (__segmented_reduce_pointer_input<_Input>::value)
    {
        using _T = std::remove_cv_t<std::remove_pointer_t<decltype(__input.begin())>>;
        if constexpr (std::is_arithmetic_v<_T> && !std::is_same_v<_T, bool> &&
                      (sizeof(_T) == 1 || sizeof(_T) == 2 || sizeof(_T) == 4 || sizeof(_T) == 8))
        {
            const _T* const __base = __input.begin();
            using _V = sycl::vec<_T, 4>;
            constexpr std::uintptr_t __vb = sizeof(_V);
            const std::uintptr_t __mis = reinterpret_cast<std::uintptr_t>(__base + __first) % __vb;
            const _Size __a = std::min<_Size>(__last, __first + _Size(__mis == 0 ? 0 : (__vb - __mis) / sizeof(_T)));
            const _Size __nv = (__last - __a) / 4;
            if (__nv >= 1)
            {
                __by_vectors = true;
                __active = std::min<_Size>(__lanes, __nv);
                auto __load4 = [&](_Size __k) -> _Acc {
                    const _V __v = *reinterpret_cast<const _V*>(__base + __a + 4 * __k);
                    return std::invoke(__binary_op,
                                       oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                           std::invoke(__binary_op, _T(__v[0]), _T(__v[1]))),
                                       oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                           std::invoke(__binary_op, _T(__v[2]), _T(__v[3]))));
                };
                if (__lane < __active)
                {
                    _Size __k = __lane;
                    __acc = __load4(__k);
                    // Four vector loads in flight
                    for (__k += __active; __k + 3 * __active < __nv; __k += 4 * __active)
                        __acc = std::invoke(
                            __binary_op, __acc,
                            std::invoke(__binary_op,
                                        oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                            std::invoke(__binary_op, __load4(__k), __load4(__k + __active))),
                                        oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(std::invoke(
                                            __binary_op, __load4(__k + 2 * __active), __load4(__k + 3 * __active)))));
                    for (; __k < __nv; __k += __active)
                        __acc = std::invoke(__binary_op, __acc, __load4(__k));
                }
                // The unaligned head and the tail, at most three elements each, to lane 0
                if (__lane == 0)
                {
                    for (_Size __i = __first; __i < __a; ++__i)
                        __acc = std::invoke(__binary_op, __acc, __input[__i]);
                    for (_Size __i = __a + 4 * __nv; __i < __last; ++__i)
                        __acc = std::invoke(__binary_op, __acc, __input[__i]);
                }
            }
        }
    }
    if (!__by_vectors && __lane < __active)
    {
        _Size __i = __first + __lane;
        __acc = std::invoke(__binary_op, __input[__i], __input[__i + __active]);
        // Four loads in flight
        for (__i += 2 * __active; __i + 3 * __active < __last; __i += 4 * __active)
            __acc =
                std::invoke(__binary_op, __acc,
                            std::invoke(__binary_op,
                                        oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                            std::invoke(__binary_op, __input[__i], __input[__i + __active])),
                                        oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(std::invoke(
                                            __binary_op, __input[__i + 2 * __active], __input[__i + 3 * __active]))));
        for (; __i < __last; __i += __active)
            __acc = std::invoke(__binary_op, __acc, __input[__i]);
    }
    for (_Size __shift = 1; __shift < __active; __shift *= 2)
    {
        const _Acc __other = sycl::shift_group_left(__sg, __acc, __shift);
        if (__lane % (2 * __shift) == 0 && __lane + __shift < __active)
            __acc = std::invoke(__binary_op, __acc, __other);
    }
    // A range of one element more than the initial value takes it alone
    if (__is_first)
        __acc = __active > 0
                    ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(std::invoke(__binary_op, __head, __acc))
                : __first < __last ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                         std::invoke(__binary_op, __head, __input[__first]))
                                   : __head;
    return __acc;
}

// On GPUs a decomposition of the input that does not depend on the segments, for arithmetic types, in portions taken
// one launch after another as the blocks of reduce-then-scan are: every launch has a number of work-groups chosen by
// the cache of the device, and every work-item of it a number of elements chosen so that the portions are equal. Every
// sub-group takes a piece of consecutive elements of its work-group, finds the segment covering its start by a binary
// search over the offsets and reduces the segments of the piece one after another, each by the whole sub-group with
// coalesced loads. A segment the piece holds whole is written. The partial result of the segment begun before the piece
// that covers its start is the head of the piece, and the one of the segment begun in the piece that goes on past its
// end is its tail; the tails are combined with the following heads in the local memory of the work-group, and the
// same for the work-groups: the last work-group of a segment to arrive combines the partial results of all of them. The
// segment going on past the end of a portion is carried into the next one.
inline constexpr std::size_t __segmented_reduce_gpu_flat_lanes = 32;
// Whether the portions are bounded by the cache as the blocks of reduce-then-scan are, or the input is one portion: the
// input is read once, so the cache gains nothing, and the launches of the portions cost time, as measured
inline constexpr bool __segmented_reduce_gpu_portions_by_cache = false;
// A work-item takes at least this many elements up to this input size, and four times as many above it, so that the
// search of a piece and the combination of its parts are paid for; in one portion it takes at most four times as many,
// so that the larger inputs are spread over more work-groups than the device runs at once, as measured
inline constexpr std::size_t __segmented_reduce_gpu_flat_min_inputs_per_item = 16;
inline constexpr std::size_t __segmented_reduce_gpu_flat_small_input = std::size_t(1) << 20;

// The index of the last of the __m segments that begins at or before __v, or -1
template <typename _Size, typename _OffsetRange>
_Size
__segmented_reduce_flat_last_begin(const _OffsetRange& __offsets, const _Size __m, const _Size __n, const _Size __v)
{
    _Size __lo = 0, __hi = __m;
    while (__lo < __hi)
    {
        const _Size __mid = __lo + (__hi - __lo) / 2;
        if (oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__mid], __n) <= __v)
            __lo = __mid + 1;
        else
            __hi = __mid;
    }
    return __lo - 1;
}

// What the decomposition above asks of the device. The queries cost more than a launch of a small input, so the answers
// for the last device asked about are kept; the holder is never destroyed, as a device must not outlive the runtime.
struct __segmented_reduce_device_info
{
    bool __supported; // whether the device runs sub-groups of the size of the decomposition
    std::size_t __max_work_group_size;
    std::size_t __compute_units;
    std::size_t __cache; // the size of the last-level cache in bytes, 0 if not reported

    static __segmented_reduce_device_info
    __get(const sycl::queue& __q)
    {
        struct __holder
        {
            std::mutex __mutex;
            std::optional<sycl::device> __device; // not constructed by default, which selects a device
            __segmented_reduce_device_info __info;
        };
        static __holder* const __last = new __holder;

        const sycl::device __device = __q.get_device();
        std::lock_guard<std::mutex> __lock(__last->__mutex);
        if (!__last->__device || *__last->__device != __device)
        {
            __last->__info = {
                oneapi::dpl::__internal::__supports_sub_group_size(__q, __segmented_reduce_gpu_flat_lanes),
                oneapi::dpl::__internal::__max_work_group_size(__q, 1024),
                oneapi::dpl::__internal::__max_compute_units(__q),
                __device.template get_info<sycl::info::device::global_mem_cache_size>()};
            __last->__device = __device;
        }
        return __last->__info;
    }
};

// The launches of the decomposition above for an input of __n elements of __bytes bytes each, as reduce-then-scan
// computes them for its blocks
struct __segmented_reduce_portions
{
    std::size_t __work_group_size; // a multiple of the size of a sub-group
    std::size_t __num_groups;      // the work-groups of a launch, but the last one
    std::size_t __num_portions;
    std::size_t __portion_size;    // the elements of a portion, but the last one
    std::size_t __inputs_per_item; // the elements of a work-item in a portion, but the last one
    std::size_t __min_inputs_per_item;

    __segmented_reduce_portions(const __segmented_reduce_device_info& __info, const std::size_t __n,
                                const std::size_t __bytes)
    {
        constexpr std::size_t __lanes = __segmented_reduce_gpu_flat_lanes;

        __work_group_size = __info.__max_work_group_size / __lanes * __lanes;
        const std::size_t __compute_units = __info.__compute_units;
        std::size_t __cache = __info.__cache;
        // A device that does not report the size of its cache is likely an older one: 32 KiB per compute unit
        if (__cache == 0)
            __cache = std::size_t{32} * 1024 * __compute_units;
        const std::size_t __target = __cache / 2;
        // Eight compute units per Xe core
        const std::size_t __cores = std::max<std::size_t>(1, __compute_units / 8);
        __min_inputs_per_item = __n <= __segmented_reduce_gpu_flat_small_input
                                    ? __segmented_reduce_gpu_flat_min_inputs_per_item
                                    : 4 * __segmented_reduce_gpu_flat_min_inputs_per_item;
        if (!__segmented_reduce_gpu_portions_by_cache)
        {
            // One portion: the work-groups are made smaller until there are at least two per Xe core, as long as the
            // input allows it
            __inputs_per_item = std::clamp<std::size_t>(
                (__n + 2 * __cores * __work_group_size - 1) / (2 * __cores * __work_group_size), __min_inputs_per_item,
                4 * __segmented_reduce_gpu_flat_min_inputs_per_item);
            auto __groups = [&](std::size_t __size) {
                return (__n + __inputs_per_item * __size - 1) / (__inputs_per_item * __size);
            };
            while (__work_group_size > __lanes && __groups(__work_group_size) < 2 * __cores)
                __work_group_size = std::max(__lanes, __work_group_size / 2 / __lanes * __lanes);
            __num_groups = __groups(__work_group_size);
            __num_portions = 1;
            __portion_size = __n;
            return;
        }

        const std::size_t __demand = __work_group_size * __bytes;
        std::size_t __budget = __n;
        if (__cores * __demand <= __target)
        {
            __num_groups = 2 * __cores;
            __budget = std::max(__target / __bytes, __num_groups * __work_group_size);
        }
        else if (__cores * __demand <= __cache)
        {
            __num_groups = __cores;
            __budget = __num_groups * __work_group_size;
        }
        else
            // The cache is spilled anyway: one portion
            __num_groups = 2 * __cores;

        const std::size_t __items = __num_groups * __work_group_size;
        if (__items < __budget && __budget < __n)
            __budget = __budget / __items * __items;
        __num_portions = (__n + __budget - 1) / __budget;
        __inputs_per_item = std::max(__min_inputs_per_item,
                                     (__n + __num_portions * __items - 1) / (__num_portions * __items));
        __portion_size = __items * __inputs_per_item;
        __num_portions = (__n + __portion_size - 1) / __portion_size;
    }
};

template <typename... _Name>
class __segmented_reduce_flat_kernel;

template <typename _Acc, typename _KernelName>
struct __segmented_reduce_flat_submitter;

template <typename _Acc, typename... _Name>
struct __segmented_reduce_flat_submitter<_Acc, __internal::__optional_kernel_name<_Name...>>
{
    template <typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size, typename _InitType,
              typename _BinaryOperation>
    void
    operator()(sycl::queue& __q, _InRange&& __input, _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n,
               const _Size __m, _InitType __init, _BinaryOperation __binary_op,
               const __segmented_reduce_device_info& __info) const
    {
        using _AtomicCount = sycl::atomic_ref<std::uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                              sycl::access::address_space::global_space>;

        constexpr std::size_t __lanes = __segmented_reduce_gpu_flat_lanes;
        const __segmented_reduce_portions __portions(
            __info, std::size_t(__n), sizeof(oneapi::dpl::__internal::__value_t<std::decay_t<_InRange>>));
        const std::size_t __work_group_size = __portions.__work_group_size;
        const std::size_t __sub_groups = __work_group_size / __lanes;
        const std::size_t __num_groups = __portions.__num_groups;

        // The number of the work-groups arrived for the segment going on past the end of every work-group, then zeroed
        // again by the last one; the partial results of the heads of the work-groups, of their tails, and the carries
        // into the portions, one for the portion in flight and one for the next, so that a launch does not read what
        // it writes
        __device_storage<std::uint32_t> __counts{__q, __num_groups};
        __device_storage<_Acc> __partials{__q, 2 * __num_groups + 2};

        sycl::event __event;
        if (__counts.__usm_buf)
            __event = __q.fill(__counts.__usm_buf.get(), std::uint32_t{0}, __num_groups);
        else
            __event = __q.submit([&](sycl::handler& __cgh) {
                sycl::accessor __counts_acc(__counts.__sycl_buf, __cgh, sycl::write_only, sycl::no_init);
                __cgh.fill(__counts_acc, std::uint32_t{0});
            });

        for (std::size_t __portion = 0; __portion < __portions.__num_portions; ++__portion)
        {
            const std::size_t __portion_begin = __portion * __portions.__portion_size;
            const std::size_t __rest = std::size_t(__n) - __portion_begin;
            const std::size_t __portion_end_index = __portion_begin + std::min(__rest, __portions.__portion_size);
            const bool __is_last = __portion + 1 == __portions.__num_portions;
            // The last portion takes as few elements per work-item as its work-groups allow, and only the work-groups
            // with elements
            const std::size_t __ipi =
                __is_last ? std::max(__portions.__min_inputs_per_item, (__rest + __num_groups * __work_group_size - 1) /
                                                                           (__num_groups * __work_group_size))
                          : __portions.__inputs_per_item;
            const std::size_t __group_size = __ipi * __work_group_size;
            const std::size_t __launch_groups = __is_last ? (__rest + __group_size - 1) / __group_size : __num_groups;

            __event = __q.submit([&](sycl::handler& __cgh) {
                oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
                auto __counts_acc = __get_accessor(sycl::read_write, __counts, __cgh);
                auto __partials_acc = __get_accessor(sycl::read_write, __partials, __cgh);
                __cgh.depends_on(__event);
                // The heads and the tails of the pieces of the work-group, and the states of the heads: 0 for none, 1
                // for a segment ending in the piece, 2 for one going on past it
                __dpl_sycl::__local_accessor<_Acc> __local_acc(sycl::range<1>(2 * __sub_groups), __cgh);
                __dpl_sycl::__local_accessor<_Size> __local_segments(sycl::range<1>(2 * __sub_groups), __cgh);
                __dpl_sycl::__local_accessor<std::uint8_t> __local_states(sycl::range<1>(__sub_groups), __cgh);
                __cgh.parallel_for<_Name...>(
                    sycl::nd_range<1>(sycl::range<1>(__launch_groups * __work_group_size),
                                      sycl::range<1>(__work_group_size)),
                    [=](sycl::nd_item<1> __item) [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(32)]] {
                        _BinaryOperation __op = __binary_op;
                        std::uint32_t* const __count = __counts_acc.__data();
                        _Acc* const __heads = __partials_acc.__data();
                        _Acc* const __tails = __heads + __num_groups;
                        _Acc* const __carries = __tails + __num_groups;
                        auto __begin_of = [&](_Size __s) {
                            return oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__s], __n);
                        };
                        auto __end_of = [&](_Size __s) {
                            return oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__s + 1], __n);
                        };
                        auto __combine = [&](const _Acc& __a, const _Acc& __b) {
                            return oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                std::invoke(__op, __a, __b));
                        };

                        // The empty segments get the initial value, in the first portion; the others are written
                        // by the pieces
                        if (__portion == 0)
                        {
                            for (std::size_t __s = __item.get_global_linear_id(); __s < std::size_t(__m);
                                 __s += __item.get_global_range(0))
                            {
                                if (__end_of(_Size(__s)) <= __begin_of(_Size(__s)))
                                    __result[__s] = oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init);
                            }
                        }

                        const auto __sg = __item.get_sub_group();
                        const std::size_t __lane = __sg.get_local_linear_id();
                        const std::size_t __sg_id = __sg.get_group_linear_id();
                        const std::size_t __group = __item.get_group(0);
                        const _Size __portion_end = _Size(__portion_end_index);
                        const _Size __group_begin = _Size(__portion_begin + __group * __group_size);
                        const _Size __group_end = std::min<_Size>(__portion_end, __group_begin + _Size(__group_size));
                        const _Size __piece_begin =
                            std::min<_Size>(__group_end, __group_begin + _Size(__sg_id * __ipi * __lanes));
                        const _Size __piece_end =
                            std::min<_Size>(__group_end, __piece_begin + _Size(__ipi * __lanes));

                        // The head and the tail of the piece, on lane 0
                        _Acc __head{}, __tail{};
                        std::uint8_t __head_state = 0;
                        _Size __head_segment = __m, __tail_segment = __m;
                        if (__piece_begin < __piece_end)
                        {
                            // The segments of the piece in order, from the one covering its start; the same for
                            // every work-item of the sub-group
                            _Size __s = std::max<_Size>(
                                0, __segmented_reduce_flat_last_begin(__offsets, __m, __n, __piece_begin));
                            for (; __s < __m; ++__s)
                            {
                                const _Size __b = __begin_of(__s);
                                if (__b >= __piece_end)
                                    break;
                                const _Size __e = __end_of(__s);
                                const _Size __lo = std::max<_Size>(__b, __piece_begin);
                                const _Size __hi = std::min<_Size>(__e, __piece_end);
                                if (__lo < __hi)
                                {
                                    const bool __first = __lo == __b;
                                    // A range of one element that does not begin its segment is that element
                                    const _Acc __acc =
                                        !__first && __hi - __lo == 1
                                            ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__input[__lo])
                                            : __segmented_reduce_by_sub_group<_Acc>(__sg, __lo, __hi, __first, __init,
                                                                                    __op, __input);
                                    if (!__first)
                                    {
                                        __head = __acc;
                                        __head_state = __e > __piece_end ? 2 : 1;
                                        __head_segment = __s;
                                    }
                                    if (__first && __e <= __piece_end && __lane == 0)
                                        __result[__s] = __acc;
                                    if (__first && __e > __piece_end)
                                    {
                                        __tail = __acc;
                                        __tail_segment = __s;
                                    }
                                }
                                // A later segment begins at or after the end of this one
                                if (__e >= __piece_end)
                                    break;
                            }
                        }
                        if (__lane == 0)
                        {
                            __local_acc[__sg_id] = __head;
                            __local_acc[__sub_groups + __sg_id] = __tail;
                            __local_segments[__sg_id] = __head_segment;
                            __local_segments[__sub_groups + __sg_id] = __tail_segment;
                            __local_states[__sg_id] = __head_state;
                        }
                        sycl::group_barrier(__item.get_group());
                        if (__sg_id != 0 || __lane >= __sub_groups)
                            return;

                        // The partial result __acc of segment __s of this work-group, of its tail if the segment
                        // begins in it, of its head otherwise, combined with those of the other work-groups of the
                        // portion and with the carry into it; the segment going on past the portion is carried
                        auto __finish = [&](const _Size __s, _Acc __acc) {
                            const _Size __b = __begin_of(__s);
                            const _Size __e = __end_of(__s);
                            const std::size_t __first_group =
                                __b > _Size(__portion_begin) ? std::size_t(__b - _Size(__portion_begin)) / __group_size
                                                             : 0;
                            const std::size_t __last_group =
                                std::size_t(std::min<_Size>(__e, __portion_end) - 1 - _Size(__portion_begin)) /
                                __group_size;
                            const bool __is_tail = __b >= __group_begin;
                            if (__first_group != __last_group)
                            {
                                (__is_tail ? __tails : __heads)[__group] = __acc;
                                // The partial results written before are visible to the last work-group arrived
                                _AtomicCount __arrived(__count[__first_group]);
                                if (__arrived.fetch_add(1) + 1 != std::uint32_t(__last_group - __first_group + 1))
                                    return;
                                __arrived.store(0, sycl::memory_order::relaxed);
                                __acc = __b >= _Size(__portion_begin + __first_group * __group_size)
                                            ? __tails[__first_group]
                                            : __heads[__first_group];
                                for (std::size_t __g = __first_group + 1; __g <= __last_group; ++__g)
                                    __acc = __combine(__acc, __heads[__g]);
                            }
                            if (__b < _Size(__portion_begin))
                                __acc = __combine(__carries[__portion % 2], __acc);
                            if (__e > __portion_end)
                                __carries[1 - __portion % 2] = __acc;
                            if (__e <= __portion_end)
                                __result[__s] = __acc;
                        };

                        // Lane j combines the tail of piece j with the heads of the following pieces of its segment
                        const _Size __s = __local_segments[__sub_groups + __lane];
                        if (__s < __m)
                        {
                            _Acc __acc = __local_acc[__sub_groups + __lane];
                            std::uint8_t __state = 2;
                            for (std::size_t __k = __lane + 1; __state == 2 && __k < __sub_groups; ++__k)
                            {
                                __state = __local_states[__k];
                                if (__state != 0)
                                    __acc = __combine(__acc, __local_acc[__k]);
                            }
                            // The segment ends in the work-group or goes on past it
                            if (__state == 1)
                                __result[__s] = __acc;
                            if (__state != 1)
                                __finish(__s, __acc);
                        }
                        // Lane 0 the head of the work-group, the segment begun before it
                        if (__lane == 0 && __local_states[0] != 0)
                        {
                            _Acc __acc = __local_acc[0];
                            std::uint8_t __state = __local_states[0];
                            for (std::size_t __k = 1; __state == 2 && __k < __sub_groups; ++__k)
                            {
                                __state = __local_states[__k];
                                if (__state != 0)
                                    __acc = __combine(__acc, __local_acc[__k]);
                            }
                            __finish(__local_segments[0], __acc);
                        }
                    });
            });
        }
        // Waited here, as the temporary data above is released on return
        __event.wait_and_throw();
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

    // On GPUs arithmetic types by pieces of the input that do not depend on the segments
    if constexpr (std::is_arithmetic_v<_Acc> &&
                  std::is_arithmetic_v<oneapi::dpl::__internal::__value_t<std::decay_t<_InRange>>>)
    {
        sycl::queue __q = __exec.queue();
        const __segmented_reduce_device_info __info =
            __is_gpu ? __segmented_reduce_device_info::__get(__q) : __segmented_reduce_device_info{};
        if (__info.__supported)
        {
            using _FlatKernel = __internal::__kernel_name_provider<
                __segmented_reduce_flat_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;

            __segmented_reduce_flat_submitter<_Acc, _FlatKernel>()(__q, __input, __offsets, __result, __n, __m, __init,
                                                                   __binary_op, __info);
            return;
        }
    }

    // Reduces the segments longer than __min_length in chunks; __max_segments bounds the number of such segments
    auto __reduce_in_chunks = [&](_Size __min_length, bool __write_empty, _Size __max_length, _Size __max_segments) {
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
        __reduce_in_chunks(_Size{0}, /*__write_empty=*/true, __max_length, __m);
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
        __reduce_in_chunks(__threshold, /*__write_empty=*/false, __max_length, __n / __threshold);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
