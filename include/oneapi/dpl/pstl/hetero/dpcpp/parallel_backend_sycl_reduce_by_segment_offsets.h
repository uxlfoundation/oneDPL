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

    _Size __n = {};         // the size of the input
    _Size __threshold = {}; // the length of the longest short segment
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
            __ref.fetch_max(static_cast<std::uint32_t>(std::min<std::uint64_t>(
                static_cast<std::uint64_t>(__end - __begin), std::numeric_limits<std::uint32_t>::max())));
            return;
        }
        std::get<0>(__result[__idx]) =
            oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(__input, __begin, __end, __init, __binary_op);
    }
};

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

// A short segment is reduced by one work-item, a long one in chunks, each reduced by one work-item, whose partial
// results one work-item combines; the values below were measured on GPUs and CPUs:
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
    _Size __n = {}; // the size of the input

    template <typename _OffsetEnd, typename _OffsetBegin>
    _Size
    operator()(_OffsetEnd __end, _OffsetBegin __begin) const
    {
        const _Size __b = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__begin, __n);
        const _Size __e = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__end, __n);
        return __b < __e ? __e - __b : _Size{0};
    }
};

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
        const _Size __length = __segmented_reduce_length<_Size>{__n}(std::get<1>(__bounds), std::get<0>(__bounds));
        return __length > __min_length ? __length / __chunk_size + (__length % __chunk_size != 0) : _Size{0};
    }
};

// Reduces one chunk of a long segment. The first chunk of a segment takes the remainder of its length and starts
// from the operation on the initial value and its first element; every other chunk has exactly __chunk_size elements
// and starts from the operation on its first two elements.
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_chunk_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _Size __n = {};            // the size of the input
    _Size __num_segments = {}; // the number of segments
    _Size __chunk_size = {};   // the length of a chunk
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

template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_combine_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _Size __num_partials = {};  // the number of the computed partial results
    bool __write_empty = false; // whether a segment without chunks gets the initial value; otherwise it is left as is
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

// Whether a range reaches its elements through a pointer, so that consecutive elements can be loaded as one vector
template <typename _Range, typename = void>
struct __segmented_reduce_pointer_input : std::false_type
{
};
template <typename _Range>
struct __segmented_reduce_pointer_input<
    _Range, std::enable_if_t<std::is_pointer_v<decltype(std::declval<const _Range&>().begin())>>> : std::true_type
{
};

// A range takes the vector loads below from this many whole vectors on: a shorter one is reduced faster element by
// element, as the alignment, the head and the tail of the vectors cost more than the vectors save, as measured
inline constexpr std::size_t __segmented_reduce_sub_group_min_vectors = 8;

// The operation on four values in two pairs, each made an accumulator; taken by value, so that the four are loaded
// before the operation waits for any of them
template <typename _Acc, typename _BinaryOperation, typename _A, typename _B, typename _C, typename _D>
_Acc
__segmented_reduce_four(_BinaryOperation& __binary_op, _A __a, _B __b, _C __c, _D __d)
{
    return std::invoke(__binary_op,
                       oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(std::invoke(__binary_op, __a, __b)),
                       oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(std::invoke(__binary_op, __c, __d)));
}

// Combines the accumulators of the first __active lanes of a sub-group on lane 0: by the sub-group reduction for an
// operation with an identity the group algorithms know, the other lanes giving it; else by a tree of shifts known at
// compile time, as a shift known only at run time reaches the registers of the other lanes indirectly
template <typename _SubGroup, typename _Acc, typename _BinaryOperation>
_Acc
__segmented_reduce_lanes(const _SubGroup& __sg, _Acc __acc, const std::uint32_t __active, _BinaryOperation& __binary_op)
{
    const std::uint32_t __lane = __sg.get_local_linear_id();
    if constexpr (oneapi::dpl::unseq_backend::__can_use_group_reduce_scan<_BinaryOperation, _Acc>::value)
    {
        return sycl::reduce_over_group(
            __sg, __lane < __active ? __acc : sycl::known_identity_v<_BinaryOperation, _Acc>, __binary_op);
    }
    else
    {
        auto __step = [&](auto __shift) {
            if (__shift < __active)
            {
                const _Acc __other = sycl::shift_group_left(__sg, __acc, __shift);
                if (__lane + __shift < __active)
                    __acc = std::invoke(__binary_op, __acc, __other);
            }
        };
        __step(std::integral_constant<std::uint32_t, 1>{});
        __step(std::integral_constant<std::uint32_t, 2>{});
        __step(std::integral_constant<std::uint32_t, 4>{});
        __step(std::integral_constant<std::uint32_t, 8>{});
        __step(std::integral_constant<std::uint32_t, 16>{});
        return __acc;
    }
}

// Reduces the elements [__base + __first, __base + __last), at least one, by a sub-group onto lane 0: the lanes take
// the elements in turn, so that the loads are coalesced, which reorders the operands as the required commutativity
// allows. A range that begins its segment starts from the operation on the initial value and its first element.
// Arithmetic types only; the positions are counted in 32 bits from __base, the start of a piece.
template <typename _Acc, typename _SubGroup, typename _Size, typename _InitType, typename _BinaryOperation,
          typename _Input>
_Acc
__segmented_reduce_by_sub_group(const _SubGroup& __sg, const _Size __base, std::uint32_t __first,
                                const std::uint32_t __last, const bool __is_first, const _InitType __init,
                                _BinaryOperation __binary_op, const _Input& __input)
{
    const std::uint32_t __lane = __sg.get_local_linear_id();
    const std::uint32_t __lanes = __sg.get_local_linear_range();
    auto __at = [&]() {
        if constexpr (__segmented_reduce_pointer_input<_Input>::value)
        {
            const auto __p = __input.begin() + __base;
            return [__p](std::uint32_t __i) -> decltype(auto) { return __p[__i]; };
        }
        else
            return [&__input, __base](std::uint32_t __i) -> decltype(auto) { return __input[__base + _Size(__i)]; };
    }();
    auto __to_acc = [](const auto& __x) { return oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__x); };
    // The accumulators are set up from their first values; a lane not taking part holds none, and what it passes on
    // to the other lanes is never used
    oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __head;
    if (__is_first)
        __head.__setup(__to_acc(std::invoke(__binary_op, __init, __at(__first++))));

    // Every lane that takes part gets at least two elements: its accumulator starts from the operation on two of them,
    // never from an element alone, which may mean something else than the element to the operation
    std::uint32_t __active = std::min<std::uint32_t>(__lanes, (__last - __first) / 2);
    oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __acc;
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
            const _T* const __p = __input.begin() + __base;
            // Four elements a vector, two of 8 bytes, so that a vector stays within 16 bytes and is aligned as often
            constexpr std::uint32_t __w = sizeof(_T) == 8 ? 2 : 4;
            using _V = sycl::vec<_T, __w>;
            constexpr std::uint32_t __vb = sizeof(_V);
            const std::uint32_t __mis = std::uint32_t(reinterpret_cast<std::uintptr_t>(__p + __first) % __vb);
            const std::uint32_t __a = std::min<std::uint32_t>(__last, __first + (__mis == 0 ? 0 : (__vb - __mis) / sizeof(_T)));
            const std::uint32_t __nv = (__last - __a) / __w;
            if (__nv >= __segmented_reduce_sub_group_min_vectors)
            {
                __by_vectors = true;
                __active = std::min<std::uint32_t>(__lanes, __nv);
                auto __load4 = [&](std::uint32_t __k) -> _Acc {
                    const _V __v = *reinterpret_cast<const _V*>(__p + __a + __w * __k);
                    if constexpr (__w == 2)
                        return oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                            std::invoke(__binary_op, _T(__v[0]), _T(__v[1])));
                    else
                        return std::invoke(__binary_op,
                                           oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                               std::invoke(__binary_op, _T(__v[0]), _T(__v[1]))),
                                           oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                               std::invoke(__binary_op, _T(__v[2]), _T(__v[3]))));
                };
                if (__lane < __active)
                {
                    std::uint32_t __k = __lane;
                    __acc.__setup(__load4(__k));
                    for (__k += __active; __k + 3 * __active < __nv; __k += 4 * __active)
                        __acc.__v = std::invoke(__binary_op, __acc.__v,
                                                __segmented_reduce_four<_Acc>(__binary_op, __load4(__k),
                                                                              __load4(__k + __active),
                                                                              __load4(__k + 2 * __active),
                                                                              __load4(__k + 3 * __active)));
                    for (; __k < __nv; __k += __active)
                        __acc.__v = std::invoke(__binary_op, __acc.__v, __load4(__k));
                }
                // The unaligned head and the tail, at most three elements each (one of 8 bytes), to lane 0
                if (__lane == 0)
                {
                    for (std::uint32_t __i = __first; __i < __a; ++__i)
                        __acc.__v = std::invoke(__binary_op, __acc.__v, __p[__i]);
                    for (std::uint32_t __i = __a + __w * __nv; __i < __last; ++__i)
                        __acc.__v = std::invoke(__binary_op, __acc.__v, __p[__i]);
                }
            }
        }
    }
    if (!__by_vectors && __lane < __active)
    {
        std::uint32_t __i = __first + __lane;
        __acc.__setup(__to_acc(std::invoke(__binary_op, __at(__i), __at(__i + __active))));
        for (__i += 2 * __active; __i + 3 * __active < __last; __i += 4 * __active)
            __acc.__v = std::invoke(__binary_op, __acc.__v,
                                    __segmented_reduce_four<_Acc>(__binary_op, __at(__i), __at(__i + __active),
                                                                  __at(__i + 2 * __active), __at(__i + 3 * __active)));
        for (; __i < __last; __i += __active)
            __acc.__v = std::invoke(__binary_op, __acc.__v, __at(__i));
    }
    if (__active > 0)
        __acc.__v = __segmented_reduce_lanes(__sg, __acc.__v, __active, __binary_op);
    // A range of one element more than the initial value takes it alone
    if (__is_first)
        return __active > 0       ? __to_acc(std::invoke(__binary_op, __head.__v, __acc.__v))
               : __first < __last ? __to_acc(std::invoke(__binary_op, __head.__v, __at(__first)))
                                  : __head.__v;
    return __acc.__v;
}

// On GPUs, for arithmetic types: every sub-group takes a piece of consecutive elements, finds the segment covering
// its start in the offsets and reduces the segments of the piece one after another with coalesced loads, writing the
// ones it holds whole. The segment begun before the piece gives its head, the one going on past its end its tail; a
// second kernel combines every tail with the heads of the following pieces its segment covers. Above a size the
// pieces of a work-group are first combined in its local memory, and the second kernel combines the work-groups.
inline constexpr std::size_t __segmented_reduce_gpu_flat_lanes = 32;
// A work-item takes at least this many elements up to this input size, and four times as many above it, so that the
// search of a piece and the combination of its parts are paid for, and at most four times as many, so that the larger
// inputs are spread over more work-groups than the device runs at once, as measured
inline constexpr std::size_t __segmented_reduce_gpu_flat_min_inputs_per_item = 16;
inline constexpr std::size_t __segmented_reduce_gpu_flat_small_input = std::size_t(1) << 20;
// Segments of at most this many elements on average make the walk of a piece cost more than its loads: there the
// elements of a work-item are chosen so that the pieces about fill the hardware threads of the device, as measured
inline constexpr std::size_t __segmented_reduce_gpu_flat_short_average_segment = 2048;
inline constexpr std::size_t __segmented_reduce_gpu_flat_max_work_group = 256;
// A segment of at most this many elements that a piece holds whole is reduced by a lane alone, with the following ones
// as long as they are such segments too, a lane each: the sub-group would spend more on combining its lanes and on
// walking the segments one at a time than on the elements, as measured
inline constexpr std::size_t __segmented_reduce_gpu_flat_tiny_segment = 32;
// Above this input the pieces of a work-group are combined in its local memory before the second kernel: there the
// stores of every piece and the second kernel over them cost more than the wait of the work-group for its slowest
// sub-group, which the many work-groups hide, as measured
inline constexpr std::size_t __segmented_reduce_gpu_flat_group_combine_input = std::size_t(1) << 24;

// The index of the last of the __m segments that begins at or before __v, or -1, searched from the segment __guess:
// steps doubling from it up or down bracket the first segment that begins after __v, and a binary search finds it in
// the bracket, so that a guess close to the answer takes a few reads of the offsets instead of log2(__m)
template <typename _Size, typename _OffsetRange>
_Size
__segmented_reduce_flat_last_begin(const _OffsetRange& __offsets, const _Size __m, const _Size __n, const _Size __v,
                                   _Size __guess, const float __segments_per_element, const int __steps)
{
    auto __begins_after = [&](_Size __i) {
        return oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__i], __n) > __v;
    };
    __guess = std::min<_Size>(std::max<_Size>(__guess, _Size{0}), __m - 1);
    // The guess moved by the distance from its offset to __v, as if the segments around it were of the same length
    for (int __step = 0; __step < __steps; ++__step)
    {
        const _Size __at = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__guess], __n);
        const _Size __next = std::min<_Size>(
            std::max<_Size>(__guess + _Size(float(__v - __at) * __segments_per_element), _Size{0}), __m - 1);
        if (__next == __guess)
            break;
        __guess = __next;
    }
    // The first segment that begins after __v is in [__lo, __hi]: every one before __lo begins at or before it
    _Size __lo = 0, __hi = __m;
    if (__begins_after(__guess))
    {
        __hi = __guess;
        for (_Size __step = 1; __hi > 0; __step *= 2)
        {
            const _Size __below = std::max<_Size>(_Size{0}, __guess - __step);
            if (!__begins_after(__below))
            {
                __lo = __below + 1;
                break;
            }
            __hi = __below;
        }
    }
    else
    {
        __lo = __guess + 1;
        for (_Size __step = 1; __lo < __m; __step *= 2)
        {
            const _Size __above = std::min<_Size>(__m, __guess + __step);
            if (__above == __m || __begins_after(__above))
            {
                __hi = __above;
                break;
            }
            __lo = __above + 1;
        }
    }
    while (__lo < __hi)
    {
        const _Size __mid = __lo + (__hi - __lo) / 2;
        if (__begins_after(__mid))
            __hi = __mid;
        else
            __lo = __mid + 1;
    }
    return __lo - 1;
}

inline bool
__segmented_reduce_flat_supported(const sycl::queue& __q)
{
    return oneapi::dpl::__internal::__supports_sub_group_size(__q, __segmented_reduce_gpu_flat_lanes);
}

// The launch for __n elements in __m segments: twice as many work-groups as Xe cores, as reduce-then-scan has, or
// with short segments on average a piece per hardware thread, the elements of a work-item within the bounds above,
// and the work-groups halved, down to one sub-group, until there are at least two per Xe core
struct __segmented_reduce_flat_geometry
{
    std::size_t __work_group_size = {}; // a multiple of the size of a sub-group
    std::size_t __num_groups = {};
    std::size_t __inputs_per_item = {};

    __segmented_reduce_flat_geometry(const sycl::queue& __q, const std::size_t __n, const std::size_t __m)
    {
        constexpr std::size_t __lanes = __segmented_reduce_gpu_flat_lanes;

        __work_group_size =
            oneapi::dpl::__internal::__max_work_group_size(__q, __segmented_reduce_gpu_flat_max_work_group) / __lanes *
            __lanes;
        const std::size_t __compute_units = oneapi::dpl::__internal::__max_compute_units(__q);
        // Eight compute units per Xe core
        const std::size_t __min_groups = 2 * std::max<std::size_t>(1, __compute_units / 8);
        // Eight hardware threads per compute unit, a sub-group each
        const bool __short_segments = __n <= __m * __segmented_reduce_gpu_flat_short_average_segment;
        const std::size_t __items = __short_segments ? 8 * std::max<std::size_t>(1, __compute_units) * __lanes
                                                     : __min_groups * __work_group_size;
        const std::size_t __min_inputs = __short_segments || __n <= __segmented_reduce_gpu_flat_small_input
                                             ? __segmented_reduce_gpu_flat_min_inputs_per_item
                                             : 4 * __segmented_reduce_gpu_flat_min_inputs_per_item;
        __inputs_per_item = std::clamp<std::size_t>((__n + __items - 1) / __items, __min_inputs,
                                                    4 * __segmented_reduce_gpu_flat_min_inputs_per_item);
        auto __groups = [&](std::size_t __size) {
            return (__n + __inputs_per_item * __size - 1) / (__inputs_per_item * __size);
        };
        while (__work_group_size > __lanes && __groups(__work_group_size) < __min_groups)
            __work_group_size = std::max(__lanes, __work_group_size / 2 / __lanes * __lanes);
        __num_groups = __groups(__work_group_size);
    }
};

template <typename... _Name>
class __segmented_reduce_flat_kernel;
template <typename... _Name>
class __segmented_reduce_flat_group_kernel;
template <typename... _Name>
class __segmented_reduce_flat_tail_kernel;

// The local memory of the first kernel where it combines the pieces of a work-group: the heads and the tails of the
// pieces, the segments of the tails and the states of the heads, 0 for none, 1 for a segment ending in the piece, 2 for
// one going on past it; none where it does not
template <bool _InGroup, typename _Acc, typename _Size>
struct __segmented_reduce_flat_local
{
    __segmented_reduce_flat_local(sycl::handler&, std::size_t) {}
};

template <typename _Acc, typename _Size>
struct __segmented_reduce_flat_local<true, _Acc, _Size>
{
    __dpl_sycl::__local_accessor<_Acc> __acc;
    __dpl_sycl::__local_accessor<_Size> __segments;
    __dpl_sycl::__local_accessor<std::uint8_t> __states;

    __segmented_reduce_flat_local(sycl::handler& __cgh, std::size_t __sub_groups)
        : __acc(sycl::range<1>(2 * __sub_groups), __cgh), __segments(sycl::range<1>(__sub_groups), __cgh),
          __states(sycl::range<1>(__sub_groups), __cgh)
    {
    }
};

template <typename _KernelName>
struct __segmented_reduce_kernel_name_tag
{
};

template <typename... _KernelName, typename _Range, typename _Kernel>
void
__segmented_reduce_parallel_for(sycl::handler& __cgh,
                                __segmented_reduce_kernel_name_tag<__internal::__optional_kernel_name<_KernelName...>>,
                                const _Range& __range, const _Kernel& __kernel)
{
    __cgh.parallel_for<_KernelName...>(__range, __kernel);
}

template <typename _Acc, typename _KernelName, typename _GroupKernelName, typename _TailKernelName>
struct __segmented_reduce_flat_submitter;

template <typename _Acc, typename... _Name, typename _GroupKernelName, typename... _TailName>
struct __segmented_reduce_flat_submitter<_Acc, __internal::__optional_kernel_name<_Name...>, _GroupKernelName,
                                         __internal::__optional_kernel_name<_TailName...>>
{
    template <typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size, typename _InitType,
              typename _BinaryOperation>
    void
    operator()(sycl::queue& __q, _InRange&& __input, _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n,
               const _Size __m, _InitType __init, _BinaryOperation __binary_op) const
    {
        // The accumulators in __lazy_ctor_storage are never destroyed: the path takes arithmetic types only
        static_assert(std::is_trivially_destructible_v<_Acc>);
        constexpr std::size_t __lanes = __segmented_reduce_gpu_flat_lanes;
        const __segmented_reduce_flat_geometry __geometry(__q, std::size_t(__n), std::size_t(__m));
        const std::size_t __work_group_size = __geometry.__work_group_size;
        const std::size_t __sub_groups = __work_group_size / __lanes;
        const std::size_t __num_groups = __geometry.__num_groups;
        const std::size_t __ipi = __geometry.__inputs_per_item;
        const std::size_t __piece_size = __ipi * __lanes;
        // The units of the second kernel: the work-groups where their pieces are combined in local memory, else the
        // pieces
        const bool __in_group = std::size_t(__n) > __segmented_reduce_gpu_flat_group_combine_input;
        const std::size_t __num_units = __in_group ? __num_groups : __num_groups * __sub_groups;
        const std::size_t __unit_size = __in_group ? __piece_size * __sub_groups : __piece_size;
        // The segments per element, from which a piece guesses the segment covering its start as if the segments were
        // of the same length; in single precision, which every GPU computes natively
        const float __segments_per_element = float(__m) / float(__n);
        // The guess refined by interpolation where there are more segments than pieces, where it misses by many of
        // them; with fewer, it is about right and a step only adds a load, as measured
        const int __interpolation_steps = std::size_t(__m) > __num_groups * __sub_groups ? 2 : 0;

        // The segments of the tails of the units and, after them, the partial results of their heads and of their
        // tails, in one storage, which costs a call less than two, as measured; of the more aligned of the two types,
        // so that both arrays start aligned
        using _Unit = std::conditional_t<(alignof(_Acc) > alignof(_Size)), _Acc, _Size>;
        const std::size_t __segment_slots = (__num_units * sizeof(_Size) + sizeof(_Unit) - 1) / sizeof(_Unit);
        const std::size_t __partial_slots = (2 * __num_units * sizeof(_Acc) + sizeof(_Unit) - 1) / sizeof(_Unit);
        __device_storage<_Unit> __units_storage{__q, __segment_slots + __partial_slots};

        // The first kernel, with the combination of the pieces of a work-group in its local memory or without it: the
        // two are apart, so that the one without has no local memory to set up
        auto __pieces = [&](auto __in_group_tag, auto __kernel_name) {
            constexpr bool __in_group_v = decltype(__in_group_tag)::value;
            return __q.submit([&](sycl::handler& __cgh) {
                oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
                auto __units_acc = __get_accessor(sycl::write_only, __units_storage, __cgh, __dpl_sycl::__no_init{});
                const __segmented_reduce_flat_local<__in_group_v, _Acc, _Size> __local(__cgh, __sub_groups);
                __segmented_reduce_parallel_for(
                    __cgh, __kernel_name,
                    sycl::nd_range<1>(sycl::range<1>(__num_groups * __work_group_size),
                                      sycl::range<1>(__work_group_size)),
                    [=](sycl::nd_item<1> __item) [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(32)]] {
                        _BinaryOperation __op = __binary_op;
                        _Size* const __tail_segment_of = reinterpret_cast<_Size*>(__units_acc.__data());
                        _Acc* const __heads = reinterpret_cast<_Acc*>(__units_acc.__data() + __segment_slots);
                        _Acc* const __tails = __heads + __num_units;
                        auto __begin_of = [&](_Size __s) {
                            return oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__s], __n);
                        };
                        auto __end_of = [&](_Size __s) {
                            return oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__s + 1], __n);
                        };

                        const _Acc __init_acc = oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init);

                        const auto __sg = __item.get_sub_group();
                        const std::uint32_t __lane = __sg.get_local_linear_id();
                        const std::uint32_t __sg_id = __sg.get_group_linear_id();
                        const std::uint32_t __group = __item.get_group(0);
                        const std::uint32_t __piece = __group * std::uint32_t(__sub_groups) + __sg_id;
                        const _Size __piece_begin = std::min<_Size>(__n, _Size(std::size_t(__piece) * __piece_size));
                        const _Size __piece_end = std::min<_Size>(__n, __piece_begin + _Size(__piece_size));

                        oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __head, __tail;
                        std::uint8_t __head_state = 0;
                        _Size __tail_segment = __m;
                        if (__piece_begin < __piece_end)
                        {
                            // The segments of the piece in order, the same for every lane: from the last one
                            // beginning before the piece, so that the walk meets the empty ones at its start too; the
                            // last piece also takes the ones at n
                            const bool __last_piece = __piece_end == __n;
                            const _Size __guess = _Size(float(__piece_begin) * __segments_per_element);
                            _Size __s = std::max<_Size>(
                                0, __segmented_reduce_flat_last_begin(__offsets, __m, __n, __piece_begin - 1, __guess,
                                                                   __segments_per_element, __interpolation_steps));
                            auto __tiny = [&](const _Size __b, const _Size __e) {
                                return __b >= __piece_begin && __e <= __piece_end &&
                                       __e - __b <= _Size(__segmented_reduce_gpu_flat_tiny_segment);
                            };
                            auto __element = [&](const std::uint32_t __i) -> decltype(auto) {
                                if constexpr (__segmented_reduce_pointer_input<std::decay_t<decltype(__input)>>::value)
                                    return (__input.begin() + __piece_begin)[__i];
                                else
                                    return __input[__piece_begin + _Size(__i)];
                            };
                            for (; __s < __m; ++__s)
                            {
                                const _Size __b = __begin_of(__s);
                                if (__b >= __piece_end && !__last_piece)
                                    break;
                                const _Size __e = __end_of(__s);
                                // A run of very short segments, a lane each: the lanes before the first one whose segment
                                // is not such a segment reduce theirs alone, from the operation on the initial value and
                                // the first element; the empty ones were written above
                                if (__tiny(__b, __e))
                                {
                                    const _Size __mine = __s + _Size(__lane);
                                    _Size __mb = __b, __me = __b;
                                    bool __run = false;
                                    if (__mine < __m)
                                    {
                                        __mb = __begin_of(__mine);
                                        __me = __end_of(__mine);
                                        __run = __tiny(__mb, __me);
                                    }
                                    const std::uint32_t __count = sycl::reduce_over_group(
                                        __sg, __run ? std::uint32_t(__lanes) : __lane, sycl::minimum<std::uint32_t>());
                                    if (__lane < __count && __mb == __me)
                                        __result[__mine] = __init_acc;
                                    if (__lane < __count && __mb < __me)
                                    {
                                        const std::uint32_t __lo = std::uint32_t(__mb - __piece_begin);
                                        const std::uint32_t __hi = std::uint32_t(__me - __piece_begin);
                                        auto __to_acc = [](const auto& __x) {
                                            return oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__x);
                                        };
                                        _Acc __acc = __to_acc(std::invoke(__op, __init, __element(__lo)));
                                        std::uint32_t __i = __lo + 1;
                                        for (; __i + 3 < __hi; __i += 4)
                                            __acc = __to_acc(std::invoke(
                                                __op, __acc,
                                                __segmented_reduce_four<_Acc>(__op, __element(__i),
                                                                              __element(__i + 1),
                                                                              __element(__i + 2),
                                                                              __element(__i + 3))));
                                        for (; __i < __hi; ++__i)
                                            __acc = __to_acc(std::invoke(__op, __acc, __element(__i)));
                                        __result[__mine] = __acc;
                                    }
                                    // The loop goes on after the run; the last segment of the run may end the piece
                                    __s += _Size(__count) - 1;
                                    if (sycl::select_from_group(__sg, __me, __count - 1) >= __piece_end &&
                                        !__last_piece)
                                        break;
                                    continue;
                                }
                                const std::uint32_t __lo = __b > __piece_begin ? std::uint32_t(__b - __piece_begin) : 0;
                                const std::uint32_t __hi =
                                    std::uint32_t(std::max<_Size>(__piece_begin, std::min<_Size>(__e, __piece_end)) -
                                                  __piece_begin);
                                if (__b == __e && __b >= __piece_begin && __lane == 0)
                                    __result[__s] = __init_acc;
                                if (__lo < __hi)
                                {
                                    const bool __first = __b >= __piece_begin;
                                    // A range of one element that does not begin its segment is that element
                                    const _Acc __acc = !__first && __hi - __lo == 1
                                                           ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                                                 __input[__piece_begin + _Size(__lo)])
                                                           : __segmented_reduce_by_sub_group<_Acc>(
                                                                 __sg, __piece_begin, __lo, __hi, __first, __init, __op, __input);
                                    if (!__first)
                                    {
                                        __head.__setup(__acc);
                                        __head_state = __e > __piece_end ? 2 : 1;
                                    }
                                    if (__first && __e <= __piece_end && __lane == 0)
                                        __result[__s] = __acc;
                                    if (__first && __e > __piece_end)
                                    {
                                        __tail.__setup(__acc);
                                        __tail_segment = __s;
                                    }
                                }
                                // A later segment begins at or after the end of this one
                                if (__e >= __piece_end && !__last_piece)
                                    break;
                            }
                        }
                        // A piece covered by a segment begun before it has a head, which only the second kernel reads
                        if constexpr (!__in_group_v)
                        {
                            if (__lane == 0 && __head_state != 0)
                                __heads[__piece] = __head.__v;
                            if (__lane == 0 && __tail_segment < __m)
                                __tails[__piece] = __tail.__v;
                            if (__lane == 0)
                                __tail_segment_of[__piece] = __tail_segment;
                        }
                        else
                        {

                            if (__lane == 0)
                            {
                                if (__head_state != 0)
                                    __local.__acc[__sg_id] = __head.__v;
                                if (__tail_segment < __m)
                                    __local.__acc[__sub_groups + __sg_id] = __tail.__v;
                                __local.__segments[__sg_id] = __tail_segment;
                                __local.__states[__sg_id] = __head_state;
                            }
                            // No tail unless found below; the barrier orders this before the writes after it
                            if (__sg_id == 0 && __lane == 0)
                                __tail_segment_of[__group] = __m;
                            sycl::group_barrier(__item.get_group());
                            if (__sg_id != 0 || __lane >= __sub_groups)
                                return;
                            auto __combine = [&](const _Acc& __a, const _Acc& __b) {
                                return oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(std::invoke(__op, __a, __b));
                            };

                            // Lane j combines the tail of piece j with the heads of the following pieces of its segment; the
                            // segment going on past the work-group is its tail
                            const _Size __s = __local.__segments[__lane];
                            if (__s < __m)
                            {
                                _Acc __acc = __local.__acc[__sub_groups + __lane];
                                std::uint8_t __state = 2;
                                for (std::size_t __k = __lane + 1; __state == 2 && __k < __sub_groups; ++__k)
                                {
                                    __state = __local.__states[__k];
                                    if (__state != 0)
                                        __acc = __combine(__acc, __local.__acc[__k]);
                                }
                                if (__state == 1)
                                    __result[__s] = __acc;
                                if (__state != 1)
                                {
                                    __tails[__group] = __acc;
                                    __tail_segment_of[__group] = __s;
                                }
                            }
                            if (__lane == 0 && __local.__states[0] != 0)
                            {
                                _Acc __acc = __local.__acc[0];
                                std::uint8_t __state = __local.__states[0];
                                for (std::size_t __k = 1; __state == 2 && __k < __sub_groups; ++__k)
                                {
                                    __state = __local.__states[__k];
                                    if (__state != 0)
                                        __acc = __combine(__acc, __local.__acc[__k]);
                                }
                                __heads[__group] = __acc;
                            }
                        }
                    });
            });
        };
        sycl::event __event =
            __in_group
            ? __pieces(std::true_type{}, __segmented_reduce_kernel_name_tag<_GroupKernelName>{})
            : __pieces(std::false_type{},
                       __segmented_reduce_kernel_name_tag<__internal::__optional_kernel_name<_Name...>>{});

        // The tail of every unit with the heads of the following units its segment covers, a sub-group per unit; a
        // single unit has none
        if (__num_units > 1)
        {
            constexpr std::size_t __tail_work_group_size = 256;
            constexpr std::size_t __tail_sub_groups = __tail_work_group_size / __lanes;
            const std::size_t __tail_groups = (__num_units + __tail_sub_groups - 1) / __tail_sub_groups;
            __event = __q.submit([&](sycl::handler& __cgh) {
                oneapi::dpl::__ranges::__require_access(__cgh, __offsets, __result);
                auto __units_acc = __get_accessor(sycl::read_only, __units_storage, __cgh);
                __cgh.depends_on(__event);
                __cgh.parallel_for<_TailName...>(
                    sycl::nd_range<1>(sycl::range<1>(__tail_groups * __tail_work_group_size),
                                      sycl::range<1>(__tail_work_group_size)),
                    [=](sycl::nd_item<1> __item) [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(32)]] {
                        _BinaryOperation __op = __binary_op;
                        const _Acc* const __heads =
                            reinterpret_cast<const _Acc*>(__units_acc.__data() + __segment_slots);
                        const _Acc* const __tails = __heads + __num_units;
                        const auto __sg = __item.get_sub_group();
                        const std::uint32_t __lane = __sg.get_local_linear_id();
                        const std::uint32_t __unit =
                            __item.get_group(0) * __tail_sub_groups + __sg.get_group_linear_id();
                        if (__unit >= __num_units)
                            return;
                        const _Size __s = reinterpret_cast<const _Size*>(__units_acc.__data())[__unit];
                        if (__s >= __m)
                            return;
                        // The tail is loaded together with the end of its segment: neither waits for the other
                        const _Acc __tail = __tails[__unit];
                        const _Size __e =
                            oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__s + 1], __n);
                        const std::uint32_t __heads_count = std::uint32_t(std::size_t(__e - 1) / __unit_size) - __unit;
                        if (__heads_count == 0 && __lane == 0)
                            __result[__s] = __tail;
                        if (__heads_count == 0)
                            return;
                        // The heads of the following units its segment covers, a head a lane at once and the rest of
                        // a long segment four heads a lane at a time, so that they do not wait for each other: a head
                        // is already an accumulator, so a lane may start from one alone
                        const _Acc* const __first_head = __heads + (__unit + 1);
                        oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __acc;
                        if (__lane < __heads_count)
                            __acc.__setup(__first_head[__lane]);
                        std::uint32_t __k = __lane + __lanes;
                        for (; __k + 3 * __lanes < __heads_count; __k += 4 * __lanes)
                            __acc.__v = std::invoke(
                                __op, __acc.__v,
                                __segmented_reduce_four<_Acc>(__op, __first_head[__k], __first_head[__k + __lanes],
                                                              __first_head[__k + 2 * __lanes],
                                                              __first_head[__k + 3 * __lanes]));
                        for (; __k < __heads_count; __k += __lanes)
                            __acc.__v = std::invoke(__op, __acc.__v, __first_head[__k]);
                        __acc.__v = __segmented_reduce_lanes(__sg, __acc.__v,
                                                             std::min<std::uint32_t>(__heads_count, __lanes), __op);
                        if (__lane == 0)
                            __result[__s] = oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                std::invoke(__op, __tail, __acc.__v));
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

    if constexpr (std::is_arithmetic_v<_Acc> &&
                  std::is_arithmetic_v<oneapi::dpl::__internal::__value_t<std::decay_t<_InRange>>>)
    {
        sycl::queue __q = __exec.queue();
        if (__is_gpu && __segmented_reduce_flat_supported(__q))
        {
            using _FlatKernel = __internal::__kernel_name_provider<
                __segmented_reduce_flat_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;
            using _FlatGroupKernel = __internal::__kernel_name_provider<
                __segmented_reduce_flat_group_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;
            using _FlatTailKernel = __internal::__kernel_name_provider<
                __segmented_reduce_flat_tail_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;

            __segmented_reduce_flat_submitter<_Acc, _FlatKernel, _FlatGroupKernel, _FlatTailKernel>()(
                __q, __input, __offsets, __result, __n, __m, __init, __binary_op);
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
                : __n;
        if (__max_length == 0)
        {
            __parallel_reduce_by_segment_offsets_fill<_Acc>(_BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
                                                            std::forward<_OutRange>(__result), __m, __init);
            return;
        }
        __reduce_in_chunks(_Size{0}, /*__write_empty=*/true, __max_length, __m);
        return;
    }

    // The short segments, one work-item each; the long ones only report the longest length, through a buffer, which
    // costs less on GPUs than a copy from device USM
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

    if (__max_length > 0)
        __reduce_in_chunks(__threshold, /*__write_empty=*/false, __max_length, __n / __threshold);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
