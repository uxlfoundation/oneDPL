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

#include <algorithm>   // std::min, std::max, std::clamp
#include <cassert>     // assert
#include <cmath>       // std::sqrt
#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint8_t, std::uint32_t, std::uint64_t, std::uintptr_t
#include <functional>  // std::invoke, std::less, std::plus
#include <limits>      // std::numeric_limits
#include <type_traits> // std::integral_constant, std::conditional_t, std::enable_if_t, std::is_*_v, ...
#include <utility>     // std::move, std::forward, std::declval, std::pair

#include "sycl_defs.h"
#include "parallel_backend_sycl.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

#include "../../functional_impl.h" // oneapi::dpl::identity
#include "../../segmented_reduce_utils.h"
#include "../../utils.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// reduce_by_segment with segments defined by offsets

// One work-item per short segment; a long one only reports its length via atomic max, capped to 32 bits: the chunk
// length takes only its square root and clamps it far below
template <typename _Acc, typename _Size, typename _BinaryOperation>
struct __segmented_reduce_variable_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _Size __n = {};         // input size
    _Size __threshold = {}; // longest short segment length
    _Acc __init;            // init converted to _Acc
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

        // A reversed segment has a negative length, so it is short and its serial reduce gives the init
        static_assert(std::is_signed_v<_Size>, "The size type must be signed");
        if (__end - __begin > __threshold)
        {
            using __atomic_ref_t =
                sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>;

            __atomic_ref_t __ref(__max_length[0]);
            __ref.fetch_max(static_cast<std::uint32_t>(std::min<std::uint64_t>(
                static_cast<std::uint64_t>(__end - __begin), std::numeric_limits<std::uint32_t>::max())));
        }
        else
        {
            std::get<0>(__result[__idx]) =
                oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(__input, __begin, __end, __init, __binary_op);
        }
    }
};

template <typename _Name>
class __segmented_reduce_fill_wrapper;

// Short segment: one work-item; long: chunks per work-item, then combined by one work-item. Tuned on GPUs and CPUs
struct __segmented_reduce_chunk_tuning
{
    // A segment up to this length is short
    static constexpr std::size_t __gpu_short_segment_max = 512;
    static constexpr std::size_t __cpu_short_segment_max = 32768;

    // GPU: best chunk ~ sqrt(longest segment), balancing both levels, within these bounds; CPU: hardly matters >1024
    static constexpr std::size_t __gpu_min_chunk_size = 64;
    static constexpr std::size_t __gpu_max_chunk_size = 16384;
    static constexpr std::size_t __cpu_chunk_size = 4096;

    static std::size_t
    __short_segment_max(bool __is_gpu)
    {
        return __is_gpu ? __gpu_short_segment_max : __cpu_short_segment_max;
    }

    static std::size_t
    __chunk_size(bool __is_gpu, std::size_t __max_length)
    {
        if (__is_gpu)
        {
            const auto __root = static_cast<std::size_t>(std::sqrt(static_cast<double>(__max_length)));
            return std::clamp<std::size_t>(oneapi::dpl::__internal::__dpl_bit_ceil(__root), __gpu_min_chunk_size,
                                           __gpu_max_chunk_size);
        }

        return __cpu_chunk_size;
    }
};

// Segment length within the input; a negatively sized segment is empty
template <typename _Size>
struct __segmented_reduce_length
{
    _Size __n = {}; // input size

    template <typename _Offset>
    _Size
    operator()(_Offset __begin, _Offset __end) const
    {
        const _Size __b = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__begin, __n);
        const _Size __e = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__end, __n);

        static_assert(std::is_signed_v<_Size>, "The size type must be signed");
        return std::max<_Size>(0, __e - __b);
    }
};

// Chunks of a segment (0 if not longer than __min_length), scanned into chunk ends; __length - 1 >= 0 when used
template <typename _Size>
struct __segmented_reduce_chunk_count
{
    _Size __n = {};          // input size
    _Size __chunk_size = {}; // chunk length
    _Size __min_length = {}; // segments not longer than this have no chunks

    template <typename _Bounds>
    _Size
    operator()(const _Bounds& __bounds) const
    {
        const _Size __length = __segmented_reduce_length<_Size>{__n}(std::get<0>(__bounds), std::get<1>(__bounds));

        return __length > __min_length ? (__length - 1) / __chunk_size + 1 : _Size{0};
    }
};

// Reduces one chunk of a long segment. The first chunk of a segment takes the remainder of its length and starts
// from the initial value converted to the accumulator type; every other chunk has exactly __chunk_size elements and
// starts from the operation on its first two elements.
template <typename _Acc, typename _Size, typename _BinaryOperation>
struct __segmented_reduce_chunk_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

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
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");

        const _Size __chunk = __idx;
        if (__chunk >= __chunk_ends[__num_segments - 1])
            return;

        // The segment of the chunk: the first one whose chunks end after it, the last one if no other does
        const _Size __lo = oneapi::dpl::__internal::__pstl_upper_bound(__chunk_ends, _Size{0}, __num_segments - 1,
                                                                       &__chunk, std::less<>{}, oneapi::dpl::identity{},
                                                                       oneapi::dpl::identity{});
        const _Size __first_chunk = __lo == 0 ? _Size{0} : _Size(__chunk_ends[__lo - 1]);
        const _Size __begin = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__lo], __n);
        const _Size __end = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__lo + 1], __n);

        // First chunk length: the remainder after the full chunks, in [1, __chunk_size]
        const _Size __head = (__end - __begin) - (_Size(__chunk_ends[__lo]) - __first_chunk - 1) * __chunk_size;

        // One loop for every chunk, so sub-group work-items do not diverge into two long loops
        const _Size __k = __chunk - __first_chunk;
        const _Size __chunk_begin = __k == 0 ? __begin : __begin + __head + (__k - 1) * __chunk_size;
        const _Size __chunk_end = __k == 0 ? __begin + __head : __chunk_begin + __chunk_size;

        // The first chunk starts from __init; non-first chunks start from op(first two elements) as std::reduce does:
        // an accumulator is never made from a lone element, which may mean something else to the operation (such a
        // chunk has >= 64 elements)
        _Acc __start = __init;
        if (__k != 0)
            __start = std::invoke(__binary_op, __input[__chunk_begin], __input[__chunk_begin + 1]);

        std::get<0>(__partials[__idx]) = oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(
            __input, __chunk_begin + (__k == 0 ? 0 : 2), __chunk_end, std::move(__start), __binary_op);
    }
};

// A work-item per segment combines its chunk partials in chunk order
template <typename _Acc, typename _Size, typename _BinaryOperation>
struct __segmented_reduce_combine_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _Size __num_partials = {}; // number of computed partial results
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _ChunkEnds, typename _Partials, typename _Result>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _ChunkEnds __chunk_ends, _Partials __partials,
               _Result __result) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");

        const _Size __first = __idx == 0 ? _Size{0} : _Size(__chunk_ends[__idx - 1]);

        // No chunks: short and already reduced. Unordered offsets may exceed the computed chunks: results unspecified
        const _Size __last = std::min<_Size>(__chunk_ends[__idx], __num_partials);
        if (__first < __last)
            std::get<0>(__result[__idx]) = oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(
                __partials, __first + 1, __last, _Acc(__partials[__first]), __binary_op);
    }
};

// Whether a range is pointer-based, so consecutive elements can be loaded as one vector
template <typename _Range, typename = void>
struct __segmented_reduce_pointer_input : std::false_type
{
};
template <typename _Range>
struct __segmented_reduce_pointer_input<
    _Range, std::enable_if_t<std::is_pointer_v<decltype(std::declval<const _Range&>().begin())>>> : std::true_type
{
};

// The launch of the flat path
// Hardware threads of a device: the unit of the input thresholds of the GPU paths
inline std::size_t
__segmented_reduce_hw_threads(const oneapi::dpl::__internal::__device_compute_units& __compute_units)
{
    return std::max<std::size_t>(1, std::size_t{__compute_units.__max_compute_units} *
                                        __compute_units.__threads_per_compute_unit);
}

struct __segmented_reduce_flat_geometry
{
    std::size_t __work_group_size = {}; // multiple of the sub-group size
    std::size_t __num_groups = {};
    std::size_t __inputs_per_item = {};
};

// GPU, arithmetic types: each sub-group takes a piece of consecutive elements, finds its first segment by search and
// reduces the segments with coalesced loads, writing whole ones. The partial head/tail segments are combined by a
// second kernel; above a size, a work-group first combines its pieces in local memory. Tuned on GPUs. The search needs
// sorted offsets: with unsorted ones (results unspecified) a segment may be met by no piece and left unwritten.
struct __segmented_reduce_flat_tuning
{
    static constexpr std::size_t __sub_group_size = 32;

    // Min elements per work-item for small inputs to amortize the search and combination; the max (4x), which large
    // inputs take, so that they spread over more work-groups than the device runs at once
    static constexpr std::size_t __min_inputs_per_item = 16;
    static constexpr std::size_t __max_inputs_per_item = 4 * __min_inputs_per_item;

    // An input is small, or tiny, up to this many elements per hardware thread of the device; small and tiny inputs
    // take fewer elements per work-item, so that more sub-groups share them
    static constexpr std::size_t __small_inputs_per_thread = 1024;
    static constexpr std::size_t __tiny_inputs_per_thread = 256;
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

    // Whole segments up to this length are reduced one per lane: cheaper than sub-group combining and serial walking
    static constexpr std::size_t __tiny_segment = 32;

    // Min whole vectors for sub-group vector loads; shorter ranges go faster per element (alignment/head/tail cost)
    static constexpr std::size_t __min_vectors = 8;

    // Above this input, combine pieces in local memory first: per-piece stores and the second kernel cost more than
    // waiting for the slowest sub-group, which many work-groups hide
    static constexpr std::size_t __group_combine_input = std::size_t(1) << 24;

    // Launch for __n elements: a piece per HW thread, elements per work-item within the bounds, and work-groups halved
    // (down to one sub-group) until there are at least two per GPU core
    static __segmented_reduce_flat_geometry
    __geometry(const sycl::queue& __q, const std::size_t __n)
    {
        std::size_t __work_group_size = __max_work_group(__q);
        const oneapi::dpl::__internal::__device_compute_units __compute_units(__q);
        const std::size_t __min_groups = 2 * std::size_t{__compute_units.__num_xe_cores};
        const std::size_t __hw_threads = __segmented_reduce_hw_threads(__compute_units);

        const std::size_t __min_inputs = __n <= __tiny_inputs_per_thread * __hw_threads    ? __tiny_min_inputs_per_item
                                         : __n <= __small_inputs_per_thread * __hw_threads ? __small_min_inputs_per_item
                                                                                           : __min_inputs_per_item;
        const std::size_t __inputs_per_item =
            std::clamp<std::size_t>(oneapi::dpl::__internal::__dpl_ceiling_div(__n, __hw_threads * __sub_group_size),
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

// First-kernel local memory when combining pieces in a work-group: heads, tails, tail segments and head states
// (0 none, 1 segment ends in the piece, 2 goes on past it); empty otherwise
template <bool _InGroup, typename _Acc, typename _Size>
struct __segmented_reduce_flat_local
{
    __segmented_reduce_flat_local(sycl::handler&, std::size_t) {}
};

template <typename _Acc, typename _Size>
struct __segmented_reduce_flat_local<true, _Acc, _Size>
{
    __dpl_sycl::__local_accessor<_Acc> __loc_heads_tails_acc;
    __dpl_sycl::__local_accessor<_Size> __loc_tail_segments_acc;
    __dpl_sycl::__local_accessor<std::uint8_t> __loc_head_states_acc;

    __segmented_reduce_flat_local(sycl::handler& __cgh, std::size_t __sub_groups)
        : __loc_heads_tails_acc(sycl::range<1>(2 * __sub_groups), __cgh),
          __loc_tail_segments_acc(sycl::range<1>(__sub_groups), __cgh),
          __loc_head_states_acc(sycl::range<1>(__sub_groups), __cgh)
    {
    }
};

// Combines the first __active lanes of a sub-group onto lane 0: sub-group reduce if the op has a known identity (the
// other lanes give it), else a tree of compile-time shifts, as run-time shifts reach other lanes' registers indirectly.
// Every lane passes a constructed value; those of the other lanes are not used
template <typename _Acc, typename _SubGroup, typename _BinaryOperation>
_Acc
__segmented_reduce_lanes(const _SubGroup __sg, _Acc __acc, const std::uint32_t __active, _BinaryOperation __binary_op)
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
                    __acc = std::invoke(__binary_op, __acc, __other);
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
        const __segmented_reduce_flat_geometry __geometry = _Tuning::__geometry(__q, std::size_t(__n));
        const std::size_t __work_group_size = __geometry.__work_group_size;
        // Kernel values in 32 bits (counted in pieces and work-groups) so kernel arguments stay in registers
        const std::uint32_t __sub_groups = __work_group_size / __sub_group_size;
        const std::size_t __num_groups = __geometry.__num_groups;
        const std::size_t __ipi = __geometry.__inputs_per_item;
        const std::uint32_t __piece_size = __ipi * __sub_group_size;

        // Second-kernel units: work-groups when pieces are combined in local memory, else pieces
        const bool __in_group = std::size_t(__n) > _Tuning::__group_combine_input;
        const std::uint32_t __num_units = __in_group ? __num_groups : __num_groups * __sub_groups;
        const std::uint32_t __unit_size = __in_group ? __piece_size * __sub_groups : __piece_size;

        // Unit tail segments, then head and tail partials, in one storage (one call fewer, measured faster), typed by
        // the more aligned type so both arrays start aligned
        using _Unit = std::conditional_t<(alignof(_Acc) > alignof(_Size)), _Acc, _Size>;
        const std::uint32_t __segment_slots =
            oneapi::dpl::__internal::__dpl_ceiling_div(__num_units * sizeof(_Size), sizeof(_Unit));
        const std::size_t __partial_slots =
            oneapi::dpl::__internal::__dpl_ceiling_div(2 * __num_units * sizeof(_Acc), sizeof(_Unit));
        __device_storage<_Unit> __units_storage{__q, __segment_slots + __partial_slots};

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
                        const std::uint32_t __group = __item.get_group(0);
                        const std::uint32_t __piece = __group * std::uint32_t(__sub_groups) + __sg_id;
                        // In 64 bits: the product may exceed 32 bits before clamping to __n
                        const _Size __piece_begin = std::min<_Size>(__n, _Size(std::size_t(__piece) * __piece_size));
                        const _Size __piece_end = std::min<_Size>(__n, __piece_begin + _Size(__piece_size));

                        __piece_parts<_Size> __parts{__m};

                        if (__piece_begin < __piece_end)
                            __reduce_piece(__sg, __input, __offsets, __result, __n, __m, __piece_begin, __piece_end,
                                           __parts);

                        if constexpr (__in_group_v)
                            __combine_pieces_in_group(__item, __sg_id, __lane, __group, __sub_groups, __m, __parts,
                                                      __local, __result, __heads, __tails, __tail_segment_of);
                        else
                            __store_piece_parts(__lane, __piece, __m, __parts, __heads, __tails, __tail_segment_of);
                    });
            });
        };

        sycl::event __event = __in_group ? __pieces(/*__in_group_tag = */ std::true_type{})
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
    // Op on four values as two accumulator pairs; by value so all four load before the op waits on any
    template <typename _T>
    _Acc
    __reduce_four(_T __a, _T __b, _T __c, _T __d) const
    {
        return std::invoke(__binary_op, _Acc(std::invoke(__binary_op, __a, __b)),
                           _Acc(std::invoke(__binary_op, __c, __d)));
    }

    // Folds __get(__i) into __acc for __i from __i below __end by __stride: four values an op while they last
    template <typename _Get>
    void
    __fold_strided(_Acc& __acc, std::uint32_t __i, const std::uint32_t __end, const std::uint32_t __stride,
                   _Get __get) const
    {
        for (; __i + 3 * __stride < __end; __i += 4 * __stride)
            __acc = std::invoke(
                __binary_op, __acc,
                __reduce_four(__get(__i), __get(__i + __stride), __get(__i + 2 * __stride), __get(__i + 3 * __stride)));
        for (; __i < __end; __i += __stride)
            __acc = std::invoke(__binary_op, __acc, __get(__i));
    }

    // A piece's output on lane 0: head partial (state 0 none, 1 ends in the piece, 2 goes on past it) and tail partial
    // (segment begun in it going on past its end) with the tail's segment, __m for none
    template <typename _Size>
    struct __piece_parts
    {
        _Size __tail_segment = {};
        std::uint8_t __head_state = 0;
        oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __head, __tail;
    };

    // A very short segment held whole by the piece
    template <typename _Size>
    static bool
    __is_tiny(const _Size __b, const _Size __e, const _Size __piece_begin, const _Size __piece_end)
    {
        // A reversed segment in the piece has a negative length, so it is tiny
        return __b >= __piece_begin && __e <= __piece_end && __e - __b <= _Size(_Tuning::__tiny_segment);
    }

    // Element at an offset from the piece start, via a pointer where the input has one
    template <typename _InRange, typename _Size>
    static decltype(auto)
    __element(const _InRange __input, const _Size __piece_begin, const std::uint32_t __i)
    {
        if constexpr (__segmented_reduce_pointer_input<_InRange>::value)
            return (__input.begin() + __piece_begin)[__i];
        else
            return __input[__piece_begin + _Size(__i)];
    }

    // Sub-group reduces [__base + __first, __base + __last) (non-empty) onto lane 0; lanes interleave for coalescing,
    // reordering as commutativity allows. A segment start adds the init to the result. Arithmetic types; 32-bit offsets
    template <typename _SubGroup, typename _Size, typename _Input>
    _Acc
    __reduce_by_sub_group(const _SubGroup __sg, const _Size __base, const std::uint32_t __first,
                          const std::uint32_t __last, const bool __is_first, const _Input __input) const
    {
        const std::uint32_t __lane = __sg.get_local_linear_id();
        const std::uint32_t __sub_group_size = __sg.get_local_linear_range();

        // Accumulators start from their first values, at least two for each active lane: an accumulator never comes
        // from a lone element (may differ to the op); an idle lane holds the init, which is passed but never used
        std::uint32_t __active = std::min<std::uint32_t>(__sub_group_size, (__last - __first) / 2);
        oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __acc;

        // With a pointer input, each lane loads aligned vectors of four; the unaligned head and tail go to lane 0.
        // Pairs within a vector keep the accumulator from a lone element.
        bool __by_vectors = false;
        if constexpr (__segmented_reduce_pointer_input<_Input>::value)
        {
            using _T = std::remove_cv_t<std::remove_pointer_t<decltype(__input.begin())>>;
            if constexpr (std::is_arithmetic_v<_T> && !std::is_same_v<_T, bool> &&
                          (sizeof(_T) == 1 || sizeof(_T) == 2 || sizeof(_T) == 4 || sizeof(_T) == 8))
            {
                const _T* const __p = __input.begin() + __base;
                // Four elements a vector, two for 8-byte types, to stay within 16 bytes and be aligned as often
                constexpr std::uint32_t __w = std::min<std::uint32_t>(4, 16 / sizeof(_T));
                using _V = sycl::vec<_T, __w>;
                constexpr std::uint32_t __vb = sizeof(_V);
                const std::uint32_t __mis = std::uint32_t(reinterpret_cast<std::uintptr_t>(__p + __first) % __vb);
                // First vector-aligned index, at most __last
                const std::uint32_t __a = std::min<std::uint32_t>(__last, __first + (__vb - __mis) % __vb / sizeof(_T));
                const std::uint32_t __nv = (__last - __a) / __w;
                if (__nv >= _Tuning::__min_vectors)
                {
                    __by_vectors = true;
                    __active = std::min<std::uint32_t>(__sub_group_size, __nv);
                    auto __load4 = [&](std::uint32_t __k) -> _Acc {
                        _V __v;
                        __v.load(0, sycl::address_space_cast<sycl::access::address_space::generic_space,
                                                             sycl::access::decorated::no>(__p + __a + __w * __k));
                        if constexpr (__w == 2)
                            return _Acc(std::invoke(__binary_op, _T(__v[0]), _T(__v[1])));
                        else
                            return __reduce_four(_T(__v[0]), _T(__v[1]), _T(__v[2]), _T(__v[3]));
                    };
                    if (__lane < __active)
                    {
                        __acc.__setup(__load4(__lane));
                        __fold_strided(__acc.__v, __lane + __active, __nv, __active, __load4);
                    }

                    // Unaligned head and tail, at most three elements each (one for 8 bytes), to lane 0
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
            const std::uint32_t __i = __first + __lane;
            __acc.__setup(
                std::invoke(__binary_op, __element(__input, __base, __i), __element(__input, __base, __i + __active)));
            __fold_strided(__acc.__v, __i + 2 * __active, __last, __active,
                           [&](std::uint32_t __j) { return __element(__input, __base, __j); });
        }
        if (__lane >= __active)
            __acc.__setup(__init);

        if (__active > 0)
            __acc.__v = __segmented_reduce_lanes<_Acc>(__sg, __acc.__v, __active, __binary_op);

        // A range of one element (no active lane) is that element; a segment start adds the init
        const _Acc __total = __active > 0 ? __acc.__v : _Acc(__element(__input, __base, __first));
        return __is_first ? _Acc(std::invoke(__binary_op, __init, __total)) : __total;
    }

    // Run of very short segments from __s, a lane each, up to the first longer one: reduced from op(init, first),
    // empty and reversed ones get the init. Returns the run's segment count and its last segment's end.
    template <typename _SubGroup, typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size>
    std::pair<std::uint32_t, _Size>
    __reduce_tiny_run(const _SubGroup __sg, const _InRange __input, const _OffsetRange __offsets,
                      const _OutRange __result, const _Size __n, const _Size __m, const _Size __s, const _Size __b,
                      const _Size __piece_begin, const _Size __piece_end, const _Acc __init_acc) const
    {
        const std::uint32_t __lane = __sg.get_local_linear_id();
        const _Size __mine = __s + _Size(__lane);
        _Size __mb = __b, __me = __b;
        bool __run = false;
        if (__mine < __m)
        {
            using oneapi::dpl::__internal::__segmented_reduce_clamp_offset;

            __mb = __segmented_reduce_clamp_offset(__offsets[__mine], __n);
            __me = __segmented_reduce_clamp_offset(__offsets[__mine + 1], __n);
            __run = __is_tiny(__mb, __me, __piece_begin, __piece_end);
        }
        // Run length: the first lane not in the run, or the sub-group size if all are
        const std::uint32_t __count = __dpl_sycl::__reduce_over_group(
            __sg, __run ? std::uint32_t(_Tuning::__sub_group_size) : __lane, sycl::minimum<std::uint32_t>());

        if (__lane < __count && __mb >= __me)
            __result[__mine] = __init_acc;
        if (__lane < __count && __mb < __me)
        {
            // Within the piece, whose size fits in 32 bits
            const std::uint32_t __lo = std::uint32_t(__mb - __piece_begin);
            const std::uint32_t __hi = std::uint32_t(__me - __piece_begin);
            _Acc __acc = std::invoke(__binary_op, __init, __element(__input, __piece_begin, __lo));
            __fold_strided(__acc, __lo + 1, __hi, 1,
                           [&](std::uint32_t __j) { return __element(__input, __piece_begin, __j); });
            __result[__mine] = __acc;
        }

        return {__count, sycl::select_from_group(__sg, __me, __count - 1)};
    }

    // Part [__b, __e) of segment __s in the piece, by the sub-group: empty or whole segments are written; a part begun
    // before the piece is its head, one going on past its end its tail
    template <typename _SubGroup, typename _InRange, typename _OutRange, typename _Size>
    void
    __reduce_segment_part(const _SubGroup __sg, const _InRange __input, const _OutRange __result, const _Size __s,
                          const _Size __b, const _Size __e, const _Size __piece_begin, const _Size __piece_end,
                          const _Acc __init_acc, __piece_parts<_Size>& __parts) const
    {
        const std::uint32_t __lane = __sg.get_local_linear_id();
        const std::uint32_t __lo = __b > __piece_begin ? std::uint32_t(__b - __piece_begin) : 0;
        const std::uint32_t __hi =
            std::uint32_t(std::max<_Size>(__piece_begin, std::min<_Size>(__e, __piece_end)) - __piece_begin);

        if (__b == __e && __b >= __piece_begin && __lane == 0)
            __result[__s] = __init_acc;

        if (__lo < __hi)
        {
            const bool __first = __b >= __piece_begin;
            const _Acc __acc = __reduce_by_sub_group(__sg, __piece_begin, __lo, __hi, __first, __input);

            const bool __ends_in_piece = __e <= __piece_end;

            if (__first)
            {
                if (__ends_in_piece && __lane == 0)
                    __result[__s] = __acc;

                if (!__ends_in_piece)
                {
                    __parts.__tail.__setup(__acc);
                    __parts.__tail_segment = __s;
                }
            }
            else
            {
                __parts.__head.__setup(__acc);
                __parts.__head_state = __ends_in_piece ? 1 : 2;
            }
        }
    }

    // Walk the piece's segments, uniform across lanes, from the last beginning before it so leading empty ones are met;
    // the last piece also takes those at n
    template <typename _SubGroup, typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size>
    void
    __reduce_piece(const _SubGroup __sg, const _InRange __input, const _OffsetRange __offsets, const _OutRange __result,
                   const _Size __n, const _Size __m, const _Size __piece_begin, const _Size __piece_end,
                   __piece_parts<_Size>& __parts) const
    {
        const bool __last_piece = __piece_end == __n;
        using oneapi::dpl::__internal::__segmented_reduce_clamp_offset;

        // The last segment beginning before the piece, 0 for the first piece
        _Size __s =
            std::max<_Size>(0, oneapi::dpl::__internal::__pstl_lower_bound_impl(_Size{0}, __m, [&](const _Size __i) {
                                   return __segmented_reduce_clamp_offset(__offsets[__i], __n) < __piece_begin;
                               }) - 1);

        for (; __s < __m; ++__s)
        {
            const _Size __b = __segmented_reduce_clamp_offset(__offsets[__s], __n);
            if (__b >= __piece_end && !__last_piece)
                break;

            const _Size __e = __segmented_reduce_clamp_offset(__offsets[__s + 1], __n);
            if (__is_tiny(__b, __e, __piece_begin, __piece_end))
            {
                const auto [__count, __run_end] = __reduce_tiny_run(__sg, __input, __offsets, __result, __n, __m, __s,
                                                                    __b, __piece_begin, __piece_end, __init);
                // Continue after the run; its last segment may end the piece. __count >= 1, as segment __s is tiny
                __s += _Size(__count) - 1;
                if (__run_end >= __piece_end && !__last_piece)
                    break;
                continue;
            }

            __reduce_segment_part(__sg, __input, __result, __s, __b, __e, __piece_begin, __piece_end, __init, __parts);

            // A later segment begins at or after the end of this one
            if (__e >= __piece_end && !__last_piece)
                break;
        }
    }

    // Piece parts for the second kernel, from lane 0
    template <typename _Size>
    static void
    __store_piece_parts(const std::uint32_t __lane, const std::uint32_t __piece, const _Size __m,
                        const __piece_parts<_Size>& __parts, _Acc* const __heads, _Acc* const __tails,
                        _Size* const __tail_segment_of)
    {
        if (__lane == 0 && __parts.__head_state != 0)
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
                              const std::uint32_t __group, const std::size_t __sub_groups, const _Size __m,
                              const __piece_parts<_Size>& __parts, const _Local __local, const _OutRange __result,
                              _Acc* const __heads, _Acc* const __tails, _Size* const __tail_segment_of) const
    {
        if (__lane == 0)
        {
            if (__parts.__head_state != 0)
                __local.__loc_heads_tails_acc[__sg_id] = __parts.__head.__v;
            if (__parts.__tail_segment < __m)
                __local.__loc_heads_tails_acc[__sub_groups + __sg_id] = __parts.__tail.__v;
            __local.__loc_tail_segments_acc[__sg_id] = __parts.__tail_segment;
            __local.__loc_head_states_acc[__sg_id] = __parts.__head_state;
        }

        // No tail unless found below; the barrier orders this before the writes after it
        if (__sg_id == 0 && __lane == 0)
            __tail_segment_of[__group] = __m;

        sycl::group_barrier(__item.get_group());
        if (__sg_id != 0 || __lane >= __sub_groups)
            return;

        // Folds the heads of the pieces from __k on into __acc while the segment goes on (state 2); returns the state
        // it stops at, 1 if the segment ends in the work-group
        auto __chain = [&](_Acc& __acc, std::uint8_t __state, std::size_t __k) {
            for (; __state == 2 && __k < __sub_groups; ++__k)
            {
                __state = __local.__loc_head_states_acc[__k];
                if (__state != 0)
                    __acc = std::invoke(__binary_op, __acc, __local.__loc_heads_tails_acc[__k]);
            }
            return __state;
        };

        const _Size __s = __local.__loc_tail_segments_acc[__lane];
        if (__s < __m)
        {
            _Acc __acc = __local.__loc_heads_tails_acc[__sub_groups + __lane];
            const std::uint8_t __state = __chain(__acc, 2, __lane + 1);
            if (__state == 1)
                __result[__s] = __acc;
            if (__state != 1)
            {
                __tails[__group] = __acc;
                __tail_segment_of[__group] = __s;
            }
        }

        if (__lane == 0 && __local.__loc_head_states_acc[0] != 0)
        {
            _Acc __acc = __local.__loc_heads_tails_acc[0];
            __chain(__acc, __local.__loc_head_states_acc[0], 1);
            __heads[__group] = __acc;
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
        constexpr std::uint32_t __sub_group_size = _Tuning::__sub_group_size;
        const std::uint32_t __lane = __sg.get_local_linear_id();

        const _Size __s = __tail_segment_of[__unit];
        if (__s >= __m)
            return;

        // Tail loaded with its segment end so neither waits for the other
        // Tails follow the heads in the storage
        const _Acc __tail = (__heads + __num_units)[__unit];
        const _Size __e = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__s + 1], __n);
        // Units after this one up to the one holding the segment's last element
        const std::uint32_t __heads_count = std::uint32_t(std::size_t(__e - 1) / __unit_size) - __unit;
        if (__heads_count == 0 && __lane == 0)
            __result[__s] = __tail;
        if (__heads_count == 0)
            return;

        // Following heads: one per lane, then four per lane at a time so loads overlap; heads are already accumulators,
        // so a lane may start from one alone; an idle lane holds the tail, which is passed but never used
        const _Acc* const __first_head = __heads + (__unit + 1);
        oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __acc;
        if (__lane < __heads_count)
            __acc.__setup(__first_head[__lane]);
        if (__lane >= __heads_count)
            __acc.__setup(__tail);
        __fold_strided(__acc.__v, __lane + __sub_group_size, __heads_count, __sub_group_size,
                       [&](std::uint32_t __k) { return __first_head[__k]; });
        __acc.__v = __segmented_reduce_lanes<_Acc>(
            __sg, __acc.__v, std::min<std::uint32_t>(__heads_count, __sub_group_size), __binary_op);
        if (__lane == 0)
            __result[__s] = _Acc(std::invoke(__binary_op, __tail, __acc.__v));
    }
};

// GPU, arithmetic types, small input: one kernel, a work-group per segment. Empty, out-of-input or reversed segments
// get the init; otherwise work-items stride by the work-group, sub-groups combine, then sub-group 0 combines them via
// local memory. Tuned on GPUs.
struct __segmented_reduce_group_tuning
{
    // As on the flat path, whose sub-group size the dispatch checks for both
    static constexpr std::size_t __sub_group_size = __segmented_reduce_flat_tuning::__sub_group_size;

    // Up to this many elements and segments, a work-group per segment beats the flat path's second kernel for any
    // lengths; above, a few long segments leave the device idle. Positions within a segment fit in 32 bits.
    static constexpr std::size_t __max_input = std::size_t(1) << 12;
    static_assert(__max_input <= std::numeric_limits<std::uint32_t>::max());

    // With at least this many segments, a work-group per segment still wins up to this many elements and segments per
    // hardware thread of the device: the segments keep enough work-groups busy
    static constexpr std::size_t __min_segments_beyond = 16;
    static constexpr std::size_t __max_inputs_per_thread = 4;

    static bool
    __applies(const sycl::queue& __q, const std::size_t __n, const std::size_t __m)
    {
        if (__n <= __max_input && __m <= __max_input)
            return true;
        if (__m < __min_segments_beyond)
            return false;

        const std::size_t __hw_threads =
            __segmented_reduce_hw_threads(oneapi::dpl::__internal::__device_compute_units(__q));
        const std::size_t __max_input_beyond =
            std::min<std::size_t>(std::numeric_limits<std::uint32_t>::max(), __max_inputs_per_thread * __hw_threads);
        return __n <= __max_input_beyond && __m <= __max_input_beyond;
    }

    // Power-of-two sub-groups, about the average segment length in work-items, within the bounds; beyond small inputs
    // the largest, as a segment may then be much longer than the average
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
        constexpr std::uint32_t __sub_group_size = _Tuning::__sub_group_size;
        const std::uint32_t __work_group_size = _Tuning::__work_group_size(__q, std::size_t(__n), std::size_t(__m));
        const std::uint32_t __sub_groups = __work_group_size / __sub_group_size;

        sycl::event __event = __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
            __dpl_sycl::__local_accessor<_Acc> __loc_partials_acc(sycl::range<1>(__sub_groups), __cgh);

            __cgh.parallel_for<_Name...>(
                sycl::nd_range<1>(sycl::range<1>(std::size_t(__m) * __work_group_size),
                                  sycl::range<1>(__work_group_size)),
                [=, *this](sycl::nd_item<1> __item) [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(__sub_group_size)]] {
                    const std::uint32_t __segment = __item.get_group(0);
                    const std::uint32_t __lid = __item.get_local_id(0);
                    using oneapi::dpl::__internal::__segmented_reduce_clamp_offset;

                    const _Size __begin = __segmented_reduce_clamp_offset(__offsets[__segment], __n);
                    const _Size __end = __segmented_reduce_clamp_offset(__offsets[__segment + 1], __n);

                    // Uniform across the work-group, so it leaves before any barrier
                    if (__begin >= __end)
                    {
                        if (__lid == 0)
                            __result[__segment] = __init;
                        return;
                    }

                    // Work-item 0 starts from op(init, first); one past the segment length starts from the first
                    // element, skipped by the combinations below
                    const std::uint32_t __length = __end - __begin;
                    const std::uint32_t __first = __lid < __length ? __lid : 0;
                    _Acc __acc = __lid == 0 ? _Acc(std::invoke(__binary_op, __init, __input[__begin]))
                                            : _Acc(__input[__begin + __first]);
                    for (std::uint32_t __i = __lid + __work_group_size; __i < __length; __i += __work_group_size)
                        __acc = std::invoke(__binary_op, __acc, __input[__begin + __i]);

                    const auto __sg = __item.get_sub_group();
                    const std::uint32_t __sg_id = __sg.get_group_linear_id();
                    const std::uint32_t __items = std::min(__length, __work_group_size);
                    const std::uint32_t __sg_first = __sg_id * __sub_group_size;
                    const std::uint32_t __active =
                        __items > __sg_first ? std::min(__sub_group_size, __items - __sg_first) : 0;
                    __acc = __segmented_reduce_lanes<_Acc>(__sg, __acc, __active, __binary_op);

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
                            __acc = __segmented_reduce_lanes<_Acc>(
                                __sg, __loc_partials_acc[__lane < __active_sub_groups ? __lane : 0],
                                __active_sub_groups, __binary_op);
                        }
                    }

                    if (__lid == 0)
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

// Writes the init into every result: the input is empty, or the offsets give no element to any segment
template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Size, typename _Acc>
void
__parallel_reduce_by_segment_offsets_fill(_BackendTag, _ExecutionPolicy&& __exec, _Range&& __result, _Size __m,
                                          const _Acc& __init)
{
    using _Fill = oneapi::dpl::__internal::__brick_fill<oneapi::dpl::__internal::__hetero_tag<_BackendTag>, _Acc>;
    // Separate kernel name, as the same call also instantiates the reduction kernel
    __parallel_for(_BackendTag{},
                   make_wrapped_policy<__segmented_reduce_fill_wrapper>(std::forward<_ExecutionPolicy>(__exec)),
                   unseq_backend::walk_n_vectors_or_scalars<_Fill>{_Fill{__init}, std::size_t(__m)}, __m,
                   std::forward<_Range>(__result))
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

        // No segments and an empty input are handled by the caller, __pattern_reduce_by_segment_offsets; the divisions
        // below need both sizes positive
        assert(__m > 0 && __n > 0);

        sycl::queue __q = __exec.queue();
        const bool __is_gpu = __q.get_device().is_gpu();

        if constexpr (std::is_arithmetic_v<_Acc> &&
                      std::is_arithmetic_v<oneapi::dpl::__internal::__value_t<std::decay_t<_InRange>>>)
        {
            if (__is_gpu && oneapi::dpl::__internal::__supports_sub_group_size(__q, _FlatTuning::__sub_group_size))
            {
                using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

                if (_GroupTuning::__applies(__q, std::size_t(__n), std::size_t(__m)))
                {
                    // Small input: a work-group per segment
                    using _GroupKernel =
                        __internal::__kernel_name_provider<__segmented_reduce_group_kernel<_CustomName>>;
                    __segmented_reduce_group_submitter<_Acc, _GroupKernel, _BinaryOperation, _GroupTuning>{
                        __init, __binary_op}(__q, __input, __offsets, __result, __n, __m);
                }
                else
                {
                    // Pieces of the input independent of the segments
                    using _FlatKernel = __internal::__kernel_name_provider<__segmented_reduce_flat_kernel<_CustomName>>;
                    using _FlatGroupKernel =
                        __internal::__kernel_name_provider<__segmented_reduce_flat_group_kernel<_CustomName>>;
                    using _FlatTailKernel =
                        __internal::__kernel_name_provider<__segmented_reduce_flat_tail_kernel<_CustomName>>;
                    __segmented_reduce_flat_submitter<_Acc, _FlatKernel, _FlatGroupKernel, _FlatTailKernel,
                                                      _BinaryOperation, _FlatTuning>{__init, __binary_op}(
                        __q, __input, __offsets, __result, __n, __m);
                }
                return;
            }
        }

        // Short segments by a work-item each, which also writes the empty ones; the longer ones in chunks
        const _Size __threshold = _ChunkTuning::__short_segment_max(__is_gpu);
        const _Size __max_length = __reduce_short_segments(_BackendTag{}, __exec, __input, __offsets, __result, __n,
                                                           __m, __threshold, __init, __binary_op);
        // At most __n / __threshold segments are longer than __threshold
        if (__max_length > 0)
            __reduce_in_chunks(_BackendTag{}, __exec, __is_gpu, __input, __offsets, __result, __n, __m, __init,
                               __binary_op, __threshold, __max_length, __n / __threshold);
    }

  private:
    // Short segments, a work-item each; returns the longest long-segment length (0 if none) via a buffer, cheaper on
    // GPUs than a copy from device USM
    template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange,
              typename _OutRange, typename _Size, typename _BinaryOperation>
    static _Size
    __reduce_short_segments(_BackendTag, _ExecutionPolicy& __exec, _InRange& __input, _OffsetRange& __offsets,
                            _OutRange& __result, const _Size __n, const _Size __m, const _Size __threshold,
                            const _Acc& __init, const _BinaryOperation& __binary_op)
    {
        auto __report_buf = __buffer<std::uint32_t>(1).get_buffer();
        // No host data: the atomic maximum starts from this zero
        __report_buf.get_host_access(sycl::write_only)[0] = 0;
        __parallel_for(
            _BackendTag{}, __exec,
            __segmented_reduce_variable_brick<_Acc, _Size, _BinaryOperation>{__n, __threshold, __init, __binary_op},
            // Zip view: a tuple value type keeps __parallel_for from vectorizing (__vector_size == 1);
            // with __iters_per_item == 1, one segment per item
            __m, __offsets, oneapi::dpl::__ranges::make_zip_view(__result), __input,
            oneapi::dpl::__ranges::views::all(__report_buf))
            .event()
            .wait_and_throw();

        return __report_buf.get_host_access(sycl::read_only)[0];
    }

    // Reduces segments longer than __min_length in chunks; __max_segments bounds their number
    template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange,
              typename _OutRange, typename _Size, typename _BinaryOperation>
    static void
    __reduce_in_chunks(_BackendTag, _ExecutionPolicy& __exec, const bool __is_gpu, _InRange& __input,
                       _OffsetRange& __offsets, _OutRange& __result, const _Size __n, const _Size __m,
                       const _Acc& __init, const _BinaryOperation& __binary_op, const _Size __min_length,
                       const _Size __max_length, const _Size __max_segments)
    {
        const _Size __chunk_size = _ChunkTuning::__chunk_size(__is_gpu, __max_length);

        // Chunk ends per segment: inclusive scan of chunk counts from adjacent offsets. Its future keeps the scan's
        // temporaries alive until combining ends, which waits for all three steps: each starts on the device after
        // the previous one. The temporaries are device USM or, without it, SYCL buffers
        sycl::queue __q = __exec.queue();
        __block_storage<_Size, 0> __chunk_ends{__q, std::size_t(__m)};
        auto __scan = __parallel_transform_scan(
            _BackendTag{}, make_wrapped_policy<__segmented_reduce_scan_wrapper>(__exec),
            oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::take_view_simple(__offsets, __m),
                                                 oneapi::dpl::__ranges::drop_view_simple(__offsets, _Size{1})),
            __chunk_ends.__all_view(), __m, __segmented_reduce_chunk_count<_Size>{__n, __chunk_size, __min_length},
            unseq_backend::__no_init_value<_Size>{}, std::plus<_Size>{}, /*_Inclusive=*/std::true_type{});

        // At most n / __chunk_size + 1 chunks per segment, unless the offsets are not ordered
        const _Size __num_partials = __n / __chunk_size + __max_segments + 1;
        __block_storage<_Acc, 0> __partials{__q, std::size_t(__num_partials)};

        auto __chunks = __parallel_for(
            _BackendTag{}, make_wrapped_policy<__segmented_reduce_chunk_wrapper>(__exec), __scan.event(),
            __segmented_reduce_chunk_brick<_Acc, _Size, _BinaryOperation>{__n, __m, __chunk_size, __init, __binary_op},
            __num_partials, __chunk_ends.__all_view(), __offsets,
            // Zip view: a tuple value type keeps __parallel_for from vectorizing (__vector_size == 1);
            // with __iters_per_item == 1, one chunk per item
            oneapi::dpl::__ranges::make_zip_view(__partials.__all_view()), __input);

        __parallel_for(_BackendTag{}, make_wrapped_policy<__segmented_reduce_combine_wrapper>(__exec), __chunks.event(),
                       __segmented_reduce_combine_brick<_Acc, _Size, _BinaryOperation>{__num_partials, __binary_op},
                       __m, __chunk_ends.__all_view(), __partials.__all_view(),
                       // Zip view: a tuple value type keeps __parallel_for from vectorizing (__vector_size == 1);
                       // with __iters_per_item == 1, one segment per item
                       oneapi::dpl::__ranges::make_zip_view(__result))
            .event()
            .wait_and_throw();
    }
};

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
