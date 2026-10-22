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
#include <cstdlib>
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

// The register file size of a kernel on Intel GPUs, where the implementation of SYCL provides it
#if __has_include(<sycl/ext/intel/experimental/grf_size_properties.hpp>)
#    include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#endif

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
        get<0>(__result[__idx]) = oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init);
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
// - on GPUs an input up to this size whose average segment is shorter than this is reduced by one kernel, one
//   work-item per segment: a work-group per segment costs more for such segments, though less for a longer one;
inline constexpr std::size_t __segmented_reduce_gpu_small_input = 1024;
inline constexpr std::size_t __segmented_reduce_gpu_small_average = 16;
// - on GPUs an input up to this size is otherwise reduced by one kernel, one work-group per segment, for at most this
//   many segments or with an average segment of at least this length: the work-groups of more and shorter segments
//   cost more than the other paths;
inline constexpr std::size_t __segmented_reduce_gpu_group_input = 32768;
inline constexpr std::size_t __segmented_reduce_gpu_group_max_segments = 1024;
inline constexpr std::size_t __segmented_reduce_gpu_group_min_average = 24;
// - on GPUs a larger input of at least this many segments is reduced one work-group per segment as well when its
//   average segment is shorter than this, as the chunks reduce longer ones faster, and fewer segments leave too few
//   work-groups. Up to this size of the input by one kernel, as no segment is longer then, with an average segment of
//   at least the least one above. A larger input only with an average segment of at least this length, as the
//   work-groups of more and shorter segments cost more than the chunks on some GPUs: its segments longer than the size
//   above are left to a second kernel, one work-group per tile of the input, of a power of two of at least this length
//   that is close to the input size per compute unit, and at most that size;
inline constexpr std::size_t __segmented_reduce_gpu_large_group_min_segments = 100;
inline constexpr std::size_t __segmented_reduce_gpu_large_group_max_average = 16384;
inline constexpr std::size_t __segmented_reduce_gpu_large_group_max_length = 65536;
inline constexpr std::size_t __segmented_reduce_gpu_large_group_long_average = 2048;
inline constexpr std::size_t __segmented_reduce_gpu_min_long_tile = 8192;
// - on GPUs the best chunk length of a long segment is close to the square root of its length, which balances the
//   serial work of its chunks and of their combination; on CPUs the chunk length hardly matters above 1024.
inline constexpr std::size_t __segmented_reduce_gpu_min_chunk_size = 64;
inline constexpr std::size_t __segmented_reduce_gpu_max_chunk_size = 16384;
inline constexpr std::size_t __segmented_reduce_cpu_chunk_size = 4096;
// - on GPUs the chunks of arithmetic types are reduced by sub-groups on devices with sub-groups of this size, each
//   reducing a chunk about this many times faster: the chunk length is close to the square root of the length times
//   the sub-group size, with at least 32 elements per work-item, as the search for the segment of a chunk costs more
//   than a few elements. The kernels do not require this size: they take the size of their sub-groups, which the
//   compiler chooses, and are correct with any.
inline constexpr std::uint8_t __segmented_reduce_gpu_sub_group_size = 32;
inline constexpr std::size_t __segmented_reduce_gpu_min_group_chunk_size = 1024;
inline constexpr std::size_t __segmented_reduce_gpu_max_group_chunk_size = 65536;
// - only when the typical segment in chunks, the average one if all the segments are and the longest one otherwise, is
//   at least this long: shorter segments leave a sub-group too few elements per work-item.
inline constexpr std::size_t __segmented_reduce_gpu_min_group_length = 1024;
// - on GPUs the ends of the chunks of at most this many segments are computed by every work-group of the kernel that
//   reduces the chunks, in its local memory, instead of by a scan before that kernel: the first for the chunks by
//   sub-groups, the second for the chunks by work-items, for which the local ends pay off for more segments;
inline constexpr std::size_t __segmented_reduce_gpu_local_ends_segments = 1024;
inline constexpr std::size_t __segmented_reduce_gpu_local_ends_item_segments = 2048;
// - on GPUs an input of arithmetic types whose average segment has at least this length is reduced by tiles, one
//   work-group per tile, of the smaller length up to the input size below and of the larger one above.
inline constexpr std::size_t __segmented_reduce_gpu_tiles_average = 32768;
inline constexpr std::size_t __segmented_reduce_gpu_small_tile = 8192;
inline constexpr std::size_t __segmented_reduce_gpu_large_tile = 16384;
inline constexpr std::size_t __segmented_reduce_gpu_small_tile_input = std::size_t(1) << 20;
// - the partial results of the segments crossing the ends of the tiles are combined one sub-group per tile, whose lanes
//   take the carries of the following tiles in turn, when the average segment spans at least this many tiles: one
//   work-item per tile follows shorter chains faster.
inline constexpr std::size_t __segmented_reduce_gpu_group_fixup_tiles = 16;

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
        _Acc __start = __is_first ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                        std::invoke(__binary_op, __init, __input[__chunk_begin]))
                                  : oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                        std::invoke(__binary_op, __input[__chunk_begin], __input[__chunk_begin + 1]));
        std::get<0>(__partials[__idx]) = oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(
            __input, __chunk_begin + (__is_first ? 1 : 2), __chunk_end, std::move(__start), __binary_op);
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
            _Size __active = std::min<_Size>(__lanes, (__last - __first) / 2);
            _Acc __acc{};
            // Where the input is reached through a pointer, every lane loads four consecutive elements at once, from the
            // first address aligned for such a vector on; the elements before it and after the last whole vector go to
            // lane 0. The pairs of a vector keep the accumulator from being made from an element alone.
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
                    const _Size __a =
                        std::min<_Size>(__last, __first + _Size(__mis == 0 ? 0 : (__vb - __mis) / sizeof(_T)));
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
                                    std::invoke(
                                        __binary_op,
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
                    __acc = std::invoke(
                        __binary_op, __acc,
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

            if (__lane == 0)
            {
                // A first chunk of one element more than the initial value takes it alone
                if (__is_first)
                    __acc = __active > 0       ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                                     std::invoke(__binary_op, __head, __acc))
                            : __first < __last ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                                     std::invoke(__binary_op, __head, __input[__first]))
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
                [=](sycl::nd_item<1> __item) { __brick(__item, __rngs...); });
        });
    }
};

template <typename... _Name>
class __segmented_reduce_chunk_local_kernel;
template <typename... _Name>
class __segmented_reduce_chunk_group_local_kernel;

template <bool _ByGroups, typename _KernelName>
struct __segmented_reduce_chunk_local_submitter;

// Reduces the chunks of the long segments of a few segments in one kernel: every work-group first computes the ends of
// the chunks of every segment, an inclusive scan of their numbers, in its local memory, and work-group 0 also writes
// them for the combination. Then every work-item, or every sub-group for _ByGroups, reduces every so-many-th chunk with
// the brick, as the kernels after a scan do.
template <bool _ByGroups, typename... _Name>
struct __segmented_reduce_chunk_local_submitter<_ByGroups, __internal::__optional_kernel_name<_Name...>>
{
    // The work of a work-item; a function rather than a lambda called by the kernel, which would have to be device
    // copyable
    template <typename _Size, typename _Brick, typename _LocalEnds, typename _ChunkEnds, typename _Offsets,
              typename _Partials, typename _Input>
    static void
    __run(const sycl::nd_item<1>& __item, const _Brick& __brick, _LocalEnds __local_ends, _ChunkEnds __chunk_ends,
          _Offsets __offsets, _Partials __partials, _Input __input, const _Size __n, const _Size __m,
          const _Size __min_length, const _Size __num_partials)
    {
        const auto __group = __item.get_group();
        const _Size __lid = __item.get_local_id(0);
        const _Size __wg = __item.get_local_range(0);
        const __segmented_reduce_chunk_count<_Size> __count_of{__n, __min_length, /*__is_gpu=*/true, _ByGroups};
        std::uint32_t __carry = 0;
        for (_Size __first = 0; __first < __m; __first += __wg)
        {
            const _Size __s = __first + __lid;
            const std::uint32_t __count =
                __s < __m ? std::uint32_t(__count_of(std::make_tuple(__offsets[__s], __offsets[__s + 1]))) : 0;
            const std::uint32_t __end =
                __carry + __dpl_sycl::__inclusive_scan_over_group(__group, __count, sycl::plus<std::uint32_t>());
            if (__s < __m)
            {
                __local_ends[__s] = __end;
                if (__item.get_group(0) == 0)
                    __chunk_ends[__s] = _Size(__end);
            }
            __carry = __dpl_sycl::__group_broadcast(__group, __end, __wg - 1);
        }
        sycl::group_barrier(__group);
        if constexpr (_ByGroups)
            __brick(__item, __local_ends, __offsets, __partials, __input);
        else
        {
            const std::size_t __num_chunks = std::min<_Size>(_Size(__local_ends[__m - 1]), __num_partials);
            for (std::size_t __idx = __item.get_global_linear_id(); __idx < __num_chunks;
                 __idx += __item.get_global_range(0))
                __brick(std::true_type{}, __idx, __pfor_params_simple{}, __local_ends, __offsets, __partials, __input);
        }
    }

    template <typename _Size, typename _Acc, typename _Brick, typename _Offsets, typename _Input>
    sycl::event
    operator()(sycl::queue& __q, const _Brick& __brick, const std::size_t __num_groups,
               const std::size_t __work_group_size, const _Size __n, const _Size __m, const _Size __min_length,
               const _Size __num_partials, __device_storage<_Size>& __chunk_ends, __device_storage<_Acc>& __partials,
               _Offsets&& __offsets, _Input&& __input) const
    {
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __offsets, __input);
            auto __ends_acc = __get_accessor(sycl::write_only, __chunk_ends, __cgh, __dpl_sycl::__no_init{});
            auto __partials_acc = __get_accessor(sycl::write_only, __partials, __cgh, __dpl_sycl::__no_init{});
            // The ends fit 32 bits: there are fewer chunks than partial results, unless the offsets are not ordered,
            // whose results are unspecified
            __dpl_sycl::__local_accessor<std::uint32_t> __local_ends(sycl::range<1>(__m), __cgh);
            const sycl::nd_range<1> __range(sycl::range<1>(__num_groups * __work_group_size),
                                            sycl::range<1>(__work_group_size));
            if constexpr (_ByGroups)
                __cgh.parallel_for<_Name...>(__range, [=](sycl::nd_item<1> __item) {
                    __run(__item, __brick, __local_ends, __ends_acc.__data(), __offsets,
                          oneapi::dpl::__ranges::make_zip_view(
                              oneapi::dpl::__ranges::guard_view<_Acc*>(__partials_acc.__data(), __num_partials)),
                          __input, __n, __m, __min_length, __num_partials);
                });
            else
                __cgh.parallel_for<_Name...>(__range, [=](sycl::nd_item<1> __item) {
                    __run(__item, __brick, __local_ends, __ends_acc.__data(), __offsets,
                          oneapi::dpl::__ranges::make_zip_view(
                              oneapi::dpl::__ranges::guard_view<_Acc*>(__partials_acc.__data(), __num_partials)),
                          __input, __n, __m, __min_length, __num_partials);
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
                std::get<0>(__result[__idx]) = oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init);
            return;
        }
        std::get<0>(__result[__idx]) = oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(
            __partials, __first + 1, __last, _Acc(__partials[__first]), __binary_op);
    }
};

template <typename... _Name>
class __segmented_reduce_combine_kernel;

template <typename _KernelName>
struct __segmented_reduce_combine_submitter;

// Combines the partial results of the chunks of every segment, one work-item per segment, after the kernel that
// computed them into the storage
template <typename... _Name>
struct __segmented_reduce_combine_submitter<__internal::__optional_kernel_name<_Name...>>
{
    template <typename _Brick, typename _Size, typename _Acc, typename _Result>
    sycl::event
    operator()(sycl::queue& __q, const _Brick& __brick, const sycl::event& __chunk_event, const _Size __m,
               __device_storage<_Size>& __chunk_ends, __device_storage<_Acc>& __partials, _Result&& __result) const
    {
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __result);
            auto __ends_acc = __get_accessor(sycl::read_only, __chunk_ends, __cgh);
            auto __partials_acc = __get_accessor(sycl::read_only, __partials, __cgh);
            __cgh.depends_on(__chunk_event);
            auto __result_rng = oneapi::dpl::__ranges::make_zip_view(__result);
            __cgh.parallel_for<_Name...>(sycl::range<1>(__m), [=](sycl::item<1> __it) {
                __brick(std::true_type{}, __it.get_linear_id(), __pfor_params_simple{}, __ends_acc.__data(),
                        __partials_acc.__data(), __result_rng);
            });
        });
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

// One work-group reduces one segment, as with __work_group_reduce_kernel of the reduction, whose bricks
// (unseq_backend::transform_reduce and unseq_backend::reduce_over_group) it uses: the operation is required to be
// commutative, so the work-items load the segment coalesced. The bricks address the input by the global position of
// a work-item; the work-group of a segment is presented to them as the only one of the kernel, and its segment as the
// whole input.
class __segmented_reduce_group_item
{
    sycl::nd_item<1> __item;

  public:
    explicit __segmented_reduce_group_item(const sycl::nd_item<1>& __nd_item) : __item(__nd_item) {}

    std::size_t
    get_global_id(int) const
    {
        return __item.get_local_id(0);
    }
    std::size_t
    get_local_id(int) const
    {
        return __item.get_local_id(0);
    }
    std::size_t
    get_local_range(int) const
    {
        return __item.get_local_range(0);
    }
    sycl::range<1>
    get_local_range() const
    {
        return __item.get_local_range();
    }
    std::size_t
    get_group(int) const
    {
        return 0;
    }
    std::size_t
    get_group_linear_id() const
    {
        return 0;
    }
    sycl::group<1>
    get_group() const
    {
        return __item.get_group();
    }
};

// The elements of a range of at least two elements, taken in pairs by the bricks of the reduction. In the first range
// of a segment the first pair starts from the operation on the initial value and the first element, as in
// std::ranges::fold_left; every other pair from the operation on its two elements, as a partial result of std::reduce
// does, so that the accumulator is never made from an element alone, which may mean something else than the element
// to the operation. With an odd length the last pair also takes the last element.
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_pairs
{
    _Size __begin;           // the first element of the range
    std::size_t __num_pairs; // the number of pairs
    bool __is_odd;           // whether the length of the range is odd
    bool __with_init;        // whether the range begins its segment, so that its first pair takes the initial value
    _InitType __init;
    _BinaryOperation __binary_op;

    template <typename _Input>
    _Acc
    operator()(const std::size_t __idx, const _Input& __input) const
    {
        const _Size __first = __begin + 2 * _Size(__idx);
        const bool __head = __with_init && __idx == 0;
        _Acc __acc = __head ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                  std::invoke(__binary_op, __init, __input[__first]))
                            : oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                  std::invoke(__binary_op, __input[__first], __input[__first + 1]));
        if (__head)
            __acc = std::invoke(__binary_op, std::move(__acc), __input[__first + 1]);
        if (__is_odd && __idx + 1 == __num_pairs)
            __acc = std::invoke(__binary_op, std::move(__acc), __input[__first + 2]);
        return __acc;
    }
};

// The operation for the bricks of the reduction within a work-group, which invoke it as a const object. An operation
// that the reduction recognizes by its type to use the group algorithm, whose call operator is const, is kept as it is,
// since it would not be recognized once wrapped; another one, whose call operator may be non-const, is invoked
// through a mutable copy.
template <typename _Acc, typename _BinaryOperation>
auto
__segmented_reduce_group_op(_BinaryOperation __binary_op)
{
    if constexpr (unseq_backend::__can_use_group_reduce_scan<_BinaryOperation, _Acc>::value)
        return __binary_op;
    else
        return oneapi::dpl::__internal::__binary_op<_BinaryOperation, oneapi::dpl::identity, oneapi::dpl::identity>{
            std::move(__binary_op), {}, {}};
}

// Reduces the range [__begin, __begin + __length) by the work-group of __nd_item, as __work_group_reduce_kernel of the
// reduction does with its bricks, and passes the result to __write on work-item 0. A range shorter than two elements
// begins its segment: it gets the initial value converted to the accumulator type, or the operation on it and its
// element.
template <typename _Acc, std::uint8_t __vector_size, typename _Op, typename _Size, typename _InitType, typename _Input,
          typename _LocalMem, typename _Write>
void
__segmented_reduce_by_work_group(const sycl::nd_item<1>& __nd_item, const _Op& __op, const _Size __begin,
                                 const _Size __length, const bool __with_init, const _InitType& __init,
                                 const _Input& __input, const _LocalMem& __local_mem, _Write __write)
{
    using _Pairs = __segmented_reduce_pairs<_Acc, _Size, _InitType, _Op>;
    using _TransformPattern =
        unseq_backend::transform_reduce<_Op, _Pairs, _Acc, /*_Commutative=*/std::true_type, __vector_size>;
    using _ReducePattern = unseq_backend::reduce_over_group<_Op, _Acc>;

    const std::size_t __local_idx = __nd_item.get_local_id(0);
    // The same for every work-item of the work-group, so none of them reaches a barrier below
    if (__length < 2)
    {
        if (__local_idx == 0)
            __write(__length == 0 ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init)
                                  : oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                        std::invoke(__op, __init, __input[__begin])));
        return;
    }

    // The bricks of the reduction take unsigned sizes
    const std::size_t __work_group_size = __nd_item.get_local_range(0);
    const std::size_t __num_pairs = __length / 2;
    const __segmented_reduce_group_item __item(__nd_item);
    const _TransformPattern __transform_pattern{
        __op, _Pairs{__begin, __num_pairs, __length % 2 != 0, __with_init, __init, __op}};
    const _ReducePattern __reduce_pattern{__op};

    // The iterations of a work-item as in __parallel_transform_reduce
    const std::size_t __iters_per_work_item = __adjust_iters_per_work_item<__vector_size>(
        oneapi::dpl::__internal::__dpl_ceiling_div(__num_pairs, __work_group_size));
    const bool __is_full = __num_pairs == __work_group_size * __iters_per_work_item;

    // 1. The partial results of the work-items
    oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __partial;
    __transform_pattern(__item, __num_pairs, __iters_per_work_item, /*__global_offset*/ std::size_t{0}, __is_full,
                        /*__n_groups*/ std::size_t{1}, __partial, __input);
    const std::size_t __n_items =
        __transform_pattern.output_size(__num_pairs, __work_group_size, __iters_per_work_item);

    // 2. Their reduction within the work-group
    _Acc __reduced = __reduce_pattern(__item, __n_items, __partial, __local_mem);
    if (__local_idx < __n_items)
    {
        if (__local_idx == 0)
            __write(std::move(__reduced));
        __partial.__destroy();
    }
}

template <typename... _Name>
class __segmented_reduce_group_kernel;

template <typename _Acc, typename _KernelName>
struct __segmented_reduce_group_submitter;

template <typename _Acc, typename... _Name>
struct __segmented_reduce_group_submitter<_Acc, __internal::__optional_kernel_name<_Name...>>
{
    // As in __parallel_transform_reduce
    static constexpr std::uint8_t __vector_size = 4;

    template <typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size, typename _InitType,
              typename _BinaryOperation>
    sycl::event
    operator()(sycl::queue& __q, _InRange&& __input, _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n,
               const _Size __m, const std::uint16_t __work_group_size, _InitType __init,
               _BinaryOperation __binary_op) const
    {
        auto __op = __segmented_reduce_group_op<_Acc>(std::move(__binary_op));
        using _Op = decltype(__op);
        using _ReducePattern = unseq_backend::reduce_over_group<_Op, _Acc>;

        const std::size_t __local_mem_size = _ReducePattern{__op}.local_mem_req(__work_group_size);
        // One work-group per segment. A launch that would not fit the int range of the work-item indices, which a SYCL
        // implementation may assume (as DPC++ does by default), has fewer work-groups, each reducing every
        // __num_groups-th segment.
        const std::size_t __num_groups =
            std::min<std::size_t>(__m, std::size_t(std::numeric_limits<int>::max()) / __work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
            __dpl_sycl::__local_accessor<_Acc> __local_mem(sycl::range<1>(__local_mem_size), __cgh);
            __cgh.parallel_for<_Name...>(
                sycl::nd_range<1>(sycl::range<1>(__num_groups * __work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __nd_item) {
                    for (std::size_t __group = __nd_item.get_group(0); __group < std::size_t(__m);
                         __group += __num_groups)
                    {
                        // The local memory is no longer read for the previous segment
                        if (__group != __nd_item.get_group(0))
                            sycl::group_barrier(__nd_item.get_group());

                        const _Size __segment = __group;
                        const _Size __begin =
                            oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment], __n);
                        const _Size __end =
                            oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment + 1], __n);
                        __segmented_reduce_by_work_group<_Acc, __vector_size>(
                            __nd_item, __op, __begin, __begin < __end ? __end - __begin : _Size{0},
                            /*__with_init=*/true, __init, __input, __local_mem,
                            [&](_Acc&& __reduced) { __result[__segment] = std::move(__reduced); });
                    }
                });
        });
    }
};

// The kernels with the smaller register file on Intel GPUs, where the implementation of SYCL provides the property: for
// the kernels of the segments longer than a work-group reduces alone the compiler chooses the larger one, which halves
// the threads an execution unit keeps resident
struct __segmented_reduce_small_grf_kernel
{
#if defined(SYCL_EXT_INTEL_GRF_SIZE) && defined(SYCL_EXT_ONEAPI_KERNEL_PROPERTIES)
    auto
    get(sycl::ext::oneapi::experimental::properties_tag) const
    {
        return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<128>};
    }
#endif
};

// The first kernel of __segmented_reduce_long_group_submitter: one work-group per segment, which reduces a segment of
// at most __max_length elements, and for a longer one zeroes the number of its pieces reduced and writes it into the
// maps of the tiles
template <typename _Acc, std::uint8_t __vector_size, typename _Size, typename _InitType, typename _Op,
          typename _InRange, typename _OffsetRange, typename _OutRange, typename _MapsAcc, typename _CountsAcc,
          typename _LocalMem>
struct __segmented_reduce_long_group_functor : __segmented_reduce_small_grf_kernel
{
    _InRange __input;
    _OffsetRange __offsets;
    _OutRange __result;
    _MapsAcc __maps;     // the segment covering the start of every tile, then the one beginning inside every tile
    _CountsAcc __counts; // the number of the reduced pieces of every long segment
    _LocalMem __local_mem;
    _Size __n;
    _Size __m;
    _Size __max_length;       // the longest segment reduced by its work-group alone
    std::uint8_t __tile_log;  // the tiles have 2^__tile_log elements
    std::size_t __num_tiles;  // the number of the tiles
    std::size_t __num_groups; // the number of the work-groups, each reducing every __num_groups-th segment
    _InitType __init;
    _Op __op;

    void
    operator()(sycl::nd_item<1> __nd_item) const
    {
        const std::size_t __local_idx = __nd_item.get_local_id(0);
        const std::size_t __work_group_size = __nd_item.get_local_range(0);
        const _Size __tile = _Size{1} << __tile_log;
        for (std::size_t __group = __nd_item.get_group(0); __group < std::size_t(__m); __group += __num_groups)
        {
            // The local memory is no longer read for the previous segment
            if (__group != __nd_item.get_group(0))
                sycl::group_barrier(__nd_item.get_group());

            const _Size __segment = __group;
            const _Size __begin = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment], __n);
            const _Size __end = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment + 1], __n);
            const _Size __length = __begin < __end ? __end - __begin : _Size{0};

            // A long segment is left to the second kernel; the same for every work-item of the work-group
            if (__length > __max_length)
            {
                _Size* __map = __maps.__data();
                if (__local_idx == 0)
                {
                    __counts.__data()[__segment] = 0;
                    if ((__begin & (__tile - 1)) != 0)
                        __map[__num_tiles + std::size_t(__begin >> __tile_log)] = __segment;
                }
                for (std::size_t __t = std::size_t((__begin + __tile - 1) >> __tile_log) + __local_idx;
                     __t <= std::size_t((__end - 1) >> __tile_log); __t += __work_group_size)
                    __map[__t] = __segment;
                continue;
            }
            __segmented_reduce_by_work_group<_Acc, __vector_size>(
                __nd_item, __op, __begin, __length, /*__with_init=*/true, __init, __input, __local_mem,
                [&](_Acc&& __reduced) { __result[__segment] = std::move(__reduced); });
        }
    }
};

// The second kernel of __segmented_reduce_long_group_submitter: one work-group per tile, which reduces the pieces of
// the long segments in its tile; the last piece of a segment reduced combines the partial results of all of them
template <typename _Acc, std::uint8_t __vector_size, typename _Size, typename _InitType, typename _Op,
          typename _InRange, typename _OffsetRange, typename _OutRange, typename _MapsAcc, typename _CountsAcc,
          typename _PartialsAcc, typename _LocalMem>
struct __segmented_reduce_long_tile_functor : __segmented_reduce_small_grf_kernel
{
    _InRange __input;
    _OffsetRange __offsets;
    _OutRange __result;
    _MapsAcc __maps;
    _CountsAcc __counts;
    _PartialsAcc __partials; // the partial results of the two pieces of every tile
    _LocalMem __local_mem;
    _Size __n;
    _Size __m;
    _Size __max_length;
    std::uint8_t __tile_log;
    std::size_t __num_tiles;
    std::size_t __num_groups; // the number of the work-groups, each reducing every __num_groups-th tile
    _InitType __init;
    _Op __op;

    void
    operator()(sycl::nd_item<1> __nd_item) const
    {
        using _AtomicCount = sycl::atomic_ref<std::uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                              sycl::access::address_space::global_space>;

        const _Size* __map = __maps.__data();
        _Acc* __partial = __partials.__data();
        const _Size __tile = _Size{1} << __tile_log;
        bool __first_piece = true;
        for (std::size_t __t = __nd_item.get_group(0); __t < __num_tiles; __t += __num_groups)
        {
            const _Size __tile_begin = _Size(__t) << __tile_log;
            const _Size __tile_end = std::min<_Size>(__n, __tile_begin + __tile);
            // The segment that covers the start of the tile (slot 0) and the one that begins inside it (slot 1)
            for (std::size_t __slot = 0; __slot < 2; ++__slot)
            {
                const _Size __segment = __map[__slot * __num_tiles + __t];
                if (std::size_t(__segment) >= std::size_t(__m))
                    continue;
                const _Size __begin =
                    oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment], __n);
                const _Size __end =
                    oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment + 1], __n);
                if (__end - __begin <= __max_length || (__slot == 0 ? __begin > __tile_begin || __tile_begin >= __end
                                                                    : __begin <= __tile_begin || __begin >= __tile_end))
                    continue;
                // The tiles of the segment, without a last one of one element
                const std::size_t __first_tile = std::size_t(__begin >> __tile_log);
                const std::size_t __end_tile = std::size_t((__end - 1) >> __tile_log);
                const std::size_t __last_tile =
                    __end_tile > __first_tile && __end - (_Size(__end_tile) << __tile_log) == 1 ? __end_tile - 1
                                                                                                : __end_tile;
                if (__t > __last_tile)
                    continue;
                const std::uint32_t __num_pieces = std::uint32_t(__last_tile - __first_tile + 1);
                const _Size __piece_begin = std::max<_Size>(__begin, __tile_begin);
                const _Size __piece_end = __t == __last_tile ? __end : __tile_end;

                // The local memory is no longer read for the previous piece
                if (!__first_piece)
                    sycl::group_barrier(__nd_item.get_group());
                __first_piece = false;

                __segmented_reduce_by_work_group<_Acc, __vector_size>(
                    __nd_item, __op, __piece_begin, __piece_end - __piece_begin,
                    /*__with_init=*/__t == __first_tile, __init, __input, __local_mem, [&](_Acc&& __reduced) {
                        __partial[2 * __t + __slot] = std::move(__reduced);
                        // The pieces reduced before are visible to the last one
                        _AtomicCount __count(__counts.__data()[__segment]);
                        if (__count.fetch_add(1) + 1 == __num_pieces)
                        {
                            _Acc __acc = __partial[2 * __first_tile + ((__begin & (__tile - 1)) == 0 ? 0 : 1)];
                            for (std::size_t __u = __first_tile + 1; __u <= __last_tile; ++__u)
                                __acc = std::invoke(__op, std::move(__acc), __partial[2 * __u]);
                            __result[__segment] = std::move(__acc);
                        }
                    });
            }
        }
    }
};

// EXPERIMENT (R4): the reduction of the elements [__first, __last), at least one, of a segment by one sub-group, as
// __segmented_reduce_chunk_group_brick does for the first chunk of a segment; the result, with the initial value, on
// lane 0
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation, typename _SubGroup,
          typename _Input>
_Acc
__segmented_reduce_by_sub_group(const _SubGroup& __sg, _Size __first, const _Size __last, const _InitType& __init,
                                _BinaryOperation& __binary_op, const _Input& __input)
{
    const _Size __lane = __sg.get_local_linear_id();
    const _Size __lanes = __sg.get_local_linear_range();
    // The first element with the initial value
    const _Acc __head = oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
        std::invoke(__binary_op, __init, __input[__first++]));

    // Every lane that takes part gets at least two elements
    _Size __active = std::min<_Size>(__lanes, (__last - __first) / 2);
    _Acc __acc{};
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
        for (__i += 2 * __active; __i + 3 * __active < __last; __i += 4 * __active)
            __acc = std::invoke(__binary_op, __acc,
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
    // A segment of one element more than the initial value takes it alone
    return __active > 0       ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                    std::invoke(__binary_op, __head, __acc))
           : __first < __last ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                    std::invoke(__binary_op, __head, __input[__first]))
                              : __head;
}

// EXPERIMENT (R4): one sub-group per segment, every sub-group reducing every so-many-th segment; with _WithLong a
// segment longer than __max_length is left to the second kernel: its count of pieces is zeroed and it is written into
// the maps of the tiles, as __segmented_reduce_long_group_functor does
template <bool _WithLong, typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation,
          typename _InRange, typename _OffsetRange, typename _OutRange, typename _MapsAcc, typename _CountsAcc>
struct __segmented_reduce_sub_group_functor : __segmented_reduce_small_grf_kernel
{
    _InRange __input;
    _OffsetRange __offsets;
    _OutRange __result;
    _MapsAcc __maps;
    _CountsAcc __counts;
    _Size __n;
    _Size __m;
    _Size __max_length;
    std::uint8_t __tile_log;
    std::size_t __num_tiles;
    _InitType __init;
    _BinaryOperation __binary_op;

    void
    operator()(sycl::nd_item<1> __item) const
    {
        _BinaryOperation __op = __binary_op;
        const __dpl_sycl::__sub_group __sg = __item.get_sub_group();
        const std::size_t __lane = __sg.get_local_linear_id();
        const std::size_t __lanes = __sg.get_local_linear_range();
        const std::size_t __per_group = __sg.get_group_linear_range();
        const std::size_t __stride = __item.get_group_range(0) * __per_group;
        for (std::size_t __s = __item.get_group_linear_id() * __per_group + __sg.get_group_linear_id();
             __s < std::size_t(__m); __s += __stride)
        {
            const _Size __segment = __s;
            const _Size __begin = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment], __n);
            const _Size __end = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment + 1], __n);
            const _Size __length = __begin < __end ? __end - __begin : _Size{0};
            if constexpr (_WithLong)
            {
                if (__length > __max_length)
                {
                    const _Size __tile = _Size{1} << __tile_log;
                    _Size* __map = __maps.__data();
                    if (__lane == 0)
                    {
                        __counts.__data()[__segment] = 0;
                        if ((__begin & (__tile - 1)) != 0)
                            __map[__num_tiles + std::size_t(__begin >> __tile_log)] = __segment;
                    }
                    for (std::size_t __t = std::size_t((__begin + __tile - 1) >> __tile_log) + __lane;
                         __t <= std::size_t((__end - 1) >> __tile_log); __t += __lanes)
                        __map[__t] = __segment;
                    continue;
                }
            }
            if (__length == 0)
            {
                if (__lane == 0)
                    __result[__segment] = oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init);
                continue;
            }
            _Acc __acc = __segmented_reduce_by_sub_group<_Acc>(__sg, __begin, __end, __init, __op, __input);
            if (__lane == 0)
                __result[__segment] = std::move(__acc);
        }
    }
};

template <typename... _Name>
class __segmented_reduce_sub_group_kernel;
template <typename... _Name>
class __segmented_reduce_sub_group_long_kernel;
template <typename... _Name>
class __segmented_reduce_sub_group_tile_kernel;

inline long long
__segred_r4_env(const char* __name, long long __default)
{
    const char* __v = std::getenv(__name);
    return __v ? std::atoll(__v) : __default;
}

// EXPERIMENT (R4): the submitter of S alone, or of S with the second kernel of R3T for the segments longer than
// __max_length
template <typename _Acc, typename _SgName, typename _SgLongName, typename _TileName>
struct __segmented_reduce_sub_group_submitter;

template <typename _Acc, typename... _SgName, typename... _SgLongName, typename... _TileName>
struct __segmented_reduce_sub_group_submitter<_Acc, __internal::__optional_kernel_name<_SgName...>,
                                              __internal::__optional_kernel_name<_SgLongName...>,
                                              __internal::__optional_kernel_name<_TileName...>>
{
    static constexpr std::uint8_t __vector_size = 4;

    template <typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size, typename _InitType,
              typename _BinaryOperation>
    sycl::event
    operator()(sycl::queue& __q, _InRange&& __input, _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n,
               const _Size __m, const std::size_t __work_group_size, const bool __with_long, const _Size __max_length,
               const std::uint8_t __tile_log, const std::uint16_t __tile_work_group_size, _InitType __init,
               _BinaryOperation __binary_op) const
    {
        // As many sub-groups as segments, of 32 lanes where supported
        const std::size_t __lanes = oneapi::dpl::__internal::__supports_sub_group_size(__q, 32) ? 32 : 16;
        const std::size_t __per_group = std::max<std::size_t>(1, __work_group_size / __lanes);
        const std::size_t __num_groups = std::min<std::size_t>(
            oneapi::dpl::__internal::__dpl_ceiling_div(std::size_t(__m), __per_group),
            std::size_t(std::numeric_limits<int>::max()) / __work_group_size);
        if (!__with_long)
        {
            return __q.submit([&](sycl::handler& __cgh) {
                oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
                __cgh.parallel_for<_SgName...>(
                    sycl::nd_range<1>(sycl::range<1>(__num_groups * __work_group_size),
                                      sycl::range<1>(__work_group_size)),
                    __segmented_reduce_sub_group_functor<false, _Acc, _Size, _InitType, _BinaryOperation,
                                                         std::decay_t<_InRange>, std::decay_t<_OffsetRange>,
                                                         std::decay_t<_OutRange>, int, int>{
                        {}, __input, __offsets, __result, 0, 0, __n, __m, __n, 0, 0, __init, __binary_op});
            });
        }

        auto __op = __segmented_reduce_group_op<_Acc>(__binary_op);
        using _Op = decltype(__op);
        using _ReducePattern = unseq_backend::reduce_over_group<_Op, _Acc>;
        using _LocalMem = __dpl_sycl::__local_accessor<_Acc>;
        const std::size_t __local_mem_size = _ReducePattern{__op}.local_mem_req(__tile_work_group_size);
        const std::size_t __num_tiles = std::size_t((__n + (_Size{1} << __tile_log) - 1) >> __tile_log);
        __device_storage<_Size> __maps{__q, 2 * __num_tiles};
        __device_storage<std::uint32_t> __counts{__q, std::size_t(__m)};
        __device_storage<_Acc> __partials{__q, 2 * __num_tiles};
        sycl::event __sg_event = __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
            auto __maps_acc = __get_accessor(sycl::write_only, __maps, __cgh, __dpl_sycl::__no_init{});
            auto __counts_acc = __get_accessor(sycl::write_only, __counts, __cgh, __dpl_sycl::__no_init{});
            __cgh.parallel_for<_SgLongName...>(
                sycl::nd_range<1>(sycl::range<1>(__num_groups * __work_group_size), sycl::range<1>(__work_group_size)),
                __segmented_reduce_sub_group_functor<true, _Acc, _Size, _InitType, _BinaryOperation,
                                                     std::decay_t<_InRange>, std::decay_t<_OffsetRange>,
                                                     std::decay_t<_OutRange>, decltype(__maps_acc),
                                                     decltype(__counts_acc)>{{},
                                                                             __input,
                                                                             __offsets,
                                                                             __result,
                                                                             __maps_acc,
                                                                             __counts_acc,
                                                                             __n,
                                                                             __m,
                                                                             __max_length,
                                                                             __tile_log,
                                                                             __num_tiles,
                                                                             __init,
                                                                             __binary_op});
        });
        const std::size_t __num_tile_groups = std::min<std::size_t>(
            __num_tiles, std::size_t(std::numeric_limits<int>::max()) / __tile_work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
            auto __maps_acc = __get_accessor(sycl::read_only, __maps, __cgh);
            auto __counts_acc = __get_accessor(sycl::read_write, __counts, __cgh);
            auto __partials_acc = __get_accessor(sycl::read_write, __partials, __cgh, __dpl_sycl::__no_init{});
            __cgh.depends_on(__sg_event);
            _LocalMem __local_mem(sycl::range<1>(__local_mem_size), __cgh);
            __cgh.parallel_for<_TileName...>(
                sycl::nd_range<1>(sycl::range<1>(__num_tile_groups * __tile_work_group_size),
                                  sycl::range<1>(__tile_work_group_size)),
                __segmented_reduce_long_tile_functor<_Acc, __vector_size, _Size, _InitType, _Op, std::decay_t<_InRange>,
                                                     std::decay_t<_OffsetRange>, std::decay_t<_OutRange>,
                                                     decltype(__maps_acc), decltype(__counts_acc),
                                                     decltype(__partials_acc), _LocalMem>{{},
                                                                                          __input,
                                                                                          __offsets,
                                                                                          __result,
                                                                                          __maps_acc,
                                                                                          __counts_acc,
                                                                                          __partials_acc,
                                                                                          __local_mem,
                                                                                          __n,
                                                                                          __m,
                                                                                          __max_length,
                                                                                          __tile_log,
                                                                                          __num_tiles,
                                                                                          __num_tile_groups,
                                                                                          __init,
                                                                                          __op});
        });
    }
};

template <typename... _Name>
class __segmented_reduce_long_group_kernel;
template <typename... _Name>
class __segmented_reduce_long_tile_kernel;

template <typename _Acc, typename _GroupName, typename _TileName>
struct __segmented_reduce_long_group_submitter;

// One work-group per segment, as __segmented_reduce_group_submitter does, for an input with segments longer than
// __max_length, which a work-group would reduce too long alone. The first kernel reduces the other segments. For every
// longer one it zeroes the number of its pieces reduced and writes it into the maps of the tiles of 2^__tile_log
// elements: the segment that covers the start of a tile, and the one that begins inside a tile, at most one, as a long
// segment is longer than a tile. The second kernel, one work-group per tile, reduces the pieces of the long segments in
// its tile, and the last piece of a segment reduced combines the partial results of all of them. The maps are not
// initialized: every entry is checked against the offsets. A last piece of one element is left to the tile before it,
// so that no partial result is made from an element alone.
template <typename _Acc, typename... _GroupName, typename... _TileName>
struct __segmented_reduce_long_group_submitter<_Acc, __internal::__optional_kernel_name<_GroupName...>,
                                               __internal::__optional_kernel_name<_TileName...>>
{
    // As in __parallel_transform_reduce
    static constexpr std::uint8_t __vector_size = 4;

    template <typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size, typename _InitType,
              typename _BinaryOperation>
    sycl::event
    operator()(sycl::queue& __q, _InRange&& __input, _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n,
               const _Size __m, const std::uint16_t __work_group_size, const _Size __max_length,
               const std::uint8_t __tile_log, _InitType __init, _BinaryOperation __binary_op) const
    {
        auto __op = __segmented_reduce_group_op<_Acc>(std::move(__binary_op));
        using _Op = decltype(__op);
        using _ReducePattern = unseq_backend::reduce_over_group<_Op, _Acc>;
        using _LocalMem = __dpl_sycl::__local_accessor<_Acc>;

        const std::size_t __local_mem_size = _ReducePattern{__op}.local_mem_req(__work_group_size);
        const std::size_t __num_tiles = std::size_t((__n + (_Size{1} << __tile_log) - 1) >> __tile_log);
        // In USM device memory where supported, as the other temporary data of the paths above
        __device_storage<_Size> __maps{__q, 2 * __num_tiles};
        __device_storage<std::uint32_t> __counts{__q, std::size_t(__m)};
        __device_storage<_Acc> __partials{__q, 2 * __num_tiles};

        // As in __segmented_reduce_group_submitter
        const std::size_t __num_groups =
            std::min<std::size_t>(__m, std::size_t(std::numeric_limits<int>::max()) / __work_group_size);
        sycl::event __group_event = __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
            auto __maps_acc = __get_accessor(sycl::write_only, __maps, __cgh, __dpl_sycl::__no_init{});
            auto __counts_acc = __get_accessor(sycl::write_only, __counts, __cgh, __dpl_sycl::__no_init{});
            _LocalMem __local_mem(sycl::range<1>(__local_mem_size), __cgh);
            __cgh.parallel_for<_GroupName...>(
                sycl::nd_range<1>(sycl::range<1>(__num_groups * __work_group_size), sycl::range<1>(__work_group_size)),
                __segmented_reduce_long_group_functor<
                    _Acc, __vector_size, _Size, _InitType, _Op, std::decay_t<_InRange>, std::decay_t<_OffsetRange>,
                    std::decay_t<_OutRange>, decltype(__maps_acc), decltype(__counts_acc), _LocalMem>{{},
                                                                                                      __input,
                                                                                                      __offsets,
                                                                                                      __result,
                                                                                                      __maps_acc,
                                                                                                      __counts_acc,
                                                                                                      __local_mem,
                                                                                                      __n,
                                                                                                      __m,
                                                                                                      __max_length,
                                                                                                      __tile_log,
                                                                                                      __num_tiles,
                                                                                                      __num_groups,
                                                                                                      __init,
                                                                                                      __op});
        });

        const std::size_t __num_tile_groups =
            std::min<std::size_t>(__num_tiles, std::size_t(std::numeric_limits<int>::max()) / __work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
            auto __maps_acc = __get_accessor(sycl::read_only, __maps, __cgh);
            auto __counts_acc = __get_accessor(sycl::read_write, __counts, __cgh);
            auto __partials_acc = __get_accessor(sycl::read_write, __partials, __cgh, __dpl_sycl::__no_init{});
            __cgh.depends_on(__group_event);
            _LocalMem __local_mem(sycl::range<1>(__local_mem_size), __cgh);
            __cgh.parallel_for<_TileName...>(
                sycl::nd_range<1>(sycl::range<1>(__num_tile_groups * __work_group_size),
                                  sycl::range<1>(__work_group_size)),
                __segmented_reduce_long_tile_functor<_Acc, __vector_size, _Size, _InitType, _Op, std::decay_t<_InRange>,
                                                     std::decay_t<_OffsetRange>, std::decay_t<_OutRange>,
                                                     decltype(__maps_acc), decltype(__counts_acc),
                                                     decltype(__partials_acc), _LocalMem>{{},
                                                                                          __input,
                                                                                          __offsets,
                                                                                          __result,
                                                                                          __maps_acc,
                                                                                          __counts_acc,
                                                                                          __partials_acc,
                                                                                          __local_mem,
                                                                                          __n,
                                                                                          __m,
                                                                                          __max_length,
                                                                                          __tile_log,
                                                                                          __num_tiles,
                                                                                          __num_tile_groups,
                                                                                          __init,
                                                                                          __op});
        });
    }
};

// The carry of a tile of the reduction by tiles: the partial result of the piece of a segment begun before the tile
// (__kind 1) or a mark for such a piece of one element (__kind 2), and the segment beginning in the tile and crossing
// its end (__cross, the number of segments if none) with the partial result of its first piece
template <typename _Acc, typename _Size>
struct __segmented_reduce_tile_carry
{
    _Acc __in;
    _Acc __out;
    _Size __cross;
    std::uint8_t __kind;
};

template <typename... _Name>
class __segmented_reduce_tile_kernel;
template <typename... _Name>
class __segmented_reduce_tile_fixup_kernel;
template <typename... _Name>
class __segmented_reduce_tile_group_fixup_kernel;

template <typename _Acc, typename _KernelName, typename _FixupName, typename _GroupFixupName>
struct __segmented_reduce_tile_submitter;

template <typename _Acc, typename... _Name, typename... _FixupName, typename... _GroupFixupName>
struct __segmented_reduce_tile_submitter<_Acc, __internal::__optional_kernel_name<_Name...>,
                                         __internal::__optional_kernel_name<_FixupName...>,
                                         __internal::__optional_kernel_name<_GroupFixupName...>>
{
    template <typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size, typename _InitType,
              typename _BinaryOperation>
    void
    operator()(sycl::queue& __q, _InRange&& __input, _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n,
               const _Size __m, const std::size_t __work_group_size, const _Size __tile, _InitType __init,
               _BinaryOperation __binary_op) const
    {
        using _Carry = __segmented_reduce_tile_carry<_Acc, _Size>;

        const _Size __num_tiles = (__n + __tile - 1) / __tile;
        // In USM device memory where supported: a buffer would add the cost of its accessors to every kernel and of
        // its destruction to the call
        __device_storage<_Carry> __carries{__q, std::size_t(__num_tiles)};
        const std::size_t __num_groups =
            std::min<std::size_t>(__num_tiles, std::size_t(std::numeric_limits<int>::max()) / __work_group_size);
        const _Size __wg = _Size(__work_group_size);

        sycl::event __tile_event = __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
            auto __carry_acc = __get_accessor(sycl::write_only, __carries, __cgh, __dpl_sycl::__no_init{});
            __dpl_sycl::__local_accessor<_Acc> __local_mem(sycl::range<1>(__work_group_size), __cgh);
            __cgh.parallel_for<_Name...>(
                sycl::nd_range<1>(sycl::range<1>(__num_groups * __work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item) {
                    _BinaryOperation __op = __binary_op;
                    _Carry* const __carry = __carry_acc.__data();
                    const _Size __lid = __item.get_local_id(0);
                    const __dpl_sycl::__sub_group __sg = __item.get_sub_group();
                    const _Size __lane = __sg.get_local_linear_id();
                    const _Size __lanes = __sg.get_local_linear_range();
                    const _Size __sg_first = _Size(__sg.get_group_linear_id()) * __lanes;
                    auto __clamp = [&](_Size __s) {
                        return oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__s], __n);
                    };
                    // The partial result of [__b, __b + __len), __len >= 2, in work-item 0; with the initial value it
                    // starts from the operation on the initial value and the first element
                    auto __together = [&](_Size __b, _Size __len, bool __with_init) {
                        // The local memory is no longer read for the previous piece
                        sycl::group_barrier(__item.get_group());
                        _Acc __head{};
                        if (__with_init)
                        {
                            __head = std::invoke(__op, __init, __input[__b]);
                            ++__b;
                            --__len;
                        }
                        // Every work-item that takes part gets at least two elements
                        const _Size __active = std::min<_Size>(__wg, __len / 2);
                        _Acc __acc{};
                        if (__lid < __active)
                        {
                            const _Size __last = __b + __len;
                            _Size __i = __b + __lid;
                            __acc = std::invoke(__op, __input[__i], __input[__i + __active]);
                            for (__i += 2 * __active; __i + 3 * __active < __last; __i += 4 * __active)
                                __acc = std::invoke(
                                    __op, __acc,
                                    std::invoke(__op,
                                                oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                                    std::invoke(__op, __input[__i], __input[__i + __active])),
                                                oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(std::invoke(
                                                    __op, __input[__i + 2 * __active], __input[__i + 3 * __active]))));
                            for (; __i < __last; __i += __active)
                                __acc = std::invoke(__op, __acc, __input[__i]);
                        }
                        const _Size __sg_active =
                            std::max<_Size>(std::min<_Size>(__active - __sg_first, __lanes), _Size{0});
                        for (_Size __shift = 1; __shift < __sg_active; __shift *= 2)
                        {
                            const _Acc __other = sycl::shift_group_left(__sg, __acc, __shift);
                            if (__lane % (2 * __shift) == 0 && __lane + __shift < __sg_active)
                                __acc = std::invoke(__op, __acc, __other);
                        }
                        const _Size __groups_active = (__active + __lanes - 1) / __lanes;
                        if (__lane == 0 && __sg_active > 0)
                            __local_mem[__sg_first / __lanes] = __acc;
                        sycl::group_barrier(__item.get_group());
                        _Acc __total{};
                        if (__lid == 0)
                        {
                            if (__active > 0)
                            {
                                __total = __local_mem[0];
                                for (_Size __k = 1; __k < __groups_active; ++__k)
                                    __total = std::invoke(__op, __total, __local_mem[__k]);
                            }
                            if (__with_init)
                                __total = __active > 0 ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                                             std::invoke(__op, __head, __total))
                                          : __len == 1 ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                                             std::invoke(__op, __head, __input[__b]))
                                                       : __head;
                        }
                        return __total;
                    };
                    for (std::size_t __t = __item.get_group(0); __t < std::size_t(__num_tiles); __t += __num_groups)
                    {
                        const _Size __tb = _Size(__t) * __tile;
                        const _Size __te = std::min<_Size>(__n, __tb + __tile);
                        // The first segment beginning at or after a position; the last tile also takes the segments
                        // beginning at the end of the input
                        auto __lower = [&](_Size __v) {
                            _Size __lo = 0, __hi = __m;
                            while (__lo < __hi)
                            {
                                const _Size __mid = __lo + (__hi - __lo) / 2;
                                if (__clamp(__mid) < __v)
                                    __lo = __mid + 1;
                                else
                                    __hi = __mid;
                            }
                            return __lo;
                        };
                        const _Size __sa = __lower(__tb);
                        const _Size __sb = __te == __n ? __m : __lower(__te);

                        // (A) The short segments wholly in the tile, one per work-item
                        for (_Size __s = __sa + __lid; __s < __sb; __s += __wg)
                        {
                            const _Size __b = __clamp(__s);
                            const _Size __e = std::max<_Size>(__clamp(__s + 1), __b);
                            if (__e <= __te && __e - __b < __wg)
                                __result[__s] = oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(
                                    __input, __b, __e, __init, __op);
                        }
                        // (B) The piece of a segment begun before the tile
                        std::uint8_t __k = 0;
                        _Acc __cin{};
                        if (__sa > 0)
                        {
                            const _Size __e = std::min<_Size>(__clamp(__sa), __te);
                            if (__e - __tb >= 2)
                            {
                                __cin = __together(__tb, __e - __tb, false);
                                __k = 1;
                            }
                            else if (__e - __tb == 1)
                                __k = 2;
                        }
                        // ... and the long ones and the one crossing the end of the tile
                        _Size __cs = __m;
                        _Acc __cout{};
                        for (_Size __s = __sa; __s < __sb; ++__s)
                        {
                            const _Size __b = __clamp(__s);
                            const _Size __e = std::max<_Size>(__clamp(__s + 1), __b);
                            if (__e <= __te && __e - __b < __wg)
                                continue;
                            const _Size __len = std::max<_Size>(std::min<_Size>(__e, __te) - __b, _Size{0});
                            if (__len >= 2)
                            {
                                _Acc __p = __together(__b, __len, true);
                                if (__lid == 0)
                                {
                                    if (__e <= __te)
                                        __result[__s] = __p;
                                    else
                                        __cout = __p;
                                }
                            }
                            else if (__lid == 0)
                            {
                                const _Acc __p = __len == 0
                                                     ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init)
                                                     : oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                                           std::invoke(__op, __init, __input[__b]));
                                if (__e <= __te || __len == 0)
                                    __result[__s] = __p;
                                else
                                    __cout = __p;
                            }
                            if (__e > __te && __len > 0)
                                __cs = __s;
                        }
                        if (__lid == 0)
                            __carry[__t] = _Carry{__cin, __cout, __cs, __k};
                    }
                });
        });

        // The long chains of carries, one sub-group per tile: the lanes take the carries of the following tiles of the
        // segment crossing its end in turn, which are partial results, and a tree combines them, so that no
        // accumulator is made from an element alone; lane 0 starts from the first piece of the segment and adds the
        // last one if it is of one element, which only the last tile of a segment can hold
        if (__n / __m >= _Size(__segmented_reduce_gpu_group_fixup_tiles) * __tile)
        {
            const std::size_t __sub_groups_per_group =
                std::max<std::size_t>(1, __work_group_size / __segmented_reduce_gpu_sub_group_size);
            const std::size_t __num_fixup_groups = std::min<std::size_t>(
                oneapi::dpl::__internal::__dpl_ceiling_div(std::size_t(__num_tiles), __sub_groups_per_group),
                std::size_t(std::numeric_limits<int>::max()) / __work_group_size);
            __q.submit([&](sycl::handler& __cgh) {
                   oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
                   auto __carry_acc = __get_accessor(sycl::read_only, __carries, __cgh);
                   __cgh.depends_on(__tile_event);
                   __cgh.parallel_for<_GroupFixupName...>(
                       sycl::nd_range<1>(sycl::range<1>(__num_fixup_groups * __work_group_size),
                                         sycl::range<1>(__work_group_size)),
                       [=](sycl::nd_item<1> __item) {
                           _BinaryOperation __op = __binary_op;
                           const _Carry* const __carry = __carry_acc.__data();
                           const __dpl_sycl::__sub_group __sg = __item.get_sub_group();
                           const _Size __lane = __sg.get_local_linear_id();
                           const _Size __lanes = __sg.get_local_linear_range();
                           const _Size __num_sub_groups =
                               _Size(__item.get_group_range(0)) * _Size(__sg.get_group_linear_range());
                           // Every sub-group takes every (number of the sub-groups)-th tile
                           for (_Size __t = _Size(__item.get_group(0)) * _Size(__sg.get_group_linear_range()) +
                                            _Size(__sg.get_group_linear_id());
                                __t < __num_tiles; __t += __num_sub_groups)
                           {
                               const _Size __s = __carry[__t].__cross;
                               if (__s >= __m)
                                   continue;
                               const _Size __e =
                                   oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__s + 1], __n);
                               const _Size __t1 = std::min<_Size>((__e - 1) / __tile, __num_tiles - 1);
                               _Acc __acc{};
                               bool __has = false;
                               for (_Size __u = __t + 1 + __lane; __u <= __t1; __u += __lanes)
                               {
                                   if (__carry[__u].__kind == 1 && __has)
                                       __acc = std::invoke(__op, __acc, __carry[__u].__in);
                                   if (__carry[__u].__kind == 1 && !__has)
                                       __acc = __carry[__u].__in;
                                   __has = __has || __carry[__u].__kind == 1;
                               }
                               // A tree within the sub-group, whatever lanes hold a partial result
                               for (_Size __shift = 1; __shift < __lanes; __shift *= 2)
                               {
                                   const _Acc __other = sycl::shift_group_left(__sg, __acc, __shift);
                                   const bool __other_has = sycl::shift_group_left(__sg, int(__has), __shift) != 0;
                                   const bool __take =
                                       __lane % (2 * __shift) == 0 && __lane + __shift < __lanes && __other_has;
                                   if (__take && __has)
                                       __acc = std::invoke(__op, __acc, __other);
                                   if (__take && !__has)
                                       __acc = __other;
                                   __has = __has || __take;
                               }
                               if (__lane == 0)
                               {
                                   _Acc __total = __carry[__t].__out;
                                   if (__has)
                                       __total = std::invoke(__op, __total, __acc);
                                   if (__t1 > __t && __carry[__t1].__kind == 2)
                                       __total = std::invoke(__op, __total, __input[__t1 * __tile]);
                                   __result[__s] = __total;
                               }
                           }
                       });
               })
                .wait_and_throw();
            return;
        }
        // The short chains, one work-item per tile
        __q.submit([&](sycl::handler& __cgh) {
               oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
               auto __carry_acc = __get_accessor(sycl::read_only, __carries, __cgh);
               __cgh.depends_on(__tile_event);
               __cgh.parallel_for<_FixupName...>(sycl::range<1>(__num_tiles), [=](sycl::item<1> __it) {
                   _BinaryOperation __op = __binary_op;
                   const _Carry* const __carry = __carry_acc.__data();
                   const _Size __t = __it.get_linear_id();
                   const _Size __s = __carry[__t].__cross;
                   if (__s >= __m)
                       return;
                   const _Size __e = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__s + 1], __n);
                   const _Size __t1 = std::min<_Size>((__e - 1) / __tile, __num_tiles - 1);
                   _Acc __acc = __carry[__t].__out;
                   for (_Size __u = __t + 1; __u <= __t1; ++__u)
                   {
                       if (__carry[__u].__kind == 1)
                           __acc = std::invoke(__op, __acc, __carry[__u].__in);
                       else if (__carry[__u].__kind == 2)
                           __acc = std::invoke(__op, __acc, __input[__u * __tile]);
                   }
                   __result[__s] = __acc;
               });
           })
            .wait_and_throw();
    }
};

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

        // On GPUs the kernel that reduces the chunks of a few segments computes their ends itself. The ends and the
        // partial results are in USM device memory where supported: buffers would add the cost of their accessors to
        // both kernels and of their destruction to the call.
        if (__is_gpu && __m <= _Size(__by_groups ? __segmented_reduce_gpu_local_ends_segments
                                                 : __segmented_reduce_gpu_local_ends_item_segments))
        {
            sycl::queue __q = __exec.queue();
            const std::size_t __work_group_size = oneapi::dpl::__internal::__max_work_group_size(__q, std::size_t(256));
            const std::size_t __max_groups = oneapi::dpl::__internal::__max_compute_units(__q);
            __device_storage<_Size> __ends_storage{__q, std::size_t(__m)};
            __device_storage<_Acc> __partials_storage{__q, std::size_t(__num_partials)};
            auto __reduce_chunks = [&]() -> sycl::event {
                if constexpr (std::is_arithmetic_v<_Acc> &&
                              std::is_arithmetic_v<oneapi::dpl::__internal::__value_t<std::decay_t<_InRange>>>)
                {
                    if (__by_groups)
                    {
                        using _GroupKernel =
                            __internal::__kernel_name_provider<__segmented_reduce_chunk_group_local_kernel<
                                oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;

                        // One sub-group per chunk, at most as many work-groups as compute units: every one of them
                        // computes the ends of the chunks of all the segments
                        const std::size_t __chunks_per_group =
                            __work_group_size / __segmented_reduce_gpu_sub_group_size;
                        const std::size_t __num_sub_groups =
                            std::min<std::size_t>(__num_partials, __max_groups * __chunks_per_group);
                        return __segmented_reduce_chunk_local_submitter</*_ByGroups=*/true, _GroupKernel>{}(
                            __q,
                            __segmented_reduce_chunk_group_brick<_Acc, _Size, _InitType, _BinaryOperation>{
                                __n, __m, __num_partials, __init, __binary_op},
                            oneapi::dpl::__internal::__dpl_ceiling_div(__num_sub_groups, __chunks_per_group),
                            __work_group_size, __n, __m, __min_length, __num_partials, __ends_storage,
                            __partials_storage, __offsets, __input);
                    }
                }
                using _ItemKernel = __internal::__kernel_name_provider<__segmented_reduce_chunk_local_kernel<
                    oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;

                // One work-item per chunk, as many work-groups as compute units at most
                const std::size_t __num_groups = std::min<std::size_t>(
                    oneapi::dpl::__internal::__dpl_ceiling_div(std::size_t(__num_partials), __work_group_size),
                    __max_groups);
                return __segmented_reduce_chunk_local_submitter</*_ByGroups=*/false, _ItemKernel>{}(
                    __q,
                    __segmented_reduce_chunk_brick<_Acc, _Size, _InitType, _BinaryOperation>{__n, __m, __is_gpu, __init,
                                                                                             __binary_op},
                    __num_groups, __work_group_size, __n, __m, __min_length, __num_partials, __ends_storage,
                    __partials_storage, __offsets, __input);
            };
            sycl::event __chunk_event = __reduce_chunks();

            using _CombineKernel = __internal::__kernel_name_provider<
                __segmented_reduce_combine_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;
            __segmented_reduce_combine_submitter<_CombineKernel>{}(
                __q,
                __segmented_reduce_combine_brick<_Acc, _Size, _InitType, _BinaryOperation>{
                    __num_partials, __write_empty, __init, __binary_op},
                __chunk_event, __m, __ends_storage, __partials_storage, __result)
                .wait_and_throw();
            return;
        }

        // The end of the chunks of every segment: an inclusive scan of their numbers, computed from the pairs of
        // the adjacent offsets
        auto __chunk_ends = __buffer<_Size>(__m).get_buffer();
        auto __partials = __buffer<_Acc>(__num_partials).get_buffer();

        // A scan otherwise. The chunks depend on it through __chunk_ends, so the host does not wait for it; its future
        // keeps the temporary data of the scan until the partial results are combined.
        auto __scan = __parallel_transform_scan(
            _BackendTag{}, make_wrapped_policy<__segmented_reduce_scan_wrapper>(__exec),
            oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::take_view_simple(__offsets, __m),
                                                 oneapi::dpl::__ranges::drop_view_simple(__offsets, _Size{1})),
            oneapi::dpl::__ranges::views::all_write(__chunk_ends), __m,
            __segmented_reduce_chunk_count<_Size>{__n, __min_length, __is_gpu, __by_groups},
            unseq_backend::__no_init_value<_Size>{}, std::plus<_Size>{}, /*_Inclusive=*/std::true_type{});
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


    // EXPERIMENT (R4): one sub-group per segment
    auto __try_sub_groups = [&]() -> bool {
        if constexpr (std::is_arithmetic_v<_Acc> &&
                      std::is_arithmetic_v<oneapi::dpl::__internal::__value_t<std::decay_t<_InRange>>>)
        {
            if (!__is_gpu || __segred_r4_env("SEGRED_S", 1) == 0)
                return false;
            const _Size __avg = __n / __m;
            if (__avg < _Size(__segred_r4_env("SEGRED_S_MIN_AVG", 24)) || __avg >= _Size(__segred_r4_env("SEGRED_S_A", 2048)) ||
                __m < _Size(__segred_r4_env("SEGRED_S_M", 100)))
                return false;
            using _SgKernel = __internal::__kernel_name_provider<
                __segmented_reduce_sub_group_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;
            using _SgLongKernel = __internal::__kernel_name_provider<__segmented_reduce_sub_group_long_kernel<
                oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;
            using _SgTileKernel = __internal::__kernel_name_provider<__segmented_reduce_sub_group_tile_kernel<
                oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;
            sycl::queue __q = __exec.queue();
            const std::size_t __wg = oneapi::dpl::__internal::__max_work_group_size(
                __q, std::size_t(__segred_r4_env("SEGRED_S_WG", 256)));
            const bool __with_long = __n > _Size(__segred_r4_env("SEGRED_S_ALONE_N", 65536));
            const _Size __max_length = _Size(__segred_r4_env("SEGRED_S_L", 8192));
            // The tile of the second kernel: as for R3T, but at most the longest segment of S
            const std::size_t __per_unit =
                std::size_t(__n) / std::max<std::size_t>(1, oneapi::dpl::__internal::__max_compute_units(__q));
            std::size_t __tile = std::min<std::size_t>(std::max<std::size_t>(__segmented_reduce_gpu_min_long_tile, __per_unit),
                                                       __segmented_reduce_gpu_large_group_max_length);
            std::uint8_t __tile_log = 0;
            while ((std::size_t{1} << (__tile_log + 1)) <= std::size_t(__max_length) && (std::size_t{1} << __tile_log) < __tile)
                ++__tile_log;
            const std::size_t __tile_wg = std::min<std::size_t>(
                oneapi::dpl::__internal::__slm_adjusted_work_group_size(__q, std::size_t(sizeof(_Acc) * 2)), 256);
            sycl::event __event = __segmented_reduce_sub_group_submitter<_Acc, _SgKernel, _SgLongKernel, _SgTileKernel>()(
                __q, __input, __offsets, __result, __n, __m, __wg, __with_long, __max_length, __tile_log,
                static_cast<std::uint16_t>(__tile_wg), __init, __binary_op);
            __event.wait_and_throw();
            return true;
        }
        else
            return false;
    };
    if (__segred_r4_env("SEGRED_S_FIRST", 0) != 0 && __try_sub_groups())
        return;

    // On GPUs a work-item reduces the short segments of a small input alone faster than a work-group per segment and
    // than the scan, the chunks and their combination; a longer segment takes a work-item longer than a work-group
    if (__is_gpu && __n <= _Size(__segmented_reduce_gpu_small_input) &&
        __n / __m < _Size(__segmented_reduce_gpu_small_average))
    {
        __parallel_for(
            _BackendTag{}, make_wrapped_policy<__segmented_reduce_short_wrapper>(__exec),
            __segmented_reduce_variable_brick<_Acc, _Size, _InitType, _BinaryOperation>{__n, __n, __init, __binary_op},
            __m, __offsets, oneapi::dpl::__ranges::make_zip_view(__result), __input)
            .__checked_deferrable_wait();
        return;
    }

    // On GPUs one work-group per segment reduces an input of medium size in one kernel, without the scan, the chunks
    // and their combination: no segment is longer than the input. The work-groups of many very short segments cost
    // more than the other paths.
    if (__is_gpu && __n <= _Size(__segmented_reduce_gpu_group_input) &&
        (__m <= _Size(__segmented_reduce_gpu_group_max_segments) ||
         __n / __m >= _Size(__segmented_reduce_gpu_group_min_average)))
    {
        using _GroupKernel = __internal::__kernel_name_provider<
            __segmented_reduce_group_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;

        sycl::queue __q = __exec.queue();
        // The work-group size as in __parallel_transform_reduce: adjusted to the local memory limit, which is doubled
        // to take into account the memory used by the compiled kernel, and at most 256
        const std::size_t __work_group_size = std::min<std::size_t>(
            oneapi::dpl::__internal::__slm_adjusted_work_group_size(__q, std::size_t(sizeof(_Acc) * 2)), 256);
        __segmented_reduce_group_submitter<_Acc, _GroupKernel>()(__q, __input, __offsets, __result, __n, __m,
                                                                 static_cast<std::uint16_t>(__work_group_size), __init,
                                                                 __binary_op)
            .wait_and_throw();
        return;
    }

    if (__try_sub_groups())
        return;

    // On GPUs one work-group per segment also reduces a larger input of enough segments, not long ones on average: in
    // one kernel when no segment can be longer than a work-group reduces alone, otherwise, for longer segments on
    // average, with a second kernel for the segments longer than that, one work-group per tile of the input
    if (__is_gpu && __m >= _Size(__segmented_reduce_gpu_large_group_min_segments) &&
        __n / __m < _Size(__segmented_reduce_gpu_large_group_max_average) &&
        __n / __m >= _Size(__n <= _Size(__segmented_reduce_gpu_large_group_max_length)
                               ? __segmented_reduce_gpu_group_min_average
                               : __segmented_reduce_gpu_large_group_long_average))
    {
        sycl::queue __q = __exec.queue();
        // As for the medium inputs above
        const std::size_t __work_group_size = std::min<std::size_t>(
            oneapi::dpl::__internal::__slm_adjusted_work_group_size(__q, std::size_t(sizeof(_Acc) * 2)), 256);
        sycl::event __event;
        if (__n <= _Size(__segmented_reduce_gpu_large_group_max_length))
        {
            using _GroupKernel = __internal::__kernel_name_provider<
                __segmented_reduce_group_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;

            __event = __segmented_reduce_group_submitter<_Acc, _GroupKernel>()(
                __q, __input, __offsets, __result, __n, __m, static_cast<std::uint16_t>(__work_group_size), __init,
                __binary_op);
        }
        else
        {
            using _LongGroupKernel = __internal::__kernel_name_provider<
                __segmented_reduce_long_group_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;
            using _LongTileKernel = __internal::__kernel_name_provider<
                __segmented_reduce_long_tile_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;

            // The tile: the least power of two of at least the least tile length and the input size per compute unit,
            // up to the longest segment a work-group reduces alone
            const std::size_t __per_unit =
                std::size_t(__n) / std::max<std::size_t>(1, oneapi::dpl::__internal::__max_compute_units(__q));
            const std::size_t __tile =
                std::min<std::size_t>(std::max<std::size_t>(__segmented_reduce_gpu_min_long_tile, __per_unit),
                                      __segmented_reduce_gpu_large_group_max_length);
            std::uint8_t __tile_log = 0;
            while ((std::size_t{1} << __tile_log) < __tile)
                ++__tile_log;
            __event = __segmented_reduce_long_group_submitter<_Acc, _LongGroupKernel, _LongTileKernel>()(
                __q, __input, __offsets, __result, __n, __m, static_cast<std::uint16_t>(__work_group_size),
                _Size(__segmented_reduce_gpu_large_group_max_length), __tile_log, __init, __binary_op);
        }
        __event.wait_and_throw();
        return;
    }

    // On GPUs the long segments of arithmetic types by tiles, one work-group per tile, whatever their lengths
    if constexpr (std::is_arithmetic_v<_Acc> &&
                  std::is_arithmetic_v<oneapi::dpl::__internal::__value_t<std::decay_t<_InRange>>>)
    {
        if (__is_gpu && __n / __m >= _Size(__segmented_reduce_gpu_tiles_average) &&
            oneapi::dpl::__internal::__supports_sub_group_size(__exec.queue(), __segmented_reduce_gpu_sub_group_size))
        {
            using _TileKernel = __internal::__kernel_name_provider<
                __segmented_reduce_tile_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;
            using _FixupKernel = __internal::__kernel_name_provider<
                __segmented_reduce_tile_fixup_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;
            using _GroupFixupKernel = __internal::__kernel_name_provider<__segmented_reduce_tile_group_fixup_kernel<
                oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;
            sycl::queue __q = __exec.queue();
            __segmented_reduce_tile_submitter<_Acc, _TileKernel, _FixupKernel, _GroupFixupKernel>()(
                __q, __input, __offsets, __result, __n, __m,
                oneapi::dpl::__internal::__max_work_group_size(__q, std::size_t(256)),
                _Size(__n <= _Size(__segmented_reduce_gpu_small_tile_input) ? __segmented_reduce_gpu_small_tile
                                                                            : __segmented_reduce_gpu_large_tile),
                __init, __binary_op);
            return;
        }
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
