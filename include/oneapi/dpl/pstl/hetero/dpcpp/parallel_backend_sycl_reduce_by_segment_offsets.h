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

// On GPUs every segment is reduced by one work-group of this size in one kernel, every work-item loading this many
// elements of every tile of the segment
inline constexpr std::size_t __segmented_reduce_gpu_group_size = 256;
inline constexpr std::size_t __segmented_reduce_gpu_items_per_item = 16;

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

template <typename... _Name>
class __segmented_reduce_block_kernel;

template <typename _Acc, typename _KernelName>
struct __segmented_reduce_block_submitter;

// One work-group per segment, for arithmetic types. Every work-item loads its elements of a whole tile of the segment
// before the operations on them: as vectors of four where the input is reached through a pointer and the segment starts
// at an address aligned for such a vector, one element at a time otherwise. The last tile, which may be partial, is
// taken one element at a time. The work-items start from the operation on two elements, so that the accumulator is
// never made from an element alone; a tree within every sub-group and the first work-item over the sub-groups combine
// their results, and the initial value is applied once, to the result of the segment. The required commutativity of
// the operation allows the order.
template <typename _Acc, typename... _Name>
struct __segmented_reduce_block_submitter<_Acc, __internal::__optional_kernel_name<_Name...>>
{
    static constexpr std::size_t __items = __segmented_reduce_gpu_items_per_item;
    static constexpr std::size_t __vec_size = 4;

    // The work of a work-item; a function rather than a lambda called by the kernel, which would have to be device
    // copyable
    template <typename _Size, typename _InitType, typename _Op, typename _Offsets, typename _Result, typename _Input,
              typename _LocalAcc, typename _LocalHas>
    static void
    __run(const sycl::nd_item<1>& __item, _Op& __op, const _InitType& __init, _Offsets __offsets, _Result __result,
          _Input __input, const _Size __n, const _Size __m, const std::size_t __num_groups, _LocalAcc __local_acc,
          _LocalHas __local_has)
    {
        const auto __group = __item.get_group();
        const __dpl_sycl::__sub_group __sg = __item.get_sub_group();
        const _Size __lid = __item.get_local_id(0);
        const _Size __wg = __item.get_local_range(0);
        const std::uint32_t __lane = __sg.get_local_linear_id();
        const std::uint32_t __lanes = __sg.get_local_linear_range();
        const std::uint32_t __sg_id = __sg.get_group_linear_id();
        const std::uint32_t __num_sg = __sg.get_group_linear_range();
        const _Size __tile = __wg * _Size(__items);
        auto __pair = [&__op](const auto& __a, const auto& __b) -> _Acc {
            return oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(std::invoke(__op, __a, __b));
        };

        for (std::size_t __s = __item.get_group(0); __s < std::size_t(__m); __s += __num_groups)
        {
            const _Size __segment = __s;
            const _Size __begin = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment], __n);
            const _Size __end = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__segment + 1], __n);
            const _Size __length = __begin < __end ? __end - __begin : _Size{0};

            // The same for every work-item of the work-group, so none of them reaches a barrier below
            if (__length < 2)
            {
                if (__lid == 0)
                    __result[__segment] = __length == 0
                                              ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init)
                                              : __pair(__init, __input[__begin]);
                continue;
            }

            _Acc __acc{};
            bool __has = false;
            _Size __pos = __begin;
            bool __by_vectors = false;
            if constexpr (__segmented_reduce_pointer_input<_Input>::value)
            {
                using _T = std::remove_cv_t<std::remove_pointer_t<decltype(__input.begin())>>;
                if constexpr (std::is_arithmetic_v<_T> && !std::is_same_v<_T, bool> &&
                              (sizeof(_T) == 1 || sizeof(_T) == 2 || sizeof(_T) == 4 || sizeof(_T) == 8))
                {
                    using _V = sycl::vec<_T, __vec_size>;
                    const _T* const __base = __input.begin();
                    __by_vectors = reinterpret_cast<std::uintptr_t>(__base + __begin) % sizeof(_V) == 0;
                    if (__by_vectors)
                    {
                        for (; __pos + __tile <= __end; __pos += __tile)
                        {
                            // All the loads of the tile before the operations on them, the vectors of the work-items
                            // one after another
                            _V __v[__items / __vec_size];
                            for (std::size_t __j = 0; __j < __items / __vec_size; ++__j)
                                __v[__j] = *reinterpret_cast<const _V*>(
                                    __base + __pos + _Size(__vec_size) * (__lid + _Size(__j) * __wg));
                            _Acc __p[__items / 2];
                            for (std::size_t __j = 0; __j < __items / __vec_size; ++__j)
                            {
                                __p[2 * __j] = __pair(_T(__v[__j][0]), _T(__v[__j][1]));
                                __p[2 * __j + 1] = __pair(_T(__v[__j][2]), _T(__v[__j][3]));
                            }
                            for (std::size_t __w = 1; __w < __items / 2; __w *= 2)
                                for (std::size_t __j = 0; __j + __w < __items / 2; __j += 2 * __w)
                                    __p[__j] = __pair(__p[__j], __p[__j + __w]);
                            __acc = __has ? __pair(__acc, __p[0]) : __p[0];
                            __has = true;
                        }
                    }
                }
            }
            if (!__by_vectors)
            {
                using _Val = std::decay_t<decltype(__input[__begin])>;
                for (; __pos + __tile <= __end; __pos += __tile)
                {
                    // All the loads of the tile before the operations on them, the elements of the work-items one
                    // after another
                    _Val __x[__items];
                    for (std::size_t __j = 0; __j < __items; ++__j)
                        __x[__j] = __input[__pos + __lid + _Size(__j) * __wg];
                    _Acc __p[__items / 2];
                    for (std::size_t __j = 0; __j < __items / 2; ++__j)
                        __p[__j] = __pair(__x[2 * __j], __x[2 * __j + 1]);
                    for (std::size_t __w = 1; __w < __items / 2; __w *= 2)
                        for (std::size_t __j = 0; __j + __w < __items / 2; __j += 2 * __w)
                            __p[__j] = __pair(__p[__j], __p[__j + __w]);
                    __acc = __has ? __pair(__acc, __p[0]) : __p[0];
                    __has = true;
                }
            }

            // The last tile: after a whole one every work-item has a result and takes one element after another;
            // otherwise the work-items that take part get at least two elements
            if (__pos < __end)
            {
                if (__has)
                {
                    for (_Size __i = __pos + __lid; __i < __end; __i += __wg)
                        __acc = __pair(__acc, __input[__i]);
                }
                else
                {
                    const _Size __active = std::min<_Size>(__wg, (__end - __pos) / 2);
                    if (__lid < __active)
                    {
                        _Size __i = __pos + __lid;
                        __acc = __pair(__input[__i], __input[__i + __active]);
                        for (__i += 2 * __active; __i < __end; __i += __active)
                            __acc = __pair(__acc, __input[__i]);
                        __has = true;
                    }
                }
            }

            // A tree within the sub-group, whatever lanes hold a result
            for (std::uint32_t __shift = 1; __shift < __lanes; __shift *= 2)
            {
                const _Acc __other = sycl::shift_group_left(__sg, __acc, __shift);
                const bool __other_has = sycl::shift_group_left(__sg, int(__has), __shift) != 0;
                const bool __take = __lane % (2 * __shift) == 0 && __lane + __shift < __lanes && __other_has;
                if (__take && __has)
                    __acc = __pair(__acc, __other);
                if (__take && !__has)
                    __acc = __other;
                __has = __has || __take;
            }
            if (__lane == 0)
            {
                __local_acc[__sg_id] = __acc;
                __local_has[__sg_id] = __has;
            }
            sycl::group_barrier(__group);
            if (__lid == 0)
            {
                // A segment of at least two elements: some sub-group holds a result
                bool __any = false;
                _Acc __sum{};
                for (std::uint32_t __k = 0; __k < __num_sg; ++__k)
                {
                    if (__local_has[__k] && __any)
                        __sum = __pair(__sum, __local_acc[__k]);
                    if (__local_has[__k] && !__any)
                        __sum = __local_acc[__k];
                    __any = __any || __local_has[__k];
                }
                __result[__segment] = __pair(__init, __sum);
            }
            // The local memory is read for this segment before the next one writes it
            sycl::group_barrier(__group);
        }
    }

    template <typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size, typename _InitType,
              typename _BinaryOperation>
    sycl::event
    operator()(sycl::queue& __q, _InRange&& __input, _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n,
               const _Size __m, const std::size_t __work_group_size, _InitType __init,
               _BinaryOperation __binary_op) const
    {
        // One work-group per segment. A launch that would not fit the int range of the work-item indices, which a SYCL
        // implementation may assume (as DPC++ does by default), has fewer work-groups, each reducing every
        // __num_groups-th segment.
        const std::size_t __num_groups =
            std::min<std::size_t>(__m, std::size_t(std::numeric_limits<int>::max()) / __work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets, __result);
            __dpl_sycl::__local_accessor<_Acc> __local_acc(sycl::range<1>(__work_group_size), __cgh);
            __dpl_sycl::__local_accessor<bool> __local_has(sycl::range<1>(__work_group_size), __cgh);
            __cgh.parallel_for<_Name...>(
                sycl::nd_range<1>(sycl::range<1>(__num_groups * __work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item) {
                    _BinaryOperation __op = __binary_op;
                    __run(__item, __op, __init, __offsets, __result, __input, __n, __m, __num_groups, __local_acc,
                          __local_has);
                });
        });
    }
};

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

// The elements of a segment of at least two elements, taken in pairs by the bricks of the reduction. The first pair
// starts from the operation on the initial value and the first element, as in std::ranges::fold_left; every other
// pair from the operation on its two elements, as a partial result of std::reduce does, so that the accumulator is
// never made from an element alone, which may mean something else than the element to the operation. With an odd
// length the last pair also takes the last element.
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_pairs
{
    _Size __begin;           // the first element of the segment
    std::size_t __num_pairs; // the number of pairs
    bool __is_odd;           // whether the length of the segment is odd
    _InitType __init;
    _BinaryOperation __binary_op;

    template <typename _Input>
    _Acc
    operator()(const std::size_t __idx, const _Input& __input) const
    {
        const _Size __first = __begin + 2 * _Size(__idx);
        _Acc __acc = __idx == 0 ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                      std::invoke(__binary_op, __init, __input[__first]))
                                : oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                      std::invoke(__binary_op, __input[__first], __input[__first + 1]));
        if (__idx == 0)
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

template <typename... _Name>
class __segmented_reduce_group_kernel;

template <typename _Acc, typename _KernelName>
struct __segmented_reduce_group_submitter;

// One work-group per segment for the other types, through the local memory
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
        using _Pairs = __segmented_reduce_pairs<_Acc, _Size, _InitType, _Op>;
        using _TransformPattern =
            unseq_backend::transform_reduce<_Op, _Pairs, _Acc, /*_Commutative=*/std::true_type, __vector_size>;
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
                    const std::size_t __local_idx = __nd_item.get_local_id(0);
                    const __segmented_reduce_group_item __item(__nd_item);
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
                        const _Size __length = __begin < __end ? __end - __begin : _Size{0};

                        // The same for every work-item of the work-group, so none of them reaches a barrier below
                        if (__length < 2)
                        {
                            if (__local_idx == 0)
                                __result[__segment] =
                                    __length == 0 ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init)
                                                  : oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                                        std::invoke(__op, __init, __input[__begin]));
                            continue;
                        }

                        // The bricks of the reduction take unsigned sizes
                        const std::size_t __num_pairs = __length / 2;
                        const _TransformPattern __transform_pattern{
                            __op, _Pairs{__begin, __num_pairs, __length % 2 != 0, __init, __op}};
                        const _ReducePattern __reduce_pattern{__op};

                        // The iterations of a work-item as in __parallel_transform_reduce
                        const std::size_t __iters_per_work_item = __adjust_iters_per_work_item<__vector_size>(
                            oneapi::dpl::__internal::__dpl_ceiling_div(__num_pairs, std::size_t(__work_group_size)));
                        const bool __is_full = __num_pairs == __work_group_size * __iters_per_work_item;

                        // 1. The partial results of the work-items
                        oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __partial;
                        __transform_pattern(__item, __num_pairs, __iters_per_work_item,
                                            /*__global_offset*/ std::size_t{0}, __is_full,
                                            /*__n_groups*/ std::size_t{1}, __partial, __input);
                        const std::size_t __n_items = __transform_pattern.output_size(
                            __num_pairs, std::size_t(__work_group_size), __iters_per_work_item);

                        // 2. Their reduction within the work-group
                        _Acc __reduced = __reduce_pattern(__item, __n_items, __partial, __local_mem);
                        if (__local_idx < __n_items)
                        {
                            if (__local_idx == 0)
                                __result[__segment] = std::move(__reduced);
                            __partial.__destroy();
                        }
                    }
                });
        });
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

    // On GPUs one kernel, one work-group per segment
    if (__is_gpu)
    {
        sycl::queue __q = __exec.queue();
        if constexpr (std::is_arithmetic_v<_Acc> &&
                      std::is_arithmetic_v<oneapi::dpl::__internal::__value_t<std::decay_t<_InRange>>>)
        {
            using _BlockKernel = __internal::__kernel_name_provider<
                __segmented_reduce_block_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;

            const std::size_t __work_group_size =
                oneapi::dpl::__internal::__max_work_group_size(__q, __segmented_reduce_gpu_group_size);
            sycl::event __event = __segmented_reduce_block_submitter<_Acc, _BlockKernel>{}(
                __q, __input, __offsets, __result, __n, __m, __work_group_size, __init, __binary_op);
            __event.wait_and_throw();
        }
        else
        {
            using _GroupKernel = __internal::__kernel_name_provider<
                __segmented_reduce_group_kernel<oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>>>;

            // The work-group size as in __parallel_transform_reduce: adjusted to the local memory limit, which is
            // doubled to take into account the memory used by the compiled kernel, and at most 256
            const std::size_t __work_group_size = std::min<std::size_t>(
                oneapi::dpl::__internal::__slm_adjusted_work_group_size(__q, std::size_t(sizeof(_Acc) * 2)),
                __segmented_reduce_gpu_group_size);
            sycl::event __event = __segmented_reduce_group_submitter<_Acc, _GroupKernel>{}(
                __q, __input, __offsets, __result, __n, __m, static_cast<std::uint16_t>(__work_group_size), __init,
                __binary_op);
            __event.wait_and_throw();
        }
        return;
    }

    const _Size __threshold = __is_gpu ? __segmented_reduce_gpu_threshold : __segmented_reduce_cpu_threshold;

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
