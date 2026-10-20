/*  Copyright (c) Intel Corporation
 *
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 *  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _ONEDPL_REDUCE_BY_SEGMENT_IMPL_H
#define _ONEDPL_REDUCE_BY_SEGMENT_IMPL_H

#include <algorithm> // std::min, std::upper_bound
#include <atomic>    // std::atomic
#include <cstdint>
#include <functional> // std::invoke
#include <type_traits>
#include <utility>
#include <iterator> // std::distance

#include "by_segment_extension_defs.h"
#include "../pstl/segmented_reduce_utils.h"

#include "../pstl/glue_numeric_impl.h"
#include "../pstl/glue_algorithm_impl.h" // TODO: replace the use of for_each & transform with __pattern_walk3

#include "../pstl/iterator_impl.h"
#include "function.h"
#include "../pstl/utils.h"

#if _ONEDPL_BACKEND_SYCL
#    include "../pstl/hetero/reduce_by_segment_impl_hetero.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace internal
{

template <typename Name>
class Reduce1;
template <typename Name>
class Reduce2;
template <typename Name>
class Reduce3;
template <typename Name>
class Reduce4;

template <class _Tag, typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
std::pair<OutputIterator1, OutputIterator2>
reduce_by_segment_impl(_Tag, Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                       OutputIterator1 result1, OutputIterator2 result2, BinaryPred binary_pred,
                       BinaryOperator binary_op)
{
    static_assert(__internal::__is_host_dispatch_tag_v<_Tag>);

    // The algorithm reduces values in [first2, first2 + (last1-first1)) where the associated
    // keys for the values are equal to the adjacent key. This function's implementation is a derivative work
    // and responsible for the second copyright notice in this header.
    //
    // Example: keys          = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first1, last1)
    //          values        = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first2, first2+n)
    //
    //          keys_result   = { 1, 2, 3, 4, 1, 3, 1, 3, 0 } -- result1
    //          values_result = { 1, 2, 3, 4, 2, 6, 2, 6, 0 } -- result2

    const auto n = std::distance(first1, last1);

    if (n <= 0)
        return std::make_pair(result1, result2);
    else if (n == 1)
    {
        *result1 = *first1;
        *result2 = *first2;
        return std::make_pair(result1 + 1, result2 + 1);
    }

    using FlagType = std::uint64_t;
    using ValueType = typename std::iterator_traits<InputIterator2>::value_type;
    using CountType = std::uint64_t;

    // buffer that is used to store a flag indicating if the associated key is not equal to
    // the next key, and thus its associated sum should be part of the final result
    oneapi::dpl::__par_backend::__buffer<FlagType> _mask(n + 1);
    auto mask = _mask.get();
    mask[0] = 1;

    // instead of copying mask, use shifted sequence:
    mask[n] = 1;

    // Identify where the first key in a sequence of equivalent keys is located
    oneapi::dpl::transform(policy, first1, last1 - 1, first1 + 1, _mask.get() + 1,
                           oneapi::dpl::__internal::__not_pred<BinaryPred>(binary_pred));

    // for example: _mask = { 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1}

    // buffer stores the sums of values associated with a given key. Sums are copied with
    // a shift into result2, and the shift is computed at the same time as the sums, so the
    // sums can't be written to result2 directly.
    oneapi::dpl::__par_backend::__buffer<ValueType> _scanned_values(n);

    // Buffer is used to store results of the scan of the mask. Values indicate which position
    // in result2 needs to be written with the scanned_values element.
    oneapi::dpl::__par_backend::__buffer<FlagType> _scanned_tail_flags(n);

    // Compute the sum of the segments. scanned_tail_flags values are not used.
    oneapi::dpl::inclusive_scan(
        policy, make_zip_iterator(first2, _mask.get()), make_zip_iterator(first2, _mask.get()) + n,
        make_zip_iterator(_scanned_values.get(), _scanned_tail_flags.get()),
        oneapi::dpl::__internal::__segmented_scan_fun<ValueType, FlagType, BinaryOperator>{binary_op});

    // for example: _scanned_values     = { 1, 2, 3, 4, 1, 2, 3, 6, 1, 2, 3, 6, 0 }

    // Compute the indices each segment sum should be written
    oneapi::dpl::exclusive_scan(policy, _mask.get() + 1, _mask.get() + n + 1, _scanned_tail_flags.get(), CountType(0),
                                std::plus<CountType>());

    // for example: _scanned_tail_flags = { 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 }

    auto scanned_tail_flags = _scanned_tail_flags.get();
    auto scanned_values = _scanned_values.get();

    // number of unique segments
    CountType N = scanned_tail_flags[n - 1] + 1;

    // scatter the keys and accumulated values
    oneapi::dpl::for_each(std::forward<Policy>(policy),
                          make_zip_iterator(first1, scanned_tail_flags, mask, scanned_values, mask + 1),
                          make_zip_iterator(first1, scanned_tail_flags, mask, scanned_values, mask + 1) + n,
                          __internal::scatter_and_accumulate_fun<OutputIterator1, OutputIterator2>(result1, result2));

    // for example: result1 = {1, 2, 3, 4, 1, 3, 1, 3, 0}
    // for example: result2 = {1, 2, 3, 4, 2, 6, 2, 6, 0}

    return std::make_pair(result1 + N, result2 + N);
}

//------------------------------------------------------------------------
// reduce_by_segment with segments defined by offsets or by a fixed length
//------------------------------------------------------------------------

// Reduces the elements [__begin, __end) of a random access iterator on the host as __segmented_reduce_serial does
template <typename _Acc, typename _RandomAccessIterator, typename _Index, typename _InitType, typename _BinaryOperation>
_Acc
__segmented_reduce_host(_RandomAccessIterator __first, _Index __begin, _Index __end, _InitType __init,
                        _BinaryOperation& __binary_op, /*is_vector=*/std::false_type)
{
    return __internal::__segmented_reduce_serial<_Acc>(__first, __begin, __end, std::move(__init), __binary_op);
}

// The same, vectorized: the vectorized reduction also starts its lanes from the operation on two elements
template <typename _Acc, typename _RandomAccessIterator, typename _Index, typename _InitType, typename _BinaryOperation>
_Acc
__segmented_reduce_host(_RandomAccessIterator __first, _Index __begin, _Index __end, _InitType __init,
                        _BinaryOperation& __binary_op, /*is_vector=*/std::true_type)
{
    if (__begin >= __end)
        return _Acc(std::move(__init));
    _Acc __start(std::invoke(__binary_op, std::move(__init), __first[__begin]));
    return __internal::__brick_transform_reduce(__first + __begin + 1, __first + __end, std::move(__start), __binary_op,
                                                oneapi::dpl::identity{}, /*is_vector=*/std::true_type{});
}

// Reduces the __count > 0 elements from __it as std::ranges::fold_left does, and moves __it past them
template <typename _Acc, typename _ForwardIterator, typename _Size, typename _InitType, typename _BinaryOperation>
_Acc
__segmented_reduce_forward(_ForwardIterator& __it, _Size __count, _InitType __init, _BinaryOperation& __binary_op)
{
    _Acc __acc(std::invoke(__binary_op, std::move(__init), *__it));
    for (++__it; --__count > 0; ++__it)
        __acc = std::invoke(__binary_op, std::move(__acc), *__it);
    return __acc;
}

// The segments one after another in the order of the offsets and the results, with one pass over the input: the input
// before the first segment is skipped once, then the iterator only moves with the reduction, as a segment of sorted
// offsets starts where the previous one ends. Unsorted offsets, whose results are unspecified, may give a segment
// starting before the current position: it is reduced from the current position.
template <typename _Acc, typename _ForwardInputIt, typename _SegmentNum, typename _ForwardOffsetIt,
          typename _ForwardOutputIt, typename _InitType, typename _BinaryOperation>
void
__brick_reduce_by_segment_offsets(_ForwardInputIt __first_value, _ForwardInputIt __last_value,
                                  _ForwardOutputIt __result_value, _SegmentNum __num_segments,
                                  _ForwardOffsetIt __offsets, _InitType __init, _BinaryOperation& __binary_op,
                                  /*is_vector=*/std::false_type)
{
    using _Size = typename std::iterator_traits<_ForwardInputIt>::difference_type;

    if (__num_segments <= 0)
        return;
    const _Size __n = std::distance(__first_value, __last_value);
    _Size __pos = __internal::__segmented_reduce_clamp_offset(*__offsets, __n); // the position of __it
    _ForwardInputIt __it = std::next(__first_value, __pos);
    for (_SegmentNum __segment = 0; __segment < __num_segments; ++__segment)
    {
        ++__offsets;
        const _Size __end = __internal::__segmented_reduce_clamp_offset(*__offsets, __n);
        if (__pos >= __end)
        {
            *__result_value = _Acc(__init);
        }
        else
        {
            *__result_value = __segmented_reduce_forward<_Acc>(__it, __end - __pos, __init, __binary_op);
            __pos = __end;
        }
        ++__result_value;
    }
}

// The segments by their indices, each one vectorized; the tag asks for vectorization only with random access iterators
template <typename _Acc, typename _RandomAccessInputIt, typename _SegmentNum, typename _RandomAccessOffsetIt,
          typename _RandomAccessOutputIt, typename _InitType, typename _BinaryOperation>
void
__brick_reduce_by_segment_offsets(_RandomAccessInputIt __first_value, _RandomAccessInputIt __last_value,
                                  _RandomAccessOutputIt __result_value, _SegmentNum __num_segments,
                                  _RandomAccessOffsetIt __offsets, _InitType __init, _BinaryOperation& __binary_op,
                                  /*is_vector=*/std::true_type)
{
    using _Size = typename std::iterator_traits<_RandomAccessInputIt>::difference_type;

    const _Size __n = __last_value - __first_value;
    for (_SegmentNum __segment = 0; __segment < __num_segments; ++__segment)
    {
        __result_value[__segment] = __segmented_reduce_host<_Acc>(
            __first_value, __internal::__segmented_reduce_clamp_offset(__offsets[__segment], __n),
            __internal::__segmented_reduce_clamp_offset(__offsets[__segment + 1], __n), __init, __binary_op,
            /*is_vector=*/std::true_type{});
    }
}

// The serial host policies: the segments one after another, vectorized when the tag asks for vectorization, which unseq
// does only with random access iterators
template <typename _IsVector, typename _Policy, typename _InputValueIt, typename _SegmentNum, typename _OffsetIt,
          typename _OutputValueIt, typename _InitType, typename _BinaryOperation>
void
__reduce_by_segment_offsets_impl(__internal::__serial_tag<_IsVector>, _Policy&&, _InputValueIt __first_value,
                                 _InputValueIt __last_value, _OutputValueIt __result_value, _SegmentNum __num_segments,
                                 _OffsetIt __offsets, _InitType __init, _BinaryOperation __binary_op)
{
    using _Acc = __internal::__segmented_reduce_acc_t<_InitType, _InputValueIt, _BinaryOperation>;

    __brick_reduce_by_segment_offsets<_Acc>(__first_value, __last_value, __result_value, __num_segments, __offsets,
                                            __init, __binary_op, _IsVector{});
}

// The parallel host policies with iterators which are not random access ones: the segments one after another
template <typename _Policy, typename _ForwardInputIt, typename _SegmentNum, typename _ForwardOffsetIt,
          typename _ForwardOutputIt, typename _InitType, typename _BinaryOperation>
void
__reduce_by_segment_offsets_impl(__internal::__parallel_forward_tag, _Policy&&, _ForwardInputIt __first_value,
                                 _ForwardInputIt __last_value, _ForwardOutputIt __result_value,
                                 _SegmentNum __num_segments, _ForwardOffsetIt __offsets, _InitType __init,
                                 _BinaryOperation __binary_op)
{
    using _Acc = __internal::__segmented_reduce_acc_t<_InitType, _ForwardInputIt, _BinaryOperation>;

    __brick_reduce_by_segment_offsets<_Acc>(__first_value, __last_value, __result_value, __num_segments, __offsets,
                                            __init, __binary_op, /*is_vector=*/std::false_type{});
}

// The parallel host policies reduce a short segment in one task, and split a longer one into chunks reduced in
// parallel, which balances the load when a few segments hold most of the input.
// - a segment up to this length is short;
inline constexpr std::size_t __segmented_reduce_host_threshold = 32768;
// - the length of a chunk of a long segment.
inline constexpr std::size_t __segmented_reduce_host_chunk_size = 16384;

template <typename _IsVector, typename _ExecutionPolicy, typename _RandomAccessInputIt, typename _SegmentNum,
          typename _RandomAccessOffsetIt, typename _RandomAccessOutputIt, typename _InitType, typename _BinaryOperation>
void
__reduce_by_segment_offsets_impl(__internal::__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec,
                                 _RandomAccessInputIt __first_value, _RandomAccessInputIt __last_value,
                                 _RandomAccessOutputIt __result_value, _SegmentNum __num_segments,
                                 _RandomAccessOffsetIt __offsets, _InitType __init, _BinaryOperation __binary_op)
{
    using __backend_tag = typename __internal::__parallel_tag<_IsVector>::__backend_tag;
    using _Acc = __internal::__segmented_reduce_acc_t<_InitType, _RandomAccessInputIt, _BinaryOperation>;
    using _Size = typename std::iterator_traits<_RandomAccessInputIt>::difference_type;

    if (__num_segments <= 0)
        return;

    const _Size __n = __last_value - __first_value;
    const _Size __m = __num_segments;
    const _Size __threshold = __segmented_reduce_host_threshold;
    const _Size __chunk_size = __segmented_reduce_host_chunk_size;
    auto __bounds = [__offsets, __n](_Size __segment) {
        return std::make_pair(__internal::__segmented_reduce_clamp_offset(__offsets[__segment], __n),
                              __internal::__segmented_reduce_clamp_offset(__offsets[__segment + 1], __n));
    };

    // Every __parallel_for below has a grain of one segment or chunk, as the OpenMP backend would otherwise run up to
    // 2048 of them in one task
    __internal::__except_handler([&]() {
        // The short segments, in parallel over the segments; the long ones, at most n / __threshold of them unless the
        // offsets are not ordered, are only listed. A long segment beyond the list is reduced as a short one.
        const _Size __capacity = __n / __threshold + 1;
        oneapi::dpl::__par_backend::__buffer<_Size> __long_buf(__capacity);
        _Size* __long = __long_buf.get();
        std::atomic<_Size> __num_listed{0};
        oneapi::dpl::__par_backend::__parallel_for(
            __backend_tag{}, __exec, _Size{0}, __m,
            [&](_Size __segment_first, _Size __segment_last) {
                for (_Size __segment = __segment_first; __segment < __segment_last; ++__segment)
                {
                    const auto [__begin, __end] = __bounds(__segment);
                    if (__end - __begin > __threshold)
                    {
                        const _Size __k = __num_listed.fetch_add(1, std::memory_order_relaxed);
                        if (__k < __capacity)
                        {
                            __long[__k] = __segment;
                            continue;
                        }
                    }
                    __result_value[__segment] =
                        __segmented_reduce_host<_Acc>(__first_value, __begin, __end, __init, __binary_op, _IsVector{});
                }
            },
            /*__grainsize=*/1);
        const _Size __num_long = std::min(__num_listed.load(), __capacity);
        if (__num_long == 0)
            return;

        // The end of the chunks of every long segment: a serial scan of their numbers
        oneapi::dpl::__par_backend::__buffer<_Size> __chunk_ends_buf(__num_long);
        _Size* __chunk_ends = __chunk_ends_buf.get();
        _Size __num_chunks = 0;
        for (_Size __k = 0; __k < __num_long; ++__k)
        {
            const auto [__begin, __end] = __bounds(__long[__k]);
            __num_chunks += (__end - __begin + __chunk_size - 1) / __chunk_size;
            __chunk_ends[__k] = __num_chunks;
        }

        // The chunks in parallel. The first chunk of a segment takes the remainder of its length and starts from the
        // operation on the initial value and its first element; every other chunk has exactly __chunk_size elements
        // and starts from the operation on its first two elements.
        oneapi::dpl::__par_backend::__buffer<_Acc> __partials_buf(__num_chunks);
        _Acc* __partials = __partials_buf.get();
        oneapi::dpl::__par_backend::__parallel_for(
            __backend_tag{}, __exec, _Size{0}, __num_chunks,
            [&](_Size __chunk_first, _Size __chunk_last) {
                for (_Size __chunk = __chunk_first; __chunk < __chunk_last; ++__chunk)
                {
                    const _Size __k = std::upper_bound(__chunk_ends, __chunk_ends + __num_long, __chunk) - __chunk_ends;
                    const _Size __first_chunk = __k == 0 ? _Size{0} : __chunk_ends[__k - 1];
                    const auto [__begin, __end] = __bounds(__long[__k]);
                    const _Size __head = (__end - __begin) - (__chunk_ends[__k] - __first_chunk - 1) * __chunk_size;
                    const _Size __j = __chunk - __first_chunk;
                    if (__j == 0)
                    {
                        ::new (__partials + __chunk) _Acc(__segmented_reduce_host<_Acc>(
                            __first_value, __begin, __begin + __head, __init, __binary_op, _IsVector{}));
                    }
                    else
                    {
                        const _Size __chunk_begin = __begin + __head + (__j - 1) * __chunk_size;
                        ::new (__partials + __chunk) _Acc(__segmented_reduce_host<_Acc>(
                            __first_value, __chunk_begin + 2, __chunk_begin + __chunk_size,
                            _Acc(std::invoke(__binary_op, __first_value[__chunk_begin],
                                             __first_value[__chunk_begin + 1])),
                            __binary_op, _IsVector{}));
                    }
                }
            },
            /*__grainsize=*/1);

        // The partial results of every long segment, combined in order
        oneapi::dpl::__par_backend::__parallel_for(
            __backend_tag{}, __exec, _Size{0}, __num_long,
            [&](_Size __k_first, _Size __k_last) {
                for (_Size __k = __k_first; __k < __k_last; ++__k)
                {
                    const _Size __first_chunk = __k == 0 ? _Size{0} : __chunk_ends[__k - 1];
                    _Acc __acc = std::move(__partials[__first_chunk]);
                    for (_Size __chunk = __first_chunk + 1; __chunk < __chunk_ends[__k]; ++__chunk)
                        __acc = std::invoke(__binary_op, std::move(__acc), std::move(__partials[__chunk]));
                    __result_value[__long[__k]] = std::move(__acc);
                }
            },
            /*__grainsize=*/1);
        if constexpr (!std::is_trivially_destructible_v<_Acc>)
        {
            for (_Size __chunk = 0; __chunk < __num_chunks; ++__chunk)
                __partials[__chunk].~_Acc();
        }
    });
}

} // namespace internal

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<
    Policy, std::enable_if_t<!std::is_integral_v<OutputIterator1>, std::pair<OutputIterator1, OutputIterator2>>>
reduce_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                  OutputIterator1 result1, OutputIterator2 result2, BinaryPred binary_pred, BinaryOperator binary_op)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(policy, first1, first2, result1, result2);

    return internal::reduce_by_segment_impl(__dispatch_tag, std::forward<Policy>(policy), first1, last1, first2,
                                            result1, result2, binary_pred, binary_op);
}

template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt1, typename OutputIt2,
          typename BinaryPred>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, std::pair<OutputIt1, OutputIt2>>
reduce_by_segment(Policy&& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt1 result1,
                  OutputIt2 result2, BinaryPred binary_pred)
{
    using T = typename std::iterator_traits<InputIt2>::value_type;

    return reduce_by_segment(std::forward<Policy>(policy), first1, last1, first2, result1, result2, binary_pred,
                             std::plus<T>());
}

template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt1, typename OutputIt2>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, std::pair<OutputIt1, OutputIt2>>
reduce_by_segment(Policy&& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt1 result1,
                  OutputIt2 result2)
{
    using T = typename std::iterator_traits<InputIt1>::value_type;

    return reduce_by_segment(std::forward<Policy>(policy), first1, last1, first2, result1, result2, std::equal_to<T>());
}

template <typename Policy, typename InputIt, typename OutputIt, typename SegmentNumT, typename OffsetIt, typename InitT,
          typename BinaryOp>
oneapi::dpl::__internal::__enable_if_execution_policy<
    Policy, std::enable_if_t<std::is_integral_v<SegmentNumT> && !std::is_integral_v<OffsetIt>>>
reduce_by_segment(Policy&& policy, InputIt first, InputIt last, OutputIt out_first, SegmentNumT num_segments,
                  OffsetIt offsets, InitT init_value, BinaryOp binary_op)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(policy, first, offsets, out_first);

    internal::__reduce_by_segment_offsets_impl(__dispatch_tag, std::forward<Policy>(policy), first, last, out_first,
                                               num_segments, offsets, init_value, binary_op);
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, std::pair<OutputIterator1, OutputIterator2>>
reduce_by_key(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
              OutputIterator1 result1, OutputIterator2 result2, BinaryPred binary_pred, BinaryOperator binary_op)
{
    return reduce_by_segment(std::forward<Policy>(policy), first1, last1, first2, result1, result2, binary_pred,
                             binary_op);
}

template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt1, typename OutputIt2,
          typename BinaryPred>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, std::pair<OutputIt1, OutputIt2>>
reduce_by_key(Policy&& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt1 result1, OutputIt2 result2,
              BinaryPred binary_pred)
{
    return reduce_by_segment(std::forward<Policy>(policy), first1, last1, first2, result1, result2, binary_pred);
}

template <typename Policy, typename InputIt1, typename InputIt2, typename OutputIt1, typename OutputIt2>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, std::pair<OutputIt1, OutputIt2>>
reduce_by_key(Policy&& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt1 result1, OutputIt2 result2)
{
    return reduce_by_segment(std::forward<Policy>(policy), first1, last1, first2, result1, result2);
}
} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_REDUCE_BY_SEGMENT_IMPL_H
