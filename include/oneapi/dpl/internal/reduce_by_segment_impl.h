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

#include <cstdint>
#include <functional> // std::invoke
#include <iterator>   // std::distance
#include <type_traits>
#include <utility>

#include "by_segment_extension_defs.h"

#include "../pstl/glue_numeric_impl.h"
#include "../pstl/glue_algorithm_impl.h" // TODO: replace the use of for_each & transform with __pattern_walk3

#include "../pstl/iterator_impl.h"
#include "function.h"
#include "../pstl/utils.h"

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

template <typename _InitType, typename _InputIterator, typename _BinaryOperation>
using __segmented_reduce_acc_t =
    std::decay_t<std::invoke_result_t<_BinaryOperation&, _InitType,
                                      typename std::iterator_traits<_InputIterator>::reference>>;

// Reduces the elements [__begin, __end) of a random access iterator or a range, starting from the initial value
template <typename _Acc, typename _Source, typename _Index, typename _InitType, typename _BinaryOperation>
_Acc
__segmented_reduce_serial(const _Source& __src, _Index __begin, _Index __end, const _InitType& __init,
                          _BinaryOperation& __binary_op)
{
    _Acc __acc(__init);
    for (; __begin < __end; ++__begin)
        __acc = std::invoke(__binary_op, std::move(__acc), __src[__begin]);
    return __acc;
}

template <typename _Offset, typename _Size>
_Size
__segmented_reduce_clamp_offset(_Offset __offset, _Size __n)
{
    static_assert(std::is_integral_v<_Offset>, "The offsets of reduce_by_segment must be of an integral type");
    static_assert(std::is_integral_v<_Size>); // an internal invariant: the difference type of the input

    if constexpr (std::is_signed_v<_Offset>)
    {
        if (__offset <= _Offset{0})
            return _Size{0};
    }

    using _Unsigned = std::common_type_t<std::make_unsigned_t<_Offset>, std::make_unsigned_t<_Size>>;
    if (static_cast<_Unsigned>(__offset) >= static_cast<_Unsigned>(__n))
        return __n;
    return static_cast<_Size>(__offset);
}

template <typename _Tag, typename _Policy, typename _InputValueIt, typename _SegmentNum, typename _OffsetIt,
          typename _OutputValueIt, typename _InitType, typename _BinaryOperation>
void
__reduce_by_segment_offsets_impl(_Tag, _Policy&&, _InputValueIt __first_value, _InputValueIt __last_value,
                                 _OutputValueIt __result_value, _SegmentNum __num_segments, _OffsetIt __offsets,
                                 _InitType __init, _BinaryOperation __binary_op)
{
    using _Acc = __segmented_reduce_acc_t<_InitType, _InputValueIt, _BinaryOperation>;
    using _Size = typename std::iterator_traits<_InputValueIt>::difference_type;

    const _Size __n = __last_value - __first_value;
    for (_SegmentNum __segment = 0; __segment < __num_segments; ++__segment)
    {
        const _Size __begin = __segmented_reduce_clamp_offset(__offsets[__segment], __n);
        const _Size __end = __segmented_reduce_clamp_offset(__offsets[__segment + 1], __n);
        __result_value[__segment] = __begin < __end
                                        ? __segmented_reduce_serial<_Acc>(__first_value, __begin, __end, __init,
                                                                          __binary_op)
                                        : _Acc(__init);
    }
}

#if _ONEDPL_BACKEND_SYCL

template <typename _BackendTag, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator1, typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
std::pair<OutputIterator1, OutputIterator2>
reduce_by_segment_impl(__internal::__hetero_tag<_BackendTag> __tag, Policy&& policy, InputIterator1 first1,
                       InputIterator1 last1, InputIterator2 first2, OutputIterator1 result1, OutputIterator2 result2,
                       BinaryPred binary_pred, BinaryOperator binary_op)
{
    // The algorithm reduces values in [first2, first2 + (last1-first1)) where the associated
    // keys for the values are equal to the adjacent key.
    //
    // Example: keys          = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first1, last1)
    //          values        = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 } -- [first2, first2+n)
    //
    //          keys_result   = { 1, 2, 3, 4, 1, 3, 1, 3, 0 } -- result1
    //          values_result = { 1, 2, 3, 4, 2, 6, 2, 6, 0 } -- result2

    if (first1 == last1)
        return std::make_pair(result1, result2);

    // number of unique keys
    const auto __n = oneapi::dpl::__internal::__pattern_reduce_by_segment(
        __tag, std::forward<Policy>(policy), first1, last1, first2, result1, result2, binary_pred, binary_op);

    return std::make_pair(result1 + __n, result2 + __n);
}

// One work-item per segment: the length of a segment is not bounded, so a work-item processes only one segment
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_variable_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _Size __n; // the size of the input
    _InitType __init;
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _Offsets, typename _Result, typename _Input>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Offsets __offsets, _Result __result, _Input __input) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");

        const _Size __begin = __segmented_reduce_clamp_offset(__offsets[__idx], __n);
        const _Size __end = __segmented_reduce_clamp_offset(__offsets[__idx + 1], __n);
        std::get<0>(__result[__idx]) =
            __begin < __end ? __segmented_reduce_serial<_Acc>(__input, __begin, __end, __init, __binary_op)
                            : _Acc(__init);
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
        get<0>(__result[__idx]) = _Acc(__init);
    }
};

template <typename _Name>
class __segmented_reduce_fill_wrapper;

template <typename _Acc, typename _BackendTag, typename _Policy, typename _OutputIterator, typename _Size,
          typename _InitType>
_OutputIterator
__segmented_reduce_fill_init(__internal::__hetero_tag<_BackendTag>, _Policy&& __policy, _OutputIterator __result,
                             _Size __num_segments, const _InitType& __init)
{
    namespace __bknd = __par_backend_hetero;

    const auto __result_last = __result + __num_segments;
    auto __keep_result =
        oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::write, /*_IsNoInitRequested=*/true>();
    auto __result_buf = __keep_result(__result, __result_last);

    // A separate kernel name: the reduction kernel of the same call is instantiated too
    __bknd::__parallel_for(
        _BackendTag{}, __bknd::make_wrapped_policy<__segmented_reduce_fill_wrapper>(std::forward<_Policy>(__policy)),
        __segmented_reduce_init_brick<_Acc, _InitType>{__init}, __num_segments,
        oneapi::dpl::__ranges::make_zip_view(__result_buf.all_view()))
        .__checked_deferrable_wait();

    return __result_last;
}

template <typename _BackendTag, typename Policy, typename InputValueIt, typename SegmentNumT, typename OffsetIt,
          typename OutputValueIt, typename InitT, typename BinaryOperator>
void
__reduce_by_segment_offsets_impl(__internal::__hetero_tag<_BackendTag> __tag, Policy&& policy,
                                 InputValueIt first_value, InputValueIt last_value, OutputValueIt result_value,
                                 SegmentNumT num_segments, OffsetIt offsets, InitT init_value,
                                 BinaryOperator binary_op)
{
    namespace __bknd = __par_backend_hetero;

    if (!(num_segments > SegmentNumT{0}))
        return;

    using _Acc = __segmented_reduce_acc_t<InitT, InputValueIt, BinaryOperator>;
    const auto __result_last = result_value + num_segments;

    if (first_value == last_value)
    {
        __segmented_reduce_fill_init<_Acc>(__tag, std::forward<Policy>(policy), result_value, num_segments, init_value);
        return;
    }

    auto __keep_input = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read>();
    auto __input_buf = __keep_input(first_value, last_value);

    auto __keep_offsets = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read>();
    auto __offsets_buf = __keep_offsets(offsets, offsets + num_segments + 1);

    auto __keep_result =
        oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::write, /*_IsNoInitRequested=*/true>();
    auto __result_buf = __keep_result(result_value, __result_last);

    using _Size = typename std::iterator_traits<InputValueIt>::difference_type;
    const _Size __n = last_value - first_value;
    __bknd::__parallel_for(
        _BackendTag{}, std::forward<Policy>(policy),
        __segmented_reduce_variable_brick<_Acc, _Size, InitT, BinaryOperator>{__n, init_value, binary_op}, num_segments,
        __offsets_buf.all_view(), oneapi::dpl::__ranges::make_zip_view(__result_buf.all_view()), __input_buf.all_view())
        .__checked_deferrable_wait();
}

#endif // _ONEDPL_BACKEND_SYCL
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

// https://github.com/uxlfoundation/oneDPL/blob/71b037751ac77e6ec4f240375d1d1e9788266bf9/rfcs/proposed/segmented_reduce/README.md#synopsis
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
