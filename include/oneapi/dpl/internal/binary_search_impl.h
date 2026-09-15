/*
 *  Copyright (c) Intel Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef _ONEDPL_BINARY_SEARCH_IMPL_H
#define _ONEDPL_BINARY_SEARCH_IMPL_H

#include "function.h"
#include "binary_search_extension_defs.h"
#include "../pstl/iterator_impl.h"
#include "../pstl/utils.h"

namespace oneapi
{
namespace dpl
{

namespace internal
{

enum class search_algorithm
{
    lower_bound,
    upper_bound,
    binary_search
};

#if _ONEDPL_BACKEND_SYCL

// Searches a work item keeps in flight at once, per index width. Each in-flight search carries a few
// registers of per-lane state and 64-bit indexing doubles that; measured on PVC, a fourth 64-bit
// search is what makes IGC select the large-GRF mode, which halves resident threads and gives back the
// parallelism the interleaving buys. Both index widths are compiled into one kernel and the GRF mode
// is chosen for the kernel, so the 64-bit bound has to stay low even where only 32-bit indexing runs.
// Check if overridden for testing
#ifndef _ONEDPL_BINARY_SEARCH_IN_FLIGHT_32
#    define _ONEDPL_BINARY_SEARCH_IN_FLIGHT_32 4
#endif
#ifndef _ONEDPL_BINARY_SEARCH_IN_FLIGHT_64
#    define _ONEDPL_BINARY_SEARCH_IN_FLIGHT_64 2
#endif

template <typename Comp, typename T, search_algorithm func>
struct __custom_brick
{
    Comp comp;
    T size;
    bool use_32bit_indexing;

    __custom_brick(Comp comp, T size, bool use_32bit_indexing)
        : comp(std::move(comp)), size(size), use_32bit_indexing(use_32bit_indexing)
    {
    }

    template <typename _Size, typename _ItemId, typename _Acc>
    void
    search_impl(_ItemId idx, _Acc acc) const
    {
        _Size start_orig = 0;
        _Size end_orig = size;
        using std::get;
        if constexpr (func == search_algorithm::lower_bound)
        {
            get<2>(acc[idx]) = oneapi::dpl::__internal::__shars_lower_bound(get<0>(acc.base()), start_orig, end_orig,
                                                                            get<1>(acc[idx]), comp);
        }
        else if constexpr (func == search_algorithm::upper_bound)
        {
            get<2>(acc[idx]) = oneapi::dpl::__internal::__shars_upper_bound(get<0>(acc.base()), start_orig, end_orig,
                                                                            get<1>(acc[idx]), comp);
        }
        else
        {
            auto value = oneapi::dpl::__internal::__shars_lower_bound(get<0>(acc.base()), start_orig, end_orig,
                                                                      get<1>(acc[idx]), comp);
            get<2>(acc[idx]) = (value != end_orig) && (get<1>(acc[idx]) == get<0>(acc[value]));
        }
    }
    template <typename _IsFull, typename _Params, typename _Acc>
    void
    operator()(_IsFull, const std::size_t idx, _Params, _Acc acc) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");
        if (use_32bit_indexing)
            search_impl<std::uint32_t>(idx, acc);
        else
            search_impl<std::uint64_t>(idx, acc);
    }

    // Each search is a chain of dependent probes, so running a work item's searches one after another
    // leaves one load outstanding at a time. Take the indices as a batch instead and interleave the
    // searches, which issues _C probes per round without changing which probes are performed.
    static constexpr bool __batched = true;
    static constexpr std::uint8_t max_in_flight_32 = _ONEDPL_BINARY_SEARCH_IN_FLIGHT_32;
    static constexpr std::uint8_t max_in_flight_64 = _ONEDPL_BINARY_SEARCH_IN_FLIGHT_64;

    template <typename _Size, std::size_t _C, typename _IsFull, typename _Acc>
    void
    search_batch(_IsFull, std::size_t bound, std::size_t idx, std::uint16_t stride, _Acc acc) const
    {
        using std::get;
        auto haystack = get<0>(acc.base());
        using _KeyType = std::decay_t<decltype(get<1>(acc[idx]))>;

        // Every probe index Shar's algorithm forms lies within [0, size), so a lane whose key is past
        // the end can safely repeat the last in-range search; only in-range lanes store a result.
        auto key_index = [=](std::size_t j) {
            const std::size_t i = idx + j * stride;
            if constexpr (_IsFull::value)
                return i;
            else
                return std::min(i, bound - 1);
        };

        _KeyType value[_C];
        _Size result[_C];
        _ONEDPL_PRAGMA_UNROLL
        for (std::size_t j = 0; j < _C; ++j)
            value[j] = get<1>(acc[key_index(j)]);

        const _Size start_orig = 0;
        const _Size end_orig = size;
        if constexpr (func == search_algorithm::upper_bound)
            oneapi::dpl::__internal::__shars_upper_bound_batched<_C>(haystack, start_orig, end_orig, value, result,
                                                                     comp);
        else
            oneapi::dpl::__internal::__shars_lower_bound_batched<_C>(haystack, start_orig, end_orig, value, result,
                                                                     comp);

        _ONEDPL_PRAGMA_UNROLL
        for (std::size_t j = 0; j < _C; ++j)
        {
            if (_IsFull::value || idx + j * stride < bound)
            {
                if constexpr (func == search_algorithm::binary_search)
                    get<2>(acc[key_index(j)]) = (result[j] != end_orig) && (value[j] == haystack[result[j]]);
                else
                    get<2>(acc[key_index(j)]) = result[j];
            }
        }
    }

    template <typename _Size, std::uint8_t _NumStrides, std::uint8_t _MaxInFlight, typename _IsFull, typename _Acc>
    void
    search_rounds(_IsFull, std::size_t bound, std::size_t idx, std::uint16_t stride, _Acc acc) const
    {
        constexpr std::size_t batch = std::min<std::size_t>(_NumStrides, _MaxInFlight);
        constexpr std::size_t full_batches = _NumStrides / batch;
        constexpr std::size_t tail = _NumStrides % batch;
        _ONEDPL_PRAGMA_UNROLL
        for (std::size_t b = 0; b < full_batches; ++b)
            search_batch<_Size, batch>(_IsFull{}, bound, idx + b * batch * stride, stride, acc);
        if constexpr (tail > 0)
            search_batch<_Size, tail>(_IsFull{}, bound, idx + full_batches * batch * stride, stride, acc);
    }

    template <std::uint8_t _NumStrides, typename _IsFull, typename _Params, typename _Acc>
    void
    __execute_batch(_IsFull, std::size_t bound, std::size_t idx, std::uint16_t stride, _Params, _Acc acc) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");
        if (use_32bit_indexing)
            search_rounds<std::uint32_t, _NumStrides, max_in_flight_32>(_IsFull{}, bound, idx, stride, acc);
        else
            search_rounds<std::uint64_t, _NumStrides, max_in_flight_64>(_IsFull{}, bound, idx, stride, acc);
    }
};
#endif

template <typename InputIterator, typename StrictWeakOrdering, typename _ValueType>
struct __lower_bound_impl_fn
{
    InputIterator start;
    InputIterator end;
    StrictWeakOrdering comp;

    auto
    operator()(const _ValueType& val) const
    {
        return std::lower_bound(start, end, val, comp) - start;
    }
};

template <class _Tag, typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator
lower_bound_impl(_Tag tag, Policy&& policy, InputIterator1 start, InputIterator1 end, InputIterator2 value_start,
                 InputIterator2 value_end, OutputIterator result, StrictWeakOrdering comp)
{
    static_assert(__internal::__is_host_dispatch_tag_v<_Tag>);

    using _ValueType = typename std::iterator_traits<InputIterator2>::value_type;

    return oneapi::dpl::__internal::__pattern_walk2(
        tag, std::forward<Policy>(policy), value_start, value_end, result,
        oneapi::dpl::__internal::__transform_functor{
            __lower_bound_impl_fn<InputIterator1, StrictWeakOrdering, _ValueType>{start, end, comp}});
}

template <typename InputIterator, typename StrictWeakOrdering, typename _ValueType>
struct __upper_bound_impl_fn
{
    InputIterator start;
    InputIterator end;
    StrictWeakOrdering comp;

    auto
    operator()(const _ValueType& val) const
    {
        return std::upper_bound(start, end, val, comp) - start;
    }
};

template <class _Tag, typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator
upper_bound_impl(_Tag tag, Policy&& policy, InputIterator1 start, InputIterator1 end, InputIterator2 value_start,
                 InputIterator2 value_end, OutputIterator result, StrictWeakOrdering comp)
{
    static_assert(__internal::__is_host_dispatch_tag_v<_Tag>);

    using _ValueType = typename std::iterator_traits<InputIterator2>::value_type;

    return oneapi::dpl::__internal::__pattern_walk2(
        tag, std::forward<Policy>(policy), value_start, value_end, result,
        oneapi::dpl::__internal::__transform_functor{
            __upper_bound_impl_fn<InputIterator1, StrictWeakOrdering, _ValueType>{start, end, comp}});
}

template <typename InputIterator, typename StrictWeakOrdering, typename _ValueType>
struct __binary_search_impl_fn
{
    InputIterator start;
    InputIterator end;
    StrictWeakOrdering comp;

    auto
    operator()(const _ValueType& val) const
    {
        return std::binary_search(start, end, val, comp);
    }
};

template <class _Tag, typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator
binary_search_impl(_Tag tag, Policy&& policy, InputIterator1 start, InputIterator1 end, InputIterator2 value_start,
                   InputIterator2 value_end, OutputIterator result, StrictWeakOrdering comp)
{
    static_assert(__internal::__is_host_dispatch_tag_v<_Tag>);

    using _ValueType = typename std::iterator_traits<InputIterator2>::value_type;

    return oneapi::dpl::__internal::__pattern_walk2(
        tag, std::forward<Policy>(policy), value_start, value_end, result,
        oneapi::dpl::__internal::__transform_functor{
            __binary_search_impl_fn<InputIterator1, StrictWeakOrdering, _ValueType>{start, end, comp}});
}

#if _ONEDPL_BACKEND_SYCL
template <typename _BackendTag, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename StrictWeakOrdering>
OutputIterator
lower_bound_impl(__internal::__hetero_tag<_BackendTag>, Policy&& policy, InputIterator1 start, InputIterator1 end,
                 InputIterator2 value_start, InputIterator2 value_end, OutputIterator result, StrictWeakOrdering comp)
{
    namespace __bknd = __par_backend_hetero;
    const auto size = ::std::distance(start, end);

    if (size <= 0)
        return result;

    const auto value_size = std::distance(value_start, value_end);

    auto keep_input = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read>();
    auto input_buf = keep_input(start, end);

    auto keep_values = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read>();
    auto value_buf = keep_values(value_start, value_end);

    auto keep_result =
        oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::write, /*_IsNoInitRequested=*/true>();
    auto result_buf = keep_result(result, result + value_size);
    auto zip_vw = make_zip_view(input_buf.all_view(), value_buf.all_view(), result_buf.all_view());
    const bool use_32bit_indexing = size <= std::numeric_limits<std::uint32_t>::max();
    __bknd::__parallel_for(_BackendTag{}, std::forward<decltype(policy)>(policy),
                           __custom_brick<StrictWeakOrdering, decltype(size), search_algorithm::lower_bound>{
                               comp, size, use_32bit_indexing},
                           value_size, zip_vw)
        .__checked_deferrable_wait();
    return result + value_size;
}

template <typename _BackendTag, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename StrictWeakOrdering>
OutputIterator
upper_bound_impl(__internal::__hetero_tag<_BackendTag>, Policy&& policy, InputIterator1 start, InputIterator1 end,
                 InputIterator2 value_start, InputIterator2 value_end, OutputIterator result, StrictWeakOrdering comp)
{
    namespace __bknd = __par_backend_hetero;
    const auto size = std::distance(start, end);

    if (size <= 0)
        return result;

    const auto value_size = std::distance(value_start, value_end);

    auto keep_input = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read>();
    auto input_buf = keep_input(start, end);

    auto keep_values = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read>();
    auto value_buf = keep_values(value_start, value_end);

    auto keep_result =
        oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::write, /*_IsNoInitRequested=*/true>();
    auto result_buf = keep_result(result, result + value_size);
    auto zip_vw = make_zip_view(input_buf.all_view(), value_buf.all_view(), result_buf.all_view());
    const bool use_32bit_indexing = size <= std::numeric_limits<std::uint32_t>::max();
    __bknd::__parallel_for(_BackendTag{}, std::forward<decltype(policy)>(policy),
                           __custom_brick<StrictWeakOrdering, decltype(size), search_algorithm::upper_bound>{
                               comp, size, use_32bit_indexing},
                           value_size, zip_vw)
        .__checked_deferrable_wait();
    return result + value_size;
}

template <typename _BackendTag, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename StrictWeakOrdering>
OutputIterator
binary_search_impl(__internal::__hetero_tag<_BackendTag>, Policy&& policy, InputIterator1 start, InputIterator1 end,
                   InputIterator2 value_start, InputIterator2 value_end, OutputIterator result, StrictWeakOrdering comp)
{
    namespace __bknd = __par_backend_hetero;
    const auto size = ::std::distance(start, end);

    if (size <= 0)
        return result;

    const auto value_size = std::distance(value_start, value_end);

    auto keep_input = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read>();
    auto input_buf = keep_input(start, end);

    auto keep_values = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read>();
    auto value_buf = keep_values(value_start, value_end);

    auto keep_result =
        oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::write, /*_IsNoInitRequested=*/true>();
    auto result_buf = keep_result(result, result + value_size);
    auto zip_vw = make_zip_view(input_buf.all_view(), value_buf.all_view(), result_buf.all_view());
    const bool use_32bit_indexing = size <= std::numeric_limits<std::uint32_t>::max();
    __bknd::__parallel_for(_BackendTag{}, std::forward<decltype(policy)>(policy),
                           __custom_brick<StrictWeakOrdering, decltype(size), search_algorithm::binary_search>{
                               comp, size, use_32bit_indexing},
                           value_size, zip_vw)
        .__checked_deferrable_wait();
    return result + value_size;
}

#endif
} // namespace internal

//Lower Bound start
template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
lower_bound(Policy&& policy, InputIterator1 start, InputIterator1 end, InputIterator2 value_start,
            InputIterator2 value_end, OutputIterator result)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(policy, start, value_start, result);

    return internal::lower_bound_impl(__dispatch_tag, ::std::forward<Policy>(policy), start, end, value_start,
                                      value_end, result, oneapi::dpl::__internal::__pstl_less());
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename StrictWeakOrdering>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
lower_bound(Policy&& policy, InputIterator1 start, InputIterator1 end, InputIterator2 value_start,
            InputIterator2 value_end, OutputIterator result, StrictWeakOrdering comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(policy, start, value_start, result);

    return internal::lower_bound_impl(__dispatch_tag, ::std::forward<Policy>(policy), start, end, value_start,
                                      value_end, result, comp);
}
//Lower Bound end

//Upper Bound start

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
upper_bound(Policy&& policy, InputIterator1 start, InputIterator1 end, InputIterator2 value_start,
            InputIterator2 value_end, OutputIterator result)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(policy, start, value_start, result);

    return internal::upper_bound_impl(__dispatch_tag, ::std::forward<Policy>(policy), start, end, value_start,
                                      value_end, result, oneapi::dpl::__internal::__pstl_less());
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename StrictWeakOrdering>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
upper_bound(Policy&& policy, InputIterator1 start, InputIterator1 end, InputIterator2 value_start,
            InputIterator2 value_end, OutputIterator result, StrictWeakOrdering comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(policy, start, value_start, result);

    return internal::upper_bound_impl(__dispatch_tag, ::std::forward<Policy>(policy), start, end, value_start,
                                      value_end, result, comp);
}

//Upper Bound end

//Binary Search start

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
binary_search(Policy&& policy, InputIterator1 start, InputIterator1 end, InputIterator2 value_start,
              InputIterator2 value_end, OutputIterator result)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(policy, start, value_start, result);

    return internal::binary_search_impl(__dispatch_tag, ::std::forward<Policy>(policy), start, end, value_start,
                                        value_end, result, oneapi::dpl::__internal::__pstl_less());
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename StrictWeakOrdering>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
binary_search(Policy&& policy, InputIterator1 start, InputIterator1 end, InputIterator2 value_start,
              InputIterator2 value_end, OutputIterator result, StrictWeakOrdering comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(policy, start, value_start, result);

    return internal::binary_search_impl(__dispatch_tag, ::std::forward<Policy>(policy), start, end, value_start,
                                        value_end, result, comp);
}

//Binary search end
} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_BINARY_SEARCH_IMPL_H
