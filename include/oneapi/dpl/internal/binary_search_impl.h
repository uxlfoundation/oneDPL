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
template <typename Comp, typename T, search_algorithm func>
struct __custom_brick
{
    constexpr static bool __can_vectorize = false;
    constexpr static bool __can_process_multiple_iters = true;

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
            get<2>(acc[idx]) = oneapi::dpl::__internal::__shars_lower_bound(get<0>(acc.tuple()), start_orig, end_orig,
                                                                            get<1>(acc[idx]), comp);
        }
        else if constexpr (func == search_algorithm::upper_bound)
        {
            get<2>(acc[idx]) = oneapi::dpl::__internal::__shars_upper_bound(get<0>(acc.tuple()), start_orig, end_orig,
                                                                            get<1>(acc[idx]), comp);
        }
        else
        {
            auto value = oneapi::dpl::__internal::__shars_lower_bound(get<0>(acc.tuple()), start_orig, end_orig,
                                                                      get<1>(acc[idx]), comp);
            get<2>(acc[idx]) = (value != end_orig) && (get<1>(acc[idx]) == get<0>(acc[value]));
        }
    }
    template <typename _IsFull, typename _Params, typename _Acc>
    void
    operator()(_IsFull, const std::size_t idx, _Params, _Acc acc) const
    {
        if (use_32bit_indexing)
            search_impl<std::uint32_t>(idx, acc);
        else
            search_impl<std::uint64_t>(idx, acc);
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

    auto keep_input = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator1>();
    auto input_buf = keep_input(start, end);

    auto keep_values = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator2>();
    auto value_buf = keep_values(value_start, value_end);

    auto keep_result = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read_write, OutputIterator>();
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

    auto keep_input = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator1>();
    auto input_buf = keep_input(start, end);

    auto keep_values = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator2>();
    auto value_buf = keep_values(value_start, value_end);

    auto keep_result = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read_write, OutputIterator>();
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

    auto keep_input = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator1>();
    auto input_buf = keep_input(start, end);

    auto keep_values = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read, InputIterator2>();
    auto value_buf = keep_values(value_start, value_end);

    auto keep_result = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read_write, OutputIterator>();
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
