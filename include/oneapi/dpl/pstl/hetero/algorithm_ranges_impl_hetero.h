// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_ALGORITHM_RANGES_IMPL_HETERO_H
#define _ONEDPL_ALGORITHM_RANGES_IMPL_HETERO_H

#include "../algorithm_fwd.h"
#include "../parallel_backend.h"
#include "utils_hetero.h"

#if _ONEDPL_BACKEND_SYCL
#    include "dpcpp/utils_ranges_sycl.h"
#    include "dpcpp/unseq_backend_sycl.h"
#    include "dpcpp/parallel_backend_sycl_utils.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{
namespace __ranges
{

//------------------------------------------------------------------------
// walk_n
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Function, typename... _Ranges>
void
__pattern_walk_n(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Function __f, _Ranges&&... __rngs)
{
    auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    if (__n > 0)
    {
        oneapi::dpl::__par_backend_hetero::__parallel_for(_BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec),
                                                          unseq_backend::walk_n<_ExecutionPolicy, _Function>{__f}, __n,
                                                          ::std::forward<_Ranges>(__rngs)...)
            .__deferrable_wait();
    }
}

//------------------------------------------------------------------------
// swap
//------------------------------------------------------------------------

template <typename _Name>
struct __swap1_wrapper;

template <typename _Name>
struct __swap2_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Function>
bool
__pattern_swap(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
               _Function __f)
{
    if (__rng1.size() <= __rng2.size())
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__swap1_wrapper>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __f, __rng1, __rng2);
        return __rng1.size();
    }

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        __tag,
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__swap2_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        __f, __rng2, __rng1);
    return __rng2.size();
}

//------------------------------------------------------------------------
// equal
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Pred>
bool
__pattern_equal(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Pred __pred)
{
    if (__rng1.empty() || __rng2.empty() || __rng1.size() != __rng2.size())
        return false;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, equal_predicate<_Pred>>;

    // TODO: in case of conflicting names
    // __par_backend_hetero::make_wrapped_policy<__par_backend_hetero::__or_policy_wrapper>()
    return !oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), _Predicate{equal_predicate<_Pred>{__pred}},
        oneapi::dpl::__par_backend_hetero::__parallel_or_tag{},
        oneapi::dpl::__ranges::zip_view(::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2)));
}

//------------------------------------------------------------------------
// find_if
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Pred>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_find_if(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _Pred __pred)
{
    //trivial pre-checks
    if (__rng.empty())
        return __rng.size();

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, _Pred>;
    using _TagType = oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_Range>;

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{},
        __par_backend_hetero::make_wrapped_policy<__par_backend_hetero::__find_policy_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        _Predicate{__pred}, _TagType{}, ::std::forward<_Range>(__rng));
}

//------------------------------------------------------------------------
// find_end
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Pred>
oneapi::dpl::__internal::__difference_t<_Range1>
__pattern_find_end(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                   _Pred __pred)
{
    //trivial pre-checks
    if (__rng1.empty() || __rng2.empty() || __rng1.size() < __rng2.size())
        return __rng1.size();

    if (__rng1.size() == __rng2.size())
    {
        const bool __res = __ranges::__pattern_equal(__tag, ::std::forward<_ExecutionPolicy>(__exec), __rng1,
                                                     ::std::forward<_Range2>(__rng2), __pred);
        return __res ? 0 : __rng1.size();
    }

    using _Predicate = unseq_backend::multiple_match_pred<_ExecutionPolicy, _Pred>;
    using _TagType = __par_backend_hetero::__parallel_find_backward_tag<_Range1>;

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{},
        __par_backend_hetero::make_wrapped_policy<__par_backend_hetero::__find_policy_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        _Predicate{__pred}, _TagType{}, ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2));
}

//------------------------------------------------------------------------
// find_first_of
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Pred>
oneapi::dpl::__internal::__difference_t<_Range1>
__pattern_find_first_of(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                        _Pred __pred)
{
    //trivial pre-checks
    if (__rng1.empty() || __rng2.empty())
        return __rng1.size();

    using _Predicate = unseq_backend::first_match_pred<_ExecutionPolicy, _Pred>;
    using _TagType = oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_Range1>;

    //TODO: To check whether it makes sense to iterate over the second sequence in case of __rng1.size() < __rng2.size()
    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{},
        __par_backend_hetero::make_wrapped_policy<__par_backend_hetero::__find_policy_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        _Predicate{__pred}, _TagType{}, ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2));
}

//------------------------------------------------------------------------
// any_of
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Pred>
bool
__pattern_any_of(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _Pred __pred)
{
    if (__rng.empty())
        return false;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, _Pred>;
    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{},
        __par_backend_hetero::make_wrapped_policy<oneapi::dpl::__par_backend_hetero::__or_policy_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        _Predicate{__pred}, oneapi::dpl::__par_backend_hetero::__parallel_or_tag{}, ::std::forward<_Range>(__rng));
}

//------------------------------------------------------------------------
// search
//------------------------------------------------------------------------

template <typename Name>
class equal_wrapper
{
};

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Pred>
oneapi::dpl::__internal::__difference_t<_Range1>
__pattern_search(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                 _Pred __pred)
{
    //trivial pre-checks
    if (__rng2.empty())
        return 0;
    if (__rng1.size() < __rng2.size())
        return __rng1.size();

    if (__rng1.size() == __rng2.size())
    {
        const bool __res = __ranges::__pattern_equal(
            __tag, __par_backend_hetero::make_wrapped_policy<equal_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
            __rng1, ::std::forward<_Range2>(__rng2), __pred);
        return __res ? 0 : __rng1.size();
    }

    using _Predicate = unseq_backend::multiple_match_pred<_ExecutionPolicy, _Pred>;
    using _TagType = oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_Range1>;

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{},
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<
            oneapi::dpl::__par_backend_hetero::__find_policy_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        _Predicate{__pred}, _TagType{}, ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2));
}

//------------------------------------------------------------------------
// search_n
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Size, typename _Tp,
          typename _BinaryPredicate>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_search_n(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range&& __rng, _Size __count,
                   const _Tp& __value, _BinaryPredicate __pred)
{
    //TODO: To consider definition a kind of special factory "multiple_view" (addition to standard "single_view").
    //The factory "multiple_view" would generate a range of N identical values.
    auto __s_rng = oneapi::dpl::experimental::ranges::views::iota(0, __count) |
                   oneapi::dpl::experimental::ranges::views::transform([__value](auto) { return __value; });

    return __ranges::__pattern_search(__tag, ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                                      __s_rng, __pred);
}

template <typename _Size>
_Size
return_value(bool __res, _Size __size, ::std::true_type)
{
    return __res ? 0 : __size;
}

template <typename _Size>
_Size
return_value(_Size __res, _Size __size, ::std::false_type)
{
    return __res == __size - 1 ? __size : __res;
}

//------------------------------------------------------------------------
// adjacent_find
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _BinaryPredicate,
          typename _OrFirstTag>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_adjacent_find(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng,
                        _BinaryPredicate __predicate, _OrFirstTag __is__or_semantic)
{
    if (__rng.size() < 2)
        return __rng.size();

    using _Predicate =
        oneapi::dpl::unseq_backend::single_match_pred<_ExecutionPolicy, adjacent_find_fn<_BinaryPredicate>>;
    using _TagType = ::std::conditional_t<__is__or_semantic(), oneapi::dpl::__par_backend_hetero::__parallel_or_tag,
                                          oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_Range>>;

    auto __rng1 = __rng | oneapi::dpl::experimental::ranges::views::take(__rng.size() - 1);
    auto __rng2 = __rng | oneapi::dpl::experimental::ranges::views::drop(1);

    // TODO: in case of conflicting names
    // __par_backend_hetero::make_wrapped_policy<__par_backend_hetero::__or_policy_wrapper>()
    auto result = oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec),
        _Predicate{adjacent_find_fn<_BinaryPredicate>{__predicate}}, _TagType{},
        oneapi::dpl::__ranges::zip_view(__rng1, __rng2));

    // inverted conditional because of
    // reorder_predicate in glue_algorithm_impl.h
    return return_value(result, __rng.size(), __is__or_semantic);
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_count(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __predicate)
{
    if (__rng.size() == 0)
        return 0;

    using _ReduceValueType = oneapi::dpl::__internal::__difference_t<_Range>;

    auto __reduce_fn = ::std::plus<_ReduceValueType>{};
    // int is being implicitly casted to difference_type
    // otherwise we can only pass the difference_type as a functor template parameter
    auto __transform_fn = [__predicate](auto __gidx, auto __acc) -> int {
        return (__predicate(__acc[__gidx]) ? 1 : 0);
    };

    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType,
                                                                          ::std::true_type /*is_commutative*/>(
               _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __reduce_fn, __transform_fn,
               unseq_backend::__no_init_value{}, // no initial value
               ::std::forward<_Range>(__rng))
        .get();
}

//------------------------------------------------------------------------
// copy_if
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Predicate,
          typename _Assign = oneapi::dpl::__internal::__pstl_assign>
oneapi::dpl::__internal::__difference_t<_Range2>
__pattern_copy_if(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                  _Predicate __pred, _Assign __assign)
{
    oneapi::dpl::__internal::__difference_t<_Range2> __n = __rng1.size();
    if (__n == 0)
        return 0;

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_copy_if(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1),
        std::forward<_Range2>(__rng2), __n, __pred, __assign);

    return __res.get(); //is a blocking call
}

//------------------------------------------------------------------------
// remove_if
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_remove_if(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    if (__rng.size() == 0)
        return __rng.size();

    using _ValueType = oneapi::dpl::__internal::__value_t<_Range>;

    oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, _ValueType> __buf(__exec, __rng.size());
    auto __copy_rng = oneapi::dpl::__ranges::views::all(__buf.get_buffer());

    auto __copy_last_id = __ranges::__pattern_copy_if(__tag, __exec, __rng, __copy_rng, __not_pred<_Predicate>{__pred},
                                                      oneapi::dpl::__internal::__pstl_assign());
    auto __copy_rng_truncated = __copy_rng | oneapi::dpl::experimental::ranges::views::take(__copy_last_id);

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        __tag, ::std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>, _ExecutionPolicy>{}, __copy_rng_truncated,
        ::std::forward<_Range>(__rng));

    return __copy_last_id;
}

//------------------------------------------------------------------------
// unique_copy
//------------------------------------------------------------------------

template <typename _Name>
struct __copy_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2,
          typename _BinaryPredicate, typename _Assign = oneapi::dpl::__internal::__pstl_assign>
oneapi::dpl::__internal::__difference_t<_Range2>
__pattern_unique_copy(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result,
                      _BinaryPredicate __pred)
{
    oneapi::dpl::__internal::__difference_t<_Range2> __n = __rng.size();
    if (__n == 0)
        return 0;
    if (__n == 1)
    {
        // For a sequence of size 1, we can just copy the only element to the result.
        using _CopyBrick = oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>, _ExecutionPolicy>;
        oneapi::dpl::__par_backend_hetero::__parallel_for(
            _BackendTag{},
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy_wrapper>(
                std::forward<_ExecutionPolicy>(__exec)),
            unseq_backend::walk_n<_ExecutionPolicy, _CopyBrick>{_CopyBrick{}}, __n, std::forward<_Range1>(__rng),
            std::forward<_Range2>(__result))
            .get();

        return 1;
    }

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_unique_copy(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng),
        std::forward<_Range2>(__result), __pred);

    return __res.get(); // is a blocking call
}

//------------------------------------------------------------------------
// unique
//------------------------------------------------------------------------

template <typename _Name>
struct __unique_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _BinaryPredicate>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_unique(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range&& __rng, _BinaryPredicate __pred)
{
    if (__rng.size() == 0)
        return __rng.size();

    using _ValueType = oneapi::dpl::__internal::__value_t<_Range>;

    oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, _ValueType> __buf(__exec, __rng.size());
    auto res_rng = oneapi::dpl::__ranges::views::all(__buf.get_buffer());
    oneapi::dpl::__internal::__difference_t<_Range> res = __ranges::__pattern_unique_copy(
        __tag, oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__unique_wrapper>(__exec), __rng, res_rng,
        __pred);

    __ranges::__pattern_walk_n(
        __tag,
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy_wrapper>(std::forward<_ExecutionPolicy>(__exec)),
        __brick_copy<__hetero_tag<_BackendTag>, _ExecutionPolicy>{}, res_rng, std::forward<_Range>(__rng));
    return res;
}

//------------------------------------------------------------------------
// merge
//------------------------------------------------------------------------

template <typename _Name>
struct __copy1_wrapper;

template <typename _Name>
struct __copy2_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3,
          typename _Compare>
oneapi::dpl::__internal::__difference_t<_Range3>
__pattern_merge(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                _Range3&& __rng3, _Compare __comp)
{
    auto __n1 = __rng1.size();
    auto __n2 = __rng2.size();
    auto __n = __n1 + __n2;
    if (__n == 0)
        return 0;

    //To consider the direct copying pattern call in case just one of sequences is empty.
    if (__n1 == 0)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy1_wrapper>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>, _ExecutionPolicy>{},
            ::std::forward<_Range2>(__rng2), ::std::forward<_Range3>(__rng3));
    }
    else if (__n2 == 0)
    {
        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy2_wrapper>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>, _ExecutionPolicy>{},
            ::std::forward<_Range1>(__rng1), ::std::forward<_Range3>(__rng3));
    }
    else
    {
        __par_backend_hetero::__parallel_merge(_BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec),
                                               ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2),
                                               ::std::forward<_Range3>(__rng3), __comp)
            .__deferrable_wait();
    }

    return __n;
}

//------------------------------------------------------------------------
// sort
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Compare, typename _Proj>
void
__pattern_sort(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp, _Proj __proj)
{
    if (__rng.size() >= 2)
        __par_backend_hetero::__parallel_stable_sort(_BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec),
                                                     ::std::forward<_Range>(__rng), __comp, __proj)
            .__deferrable_wait();
}

//------------------------------------------------------------------------
// min_element
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_min_element(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    //If size == 1, result is the zero-indexed element. If size == 0, result is 0.
    if (__rng.size() < 2)
        return 0;

    using _IteratorValueType = typename ::std::iterator_traits<decltype(__rng.begin())>::value_type;
    using _IndexValueType = oneapi::dpl::__internal::__difference_t<_Range>;
    using _ReduceValueType = oneapi::dpl::__internal::tuple<_IndexValueType, _IteratorValueType>;

    // This operator doesn't track the lowest found index in case of equal min. or max. values. Thus, this operator is
    // not commutative.
    auto __reduce_fn = [__comp](_ReduceValueType __a, _ReduceValueType __b) {
        using ::std::get;
        if (__comp(get<1>(__b), get<1>(__a)))
        {
            return __b;
        }
        return __a;
    };
    auto __transform_fn = [](auto __gidx, auto __acc) { return _ReduceValueType{__gidx, __acc[__gidx]}; };

    auto __ret_idx =
        oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType,
                                                                       ::std::false_type /*is_commutative*/>(
            _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __reduce_fn, __transform_fn,
            unseq_backend::__no_init_value{}, // no initial value
            ::std::forward<_Range>(__rng))
            .get();

    using ::std::get;
    return get<0>(__ret_idx);
}

//------------------------------------------------------------------------
// minmax_element
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Compare>
::std::pair<oneapi::dpl::__internal::__difference_t<_Range>, oneapi::dpl::__internal::__difference_t<_Range>>
__pattern_minmax_element(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    //If size == 1, result is the zero-indexed element. If size == 0, result is 0.
    if (__rng.size() < 2)
        return ::std::make_pair(0, 0);

    using _IteratorValueType = typename ::std::iterator_traits<decltype(__rng.begin())>::value_type;
    using _IndexValueType = oneapi::dpl::__internal::__difference_t<_Range>;
    using _ReduceValueType =
        oneapi::dpl::__internal::tuple<_IndexValueType, _IndexValueType, _IteratorValueType, _IteratorValueType>;

    // This operator doesn't track the lowest found index in case of equal min. values and the highest found index in
    // case of equal max. values. Thus, this operator is not commutative.
    auto __reduce_fn = [__comp](_ReduceValueType __a, _ReduceValueType __b) {
        using ::std::get;
        auto __chosen_for_min = __a;
        auto __chosen_for_max = __b;

        if (__comp(get<2>(__b), get<2>(__a)))
            __chosen_for_min = ::std::move(__b);
        if (__comp(get<3>(__b), get<3>(__a)))
            __chosen_for_max = ::std::move(__a);
        return _ReduceValueType{get<0>(__chosen_for_min), get<1>(__chosen_for_max), get<2>(__chosen_for_min),
                                get<3>(__chosen_for_max)};
    };

    // TODO: Doesn't work with `zip_iterator`.
    //       In that case the first and the second arguments of `_ReduceValueType` will be
    //       a `tuple` of `difference_type`, not the `difference_type` itself.
    auto __transform_fn = [](auto __gidx, auto __acc) {
        return _ReduceValueType{__gidx, __gidx, __acc[__gidx], __acc[__gidx]};
    };

    _ReduceValueType __ret =
        oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType,
                                                                       ::std::false_type /*is_commutative*/>(
            _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __reduce_fn, __transform_fn,
            unseq_backend::__no_init_value{}, // no initial value
            ::std::forward<_Range>(__rng))
            .get();

    using ::std::get;
    return ::std::make_pair(get<0>(__ret), get<1>(__ret));
}

//------------------------------------------------------------------------
// reduce_by_segment
//------------------------------------------------------------------------

template <typename _Name>
struct __copy_keys_wrapper;

template <typename _Name>
struct __copy_values_wrapper;

template <typename _Name>
struct __reduce1_wrapper;

template <typename _Name>
struct __reduce2_wrapper;

template <typename _Name>
struct __assign_key1_wrapper;

template <typename _Name>
struct __assign_key2_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3,
          typename _Range4, typename _BinaryPredicate, typename _BinaryOperator>
oneapi::dpl::__internal::__difference_t<_Range3>
__pattern_reduce_by_segment(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range1&& __keys,
                            _Range2&& __values, _Range3&& __out_keys, _Range4&& __out_values,
                            _BinaryPredicate __binary_pred, _BinaryOperator __binary_op)
{
    // The algorithm reduces values in __values where the
    // associated keys for the values are equal to the adjacent key.
    //
    // Example: __keys       = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 }
    //          __values     = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 }
    //
    //          __out_keys   = { 1, 2, 3, 4, 1, 3, 1, 3, 0 }
    //          __out_values = { 1, 2, 3, 4, 2, 6, 2, 6, 0 }

    const auto __n = __keys.size();

    if (__n == 0)
        return 0;

    if (__n == 1)
    {
        __brick_copy<__hetero_tag<_BackendTag>, _ExecutionPolicy> __copy_range{};

        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy_keys_wrapper>(__exec), __copy_range,
            ::std::forward<_Range1>(__keys), ::std::forward<_Range3>(__out_keys));

        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy_values_wrapper>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __copy_range, ::std::forward<_Range2>(__values), ::std::forward<_Range4>(__out_values));

        return 1;
    }

    using __diff_type = oneapi::dpl::__internal::__difference_t<_Range1>;
    using __key_type = oneapi::dpl::__internal::__value_t<_Range1>;
    using __val_type = oneapi::dpl::__internal::__value_t<_Range2>;

    // Round 1: reduce with extra indices added to avoid long segments
    // TODO: At threshold points check if the key is equal to the key at the previous threshold point, indicating a long sequence.
    // Skip a round of copy_if and reduces if there are none.
    auto __idx = oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, __diff_type>(__exec, __n).get_buffer();
    auto __tmp_out_keys =
        oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, __key_type>(__exec, __n).get_buffer();
    auto __tmp_out_values =
        oneapi::dpl::__par_backend_hetero::__buffer<_ExecutionPolicy, __val_type>(__exec, __n).get_buffer();

    // Replicating first element of keys view to be able to compare (i-1)-th and (i)-th key with aligned sequences,
    //  dropping the last key for the i-1 sequence.
    auto __k1 =
        oneapi::dpl::__ranges::take_view_simple(oneapi::dpl::__ranges::replicate_start_view_simple(__keys, 1), __n);

    // view1 elements are a tuple of the element index and pairs of adjacent keys
    // view2 elements are a tuple of the elements where key-index pairs will be written by copy_if
    auto __view1 = experimental::ranges::zip_view(experimental::ranges::views::iota(0, __n), __k1, __keys);
    auto __view2 = experimental::ranges::zip_view(experimental::ranges::views::all_write(__tmp_out_keys),
                                                  experimental::ranges::views::all_write(__idx));

    // use work group size adjusted to shared local memory as the maximum segment size.
    ::std::size_t __wgroup_size =
        oneapi::dpl::__internal::__slm_adjusted_work_group_size(__exec, sizeof(__key_type) + sizeof(__val_type));

    // element is copied if it is the 0th element (marks beginning of first segment), is in an index
    // evenly divisible by wg size (ensures segments are not long), or has a key not equal to the
    // adjacent element (marks end of real segments)
    // TODO: replace wgroup size with segment size based on platform specifics.
    auto __intermediate_result_end = __ranges::__pattern_copy_if(
        __tag, oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__assign_key1_wrapper>(__exec), __view1, __view2,
        [__n, __binary_pred, __wgroup_size](const auto& __a) {
            // The size of key range for the (i-1) view is one less, so for the 0th index we do not check the keys
            // for (i-1), but we still need to get its key value as it is the start of a segment
            const auto index = ::std::get<0>(__a);
            if (index == 0)
                return true;
            return index % __wgroup_size == 0                                 // segment size
                   || !__binary_pred(::std::get<1>(__a), ::std::get<2>(__a)); // key comparison
        },
        unseq_backend::__brick_assign_key_position{});

    //reduce by segment
    oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{}, oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__reduce1_wrapper>(__exec),
        unseq_backend::__brick_reduce_idx<_BinaryOperator, decltype(__n)>(__binary_op, __n), __intermediate_result_end,
        oneapi::dpl::__ranges::take_view_simple(experimental::ranges::views::all_read(__idx),
                                                __intermediate_result_end),
        ::std::forward<_Range2>(__values), experimental::ranges::views::all_write(__tmp_out_values))
        .wait();

    // Round 2: final reduction to get result for each segment of equal adjacent keys
    // create views over adjacent keys
    oneapi::dpl::__ranges::all_view<__key_type, __par_backend_hetero::access_mode::read_write> __new_keys(
        __tmp_out_keys);

    // Replicating first element of key views to be able to compare (i-1)-th and (i)-th key,
    //  dropping the last key for the i-1 sequence.  Only taking the appropriate number of keys to start with here.
    auto __clipped_new_keys = oneapi::dpl::__ranges::take_view_simple(__new_keys, __intermediate_result_end);

    auto __k3 = oneapi::dpl::__ranges::take_view_simple(
        oneapi::dpl::__ranges::replicate_start_view_simple(__clipped_new_keys, 1), __intermediate_result_end);

    // view3 elements are a tuple of the element index and pairs of adjacent keys
    // view4 elements are a tuple of the elements where key-index pairs will be written by copy_if
    auto __view3 = experimental::ranges::zip_view(experimental::ranges::views::iota(0, __intermediate_result_end), __k3,
                                                  __clipped_new_keys);
    auto __view4 = experimental::ranges::zip_view(experimental::ranges::views::all_write(__out_keys),
                                                  experimental::ranges::views::all_write(__idx));

    // element is copied if it is the 0th element (marks beginning of first segment), or has a key not equal to
    // the adjacent element (end of a segment). Artificial segments based on wg size are not created.
    auto __result_end = __ranges::__pattern_copy_if(
        __tag, oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__assign_key2_wrapper>(__exec), __view3, __view4,
        [__binary_pred](const auto& __a) {
            // The size of key range for the (i-1) view is one less, so for the 0th index we do not check the keys
            // for (i-1), but we still need to get its key value as it is the start of a segment
            if (::std::get<0>(__a) == 0)
                return true;
            return !__binary_pred(::std::get<1>(__a), ::std::get<2>(__a)); // keys comparison
        },
        unseq_backend::__brick_assign_key_position{});

    //reduce by segment
    oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{},
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__reduce2_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        unseq_backend::__brick_reduce_idx<_BinaryOperator, decltype(__intermediate_result_end)>(
            __binary_op, __intermediate_result_end),
        __result_end,
        oneapi::dpl::__ranges::take_view_simple(experimental::ranges::views::all_read(__idx), __result_end),
        experimental::ranges::views::all_read(__tmp_out_values), ::std::forward<_Range4>(__out_values))
        .__deferrable_wait();

    return __result_end;
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_ALGORITHM_RANGES_IMPL_HETERO_H
