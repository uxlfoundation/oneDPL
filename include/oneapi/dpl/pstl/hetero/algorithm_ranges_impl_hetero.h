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
#include "../utils_ranges.h"
#include "utils_hetero.h"
#include "../functional_impl.h" // for oneapi::dpl::identity

#if _ONEDPL_BACKEND_SYCL
#    include "dpcpp/utils_ranges_sycl.h"
#    include "dpcpp/unseq_backend_sycl.h"
#    include "dpcpp/parallel_backend_sycl_utils.h"
#    include "dpcpp/execution_sycl_defs.h"
#endif

#if _ONEDPL_CPP20_RANGES_PRESENT
#include <ranges>
#include <utility>
#include <cassert>
#include <cstddef>
#include <functional>
#include <type_traits>
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
std::make_unsigned_t<std::common_type_t<oneapi::dpl::__internal::__difference_t<_Ranges>...>>
__pattern_walk_n(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Function __f, _Ranges&&... __rngs)
{
    const auto __n = oneapi::dpl::__ranges::__min_size_calc{}(__rngs...);
    if (__n > 0)
    {
        oneapi::dpl::__par_backend_hetero::__parallel_for(
            _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
            unseq_backend::walk_n_vectors_or_scalars<_Function>{__f, static_cast<std::size_t>(__n)}, __n,
            std::forward<_Ranges>(__rngs)...)
            .__checked_deferrable_wait();
    }
    return __n;
}

#if _ONEDPL_CPP20_RANGES_PRESENT

//---------------------------------------------------------------------------------------------------------------------
// pattern_for_each
//---------------------------------------------------------------------------------------------------------------------
template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _Fun, typename _Proj>
void
__pattern_for_each(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Fun __f, _Proj __proj)
{
    oneapi::dpl::__internal::__unary_op<_Fun, _Proj> __f_1{__f, __proj};

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(__tag, std::forward<_ExecutionPolicy>(__exec), __f_1,
                                                            oneapi::dpl::__ranges::views::all(std::forward<_R>(__r)));
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_transform
//---------------------------------------------------------------------------------------------------------------------
template<typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _F,
         typename _Proj>
void
__pattern_transform(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r,
                    _F __op, _Proj __proj)
{
    assert(std::ranges::size(__in_r) <= std::ranges::size(__out_r)); // for debug purposes only
    oneapi::dpl::__internal::__unary_op<_F, _Proj> __unary_op{__op, __proj};

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(__tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__transform_functor<decltype(__unary_op)>{std::move(__unary_op)},
            oneapi::dpl::__ranges::views::all_read(std::forward<_InRange>(__in_r)),
            oneapi::dpl::__ranges::views::all_write(std::forward<_OutRange>(__out_r)));
}

template<typename _BackendTag, typename _ExecutionPolicy, typename _InRange1, typename _InRange2, typename _OutRange, typename _F,
         typename _Proj1, typename _Proj2>
void
__pattern_transform(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _InRange1&& __in_r1,
                    _InRange2&& __in_r2, _OutRange&& __out_r, _F __binary_op, _Proj1 __proj1, _Proj2 __proj2)
{
    oneapi::dpl::__internal::__binary_op<_F, _Proj1, _Proj2> __f{__binary_op, __proj1, __proj2};

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(__tag, std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__internal::__transform_functor<decltype(__f)>{std::move(__f)},
            oneapi::dpl::__ranges::views::all_read(std::forward<_InRange1>(__in_r1)),
            oneapi::dpl::__ranges::views::all_read(std::forward<_InRange2>(__in_r2)),
            oneapi::dpl::__ranges::views::all_write(std::forward<_OutRange>(__out_r)));
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
void
__pattern_copy(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r)
{
    assert(oneapi::dpl::__ranges::__size(__in_r) <= oneapi::dpl::__ranges::__size(__out_r)); // for debug purposes only

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        __tag, ::std::forward<_ExecutionPolicy>(__exec), oneapi::dpl::__internal::__brick_copy<decltype(__tag)>{},
        oneapi::dpl::__ranges::views::all_read(std::forward<_InRange>(__in_r)),
        oneapi::dpl::__ranges::views::all_write(std::forward<_OutRange>(__out_r)));
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _T>
std::ranges::borrowed_iterator_t<_R>
__pattern_fill(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, const _T& __value)
{
    oneapi::dpl::__internal::__set_value __f{__value};

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(__tag, std::forward<_ExecutionPolicy>(__exec), __f,
                                                        oneapi::dpl::__ranges::views::all_write(std::forward<_R>(__r)));

    return {std::ranges::begin(__r) + oneapi::dpl::__ranges::__size(__r)};
}

#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// swap
//------------------------------------------------------------------------

template <typename _Name>
struct __swap1_wrapper;

template <typename _Name>
struct __swap2_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__difference_t<_Range1>
__pattern_swap(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    const std::size_t __n1 = oneapi::dpl::__ranges::__size(__rng1);
    const std::size_t __n2 = oneapi::dpl::__ranges::__size(__rng2);

    //a trivial pre-check
    if (__n1 == 0 || __n2 == 0)
        return 0;

    using _Function = oneapi::dpl::__internal::__swap_fn;

    if (__n1 <= __n2)
    {
        oneapi::dpl::__par_backend_hetero::__parallel_for(
            _BackendTag{},
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__swap1_wrapper>(
                std::forward<_ExecutionPolicy>(__exec)),
            unseq_backend::__brick_swap<_Function>{_Function{}, __n1}, __n1,
            oneapi::dpl::__ranges::__get_subscription_view(__rng1),
            oneapi::dpl::__ranges::__get_subscription_view(__rng2))
            .__checked_deferrable_wait();
        return __n1;
    }

    oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{},
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__swap2_wrapper>(std::forward<_ExecutionPolicy>(__exec)),
        unseq_backend::__brick_swap<_Function>{_Function{}, __n2}, __n2,
        oneapi::dpl::__ranges::__get_subscription_view(__rng2), oneapi::dpl::__ranges::__get_subscription_view(__rng1))
        .__checked_deferrable_wait();
    return __n2;
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R1, typename _R2>
void
__pattern_swap_ranges(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2)
{
    oneapi::dpl::__internal::__ranges::__pattern_swap(__tag, std::forward<_ExecutionPolicy>(__exec),
                                                      oneapi::dpl::__ranges::views::all(std::forward<_R1>(__r1)),
                                                      oneapi::dpl::__ranges::views::all(std::forward<_R2>(__r2)));
}
#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// equal
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Pred>
bool
__pattern_equal(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Pred __pred)
{
    const auto __n1 = oneapi::dpl::__ranges::__size(__rng1);
    const auto __n2 = oneapi::dpl::__ranges::__size(__rng2);
    if (__n1 != __n2)
        return false;

    if (__n1 == 0)
        return true; //both sequences are empty

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<oneapi::dpl::__internal::__not_pred<_Pred>>;
    using __or_tag = oneapi::dpl::__par_backend_hetero::__parallel_or_tag;
    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    assert(__n1 == __n2);

    return !oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        _Predicate{oneapi::dpl::__internal::__not_pred<_Pred>{__pred}}, __or_tag{}, __size_calc{},
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range1>(__rng1)),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range2>(__rng2)));
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template<typename _BackendTag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
         typename _Proj2>
bool
__pattern_equal(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred,
                 _Proj1 __proj1, _Proj2 __proj2)
{
    oneapi::dpl::__internal::__binary_op<_Pred, _Proj1, _Proj2> __pred_2(__pred, __proj1, __proj2);

    return oneapi::dpl::__internal::__ranges::__pattern_equal(
        __tag, ::std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::views::all_read(std::forward<_R1>(__r1)),
        oneapi::dpl::__ranges::views::all_read(std::forward<_R2>(__r2)), __pred_2);
}
#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// find_if
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Pred>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_find_if(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _Pred __pred)
{
    //trivial pre-checks
    if (oneapi::dpl::__ranges::__empty(__rng))
        return 0;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_Pred>;
    using _IndexType = std::make_unsigned_t<oneapi::dpl::__internal::__difference_t<_Range>>;
    using _TagType = oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_IndexType>;
    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), _Predicate{__pred}, _TagType{}, __size_calc{},
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range>(__rng)));
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
std::ranges::borrowed_iterator_t<_R>
__pattern_find_if(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    oneapi::dpl::__internal::__unary_op<_Pred, _Proj> __pred_1{__pred, __proj};

    auto __idx = oneapi::dpl::__internal::__ranges::__pattern_find_if(__tag, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::views::all_read(__r), __pred_1);

    return std::ranges::begin(__r) + __idx;
}
#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// find_end
//------------------------------------------------------------------------

template <typename Name>
struct __equal_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Pred>
oneapi::dpl::__internal::__difference_t<_Range1>
__pattern_find_end(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                   _Pred __pred)
{
    const auto __n1 = oneapi::dpl::__ranges::__size(__rng1);
    const auto __n2 = oneapi::dpl::__ranges::__size(__rng2);

    //trivial pre-checks
    if (__n1 == 0 || __n2 == 0 || __n1 < __n2)
        return __n1;

    if (__n1 == __n2)
    {
        const bool __res = __ranges::__pattern_equal(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__equal_wrapper>(
                std::forward<_ExecutionPolicy>(__exec)),
            oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range1>(__rng1)),
            oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range2>(__rng2)), __pred);
        return __res ? 0 : __n1;
    }

    using _Predicate = unseq_backend::multiple_match_pred<_Pred>;
    using _IndexType = oneapi::dpl::__internal::__difference_t<_Range1>;
    using _TagType = __par_backend_hetero::__parallel_find_backward_tag<_IndexType>;
    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), _Predicate{__pred}, _TagType{}, __size_calc{},
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range1>(__rng1)),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range2>(__rng2)));
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
          typename _Proj2>
std::ranges::borrowed_subrange_t<_R1>
__pattern_find_end(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred,
                   _Proj1 __proj1, _Proj2 __proj2)
{
    oneapi::dpl::__internal::__binary_op<_Pred, _Proj1, _Proj2> __bin_pred{__pred, __proj1, __proj2};

    auto __idx = oneapi::dpl::__internal::__ranges::__pattern_find_end(__tag, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::views::all_read(__r1), oneapi::dpl::__ranges::views::all_read(__r2), __bin_pred);

    auto __first1 = std::ranges::begin(__r1);
    auto __last1 = __first1 + oneapi::dpl::__ranges::__size(__r1);

    auto __it = __first1 + __idx;

    return {__it, __it + (__it == __last1 ? 0 : oneapi::dpl::__ranges::__size(__r2))};
}

#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// find_first_of
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Pred>
oneapi::dpl::__internal::__difference_t<_Range1>
__pattern_find_first_of(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                        _Pred __pred)
{
    const auto __n1 = oneapi::dpl::__ranges::__size(__rng1);
    const auto __n2 = oneapi::dpl::__ranges::__size(__rng2);
    //trivial pre-checks
    if (__n1 == 0 || __n2 == 0)
        return __n1;

    using _Predicate = unseq_backend::first_match_pred<_Pred>;
    using _IndexType = std::make_unsigned_t<oneapi::dpl::__internal::__difference_t<_Range1>>;
    using _TagType = oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_IndexType>;
    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    //TODO: To check whether it makes sense to iterate over the second sequence in case of __n1 < __n2
    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), _Predicate{__pred}, _TagType{}, __size_calc{},
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range1>(__rng1)),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range2>(__rng2)));
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
          typename _Proj2>
std::ranges::borrowed_iterator_t<_R1>
__pattern_find_first_of(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2,
                        _Pred __pred, _Proj1 __proj1, _Proj2 __proj2)
{
    oneapi::dpl::__internal::__binary_op<_Pred, _Proj1, _Proj2> __bin_pred{__pred, __proj1, __proj2};

    auto __idx = oneapi::dpl::__internal::__ranges::__pattern_find_first_of(__tag, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::views::all_read(__r1), oneapi::dpl::__ranges::views::all_read(__r2), __bin_pred);

    return {std::ranges::begin(__r1) + __idx};
}
#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// any_of
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Pred>
bool
__pattern_any_of(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _Pred __pred)
{
    if (oneapi::dpl::__ranges::__empty(__rng))
        return false;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_Pred>;
    using __or_tag = oneapi::dpl::__par_backend_hetero::__parallel_or_tag;
    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), _Predicate{__pred}, __or_tag{}, __size_calc{},
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range>(__rng)));
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
bool
__pattern_any_of(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    oneapi::dpl::__internal::__unary_op<_Pred, _Proj> __pred_1{__pred, __proj};

    return oneapi::dpl::__internal::__ranges::__pattern_any_of(__tag, std::forward<_ExecutionPolicy>(__exec),
                oneapi::dpl::__ranges::views::all_read(std::forward<_R>(__r)), __pred_1);
}
#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// search
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Pred>
oneapi::dpl::__internal::__difference_t<_Range1>
__pattern_search(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                 _Pred __pred)
{
    const auto __n1 = oneapi::dpl::__ranges::__size(__rng1);
    const auto __n2 = oneapi::dpl::__ranges::__size(__rng2);

    //trivial pre-checks
    if (__n2 == 0)
        return 0;
    if (__n1 < __n2)
        return __n1;

    if (__n1 == __n2)
    {
        const bool __res = __ranges::__pattern_equal(
            __tag, __par_backend_hetero::make_wrapped_policy<__equal_wrapper>(std::forward<_ExecutionPolicy>(__exec)),
            oneapi::dpl::__ranges::__get_subscription_view(__rng1),
            oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range2>(__rng2)), __pred);
        return __res ? 0 : __n1;
    }

    using _Predicate = unseq_backend::multiple_match_pred<_Pred>;
    using _IndexType = std::make_unsigned_t<oneapi::dpl::__internal::__difference_t<_Range1>>;
    using _TagType = oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_IndexType>;
    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), _Predicate{__pred}, _TagType{}, __size_calc{},
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range1>(__rng1)),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range2>(__rng2)));
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
          typename _Proj2>
std::ranges::borrowed_subrange_t<_R1>
__pattern_search(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred,
                 _Proj1 __proj1, _Proj2 __proj2)
{
    oneapi::dpl::__internal::__binary_op<_Pred, _Proj1, _Proj2> __pred_2{__pred, __proj1, __proj2};

    auto __idx = oneapi::dpl::__internal::__ranges::__pattern_search(__tag, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::views::all_read(__r1),
        oneapi::dpl::__ranges::views::all_read(__r2), __pred_2);
    auto __res = std::ranges::begin(__r1) + __idx;

    return {__res, __res == std::ranges::end(__r1) ? __res : __res + oneapi::dpl::__ranges::__size(__r2)};
}
#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// search_n
//------------------------------------------------------------------------

template <typename _Tp>
struct __pattern_search_n_fn
{
    _Tp __value;

    template <typename _TValue1>
    _Tp
    operator()(_TValue1&&) const
    {
        return __value;
    }
};

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Size, typename _Tp,
          typename _BinaryPredicate>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_search_n(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range&& __rng, _Size __count,
                   const _Tp& __value, _BinaryPredicate __pred)
{
    __pattern_search_n_fn<_Tp> __fn{__value};

    //TODO: To consider definition a kind of special factory "multiple_view" (addition to standard "single_view").
    //The factory "multiple_view" would generate a range of N identical values.
    auto __s_rng = oneapi::dpl::experimental::ranges::views::iota(0, __count) |
                   oneapi::dpl::experimental::ranges::views::transform(__fn);

    return __ranges::__pattern_search(__tag, std::forward<_ExecutionPolicy>(__exec),
                                      oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range>(__rng)),
                                      __s_rng, __pred);
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _T, typename _Pred, typename _Proj>
std::ranges::borrowed_subrange_t<_R>
__pattern_search_n(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r,
                   std::ranges::range_difference_t<_R> __count, const _T& __value, _Pred __pred, _Proj __proj)
{
    oneapi::dpl::__internal::__binary_op<_Pred, _Proj, std::identity> __pred_2{__pred, __proj, std::identity{}};

    auto __idx = oneapi::dpl::__internal::__ranges::__pattern_search_n(__tag, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::views::all_read(__r), __count, __value, __pred_2);

    auto __found_first = std::ranges::begin(__r) + __idx;
    auto __found_last = (__idx == oneapi::dpl::__ranges::__size(__r) ? __found_first : __found_first + __count);

    return {__found_first, __found_last};
}
#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// adjacent_find
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _BinaryPredicate,
          typename _OrFirstTag>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_adjacent_find(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _BinaryPredicate __pred,
                        _OrFirstTag __is_or_semantic)
{
    const auto __n = oneapi::dpl::__ranges::__size(__rng);
    if (__n < 2)
        return __n;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_BinaryPredicate>;
    using _IndexType = std::make_unsigned_t<oneapi::dpl::__internal::__difference_t<_Range>>;
    using _TagType = std::conditional_t<__is_or_semantic(), oneapi::dpl::__par_backend_hetero::__parallel_or_tag,
                                        oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_IndexType>>;

    //ATTENTION!!! oneDPL supports SYCL buffer via a placeholder accessor; a 'subrange' cannot be used here because
    //getting an iterator for the placeholder accessor is incorrect on the host; so, oneDPL uses lazy-access views
    //for range transformations.
    //For _ONEDPL_CPP20_RANGES_PRESENT, oneDPL may use std::ranges::take_view and std::ranges::drop_view, but there are
    //C++ standard libraries (f.e libstdc++ 10), where the implementation might throw C++ exceptions, that is an issue,
    //because "SYCL kernel cannot use exceptions".

    auto&& __rng_s = oneapi::dpl::__ranges::__get_subscription_view(__rng);

    auto __rng1 = oneapi::dpl::__ranges::take_view_simple(__rng_s, __n - 1);
    auto __rng2 = oneapi::dpl::__ranges::drop_view_simple(__rng_s, 1);

    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    assert(oneapi::dpl::__ranges::__size(__rng1) == oneapi::dpl::__ranges::__size(__rng2));

    auto result = oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), _Predicate{__pred}, _TagType{}, __size_calc{}, __rng1,
        __rng2);

    // inverted conditional because of
    // reorder_predicate in glue_algorithm_impl.h
    if constexpr (__is_or_semantic())
        return result ? 0 : __n;
    else
        return result == __n - 1 ? __n : result;
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
std::ranges::borrowed_iterator_t<_R>
__pattern_adjacent_find_ranges(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred,
                               _Proj __proj)
{
    oneapi::dpl::__internal::__binary_op<_Pred, _Proj, _Proj> __pred_2{__pred, __proj, __proj};

    auto __idx =
        oneapi::dpl::__internal::__ranges::__pattern_adjacent_find(__tag, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::views::all_read(__r), __pred_2,
        oneapi::dpl::__internal::__first_semantic());

    return std::ranges::begin(__r) + __idx;
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
bool
__pattern_is_sorted(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    oneapi::dpl::__internal::__binary_op<_Comp, _Proj, _Proj> __pred_2{__comp, __proj, __proj};

    return oneapi::dpl::__internal::__ranges::__pattern_adjacent_find(
               __tag, std::forward<_ExecutionPolicy>(__exec), oneapi::dpl::__ranges::views::all_read(__r),
               oneapi::dpl::__internal::__reorder_pred(__pred_2),
               oneapi::dpl::__internal::__or_semantic()) == oneapi::dpl::__ranges::__size(__r);
}
#endif //_ONEDPL_CPP20_RANGES_PRESENT

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_count(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __predicate)
{
    if (oneapi::dpl::__ranges::__empty(__rng))
        return 0;

    using _ReduceValueType = oneapi::dpl::__internal::__difference_t<_Range>;

    auto __reduce_fn = ::std::plus<_ReduceValueType>{};
    oneapi::dpl::__internal::__pattern_count_transform_fn<_Predicate> __transform_fn{__predicate};

    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType,
                                                                          ::std::true_type /*is_commutative*/>(
               _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __reduce_fn, __transform_fn,
               unseq_backend::__no_init_value{}, // no initial value
               oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range>(__rng)))
        .get();
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
std::ranges::range_difference_t<_R>
__pattern_count_if(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    oneapi::dpl::__internal::__unary_op<_Pred, _Proj> __pred_1{__pred, __proj};

    return oneapi::dpl::__internal::__ranges::__pattern_count(
        __tag, std::forward<_ExecutionPolicy>(__exec), oneapi::dpl::__ranges::views::all_read(std::forward<_R>(__r)),
        __pred_1);
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _T, typename _Proj>
std::ranges::range_difference_t<_R>
__pattern_count(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, const _T& __value, _Proj __proj)
{
    oneapi::dpl::__internal::__count_fn_pred<_T, _Proj> __pred{__value, __proj};
    return oneapi::dpl::__internal::__ranges::__pattern_count(
        __tag, std::forward<_ExecutionPolicy>(__exec), oneapi::dpl::__ranges::views::all_read(std::forward<_R>(__r)),
        __pred);
}

#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// copy_if
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Predicate,
          typename _Assign = oneapi::dpl::__internal::__pstl_assign>
oneapi::dpl::__internal::__difference_t<_Range2>
__pattern_copy_if(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                  _Predicate __pred, _Assign __assign)
{
    oneapi::dpl::__internal::__difference_t<_Range2> __n = oneapi::dpl::__ranges::__size(__rng1);
    if (__n == 0)
        return 0;

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_copy_if(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range1>(__rng1)),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range2>(__rng2)), __n, __pred, __assign);

    return __res.get(); //is a blocking call
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _Pred,
          typename _Proj>
std::ranges::copy_if_result<std::ranges::borrowed_iterator_t<_InRange>, std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_copy_if_ranges(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r,
                         _OutRange&& __out_r, _Pred __pred, _Proj __proj)
{
    oneapi::dpl::__internal::__unary_op<_Pred, _Proj> __pred_1{__pred, __proj};

    auto __res_idx = oneapi::dpl::__internal::__ranges::__pattern_copy_if(__tag,
        std::forward<_ExecutionPolicy>(__exec), oneapi::dpl::__ranges::views::all_read(__in_r),
        oneapi::dpl::__ranges::views::all_write(__out_r), __pred_1,
        oneapi::dpl::__internal::__pstl_assign());

    return {std::ranges::begin(__in_r) + oneapi::dpl::__ranges::__size(__in_r),
            std::ranges::begin(__out_r) + __res_idx};
}
#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// remove_if
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_remove_if(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    const auto __n = oneapi::dpl::__ranges::__size(__rng);
    if (__n == 0)
        return 0;

    using _ValueType = oneapi::dpl::__internal::__value_t<_Range>;

    oneapi::dpl::__par_backend_hetero::__buffer<_ValueType> __buf(__n);
    auto __copy_rng = oneapi::dpl::__ranges::views::all(__buf.get_buffer());

    auto __copy_last_id = __ranges::__pattern_copy_if(__tag, __exec, __rng, __copy_rng, __not_pred<_Predicate>{__pred},
                                                      oneapi::dpl::__internal::__pstl_assign());
    auto __copy_rng_truncated = __copy_rng | oneapi::dpl::experimental::ranges::views::take(__copy_last_id);

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        __tag, ::std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{}, __copy_rng_truncated,
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range>(__rng)));

    return __copy_last_id;
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
std::ranges::borrowed_subrange_t<_R>
__pattern_remove_if(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    oneapi::dpl::__internal::__unary_op<_Pred, _Proj> __pred_1{__pred, __proj};

    auto __n = oneapi::dpl::__ranges::__size(__r);
    auto __idx = oneapi::dpl::__internal::__ranges::__pattern_remove_if(
        __tag, std::forward<_ExecutionPolicy>(__exec), oneapi::dpl::__ranges::views::all(std::forward<_R>(__r)),
        __pred_1);

    return {std::ranges::begin(__r) + __idx, std::ranges::begin(__r) + __n};
}

//------------------------------------------------------------------------
// reverse
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _R>
void
__pattern_reverse(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _R&& __r)
{
    const auto __n = oneapi::dpl::__ranges::__size(__r);
    if (__n <= 1)
        return;

    oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), unseq_backend::__reverse_functor<decltype(__n)>{__n},
        __n / 2, oneapi::dpl::__ranges::__get_subscription_view(std::forward<_R>(__r)))
        .__checked_deferrable_wait();
}

//------------------------------------------------------------------------
// reverse_copy
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
void
__pattern_reverse_copy(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r)
{
    const auto __n = oneapi::dpl::__ranges::__size(__in_r);

    assert(__n == oneapi::dpl::__ranges::__size(__out_r)); // sizes must be made equal on the caller side

    if (__n == 0)
        return;

    oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), unseq_backend::__reverse_copy<decltype(__n)>{__n}, __n,
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_InRange>(__in_r)),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_OutRange>(__out_r)))
        .__checked_deferrable_wait();
}

//------------------------------------------------------------------------
// move
//------------------------------------------------------------------------

template<typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
void
__pattern_move(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _InRange&& __r, _OutRange&& __out_r)
{
    oneapi::dpl::__internal::__ranges::__pattern_walk_n(__tag, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__internal::__brick_move<decltype(__tag)>{},
        oneapi::dpl::__ranges::views::all_read(std::forward<_InRange>(__r)),
        oneapi::dpl::__ranges::views::all_write(std::forward<_OutRange>(__out_r)));
}

#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// unique_copy
//------------------------------------------------------------------------

template <typename _Name>
struct __copy_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2,
          typename _BinaryPredicate, typename _Assign = oneapi::dpl::__internal::__pstl_assign>
oneapi::dpl::__internal::__difference_t<_Range2>
__pattern_unique_copy(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result,
                      _BinaryPredicate __pred)
{
    oneapi::dpl::__internal::__difference_t<_Range2> __n = oneapi::dpl::__ranges::__size(__rng);
    if (__n == 0)
        return 0;
    if (__n == 1)
    {
        // For a sequence of size 1, we can just copy the only element to the result.
        using _CopyBrick = oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>;
        oneapi::dpl::__par_backend_hetero::__parallel_for(
            _BackendTag{},
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy_wrapper>(
                std::forward<_ExecutionPolicy>(__exec)),
            unseq_backend::walk_n_vectors_or_scalars<_CopyBrick>{_CopyBrick{}, static_cast<std::size_t>(__n)}, __n,
            oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range1>(__rng)),
            oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range2>(__result)))
            .get();

        return 1;
    }

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_unique_copy(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range1>(__rng)),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range2>(__result)), __pred);

    return __res.get(); // is a blocking call
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _OutRange, typename _Comp,
          typename _Proj>
std::ranges::unique_copy_result<std::ranges::borrowed_iterator_t<_R>, std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_unique_copy(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, _OutRange&& __out_r,
                      _Comp __comp, _Proj __proj)
{
    oneapi::dpl::__internal::__binary_op<_Comp, _Proj, _Proj> __pred_2{__comp, __proj, __proj};

    auto __beg = std::ranges::begin(__r);
    auto __end = __beg + oneapi::dpl::__ranges::__size(__r);
    auto __beg_out = std::ranges::begin(__out_r);

    auto __idx = oneapi::dpl::__internal::__ranges::__pattern_unique_copy(
        __tag, std::forward<_ExecutionPolicy>(__exec), oneapi::dpl::__ranges::views::all_read(__r),
        oneapi::dpl::__ranges::views::all_write(__out_r), __pred_2);

    return {__end, __beg_out + __idx};
}

#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// unique
//------------------------------------------------------------------------

template <typename _Name>
struct __unique_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _BinaryPredicate>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_unique(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range&& __rng, _BinaryPredicate __pred)
{
    const auto __n = oneapi::dpl::__ranges::__size(__rng);
    if (__n == 0)
        return __n;

    using _ValueType = oneapi::dpl::__internal::__value_t<_Range>;

    auto&& __rng_s = oneapi::dpl::__ranges::__get_subscription_view(__rng);

    oneapi::dpl::__par_backend_hetero::__buffer<_ValueType> __buf(__n);
    auto __res_rng = oneapi::dpl::__ranges::views::all(__buf.get_buffer());

    oneapi::dpl::__internal::__difference_t<_Range> res = __ranges::__pattern_unique_copy(
        __tag, oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__unique_wrapper>(__exec), __rng_s, __res_rng,
        __pred);

    __ranges::__pattern_walk_n(
        __tag,
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy_wrapper>(std::forward<_ExecutionPolicy>(__exec)),
        __brick_copy<__hetero_tag<_BackendTag>>{}, __res_rng, std::forward<decltype(__rng_s)>(__rng_s));

    return res;
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _Comp, typename _Proj>
std::ranges::borrowed_subrange_t<_R>
__pattern_unique(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    oneapi::dpl::__internal::__binary_op<_Comp, _Proj, _Proj> __pred_2{__comp, __proj, __proj};

    auto __beg = std::ranges::begin(__r);
    auto __end = __beg + oneapi::dpl::__ranges::__size(__r);

    auto __idx = oneapi::dpl::__internal::__ranges::__pattern_unique(__tag, std::forward<_ExecutionPolicy>(__exec),
                                                                     oneapi::dpl::__ranges::views::all(__r), __pred_2);

    return {__beg + __idx, __end};
}
#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// merge
//------------------------------------------------------------------------

template <typename _Name>
struct __copy1_wrapper;

template <typename _Name>
struct __copy2_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3,
          typename _Compare, typename _Proj1, typename _Proj2>
std::pair<oneapi::dpl::__internal::__difference_t<_Range1>, oneapi::dpl::__internal::__difference_t<_Range2>>
__pattern_merge(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                _Range3&& __rng3, _Compare __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    if (oneapi::dpl::__ranges::__empty(__rng3))
        return {0, 0};

    const auto __n1 = oneapi::dpl::__ranges::__size(__rng1);
    const auto __n2 = oneapi::dpl::__ranges::__size(__rng2);

    //To consider the direct copying pattern call in case just one of sequences is empty.
    if (__n1 == 0)
    {
        auto __res = oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy1_wrapper>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{},
            oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range2>(__rng2)),
            oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range3>(__rng3)));
        return {0, __res};
    }

    if (__n2 == 0)
    {
        auto __res = oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy2_wrapper>(
                std::forward<_ExecutionPolicy>(__exec)),
            oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{},
            oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range1>(__rng1)),
            oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range3>(__rng3)));
        return {__res, 0};
    }

    auto __res = __par_backend_hetero::__parallel_merge<std::true_type /*out size limit*/>(
        _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range1>(__rng1)),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range2>(__rng2)),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range3>(__rng3)), __comp, __proj1, __proj2);

    auto __val = __res.get();
    return {__val.first, __val.second};
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange,
          typename _Comp, typename _Proj1, typename _Proj2>
std::ranges::merge_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_R2>,
                          std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_merge_ranges(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2,
                       _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    using _Index1 = std::ranges::range_difference_t<_R1>;
    using _Index2 = std::ranges::range_difference_t<_R2>;
    using _Index3 = std::ranges::range_difference_t<_OutRange>;

    const _Index1 __n_1 = oneapi::dpl::__ranges::__size(__r1);
    const _Index2 __n_2 = oneapi::dpl::__ranges::__size(__r2);
    const _Index3 __n_out = std::min<_Index3>(__n_1 + __n_2, oneapi::dpl::__ranges::__size(__out_r));

    const std::pair __res = oneapi::dpl::__internal::__ranges::__pattern_merge(
        __tag, std::forward<_ExecutionPolicy>(__exec), oneapi::dpl::__ranges::views::all_read(__r1),
        oneapi::dpl::__ranges::views::all_read(__r2), oneapi::dpl::__ranges::views::all_write(__out_r), __comp, __proj1,
        __proj2);

    return {std::ranges::begin(__r1) + __res.first, std::ranges::begin(__r2) + __res.second,
            std::ranges::begin(__out_r) + __n_out};
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Comp, typename _Proj1,
          typename _Proj2>
bool
__pattern_includes(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Comp __comp,
                   _Proj1 __proj1, _Proj2 __proj2)
{
    const auto __n1 = oneapi::dpl::__ranges::__size(__r1);
    const auto __n2 = oneapi::dpl::__ranges::__size(__r2);

    //according to the spec
    if (__n2 == 0)
        return true;

    //optimization; {1} - the first sequence, {2} - the second sequence
    //{1} is empty or size_of{2} > size_of{1}
    if (__n1 == 0 || __n2 > __n1)
        return false;

    using __brick_include_type = unseq_backend::__brick_includes<decltype(__n1), decltype(__n2), _Comp, _Proj1, _Proj2>;
    using _TagType = __par_backend_hetero::__parallel_or_tag;
    using __size_calc = oneapi::dpl::__ranges::__second_size_calc;

    return !oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        __brick_include_type{__n1, __n2, __comp, __proj1, __proj2}, _TagType{}, __size_calc{},
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_R1>(__r1)),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_R2>(__r2)));
}

//Dummy names to avoid kernel problems
template <typename Name>
struct __set_union_copy_case_1;

template <typename Name>
struct __set_union_copy_case_2;

template <typename _BackendTag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange,
          typename _Comp, typename _Proj1, typename _Proj2>
std::ranges::set_union_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_R2>,
                              std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_set_union(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2,
                    _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    const auto __first1 = std::ranges::begin(__r1);
    const auto __first2 = std::ranges::begin(__r2);
    const auto __result = std::ranges::begin(__out_r);

    const auto __n1 = oneapi::dpl::__ranges::__size(__r1);
    const auto __n2 = oneapi::dpl::__ranges::__size(__r2);

    if (__n1 == 0 && __n2 == 0)
        return {__first1, __first2, __result};

    //{1} is empty
    if (__n1 == 0)
    {
        const auto __idx = oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_union_copy_case_1>(
                std::forward<_ExecutionPolicy>(__exec)),
            oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{},
            oneapi::dpl::__ranges::__get_subscription_view(__r2),
            oneapi::dpl::__ranges::__get_subscription_view(__out_r));

        return {__first1, __first2 + __n2, __result + __idx};
    }

    //{2} is empty
    if (__n2 == 0)
    {
        const auto __idx = oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_union_copy_case_2>(
                std::forward<_ExecutionPolicy>(__exec)),
            oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{},
            oneapi::dpl::__ranges::__get_subscription_view(__r1),
            oneapi::dpl::__ranges::__get_subscription_view(__out_r));

        return {__first1 + __n1, __first2, __result + __idx};
    }

    const std::size_t __result_size = __par_backend_hetero::__parallel_set_op<unseq_backend::_UnionTag>(
        _BackendTag{}, unseq_backend::_UnionTag{}, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::__get_subscription_view(__r1), oneapi::dpl::__ranges::__get_subscription_view(__r2),
        oneapi::dpl::__ranges::__get_subscription_view(__out_r), __comp, __proj1, __proj2);

    return {__first1 + __n1, __first2 + __n2, __result + __result_size};
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange,
          typename _Comp, typename _Proj1, typename _Proj2>
std::ranges::set_intersection_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_R2>,
                                     std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_set_intersection(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2,
                           _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    const auto __first1 = std::ranges::begin(__r1);
    const auto __first2 = std::ranges::begin(__r2);
    const auto __result = std::ranges::begin(__out_r);

    const auto __n1 = oneapi::dpl::__ranges::__size(__r1);
    const auto __n2 = oneapi::dpl::__ranges::__size(__r2);

    // intersection is empty
    if (__n1 == 0 || __n2 == 0)
        return {__first1 + __n1, __first2 + __n2, __result};

    const std::size_t __result_size = __par_backend_hetero::__parallel_set_op<unseq_backend::_IntersectionTag>(
        _BackendTag{}, unseq_backend::_IntersectionTag{}, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::__get_subscription_view(__r1), oneapi::dpl::__ranges::__get_subscription_view(__r2),
        oneapi::dpl::__ranges::__get_subscription_view(__out_r), __comp, __proj1, __proj2);

    return {__first1 + __n1, __first2 + __n2, __result + __result_size};
}

//Dummy names to avoid kernel problems
template <typename Name>
struct __set_difference_copy_case_1;

template <typename _BackendTag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange,
          typename _Comp, typename _Proj1, typename _Proj2>
std::ranges::set_difference_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_set_difference(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2,
                         _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    const auto __first1 = std::ranges::begin(__r1);
    const auto __result = std::ranges::begin(__out_r);

    const auto __n1 = oneapi::dpl::__ranges::__size(__r1);

    // {} \ {2}: the difference is empty
    if (__n1 == 0)
        return {__first1, __result};

    // {1} \ {}: the difference is {1}
    if (oneapi::dpl::__ranges::__empty(__r2))
    {
        const auto __idx = oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_difference_copy_case_1>(
                std::forward<_ExecutionPolicy>(__exec)),
            oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{},
            oneapi::dpl::__ranges::__get_subscription_view(__r1),
            oneapi::dpl::__ranges::__get_subscription_view(__out_r));

        return {__first1 + __n1, __result + __idx};
    }

    const std::size_t __result_size = __par_backend_hetero::__parallel_set_op<unseq_backend::_DifferenceTag>(
        _BackendTag{}, unseq_backend::_DifferenceTag{}, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::__get_subscription_view(__r1),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_R2>(__r2)),
        oneapi::dpl::__ranges::__get_subscription_view(__out_r), __comp, __proj1, __proj2);

    return {__first1 + __n1, __result + __result_size};
}

//Dummy names to avoid kernel problems
template <typename Name>
struct __set_symmetric_difference_copy_case_1;

template <typename Name>
struct __set_symmetric_difference_copy_case_2;

template <typename _BackendTag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange,
          typename _Comp, typename _Proj1, typename _Proj2>
std::ranges::set_symmetric_difference_result<std::ranges::borrowed_iterator_t<_R1>,
                                             std::ranges::borrowed_iterator_t<_R2>,
                                             std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_set_symmetric_difference(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2,
                                   _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    const auto __first1 = std::ranges::begin(__r1);
    const auto __first2 = std::ranges::begin(__r2);
    const auto __result = std::ranges::begin(__out_r);

    const auto __n1 = oneapi::dpl::__ranges::__size(__r1);
    const auto __n2 = oneapi::dpl::__ranges::__size(__r2);

    if (__n1 == 0 && __n2 == 0)
        return {__first1, __first2, __result};

    //{1} is empty
    if (__n1 == 0)
    {
        const auto __idx = oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_symmetric_difference_copy_case_1>(
                std::forward<_ExecutionPolicy>(__exec)),
            oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{},
            oneapi::dpl::__ranges::__get_subscription_view(__r2),
            oneapi::dpl::__ranges::__get_subscription_view(__out_r));

        return {__first1, __first2 + __n2, __result + __idx};
    }

    //{2} is empty
    if (__n2 == 0)
    {
        const auto __idx = oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_symmetric_difference_copy_case_2>(
                std::forward<_ExecutionPolicy>(__exec)),
            oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{},
            oneapi::dpl::__ranges::__get_subscription_view(__r1),
            oneapi::dpl::__ranges::__get_subscription_view(__out_r));

        return {__first1 + __n1, __first2, __result + __idx};
    }

    const std::size_t __result_size = __par_backend_hetero::__parallel_set_op<unseq_backend::_SymmetricDifferenceTag>(
        _BackendTag{}, unseq_backend::_SymmetricDifferenceTag{}, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::__get_subscription_view(__r1), oneapi::dpl::__ranges::__get_subscription_view(__r2),
        oneapi::dpl::__ranges::__get_subscription_view(__out_r), __comp, __proj1, __proj2);

    return {__first1 + __n1, __first2 + __n2, __result + __result_size};
}

#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// sort
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Compare, typename _Proj>
void
__pattern_stable_sort(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp,
                      _Proj __proj)
{
    if (oneapi::dpl::__ranges::__size(__rng) >= 2)
    {
        __par_backend_hetero::__parallel_stable_sort(
            _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec),
            oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range>(__rng)), __comp, __proj)
            .__checked_deferrable_wait();
    }
}

#if _ONEDPL_CPP20_RANGES_PRESENT

template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _Comp, typename _Proj,
          typename _LeafSort = std::nullptr_t>
std::ranges::borrowed_iterator_t<_R>
__pattern_sort_ranges(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj,
                      _LeafSort = {})
{
    oneapi::dpl::__internal::__ranges::__pattern_stable_sort(__tag, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::views::all(__r), __comp, __proj);

    return std::ranges::begin(__r) + oneapi::dpl::__ranges::__size(__r);
}

#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// min_element
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Compare>
std::pair<oneapi::dpl::__internal::__difference_t<_Range>, oneapi::dpl::__internal::__value_t<_Range>>
__pattern_min_element_impl(_BackendTag __tag, _ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    assert(oneapi::dpl::__ranges::__size(__rng) > 0);

    using _IteratorValueType = typename ::std::iterator_traits<decltype(__rng.begin())>::value_type;
    using _IndexValueType = oneapi::dpl::__internal::__difference_t<_Range>;
    using _ReduceValueType = oneapi::dpl::__internal::tuple<_IndexValueType, _IteratorValueType>;

    // This operator doesn't track the lowest found index in case of equal min. or max. values. Thus, this operator is
    // not commutative.
    __pattern_min_element_reduce_fn<_ReduceValueType, _Compare> __reduce_fn{__comp};
    oneapi::dpl::__internal::__pattern_min_element_transform_fn<_ReduceValueType> __transform_fn;

    [[maybe_unused]] auto [__idx, __val] =
        oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType,
                                                                       ::std::false_type /*is_commutative*/>(
            __tag, ::std::forward<_ExecutionPolicy>(__exec), __reduce_fn, __transform_fn,
            unseq_backend::__no_init_value{}, // no initial value
            oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range>(__rng)))
            .get();

    return {__idx, __val};
}


template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_min_element(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    //If size == 1, result is the zero-indexed element. If size == 0, result is 0.
    if (oneapi::dpl::__ranges::__size(__rng) < 2)
        return 0;

    [[maybe_unused]] auto [__idx, __val] =
        __pattern_min_element_impl(_BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
                                   oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range>(__rng)), __comp);

    return __idx;
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
std::ranges::borrowed_iterator_t<_R>
__pattern_min_element(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    oneapi::dpl::__internal::__binary_op<_Comp, _Proj, _Proj> __comp_2{__comp, __proj, __proj};

    const auto __idx = oneapi::dpl::__internal::__ranges::__pattern_min_element(
        __tag, std::forward<_ExecutionPolicy>(__exec), oneapi::dpl::__ranges::views::all_read(__r), __comp_2);

    return {std::ranges::begin(__r) + __idx};
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
std::ranges::range_value_t<_R>
__pattern_min(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    oneapi::dpl::__internal::__binary_op<_Comp, _Proj, _Proj> __comp_2{__comp, __proj, __proj};

    [[maybe_unused]] const auto& [__idx, __val] =
        __pattern_min_element_impl(_BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::views::all_read(__r), __comp_2);

    return __val;
}

#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// minmax_element
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Compare>
std::pair<std::pair<oneapi::dpl::__internal::__difference_t<_Range>, oneapi::dpl::__internal::__value_t<_Range>>,
          std::pair<oneapi::dpl::__internal::__difference_t<_Range>, oneapi::dpl::__internal::__value_t<_Range>>>
__pattern_minmax_element_impl(_BackendTag, _ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    assert(oneapi::dpl::__ranges::__size(__rng) > 0);

    using _IteratorValueType = typename ::std::iterator_traits<decltype(__rng.begin())>::value_type;
    using _IndexValueType = oneapi::dpl::__internal::__difference_t<_Range>;
    using _ReduceValueType =
        oneapi::dpl::__internal::tuple<_IndexValueType, _IndexValueType, _IteratorValueType, _IteratorValueType>;

    // This operator doesn't track the lowest found index in case of equal min. values and the highest found index in
    // case of equal max. values. Thus, this operator is not commutative.
    oneapi::dpl::__internal::__pattern_minmax_element_reduce_fn<_Compare, _ReduceValueType> __reduce_fn{__comp};

    // TODO: Doesn't work with `zip_iterator`.
    //       In that case the first and the second arguments of `_ReduceValueType` will be
    //       a `tuple` of `difference_type`, not the `difference_type` itself.
    oneapi::dpl::__internal::__pattern_minmax_element_transform_fn<_ReduceValueType> __transform_fn;

    const auto& [__idx_min, __idx_max, __min, __max] =
        oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType,
                                                                       ::std::false_type /*is_commutative*/>(
            _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __reduce_fn, __transform_fn,
            unseq_backend::__no_init_value{}, // no initial value
            oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range>(__rng)))
            .get();

    return {{__idx_min, __min}, {__idx_max, __max}};
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Compare>
std::pair<oneapi::dpl::__internal::__difference_t<_Range>, oneapi::dpl::__internal::__difference_t<_Range>>
__pattern_minmax_element(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    //If size == 1, result is the zero-indexed element. If size == 0, result is 0.
    if (oneapi::dpl::__ranges::__size(__rng) < 2)
        return {0, 0};

    [[maybe_unused]] const auto& [__res_min, __res_max] = __pattern_minmax_element_impl(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range>(__rng)), __comp);

    [[maybe_unused]] const auto& [__idx_min, __min] = __res_min;
    [[maybe_unused]] const auto& [__idx_max, __max] = __res_max;

    return {__idx_min, __idx_max};
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
std::pair<std::ranges::borrowed_iterator_t<_R>, std::ranges::borrowed_iterator_t<_R>>
__pattern_minmax_element(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp,
                         _Proj __proj)
{
    oneapi::dpl::__internal::__binary_op<_Comp, _Proj, _Proj> __comp_2{__comp, __proj, __proj};

    const auto [__min_idx, __max_idx] =
        oneapi::dpl::__internal::__ranges::__pattern_minmax_element(__tag, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::views::all_read(__r), __comp_2);

    return {std::ranges::begin(__r) + __min_idx, std::ranges::begin(__r) + __max_idx};
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
std::pair<std::ranges::range_value_t<_R>, std::ranges::range_value_t<_R>>
__pattern_minmax(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    oneapi::dpl::__internal::__binary_op<_Comp, _Proj, _Proj> __comp_2{__comp, __proj, __proj};

    [[maybe_unused]] const auto& [__res_min, __res_max] =
        __pattern_minmax_element_impl(_BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
                                      oneapi::dpl::__ranges::__get_subscription_view(std::forward<_R>(__r)), __comp_2);

    [[maybe_unused]] const auto& [__idx_min, __min] = __res_min;
    [[maybe_unused]] const auto& [__idx_max, __max] = __res_max;

    return {__min, __max};
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
          typename _Proj2>
std::pair<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_R2>>
__pattern_mismatch(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred,
                   _Proj1 __proj1, _Proj2 __proj2)
{
    if (std::ranges::empty(__r1) || std::ranges::empty(__r2))
        return {std::ranges::begin(__r1), std::ranges::begin(__r2)};

    oneapi::dpl::__internal::__binary_op<_Pred, _Proj1, _Proj2> __bin_pred{__pred, __proj1, __proj2};

    using __bin_pred_type = decltype(__bin_pred);
    using _IndexType = std::make_unsigned_t<
        std::common_type_t<oneapi::dpl::__internal::__difference_t<_R1>, oneapi::dpl::__internal::__difference_t<_R2>>>;
    using _TagType = oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_IndexType>;
    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<oneapi::dpl::__internal::__not_pred<__bin_pred_type>>;
    using __size_calc = oneapi::dpl::__ranges::__min_size_calc;

    auto __idx = oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        _Predicate{oneapi::dpl::__internal::__not_pred<__bin_pred_type>(__bin_pred)}, _TagType{}, __size_calc{},
        oneapi::dpl::__ranges::views::all_read(__r1), oneapi::dpl::__ranges::views::all_read(__r2));

    return {std::ranges::begin(__r1) + __idx, std::ranges::begin(__r2) + __idx};
}

#endif //_ONEDPL_CPP20_RANGES_PRESENT

//------------------------------------------------------------------------
// reduce_by_segment
//------------------------------------------------------------------------

template <typename _Name>
struct __copy_keys_values_range_wrapper;

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

    const auto __n = oneapi::dpl::__ranges::__size(__keys);

    if (__n == 0)
        return 0;

    if (__n == 1)
    {
        __brick_copy<__hetero_tag<_BackendTag>> __copy_range{};

        oneapi::dpl::__internal::__ranges::__pattern_walk_n(
            __tag, oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy_keys_values_range_wrapper>(__exec),
            __copy_range,
            oneapi::dpl::__ranges::zip_view(
                oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range1>(__keys)),
                oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range2>(__values))),
            oneapi::dpl::__ranges::zip_view(
                oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range3>(__out_keys)),
                oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range4>(__out_values))));

        return 1;
    }

    return oneapi::dpl::__par_backend_hetero::__parallel_reduce_by_segment(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range1>(__keys)),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range2>(__values)),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range3>(__out_keys)),
        oneapi::dpl::__ranges::__get_subscription_view(std::forward<_Range4>(__out_values)), __binary_pred,
        __binary_op);
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_ALGORITHM_RANGES_IMPL_HETERO_H
