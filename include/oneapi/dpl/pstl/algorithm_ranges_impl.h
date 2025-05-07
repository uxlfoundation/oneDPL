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

#ifndef _ONEDPL_ALGORITHM_RANGES_IMPL_H
#define _ONEDPL_ALGORITHM_RANGES_IMPL_H

#if _ONEDPL_CPP20_RANGES_PRESENT

#include <ranges>
#include <utility>
#include <cassert>
#include <functional>
#include <type_traits>

#include "algorithm_fwd.h"
#include "execution_impl.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{
namespace __ranges
{

//---------------------------------------------------------------------------------------------------------------------
// pattern_for_each
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Fun>
void
__pattern_for_each(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Fun __f, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __f_1 =
        [__f, __proj](auto&& __val) { std::invoke(__f, std::invoke(__proj, std::forward<decltype(__val)>(__val)));};

    oneapi::dpl::__internal::__pattern_walk1(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r),
        std::ranges::begin(__r) + std::ranges::size(__r), __f_1);
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Fun>
void
__pattern_for_each(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _R&& __r, _Fun __f, _Proj __proj)
{
    std::ranges::for_each(std::forward<_R>(__r), __f, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_transform
//---------------------------------------------------------------------------------------------------------------------

template<typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _F, typename _Proj>
void
__pattern_transform(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r, _F __op,
                    _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});
    assert(std::ranges::size(__in_r) <= std::ranges::size(__out_r)); // for debug purposes only

    auto __unary_op = [__op, __proj](auto&& __val) {
        return std::invoke(__op, std::invoke(__proj, std::forward<decltype(__val)>(__val)));};

    oneapi::dpl::__internal::__pattern_walk2(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__in_r),
        std::ranges::begin(__in_r) + std::ranges::size(__in_r), std::ranges::begin(__out_r),
        oneapi::dpl::__internal::__transform_functor<decltype(__unary_op)>{std::move(__unary_op)});
}

template<typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _F, typename _Proj>
void
__pattern_transform(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _InRange&& __in_r, _OutRange&& __out_r,
                    _F __op, _Proj __proj)
{
    std::ranges::transform(std::forward<_InRange>(__in_r), std::ranges::begin(__out_r), __op, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_transform (binary vesrion)
//---------------------------------------------------------------------------------------------------------------------

template<typename _Tag, typename _ExecutionPolicy, typename _InRange1, typename _InRange2, typename _OutRange,
         typename _F, typename _Proj1, typename _Proj2>
void
__pattern_transform(_Tag __tag, _ExecutionPolicy&& __exec, _InRange1&& __in_r1, _InRange2&& __in_r2,
                    _OutRange&& __out_r, _F __binary_op, _Proj1 __proj1,_Proj2 __proj2)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __f = [__binary_op, __proj1, __proj2](auto&& __val1, auto&& __val2) {
        return std::invoke(__binary_op, std::invoke(__proj1, std::forward<decltype(__val1)>(__val1)),
            std::invoke(__proj2, std::forward<decltype(__val2)>(__val2)));};

    oneapi::dpl::__internal::__pattern_walk3(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__in_r1),
        std::ranges::begin(__in_r1) + std::ranges::size(__in_r1), std::ranges::begin(__in_r2),
        std::ranges::begin(__out_r), oneapi::dpl::__internal::__transform_functor<decltype(__f)>{std::move(__f)});
}

template<typename _ExecutionPolicy, typename _InRange1, typename _InRange2, typename _OutRange, typename _F,
         typename _Proj1, typename _Proj2>
void
__pattern_transform(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _InRange1&& __in_r1, _InRange2&& __in_r2, _OutRange&& __out_r,
                    _F __binary_op, _Proj1 __proj1, _Proj2 __proj2)
{
    std::ranges::transform(std::forward<_InRange1>(__in_r1), std::forward<_InRange2>(__in_r2),
                           std::ranges::begin(__out_r), __binary_op, __proj1, __proj2);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_find_if
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
auto
__pattern_find_if(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_1 = [__pred, __proj](auto&& __val) {
        return std::invoke(__pred, std::invoke(__proj, std::forward<decltype(__val)>(__val)));};

    return std::ranges::borrowed_iterator_t<_R>(oneapi::dpl::__internal::__pattern_find_if(__tag,
        std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r), std::ranges::begin(__r) +
        std::ranges::size(__r), __pred_1));
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
auto
__pattern_find_if(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _R&& __r, _Pred __pred, _Proj __proj)
{
    return std::ranges::find_if(std::forward<_R>(__r), __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// __pattern_find_first_of
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
          typename _Proj2>
auto
__pattern_find_first_of(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2,
                        _Pred __pred, _Proj1 __proj1, _Proj2 __proj2)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __bin_pred = [__pred, __proj1, __proj2](auto&& __val1, auto&& __val2) {
        return std::invoke(__pred, std::invoke(__proj1, std::forward<decltype(__val1)>(__val1)),
                           std::invoke(__proj2, std::forward<decltype(__val2)>(__val2)));
    };

    return std::ranges::borrowed_iterator_t<_R1>(oneapi::dpl::__internal::__pattern_find_first_of(
        __tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r1),
        std::ranges::begin(__r1) + std::ranges::size(__r1), std::ranges::begin(__r2),
        std::ranges::begin(__r2) + std::ranges::size(__r2), __bin_pred));
}

template <typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1, typename _Proj2>
auto
__pattern_find_first_of(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2,
                        _Pred __pred, _Proj1 __proj1, _Proj2 __proj2)
{
    return std::ranges::find_first_of(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __pred, __proj1, __proj2);
}

//---------------------------------------------------------------------------------------------------------------------
// __pattern_find_end
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
          typename _Proj2>
auto
__pattern_find_end(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred, _Proj1 __proj1,
                   _Proj2 __proj2)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __bin_pred = [__pred, __proj1, __proj2](auto&& __val1, auto&& __val2) {
        return std::invoke(__pred, std::invoke(__proj1, std::forward<decltype(__val1)>(__val1)),
                           std::invoke(__proj2, std::forward<decltype(__val2)>(__val2)));
    };

    auto __last1 = std::ranges::begin(__r1) + std::ranges::size(__r1);
    if (std::ranges::empty(__r2))
        return std::ranges::borrowed_subrange_t<_R1>(__last1, __last1);

    auto __it = oneapi::dpl::__internal::__pattern_find_end(__tag,
        std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r1), __last1, std::ranges::begin(__r2),
        std::ranges::begin(__r2) + std::ranges::size(__r2), __bin_pred);

    return std::ranges::borrowed_subrange_t<_R1>(__it, __it + (__it == __last1 ? 0 : std::ranges::size(__r2)));
}

template <typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1, typename _Proj2>
auto
__pattern_find_end(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2,
                   _Pred __pred, _Proj1 __proj1, _Proj2 __proj2)
{
    return std::ranges::find_end(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __pred, __proj1, __proj2);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_any_of
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
bool
__pattern_any_of(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_1 = [__pred, __proj](auto&& __val) {
        return std::invoke(__pred, std::invoke(__proj, std::forward<decltype(__val)>(__val)));};
    return oneapi::dpl::__internal::__pattern_any_of(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + std::ranges::size(__r), __pred_1);
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
bool
__pattern_any_of(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _R&& __r, _Pred __pred, _Proj __proj)
{
    return std::ranges::any_of(std::forward<_R>(__r), __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_adjacent_find
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
auto
__pattern_adjacent_find_ranges(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred,
                        _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_2 = [__pred, __proj](auto&& __val, auto&& __next) { return std::invoke(__pred, std::invoke(__proj,
        std::forward<decltype(__val)>(__val)), std::invoke(__proj, std::forward<decltype(__next)>(__next)));};

    auto __res = oneapi::dpl::__internal::__pattern_adjacent_find(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + std::ranges::size(__r), __pred_2,
        oneapi::dpl::__internal::__first_semantic());
    return std::ranges::borrowed_iterator_t<_R>(__res);
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
auto
__pattern_adjacent_find_ranges(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _R&& __r, _Pred __pred, _Proj __proj)
{
    return std::ranges::adjacent_find(std::forward<_R>(__r), __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_search
//---------------------------------------------------------------------------------------------------------------------

template<typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
         typename _Proj2>
auto
__pattern_search(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred,
                 _Proj1 __proj1, _Proj2 __proj2)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_2 = [__pred, __proj1, __proj2](auto&& __val1, auto&& __val2) { return std::invoke(__pred,
        std::invoke(__proj1, std::forward<decltype(__val1)>(__val1)),
        std::invoke(__proj2, std::forward<decltype(__val2)>(__val2)));};

    auto __res = oneapi::dpl::__internal::__pattern_search(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r1), std::ranges::begin(__r1) + std::ranges::size(__r1), std::ranges::begin(__r2),
        std::ranges::begin(__r2) + std::ranges::size(__r2), __pred_2);

    return std::ranges::borrowed_subrange_t<_R1>(__res, __res == std::ranges::end(__r1)
        ? __res : __res + std::ranges::size(__r2));
}

template<typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
         typename _Proj2>
auto
__pattern_search(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _R1&& __r1, _R2&& __r2, _Pred __pred, _Proj1 __proj1, _Proj2 __proj2)
{
    return std::ranges::search(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __pred, __proj1, __proj2);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_search_n
//---------------------------------------------------------------------------------------------------------------------

template<typename _Tag, typename _ExecutionPolicy, typename _R, typename _T, typename _Pred, typename _Proj>
auto
__pattern_search_n(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r,
                   std::ranges::range_difference_t<_R> __count, const _T& __value, _Pred __pred, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_2 = [__pred, __proj](auto&& __val1, auto&& __val2) { return std::invoke(__pred,
        std::invoke(__proj, std::forward<decltype(__val1)>(__val1)), std::forward<decltype(__val2)>(__val2));};

    auto __res = oneapi::dpl::__internal::__pattern_search_n(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + std::ranges::size(__r), __count, __value, __pred_2);

    return std::ranges::borrowed_subrange_t<_R>(__res, __res == std::ranges::end(__r) ? __res : __res + __count);
}

template<typename _ExecutionPolicy, typename _R, typename _T, typename _Pred, typename _Proj>
auto
__pattern_search_n(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _R&& __r, std::ranges::range_difference_t<_R> __count, const _T& __value,
                   _Pred __pred, _Proj __proj)
{
    return std::ranges::search_n(std::forward<_R>(__r), __count, __value, __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_count_if
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
std::ranges::range_difference_t<_R>
__pattern_count_if(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_1 = [__pred, __proj](auto&& __val) { return std::invoke(__pred, std::invoke(__proj,
        std::forward<decltype(__val)>(__val)));};
    return oneapi::dpl::__internal::__pattern_count(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + std::ranges::size(__r), __pred_1);
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
std::ranges::range_difference_t<_R>
__pattern_count_if(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _R&& __r, _Pred __pred, _Proj __proj)
{
    return std::ranges::count_if(std::forward<_R>(__r), __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_count
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _T, typename _Proj>
std::ranges::range_difference_t<_R>
__pattern_count(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, const _T& __value, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    oneapi::dpl::__internal::__count_fn_pred<_T, _Proj> __pred{__value, __proj};

    return oneapi::dpl::__internal::__pattern_count(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + std::ranges::size(__r), __pred);
}

template <typename _ExecutionPolicy, typename _R, typename _T, typename _Proj>
std::ranges::range_difference_t<_R>
__pattern_count(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r, const _T& __value,
                _Proj __proj)
{
    return std::ranges::count(std::forward<_R>(__r), __value, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_equal
//---------------------------------------------------------------------------------------------------------------------
template<typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
         typename _Proj2>
bool
__pattern_equal(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred,
                _Proj1 __proj1, _Proj2 __proj2)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_2 = [__pred, __proj1, __proj2](auto&& __val1, auto&& __val2)
        { return std::invoke(__pred, std::invoke(__proj1, std::forward<decltype(__val1)>(__val1)),
        std::invoke(__proj2, std::forward<decltype(__val2)>(__val2)));};

    return oneapi::dpl::__internal::__pattern_equal(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r1), std::ranges::begin(__r1) + std::ranges::size(__r1), std::ranges::begin(__r2),
        std::ranges::begin(__r2) + std::ranges::size(__r2), __pred_2);
}

template<typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
         typename _Proj2>
bool
__pattern_equal(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _R1&& __r1, _R2&& __r2, _Pred __pred, _Proj1 __proj1, _Proj2 __proj2)
{
    return std::ranges::equal(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __pred, __proj1, __proj2);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_is_sorted
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
bool
__pattern_is_sorted(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_2 = [__comp, __proj](auto&& __val1, auto&& __val2) { return std::invoke(__comp, std::invoke(__proj,
        std::forward<decltype(__val1)>(__val1)), std::invoke(__proj, std::forward<decltype(__val2)>(__val2)));};

    return oneapi::dpl::__internal::__pattern_adjacent_find(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + std::ranges::size(__r),
        oneapi::dpl::__internal::__reorder_pred(__pred_2), oneapi::dpl::__internal::__or_semantic()) == __r.end();
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
bool
__pattern_is_sorted(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _R&& __r, _Comp __comp, _Proj __proj)
{
    return std::ranges::is_sorted(std::forward<_R>(__r), __comp, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_sort_ranges
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp, typename _LeafSort>
auto
__pattern_sort_ranges(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj,
                      _LeafSort __leaf_sort)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __comp_2 = [__comp, __proj](auto&& __val1, auto&& __val2) { return std::invoke(__comp, std::invoke(__proj,
        std::forward<decltype(__val1)>(__val1)), std::invoke(__proj, std::forward<decltype(__val2)>(__val2)));};
    oneapi::dpl::__internal::__pattern_sort(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r),
                                            std::ranges::begin(__r) + std::ranges::size(__r), __comp_2, __leaf_sort);

    return std::ranges::borrowed_iterator_t<_R>(std::ranges::begin(__r) + std::ranges::size(__r));
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp, typename _LeafSort>
auto
__pattern_sort_ranges(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r, _Comp __comp,
                      _Proj __proj, _LeafSort __leaf_sort)
{
    return __leaf_sort(std::forward<_R>(__r), __comp, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_min_element
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
auto
__pattern_min_element(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __comp_2 = [__comp, __proj](auto&& __val1, auto&& __val2) { return std::invoke(__comp, std::invoke(__proj,
        std::forward<decltype(__val1)>(__val1)), std::invoke(__proj, std::forward<decltype(__val2)>(__val2)));};

    return oneapi::dpl::__internal::__pattern_min_element(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r),
        std::ranges::begin(__r) + std::ranges::size(__r), __comp_2);
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
auto
__pattern_min_element(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _R&& __r, _Comp __comp, _Proj __proj)
{
    return std::ranges::min_element(std::forward<_R>(__r), __comp, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_min
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
std::ranges::range_value_t<_R>
__pattern_min(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    return *__pattern_min_element(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __comp, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_minmax_element
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
auto
__pattern_minmax_element(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __comp_2 = [__comp, __proj](auto&& __val1, auto&& __val2) { return std::invoke(__comp, std::invoke(__proj,
        std::forward<decltype(__val1)>(__val1)), std::invoke(__proj, std::forward<decltype(__val2)>(__val2)));};

    return oneapi::dpl::__internal::__pattern_minmax_element(
        __tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r),
        std::ranges::begin(__r) + std::ranges::size(__r), __comp_2);
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
auto
__pattern_minmax_element(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r, _Comp __comp,
                         _Proj __proj)
{
    return std::ranges::minmax_element(std::forward<_R>(__r), __comp, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// __pattern_minmax
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
std::pair<std::ranges::range_value_t<_R>, std::ranges::range_value_t<_R>>
__pattern_minmax(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    auto [__it_min, __it_max] =
        __pattern_minmax_element(__tag, std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __comp, __proj);

    return {*__it_min, *__it_max};
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_copy
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
void
__pattern_copy(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    assert(std::ranges::size(__in_r) <= std::ranges::size(__out_r)); // for debug purposes only

    oneapi::dpl::__internal::__pattern_walk2_brick(
        __tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__in_r),
        std::ranges::begin(__in_r) + std::ranges::size(__in_r), std::ranges::begin(__out_r),
        oneapi::dpl::__internal::__brick_copy<decltype(__tag)>{});
}

template<typename _ExecutionPolicy, typename _InRange, typename _OutRange>
void
__pattern_copy(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _InRange&& __in_r, _OutRange&& __out_r)
{
    std::ranges::copy(std::forward<_InRange>(__in_r), std::ranges::begin(__out_r));
}
//---------------------------------------------------------------------------------------------------------------------
// pattern_copy_if
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _Pred,
          typename _Proj>
auto
__pattern_copy_if_ranges(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r, _Pred __pred, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_1 = [__pred, __proj](auto&& __val) { return std::invoke(__pred, std::invoke(__proj,
        std::forward<decltype(__val)>(__val)));};

    auto __res_idx = oneapi::dpl::__internal::__pattern_copy_if(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__in_r), std::ranges::begin(__in_r) + std::ranges::size(__in_r),
        std::ranges::begin(__out_r), __pred_1) - std::ranges::begin(__out_r);

    using __return_type = std::ranges::copy_if_result<std::ranges::borrowed_iterator_t<_InRange>,
        std::ranges::borrowed_iterator_t<_OutRange>>;

    return __return_type{std::ranges::begin(__in_r) + std::ranges::size(__in_r), std::ranges::begin(__out_r) + __res_idx};
}

template<typename _ExecutionPolicy, typename _InRange, typename _OutRange, typename _Pred, typename _Proj>
auto
__pattern_copy_if_ranges(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&&, _InRange&& __in_r, _OutRange&& __out_r,
                         _Pred __pred, _Proj __proj)
{
    return std::ranges::copy_if(std::forward<_InRange>(__in_r), std::ranges::begin(__out_r), __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// __pattern_fill
//---------------------------------------------------------------------------------------------------------------------
template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _T>
std::ranges::borrowed_iterator_t<_R>
__pattern_fill(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, const _T& __value)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    const auto __first = std::ranges::begin(__r);
    const auto __last = __first + std::ranges::size(__r);
    oneapi::dpl::__internal::__pattern_fill(__tag, std::forward<_ExecutionPolicy>(__exec), __first, __last, __value);

    return {__last};
}

template <typename _ExecutionPolicy, typename _R, typename _T>
std::ranges::borrowed_iterator_t<_R>
__pattern_fill(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r, const _T& __value)
{
    return std::ranges::fill(std::forward<_R>(__r), __value);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_merge
//---------------------------------------------------------------------------------------------------------------------

template<typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
         typename _Proj1, typename _Proj2>
auto
__pattern_merge(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp,
                _Proj1 __proj1, _Proj2 __proj2)
{
    using __return_type =
        std::ranges::merge_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_R2>,
                                  std::ranges::borrowed_iterator_t<_OutRange>>;

    auto __comp_2 = [__comp, __proj1, __proj2](auto&& __val1, auto&& __val2) { return std::invoke(__comp,
        std::invoke(__proj1, std::forward<decltype(__val1)>(__val1)), std::invoke(__proj2,
        std::forward<decltype(__val2)>(__val2)));};

    using _Index1 = std::ranges::range_difference_t<_R1>;
    using _Index2 = std::ranges::range_difference_t<_R2>;
    using _Index3 = std::ranges::range_difference_t<_OutRange>;

    const _Index1 __n_1 = std::ranges::size(__r1);
    const _Index2 __n_2 = std::ranges::size(__r2);
    const _Index3 __n_out = std::min<_Index3>(__n_1 + __n_2, std::ranges::size(__out_r));

    auto __it_1 = std::ranges::begin(__r1);
    auto __it_2 = std::ranges::begin(__r2);
    auto __it_out = std::ranges::begin(__out_r);

    if (__n_out == 0)
        return __return_type{__it_1, __it_2, __it_out};

    // Parallel and serial versions of ___merge_path_out_lim merge the 1st sequence and the 2nd sequence in "reverse order":
    // the identical elements from the 2nd sequence are merged first.
    // So, the call to ___merge_path_out_lim swaps the order of sequences.
    std::pair __res = ___merge_path_out_lim(__tag, std::forward<_ExecutionPolicy>(__exec), __it_2, __n_2, __it_1, __n_1,
                                            __it_out, __n_out, __comp_2);

    return __return_type{__res.second, __res.first, __it_out + __n_out};
}

//---------------------------------------------------------------------------------------------------------------------
// __pattern_mismatch
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
          typename _Proj2>
auto
__pattern_mismatch(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred, _Proj1 __proj1,
                   _Proj2 __proj2)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __bin_pred = [__pred, __proj1, __proj2](auto&& __val1, auto&& __val2) {
        return std::invoke(__pred, std::invoke(__proj1, std::forward<decltype(__val1)>(__val1)),
                           std::invoke(__proj2, std::forward<decltype(__val2)>(__val2)));
    };

    return oneapi::dpl::__internal::__pattern_mismatch(
        __tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r1),
        std::ranges::begin(__r1) + std::ranges::size(__r1), std::ranges::begin(__r2),
        std::ranges::begin(__r2) + std::ranges::size(__r2), __bin_pred);
}

template <typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1, typename _Proj2>
auto
__pattern_mismatch(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R1&& __r1, _R2&& __r2, _Pred __pred,
                   _Proj1 __proj1, _Proj2 __proj2)
{
    return std::ranges::mismatch(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __pred, __proj1, __proj2);
}

//---------------------------------------------------------------------------------------------------------------------
// __pattern_remove_if
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
auto
__pattern_remove_if(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    auto __pred_1 = [__pred, __proj](auto&& __val) {
        return std::invoke(__pred, std::invoke(__proj, std::forward<decltype(__val)>(__val)));
    };

    auto __end = std::ranges::begin(__r) + std::ranges::size(__r);

    auto __it = oneapi::dpl::__internal::__pattern_remove_if(__tag, std::forward<_ExecutionPolicy>(__exec),
                                                             std::ranges::begin(__r), __end, __pred_1);

    return std::ranges::borrowed_subrange_t<_R>(__it, __end);
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
auto
__pattern_remove_if(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred,
                    _Proj __proj)
{
    return std::ranges::remove_if(std::forward<_R>(__r), __pred, __proj);
}

template <typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
void
__pattern_move(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __r, _OutRange&& __out_r)
{
    auto __end = std::ranges::begin(__r) + std::ranges::size(__r);
    oneapi::dpl::__internal::__pattern_walk2_brick(__tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                   std::ranges::begin(__r), __end, std::ranges::begin(__out_r),
                                                   oneapi::dpl::__internal::__brick_move<decltype(__tag)>{});
}

template <typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
void
__pattern_move(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&& __exec, _InRange&& __r,
               _OutRange&& __out_r)
{
    std::ranges::move(std::forward<_InRange>(__r), std::forward<_OutRange>(__out_r));
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_CPP20_RANGES_PRESENT

#endif // _ONEDPL_ALGORITHM_RANGES_IMPL_H
