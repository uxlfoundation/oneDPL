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

#    include <ranges>
#    include <utility>
#    include <cassert>
#    include <functional>
#    include <type_traits>

#    include "execution_impl.h"
#    include "algorithm_impl.h"

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

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Fun, typename _Proj>
void
__pattern_for_each(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Fun __f, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __f_1 =
        [__f, __proj](auto&& __val) { std::invoke(__f, std::invoke(__proj, std::forward<decltype(__val)>(__val)));};

    oneapi::dpl::__internal::__pattern_walk1(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r),
        std::ranges::begin(__r) + std::ranges::size(__r), __f_1);
}

template <typename _ExecutionPolicy, typename _R, typename _Fun, typename _Proj>
void
__pattern_for_each(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r, _Fun __f, _Proj __proj)
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
std::ranges::borrowed_iterator_t<_R>
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
std::ranges::borrowed_iterator_t<_R>
__pattern_find_if(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r, _Pred __pred, _Proj __proj)
{
    return std::ranges::find_if(std::forward<_R>(__r), __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// __pattern_find_first_of
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
          typename _Proj2>
std::ranges::borrowed_iterator_t<_R1>
__pattern_find_first_of(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred, _Proj1 __proj1,
                        _Proj2 __proj2)
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
std::ranges::borrowed_iterator_t<_R1>
__pattern_find_first_of(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R1&& __r1, _R2&& __r2,
                        _Pred __pred, _Proj1 __proj1, _Proj2 __proj2)
{
    return std::ranges::find_first_of(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __pred, __proj1, __proj2);
}

//---------------------------------------------------------------------------------------------------------------------
// __pattern_find_end
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
          typename _Proj2>
std::ranges::borrowed_subrange_t<_R1>
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
std::ranges::borrowed_subrange_t<_R1>
__pattern_find_end(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R1&& __r1, _R2&& __r2, _Pred __pred,
                   _Proj1 __proj1, _Proj2 __proj2)
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
std::ranges::borrowed_iterator_t<_R>
__pattern_adjacent_find_ranges(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    oneapi::dpl::__internal::__compare<_Pred, _Proj> __pred_2{__pred, __proj};

    auto __res = oneapi::dpl::__internal::__pattern_adjacent_find(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + std::ranges::size(__r), __pred_2,
        oneapi::dpl::__internal::__first_semantic());
    return std::ranges::borrowed_iterator_t<_R>(__res);
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
std::ranges::borrowed_iterator_t<_R>
__pattern_adjacent_find_ranges(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r, _Pred __pred,
                               _Proj __proj)
{
    return std::ranges::adjacent_find(std::forward<_R>(__r), __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_search
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1,
          typename _Proj2>
std::ranges::borrowed_subrange_t<_R1>
__pattern_search(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred, _Proj1 __proj1,
                 _Proj2 __proj2)
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

template <typename _ExecutionPolicy, typename _R1, typename _R2, typename _Pred, typename _Proj1, typename _Proj2>
std::ranges::borrowed_subrange_t<_R1>
__pattern_search(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R1&& __r1, _R2&& __r2, _Pred __pred,
                 _Proj1 __proj1, _Proj2 __proj2)
{
    return std::ranges::search(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __pred, __proj1, __proj2);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_search_n
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _T, typename _Pred, typename _Proj>
std::ranges::borrowed_subrange_t<_R>
__pattern_search_n(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, std::ranges::range_difference_t<_R> __count,
                   const _T& __value, _Pred __pred, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __pred_2 = [__pred, __proj](auto&& __val1, auto&& __val2) { return std::invoke(__pred,
        std::invoke(__proj, std::forward<decltype(__val1)>(__val1)), std::forward<decltype(__val2)>(__val2));};

    auto __res = oneapi::dpl::__internal::__pattern_search_n(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + std::ranges::size(__r), __count, __value, __pred_2);

    return std::ranges::borrowed_subrange_t<_R>(__res, __res == std::ranges::end(__r) ? __res : __res + __count);
}

template <typename _ExecutionPolicy, typename _R, typename _T, typename _Pred, typename _Proj>
std::ranges::borrowed_subrange_t<_R>
__pattern_search_n(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r,
                   std::ranges::range_difference_t<_R> __count, const _T& __value, _Pred __pred, _Proj __proj)
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

    auto __pred_2 = [__comp, __proj](auto&& __val1, auto&& __val2) {
        return __comp(std::invoke(__proj, std::forward<decltype(__val1)>(__val1)),
                      std::invoke(__proj, std::forward<decltype(__val2)>(__val2)));
    };

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
std::ranges::borrowed_iterator_t<_R>
__pattern_sort_ranges(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj,
                      _LeafSort __leaf_sort)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __comp_2 = [__comp, __proj](auto&& __val1, auto&& __val2) {
        return __comp(std::invoke(__proj, std::forward<decltype(__val1)>(__val1)),
                      std::invoke(__proj, std::forward<decltype(__val2)>(__val2)));
    };
    oneapi::dpl::__internal::__pattern_sort(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r),
                                            std::ranges::begin(__r) + std::ranges::size(__r), __comp_2, __leaf_sort);

    return std::ranges::borrowed_iterator_t<_R>(std::ranges::begin(__r) + std::ranges::size(__r));
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp, typename _LeafSort>
std::ranges::borrowed_iterator_t<_R>
__pattern_sort_ranges(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r, _Comp __comp,
                      _Proj __proj, _LeafSort __leaf_sort)
{
    return __leaf_sort(std::forward<_R>(__r), __comp, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_min_element
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
std::ranges::borrowed_iterator_t<_R>
__pattern_min_element(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __comp_2 = [__comp, __proj](auto&& __val1, auto&& __val2) {
        return __comp(std::invoke(__proj, std::forward<decltype(__val1)>(__val1)),
                      std::invoke(__proj, std::forward<decltype(__val2)>(__val2)));
    };
    return oneapi::dpl::__internal::__pattern_min_element(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r),
        std::ranges::begin(__r) + std::ranges::size(__r), __comp_2);
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
std::ranges::borrowed_iterator_t<_R>
__pattern_min_element(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r, _Comp __comp,
                      _Proj __proj)
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

    auto __comp_2 = [__comp, __proj](auto&& __val1, auto&& __val2) {
        return __comp(std::invoke(__proj, std::forward<decltype(__val1)>(__val1)),
                      std::invoke(__proj, std::forward<decltype(__val2)>(__val2)));
    };
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

    return __last;
}

template <typename _ExecutionPolicy, typename _R, typename _T>
std::ranges::borrowed_iterator_t<_R>
__pattern_fill(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r, const _T& __value)
{
    return std::ranges::fill(std::forward<_R>(__r), __value);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_merge_ranges
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
          typename _Proj1, typename _Proj2>
auto
__pattern_merge_ranges(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp,
                       _Proj1 __proj1, _Proj2 __proj2)
{
    using __return_type =
        std::ranges::merge_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_R2>,
                                  std::ranges::borrowed_iterator_t<_OutRange>>;

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

    auto [__res1, __res2] = ___merge_path_out_lim(__tag, std::forward<_ExecutionPolicy>(__exec), __it_1, __n_1, __it_2,
                                                  __n_2, __it_out, __n_out, __comp, __proj1, __proj2);

    return __return_type{__res1, __res2, __it_out + __n_out};
}

//---------------------------------------------------------------------------------------------------------------------
// includes
//---------------------------------------------------------------------------------------------------------------------

template <typename _R1, typename _R2, typename _Comp, typename _Proj1, typename _Proj2>
bool
__brick_includes(_R1&& __r1, _R2&& __r2, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2,
                 /*__is_vector=*/std::false_type) noexcept
{
    return std::ranges::includes(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __comp, __proj1, __proj2);
}

template <typename _R1, typename _R2, typename _Comp, typename _Proj1, typename _Proj2>
bool
__brick_includes(_R1&& __r1, _R2&& __r2, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2,
                 /*__is_vector=*/std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::ranges::includes(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __comp, __proj1, __proj2);
}

template <typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Comp, typename _Proj1,
          typename _Proj2>
bool
__pattern_includes(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Comp __comp, _Proj1 __proj1,
                   _Proj2 __proj2)
{
    static_assert(__is_serial_tag_v<_Tag>);

    return __brick_includes(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __comp, __proj1, __proj2,
                            typename _Tag::__is_vector{});
}

template <class _IsVector, typename _ExecutionPolicy, typename _R1, typename _R2, typename _Comp, typename _Proj1,
          typename _Proj2>
bool
__pattern_includes(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Comp __comp,
                   _Proj1 __proj1, _Proj2 __proj2)
{
    using _RandomAccessIterator2 = std::ranges::iterator_t<_R2>;

    const auto __n1 = std::ranges::size(__r1);
    const auto __n2 = std::ranges::size(__r2);

    // use serial algorithm
    if (__n1 + __n2 <= oneapi::dpl::__internal::__set_algo_cut_off)
        return std::ranges::includes(std::forward<_R1>(__r1), std::forward<_R2>(__r2), __comp, __proj1, __proj2);

    auto __first1 = std::ranges::begin(__r1);
    auto __last1 = __first1 + __n1;
    auto __first2 = std::ranges::begin(__r2);
    auto __last2 = __first2 + __n2;

    using _DifferenceType1 = typename std::iterator_traits<decltype(__first1)>::difference_type;
    using _DifferenceType2 = typename std::iterator_traits<decltype(__first2)>::difference_type;

    if (__first2 == __last2)
        return true;

    //optimization; {1} - the first sequence, {2} - the second sequence
    //{1} is empty or size_of{2} > size_of{1}
    if (__first1 == __last1 || __last2 - __first2 > __last1 - __first1 ||
        // {1}:     [**********]     or   [**********]
        // {2}: [***********]                   [***********]
        std::invoke(__comp, std::invoke(__proj2, *__first2), std::invoke(__proj1, *__first1)) ||
        std::invoke(__comp, std::invoke(__proj1, *(__last1 - 1)), std::invoke(__proj2, *(__last2 - 1))))
        return false;

    __first1 += oneapi::dpl::__internal::__pstl_lower_bound(__first1, _DifferenceType1{0}, __last1 - __first1,
                                                            std::invoke(__proj2, *__first2), __comp, __proj1);
    if (__first1 == __last1)
        return false;

    if (__last2 - __first2 == 1)
        return !std::invoke(__comp, std::invoke(__proj1, *__first1), std::invoke(__proj2, *__first2)) &&
               !std::invoke(__comp, std::invoke(__proj2, *__first2), std::invoke(__proj1, *__first1));

    return !__internal::__parallel_or(
        __tag, std::forward<_ExecutionPolicy>(__exec), __first2, __last2,
        [__first1, __last1, __first2, __last2, __comp, __proj1, __proj2](_RandomAccessIterator2 __i,
                                                                         _RandomAccessIterator2 __j) {
            assert(__j > __i);

            //1. moving boundaries to "consume" subsequence of equal elements
            auto __is_equal_sorted = [&__comp, __proj2](_RandomAccessIterator2 __a,
                                                        _RandomAccessIterator2 __b) -> bool {
                //enough one call of __comp due to compared couple belongs to one sorted sequence
                return !std::invoke(__comp, std::invoke(__proj2, *__a), std::invoke(__proj2, *__b));
            };

            //1.1 left bound, case "aaa[aaaxyz...]" - searching "x"
            if (__i > __first2 && __is_equal_sorted(__i - 1, __i))
            {
                //whole subrange continues to have equal elements - return "no op"
                if (__is_equal_sorted(__i, __j - 1))
                    return false;

                __i += oneapi::dpl::__internal::__pstl_upper_bound(__i, _DifferenceType2{0}, __last2 - __i,
                                                                   std::invoke(__proj2, *__i), __comp, __proj2);
            }

            //1.2 right bound, case "[...aaa]aaaxyz" - searching "x"
            if (__j < __last2 && __is_equal_sorted(__j - 1, __j))
                __j += oneapi::dpl::__internal::__pstl_upper_bound(__j, _DifferenceType2{0}, __last2 - __j,
                                                                   std::invoke(__proj2, *__j), __comp, __proj2);

            //2. testing is __a subsequence of the second range included into the first range
            auto __b = __first1 +
                       oneapi::dpl::__internal::__pstl_lower_bound(__first1, _DifferenceType1{0}, __last1 - __first1,
                                                                   std::invoke(__proj2, *__i), __comp, __proj1);

            return !std::ranges::includes(__b, __last1, __i, __j, __comp, __proj1, __proj2);
        });
}

//---------------------------------------------------------------------------------------------------------------------
// set_union
//---------------------------------------------------------------------------------------------------------------------

template <typename _R1, typename _R2, typename _OutRange, typename _Comp, typename _Proj1, typename _Proj2>
auto
__brick_set_union(_R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2,
                  /*__is_vector=*/std::false_type) noexcept
{
    return std::ranges::set_union(std::forward<_R1>(__r1), std::forward<_R2>(__r2), std::ranges::begin(__out_r), __comp,
                                  __proj1, __proj2);
}

template <typename _R1, typename _R2, typename _OutRange, typename _Comp, typename _Proj1, typename _Proj2>
auto
__brick_set_union(_R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2,
                  /*__is_vector=*/std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::ranges::set_union(std::forward<_R1>(__r1), std::forward<_R2>(__r2), std::ranges::begin(__out_r), __comp,
                                  __proj1, __proj2);
}

template <typename _R1, typename _R2, typename _OutRange>
using __pattern_set_union_return_t =
    std::ranges::set_union_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_R2>,
                                  std::ranges::borrowed_iterator_t<_OutRange>>;

template <typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
          typename _Proj1, typename _Proj2>
__pattern_set_union_return_t<_R1, _R2, _OutRange>
__pattern_set_union(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp,
                    _Proj1 __proj1, _Proj2 __proj2)
{
    static_assert(__is_serial_tag_v<_Tag>);

    return __brick_set_union(std::forward<_R1>(__r1), std::forward<_R2>(__r2), std::forward<_OutRange>(__out_r), __comp,
                             __proj1, __proj2, typename _Tag::__is_vector{});
}

template <class _IsVector, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
          typename _Proj1, typename _Proj2>
__pattern_set_union_return_t<_R1, _R2, _OutRange>
__pattern_set_union(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2,
                    _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    using _RandomAccessIterator1 = std::ranges::iterator_t<_R1>;
    using _RandomAccessIterator2 = std::ranges::iterator_t<_R2>;
    using _Tp = std::ranges::range_value_t<_OutRange>;

    const auto __n1 = std::ranges::size(__r1);
    const auto __n2 = std::ranges::size(__r2);

    // use serial algorithm
    if (__n1 + __n2 <= oneapi::dpl::__internal::__set_algo_cut_off)
        return std::ranges::set_union(__r1, __r2, std::begin(__out_r), __comp, __proj1, __proj2);

    auto __first1 = std::ranges::begin(__r1);
    auto __last1 = __first1 + __n1;
    auto __first2 = std::ranges::begin(__r2);
    auto __last2 = __first2 + __n2;
    auto __result = std::ranges::begin(__out_r);

    auto __out_last = oneapi::dpl::__internal::__parallel_set_union_op(
        __tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2, __result,
        [](_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
           _RandomAccessIterator2 __last2, _Tp* __result, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2) {
            return oneapi::dpl::__utils::__set_union_construct(
                __first1, __last1, __first2, __last2, __result,
                oneapi::dpl::__internal::__BrickCopyConstruct<_IsVector>(), __comp, __proj1, __proj2);
        },
        __comp, __proj1, __proj2);

    return __pattern_set_union_return_t<_R1, _R2, _OutRange>{__first1 + __n1, __first2 + __n2,
                                                             __result + (__out_last - __result)};
}

//---------------------------------------------------------------------------------------------------------------------
// set_intersection
//---------------------------------------------------------------------------------------------------------------------

template <typename _R1, typename _R2, typename _OutRange, typename _Comp, typename _Proj1, typename _Proj2>
auto
__brick_set_intersection(_R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2,
                         /*__is_vector=*/std::false_type) noexcept
{
    return std::ranges::set_intersection(std::forward<_R1>(__r1), std::forward<_R2>(__r2), std::ranges::begin(__out_r),
                                         __comp, __proj1, __proj2);
}

template <typename _R1, typename _R2, typename _OutRange, typename _Comp, typename _Proj1, typename _Proj2>
auto
__brick_set_intersection(_R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2,
                         /*__is_vector=*/std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::ranges::set_intersection(std::forward<_R1>(__r1), std::forward<_R2>(__r2), std::ranges::begin(__out_r),
                                         __comp, __proj1, __proj2);
}

template <typename _R1, typename _R2, typename _OutRange>
using __pattern_set_intersection_return_t =
    std::ranges::set_intersection_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_R2>,
                                         std::ranges::borrowed_iterator_t<_OutRange>>;

template <typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
          typename _Proj1, typename _Proj2>
__pattern_set_intersection_return_t<_R1, _R2, _OutRange>
__pattern_set_intersection(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _OutRange&& __out_r,
                           _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    static_assert(__is_serial_tag_v<_Tag>);

    return __brick_set_intersection(std::forward<_R1>(__r1), std::forward<_R2>(__r2), std::forward<_OutRange>(__out_r),
                                    __comp, __proj1, __proj2, typename _Tag::__is_vector{});
}

template <class _IsVector, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
          typename _Proj1, typename _Proj2>
__pattern_set_intersection_return_t<_R1, _R2, _OutRange>
__pattern_set_intersection(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2,
                           _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    using _RandomAccessIterator1 = std::ranges::iterator_t<_R1>;
    using _RandomAccessIterator2 = std::ranges::iterator_t<_R2>;
    using _T = std::ranges::range_value_t<_OutRange>;

    using _DifferenceType1 = typename std::iterator_traits<_RandomAccessIterator1>::difference_type;
    using _DifferenceType2 = typename std::iterator_traits<_RandomAccessIterator2>::difference_type;
    using _DifferenceType = std::common_type_t<_DifferenceType1, _DifferenceType2>;

    const auto __n1 = std::ranges::size(__r1);
    const auto __n2 = std::ranges::size(__r2);

    auto __first1 = std::ranges::begin(__r1);
    auto __last1 = __first1 + __n1;
    auto __first2 = std::ranges::begin(__r2);
    auto __last2 = __first2 + __n2;
    auto __result = std::ranges::begin(__out_r);

    // intersection is empty
    if (__n1 == 0 || __n2 == 0)
        return __pattern_set_intersection_return_t<_R1, _R2, _OutRange>{__last1, __last2, __result};

    // testing  whether the sequences are intersected
    auto __left_bound_seq_1 =
        __first1 + oneapi::dpl::__internal::__pstl_lower_bound(__first1, _DifferenceType1{0}, __last1 - __first1,
                                                               std::invoke(__proj2, *__first2), __comp, __proj1);
    //{1} < {2}: seq 2 is wholly greater than seq 1, so, the intersection is empty
    if (__left_bound_seq_1 == __last1)
        return __pattern_set_intersection_return_t<_R1, _R2, _OutRange>{__last1, __last2, __result};

    // testing  whether the sequences are intersected
    auto __left_bound_seq_2 =
        __first2 + oneapi::dpl::__internal::__pstl_lower_bound(__first2, _DifferenceType2{0}, __last2 - __first2,
                                                               std::invoke(__proj1, *__first1), __comp, __proj2);
    //{2} < {1}: seq 1 is wholly greater than seq 2, so, the intersection is empty
    if (__left_bound_seq_2 == __last2)
        return __pattern_set_intersection_return_t<_R1, _R2, _OutRange>{__last1, __last2, __result};

    const auto __m1 = __last1 - __left_bound_seq_1 + __n2;
    if (__m1 > oneapi::dpl::__internal::__set_algo_cut_off)
    {
        //we know proper offset due to [first1; left_bound_seq_1) < [first2; last2)
        return __internal::__except_handler([&]() {
            auto __out_last = __internal::__parallel_set_op(
                __tag, std::forward<_ExecutionPolicy>(__exec), __left_bound_seq_1, __last1, __first2, __last2, __result,
                [](_DifferenceType __n, _DifferenceType __m) { return std::min(__n, __m); },
                [](_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
                   _RandomAccessIterator2 __last2, _T* __result, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2) {
                    return oneapi::dpl::__utils::__set_intersection_construct(
                        __first1, __last1, __first2, __last2, __result,
                        oneapi::dpl::__internal::__op_uninitialized_copy<_ExecutionPolicy>{},
                        /*CopyFromFirstSet = */ std::true_type{}, __comp, __proj1, __proj2);
                },
                __comp, __proj1, __proj2);
            return __pattern_set_intersection_return_t<_R1, _R2, _OutRange>{__last1, __last2, __out_last};
        });
    }

    const auto __m2 = __last2 - __left_bound_seq_2 + __n1;
    if (__m2 > oneapi::dpl::__internal::__set_algo_cut_off)
    {
        //we know proper offset due to [first2; left_bound_seq_2) < [first1; last1)
        return __internal::__except_handler([&]() {
            auto __out_last = __internal::__parallel_set_op(
                __tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __left_bound_seq_2, __last2, __result,
                [](_DifferenceType __n, _DifferenceType __m) { return std::min(__n, __m); },
                [](_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
                   _RandomAccessIterator2 __last2, _T* __result, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2) {
                    return oneapi::dpl::__utils::__set_intersection_construct(
                        __first2, __last2, __first1, __last1, __result,
                        oneapi::dpl::__internal::__op_uninitialized_copy<_ExecutionPolicy>{},
                        /*CopyFromFirstSet = */ std::false_type{}, __comp, __proj2, __proj1);
                },
                __comp, __proj1, __proj2);
            return __pattern_set_intersection_return_t<_R1, _R2, _OutRange>{__last1, __last2, __out_last};
        });
    }

    // [left_bound_seq_1; last1) and [left_bound_seq_2; last2) - use serial algorithm
    return std::ranges::set_intersection(__left_bound_seq_1, __last1, __left_bound_seq_2, __last2,
                                         std::ranges::begin(__out_r), __comp, __proj1, __proj2);
}

//---------------------------------------------------------------------------------------------------------------------
// set_difference
//---------------------------------------------------------------------------------------------------------------------

template <typename _R1, typename _R2, typename _OutRange, typename _Comp, typename _Proj1, typename _Proj2>
auto
__brick_set_difference(_R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2,
                       /*__is_vector=*/std::false_type) noexcept
{
    return std::ranges::set_difference(std::forward<_R1>(__r1), std::forward<_R2>(__r2), std::ranges::begin(__out_r),
                                       __comp, __proj1, __proj2);
}

template <typename _R1, typename _R2, typename _OutRange, typename _Comp, typename _Proj1, typename _Proj2>
auto
__brick_set_difference(_R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2,
                       /*__is_vector=*/std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::ranges::set_difference(std::forward<_R1>(__r1), std::forward<_R2>(__r2), std::ranges::begin(__out_r),
                                       __comp, __proj1, __proj2);
}

template <typename _R1, typename _OutRange>
using __pattern_set_difference_return_t =
    std::ranges::set_difference_result<std::ranges::borrowed_iterator_t<_R1>,
                                       std::ranges::borrowed_iterator_t<_OutRange>>;

template <typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
          typename _Proj1, typename _Proj2>
__pattern_set_difference_return_t<_R1, _OutRange>
__pattern_set_difference(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _OutRange&& __out_r,
                         _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    static_assert(__is_serial_tag_v<_Tag>);

    return __brick_set_difference(std::forward<_R1>(__r1), std::forward<_R2>(__r2), std::forward<_OutRange>(__out_r),
                                  __comp, __proj1, __proj2, typename _Tag::__is_vector{});
}

template <class _IsVector, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
          typename _Proj1, typename _Proj2>
__pattern_set_difference_return_t<_R1, _OutRange>
__pattern_set_difference(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2,
                         _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    using _RandomAccessIterator1 = std::ranges::iterator_t<_R1>;
    using _RandomAccessIterator2 = std::ranges::iterator_t<_R2>;
    using _T = std::ranges::range_value_t<_OutRange>;

    using _DifferenceType1 = typename std::iterator_traits<_RandomAccessIterator1>::difference_type;
    using _DifferenceType2 = typename std::iterator_traits<_RandomAccessIterator2>::difference_type;
    using _DifferenceType = std::common_type_t<_DifferenceType1, _DifferenceType2>;

    const auto __n1 = std::ranges::size(__r1);
    const auto __n2 = std::ranges::size(__r2);

    auto __first1 = std::ranges::begin(__r1);
    auto __last1 = __first1 + __n1;
    auto __first2 = std::ranges::begin(__r2);
    auto __last2 = __first2 + __n2;
    auto __result = std::ranges::begin(__out_r);

    // {} \ {2}: the difference is empty
    if (__n1 == 0)
        return __pattern_set_difference_return_t<_R1, _OutRange>{__first1, __result};

    // {1} \ {}: parallel copying just first sequence
    if (__n2 == 0)
    {
        auto __out_last = __pattern_walk2_brick(__tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
                                                __result, __internal::__brick_copy<__parallel_tag<_IsVector>>{});
        return __pattern_set_difference_return_t<_R1, _OutRange>{__last1, __out_last};
    }

    // testing  whether the sequences are intersected
    auto __left_bound_seq_1 =
        __first1 + oneapi::dpl::__internal::__pstl_lower_bound(__first1, _DifferenceType1{0}, __last1 - __first1,
                                                               std::invoke(__proj2, *__first2), __comp, __proj1);
    //{1} < {2}: seq 2 is wholly greater than seq 1, so, parallel copying just first sequence
    if (__left_bound_seq_1 == __last1)
    {
        auto __out_last = __pattern_walk2_brick(__tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
                                                __result, __internal::__brick_copy<__parallel_tag<_IsVector>>{});
        return __pattern_set_difference_return_t<_R1, _OutRange>{__last1, __out_last};
    }

    // testing  whether the sequences are intersected
    auto __left_bound_seq_2 =
        __first2 + oneapi::dpl::__internal::__pstl_lower_bound(__first2, _DifferenceType2{0}, __last2 - __first2,
                                                               std::invoke(__proj1, *__first1), __comp, __proj2);
    //{2} < {1}: seq 1 is wholly greater than seq 2, so, parallel copying just first sequence
    if (__left_bound_seq_2 == __last2)
    {
        auto __out_last =
            __internal::__pattern_walk2_brick(__tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
                                              __result, __brick_copy<__parallel_tag<_IsVector>>{});
        return __pattern_set_difference_return_t<_R1, _OutRange>{__last1, __out_last};
    }

    if (__n1 + __n2 > __set_algo_cut_off)
    {
        auto __out_last = __parallel_set_op(
            __tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2, __result,
            [](_DifferenceType __n, _DifferenceType) { return __n; },
            [](_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
               _RandomAccessIterator2 __last2, _T* __result, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2) {
                return oneapi::dpl::__utils::__set_difference_construct(__first1, __last1, __first2, __last2, __result,
                                                                        __BrickCopyConstruct<_IsVector>(), __comp,
                                                                        __proj1, __proj2);
            },
            __comp, __proj1, __proj2);
        return __pattern_set_difference_return_t<_R1, _OutRange>{__last1, __result + (__out_last - __result)};
    }

    // use serial algorithm
    return std::ranges::set_difference(std::forward<_R1>(__r1), std::forward<_R2>(__r2), std::ranges::begin(__out_r),
                                       __comp, __proj1, __proj2);
}

//---------------------------------------------------------------------------------------------------------------------
// set_symmetric_difference
//---------------------------------------------------------------------------------------------------------------------

template <typename _R1, typename _R2, typename _OutRange, typename _Comp, typename _Proj1, typename _Proj2>
auto
__brick_set_symmetric_difference(_R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1,
                                 _Proj2 __proj2,
                                 /*__is_vector=*/std::false_type) noexcept
{
    return std::ranges::set_symmetric_difference(std::forward<_R1>(__r1), std::forward<_R2>(__r2),
                                                 std::ranges::begin(__out_r), __comp, __proj1, __proj2);
}

template <typename _R1, typename _R2, typename _OutRange, typename _Comp, typename _Proj1, typename _Proj2>
auto
__brick_set_symmetric_difference(_R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1,
                                 _Proj2 __proj2,
                                 /*__is_vector=*/std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::ranges::set_symmetric_difference(std::forward<_R1>(__r1), std::forward<_R2>(__r2),
                                                 std::ranges::begin(__out_r), __comp, __proj1, __proj2);
}

template <typename _R1, typename _R2, typename _OutRange>
using __pattern_set_symmetric_difference_return_t =
    std::ranges::set_symmetric_difference_result<std::ranges::borrowed_iterator_t<_R1>,
                                                 std::ranges::borrowed_iterator_t<_R2>,
                                                 std::ranges::borrowed_iterator_t<_OutRange>>;

template <typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
          typename _Proj1, typename _Proj2>
__pattern_set_symmetric_difference_return_t<_R1, _R2, _OutRange>
__pattern_set_symmetric_difference(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _OutRange&& __out_r,
                                   _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    static_assert(__is_serial_tag_v<_Tag>);

    return __brick_set_symmetric_difference(std::forward<_R1>(__r1), std::forward<_R2>(__r2),
                                            std::forward<_OutRange>(__out_r), __comp, __proj1, __proj2,
                                            typename _Tag::__is_vector{});
}

template <class _IsVector, typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
          typename _Proj1, typename _Proj2>
__pattern_set_symmetric_difference_return_t<_R1, _R2, _OutRange>
__pattern_set_symmetric_difference(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2,
                                   _OutRange&& __out_r, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    using _RandomAccessIterator1 = std::ranges::iterator_t<_R1>;
    using _RandomAccessIterator2 = std::ranges::iterator_t<_R2>;
    using _Tp = std::ranges::range_value_t<_OutRange>;

    const auto __n1 = std::ranges::size(__r1);
    const auto __n2 = std::ranges::size(__r2);

    // use serial algorithm
    if (__n1 + __n2 <= oneapi::dpl::__internal::__set_algo_cut_off)
        return std::ranges::set_symmetric_difference(std::forward<_R1>(__r1), std::forward<_R2>(__r2),
                                                     std::ranges::begin(__out_r), __comp, __proj1, __proj2);

    auto __first1 = std::ranges::begin(__r1);
    auto __last1 = __first1 + __n1;
    auto __first2 = std::ranges::begin(__r2);
    auto __last2 = __first2 + __n2;
    auto __result = std::ranges::begin(__out_r);

    auto __out_last = oneapi::dpl::__internal::__parallel_set_union_op(
        __tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2, __result,
        [](_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
           _RandomAccessIterator2 __last2, _Tp* __result, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2) {
            return oneapi::dpl::__utils::__set_symmetric_difference_construct(
                __first1, __last1, __first2, __last2, __result,
                oneapi::dpl::__internal::__BrickCopyConstruct<_IsVector>(), __comp, __proj1, __proj2);
        },
        __comp, __proj1, __proj2);

    return __pattern_set_symmetric_difference_return_t<_R1, _R2, _OutRange>{__last1, __last2, __out_last};
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

    oneapi::dpl::__internal::__binary_op __bin_pred{__pred, __proj1, __proj2};

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
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    oneapi::dpl::__internal::__predicate __pred_1{__pred, __proj};

    auto __end = std::ranges::begin(__r) + std::ranges::size(__r);

    auto __it = oneapi::dpl::__internal::__pattern_remove_if(__tag, std::forward<_ExecutionPolicy>(__exec),
                                                             std::ranges::begin(__r), __end, __pred_1);

    return std::ranges::borrowed_subrange_t<_R>(__it, __end);
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Pred>
auto
__pattern_remove_if(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r, _Pred __pred,
                    _Proj __proj)
{
    return std::ranges::remove_if(std::forward<_R>(__r), __pred, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// __pattern_reverse
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R>
void
__pattern_reverse(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __first = std::ranges::begin(__r);
    auto __last = __first + std::ranges::size(__r);
    oneapi::dpl::__internal::__pattern_reverse(__tag, std::forward<_ExecutionPolicy>(__exec), __first, __last);
}

template <typename _ExecutionPolicy, typename _R>
void
__pattern_reverse(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r)
{
    std::ranges::reverse(std::forward<_R>(__r));
}

//---------------------------------------------------------------------------------------------------------------------
// __pattern_reverse_copy
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
void
__pattern_reverse_copy(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __first_in = std::ranges::begin(__in_r);
    auto __last_in = __first_in + std::ranges::size(__in_r);
    auto __first_out = std::ranges::begin(__out_r);
    oneapi::dpl::__internal::__pattern_reverse_copy(__tag, std::forward<_ExecutionPolicy>(__exec), __first_in,
                                                    __last_in, __first_out);
}

template <typename _ExecutionPolicy, typename _InRange, typename _OutRange>
void
__pattern_reverse_copy(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _InRange&& __in_r,
                       _OutRange&& __out_r)
{
    std::ranges::reverse_copy(std::forward<_InRange>(__in_r), std::ranges::begin(__out_r));
}

//---------------------------------------------------------------------------------------------------------------------
// __pattern_move
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _InRange, typename _OutRange>
void
__pattern_move(_Tag __tag, _ExecutionPolicy&& __exec, _InRange&& __r, _OutRange&& __out_r)
{
    auto __end = std::ranges::begin(__r) + std::ranges::size(__r);
    oneapi::dpl::__internal::__pattern_walk2_brick(__tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                   std::ranges::begin(__r), __end, std::ranges::begin(__out_r),
                                                   oneapi::dpl::__internal::__brick_move<decltype(__tag)>{});
}

template <typename _ExecutionPolicy, typename _InRange, typename _OutRange>
void
__pattern_move(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _InRange&& __r, _OutRange&& __out_r)
{
    std::ranges::move(std::forward<_InRange>(__r), std::ranges::begin(__out_r));
}

template <typename _Tag, typename _ExecutionPolicy, typename _R1, typename _R2>
void
__pattern_swap_ranges(_Tag __tag, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __beg1 = std::ranges::begin(__r1);
    auto __end1 = __beg1 + std::ranges::size(__r1);
    auto __beg2 = std::ranges::begin(__r2);
    oneapi::dpl::__internal::__pattern_swap(__tag, std::forward<_ExecutionPolicy>(__exec), __beg1, __end1, __beg2);
}

template <typename _ExecutionPolicy, typename _R1, typename _R2>
void
__pattern_swap_ranges(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R1&& __r1, _R2&& __r2)
{
    std::ranges::swap_ranges(std::forward<_R1>(__r1), std::forward<_R2>(__r2));
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Comp, typename _Proj>
std::ranges::borrowed_subrange_t<_R>
__pattern_unique(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __beg = std::ranges::begin(__r);
    auto __end = __beg + std::ranges::size(__r);
    auto __it =
        oneapi::dpl::__internal::__pattern_unique(__tag, std::forward<_ExecutionPolicy>(__exec), __beg, __end,
                                                  oneapi::dpl::__internal::__compare<_Comp, _Proj>{__comp, __proj});

    return {__it, __end};
}

template <typename _ExecutionPolicy, typename _R, typename _Comp, typename _Proj>
std::ranges::borrowed_subrange_t<_R>
__pattern_unique(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r, _Comp __comp, _Proj __proj)
{
    return std::ranges::unique(std::forward<_R>(__r), __comp, __proj);
}

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _OutRange, typename _Comp, typename _Proj>
std::ranges::unique_copy_result<std::ranges::borrowed_iterator_t<_R>, std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_unique_copy(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _OutRange&& __out_r, _Comp __comp, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __beg = std::ranges::begin(__r);
    auto __end = __beg + std::ranges::size(__r);
    auto __it = oneapi::dpl::__internal::__pattern_unique_copy(
        __tag, std::forward<_ExecutionPolicy>(__exec), __beg, __end, std::ranges::begin(__out_r),
        oneapi::dpl::__internal::__compare<_Comp, _Proj>{__comp, __proj});
    return {__end, __it};
}

template <typename _ExecutionPolicy, typename _R, typename _OutRange, typename _Comp, typename _Proj>
std::ranges::unique_copy_result<std::ranges::borrowed_iterator_t<_R>, std::ranges::borrowed_iterator_t<_OutRange>>
__pattern_unique_copy(__serial_tag</*IsVector*/ std::false_type>, _ExecutionPolicy&&, _R&& __r, _OutRange&& __out_r,
                      _Comp __comp, _Proj __proj)
{
    return std::ranges::unique_copy(std::forward<_R>(__r), std::ranges::begin(__out_r), __comp, __proj);
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_CPP20_RANGES_PRESENT

#endif // _ONEDPL_ALGORITHM_RANGES_IMPL_H
