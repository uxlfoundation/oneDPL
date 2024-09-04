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
__pattern_for_each(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _R&& __r, _Fun __f, _Proj __proj)
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
__pattern_transform(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r,
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
__pattern_transform(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _InRange1&& __in_r1, _InRange2&& __in_r2, _OutRange&& __out_r,
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
__pattern_find_if(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    return std::ranges::find_if(std::forward<_R>(__r), __pred, __proj);
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
__pattern_any_of(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
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
__pattern_adjacent_find_ranges(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
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
__pattern_search(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred, _Proj1 __proj1, _Proj2 __proj2)
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
__pattern_search_n(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _R&& __r, std::ranges::range_difference_t<_R> __count, const _T& __value,
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
__pattern_count_if(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj)
{
    return std::ranges::count_if(std::forward<_R>(__r), __pred, __proj);
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
__pattern_equal(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred, _Proj1 __proj1, _Proj2 __proj2)
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
__pattern_is_sorted(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    return std::ranges::is_sorted(std::forward<_R>(__r), __comp, __proj);
}

//---------------------------------------------------------------------------------------------------------------------
// pattern_sort
//---------------------------------------------------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
auto
__pattern_sort_ranges(_Tag __tag, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});

    auto __comp_2 = [__comp, __proj](auto&& __val1, auto&& __val2) { return std::invoke(__comp, std::invoke(__proj,
        std::forward<decltype(__val1)>(__val1)), std::invoke(__proj, std::forward<decltype(__val2)>(__val2)));};
    // Call stable_sort pattern since __pattern_sort_ranges is shared between sort and stable_sort
    // TODO: add a separate pattern for ranges::sort for better performance
    oneapi::dpl::__internal::__pattern_stable_sort(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r), std::ranges::begin(__r) + std::ranges::size(__r), __comp_2);

    return std::ranges::borrowed_iterator_t<_R>(std::ranges::begin(__r) + std::ranges::size(__r));
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
auto
__pattern_sort_ranges(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    return std::ranges::stable_sort(std::forward<_R>(__r), __comp, __proj);
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

    auto __res = oneapi::dpl::__internal::__pattern_min_element(__tag, std::forward<_ExecutionPolicy>(__exec), std::ranges::begin(__r),
        std::ranges::begin(__r) + std::ranges::size(__r), __comp_2);

    return std::ranges::borrowed_iterator_t<_R>(__res);
}

template <typename _ExecutionPolicy, typename _R, typename _Proj, typename _Comp>
auto
__pattern_min_element(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _R&& __r, _Comp __comp, _Proj __proj)
{
    return std::ranges::min_element(std::forward<_R>(__r), __comp, __proj);
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

    oneapi::dpl::__internal::__pattern_walk2_brick(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__in_r), std::ranges::begin(__in_r) + std::ranges::size(__in_r), std::ranges::begin(__out_r),
        oneapi::dpl::__internal::__brick_copy<decltype(__tag), _ExecutionPolicy>{});
}

template<typename _ExecutionPolicy, typename _InRange, typename _OutRange>
void
__pattern_copy(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r)
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
__pattern_copy_if_ranges(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r,
                         _Pred __pred, _Proj __proj)
{
    return std::ranges::copy_if(std::forward<_InRange>(__in_r), std::ranges::begin(__out_r), __pred, __proj);
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
    static_assert(__is_parallel_tag_v<_Tag> || typename _Tag::__is_vector{});
    assert(std::ranges::size(__r1) + std::ranges::size(__r2) <= std::ranges::size(__out_r)); // for debug purposes only

    auto __comp_2 = [__comp, __proj1, __proj2](auto&& __val1, auto&& __val2) { return std::invoke(__comp,
        std::invoke(__proj1, std::forward<decltype(__val1)>(__val1)), std::invoke(__proj2,
        std::forward<decltype(__val2)>(__val2)));};

    auto __res = oneapi::dpl::__internal::__pattern_merge(__tag, std::forward<_ExecutionPolicy>(__exec),
        std::ranges::begin(__r1), std::ranges::begin(__r1) + std::ranges::size(__r1), std::ranges::begin(__r2),
        std::ranges::begin(__r2) + std::ranges::size(__r2), std::ranges::begin(__out_r), __comp_2);

    using __return_type = std::ranges::merge_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_R2>,
        std::ranges::borrowed_iterator_t<_OutRange>>;

    return __return_type{std::ranges::begin(__r1) + std::ranges::size(__r1), std::ranges::begin(__r2) + std::ranges::size(__r2), __res};
}

template<typename _ExecutionPolicy, typename _R1, typename _R2, typename _OutRange, typename _Comp,
         typename _Proj1, typename _Proj2>
auto
__pattern_merge(__serial_tag</*IsVector*/std::false_type>, _ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp,
                _Proj1 __proj1, _Proj2 __proj2)
{
    return std::ranges::merge(std::forward<_R1>(__r1), std::forward<_R2>(__r2), std::ranges::begin(__out_r), __comp, __proj1,
                              __proj2);
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_CPP20_RANGES_PRESENT

#endif // _ONEDPL_ALGORITHM_RANGES_IMPL_H
