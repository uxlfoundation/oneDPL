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

#ifndef _ONEDPL_GLUE_ALGORITHM_RANGES_IMPL_H
#define _ONEDPL_GLUE_ALGORITHM_RANGES_IMPL_H

#include <utility>
#if _ONEDPL_CPP20_RANGES_PRESENT
#    include <ranges>
#    include <functional>
#    include <type_traits>
#    include <iterator>
#endif

#include "execution_defs.h"
#include "oneapi/dpl/pstl/ranges_defs.h"

#if _ONEDPL_CPP20_RANGES_PRESENT
#    include "algorithm_ranges_impl.h"
#endif

#if _ONEDPL_HETERO_BACKEND
#    include "hetero/algorithm_ranges_impl_hetero.h"
#    include "hetero/algorithm_impl_hetero.h" //TODO: for __brick_copy
#endif

namespace oneapi
{
namespace dpl
{

#if _ONEDPL_CPP20_RANGES_PRESENT
namespace ranges
{

// [alg.foreach]

namespace __internal
{

struct __for_each_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Proj = std::identity,
             std::indirectly_unary_invocable<std::projected<std::ranges::iterator_t<_R>, _Proj>>  _Fun>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>

    std::ranges::borrowed_iterator_t<_R>
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _Fun __f, _Proj __proj = {}) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);
        oneapi::dpl::__internal::__ranges::__pattern_for_each(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __r, __f, __proj);

        return {std::ranges::begin(__r) + std::ranges::size(__r)};
    }
}; //__for_each_fn
}  //__internal

inline constexpr __internal::__for_each_fn for_each;

// [alg.transform]

namespace __internal
{
struct __transform_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R,
             std::ranges::random_access_range _OutRange, std::copy_constructible _F, typename _Proj = std::identity>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
        && std::ranges::sized_range<_OutRange> && std::indirectly_writable<std::ranges::iterator_t<_OutRange>,
                 std::indirect_result_t<_F&, std::projected<std::ranges::iterator_t<_R>, _Proj>>>

    std::ranges::unary_transform_result<std::ranges::borrowed_iterator_t<_R>, std::ranges::borrowed_iterator_t<_OutRange>>
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _OutRange&& __out_r, _F __op, _Proj __proj = {}) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);

        using _Size = std::common_type_t<std::ranges::range_size_t<_R>, std::ranges::range_size_t<_OutRange>>;
        const _Size __size = std::ranges::min((_Size)std::ranges::size(__r), (_Size)std::ranges::size(__out_r));

        oneapi::dpl::__internal::__ranges::__pattern_transform(__dispatch_tag, std::forward<_ExecutionPolicy>(__exec),
            std::ranges::take_view(__r, __size), std::ranges::take_view(__out_r, __size), __op, __proj);

        return {std::ranges::begin(__r) + __size, std::ranges::begin(__out_r) +  __size};
    }

    template<typename _ExecutionPolicy, std::ranges::random_access_range _R1, std::ranges::random_access_range _R2,
             std::ranges::random_access_range _OutRange, std::copy_constructible _F, typename _Proj1 = std::identity,
             typename _Proj2 = std::identity>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && (std::ranges::sized_range<_R1>
        || std::ranges::sized_range<_R2>) && std::ranges::sized_range<_OutRange>
        && std::indirectly_writable<std::ranges::iterator_t<_OutRange>,
            std::indirect_result_t<_F&, std::projected<std::ranges::iterator_t<_R1>, _Proj1>,
            std::projected<std::ranges::iterator_t<_R2>, _Proj2>>>

    std::ranges::binary_transform_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_R2>,
        std::ranges::borrowed_iterator_t<_OutRange>>
    operator()(_ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _F __binary_op,
               _Proj1 __proj1 = {}, _Proj2 __proj2 = {}) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);

        using _Size = std::common_type_t<oneapi::dpl::__internal::__range_size_t<_R1>,
            oneapi::dpl::__internal::__range_size_t<_R2>, std::ranges::range_size_t<_OutRange>>;
        _Size __size = std::ranges::size(__out_r);
        if constexpr(std::ranges::sized_range<_R1>)
            __size = std::ranges::min(__size, (_Size)std::ranges::size(__r1));
        if constexpr(std::ranges::sized_range<_R2>)
            __size = std::ranges::min(__size, (_Size)std::ranges::size(__r2));

        //take_view doesn't make unsized range sized, so subrange is used below
        oneapi::dpl::__internal::__ranges::__pattern_transform(__dispatch_tag, std::forward<_ExecutionPolicy>(__exec),
            std::ranges::subrange(std::ranges::begin(__r1), std::ranges::begin(__r1) + __size),
            std::ranges::subrange(std::ranges::begin(__r2), std::ranges::begin(__r2) + __size),
            std::ranges::take_view(__out_r, __size), __binary_op, __proj1, __proj2);

        return {std::ranges::begin(__r1) + __size, std::ranges::begin(__r2) + __size, std::ranges::begin(__out_r) + __size};
    }
}; //__transform_fn
}  //__internal

inline constexpr __internal::__transform_fn transform;

// [alg.find_if]

namespace __internal
{
struct __find_if_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Proj = std::identity,
             std::indirect_unary_predicate<std::projected<std::ranges::iterator_t<_R>, _Proj>> _Pred>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
    auto
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj = {}) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);
        return oneapi::dpl::__internal::__ranges::__pattern_find_if(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __pred, __proj);
    }
}; //__find_if_fn
}  //__internal

inline constexpr __internal::__find_if_fn find_if;

// [alg.find_if_not]

namespace __internal
{
struct __find_if_not_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Proj = std::identity,
             std::indirect_unary_predicate<std::projected<std::ranges::iterator_t<_R>, _Proj>> _Pred>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
    auto
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj = {}) const
    {
        return oneapi::dpl::ranges::find_if(std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r),
            oneapi::dpl::__internal::__not_pred<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy,
            _Pred>>(__pred), __proj);
    }
}; //__find_if_not_fn
}  //__internal

inline constexpr __internal::__find_if_not_fn find_if_not;

// [alg.find]

namespace __internal
{
struct __find_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Proj = std::identity,
             typename _T>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
        && std::indirect_binary_predicate<std::ranges::equal_to, std::projected<std::ranges::iterator_t<_R>, _Proj>,
        const _T*>
    auto
    operator()(_ExecutionPolicy&& __exec, _R&& __r, const _T& __value, _Proj __proj = {}) const
    {
        return oneapi::dpl::ranges::find_if(std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r),
            oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy,
            const _T>>(__value), __proj);
    }
}; //__find_fn
}  //__internal

inline constexpr __internal::__find_fn find;

namespace __internal
{
struct __any_of_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Proj = std::identity,
             std::indirect_unary_predicate<std::projected<std::ranges::iterator_t<_R>, _Proj>> _Pred>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
    bool
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj = {}) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);
        return oneapi::dpl::__internal::__ranges::__pattern_any_of(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __pred, __proj);
    }
}; //__any_of_fn
}  //__internal

inline constexpr __internal::__any_of_fn any_of;

namespace __internal
{
struct __all_of_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Proj = std::identity,
             std::indirect_unary_predicate<std::projected<std::ranges::iterator_t<_R>, _Proj>> _Pred>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
    bool
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj = {}) const
    {
        return !oneapi::dpl::ranges::any_of(std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r),
            oneapi::dpl::__internal::__not_pred<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _Pred>>(__pred),
            __proj);
    }
}; //__all_of_fn
}  //__internal

inline constexpr __internal::__all_of_fn all_of;

namespace __internal
{
struct __none_of_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Proj = std::identity,
             std::indirect_unary_predicate<std::projected<std::ranges::iterator_t<_R>, _Proj>> _Pred>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
    bool
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj = {}) const
    {
        return !oneapi::dpl::ranges::any_of(std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r),
            __pred, __proj);
    }
}; //__none_of_fn
}  //__internal

inline constexpr __internal::__none_of_fn none_of;

namespace __internal
{
struct __adjacent_find_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Proj = std::identity,
             std::indirect_binary_predicate<std::projected<std::ranges::iterator_t<_R>, _Proj>,
             std::projected<std::ranges::iterator_t<_R>, _Proj>> _Pred = std::ranges::equal_to>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
    auto
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _Pred __pred = {}, _Proj __proj = {}) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);
        return oneapi::dpl::__internal::__ranges::__pattern_adjacent_find_ranges(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __pred, __proj);
    }
}; //__adjacent_find_fn
}  //__internal

inline constexpr __internal::__adjacent_find_fn adjacent_find;

namespace __internal
{
struct __search_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R1, std::ranges::random_access_range _R2,
             typename _Pred = std::ranges::equal_to, typename _Proj1 = std::identity, typename _Proj2 = std::identity>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>>
        && std::ranges::sized_range<_R1> && std::ranges::sized_range<_R2>
        && std::indirectly_comparable<std::ranges::iterator_t<_R1>, std::ranges::iterator_t<_R2>, _Pred, _Proj1, _Proj2>
    auto
    operator()(_ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred = {}, _Proj1 __proj1 = {},
        _Proj2 __proj2 = {}) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);
        return oneapi::dpl::__internal::__ranges::__pattern_search(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::forward<_R1>(__r1), std::forward<_R2>(__r2), __pred, __proj1,
            __proj2);
    }
}; //__search_fn
}  //__internal

inline constexpr __internal::__search_fn search;

namespace __internal
{
struct __search_n_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _T,
             typename _Pred = std::ranges::equal_to, typename _Proj = std::identity>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
        && std::indirectly_comparable<std::ranges::iterator_t<_R>, const _T*, _Pred, _Proj>
    auto
    operator()(_ExecutionPolicy&& __exec, _R&& __r, std::ranges::range_difference_t<_R> __count, const _T& __value,
        _Pred __pred = {}, _Proj __proj = {}) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);
        return oneapi::dpl::__internal::__ranges::__pattern_search_n(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __count, __value, __pred, __proj);
    }
}; //__search_n_fn
}  //__internal

inline constexpr __internal::__search_n_fn search_n;

namespace __internal
{
struct __count_if_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Proj = std::identity,
             std::indirect_unary_predicate<std::projected<std::ranges::iterator_t<_R>, _Proj>> _Pred>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>

    std::ranges::range_difference_t<_R>
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _Pred __pred, _Proj __proj = {}) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);
        return oneapi::dpl::__internal::__ranges::__pattern_count_if(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __pred, __proj);
    }
}; //__count_if_fn
}  //__internal

inline constexpr __internal::__count_if_fn count_if;

namespace __internal
{
struct __count_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Proj = std::identity,
             typename _T>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
        && std::indirect_binary_predicate<std::ranges::equal_to, std::projected<std::ranges::iterator_t<_R>, _Proj>,
        const _T*>
    std::ranges::range_difference_t<_R>
    operator()(_ExecutionPolicy&& __exec, _R&& __r, const _T& __value, _Proj __proj = {}) const
    {
        auto __pred = [__value](auto&& __val) { return std::ranges::equal_to{}(
            std::forward<decltype(__val)>(__val), __value);};
        return oneapi::dpl::ranges::count_if(std::forward<_ExecutionPolicy>(__exec),
            std::forward<_R>(__r), __pred, __proj);
    }
}; //__count_fn
}  //__internal

inline constexpr __internal::__count_fn count;

namespace __internal
{
struct __equal_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R1, std::ranges::random_access_range _R2,
             typename _Pred = std::ranges::equal_to, typename _Proj1 = std::identity, typename _Proj2 = std::identity>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>>
           && (std::ranges::sized_range<_R1> || std::ranges::sized_range<_R2>)
           && std::indirectly_comparable<std::ranges::iterator_t<_R1>, std::ranges::iterator_t<_R2>, _Pred, _Proj1,
           _Proj2>
    bool
    operator()(_ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _Pred __pred = {}, _Proj1 __proj1 = {},
               _Proj2 __proj2 = {}) const
    {
        if constexpr(!std::ranges::sized_range<_R1> || !std::ranges::sized_range<_R2>)
            return false;
        else
        {
            const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);
            return oneapi::dpl::__internal::__ranges::__pattern_equal(__dispatch_tag,
                std::forward<_ExecutionPolicy>(__exec), std::forward<_R1>(__r1), std::forward<_R2>(__r2), __pred,
                __proj1, __proj2);
        }
    }
}; //__equal_fn
}  //__internal

inline constexpr __internal::__equal_fn equal;

namespace __internal
{
struct __is_sorted_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Proj = std::identity,
             std::indirect_strict_weak_order<std::projected<std::ranges::iterator_t<_R>, _Proj>>
             _Comp = std::ranges::less>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
    bool
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _Comp __comp = {}, _Proj __proj = {}) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);
        return oneapi::dpl::__internal::__ranges::__pattern_is_sorted(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __comp, __proj);
    }
}; //__is_sorted_fn
}  //__internal

inline constexpr __internal::__is_sorted_fn is_sorted;

namespace __internal
{
struct __stable_sort_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Comp = std::ranges::less,
             typename _Proj = std::identity>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
           && std::sortable<std::ranges::iterator_t<_R>, _Comp, _Proj>
    auto
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _Comp __comp = {}, _Proj __proj = {}) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);
        return oneapi::dpl::__internal::__ranges::__pattern_sort_ranges(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __comp, __proj);
    }
}; //__stable_sort_fn
}  //__internal

inline constexpr __internal::__stable_sort_fn stable_sort;

namespace __internal
{
struct __sort_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Comp = std::ranges::less,
             typename _Proj = std::identity>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
           && std::sortable<std::ranges::iterator_t<_R>, _Comp, _Proj>
    auto
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _Comp __comp = {}, _Proj __proj = {}) const
    {
        return oneapi::dpl::ranges::stable_sort(std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r),
            __comp, __proj);
    }
}; //__sort_fn
}  //__internal

inline constexpr __internal::__sort_fn sort;

namespace __internal
{
struct __min_element_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Proj = std::identity,
             std::indirect_strict_weak_order<std::projected<std::ranges::iterator_t<_R>, _Proj>>
             _Comp = std::ranges::less>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
    auto
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _Comp __comp = {}, _Proj __proj = {}) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);
        return oneapi::dpl::__internal::__ranges::__pattern_min_element(__dispatch_tag,
            std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r), __comp, __proj);
    }
}; //__min_element_fn
}  //__internal

inline constexpr __internal::__min_element_fn min_element;

namespace __internal
{
struct __max_element_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R, typename _Proj = std::identity,
             std::indirect_strict_weak_order<std::projected<std::ranges::iterator_t<_R>, _Proj>> _Comp = std::ranges::less>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R>
    auto
    operator()(_ExecutionPolicy&& __exec, _R&& __r, _Comp __comp = {}, _Proj __proj = {}) const
    {
        return oneapi::dpl::ranges::min_element(std::forward<_ExecutionPolicy>(__exec), std::forward<_R>(__r),
            oneapi::dpl::__internal::__reorder_pred(__comp), __proj);
    }
}; //__max_element_fn
}  //__internal

inline constexpr __internal::__max_element_fn max_element;

namespace __internal
{
struct __copy_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _InRange,
             std::ranges::random_access_range _OutRange>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>>
        && std::ranges::sized_range<_InRange> && std::ranges::sized_range<_OutRange>
        && std::indirectly_copyable<std::ranges::iterator_t<_InRange>, std::ranges::iterator_t<_OutRange>>

    std::ranges::copy_result<std::ranges::borrowed_iterator_t<_InRange>, std::ranges::borrowed_iterator_t<_OutRange>>
    operator()(_ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);

        using _Size = std::common_type_t<std::ranges::range_size_t<_InRange>, std::ranges::range_size_t<_OutRange>>;
        const _Size __size = std::ranges::min((_Size)std::ranges::size(__in_r), (_Size)std::ranges::size(__out_r));

        oneapi::dpl::__internal::__ranges::__pattern_copy(__dispatch_tag, std::forward<_ExecutionPolicy>(__exec),
            std::ranges::take_view(__in_r, __size), std::ranges::take_view(__out_r, __size));

        return {std::ranges::begin(__in_r) + __size, std::ranges::begin(__out_r) +  __size};
    }
}; //__copy_fn
}  //__internal

inline constexpr __internal::__copy_fn copy;

namespace __internal
{
struct __copy_if_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _InRange,
             std::ranges::random_access_range _OutRange, typename _Proj = std::identity,
             std::indirect_unary_predicate<std::projected<std::ranges::iterator_t<_InRange>, _Proj>> _Pred>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>>
        && std::ranges::sized_range<_InRange> && std::ranges::sized_range<_OutRange>
        && std::indirectly_copyable<std::ranges::iterator_t<_InRange>, std::ranges::iterator_t<_OutRange>>
    auto
    operator()(_ExecutionPolicy&& __exec, _InRange&& __in_r, _OutRange&& __out_r, _Pred __pred, _Proj __proj = {}) const
    {
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);
        // No minimum common size is calculated here, because the size of the output range is unknown
        return oneapi::dpl::__internal::__ranges::__pattern_copy_if_ranges(__dispatch_tag, std::forward<_ExecutionPolicy>(__exec),
            std::forward<_InRange>(__in_r), std::forward<_OutRange>(__out_r), __pred, __proj);
    }
}; //__copy_if_fn
}  //__internal

inline constexpr __internal::__copy_if_fn copy_if;

namespace __internal
{
struct __merge_fn
{
    template<typename _ExecutionPolicy, std::ranges::random_access_range _R1, std::ranges::random_access_range _R2,
             std::ranges::random_access_range _OutRange, typename _Comp = std::ranges::less,
             typename _Proj1 = std::identity, typename _Proj2 = std::identity>
    requires oneapi::dpl::is_execution_policy_v<std::remove_cvref_t<_ExecutionPolicy>> && std::ranges::sized_range<_R1>
        && std::ranges::sized_range<_R2> && std::ranges::sized_range<_OutRange>
        && std::mergeable<std::ranges::iterator_t<_R1>, std::ranges::iterator_t<_R2>,
        std::ranges::iterator_t<_OutRange>, _Comp, _Proj1, _Proj2>
    auto
    operator()(_ExecutionPolicy&& __exec, _R1&& __r1, _R2&& __r2, _OutRange&& __out_r, _Comp __comp = {}, _Proj1 __proj1 = {},
               _Proj2 __proj2 = {}) const
    {
        // TODO: develop a strategy to get a common minimum size
        const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec);
        return oneapi::dpl::__internal::__ranges::__pattern_merge(__dispatch_tag, std::forward<_ExecutionPolicy>(__exec),
            std::forward<_R1>(__r1), std::forward<_R2>(__r2), std::forward<_OutRange>(__out_r), __comp, __proj1, __proj2);
    }
}; //__merge_fn
}  //__internal

inline constexpr __internal::__merge_fn merge;

} //ranges

#endif //_ONEDPL_CPP20_RANGES_PRESENT

#if _ONEDPL_HETERO_BACKEND
namespace experimental
{
namespace ranges
{

// [alg.any_of]

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
any_of(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    return oneapi::dpl::__internal::__ranges::__pattern_any_of(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                               views::all_read(::std::forward<_Range>(__rng)), __pred);
}

// [alg.all_of]

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
all_of(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    return !any_of(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
        oneapi::dpl::__internal::__not_pred<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _Predicate>>(
            __pred));
}

// [alg.none_of]

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
none_of(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    return !any_of(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __pred);
}

// [alg.foreach]

template <typename _ExecutionPolicy, typename _Range, typename _Function>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
for_each(_ExecutionPolicy&& __exec, _Range&& __rng, _Function __f)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __f,
                                                        views::all(::std::forward<_Range>(__rng)));
}

// [alg.find]

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
find_if(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    return oneapi::dpl::__internal::__ranges::__pattern_find_if(__dispatch_tag,
                                                                ::std::forward<_ExecutionPolicy>(__exec),
                                                                views::all_read(::std::forward<_Range>(__rng)), __pred);
}

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
find_if_not(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    return find_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
        oneapi::dpl::__internal::__not_pred<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _Predicate>>(
            __pred));
}

template <typename _ExecutionPolicy, typename _Range, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
find(_ExecutionPolicy&& __exec, _Range&& __rng, const _Tp& __value)
{
    return find_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __value));
}

// [alg.find.end]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
find_end(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng1, __rng2);

    return oneapi::dpl::__internal::__ranges::__pattern_find_end(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng1)),
        views::all_read(::std::forward<_Range2>(__rng2)), __pred);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
find_end(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    return find_end(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                    ::std::forward<_Range2>(__rng2), oneapi::dpl::__internal::__pstl_equal());
}

// [alg.find_first_of]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
find_first_of(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng1, __rng2);

    return oneapi::dpl::__internal::__ranges::__pattern_find_first_of(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng1)),
        views::all_read(::std::forward<_Range2>(__rng2)), __pred);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
find_first_of(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    return find_first_of(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                         ::std::forward<_Range2>(__rng2), oneapi::dpl::__internal::__pstl_equal());
}

// [alg.adjacent_find]

template <typename _ExecutionPolicy, typename _Range, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
adjacent_find(_ExecutionPolicy&& __exec, _Range&& __rng, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    return oneapi::dpl::__internal::__ranges::__pattern_adjacent_find(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range>(__rng)),
        __pred, oneapi::dpl::__internal::__first_semantic());
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
adjacent_find(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    using _ValueType = oneapi::dpl::__internal::__value_t<_Range>;
    return adjacent_find(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                         ::std::equal_to<_ValueType>());
}

// [alg.count]

template <typename _ExecutionPolicy, typename _Range, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
count_if(_ExecutionPolicy&& __exec, _Range&& __rng, _Predicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    return oneapi::dpl::__internal::__ranges::__pattern_count(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                              views::all_read(::std::forward<_Range>(__rng)), __pred);
}

template <typename _ExecutionPolicy, typename _Range, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
count(_ExecutionPolicy&& __exec, _Range&& __rng, const _Tp& __value)
{
    return count_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __value));
}

// [alg.search]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
search(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng1, __rng2);

    return oneapi::dpl::__internal::__ranges::__pattern_search(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng1)),
        views::all_read(::std::forward<_Range2>(__rng2)), __pred);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
search(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    return search(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                  ::std::forward<_Range2>(__rng2), oneapi::dpl::__internal::__pstl_equal());
}

template <typename _ExecutionPolicy, typename _Range, typename _Size, typename _Tp, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
search_n(_ExecutionPolicy&& __exec, _Range&& __rng, _Size __count, const _Tp& __value, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    return oneapi::dpl::__internal::__ranges::__pattern_search_n(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range>(__rng)),
        __count, __value, __pred);
}

template <typename _ExecutionPolicy, typename _Range, typename _Size, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
search_n(_ExecutionPolicy&& __exec, _Range&& __rng, _Size __count, const _Tp& __value)
{
    return search_n(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __count, __value,
                    oneapi::dpl::__internal::__pstl_equal());
}

// [alg.copy]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result)
{
    auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng, __result);

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__internal::__brick_copy<decltype(__dispatch_tag), _ExecutionPolicy>{},
        views::all_read(::std::forward<_Range1>(__rng)), views::all_write(::std::forward<_Range2>(__result)));
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
copy_if(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, _Predicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng, __result);

    return oneapi::dpl::__internal::__ranges::__pattern_copy_if(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng)),
        views::all_write(::std::forward<_Range2>(__result)), __pred, oneapi::dpl::__internal::__pstl_assign());
}

// [alg.swap]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
swap_ranges(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng1, __rng2);

    using _ReferenceType1 = oneapi::dpl::__internal::__value_t<_Range1>&;
    using _ReferenceType2 = oneapi::dpl::__internal::__value_t<_Range2>&;

    return oneapi::dpl::__internal::__ranges::__pattern_swap(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all(::std::forward<_Range1>(__rng1)),
        views::all(::std::forward<_Range2>(__rng2)), [](_ReferenceType1 __x, _ReferenceType2 __y) {
            using ::std::swap;
            swap(__x, __y);
        });
}

// [alg.transform]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
transform(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, _UnaryOperation __op)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng, __result);

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), [__op](auto x, auto& z) { z = __op(x); },
        views::all_read(::std::forward<_Range1>(__rng)), views::all_write(::std::forward<_Range2>(__result)));
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
transform(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __result, _BinaryOperation __op)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng1, __rng2, __result);

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), [__op](auto x, auto y, auto& z) { z = __op(x, y); },
        views::all_read(::std::forward<_Range1>(__rng1)), views::all_read(::std::forward<_Range2>(__rng2)),
        views::all_write(::std::forward<_Range3>(__result)));
}

// [alg.remove]

template <typename _ExecutionPolicy, typename _Range, typename _UnaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
remove_if(_ExecutionPolicy&& __exec, _Range&& __rng, _UnaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    return oneapi::dpl::__internal::__ranges::__pattern_remove_if(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all(::std::forward<_Range>(__rng)), __pred);
}

template <typename _ExecutionPolicy, typename _Range, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
remove(_ExecutionPolicy&& __exec, _Range&& __rng, const _Tp& __value)
{
    return remove_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __value));
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
remove_copy_if(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, _Predicate __pred)
{
    return copy_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng), ::std::forward<_Range2>(__result),
        oneapi::dpl::__internal::__not_pred<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _Predicate>>(
            __pred));
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
remove_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, const _Tp& __value)
{
    return copy_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng), ::std::forward<_Range2>(__result),
        oneapi::dpl::__internal::__not_equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __value));
}

// [alg.unique]

template <typename _ExecutionPolicy, typename _Range, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
unique(_ExecutionPolicy&& __exec, _Range&& __rng, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    return oneapi::dpl::__internal::__ranges::__pattern_unique(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                               views::all(::std::forward<_Range>(__rng)), __pred);
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
unique(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    return unique(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                  oneapi::dpl::__internal::__pstl_equal());
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
unique_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng, __result);

    return oneapi::dpl::__internal::__ranges::__pattern_unique_copy(
        __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), views::all_read(std::forward<_Range1>(__rng)),
        views::all_write(std::forward<_Range2>(__result)), __pred);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
unique_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result)
{
    return unique_copy(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng),
                       ::std::forward<_Range2>(__result), oneapi::dpl::__internal::__pstl_equal());
}

// [alg.reverse]

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
reverse(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    auto __v = views::all(::std::forward<_Range>(__rng));
    auto __n = __v.size();
    auto __n_2 = __n / 2;

    auto __r1 = __v | views::take(__n_2);
    auto __r2 = __v | views::reverse | views::take(__n_2);
    swap_ranges(::std::forward<_ExecutionPolicy>(__exec), __r1, __r2);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
reverse_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result)
{
    auto __src = views::all_read(::std::forward<_Range1>(__rng));
    copy(::std::forward<_ExecutionPolicy>(__exec), __src | views::reverse, ::std::forward<_Range2>(__result));
    return __src.size();
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
rotate_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, oneapi::dpl::__internal::__difference_t<_Range1> __rotate_value,
            _Range2&& __result)
{
    auto __src = views::all_read(::std::forward<_Range1>(__rng));
    copy(::std::forward<_ExecutionPolicy>(__exec), __src | views::rotate(__rotate_value),
         ::std::forward<_Range2>(__result));
    return __src.size();
}

// [alg.replace]

template <typename _ExecutionPolicy, typename _Range, typename _UnaryPredicate, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
replace_if(_ExecutionPolicy&& __exec, _Range&& __rng, _UnaryPredicate __pred, const _Tp& __new_value)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__internal::__replace_functor<
            oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>,
            oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _UnaryPredicate>>(__new_value, __pred),
        views::all(::std::forward<_Range>(__rng)));
}

template <typename _ExecutionPolicy, typename _Range, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
replace(_ExecutionPolicy&& __exec, _Range&& __rng, const _Tp& __old_value, const _Tp& __new_value)
{
    replace_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __old_value),
        __new_value);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryPredicate, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
replace_copy_if(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, _UnaryPredicate __pred,
                const _Tp& __new_value)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng, __result);

    auto __src = views::all_read(::std::forward<_Range1>(__rng));
    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__internal::__replace_copy_functor<
            oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>,
            ::std::conditional_t<oneapi::dpl::__internal::__is_const_callable_object_v<_UnaryPredicate>,
                                 _UnaryPredicate,
                                 oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _UnaryPredicate>>>(
            __new_value, __pred),
        __src, views::all_write(::std::forward<_Range2>(__result)));
    return __src.size();
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range1>>
replace_copy(_ExecutionPolicy&& __exec, _Range1&& __rng, _Range2&& __result, const _Tp& __old_value,
             const _Tp& __new_value)
{
    return replace_copy_if(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng), ::std::forward<_Range2>(__result),
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __old_value),
        __new_value);
}

// [alg.sort]

template <typename _ExecutionPolicy, typename _Range, typename _Compare, typename _Proj>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
sort(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp, _Proj __proj)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    oneapi::dpl::__internal::__ranges::__pattern_sort(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                      views::all(::std::forward<_Range>(__rng)), __comp, __proj);
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
sort(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    sort(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
         oneapi::dpl::__internal::__pstl_less());
}

// [stable.sort]

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
stable_sort(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    sort(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __comp);
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
stable_sort(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    sort(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
         oneapi::dpl::__internal::__pstl_less());
}

// [is.sorted]

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
is_sorted_until(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    auto __view = views::all_read(::std::forward<_Range>(__rng));
    const auto __res = oneapi::dpl::__internal::__ranges::__pattern_adjacent_find(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __view,
        oneapi::dpl::__internal::__reorder_pred<_Compare>(__comp), oneapi::dpl::__internal::__first_semantic());

    return __res == __view.size() ? __res : __res + 1;
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
is_sorted_until(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    return is_sorted_until(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                           oneapi::dpl::__internal::__pstl_less());
}

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
is_sorted(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    auto __view = views::all_read(::std::forward<_Range>(__rng));
    return oneapi::dpl::__internal::__ranges::__pattern_adjacent_find(
               __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __view,
               oneapi::dpl::__internal::__reorder_pred<_Compare>(__comp),
               oneapi::dpl::__internal::__or_semantic()) == __view.size();
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
is_sorted(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    return is_sorted(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                     oneapi::dpl::__internal::__pstl_less());
}

// [alg.equal]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryPredicate __p)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng1, __rng2);

    return oneapi::dpl::__internal::__ranges::__pattern_equal(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                              views::all_read(::std::forward<_Range1>(__rng1)),
                                                              views::all_read(::std::forward<_Range2>(__rng2)), __p);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    return equal(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                 ::std::forward<_Range2>(__rng2), oneapi::dpl::__internal::__pstl_equal());
}

// [alg.move]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
move(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng1, __rng2);

    using _DecayedExecutionPolicy = ::std::decay_t<_ExecutionPolicy>;

    oneapi::dpl::__internal::__ranges::__pattern_walk_n(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
        oneapi::dpl::__internal::__brick_move<decltype(__dispatch_tag), _DecayedExecutionPolicy>{},
        views::all_read(::std::forward<_Range1>(__rng1)), views::all_write(::std::forward<_Range2>(__rng2)));
}

// [alg.merge]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range3>>
merge(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng1, __rng2, __rng3);

    return oneapi::dpl::__internal::__ranges::__pattern_merge(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng1)),
        views::all_read(::std::forward<_Range2>(__rng2)), views::all_write(::std::forward<_Range3>(__rng3)), __comp);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range3>>
merge(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3)
{
    return merge(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                 ::std::forward<_Range2>(__rng2), ::std::forward<_Range3>(__rng3),
                 oneapi::dpl::__internal::__pstl_less());
}

// [alg.min.max]

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
min_element(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    return oneapi::dpl::__internal::__ranges::__pattern_min_element(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range>(__rng)),
        __comp);
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
min_element(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    return min_element(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                       oneapi::dpl::__internal::__pstl_less());
}

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
max_element(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    return min_element(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                       oneapi::dpl::__internal::__reorder_pred<_Compare>(__comp));
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__difference_t<_Range>>
max_element(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    return min_element(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                       oneapi::dpl::__internal::__reorder_pred<oneapi::dpl::__internal::__pstl_less>(
                           oneapi::dpl::__internal::__pstl_less()));
}

template <typename _ExecutionPolicy, typename _Range, typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<
    _ExecutionPolicy,
    ::std::pair<oneapi::dpl::__internal::__difference_t<_Range>, oneapi::dpl::__internal::__difference_t<_Range>>>
minmax_element(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    return oneapi::dpl::__internal::__ranges::__pattern_minmax_element(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range>(__rng)),
        __comp);
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<
    _ExecutionPolicy,
    ::std::pair<oneapi::dpl::__internal::__difference_t<_Range>, oneapi::dpl::__internal::__difference_t<_Range>>>
minmax_element(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    return minmax_element(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                          oneapi::dpl::__internal::__pstl_less());
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Range4,
          typename _BinaryPredicate, typename _BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range3>>
reduce_by_segment(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values, _Range3&& __out_keys,
                  _Range4&& __out_values, _BinaryPredicate __binary_pred, _BinaryOperator __binary_op)
{
    const auto __dispatch_tag =
        oneapi::dpl::__ranges::__select_backend(__exec, __keys, __values, __out_keys, __out_values);

    return oneapi::dpl::__internal::__ranges::__pattern_reduce_by_segment(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__keys)),
        views::all_read(::std::forward<_Range2>(__values)), views::all_write(::std::forward<_Range3>(__out_keys)),
        views::all_write(::std::forward<_Range4>(__out_values)), __binary_pred, __binary_op);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Range4,
          typename _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range3>>
reduce_by_segment(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values, _Range3&& __out_keys,
                  _Range4&& __out_values, _BinaryPredicate __binary_pred)
{
    return reduce_by_segment(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__keys),
                             ::std::forward<_Range2>(__values), ::std::forward<_Range3>(__out_keys),
                             ::std::forward<_Range4>(__out_values), __binary_pred,
                             oneapi::dpl::__internal::__pstl_plus());
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Range4>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range3>>
reduce_by_segment(_ExecutionPolicy&& __exec, _Range1&& __keys, _Range2&& __values, _Range3&& __out_keys,
                  _Range4&& __out_values)
{
    return reduce_by_segment(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__keys),
                             ::std::forward<_Range2>(__values), ::std::forward<_Range3>(__out_keys),
                             ::std::forward<_Range4>(__out_values), oneapi::dpl::__internal::__pstl_equal());
}

} // namespace ranges
} // namespace experimental

#endif //_ONEDPL_HETERO_BACKEND

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_GLUE_ALGORITHM_RANGES_IMPL_H
