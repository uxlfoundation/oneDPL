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

#ifndef _ONEDPL_GLUE_ALGORITHM_IMPL_H
#define _ONEDPL_GLUE_ALGORITHM_IMPL_H

#include <functional>

#include "execution_defs.h"

#include "utils.h"

#if _ONEDPL_HETERO_BACKEND
#    include "hetero/algorithm_impl_hetero.h"
#    include "hetero/numeric_impl_hetero.h"
#endif

#include "algorithm_fwd.h"
#include "numeric_fwd.h" /* count and count_if use __pattern_transform_reduce */

#include "execution_impl.h"

namespace oneapi
{
namespace dpl
{

// [alg.any_of]

template <class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
any_of(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_any_of(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                     __last, __pred);
}

// [alg.all_of]

template <class _ExecutionPolicy, class _ForwardIterator, class _Pred>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
all_of(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Pred __pred)
{
    return !oneapi::dpl::any_of(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
        oneapi::dpl::__internal::__not_pred<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _Pred>>(__pred));
}

// [alg.none_of]

template <class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
none_of(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    return !oneapi::dpl::any_of(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __pred);
}

// [alg.foreach]

template <class _ExecutionPolicy, class _ForwardIterator, class _Function>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
for_each(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Function __f)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    oneapi::dpl::__internal::__pattern_walk1(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                             __f);
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Function>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
for_each_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n, _Function __f)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_walk1_n(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                      __n, __f);
}

// [alg.find]

template <class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
find_if(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_find_if(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                      __last, __pred);
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
find_if_not(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    return oneapi::dpl::find_if(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
        oneapi::dpl::__internal::__not_pred<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _Predicate>>(
            __pred));
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
find(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    return oneapi::dpl::find_if(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __value));
}

// [alg.find.end]
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator1>
find_end(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first,
         _ForwardIterator2 __s_last, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __s_first);

    return oneapi::dpl::__internal::__pattern_find_end(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                       __first, __last, __s_first, __s_last, __pred);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator1>
find_end(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first,
         _ForwardIterator2 __s_last)
{
    return oneapi::dpl::find_end(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __s_first, __s_last,
                                 oneapi::dpl::__internal::__pstl_equal());
}

// [alg.find_first_of]
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator1>
find_first_of(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
              _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __s_first);

    return oneapi::dpl::__internal::__pattern_find_first_of(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                            __first, __last, __s_first, __s_last, __pred);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator1>
find_first_of(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
              _ForwardIterator2 __s_first, _ForwardIterator2 __s_last)
{
    return oneapi::dpl::find_first_of(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __s_first, __s_last,
                                      oneapi::dpl::__internal::__pstl_equal());
}

// [alg.adjacent_find]
template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
adjacent_find(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    typedef typename ::std::iterator_traits<_ForwardIterator>::value_type _ValueType;

    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_adjacent_find(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                            __first, __last, ::std::equal_to<_ValueType>(),
                                                            oneapi::dpl::__internal::__first_semantic());
}

template <class _ExecutionPolicy, class _ForwardIterator, class _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
adjacent_find(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_adjacent_find(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                            __first, __last, __pred,
                                                            oneapi::dpl::__internal::__first_semantic());
}

// [alg.count]

// Implementation note: count and count_if call the pattern directly instead of calling ::std::transform_reduce
// so that we do not have to include <numeric>.

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<
    _ExecutionPolicy, typename ::std::iterator_traits<_ForwardIterator>::difference_type>
count(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_count(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __value));
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<
    _ExecutionPolicy, typename ::std::iterator_traits<_ForwardIterator>::difference_type>
count_if(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_count(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                    __last, __pred);
}

// [alg.search]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator1>
search(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first,
       _ForwardIterator2 __s_last, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __s_first);

    return oneapi::dpl::__internal::__pattern_search(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                     __last, __s_first, __s_last, __pred);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator1>
search(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first,
       _ForwardIterator2 __s_last)
{
    return oneapi::dpl::search(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __s_first, __s_last,
                               oneapi::dpl::__internal::__pstl_equal());
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Tp, class _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
search_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Size __count,
         const _Tp& __value, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_search_n(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                       __first, __last, __count, __value, __pred);
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
search_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Size __count,
         const _Tp& __value)
{
    return oneapi::dpl::search_n(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __count, __value,
                                 ::std::equal_to<typename ::std::iterator_traits<_ForwardIterator>::value_type>());
}

// [alg.copy]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
copy(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result)
{
    auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    return oneapi::dpl::__internal::__pattern_walk2_brick(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
        oneapi::dpl::__internal::__brick_copy<decltype(__dispatch_tag), _ExecutionPolicy>{});
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _Size, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
copy_n(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _Size __n, _ForwardIterator2 __result)
{
    using _DecayedExecutionPolicy = ::std::decay_t<_ExecutionPolicy>;

    auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    return oneapi::dpl::__internal::__pattern_walk2_brick_n(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __n, __result,
        oneapi::dpl::__internal::__brick_copy_n<decltype(__dispatch_tag), _DecayedExecutionPolicy>{});
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
copy_if(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result,
        _Predicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    return oneapi::dpl::__internal::__pattern_copy_if(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                      __last, __result, __pred);
}

// [alg.swap]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
swap_ranges(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
            _ForwardIterator2 __first2)
{
    typedef typename ::std::iterator_traits<_ForwardIterator1>::reference _ReferenceType1;
    typedef typename ::std::iterator_traits<_ForwardIterator2>::reference _ReferenceType2;

    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    return oneapi::dpl::__internal::__pattern_swap(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1,
                                                   __last1, __first2, [](_ReferenceType1 __x, _ReferenceType2 __y) {
                                                       using ::std::swap;
                                                       swap(__x, __y);
                                                   });
}

// [alg.transform]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _UnaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
transform(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result,
          _UnaryOperation __op)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    return oneapi::dpl::__internal::__pattern_walk2(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
        oneapi::dpl::__internal::__transform_functor<_UnaryOperation>{::std::move(__op)});
}

// we can't use non-const __op here
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator,
          class _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
transform(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
          _ForwardIterator __result, _BinaryOperation __op)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2, __result);

    return oneapi::dpl::__internal::__pattern_walk3(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __result,
        oneapi::dpl::__internal::__transform_functor<_BinaryOperation>(::std::move(__op)));
}

// [alg.transform_if]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _UnaryOperation,
          class _UnaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
transform_if(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result,
             _UnaryOperation __op, _UnaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    return oneapi::dpl::__internal::__pattern_walk2_transform_if(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
        oneapi::dpl::__internal::__transform_if_unary_functor<_UnaryOperation, _UnaryPredicate>(::std::move(__op),
                                                                                                ::std::move(__pred)));
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator3,
          class _BinaryOperation, class _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator3>
transform_if(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
             _ForwardIterator2 __first2, _ForwardIterator3 __result, _BinaryOperation __op, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2, __result);

    return oneapi::dpl::__internal::__pattern_walk3_transform_if(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __result,
        oneapi::dpl::__internal::__transform_if_binary_functor<_BinaryOperation, _BinaryPredicate>(
            ::std::move(__op), ::std::move(__pred)));
}

// [alg.replace]

template <class _ExecutionPolicy, class _ForwardIterator, class _UnaryPredicate, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
replace_if(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred,
           const _Tp& __new_value)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    oneapi::dpl::__internal::__pattern_walk1(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
        oneapi::dpl::__internal::__replace_functor<
            oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>,
            oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _UnaryPredicate>>(__new_value, __pred));
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
replace(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __old_value,
        const _Tp& __new_value)
{
    oneapi::dpl::replace_if(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __old_value),
        __new_value);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _UnaryPredicate, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
replace_copy_if(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                _ForwardIterator2 __result, _UnaryPredicate __pred, const _Tp& __new_value)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    return oneapi::dpl::__internal::__pattern_walk2(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
        oneapi::dpl::__internal::__replace_copy_functor<
            oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>,
            ::std::conditional_t<oneapi::dpl::__internal::__is_const_callable_object_v<_UnaryPredicate>,
                                 _UnaryPredicate,
                                 oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _UnaryPredicate>>>(
            __new_value, __pred));
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
replace_copy(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result,
             const _Tp& __old_value, const _Tp& __new_value)
{
    return oneapi::dpl::replace_copy_if(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __old_value),
        __new_value);
}

// [alg.fill]

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
fill(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    oneapi::dpl::__internal::__pattern_fill(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                            __value);
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
fill_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __count, const _Tp& __value)
{
    if (__count <= 0)
        return __first;

    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_fill_n(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                     __count, __value);
}

// [alg.generate]
template <class _ExecutionPolicy, class _ForwardIterator, class _Generator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
generate(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Generator __g)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    oneapi::dpl::__internal::__pattern_generate(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                __last, __g);
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Generator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
generate_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __count, _Generator __g)
{
    if (__count <= 0)
        return __first;

    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_generate_n(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                         __first, __count, __g);
}

// [alg.remove]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Predicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
remove_copy_if(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _Predicate __pred)
{
    return oneapi::dpl::copy_if(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
        oneapi::dpl::__internal::__not_pred<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, _Predicate>>(
            __pred));
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
remove_copy(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result,
            const _Tp& __value)
{
    return oneapi::dpl::copy_if(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
        oneapi::dpl::__internal::__not_equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __value));
}

template <class _ExecutionPolicy, class _ForwardIterator, class _UnaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
remove_if(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_remove_if(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                        __first, __last, __pred);
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
remove(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    return oneapi::dpl::remove_if(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
        oneapi::dpl::__internal::__equal_value<oneapi::dpl::__internal::__ref_or_copy<_ExecutionPolicy, const _Tp>>(
            __value));
}

// [alg.unique]

template <class _ExecutionPolicy, class _ForwardIterator, class _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
unique(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_unique(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                     __last, __pred);
}

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
unique(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::unique(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                               oneapi::dpl::__internal::__pstl_equal());
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
unique_copy(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result,
            _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    return oneapi::dpl::__internal::__pattern_unique_copy(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                          __first, __last, __result, __pred);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
unique_copy(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result)
{
    return oneapi::dpl::unique_copy(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
                                    oneapi::dpl::__internal::__pstl_equal());
}

// [alg.reverse]

template <class _ExecutionPolicy, class _BidirectionalIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
reverse(_ExecutionPolicy&& __exec, _BidirectionalIterator __first, _BidirectionalIterator __last)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    oneapi::dpl::__internal::__pattern_reverse(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                               __last);
}

template <class _ExecutionPolicy, class _BidirectionalIterator, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
reverse_copy(_ExecutionPolicy&& __exec, _BidirectionalIterator __first, _BidirectionalIterator __last,
             _ForwardIterator __d_first)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __d_first);

    return oneapi::dpl::__internal::__pattern_reverse_copy(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                           __first, __last, __d_first);
}

// [alg.rotate]

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
rotate(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_rotate(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                     __middle, __last);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
rotate_copy(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __middle, _ForwardIterator1 __last,
            _ForwardIterator2 __result)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    return oneapi::dpl::__internal::__pattern_rotate_copy(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                          __first, __middle, __last, __result);
}

// [alg.partitions]

template <class _ExecutionPolicy, class _ForwardIterator, class _UnaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
is_partitioned(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_is_partitioned(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                             __first, __last, __pred);
}

template <class _ExecutionPolicy, class _ForwardIterator, class _UnaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
partition(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_partition(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                        __first, __last, __pred);
}

template <class _ExecutionPolicy, class _BidirectionalIterator, class _UnaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _BidirectionalIterator>
stable_partition(_ExecutionPolicy&& __exec, _BidirectionalIterator __first, _BidirectionalIterator __last,
                 _UnaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_stable_partition(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                               __first, __last, __pred);
}

template <class _ExecutionPolicy, class _ForwardIterator, class _ForwardIterator1, class _ForwardIterator2,
          class _UnaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      ::std::pair<_ForwardIterator1, _ForwardIterator2>>
partition_copy(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last,
               _ForwardIterator1 __out_true, _ForwardIterator2 __out_false, _UnaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __out_true, __out_false);

    return oneapi::dpl::__internal::__pattern_partition_copy(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                             __first, __last, __out_true, __out_false, __pred);
}

// [alg.sort]

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    oneapi::dpl::__internal::__pattern_sort(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                            __comp);
}

template <class _ExecutionPolicy, class _RandomAccessIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last)
{
    oneapi::dpl::sort(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                      oneapi::dpl::__internal::__pstl_less());
}

// [stable.sort]

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
stable_sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    oneapi::dpl::__internal::__pattern_stable_sort(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                   __last, __comp);
}

template <class _ExecutionPolicy, class _RandomAccessIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
stable_sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last)
{
    oneapi::dpl::stable_sort(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                             oneapi::dpl::__internal::__pstl_less());
}

// oneapi::dpl::sort_by_key

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
          typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
sort_by_key(_ExecutionPolicy&& __exec, _RandomAccessIterator1 __keys_first, _RandomAccessIterator1 __keys_last,
            _RandomAccessIterator2 __values_first, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __keys_first, __values_first);

    oneapi::dpl::__internal::__pattern_sort_by_key(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                   __keys_first, __keys_last, __values_first, __comp);
}

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
sort_by_key(_ExecutionPolicy&& __exec, _RandomAccessIterator1 __keys_first, _RandomAccessIterator1 __keys_last,
            _RandomAccessIterator2 __values_first)
{
    oneapi::dpl::sort_by_key(::std::forward<_ExecutionPolicy>(__exec), __keys_first, __keys_last, __values_first,
                             oneapi::dpl::__internal::__pstl_less());
}

// oneapi::dpl::stable_sort_by_key

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
          typename _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
stable_sort_by_key(_ExecutionPolicy&& __exec, _RandomAccessIterator1 __keys_first, _RandomAccessIterator1 __keys_last,
                   _RandomAccessIterator2 __values_first, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __keys_first, __values_first);

    oneapi::dpl::__internal::__pattern_stable_sort_by_key(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                          __keys_first, __keys_last, __values_first, __comp);
}

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
stable_sort_by_key(_ExecutionPolicy&& __exec, _RandomAccessIterator1 __keys_first, _RandomAccessIterator1 __keys_last,
                   _RandomAccessIterator2 __values_first)
{
    oneapi::dpl::stable_sort_by_key(::std::forward<_ExecutionPolicy>(__exec), __keys_first, __keys_last, __values_first,
                                    oneapi::dpl::__internal::__pstl_less());
}

// [mismatch]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      ::std::pair<_ForwardIterator1, _ForwardIterator2>>
mismatch(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
         _ForwardIterator2 __last2, _BinaryPredicate __pred)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    return oneapi::dpl::__internal::__pattern_mismatch(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                       __first1, __last1, __first2, __last2, __pred);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      ::std::pair<_ForwardIterator1, _ForwardIterator2>>
mismatch(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
         _BinaryPredicate __pred)
{
    return oneapi::dpl::mismatch(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
                                 oneapi::dpl::__internal::__pstl_next(__first2, ::std::distance(__first1, __last1)),
                                 __pred);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      ::std::pair<_ForwardIterator1, _ForwardIterator2>>
mismatch(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
         _ForwardIterator2 __last2)
{
    return oneapi::dpl::mismatch(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2,
                                 oneapi::dpl::__internal::__pstl_equal());
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      ::std::pair<_ForwardIterator1, _ForwardIterator2>>
mismatch(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2)
{
    //TODO: to get rid of "distance"
    return oneapi::dpl::mismatch(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
                                 oneapi::dpl::__internal::__pstl_next(__first2, ::std::distance(__first1, __last1)));
}

// [alg.equal]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
      _BinaryPredicate __p)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    return oneapi::dpl::__internal::__pattern_equal(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1,
                                                    __last1, __first2, __p);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2)
{
    return oneapi::dpl::equal(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
                              oneapi::dpl::__internal::__pstl_equal());
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
      _ForwardIterator2 __last2, _BinaryPredicate __p)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    return oneapi::dpl::__internal::__pattern_equal(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1,
                                                    __last1, __first2, __last2, __p);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
equal(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
      _ForwardIterator2 __last2)
{
    return oneapi::dpl::equal(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2,
                              oneapi::dpl::__internal::__pstl_equal());
}

// [alg.move]
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
move(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __d_first)
{
    using _DecayedExecutionPolicy = ::std::decay_t<_ExecutionPolicy>;

    auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __d_first);

    return oneapi::dpl::__internal::__pattern_walk2_brick(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __d_first,
        oneapi::dpl::__internal::__brick_move<decltype(__dispatch_tag), _DecayedExecutionPolicy>{});
}

// [partial.sort]

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
partial_sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __middle,
             _RandomAccessIterator __last, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    oneapi::dpl::__internal::__pattern_partial_sort(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                    __middle, __last, __comp);
}

template <class _ExecutionPolicy, class _RandomAccessIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
partial_sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __middle,
             _RandomAccessIterator __last)
{
    oneapi::dpl::partial_sort(::std::forward<_ExecutionPolicy>(__exec), __first, __middle, __last,
                              oneapi::dpl::__internal::__pstl_less());
}

// [partial.sort.copy]

template <class _ExecutionPolicy, class _ForwardIterator, class _RandomAccessIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator>
partial_sort_copy(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last,
                  _RandomAccessIterator __d_first, _RandomAccessIterator __d_last, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __d_first);

    return oneapi::dpl::__internal::__pattern_partial_sort_copy(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __d_first, __d_last, __comp);
}

template <class _ExecutionPolicy, class _ForwardIterator, class _RandomAccessIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator>
partial_sort_copy(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last,
                  _RandomAccessIterator __d_first, _RandomAccessIterator __d_last)
{
    return oneapi::dpl::partial_sort_copy(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __d_first,
                                          __d_last, oneapi::dpl::__internal::__pstl_less());
}

// [is.sorted]
template <class _ExecutionPolicy, class _ForwardIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
is_sorted_until(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    const _ForwardIterator __res = oneapi::dpl::__internal::__pattern_adjacent_find(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
        oneapi::dpl::__internal::__reorder_pred<_Compare>(__comp), oneapi::dpl::__internal::__first_semantic());
    return __res == __last ? __last : oneapi::dpl::__internal::__pstl_next(__res);
}

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
is_sorted_until(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::is_sorted_until(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                        oneapi::dpl::__internal::__pstl_less());
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
is_sorted(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_adjacent_find(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                            __first, __last,
                                                            oneapi::dpl::__internal::__reorder_pred<_Compare>(__comp),
                                                            oneapi::dpl::__internal::__or_semantic()) == __last;
}

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
is_sorted(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::is_sorted(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                  oneapi::dpl::__internal::__pstl_less());
}

// [alg.merge]
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator,
          class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
merge(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
      _ForwardIterator2 __last2, _ForwardIterator __d_first, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2, __d_first);

    return oneapi::dpl::__internal::__pattern_merge(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1,
                                                    __last1, __first2, __last2, __d_first, __comp);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
merge(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
      _ForwardIterator2 __last2, _ForwardIterator __d_first)
{
    return oneapi::dpl::merge(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2, __d_first,
                              oneapi::dpl::__internal::__pstl_less());
}

template <class _ExecutionPolicy, class _BidirectionalIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
inplace_merge(_ExecutionPolicy&& __exec, _BidirectionalIterator __first, _BidirectionalIterator __middle,
              _BidirectionalIterator __last, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    oneapi::dpl::__internal::__pattern_inplace_merge(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                     __middle, __last, __comp);
}

template <class _ExecutionPolicy, class _BidirectionalIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
inplace_merge(_ExecutionPolicy&& __exec, _BidirectionalIterator __first, _BidirectionalIterator __middle,
              _BidirectionalIterator __last)
{
    oneapi::dpl::inplace_merge(::std::forward<_ExecutionPolicy>(__exec), __first, __middle, __last,
                               oneapi::dpl::__internal::__pstl_less());
}

// [includes]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
includes(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
         _ForwardIterator2 __last2, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    return oneapi::dpl::__internal::__pattern_includes(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                       __first1, __last1, __first2, __last2, __comp);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
includes(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
         _ForwardIterator2 __last2)
{
    return oneapi::dpl::includes(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2,
                                 oneapi::dpl::__internal::__pstl_less());
}

// [set.union]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator,
          class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_union(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
          _ForwardIterator2 __last2, _ForwardIterator __result, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2, __result);

    return oneapi::dpl::__internal::__pattern_set_union(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                        __first1, __last1, __first2, __last2, __result, __comp);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_union(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
          _ForwardIterator2 __last2, _ForwardIterator __result)
{
    return oneapi::dpl::set_union(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2,
                                  __result, oneapi::dpl::__internal::__pstl_less());
}

// [set.intersection]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator,
          class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_intersection(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                 _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2, __result);

    return oneapi::dpl::__internal::__pattern_set_intersection(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                               __first1, __last1, __first2, __last2, __result, __comp);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_intersection(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                 _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result)
{
    return oneapi::dpl::set_intersection(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2,
                                         __result, oneapi::dpl::__internal::__pstl_less());
}

// [set.difference]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator,
          class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
               _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2, __result);

    return oneapi::dpl::__internal::__pattern_set_difference(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                             __first1, __last1, __first2, __last2, __result, __comp);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
               _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result)
{
    return oneapi::dpl::set_difference(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2,
                                       __result, oneapi::dpl::__internal::__pstl_less());
}

// [set.symmetric.difference]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator,
          class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_symmetric_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                         _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result,
                         _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2, __result);

    return oneapi::dpl::__internal::__pattern_set_symmetric_difference(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2, __result,
        __comp);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
set_symmetric_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                         _ForwardIterator2 __first2, _ForwardIterator2 __last2, _ForwardIterator __result)
{
    return oneapi::dpl::set_symmetric_difference(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
                                                 __last2, __result, oneapi::dpl::__internal::__pstl_less());
}

// [is.heap]
template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator>
is_heap_until(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_is_heap_until(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                            __first, __last, __comp);
}

template <class _ExecutionPolicy, class _RandomAccessIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator>
is_heap_until(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last)
{
    return oneapi::dpl::is_heap_until(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                      oneapi::dpl::__internal::__pstl_less());
}

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
is_heap(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_is_heap(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                      __last, __comp);
}

template <class _ExecutionPolicy, class _RandomAccessIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
is_heap(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last)
{
    return oneapi::dpl::is_heap(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                oneapi::dpl::__internal::__pstl_less());
}

// [alg.min.max]

template <class _ExecutionPolicy, class _ForwardIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
min_element(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_min_element(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                          __first, __last, __comp);
}

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
min_element(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::min_element(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                    oneapi::dpl::__internal::__pstl_less());
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
max_element(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    return oneapi::dpl::min_element(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                    oneapi::dpl::__internal::__reorder_pred<_Compare>(__comp));
}

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
max_element(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::min_element(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                    oneapi::dpl::__internal::__reorder_pred<oneapi::dpl::__internal::__pstl_less>(
                                        oneapi::dpl::__internal::__pstl_less()));
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, ::std::pair<_ForwardIterator, _ForwardIterator>>
minmax_element(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_minmax_element(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                             __first, __last, __comp);
}

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, ::std::pair<_ForwardIterator, _ForwardIterator>>
minmax_element(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::minmax_element(::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                       oneapi::dpl::__internal::__pstl_less());
}

// [alg.nth.element]

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
nth_element(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __nth,
            _RandomAccessIterator __last, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    oneapi::dpl::__internal::__pattern_nth_element(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first,
                                                   __nth, __last, __comp);
}

template <class _ExecutionPolicy, class _RandomAccessIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
nth_element(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __nth,
            _RandomAccessIterator __last)
{
    oneapi::dpl::nth_element(::std::forward<_ExecutionPolicy>(__exec), __first, __nth, __last,
                             oneapi::dpl::__internal::__pstl_less());
}

// [alg.lex.comparison]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Compare>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
lexicographical_compare(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                        _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    return oneapi::dpl::__internal::__pattern_lexicographical_compare(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2, __comp);
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, bool>
lexicographical_compare(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                        _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
    return oneapi::dpl::lexicographical_compare(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
                                                __last2, oneapi::dpl::__internal::__pstl_less());
}

// [shift.left]

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
shift_left(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last,
           typename ::std::iterator_traits<_ForwardIterator>::difference_type __n)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_shift_left(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                         __first, __last, __n);
}

// [shift.right]

template <class _ExecutionPolicy, class _BidirectionalIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _BidirectionalIterator>
shift_right(_ExecutionPolicy&& __exec, _BidirectionalIterator __first, _BidirectionalIterator __last,
            typename ::std::iterator_traits<_BidirectionalIterator>::difference_type __n)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    return oneapi::dpl::__internal::__pattern_shift_right(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                          __first, __last, __n);
}

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_GLUE_ALGORITHM_IMPL_H
