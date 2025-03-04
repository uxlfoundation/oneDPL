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

#ifndef _ONEDPL_GLUE_ASYNC_IMPL_H
#define _ONEDPL_GLUE_ASYNC_IMPL_H

#include "../async_extension_defs.h"
#include "async_impl_hetero.h"

namespace oneapi
{
namespace dpl
{
namespace experimental
{
// [wait_for_all]
template <typename... _Ts>
oneapi::dpl::__internal::__enable_if_convertible_to_events<void, _Ts...>
wait_for_all(_Ts&&... __events)
{
    // TODO design a backend API function for waiting, and move this implementation into the SYCL backend
    ::std::initializer_list<int> i = {0, (static_cast<sycl::event>(__events).wait_and_throw(), 0)...};
    (void)i;
}

// [async.transform]
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _UnaryOperation,
          class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...>>
auto
transform_async(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                _ForwardIterator2 __result, _UnaryOperation __op, _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    auto ret_val = oneapi::dpl::__internal::__pattern_walk2_async(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
        oneapi::dpl::__internal::__transform_functor<_UnaryOperation>{::std::move(__op)});
    return ret_val;
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator,
          class _BinaryOperation, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<_ExecutionPolicy, int,
                                                                                         _BinaryOperation, _Events...>>
auto
transform_async(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                _ForwardIterator2 __first2, _ForwardIterator __result, _BinaryOperation __op,
                _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2, __result);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    auto ret_val = oneapi::dpl::__internal::__pattern_walk3_async(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __result,
        oneapi::dpl::__internal::__transform_functor<_BinaryOperation>(::std::move(__op)));
    return ret_val;
}

// [async.copy]
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...>>
auto
copy_async(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result,
           _Events&&... __dependencies)
{
    auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    auto ret_val = oneapi::dpl::__internal::__pattern_walk2_brick_async(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
        oneapi::dpl::__internal::__brick_copy<decltype(__dispatch_tag), _ExecutionPolicy>{});
    return ret_val;
}

// [async.sort]
template <class _ExecutionPolicy, class _Iterator, class _Compare, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<_ExecutionPolicy, int,
                                                                                         _Compare, _Events...>>
auto
sort_async(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Compare __comp, _Events&&... __dependencies)
{
    wait_for_all(::std::forward<_Events>(__dependencies)...);
    assert(__last - __first >= 2);

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);

    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);
    using __backend_tag = typename decltype(__dispatch_tag)::__backend_tag;

    return __par_backend_hetero::__parallel_stable_sort(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec),
                                                        __buf.all_view(), __comp, oneapi::dpl::identity{});
}

template <class _ExecutionPolicy, class _RandomAccessIterator, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...>>
auto
sort_async(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last,
           _Events&&... __dependencies)
{
    using _ValueType = typename ::std::iterator_traits<_RandomAccessIterator>::value_type;
    return sort_async(::std::forward<_ExecutionPolicy>(__exec), __first, __last, ::std::less<_ValueType>(),
                      ::std::forward<_Events>(__dependencies)...);
}

// [async.for_each]
template <class _ExecutionPolicy, class _ForwardIterator, class _Function, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...>>
auto
for_each_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Function __f,
               _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    auto ret_val = oneapi::dpl::__internal::__pattern_walk1_async(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __f);
    return ret_val;
}

// [async.reduce]

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_double_no_default<_ExecutionPolicy, int, _Tp,
                                                                                         _BinaryOperation, _Events...>>
auto
reduce_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
             _BinaryOperation __binary_op, _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    auto ret_val = oneapi::dpl::__internal::__pattern_transform_reduce_async(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, __binary_op,
        oneapi::dpl::__internal::__no_op());
    return ret_val;
}

template <class _ExecutionPolicy, class _ForwardIt, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...>>
auto
reduce_async(_ExecutionPolicy&& __exec, _ForwardIt __first, _ForwardIt __last, _Events&&... __dependencies)
{
    using _ValueType = typename ::std::iterator_traits<_ForwardIt>::value_type;
    return reduce_async(::std::forward<_ExecutionPolicy>(__exec), __first, __last, _ValueType(0),
                        ::std::plus<_ValueType>(), ::std::forward<_Events>(__dependencies)...);
}

template <class _ExecutionPolicy, class _ForwardIt, class _T, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<_ExecutionPolicy, int, _T,
                                                                                         _Events...>>
auto
reduce_async(_ExecutionPolicy&& __exec, _ForwardIt __first, _ForwardIt __last, _T __init, _Events&&... __dependencies)
{
    return reduce_async(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, ::std::plus<_T>(),
                        ::std::forward<_Events>(__dependencies)...);
}

// [async.fill]

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...>>
auto
fill_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value,
           _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    return oneapi::dpl::__internal::__pattern_fill_async(__dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                         __first, __last, __value);
}

// [async.transform_reduce]

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class _BinaryOp1, class _BinaryOp2,
          class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_double_no_default<
              _ExecutionPolicy, int, _BinaryOp1, _BinaryOp2, _Events...>>
auto
transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                       _T __init, _BinaryOp1 __binary_op1, _BinaryOp2 __binary_op2, _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    return oneapi::dpl::__internal::__pattern_transform_reduce_async(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __init, __binary_op1,
        __binary_op2);
}

template <class _ExecutionPolicy, class _ForwardIt, class _T, class _BinaryOp, class _UnaryOp, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<_ExecutionPolicy, int,
                                                                                         _UnaryOp, _Events...>>
auto
transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIt __first, _ForwardIt __last, _T __init,
                       _BinaryOp __binary_op, _UnaryOp __unary_op, _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    return oneapi::dpl::__internal::__pattern_transform_reduce_async(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, __binary_op, __unary_op);
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...>>
auto
transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                       _T __init, _Events&&... __dependencies)
{
    using _ValueType = typename ::std::iterator_traits<_ForwardIt1>::value_type;
    return transform_reduce_async(::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __init,
                                  ::std::plus<_T>(), ::std::multiplies<_ValueType>(),
                                  ::std::forward<_Events>(__dependencies)...);
}

// [async.scan]

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...>>
auto
inclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                     _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    using _ValueType = typename ::std::iterator_traits<_ForwardIt1>::value_type;
    wait_for_all(::std::forward<_Events>(__dependencies)...);
    return oneapi::dpl::__internal::__pattern_transform_scan_async(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
        oneapi::dpl::__internal::__no_op(), ::std::plus<_ValueType>(), /*inclusive=*/::std::true_type());
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _BinaryOperation, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<_ExecutionPolicy, int,
                                                                                         _BinaryOperation, _Events...>>
auto
inclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                     _BinaryOperation __binary_op, _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    return oneapi::dpl::__internal::__pattern_transform_scan_async(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
        oneapi::dpl::__internal::__no_op(), __binary_op, /*inclusive=*/::std::true_type());
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _BinaryOperation, class _T,
          class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_double_no_default<
              _ExecutionPolicy, int, _BinaryOperation, _T, _Events...>>
auto
inclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                     _BinaryOperation __binary_op, _T __init, _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    return oneapi::dpl::__internal::__pattern_transform_scan_async(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
        oneapi::dpl::__internal::__no_op(), __init, __binary_op, /*inclusive=*/::std::true_type());
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...>>
auto
exclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                     _T __init, _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    return oneapi::dpl::__internal::__pattern_transform_scan_async(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
        oneapi::dpl::__internal::__no_op(), __init, ::std::plus<_T>(), /*exclusive=*/::std::false_type());
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class _BinaryOperation,
          class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<_ExecutionPolicy, int,
                                                                                         _BinaryOperation, _Events...>>
auto
exclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                     _T __init, _BinaryOperation __binary_op, _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    return oneapi::dpl::__internal::__pattern_transform_scan_async(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
        oneapi::dpl::__internal::__no_op(), __init, __binary_op, /*exclusive=*/::std::false_type());
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class _BinaryOperation,
          class _UnaryOperation, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...>>
auto
transform_exclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1,
                               _ForwardIt2 __first2, _T __init, _BinaryOperation __binary_op,
                               _UnaryOperation __unary_op, _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    return oneapi::dpl::__internal::__pattern_transform_scan_async(__dispatch_tag,
                                                                   ::std::forward<_ExecutionPolicy>(__exec), __first1,
                                                                   __last1, __first2, __unary_op, __init, __binary_op,
                                                                   /*exclusive=*/::std::false_type());
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _BinaryOperation, class _UnaryOperation,
          class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...>>
auto
transform_inclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1,
                               _ForwardIt2 __first2, _BinaryOperation __binary_op, _UnaryOperation __unary_op,
                               _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    return oneapi::dpl::__internal::__pattern_transform_scan_async(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __unary_op, __binary_op,
        /*inclusive=*/::std::true_type());
}

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _BinaryOperation, class _UnaryOperation,
          class _T, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<_ExecutionPolicy, int, _T,
                                                                                         _Events...>>
auto
transform_inclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1,
                               _ForwardIt2 __first2, _BinaryOperation __binary_op, _UnaryOperation __unary_op,
                               _T __init, _Events&&... __dependencies)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first1, __first2);

    wait_for_all(::std::forward<_Events>(__dependencies)...);
    return oneapi::dpl::__internal::__pattern_transform_scan_async(__dispatch_tag,
                                                                   ::std::forward<_ExecutionPolicy>(__exec), __first1,
                                                                   __last1, __first2, __unary_op, __init, __binary_op,
                                                                   /*inclusive=*/::std::true_type());
}

} // namespace experimental

} // namespace dpl

} // namespace oneapi

#endif // _ONEDPL_GLUE_ASYNC_IMPL_H
