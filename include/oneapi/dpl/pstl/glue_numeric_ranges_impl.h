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

#ifndef _ONEDPL_GLUE_NUMERIC_RANGES_IMPL_H
#define _ONEDPL_GLUE_NUMERIC_RANGES_IMPL_H

#include "execution_defs.h"
#include "glue_numeric_defs.h"

#if _ONEDPL_HETERO_BACKEND
#    include "hetero/numeric_ranges_impl_hetero.h"

namespace oneapi
{
namespace dpl
{
namespace experimental
{
namespace ranges
{

// [reduce]

template <typename _ExecutionPolicy, typename _Range, typename _Tp, typename _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
reduce(_ExecutionPolicy&& __exec, _Range&& __rng, _Tp __init, _BinaryOperation __binary_op)
{
    return transform_reduce(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __init,
                            __binary_op, oneapi::dpl::__internal::__no_op());
}

template <typename _ExecutionPolicy, typename _Range, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
reduce(_ExecutionPolicy&& __exec, _Range&& __rng, _Tp __init)
{
    return transform_reduce(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __init,
                            ::std::plus<_Tp>(), oneapi::dpl::__internal::__no_op());
}

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__value_t<_Range>>
reduce(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    using _ValueType = oneapi::dpl::__internal::__value_t<_Range>;
    return transform_reduce(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), _ValueType{},
                            ::std::plus<_ValueType>(), oneapi::dpl::__internal::__no_op());
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Tp __init)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng1, __rng2);

    using _ValueType = oneapi::dpl::__internal::__value_t<_Range1>;
    return oneapi::dpl::__internal::__ranges::__pattern_transform_reduce(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng1)),
        views::all_read(::std::forward<_Range2>(__rng2)), __init, ::std::plus<_ValueType>(),
        ::std::multiplies<_ValueType>());
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp, typename _BinaryOperation1,
          typename _BinaryOperation2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Tp __init,
                 _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng1, __rng2);

    return oneapi::dpl::__internal::__ranges::__pattern_transform_reduce(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng1)),
        views::all_read(::std::forward<_Range2>(__rng2)), __init, __binary_op1, __binary_op2);
}

template <typename _ExecutionPolicy, typename _Range, typename _Tp, typename _BinaryOperation, typename _UnaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _Range&& __rng, _Tp __init, _BinaryOperation __binary_op,
                 _UnaryOperation __unary_op)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng);

    return oneapi::dpl::__internal::__ranges::__pattern_transform_reduce(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range>(__rng)),
        __init, __binary_op, __unary_op);
}

// [exclusive.scan]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
exclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Tp __init)
{
    return transform_exclusive_scan(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                                    ::std::forward<_Range2>(__rng2), __init, ::std::plus<_Tp>(),
                                    oneapi::dpl::__internal::__no_op());
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp, typename _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
exclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Tp __init, _BinaryOperation __binary_op)
{
    return transform_exclusive_scan(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                                    ::std::forward<_Range2>(__rng2), __init, __binary_op,
                                    oneapi::dpl::__internal::__no_op());
}

// [inclusive.scan]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
inclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2)
{
    using _ValueType = oneapi::dpl::__internal::__value_t<_Range1>;
    return transform_inclusive_scan(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                                    ::std::forward<_Range2>(__rng2), ::std::plus<_ValueType>(),
                                    oneapi::dpl::__internal::__no_op());
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
inclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryOperation __binary_op)
{
    return transform_inclusive_scan(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                                    ::std::forward<_Range2>(__rng2), __binary_op, oneapi::dpl::__internal::__no_op());
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp, typename _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
inclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryOperation __binary_op, _Tp __init)
{
    return transform_inclusive_scan(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                                    ::std::forward<_Range2>(__rng2), __binary_op, oneapi::dpl::__internal::__no_op(),
                                    __init);
}

// [transform.exclusive.scan]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp, typename _BinaryOperation,
          typename _UnaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
transform_exclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Tp __init,
                         _BinaryOperation __binary_op, _UnaryOperation __unary_op)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng1, __rng2);

    return oneapi::dpl::__internal::__ranges::__pattern_transform_scan(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng1)),
        views::all_write(::std::forward<_Range2>(__rng2)), __unary_op, __init, __binary_op,
        /*inclusive=*/::std::false_type());
}

// [transform.inclusive.scan]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryOperation,
          typename _UnaryOperation, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
transform_inclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryOperation __binary_op,
                         _UnaryOperation __unary_op, _Tp __init)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng1, __rng2);

    return oneapi::dpl::__internal::__ranges::__pattern_transform_scan(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng1)),
        views::all_write(::std::forward<_Range2>(__rng2)), __unary_op, __init, __binary_op,
        /*inclusive=*/::std::true_type());
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation,
          typename _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
transform_inclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryOperation __binary_op,
                         _UnaryOperation __unary_op)
{
    const auto __dispatch_tag = oneapi::dpl::__ranges::__select_backend(__exec, __rng1, __rng2);

    return oneapi::dpl::__internal::__ranges::__pattern_transform_scan(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(__exec), views::all_read(::std::forward<_Range1>(__rng1)),
        views::all_write(::std::forward<_Range2>(__rng2)), __unary_op, __binary_op, /*inclusive=*/::std::true_type());
}

} // namespace ranges
} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_HETERO_BACKEND

#endif // _ONEDPL_GLUE_NUMERIC_RANGES_IMPL_H
