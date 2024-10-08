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

#ifndef _ONEDPL_GLUE_NUMERIC_RANGES_DEFS_H
#define _ONEDPL_GLUE_NUMERIC_RANGES_DEFS_H

#include "execution_defs.h"
#include "utils_ranges.h"

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
reduce(_ExecutionPolicy&& __exec, _Range&& __rng, _Tp __init, _BinaryOperation __binary_op);

template <typename _ExecutionPolicy, typename _Range, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
reduce(_ExecutionPolicy&& __exec, _Range&& __rng, _Tp __init);

template <typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, oneapi::dpl::__internal::__value_t<_Range>>
reduce(_ExecutionPolicy&& __exec, _Range&& __rng);

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Tp __init);

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp, typename _BinaryOperation1,
          typename _BinaryOperation2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Tp __init,
                 _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2);

template <typename _ExecutionPolicy, typename _Range, typename _Tp, typename _BinaryOperation, typename _UnaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _Range&& __rng, _Tp __init, _BinaryOperation __binary_op,
                 _UnaryOperation __unary_op);

// [exclusive.scan]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
exclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Tp __init);

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp, typename _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
exclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Tp __init, _BinaryOperation __binary_op);

// [inclusive.scan]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
inclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2);

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
inclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryOperation __binary_op);

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp, typename _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
inclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryOperation __binary_op, _Tp __init);

// [transform.exclusive.scan]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp, typename _BinaryOperation,
          typename _UnaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
transform_exclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Tp __init,
                         _BinaryOperation __binary_op, _UnaryOperation __unary_op);

// [transform.inclusive.scan]

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryOperation,
          typename _UnaryOperation, typename _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
transform_inclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryOperation __binary_op,
                         _UnaryOperation __unary_op, _Tp __init);

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation,
          typename _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      oneapi::dpl::__internal::__difference_t<_Range2>>
transform_inclusive_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryOperation __binary_op,
                         _UnaryOperation __unary_op);

} // namespace ranges
} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_GLUE_NUMERIC_RANGES_DEFS_H
