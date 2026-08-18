/*
 *  Copyright (c) Intel Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef _ONEDPL_ASYNC_EXTENSION_DEFS_H
#define _ONEDPL_ASYNC_EXTENSION_DEFS_H

#include "../pstl/hetero/dpcpp/execution_sycl_defs.h"

namespace oneapi
{
namespace dpl
{

// Public API for asynch algorithms:
namespace experimental
{

template <typename... _Ts>
oneapi::dpl::__internal::__enable_if_convertible_to_events<void, _Ts...>
wait_for_all(_Ts&&... __events);

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...> = 0>
auto
copy_async(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result,
           _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIterator, class _Function, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...> = 0>
auto
for_each_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Function __f,
               _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...> = 0>
auto
reduce_async(_ExecutionPolicy&& __exec, _ForwardIt __first, _ForwardIt __last, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt, class _T, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<_ExecutionPolicy, int, _T,
                                                                                         _Events...> = 0>
auto
reduce_async(_ExecutionPolicy&& __exec, _ForwardIt __first, _ForwardIt __last, _T init, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_double_no_default<
              _ExecutionPolicy, int, _Tp, _BinaryOperation, _Events...> = 0>
auto
reduce_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
             _BinaryOperation __binary_op, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _UnaryOperation, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...> = 0>
auto
transform_async(_ExecutionPolicy&& __exec, _ForwardIt1 first1, _ForwardIt1 last1, _ForwardIt2 d_first,
                _UnaryOperation unary_op, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _ForwardIt3, class _BinaryOperation,
          class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<
              _ExecutionPolicy, int, _BinaryOperation, _Events...> = 0>
auto
transform_async(_ExecutionPolicy&& __exec, _ForwardIt1 first1, _ForwardIt1 last1, _ForwardIt2 first2,
                _ForwardIt3 d_first, _BinaryOperation binary_op, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class _BinaryOp1, class _BinaryOp2,
          class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_double_no_default<
              _ExecutionPolicy, int, _BinaryOp1, _BinaryOp2, _Events...> = 0>
auto
transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                       _T __init, _BinaryOp1 __binary_op1, _BinaryOp2 __binary_op2, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...> = 0>
auto
transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                       _T __init, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt, class _T, class _BinaryOp, class _UnaryOp, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<_ExecutionPolicy, int,
                                                                                         _UnaryOp, _Events...> = 0>
auto
transform_reduce_async(_ExecutionPolicy&& __exec, _ForwardIt __first, _ForwardIt __last, _T __init,
                       _BinaryOp __binary_op, _UnaryOp __unary_op, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _RandomAccessIterator, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...> = 0>
auto
sort_async(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last,
           _Events&&... __dependencies);

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Compare, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<_ExecutionPolicy, int,
                                                                                         _Compare, _Events...> = 0>
auto
sort_async(_ExecutionPolicy&& __exec, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
           _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...> = 0>
auto
fill_async(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value,
           _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...> = 0>
auto
inclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                     _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _BinaryOperation, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<
              _ExecutionPolicy, int, _BinaryOperation, _Events...> = 0>
auto
inclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                     _BinaryOperation __binary_op, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _BinaryOperation, class _T,
          class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_double_no_default<
              _ExecutionPolicy, int, _BinaryOperation, _T, _Events...> = 0>
auto
inclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                     _BinaryOperation __binary_op, _T __init, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...> = 0>
auto
exclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                     _T __init, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class _BinaryOperation,
          class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<
              _ExecutionPolicy, int, _BinaryOperation, _Events...> = 0>
auto
exclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 first1, _ForwardIt1 __last1, _ForwardIt2 __first2,
                     _T __init, _BinaryOperation __binary_op, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _T, class _BinaryOperation,
          class _UnaryOperation, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...> = 0>
auto
transform_exclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1,
                               _ForwardIt2 __first2, _T __init, _BinaryOperation __binary_op,
                               _UnaryOperation __unary_op, _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _BinaryOperation, class _UnaryOperation,
          class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int, _Events...> = 0>
auto
transform_inclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1,
                               _ForwardIt2 __first2, _BinaryOperation __binary_op, _UnaryOperation __unary_op,
                               _Events&&... __dependencies);

template <class _ExecutionPolicy, class _ForwardIt1, class _ForwardIt2, class _BinaryOperation, class _UnaryOperation,
          class _T, class... _Events,
          oneapi::dpl::__internal::__enable_if_device_execution_policy_single_no_default<_ExecutionPolicy, int, _T,
                                                                                         _Events...> = 0>
auto
transform_inclusive_scan_async(_ExecutionPolicy&& __exec, _ForwardIt1 __first1, _ForwardIt1 __last1,
                               _ForwardIt2 __first2, _BinaryOperation __binary_op, _UnaryOperation __unary_op,
                               _T __init, _Events&&... __dependencies);

} // namespace experimental

} // namespace dpl

} // namespace oneapi

#endif // _ONEDPL_ASYNC_EXTENSION_DEFS_H
