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

#ifndef _ONEDPL_GLUE_MEMORY_DEFS_H
#define _ONEDPL_GLUE_MEMORY_DEFS_H

#include "execution_defs.h"

namespace oneapi
{
namespace dpl
{

// [uninitialized.copy]

template <class _ExecutionPolicy, class _InputIterator, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_copy(_ExecutionPolicy&& __exec, _InputIterator __first, _InputIterator __last, _ForwardIterator __result);

template <class _ExecutionPolicy, class _InputIterator, class _Size, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_copy_n(_ExecutionPolicy&& __exec, _InputIterator __first, _Size __n, _ForwardIterator __result);

// [uninitialized.move]

template <class _ExecutionPolicy, class _InputIterator, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_move(_ExecutionPolicy&& __exec, _InputIterator __first, _InputIterator __last, _ForwardIterator __result);

template <class _ExecutionPolicy, class _InputIterator, class _Size, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_move_n(_ExecutionPolicy&& __exec, _InputIterator __first, _Size __n, _ForwardIterator __result);

// [uninitialized.fill]

template <class _ExecutionPolicy, class _ForwardIterator,
          class _Tp = typename std::iterator_traits<_ForwardIterator>::value_type>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
uninitialized_fill(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value);

template <class _ExecutionPolicy, class _ForwardIterator, class _Size,
          class _Tp = typename std::iterator_traits<_ForwardIterator>::value_type>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_fill_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n, const _Tp& __value);

// [specialized.destroy]

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
destroy(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
destroy_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n);

// [uninitialized.construct.default]

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
uninitialized_default_construct(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_default_construct_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n);

// [uninitialized.construct.value]

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
uninitialized_value_construct(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_value_construct_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n);

} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL_GLUE_MEMORY_DEFS_H
