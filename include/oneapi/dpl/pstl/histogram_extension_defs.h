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

#ifndef _ONEDPL_HISTOGRAM_EXTENSION_DEFS_H
#define _ONEDPL_HISTOGRAM_EXTENSION_DEFS_H

#include "onedpl_config.h"

namespace oneapi
{
namespace dpl
{

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _Size, typename _ValueType,
          typename _RandomAccessIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator2>
histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 first, _RandomAccessIterator1 last, _Size num_bins,
          _ValueType first_bin_min_val, _ValueType last_bin_max_val, _RandomAccessIterator2 histogram_first);

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
          typename _RandomAccessIterator3>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator3>
histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 first, _RandomAccessIterator1 last,
          _RandomAccessIterator2 boundary_first, _RandomAccessIterator2 boundary_last,
          _RandomAccessIterator3 histogram_first);

} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_HISTOGRAM_EXTENSION_DEFS_H
