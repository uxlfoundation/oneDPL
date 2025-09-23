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

#ifndef _ONEDPL_RANGES_DEFS_H
#define _ONEDPL_RANGES_DEFS_H

#include "ranges/nanorange.hpp"
#include "ranges/nanorange_ext.h"

#if _ONEDPL_CPP20_RANGES_PRESENT
#include <ranges>
#endif

#include "utils_ranges.h"
#if _ONEDPL_BACKEND_SYCL
#    include "hetero/dpcpp/utils_ranges_sycl.h"
#endif

namespace oneapi
{
namespace dpl
{

namespace experimental
{
namespace ranges
{

//custom views
#if _ONEDPL_BACKEND_SYCL
using oneapi::dpl::__ranges::all_view;
#endif // _ONEDPL_BACKEND_SYCL
using oneapi::dpl::__ranges::guard_view;
using oneapi::dpl::__ranges::zip_view;

//views
using __nanorange::nano::ranges::drop_view;
using __nanorange::nano::ranges::iota_view;
using __nanorange::nano::ranges::reverse_view;
using __nanorange::nano::ranges::take_view;
using __nanorange::nano::ranges::transform_view;

//adaptors
namespace views
{
#if _ONEDPL_BACKEND_SYCL
using oneapi::dpl::__ranges::views::all;
using oneapi::dpl::__ranges::views::all_read;
using oneapi::dpl::__ranges::views::all_write;
using oneapi::dpl::__ranges::views::host_all;
#endif // _ONEDPL_BACKEND_SYCL

using __nanorange::nano::views::drop;
using __nanorange::nano::views::fill;
using __nanorange::nano::views::generate;
using __nanorange::nano::views::iota;
using __nanorange::nano::views::reverse;
using __nanorange::nano::views::rotate;
using __nanorange::nano::views::take;
using __nanorange::nano::views::transform;

using __nanorange::nano::subrange;
} // namespace views

} // namespace ranges
} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_RANGES_DEFS_H
