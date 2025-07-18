// -*- C++ -*-
//===-- get_impl.h --------------------------------------------------------===//
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

#ifndef _ONEDPL_GET_IMPL_H
#define _ONEDPL_GET_IMPL_H

#include <cstddef> // for std::size_t
#include <utility> // for std::get, std::forward

namespace oneapi
{
namespace dpl
{
namespace __internal
{
template <std::size_t _Idx, typename _T>
decltype(auto)
__get(_T&& __t)
{
    using std::get;

    return get<_Idx>(std::forward<_T>(__t));
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

namespace __dpl_internal = oneapi::dpl::__internal;

#endif // _ONEDPL_GET_IMPL_H
