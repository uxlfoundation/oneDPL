// -*- C++ -*-
//===-- functional_impl.h -------------------------------------------------===//
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

#ifndef _ONEDPL_FUNCTIONAL_IMPL_H
#define _ONEDPL_FUNCTIONAL_IMPL_H

#include "onedpl_config.h"

#include <utility>    // for  std::forward
#include <functional> // for std::greater, std::less

namespace oneapi
{
namespace dpl
{

struct identity
{
    using is_transparent = void;

    template <typename _T>
    constexpr _T&&
    operator()(_T&& t) const noexcept
    {
        return std::forward<_T>(t);
    }
};

template <typename _T>
struct maximum
{
    constexpr const _T&
    operator()(const _T& a, const _T& b) const
    {
        return std::greater<_T>()(a, b) ? a : b;
    }
};

template <typename _T>
struct minimum
{
    constexpr const _T&
    operator()(const _T& a, const _T& b) const
    {
        return std::less<_T>()(a, b) ? a : b;
    }
};

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_FUNCTIONAL_IMPL_H
