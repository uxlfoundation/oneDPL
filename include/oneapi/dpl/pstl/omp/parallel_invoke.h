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

#ifndef _ONEDPL_INTERNAL_OMP_PARALLEL_INVOKE_H
#define _ONEDPL_INTERNAL_OMP_PARALLEL_INVOKE_H

#include "util.h"

namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <typename _F1, typename _F2>
void
__parallel_invoke_body(_F1&& __f1, _F2&& __f2)
{
    _ONEDPL_PRAGMA(omp taskgroup)
    {
        _ONEDPL_PRAGMA(omp task untied mergeable) { std::forward<_F1>(__f1)(); }
        _ONEDPL_PRAGMA(omp task untied mergeable) { std::forward<_F2>(__f2)(); }
    }
}

template <class _ExecutionPolicy, typename _F1, typename _F2>
void
__parallel_invoke(oneapi::dpl::__internal::__omp_backend_tag, _ExecutionPolicy&&, _F1&& __f1, _F2&& __f2)
{
    if (omp_in_parallel())
    {
        oneapi::dpl::__omp_backend::__parallel_invoke_body(std::forward<_F1>(__f1), std::forward<_F2>(__f2));
    }
    else
    {
        _ONEDPL_PRAGMA(omp parallel)
        _ONEDPL_PRAGMA(omp single nowait)
        oneapi::dpl::__omp_backend::__parallel_invoke_body(std::forward<_F1>(__f1), std::forward<_F2>(__f2));
    }
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL_INTERNAL_OMP_PARALLEL_INVOKE_H
