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

#ifndef _ONEDPL_INTERNAL_OMP_PARALLEL_TRANSFORM_SCAN_H
#define _ONEDPL_INTERNAL_OMP_PARALLEL_TRANSFORM_SCAN_H

#include "util.h"

namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <class _ExecutionPolicy, class _Index, class _Up, class _Tp, class _Cp, class _Rp, class _Sp>
_Tp
__parallel_transform_scan(oneapi::dpl::__internal::__omp_backend_tag, _ExecutionPolicy&&, _Index __n, _Up /* __u */,
                          _Tp __init, _Cp /* __combine */, _Rp /* __brick_reduce */, _Sp __scan)
{
    // TODO: parallelize this function.
    return __scan(_Index(0), __n, __init);
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL_INTERNAL_OMP_PARALLEL_TRANSFORM_SCAN_H
