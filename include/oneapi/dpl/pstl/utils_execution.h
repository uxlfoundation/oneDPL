
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
#ifndef _ONEDPL_UTILS_EXECUTION_H
#define _ONEDPL_UTILS_EXECUTION_H

#include "onedpl_config.h"

#if (_PSTL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN || _ONEDPL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN)

#include "execution_defs.h"

namespace oneapi::dpl::__internal
{

inline const oneapi::dpl::execution::parallel_policy&
get_unvectorized_policy(const oneapi::dpl::execution::parallel_unsequenced_policy&)
{
    return oneapi::dpl::execution::par;
}

inline const oneapi::dpl::execution::sequenced_policy&
get_unvectorized_policy(const oneapi::dpl::execution::unsequenced_policy&)
{
    return oneapi::dpl::execution::seq;
}

template <typename _ExecutionPolicy>
const _ExecutionPolicy&
get_unvectorized_policy(const _ExecutionPolicy& __exec)
{
    return __exec;
}

} // namespace oneapi::dpl::__internal

#endif // (_PSTL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN || _ONEDPL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN)

#endif // _ONEDPL_UTILS_EXECUTION_H
