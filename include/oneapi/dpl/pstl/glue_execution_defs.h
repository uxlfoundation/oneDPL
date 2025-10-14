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

#ifndef _ONEDPL_GLUE_EXECUTION_DEFS_H
#define _ONEDPL_GLUE_EXECUTION_DEFS_H

#include <type_traits>

#include "execution_defs.h"

#if !_ONEDPL_CPP17_EXECUTION_POLICIES_PRESENT
namespace std
{
// Type trait
using oneapi::dpl::execution::is_execution_policy;
#    if __INTEL_COMPILER
template <class T>
inline constexpr bool is_execution_policy_v = is_execution_policy<T>::value;
#    else
using oneapi::dpl::is_execution_policy_v;
#    endif

namespace execution
{
// Standard C++ policy classes
using oneapi::dpl::execution::parallel_policy;
using oneapi::dpl::execution::parallel_unsequenced_policy;
using oneapi::dpl::execution::sequenced_policy;
// Standard predefined policy instances
using oneapi::dpl::execution::par;
using oneapi::dpl::execution::par_unseq;
using oneapi::dpl::execution::seq;
// Implementation-defined names
// Unsequenced policy is not yet standard, but for consistency
// we include it into namespace std::execution as well
using oneapi::dpl::execution::unseq;
using oneapi::dpl::execution::unsequenced_policy;
} // namespace execution
} // namespace std
#endif // !_ONEDPL_CPP17_EXECUTION_POLICIES_PRESENT

#if _ONEDPL_BACKEND_SYCL
#    include "hetero/algorithm_impl_hetero.h"
#    include "hetero/numeric_impl_hetero.h"
#endif

#include "algorithm_impl.h"
#include "numeric_impl.h"
#include "parallel_backend.h"

#endif // _ONEDPL_GLUE_EXECUTION_DEFS_H
