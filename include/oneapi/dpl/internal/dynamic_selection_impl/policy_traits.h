// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_POLICY_TRAITS_H
#define _ONEDPL_POLICY_TRAITS_H
#include <type_traits>
namespace oneapi
{
namespace dpl
{
namespace experimental
{

template <typename Policy>
struct policy_traits
{
    using resource_type = typename std::decay_t<Policy>::resource_type;   //resource type
};

template <typename Policy>
using resource_t = typename policy_traits<Policy>::resource_type;

} // namespace experimental
} // namespace dpl
} // namespace oneapi
#endif //_ONEDPL_POLICY_TRAITS_H
