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
    using selection_type = typename std::decay_t<Policy>::selection_type; //selection type
    using resource_type = typename std::decay_t<Policy>::resource_type;   //resource type
    using extra_resource_type = typename std::decay_t<Policy>::extra_resource_type; //extra resource type
    using wait_type = typename std::decay_t<Policy>::wait_type; //wait_type
};

template <typename Policy>
using selection_t = typename policy_traits<Policy>::selection_type;

template <typename Policy>
using resource_t = typename policy_traits<Policy>::resource_type;

template <typename Policy>
using extra_resource_t = typename policy_traits<Policy>::extra_resource_type;


template <typename Policy>
using wait_t = typename policy_traits<Policy>::wait_type;
} // namespace experimental
} // namespace dpl
} // namespace oneapi
#endif //_ONEDPL_POLICY_TRAITS_H
