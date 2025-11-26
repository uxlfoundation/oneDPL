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
#include "backend_traits.h"
namespace oneapi
{
namespace dpl
{
namespace experimental
{

template <typename Policy>
struct policy_traits
{
    using backend_type = typename std::decay_t<Policy>::backend_type;
    using resource_type = typename std::decay_t<Policy>::resource_type;
    static constexpr bool has_wait_type_v = oneapi::dpl::experimental::backend_traits<backend_type>::has_wait_type_v;
    using wait_type = typename oneapi::dpl::experimental::backend_traits<backend_type>::wait_type;
};

template <typename Policy>
using backend_t = typename policy_traits<Policy>::backend_type;

template <typename Policy>
using resource_t = typename policy_traits<Policy>::resource_type;

template <typename Policy>
inline constexpr bool has_wait_type_v = policy_traits<Policy>::has_wait_type_v;

template <typename Policy>
using wait_t = typename policy_traits<Policy>::wait_type;

} // namespace experimental
} // namespace dpl
} // namespace oneapi
#endif //_ONEDPL_POLICY_TRAITS_H
