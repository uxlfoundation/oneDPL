// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_SCORING_POLICY_DEFS_H
#define _ONEDPL_SCORING_POLICY_DEFS_H

#include "oneapi/dpl/internal/dynamic_selection_traits.h"
namespace oneapi
{
namespace dpl
{
namespace experimental
{
template <typename Policy, typename Resource>
class basic_selection_handle_t
{
    Policy p_;
    Resource e_;

  public:
    explicit basic_selection_handle_t(const Policy& p, Resource e = Resource{}) : p_(p), e_(std::move(e)) {}
    auto
    unwrap()
    {
        return oneapi::dpl::experimental::unwrap(e_);
    }
    Policy
    get_policy()
    {
        return p_;
    }
};

} // namespace experimental
} // namespace dpl
} //namespace oneapi

#endif /* _ONEDPL_SCORING_POLICY_DEFS_H */
