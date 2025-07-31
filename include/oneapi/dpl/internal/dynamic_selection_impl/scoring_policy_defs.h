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

class empty_extra_resource
{
    // This class is used to indicate that no extra resource is needed.
    // It can be used as a template parameter for default_backend.
};

struct no_extra_resources
{
    using type = void;
    std::size_t
    size() const noexcept
    {
        return 0;
    }
};

template <typename Policy, typename Resource, typename ExtraResourceType = oneapi::dpl::experimental::empty_extra_resource>
class basic_selection_handle_t
{
    Policy p_;
    Resource e_;
    ExtraResourceType r_;

  public:
    explicit basic_selection_handle_t(const Policy& p, Resource e = Resource{}, ExtraResourceType r = ExtraResourceType{}) : p_(p), e_(std::move(e)), r_(std::move(r)) {}
    auto
    unwrap()
    {
        return oneapi::dpl::experimental::unwrap(e_);
    }

    ExtraResourceType
    get_extra_resource()
    {
        return r_;
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
