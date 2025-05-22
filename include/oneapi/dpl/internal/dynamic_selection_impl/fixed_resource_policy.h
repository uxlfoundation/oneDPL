// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_FIXED_RESOURCE_POLICY_H
#define _ONEDPL_FIXED_RESOURCE_POLICY_H

#include "oneapi/dpl/internal/dynamic_selection_impl/policy_base.h"

#if _DS_BACKEND_SYCL != 0
#    include "oneapi/dpl/internal/dynamic_selection_impl/sycl_backend.h"
#endif

namespace oneapi 
{
namespace dpl 
{
namespace experimental 
{

#if _DS_BACKEND_SYCL != 0
template <typename ResourceType = sycl::queue, typename Backend = default_backend<ResourceType>>
#else
template <typename ResourceType, typename Backend>
#endif
class fixed_resource_policy : public policy_base<fixed_resource_policy<ResourceType, Backend>, ResourceType, Backend> 
{
  protected:
    using base_t = policy_base<fixed_resource_policy<ResourceType, Backend>, ResourceType, Backend>;
    using resource_container_size_t = typename base_t::resource_container_size_t;

    struct selector_t 
    {
        typename base_t::resource_container_t resources_;
	::std::size_t index_ = 0;
    };

    std::shared_ptr<selector_t> selector_;

  public:
    using resource_type = typename base_t::resource_type;
    using typename base_t::selection_type;

    fixed_resource_policy(::std::size_t index = 0) 
    { 
        base_t::initialize(); 
	selector_->index_ = index;
    }
    fixed_resource_policy(deferred_initialization_t) {}
    fixed_resource_policy(const std::vector<resource_type>& u, ::std::size_t index = 0) 
    { 
        base_t::initialize(u); 
	selector_->index_ = index;
    }

    void 
    initialize_impl() 
    {
        if (!selector_) 
	{
	    selector_ = std::make_shared<selector_t>();
	}
	auto u = base_t::get_resources();
        selector_->resources_ = u;
        selector_->index_ = 0;
    }

    template <typename... Args>
    selection_type 
    select_impl(Args&&...) 
    {
	if (selector_) 
	{
            return selection_type{*this, selector_->resources_[selector_->index_]};
	}
	else
	{
	    throw std::logic_error("select called before initialization");
	}
    }
};

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_FIXED_RESOURCE_POLICY_H

