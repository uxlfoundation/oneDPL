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
#else
#    include "oneapi/dpl/internal/dynamic_selection_impl/default_backend.h"
#endif

namespace oneapi 
{
namespace dpl 
{
namespace experimental 
{

#if _DS_BACKEND_SYCL != 0
template <typename ResourceType = sycl::queue, typename ExtraResourceType = oneapi::dpl::experimental::empty_extra_resource, typename Backend = default_backend<ResourceType, ExtraResourceType>>
#else
template <typename ResourceType, typename ExtraResourceType = oneapi::dpl::experimental::empty_extra_resource, typename Backend = default_backend<ResourceType, ExtraResourceType>>
#endif
class fixed_resource_policy : public policy_base<fixed_resource_policy<ResourceType, ExtraResourceType, Backend>, ResourceType, ExtraResourceType, Backend> 
{
  protected:
    using base_t = policy_base<fixed_resource_policy<ResourceType, ExtraResourceType, Backend>, ResourceType, ExtraResourceType, Backend>;
    using resource_container_size_t = typename base_t::resource_container_size_t;

    struct selector_t 
    {
        typename base_t::resource_container_t resources_;
        typename base_t::extra_resource_container_t extra_resources_;
        std::size_t index_ = 0;

        auto
        get_extra_resource() const
        {
            if constexpr (base_t::has_extra_resources_v)
            {
                return extra_resources_[index_];
            }
            else
            {
                return oneapi::dpl::experimental::empty_extra_resource{};
            }
        }
    };

    std::shared_ptr<selector_t> selector_;

  public:
    using resource_type = typename base_t::resource_type;
    using typename base_t::selection_type;
    using wait_type = typename Backend::wait_type; //TODO: Get from policy_base instead?
    using extra_resource_type = ExtraResourceType;

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

    fixed_resource_policy(const std::vector<resource_type>& u, const std::vector<ExtraResourceType>& v, std::size_t index = 0) 
    { 
        base_t::initialize(u, v); 
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
        if constexpr (base_t::has_extra_resources_v)
        {
            auto er = base_t::get_extra_resources();
            selector_->extra_resources_.clear();
            selector_->extra_resources_.reserve(er.size());
            for (const auto& e : er)
            {
                selector_->extra_resources_.push_back(ExtraResourceType{e});
            }
        }
    }

    template <typename... Args>
    selection_type 
    select_impl(Args&&...) 
    {
	if (selector_) 
	{
            return selection_type{*this, selector_->resources_[selector_->index_], selector_->get_extra_resource()};
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

