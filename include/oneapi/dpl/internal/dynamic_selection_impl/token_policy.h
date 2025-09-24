// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_TOKEN_POLICY_H
#define _ONEDPL_TOKEN_POLICY_H

#include <mutex>
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
template <typename ResourceType = sycl::queue, typename ResourceAdapter = oneapi::dpl::identity, typename Backend = default_backend<ResourceType, ResourceAdapter>>
#else
template <typename ResourceType, typename ResourceAdapter = oneapi::dpl::identity, typename Backend = default_backend<ResourceType, ResourceAdapter>>
#endif
class token_policy : public policy_base<token_policy<ResourceType, ResourceAdapter, Backend>, ResourceType, Backend>
{
    int capacity;
  protected:
    using base_t = policy_base<token_policy<ResourceType, ResourceAdapter, Backend>, ResourceType, Backend>;
    using resource_container_size_t = typename base_t::resource_container_size_t;

    using execution_resource_t = typename base_t::execution_resource_t;

    //token
    struct token_t
    {
        std::atomic<int>& token_availability_;
        token_t(std::atomic<int>& a) : token_availability_(a) {}
        ~token_t()
       	{ 
	    token_availability_.fetch_sub(1); 
	}
    };
	
    struct resource_t
    {
        execution_resource_t e_;
        std::atomic<int> availability_; 
        resource_t(execution_resource_t e) : e_(e), availability_(0) {}
    };
    using resource_container_t = std::vector<std::shared_ptr<resource_t>>;


    template <typename Policy>
    class token_selection_handle_t
    {
        Policy policy_;
        std::shared_ptr<resource_t> resource_;
	std::shared_ptr<token_t> token_; 

      public:
        token_selection_handle_t(const Policy& p, std::shared_ptr<resource_t> r, std::shared_ptr<token_t> t) : policy_(p), resource_(std::move(r)), token_(std::move(t)) {} /// 
        auto
        unwrap()
        {
            return ::oneapi::dpl::experimental::unwrap(resource_->e_);
        }

        Policy
        get_policy()
        {
            return policy_;
        };

	// keeps the selection handle alive TODO: Replace with a trait implementation
        void
        report(const execution_info::task_completion_t&) const
        {

        }
    };

    struct selector_t
    {
        resource_container_t resources_;
    };

    std::shared_ptr<selector_t> selector_;

  public:
    using selection_type = token_selection_handle_t<token_policy<ResourceType, ResourceAdapter, Backend>>;
    using resource_type = typename base_t::resource_type;
    using wait_type = typename Backend::wait_type; //TODO: Get from policy_base instead?

    token_policy(const int& c = 1):capacity(c) { base_t::initialize(); }
    token_policy(deferred_initialization_t, const int& c = 1) {}
    token_policy(const std::vector<resource_type>& u, ResourceAdapter adapter = {}, const int& c = 1):capacity(c) { base_t::initialize(u, adapter); }

    void
    initialize_impl()
    {
        if (!selector_)
	{
	    selector_ = std::make_shared<selector_t>();
	}
	auto u = base_t::get_resources();
	selector_->resources_.clear();
        for (auto x : u)
        {
            selector_->resources_.push_back(std::make_shared<resource_t>(x));
        }
    }

    template <typename... Args>
    auto
    select_impl(Args&&...)
    {
	if (selector_)
        {
            std::shared_ptr<resource_t> available_resource;

            while(true) 
	    {
                for (auto r : selector_->resources_)
                {
		    int expected = r->availability_.load();
		    while (expected < capacity) //resource is available 
		    {
	                if (r->availability_.compare_exchange_weak(expected, expected + 1)) 
		        {
                            available_resource = ::std::move(r);
			    auto token = std::make_shared<token_t>(available_resource->availability_);
                            return selection_type{token_policy<ResourceType, ResourceAdapter, Backend>(*this), available_resource, token};
		        }
		    }  
                }
	        std::this_thread::yield();
	    }
        }
        else
        {
            throw std::logic_error("select called before initialization");
        }
    }
};

//CTAD deduction guide for initializer_list
template<typename T>
token_policy(std::initializer_list<T>) -> token_policy<T>; //token_policy p{ {t1, t2} }

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_TOKEN_POLICY_H

