// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_ROUND_ROBIN_POLICY_H
#define _ONEDPL_ROUND_ROBIN_POLICY_H

#include <optional>
#include "oneapi/dpl/internal/dynamic_selection_impl/policy_base.h"
#include "oneapi/dpl/functional"

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
template <typename ResourceType = sycl::queue, typename ResourceAdapter = oneapi::dpl::identity,
          typename Backend = default_backend<ResourceType, ResourceAdapter>>
#else
template <typename ResourceType, typename ResourceAdapter = oneapi::dpl::identity,
          typename Backend = default_backend<ResourceType, ResourceAdapter>>
#endif
class round_robin_policy
    : public policy_base<round_robin_policy<ResourceType, ResourceAdapter, Backend>, ResourceType, Backend>
{
  protected:
    using base_t = policy_base<round_robin_policy<ResourceType, ResourceAdapter, Backend>, ResourceType, Backend>;
    using resource_container_size_t = typename base_t::resource_container_size_t;
    using typename base_t::selection_type;

    struct selector_t
    {
        typename base_t::resource_container_t resources_;
        resource_container_size_t num_contexts_;
        std::atomic<resource_container_size_t> next_context_;
    };

    std::shared_ptr<selector_t> selector_;

  public:
    using resource_type = typename base_t::resource_type;
    using typename base_t::backend_t;

    round_robin_policy() { base_t::initialize(); }
    round_robin_policy(deferred_initialization_t) {}
    round_robin_policy(const std::vector<resource_type>& u, ResourceAdapter adapter = {})
    {
        base_t::initialize(u, adapter);
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
        selector_->num_contexts_ = u.size();
        selector_->next_context_ = 0;
    }

    template <typename... Args>
    std::optional<selection_type>
    try_select_impl(Args&&...)
    {
        if (selector_)
        {
            resource_container_size_t current;

            while (true)
            {
                current = selector_->next_context_.load();
                auto next = (current + 1) % selector_->num_contexts_;
                if (selector_->next_context_.compare_exchange_strong(current, next))
                    break;
            }
            return std::make_optional<selection_type>(*this, selector_->resources_[current]);
        }
        else
        {
            throw std::logic_error("select called before initialization");
        }
    }
};

//CTAD deduction guide for initializer_list
template <typename T>
round_robin_policy(std::initializer_list<T>) -> round_robin_policy<T>; //supports round_robin_policy p{ {t1, t2} }

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_ROUND_ROBIN_POLICY_H
