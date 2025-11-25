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
#include "oneapi/dpl/internal/dynamic_selection_impl/default_backend.h"

#if _DS_BACKEND_SYCL
#    include "oneapi/dpl/internal/dynamic_selection_impl/sycl_backend.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace experimental
{

#if _DS_BACKEND_SYCL
template <typename ResourceType = sycl::queue, typename ResourceAdapter = oneapi::dpl::identity,
          typename Backend = default_backend<ResourceType, ResourceAdapter>>
#else
template <typename ResourceType, typename ResourceAdapter = oneapi::dpl::identity,
          typename Backend = default_backend<ResourceType, ResourceAdapter>>
#endif
class round_robin_policy
    : public policy_base<round_robin_policy<ResourceType, ResourceAdapter, Backend>, ResourceAdapter, Backend>
{
  protected:
    using base_t = policy_base<round_robin_policy<ResourceType, ResourceAdapter, Backend>, ResourceAdapter, Backend>;
    friend base_t;

  public:
    using resource_type = typename base_t::resource_type;

  protected:
    using typename base_t::selection_type;
    using resource_container_size_t = typename std::vector<resource_type>::size_type;

    struct selector_t
    {
        std::vector<resource_type> resources_;
        std::vector<resource_type>::size_type num_contexts_;
        std::atomic<resource_container_size_t> next_context_;
    };

    std::shared_ptr<selector_t> selector_;

    void
    initialize_state()
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
    try_select(Args&&...)
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

  public:
    using typename base_t::backend_t;

    round_robin_policy() { base_t::initialize(); }
    round_robin_policy(deferred_initialization_t) {}
    round_robin_policy(const std::vector<ResourceType>& u, ResourceAdapter adapter = {})
    {
        base_t::initialize(u, adapter);
    }
};

//CTAD deduction guides for initializer_list
template <typename T>
round_robin_policy(std::initializer_list<T>)
    -> round_robin_policy<T, oneapi::dpl::identity,
                          oneapi::dpl::experimental::default_backend<
                              T, oneapi::dpl::identity>>; //supports round_robin_policy p{ {t1, t2} }

template <typename T, typename Adapter>
round_robin_policy(std::initializer_list<T>, Adapter)
    -> round_robin_policy<
        T, Adapter,
        oneapi::dpl::experimental::default_backend<T, Adapter>>; //supports round_robin_policy p{ {t1, t2}, adapter }

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_ROUND_ROBIN_POLICY_H
