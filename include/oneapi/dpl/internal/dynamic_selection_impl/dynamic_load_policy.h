// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_DYNAMIC_LOAD_POLICY_H
#define _ONEDPL_DYNAMIC_LOAD_POLICY_H

#include <mutex>
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
class dynamic_load_policy
    : public policy_base<dynamic_load_policy<ResourceType, ResourceAdapter, Backend>, ResourceAdapter, Backend,
                         execution_info::task_submission_t, execution_info::task_completion_t>
{
  protected:
    using base_t = policy_base<dynamic_load_policy<ResourceType, ResourceAdapter, Backend>, ResourceAdapter, Backend,
                               execution_info::task_submission_t, execution_info::task_completion_t>;
    friend base_t;

  public:
    using resource_type = typename base_t::resource_type;

  protected:
    using load_t = int;

    struct resource_t
    {
        resource_type e_;
        std::atomic<load_t> load_;
        resource_t(resource_type e) : e_(e), load_(0) {}
    };

    template <typename Policy>
    class dl_selection_handle_t
    {
        Policy policy_;
        std::shared_ptr<resource_t> resource_;

      public:
        dl_selection_handle_t(const Policy& p, std::shared_ptr<resource_t> r) : policy_(p), resource_(std::move(r)) {}
        using scratch_space_t =
            typename backend_traits<Backend>::template selection_scratch_t<execution_info::task_submission_t,
                                                                           execution_info::task_completion_t>;
        scratch_space_t scratch_space;

        auto
        unwrap()
        {
            return ::oneapi::dpl::experimental::unwrap(resource_->e_);
        }

        Policy
        get_policy()
        {
            return policy_;
        }

        void
        report(const execution_info::task_submission_t&) const
        {
            resource_->load_.fetch_add(1);
        }

        void
        report(const execution_info::task_completion_t&) const
        {
            resource_->load_.fetch_sub(1);
        }
    };
    using selection_type = dl_selection_handle_t<dynamic_load_policy<ResourceType, ResourceAdapter, Backend>>;

    struct selector_t
    {
        std::vector<std::shared_ptr<resource_t>> resources_;
        std::mutex m_;
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
        selector_->resources_.clear();
        selector_->resources_.reserve(u.size());
        for (auto& x : u)
        {
            selector_->resources_.push_back(std::make_shared<resource_t>(x));
        }
    }

    template <typename... Args>
    std::optional<selection_type>
    try_select(Args&&...)
    {
        if constexpr (backend_traits<Backend>::lazy_report_v)
        {
            if (this->backend_)
              this->backend_->lazy_report();
        }
        if (selector_)
        {
            std::shared_ptr<resource_t> least_loaded;
            int least_load = std::numeric_limits<load_t>::max();

            std::lock_guard<std::mutex> l(selector_->m_);
            for (auto r : selector_->resources_)
            {
                load_t v = r->load_.load();
                if (!least_loaded || v < least_load)
                {
                    least_load = v;
                    least_loaded = ::std::move(r);
                }
            }
            return std::make_optional<selection_type>(
                dynamic_load_policy<ResourceType, ResourceAdapter, Backend>(*this), least_loaded);
        }
        else
        {
            throw std::logic_error("select called before initialization");
        }
    }

  public:
    dynamic_load_policy() { base_t::initialize(); }
    dynamic_load_policy(deferred_initialization_t) {}
    dynamic_load_policy(const std::vector<ResourceType>& u, ResourceAdapter adapter = {})
    {
        base_t::initialize(u, adapter);
    }
};

//CTAD deduction guides for initializer_list
template <typename T>
dynamic_load_policy(std::initializer_list<T>)
    -> dynamic_load_policy<T, oneapi::dpl::identity,
                           oneapi::dpl::experimental::default_backend<
                               T, oneapi::dpl::identity>>; //supports dynamic_load_policy p{ {t1, t2} }
template <typename T, typename Adapter>
dynamic_load_policy(std::initializer_list<T>, Adapter)
    -> dynamic_load_policy<
        T, Adapter,
        oneapi::dpl::experimental::default_backend<T, Adapter>>; //supports dynamic_load_policy p{ {t1, t2}, adapter }

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_DYNAMIC_LOAD_POLICY_H
