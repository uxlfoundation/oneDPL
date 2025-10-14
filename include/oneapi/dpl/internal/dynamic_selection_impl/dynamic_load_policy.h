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

#include <atomic>
#include <memory>
#include <limits>
#include <mutex>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include "oneapi/dpl/internal/dynamic_selection_traits.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/backend_traits.h"
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
template <typename Backend = sycl_backend>
#else
template <typename Backend>
#endif
struct dynamic_load_policy
{
  private:
    using backend_t = Backend;
    using load_t = int;
    using execution_resource_t = typename backend_t::execution_resource_t;

  public:
    //Policy Traits
    using resource_type = typename backend_t::resource_type;
    using wait_type = typename backend_t::wait_type;

    struct resource_t
    {
        execution_resource_t e_;
        std::atomic<load_t> load_;
        resource_t(execution_resource_t e) : e_(e), load_(0) {}
    };
    using resource_container_t = std::vector<std::shared_ptr<resource_t>>;

    template <typename Policy>
    class dl_selection_handle_t
    {
        Policy policy_;
        std::shared_ptr<resource_t> resource_;

      public:
        dl_selection_handle_t(const Policy& p, std::shared_ptr<resource_t> r) : policy_(p), resource_(std::move(r)) {}

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

    std::shared_ptr<backend_t> backend_;

    using selection_type = dl_selection_handle_t<dynamic_load_policy<Backend>>;

    struct state_t
    {
        resource_container_t resources_;
        std::mutex m_;
    };

    std::shared_ptr<state_t> state_;

    void
    initialize()
    {
        if (!state_)
        {
            backend_ = std::make_shared<backend_t>();
            state_ = std::make_shared<state_t>();
            auto u = get_resources();
            for (auto x : u)
            {
                state_->resources_.push_back(std::make_shared<resource_t>(x));
            }
        }
    }

    void
    initialize(const std::vector<resource_type>& u)
    {
        if (!state_)
        {
            backend_ = std::make_shared<backend_t>(u);
            state_ = std::make_shared<state_t>();
            auto container = get_resources();
            for (auto x : container)
            {
                state_->resources_.push_back(std::make_shared<resource_t>(x));
            }
        }
    }

    dynamic_load_policy(deferred_initialization_t) {}

    dynamic_load_policy() { initialize(); }

    dynamic_load_policy(const std::vector<resource_type>& u) { initialize(u); }

    auto
    get_resources()
    {
        if (backend_)
            return backend_->get_resources();
        else
            throw std::logic_error("get_resources called before initialization");
    }

    template <typename... Args>
    selection_type
    select(Args&&...)
    {
        if constexpr (backend_traits::lazy_report_v<Backend>)
        {
            backend_->lazy_report();
        }
        if (state_)
        {
            std::shared_ptr<resource_t> least_loaded;
            int least_load = std::numeric_limits<load_t>::max();

            std::lock_guard<std::mutex> l(state_->m_);
            for (auto r : state_->resources_)
            {
                load_t v = r->load_.load();
                if (!least_loaded || v < least_load)
                {
                    least_load = v;
                    least_loaded = ::std::move(r);
                }
            }
            return selection_type{dynamic_load_policy<Backend>(*this), least_loaded};
        }
        else
        {
            throw std::logic_error("select called before initialization");
        }
    }

    template <typename Function, typename... Args>
    auto
    submit(selection_type e, Function&& f, Args&&... args)
    {
        if (backend_)
            return backend_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
        else
            throw std::logic_error("submit called before initialization");
    }

    auto
    get_submission_group()
    {
        if (backend_)
        {
            return backend_->get_submission_group();
        }
        else
        {
            throw std::logic_error("get_submission_group called before initialization");
        }
    }
};
} // namespace experimental

} // namespace dpl

} // namespace oneapi

#endif //_ONEDPL_DYNAMIC_LOAD_POLICY_H
