// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef _ONEDPL_POLICY_BASE_H
#define _ONEDPL_POLICY_BASE_H

#include <memory>
#include <vector>
#include <stdexcept>
#include <utility>
#include <optional>
#include <thread>
#include "oneapi/dpl/functional"
#include "oneapi/dpl/internal/dynamic_selection_traits.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"

namespace oneapi 
{
namespace dpl 
{
namespace experimental 
{

template <typename Policy, typename ResourceType, typename Backend, typename... ReportReqs>
class policy_base 
{
  protected:
    using backend_t = Backend;
    using resource_container_t = typename backend_t::resource_container_t;
    using resource_container_size_t = typename resource_container_t::size_type;
    using execution_resource_t = typename backend_t::execution_resource_t;
    using wrapped_resource_t = execution_resource_t;
    using selection_type = basic_selection_handle_t<Policy, execution_resource_t>;
    using report_reqs_t = std::tuple<ReportReqs...>;

  public:
    using resource_type = decltype(unwrap(std::declval<wrapped_resource_t>()));

  protected:
    std::shared_ptr<backend_t> backend_;

  public:
    auto 
    get_resources() const 
    {
        if (backend_) return backend_->get_resources();
        throw std::logic_error("get_resources called before initialization");
    }

    void 
    initialize()
    {
        if (!backend_) backend_ = std::make_shared<backend_t>(report_reqs_t{});
        static_cast<Policy*>(this)->initialize_impl();
    }

    void 
    initialize(const std::vector<resource_type>& u)
    {
        if (!backend_) backend_ = std::make_shared<backend_t>(u, oneapi::dpl::identity());
        static_cast<Policy*>(this)->initialize_impl();
    }

    template <typename ResourceAdapter, typename... Args>
    void
    initialize(const std::vector<resource_type>& u, ResourceAdapter adapter, Args... args)
    {
        if (!backend_) backend_ = std::make_shared<backend_t>(u, adapter);
        static_cast<Policy*>(this)->initialize_impl(args...);
    }

    template <typename... Args>
    auto
    select_impl(Args&&... args)
    {
        auto e = static_cast<Policy*>(this)->try_select_impl(args...);
        while (!e.has_value())
        {
            e = static_cast<Policy*>(this)->try_select_impl(args...);
            std::this_thread::yield();
        }
        return e.value();
    }

    template <typename Function, typename... Args>
    auto //std::optional of the "wait type"
    try_submit(Function&& f, Args&&... args) -> 
        std::optional<decltype(backend_->submit(std::declval<decltype(select_impl(f, args...))>(), std::forward<Function>(f), std::forward<Args>(args)...))>
    {
        if (backend_)
        {
            auto e = static_cast<Policy*>(this)->try_select_impl(f, args...);
            if (!e.has_value())
            {
                // return an empty optional
                return {};
            }
            else
            {
                return std::make_optional<decltype(backend_->submit(e.value(), std::forward<Function>(f), std::forward<Args>(args)...))>(
                    backend_->submit(e.value(), std::forward<Function>(f), std::forward<Args>(args)...));
            }
        }
        throw std::logic_error("submit called before initialization");
    }

    template <typename Function, typename... Args>
    auto 
    submit(Function&& f, Args&&... args) 
    {
        if (backend_)
        {
            auto e = static_cast<Policy*>(this)->select_impl(f, args...);
            return backend_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
        }
        throw std::logic_error("submit called before initialization");
    }

    template <typename Function, typename... Args>
    void
    submit_and_wait(Function&& f, Args&&... args) 
    {
        oneapi::dpl::experimental::wait(static_cast<Policy*>(this)->submit(std::forward<Function>(f), std::forward<Args>(args)...));
    }

    auto 
    get_submission_group() 
    {
        if (backend_) return backend_->get_submission_group();
        throw std::logic_error("get_submission_group called before initialization");
    }
};

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_POLICY_BASE_H

