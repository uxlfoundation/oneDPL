// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_STATIC_POLICY_IMPL_H
#define _ONEDPL_STATIC_POLICY_IMPL_H
#include <vector>
#include <type_traits>
#include <memory>
#include <stdexcept>
#include <utility>
#include "oneapi/dpl/internal/dynamic_selection_traits.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"
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
struct fixed_resource_policy
{
  private:
    using backend_t = Backend;
    using resource_container_t = typename backend_t::resource_container_t;
    using execution_resource_t = typename backend_t::execution_resource_t;
    using wrapped_resource_t = execution_resource_t;

  public:
    //policy traits
    using resource_type = decltype(unwrap(std::declval<wrapped_resource_t>()));
    using selection_type =
        oneapi::dpl::experimental::basic_selection_handle_t<fixed_resource_policy<Backend>, execution_resource_t>;
    using wait_type = typename backend_t::wait_type;

  private:
    std::shared_ptr<backend_t> backend_;

    struct state_t
    {
        resource_container_t resources_;
        ::std::size_t index_ = 0;
    };

    std::shared_ptr<state_t> state_;

  public:
    fixed_resource_policy(::std::size_t index = 0) { initialize(index); }

    fixed_resource_policy(deferred_initialization_t) {}

    fixed_resource_policy(const std::vector<resource_type>& u, ::std::size_t index = 0) { initialize(u, index); }

    auto
    get_resources() const
    {
        if (backend_)
        {
            return backend_->get_resources();
        }
        else
        {
            throw std::logic_error("get_resources called before initialization");
        }
    }

    void
    initialize(::std::size_t index = 0)
    {
        if (!state_)
        {
            backend_ = std::make_shared<backend_t>();
            state_ = std::make_shared<state_t>();
            state_->resources_ = get_resources();
            state_->index_ = index;
        }
    }

    void
    initialize(const std::vector<resource_type>& u, ::std::size_t index = 0)
    {
        if (!state_)
        {
            backend_ = std::make_shared<backend_t>(u);
            state_ = std::make_shared<state_t>();
            state_->resources_ = get_resources();
            state_->index_ = index;
        }
    }

    template <typename... Args>
    selection_type
    select(Args&&...)
    {
        if (state_)
        {
            if (!state_->resources_.empty())
            {
                return selection_type{*this, state_->resources_[state_->index_]};
            }
            return selection_type{*this};
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
        {
            return backend_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
        }
        else
        {
            throw std::logic_error("submit called before initialization");
        }
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
} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_STATIC_POLICY_IMPL_H*/
