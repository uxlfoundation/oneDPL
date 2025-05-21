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
#include "oneapi/dpl/internal/dynamic_selection_traits.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"

namespace oneapi 
{
namespace dpl 
{
namespace experimental 
{

template <typename Policy, typename ResourceType, typename Backend>
class policy_base 
{
  protected:
    using backend_t = Backend;
    using resource_container_t = typename backend_t::resource_container_t;
    using resource_container_size_t = typename resource_container_t::size_type;
    using execution_resource_t = typename backend_t::execution_resource_t;
    using wrapped_resource_t = execution_resource_t;

  public:
    using resource_type = decltype(unwrap(std::declval<wrapped_resource_t>()));
    using wait_type = typename backend_t::wait_type;
    using selection_type = basic_selection_handle_t<Policy, execution_resource_t>;

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
        if (!backend_) backend_ = std::make_shared<backend_t>();
        static_cast<Policy*>(this)->initialize_impl();
    }

    void 
    initialize(const std::vector<resource_type>& u) 
    {
        if (!backend_) backend_ = std::make_shared<backend_t>(u);
        static_cast<Policy*>(this)->initialize_impl();
    }

    template <typename... Args>
    selection_type 
    select(Args&&... args) 
    {
        return static_cast<Policy*>(this)->select_impl(std::forward<Args>(args)...);
    }

    template <typename Function, typename... Args>
    auto 
    submit(selection_type e, Function&& f, Args&&... args) 
    {
        if (backend_) 
	{
            return backend_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
        }
        throw std::logic_error("submit called before initialization");
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

