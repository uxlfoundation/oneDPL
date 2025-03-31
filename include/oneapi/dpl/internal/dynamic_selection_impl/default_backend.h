// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_DEFAULT_BACKEND_H
#define _ONEDPL_DEFAULT_BACKEND_H

#include <atomic>
#include <type_traits>
#include <vector>
#include <memory>
#include <stdexcept>
#include <limits>
#include <utility>
#include <chrono>
#include "oneapi/dpl/internal/dynamic_selection_traits.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"

namespace oneapi
{
namespace dpl
{
namespace experimental
{

template<typename ResourceType, typename Backend>
class backend_base
{
  public:
    using resource_type = ResourceType;
    using wait_type = resource_type; //TODO: Remove?
    using execution_resource_t = resource_type;
    using resource_container_t = std::vector<resource_type>;
    using report_duration = std::chrono::milliseconds;

  private:
    class async_waiter
    {
        wait_type w_;

      public:
        async_waiter(wait_type w) : w_{w} {}
        void
        wait()
        {
        }
        wait_type
        unwrap()
        {
            return w_;
        }
    };

    class submission_group
    {
      public:
        void
        wait()
        {
            return;
        }
    };

  public:
    backend_base()
    {
    }

    backend_base(const resource_container_t& u)
    {
        for (const auto& e : u)
            resources_.push_back(e);
    }
  protected:
    resource_container_t
    get_resources_impl() const noexcept
    {
        return resources_;
    }

    template <typename SelectionHandle>
    void
    instrument_before_impl(SelectionHandle /*s*/)
    {
    }

    template <typename SelectionHandle, typename WaitType>
    auto
    instrument_after_impl(SelectionHandle /*s*/, WaitType w)
    {
        return async_waiter{w};
    }

    template <typename SelectionHandle, typename Function, typename... Args>
    auto
    submit_impl(SelectionHandle s, Function&& f, Args&&... args)
    {
	instrument_before(s);
        auto w = std::forward<Function>(f)(oneapi::dpl::experimental::unwrap(s), std::forward<Args>(args)...);
	return instrument_after(s, w);
    }

  public:
    auto
    get_submission_group()
    {
        return submission_group{};
    }

  
    auto get_resources() {
        return static_cast<Backend*>(this)->get_resources_impl();
    }

    template <typename SelectionHandle>
    void
    instrument_before(SelectionHandle s)
    {
        return static_cast<Backend *>(this)->instrument_before_impl(s);
    }

    template <typename SelectionHandle, typename WaitType>
    auto
    instrument_after(SelectionHandle s, WaitType w)
    {
        return static_cast<Backend*>(this)->instrument_after_impl(s, w);
    }

    template <typename SelectionHandle, typename Function, typename... Args>
    auto
    submit(SelectionHandle s, Function&& f, Args&&... args)
    {
        return static_cast<Backend*>(this)->submit_impl(s, f, args...);
    }

  private:
    resource_container_t resources_;
};

template< typename ResourceType >
class default_backend : public backend_base<ResourceType, default_backend<ResourceType>> {
public:
    using resource_type = ResourceType;
};

} // namespace experimental

} // namespace dpl

} // namespace oneapi

#endif //_ONEDPL_DEFAULT_BACKEND_H
