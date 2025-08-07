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
#include "oneapi/dpl/internal/dynamic_selection_impl/backend_traits.h"
#include "oneapi/dpl/functional"

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
    using execution_resource_t = resource_type;
    using resource_container_t = std::vector<ResourceType>;
    using report_duration = std::chrono::milliseconds;

    template <bool needs_scratch = false, typename...Req>
    struct scratch_t_check : public no_scratch_t<Req...>
    {
    };

    // maximal scratch space to handle all requirements
    template <typename ...Req>
    struct scratch_t_check<true, Req...>
    {
    };

    // check for task_time_t and define scratch space
    template<typename ...Req>
    struct scratch_t : public scratch_t_check<any_of_v<execution_info::task_time_t, Req...>, Req...>
    {
    };

    backend_base()
    {
    }

    backend_base(const std::vector<ResourceType>& u)
    {
        for (const auto& e : u)
            resources_.push_back(e);
    }

    auto
    get_submission_group()
    {
        return static_cast<Backend*>(this)->get_submission_group_impl();
    }

    auto get_resources() {
        return static_cast<Backend*>(this)->get_resources_impl();
    }

    template <typename SelectionHandle, typename Function, typename... Args>
    auto
    submit(SelectionHandle s, Function&& f, Args&&... args)
    {
        return static_cast<Backend*>(this)->submit_impl(s, f, args...);
    }

  protected:
  
    resource_container_t resources_;

    resource_container_t
    get_resources_impl() const noexcept
    {
        return resources_;
    }

    auto
    get_submission_group_impl()
    {
        return default_submission_group{resources_};
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
        return default_submission{w};
    }

    template <typename SelectionHandle, typename Function, typename... Args>
    auto
    submit_impl(SelectionHandle s, Function&& f, Args&&... args)
    {
      static_cast<Backend*>(this)->instrument_before_impl(s);
      auto w = std::forward<Function>(f)(oneapi::dpl::experimental::unwrap(s), std::forward<Args>(args)...);
	    return static_cast<Backend*>(this)->instrument_after_impl(s, w);
    }

    template<typename WaitType>
     class default_submission
    {
      WaitType w_;

      public:
        default_submission(WaitType w) : w_{w} {}

        void
        wait()
        {
          if constexpr (internal::has_wait<WaitType>::value)
            w_.wait();
        }
        
        WaitType
        unwrap()
        {
            return w_;
        }
    };

    class default_submission_group
    {
      std::vector<ResourceType>& r_;
    
      public:
      default_submission_group(std::vector<ResourceType>& r) : r_(r) {}

        void
        wait()
        {
          if constexpr (internal::has_wait<ResourceType>::value) {
              for (auto& r : r_) 
                r.wait();
          } else {
            throw std::logic_error("wait called on unsupported submission_group.");
          }
        }
    };

};


template< typename BaseResourceType, typename ResourceType, typename ResourceAdapter >
class default_backend_impl : public backend_base<ResourceType, default_backend_impl<BaseResourceType, ResourceType, ResourceAdapter>> {
public:
    using resource_type = ResourceType;
    using my_base = backend_base<ResourceType, default_backend_impl<BaseResourceType, ResourceType, ResourceAdapter>>;

    default_backend_impl() : my_base() {}
    default_backend_impl(const std::vector<ResourceType>& u, ResourceAdapter adapter_) : my_base(u), adapter(adapter_) {}
  private:
    ResourceAdapter adapter;
};

template <typename ResourceType, typename ResourceAdapter = oneapi::dpl::identity>
class default_backend : public default_backend_impl<std::decay_t<decltype(std::declval<ResourceAdapter>()(std::declval<ResourceType>()))>, ResourceType, ResourceAdapter>
{
  public:
    using base_t = default_backend_impl<std::decay_t<decltype(std::declval<ResourceAdapter>()(std::declval<ResourceType>()))>, ResourceType, ResourceAdapter>;
    using wait_type = typename base_t::wait_type;
  public:
    default_backend()
    {
    }
    default_backend(const std::vector<ResourceType>& r, ResourceAdapter adapt = {}) : base_t(r, adapt)
    {
    }
};


} // namespace experimental

} // namespace dpl

} // namespace oneapi

#endif //_ONEDPL_DEFAULT_BACKEND_H
