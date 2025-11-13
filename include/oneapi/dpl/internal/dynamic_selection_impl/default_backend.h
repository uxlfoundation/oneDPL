// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
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

template <typename ResourceType>
class backend_base
{

  public:
    using resource_type = ResourceType;
    using execution_resource_t = resource_type;
    using resource_container_t = std::vector<ResourceType>;
    using report_duration = std::chrono::milliseconds;

    template <typename... Req>
    struct scratch_t : public no_scratch_t<Req...>
    {
        // default backend does not support any reporting requirements and has no scratch space needs
    };

    template <typename... ReportReqs>
    backend_base(ReportReqs...)
    {
        static_assert(sizeof...(ReportReqs) == 0, "Default backend does not support reporting");
    }

    template <typename... ReportReqs>
    backend_base(const std::vector<ResourceType>& u, ReportReqs...)
    {
        static_assert(sizeof...(ReportReqs) == 0, "Default backend does not support reporting");

        for (const auto& e : u)
            resources_.push_back(e);
    }

    auto
    get_submission_group()
    {
        return default_submission_group{resources_};
    }

    auto
    get_resources()
    {
        return resources_;
    }

    template <typename SelectionHandle, typename Function, typename... Args>
    auto
    submit(SelectionHandle s, Function&& f, Args&&... args)
    {
        auto w = std::forward<Function>(f)(oneapi::dpl::experimental::unwrap(s), std::forward<Args>(args)...);
        return default_submission{w};
    }

  protected:
    resource_container_t resources_;

    template <typename WaitType>
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
            if constexpr (internal::has_wait<ResourceType>::value)
            {
                for (auto& r : r_)
                    r.wait();
            }
            else
            {
                throw std::logic_error("wait called on unsupported submission_group.");
            }
        }
    };
};

template <typename BaseResourceType, typename ResourceType, typename ResourceAdapter>
class default_backend_impl : public backend_base<ResourceType>
{
  public:
    using resource_type = ResourceType;
    using my_base = backend_base<ResourceType>;

    template <typename... ReportReqs>
    default_backend_impl(ReportReqs... reqs) : my_base(reqs...)
    {
    }
    template <typename... ReportReqs>
    default_backend_impl(const std::vector<ResourceType>& u, ResourceAdapter adapter_, ReportReqs... reqs)
        : my_base(u, reqs...), adapter(adapter_)
    {
    }

  private:
    ResourceAdapter adapter;
};

template <typename ResourceType, typename ResourceAdapter = oneapi::dpl::identity>
class default_backend
    : public default_backend_impl<std::decay_t<decltype(std::declval<ResourceAdapter>()(std::declval<ResourceType>()))>,
                                  ResourceType, ResourceAdapter>
{
  public:
    using base_t =
        default_backend_impl<std::decay_t<decltype(std::declval<ResourceAdapter>()(std::declval<ResourceType>()))>,
                             ResourceType, ResourceAdapter>;

  public:
    template <typename... ReportReqs>
    default_backend(ReportReqs... reqs) : base_t(reqs...)
    {
    }
    template <typename... ReportReqs>
    default_backend(const std::vector<ResourceType>& r, ResourceAdapter adapt = {}, ReportReqs... reqs)
        : base_t(r, adapt, reqs...)
    {
    }
};

} // namespace experimental

} // namespace dpl

} // namespace oneapi

#endif //_ONEDPL_DEFAULT_BACKEND_H
