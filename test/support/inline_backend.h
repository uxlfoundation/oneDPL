// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_INLINE_SCHEDULER_H
#define _ONEDPL_INLINE_SCHEDULER_H

#include "oneapi/dpl/dynamic_selection"

#include <vector>
#include <atomic>
#include <chrono>

namespace TestUtils
{
template <typename ResourceType = int, typename ResourceAdapter = oneapi::dpl::identity>
class int_inline_backend_t
{
    template <typename Resource, typename ResourceAdapter_>
    class basic_execution_resource_t
    {
        using resource_t = Resource;
        using base_resource_t = decltype(ResourceAdapter_{}(std::declval<resource_t>()));
        resource_t resource_;
        ResourceAdapter_ adapter_;
        
      public:
        basic_execution_resource_t() : resource_(resource_t{}) {}
        basic_execution_resource_t(resource_t r, ResourceAdapter_ a) : resource_(r), adapter_(a) {}
        resource_t
        unwrap() const
        {
            return resource_;
        }
        
        friend bool
        operator==(const basic_execution_resource_t& lhs, const basic_execution_resource_t& rhs)
        {
            return lhs.adapter_(lhs.resource_) == rhs.adapter_(rhs.resource_);
        }
        
        // Handle comparison with base resource type (handles both const and non-const)
        template<typename U, std::enable_if_t<std::is_same_v<std::decay_t<U>, std::decay_t<base_resource_t>>, int> = 0>
        friend bool
        operator==(const basic_execution_resource_t& lhs, U&& rhs)
        {
            return lhs.adapter_(lhs.resource_) == std::forward<U>(rhs);
        }
        
        template<typename U, std::enable_if_t<std::is_same_v<std::decay_t<U>, std::decay_t<base_resource_t>>, int> = 0>
        friend bool
        operator==(U&& lhs, const basic_execution_resource_t& rhs)
        {
            return std::forward<U>(lhs) == rhs.adapter_(rhs.resource_);
        }
    };

  public:
    using wait_type = int;
    using execution_resource_t = basic_execution_resource_t<ResourceType, ResourceAdapter>;
    using resource_container_t = std::vector<execution_resource_t>;
    using report_duration = std::chrono::milliseconds;
    using resource_adapter_t = ResourceAdapter;

  private:
    using native_resource_container_t = std::vector<ResourceType>;
    ResourceAdapter adapter_;

    class async_waiter
    {
        wait_type w_;

      public:
        async_waiter(wait_type w) : w_{w} {}
        void
        wait()
        { /* inline scheduler tasks are always complete */
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
    int_inline_backend_t()
    {
        for (int i = 1; i < 4; ++i)
            resources_.push_back(execution_resource_t{i, ResourceAdapter{}});
    }

    int_inline_backend_t(const native_resource_container_t& u, ResourceAdapter a = {}) : adapter_(a)
    {
        for (const auto& e : u)
            resources_.push_back(execution_resource_t{e, a});
    }

    template <typename SelectionHandle, typename Function, typename... Args>
    auto
    submit(SelectionHandle s, Function&& f, Args&&... args)
    {
        std::chrono::steady_clock::time_point t0;
        if constexpr (oneapi::dpl::experimental::internal::report_value_v<
                          SelectionHandle, oneapi::dpl::experimental::execution_info::task_time_t, report_duration>)
        {
            t0 = std::chrono::steady_clock::now();
        }
        if constexpr (oneapi::dpl::experimental::internal::report_info_v<
                          SelectionHandle, oneapi::dpl::experimental::execution_info::task_submission_t>)
        {
            s.report(oneapi::dpl::experimental::execution_info::task_submission);
        }
        auto w = std::forward<Function>(f)(oneapi::dpl::experimental::unwrap(s), std::forward<Args>(args)...);

        if constexpr (oneapi::dpl::experimental::internal::report_info_v<
                          SelectionHandle, oneapi::dpl::experimental::execution_info::task_completion_t>)
        {
            oneapi::dpl::experimental::internal::report(s, oneapi::dpl::experimental::execution_info::task_completion);
        }
        if constexpr (oneapi::dpl::experimental::internal::report_value_v<
                          SelectionHandle, oneapi::dpl::experimental::execution_info::task_time_t, report_duration>)
        {
            report(s, oneapi::dpl::experimental::execution_info::task_time,
                   std::chrono::duration_cast<report_duration>(std::chrono::steady_clock::now() - t0));
        }
        return async_waiter{w};
    }

    auto
    get_submission_group()
    {
        return submission_group{};
    }

    resource_container_t
    get_resources() const noexcept
    {
        return resources_;
    }

  private:
    resource_container_t resources_;
};

} // namespace TestUtils

#endif /* _ONEDPL_INLINE_SCHEDULER_H */
