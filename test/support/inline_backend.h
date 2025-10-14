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
class int_inline_backend_t
{
    template <typename Resource>
    class basic_execution_resource_t
    {
        using resource_t = Resource;
        resource_t resource_;

      public:
        basic_execution_resource_t() : resource_(resource_t{}) {}
        basic_execution_resource_t(resource_t r) : resource_(r) {}
        resource_t
        unwrap() const
        {
            return resource_;
        }
        bool
        operator==(const basic_execution_resource_t& e) const
        {
            return resource_ == e.resource_;
        }
        bool
        operator==(const resource_t& e) const
        {
            return resource_ == e;
        }
    };

  public:
    using resource_type = int;
    using wait_type = int;
    using execution_resource_t = basic_execution_resource_t<resource_type>;
    using resource_container_t = std::vector<execution_resource_t>;
    using report_duration = std::chrono::milliseconds;

  private:
    using native_resource_container_t = std::vector<resource_type>;

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
            resources_.push_back(execution_resource_t{i});
    }

    int_inline_backend_t(const native_resource_container_t& u)
    {
        for (const auto& e : u)
            resources_.push_back(execution_resource_t{e});
    }

    template <typename SelectionHandle, typename Function, typename... Args>
    auto
    submit(SelectionHandle s, Function&& f, Args&&... args)
    {
        std::chrono::steady_clock::time_point t0;
        if constexpr (oneapi::dpl::experimental::report_value_v<
                          SelectionHandle, oneapi::dpl::experimental::execution_info::task_time_t, report_duration>)
        {
            t0 = std::chrono::steady_clock::now();
        }
        if constexpr (oneapi::dpl::experimental::report_info_v<
                          SelectionHandle, oneapi::dpl::experimental::execution_info::task_submission_t>)
        {
            s.report(oneapi::dpl::experimental::execution_info::task_submission);
        }
        auto w = std::forward<Function>(f)(oneapi::dpl::experimental::unwrap(s), std::forward<Args>(args)...);

        if constexpr (oneapi::dpl::experimental::report_info_v<
                          SelectionHandle, oneapi::dpl::experimental::execution_info::task_completion_t>)
        {
            oneapi::dpl::experimental::report(s, oneapi::dpl::experimental::execution_info::task_completion);
        }
        if constexpr (oneapi::dpl::experimental::report_value_v<
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

inline int_inline_backend_t int_inline_backend;

} //namespace TestUtils

#endif /* _ONEDPL_INLINE_SCHEDULER_H */
