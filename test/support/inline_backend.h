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
#include <iostream>

namespace TestUtils
{
template <typename ResourceType = int, typename ResourceAdapter = oneapi::dpl::identity>
class int_inline_backend_t
{

  public:
    using resource_type = ResourceType;
    using wait_type = int;
    using report_duration = std::chrono::milliseconds;
    using resource_adapter_t = ResourceAdapter;

  private:
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
    template <typename... ReportReqs>
    int_inline_backend_t(ReportReqs...)
    {
        static_assert(
            (oneapi::dpl::experimental::execution_info::contains_reporting_req_v<
                 ReportReqs, oneapi::dpl::experimental::execution_info::task_submission_t,
                 oneapi::dpl::experimental::execution_info::task_completion_t,
                 oneapi::dpl::experimental::execution_info::task_time_t> &&
             ...),
            "Only reporting for task_submission, task_completion and task_time are supported by the inline backend");

        for (int i = 1; i < 4; ++i)
            resources_.push_back(i);
    }

    template <typename... ReportReqs>
    int_inline_backend_t(const std::vector<resource_type>& u, ResourceAdapter a = {}, ReportReqs...) : adapter_(a)
    {
        static_assert(
            (oneapi::dpl::experimental::execution_info::contains_reporting_req_v<
                 ReportReqs, oneapi::dpl::experimental::execution_info::task_submission_t,
                 oneapi::dpl::experimental::execution_info::task_completion_t,
                 oneapi::dpl::experimental::execution_info::task_time_t> &&
             ...),
            "Only reporting for task_submission, task_completion and task_time are supported by the inline backend");
        for (const auto& e : u)
            resources_.push_back(e);
    }

    template <typename SelectionHandle, typename Function, typename... Args>
    auto
    submit(SelectionHandle s, Function&& f, Args&&... args)
    {
        std::cout << "inline_backend submit called\n";
        std::chrono::steady_clock::time_point t0;
        if constexpr (oneapi::dpl::experimental::report_value_v<
                          SelectionHandle, oneapi::dpl::experimental::execution_info::task_time_t, report_duration>)
        {
            std::cout << "inline_backend: recording start time for task_time_t\n";
            t0 = std::chrono::steady_clock::now();
        }
        if constexpr (oneapi::dpl::experimental::report_info_v<
                          SelectionHandle, oneapi::dpl::experimental::execution_info::task_submission_t>)
        {
            std::cout << "inline_backend: reporting task_submission\n";
            s.report(oneapi::dpl::experimental::execution_info::task_submission);
        }
        std::cout << "inline_backend: calling unwrap and executing function\n";
        auto w = std::forward<Function>(f)(oneapi::dpl::experimental::unwrap(s), std::forward<Args>(args)...);

        if constexpr (oneapi::dpl::experimental::report_info_v<
                          SelectionHandle, oneapi::dpl::experimental::execution_info::task_completion_t>)
        {
            oneapi::dpl::experimental::report(s, oneapi::dpl::experimental::execution_info::task_completion);
        }
        if constexpr (oneapi::dpl::experimental::report_value_v<
                          SelectionHandle, oneapi::dpl::experimental::execution_info::task_time_t, report_duration>)
        {
            oneapi::dpl::experimental::report(
                s, oneapi::dpl::experimental::execution_info::task_time,
                std::chrono::duration_cast<report_duration>(std::chrono::steady_clock::now() - t0));
        }
        return async_waiter{w};
    }

    auto
    get_submission_group()
    {
        return submission_group{};
    }

    std::vector<resource_type>
    get_resources() const noexcept
    {
        return resources_;
    }

  private:
    std::vector<resource_type> resources_;
};

} // namespace TestUtils

#endif /* _ONEDPL_INLINE_SCHEDULER_H */
