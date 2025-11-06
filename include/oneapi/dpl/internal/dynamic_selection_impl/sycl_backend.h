// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_SYCL_BACKEND_IMPL_H
#define _ONEDPL_SYCL_BACKEND_IMPL_H

#include <sycl/sycl.hpp>
#include "oneapi/dpl/internal/dynamic_selection_traits.h"

#include "oneapi/dpl/internal/dynamic_selection_impl/scoring_policy_defs.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/default_backend.h"
#include "oneapi/dpl/functional"

#include <chrono>
#include <ratio>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>

namespace oneapi
{
namespace dpl
{
namespace experimental
{

template <typename ResourceType, typename ResourceAdapter>
class default_backend_impl<sycl::queue, ResourceType, ResourceAdapter>
    : public backend_base<ResourceType, default_backend_impl<sycl::queue, ResourceType, ResourceAdapter>>
{
  private:
    // Base template for scratch storage - empty by default
    template <bool HasStart>
    struct scratch_storage
    {
    };

    // Specialization: needs start event for timing
    template <>
    struct scratch_storage<true>
    {
        sycl::event my_start_event;
    };

  public:
    using resource_type = ResourceType;

    // The return type for user functions must be a sycl::event (to be able to judge completion for lazy reporting of task_completeion)
    using wait_type = sycl::event;

    template <typename... Req>
    struct scratch_t : scratch_storage<execution_info::contains_reporting_req_v<execution_info::task_time_t, Req...>>
    {
    };

    using execution_resource_t = resource_type;
    using resource_container_t = std::vector<execution_resource_t>;
    using resource_adapter_t = ResourceAdapter;
    using base_resource_t = sycl::queue;

  private:
    using base_t = backend_base<ResourceType, default_backend_impl<sycl::queue, ResourceType, ResourceAdapter>>;
    static inline bool is_lazy_reporting_enabled = false;
    using report_clock_type = std::chrono::steady_clock;
    using report_duration = std::chrono::milliseconds;

    ResourceAdapter adapter;
    class async_waiter_base
    {
      public:
        virtual void
        report() const = 0;
        virtual bool
        is_complete() const = 0;
        virtual ~async_waiter_base() = default;
    };

    template <typename Selection>
    class async_waiter : public async_waiter_base
    {
      private:
        // Always need the end event for waiting independent from any reporting
        sycl::event my_end_event;

        std::shared_ptr<Selection> s;

      public:
        async_waiter() = default;
        async_waiter(std::shared_ptr<Selection> selection) : s(selection) {}

        void
        set_end_event(sycl::event e)
        {
            my_end_event = e;
        }

        void
        wait()
        {
            my_end_event.wait();
        }

        void
        report() const override
        {
            if constexpr (report_value_v<Selection, execution_info::task_time_t, report_duration>)
            {
                if (s != nullptr)
                {
                    const auto time_start =
                        s->scratch_space.my_start_event
                            .template get_profiling_info<sycl::info::event_profiling::command_start>();
                    const auto time_end =
                        my_end_event.template get_profiling_info<sycl::info::event_profiling::command_end>();

                    s->report(execution_info::task_time, std::chrono::duration_cast<report_duration>(
                                                             std::chrono::nanoseconds(time_end - time_start)));
                }
            }
            if constexpr (report_info_v<Selection, execution_info::task_completion_t>)
            {
                if (s != nullptr)
                {
                    s->report(execution_info::task_completion);
                }
            }
        }

        bool
        is_complete() const override
        {
            return my_end_event.template get_info<sycl::info::event::command_execution_status>() ==
                   sycl::info::event_command_status::complete;
        }
    };

    struct async_waiter_list_t
    {

        std::mutex m_;
        std::vector<std::unique_ptr<async_waiter_base>> async_waiters;

        void
        add_waiter(async_waiter_base* t)
        {
            std::lock_guard<std::mutex> l(m_);
            async_waiters.push_back(std::unique_ptr<async_waiter_base>(t));
        }

        void
        lazy_report()
        {
            std::lock_guard<std::mutex> l(m_);
            async_waiters.erase(std::remove_if(async_waiters.begin(), async_waiters.end(),
                                               [](std::unique_ptr<async_waiter_base>& async_waiter) {
                                                   if (async_waiter->is_complete())
                                                   {
                                                       async_waiter->report();
                                                       return true;
                                                   }
                                                   return false;
                                               }),
                                async_waiters.end());
        }
    };

    async_waiter_list_t async_waiter_list;

    class submission_group
    {
        resource_container_t resources_;
        ResourceAdapter adapter_;

      public:
        submission_group(const resource_container_t& v, ResourceAdapter adapter) : resources_(v), adapter_(adapter) {}

        void
        wait()
        {
            for (auto& r : resources_)
            {
                adapter_(unwrap(r)).wait();
            }
        }
    };

  public:
    default_backend_impl(const default_backend_impl& v) = delete;
    default_backend_impl&
    operator=(const default_backend_impl&) = delete;

    template <typename... ReportReqs, typename T = ResourceAdapter,
              typename = std::enable_if_t<std::is_same_v<T, oneapi::dpl::identity>>>
    default_backend_impl(ReportReqs... report_reqs) : base_t(), adapter()
    {
        static_assert(
            (execution_info::contains_reporting_req_v<ReportReqs, execution_info::task_submission_t,
                                                      execution_info::task_completion_t, execution_info::task_time_t> &&
             ...),
            "Only reporting for task_submission, task_completion and task_time are supported by the SYCL backend");

        if constexpr (execution_info::contains_reporting_req_v<execution_info::task_time_t, ReportReqs...> ||
                      execution_info::contains_reporting_req_v<execution_info::task_completion_t, ReportReqs...>)
        {
            is_lazy_reporting_enabled = true;
        }
        initialize_default_resources(report_reqs...);
        sgroup_ptr_ = std::make_unique<submission_group>(this->resources_, adapter);
    }

    template <typename NativeUniverseVector, typename... ReportReqs>
    default_backend_impl(const NativeUniverseVector& v, ResourceAdapter adapter_, ReportReqs... report_reqs)
        : base_t(), adapter(adapter_)
    {
        static_assert(
            (execution_info::contains_reporting_req_v<ReportReqs, execution_info::task_submission_t,
                                                      execution_info::task_completion_t, execution_info::task_time_t> &&
             ...),
            "Only reporting for task_submission, task_completion and task_time are supported by the SYCL backend");
        if constexpr (execution_info::contains_reporting_req_v<execution_info::task_time_t, ReportReqs...> ||
                      execution_info::contains_reporting_req_v<execution_info::task_completion_t, ReportReqs...>)
        {
            is_lazy_reporting_enabled = true;
        }
        filter_add_resources(v, report_reqs...);
        sgroup_ptr_ = std::make_unique<submission_group>(this->get_resources(), adapter);
    }

    template <typename SelectionHandle, typename Function, typename... Args>
    auto
    submit_impl(SelectionHandle s, Function&& f, Args&&... args)
    {
        constexpr bool report_task_submission = report_info_v<SelectionHandle, execution_info::task_submission_t>;
        constexpr bool report_task_time = report_value_v<SelectionHandle, execution_info::task_time_t, report_duration>;

        auto resource = unwrap(s);
        auto q = adapter(resource);

        if constexpr (report_task_submission)
            oneapi::dpl::experimental::report(s, execution_info::task_submission);

        sycl::event my_start_event{};
        if constexpr (report_task_time)
        {
#ifdef SYCL_EXT_ONEAPI_PROFILING_TAG
            s.scratch_space.my_start_event = sycl::ext::oneapi::experimental::submit_profiling_tag(q); //starting tag
#else
            static_assert(false, "The sycl version does not support the macro SYCL_EXT_ONEAPI_PROFILING_TAG "
                                 "and profiling is required for this policy submission.");
#endif
        }
        [[maybe_unused]] sycl::event workflow_return = f(resource, std::forward<Args>(args)...);
        async_waiter<SelectionHandle> waiter{std::make_shared<SelectionHandle>(s)};
        async_waiter_list.add_waiter(new async_waiter(waiter));
        if constexpr (report_task_time)
        {
#ifdef SYCL_EXT_ONEAPI_PROFILING_TAG
            // if we are using timing, we use the profiling event as the end event
            waiter.set_end_event(sycl::ext::oneapi::experimental::submit_profiling_tag(q)); //ending tag
#endif
        }
        else
        {
            // if not using profiling, use the normal event
            waiter.set_end_event(workflow_return);
        }

        return waiter;
    }

    auto
    get_submission_group_impl()
    {
        return *sgroup_ptr_;
    }

    void
    lazy_report()
    {
        if (is_lazy_reporting_enabled)
        {
            async_waiter_list.lazy_report();
        }
    }

  private:
    std::unique_ptr<submission_group> sgroup_ptr_;

    template <typename NativeUniverseVector, typename... ReportReqs>
    void
    filter_add_resources(const NativeUniverseVector& v, ReportReqs...)
    {
        if constexpr (execution_info::contains_reporting_req_v<execution_info::task_time_t, ReportReqs...>)
        {
#ifdef SYCL_EXT_ONEAPI_PROFILING_TAG
            for (auto& x : v)
            {
                if (adapter(x).get_device().has(sycl::aspect::ext_oneapi_queue_profiling_tag))
                    this->resources_.push_back(x);
            }

            if (this->resources_.empty())
            {
                throw std::runtime_error(
                    "Either the sycl version does not support the macro SYCL_EXT_ONEAPI_PROFILING_TAG "
                    "or the devices do not have the sycl::aspect ext_oneapi_queue_profiling_tag, "
                    "both of these are required to time kernels.");
            }
#else
            static_assert(
                false, "SYCL_EXT_ONEAPI_PROFILING_TAG is not defined, but it is required to time kernels. Please use "
                       "a SYCL version that supports this tag");
#endif
        }
        else // other reporting requirements beside task_time
        {
            for (auto& x : v)
            {
                this->resources_.push_back(x);
            }
        }
    }

    // We can only default initialize adapter is oneapi::dpl::identity. If a non base resource is provided with an adapter, then
    // it is the user's responsibility to initialize the resources
    template <typename... ReportReqs, typename T = ResourceAdapter,
              typename = std::enable_if_t<std::is_same_v<T, oneapi::dpl::identity>>>
    void
    initialize_default_resources(ReportReqs... report_reqs)
    {
        auto devices = sycl::device::get_devices();
        std::vector<sycl::queue> v;

        auto prop_list = sycl::property_list{};
        if constexpr (execution_info::contains_reporting_req_v<execution_info::task_time_t, ReportReqs...>)
        {
            prop_list = sycl::property_list{sycl::property::queue::enable_profiling()};
        }

        for (auto& x : devices)
        {
            if constexpr (execution_info::contains_reporting_req_v<execution_info::task_time_t, ReportReqs...>)
            {
                if (adapter(x).template has_property<sycl::property::queue::enable_profiling>())
                {
                    v.push_back(sycl::queue(x, prop_list));
                }
            }
            else
            {
                v.push_back(sycl::queue(x, prop_list));
            }
        }
        filter_add_resources(v, report_reqs...);
    }
};

using sycl_backend = default_backend<sycl::queue, oneapi::dpl::identity>;

} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_SYCL_BACKEND_IMPL_H*/
