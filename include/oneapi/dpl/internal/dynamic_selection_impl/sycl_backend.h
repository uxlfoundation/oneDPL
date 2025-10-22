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
  public:
    using resource_type = ResourceType;
    using wait_type = sycl::event;
    template <typename... Req>
    struct scratch_t
    {
    };

    template <>
    struct scratch_t<execution_info::task_time_t>
    {
        sycl::event my_start_event;
        sycl::event my_end_event;
    };

    using execution_resource_t = resource_type;
    using resource_container_t = std::vector<execution_resource_t>;
    using resource_adapter_t = ResourceAdapter;
    using base_resource_t = sycl::queue;

  private:
    using base_t = backend_base<ResourceType, default_backend_impl<sycl::queue, ResourceType, ResourceAdapter>>;
    static inline bool is_profiling_enabled = false;
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
        sycl::event e_start_;
        sycl::event e_end_;
        std::shared_ptr<Selection> s;

      public:
        async_waiter() = default;
        async_waiter(sycl::event start, sycl::event end, std::shared_ptr<Selection> selection) : e_start_(start), e_end_(end), s(selection) {}

        void
        wait()
        {
            e_end_.wait();
        }

        void
        report() const override
        {
            if constexpr (internal::report_value_v<Selection, execution_info::task_time_t, report_duration>)
            {
                if (s != nullptr)
                {
                    const auto time_start =
                        e_start_.template get_profiling_info<sycl::info::event_profiling::command_start>();
                    const auto time_end = e_end_.template get_profiling_info<sycl::info::event_profiling::command_end>();

                    s->report(execution_info::task_time, std::chrono::duration_cast<report_duration>(
                                                             std::chrono::nanoseconds(time_end - time_start)));
                }
            }
            if constexpr (internal::report_info_v<Selection, execution_info::task_completion_t>)
	    {
                if (s != nullptr)
		{
                    s.report(execution_info::task_completion);
		}
	    }
        }

        bool
        is_complete() const override
        {
            return e_end_.get_info<sycl::info::event::command_execution_status>() ==
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

    template <typename T = ResourceAdapter>
    default_backend_impl(std::enable_if_t<std::is_same_v<T, oneapi::dpl::identity>, int> = 0)
    {
        initialize_default_resources();
        sgroup_ptr_ = std::make_unique<submission_group>(this->resources_, adapter);
    }

    template <typename NativeUniverseVector>
    default_backend_impl(const NativeUniverseVector& v, ResourceAdapter adapter_) : base_t(v), adapter(adapter_)
    {
        bool profiling = true;
        for (auto e : this->get_resources())
        {
            if (!adapter(e).template has_property<sycl::property::queue::enable_profiling>())
            {
                profiling = false;
            }
        }
        is_profiling_enabled = profiling;
        sgroup_ptr_ = std::make_unique<submission_group>(this->get_resources(), adapter);
    }

template <typename SelectionHandle, typename Function, typename... Args>
auto
submit_impl(SelectionHandle s, Function&& f, Args&&... args)
    {
        constexpr bool report_task_completion = internal::report_info_v<SelectionHandle, execution_info::task_completion_t>;
        constexpr bool report_task_submission = internal::report_info_v<SelectionHandle, execution_info::task_submission_t>;
        constexpr bool report_task_time = internal::report_value_v<SelectionHandle, execution_info::task_time_t, report_duration>;

	auto resource = unwrap(s);
        auto q = adapter(resource);

        if constexpr (report_task_submission)
            report(s, execution_info::task_submission);

        if constexpr (report_task_completion || report_task_time)
        {
#ifdef SYCL_EXT_ONEAPI_PROFILING_TAG
            if constexpr (internal::scratch_space_member<SelectionHandle>::value)
                s.scratch_space.my_start_event =
                    sycl::ext::oneapi::experimental::submit_profiling_tag(q); //starting tag
            /*auto e1 =*/ f(resource, std::forward<Args>(args)...);
                s.scratch_space.my_end_event =
                    sycl::ext::oneapi::experimental::submit_profiling_tag(q); //ending tag
            async_waiter<SelectionHandle> waiter{s.scratch_space.my_start_event, s.scratch_space.my_end_event, std::make_shared<SelectionHandle>(s)};
            if (report_task_time || report_task_completion)
	    {
                async_waiter_list.add_waiter(new async_waiter(waiter));
	    }

#endif
            return waiter;
        }

        return async_waiter{f(resource, std::forward<Args>(args)...), sycl::event{}, std::make_shared<SelectionHandle>(s)};
    }


    auto
    get_submission_group_impl()
    {
        return *sgroup_ptr_;
    }


    void
    lazy_report()
    {
        if (is_profiling_enabled)
        {
            async_waiter_list.lazy_report();
        }
    }

  private:
    std::unique_ptr<submission_group> sgroup_ptr_;


    // We can only default initialize adapter is oneapi::dpl::identity. If a non base resource is provided with an adapter, then
    // it is the user's responsibilty to initialize the resources
    template <typename T = ResourceAdapter>
    void
    initialize_default_resources(std::enable_if_t<std::is_same_v<T, oneapi::dpl::identity>, int> = 0)
    {
        bool profiling = true;
        auto prop_list = sycl::property_list{};
        auto devices = sycl::device::get_devices();
        for (auto& x : devices)
        {
            if (!x.has(sycl::aspect::queue_profiling))
            {
                profiling = false;
            }
        }
        is_profiling_enabled = profiling;
        if (is_profiling_enabled)
        {
            prop_list = sycl::property_list{sycl::property::queue::enable_profiling()};
        }
        for (auto& x : devices)
        {
            this->resources_.push_back(sycl::queue{x, prop_list});
        }
    }
};

} //namespace experimental
} //namespace dpl
} //namespace oneapi

#endif /*_ONEDPL_SYCL_BACKEND_IMPL_H*/
