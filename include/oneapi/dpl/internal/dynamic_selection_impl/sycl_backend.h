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

template <typename _ExtraResource>
class default_backend<sycl::queue, _ExtraResource>
    : public backend_base<sycl::queue, _ExtraResource, default_backend<sycl::queue, _ExtraResource>>
{
  public:
    using resource_type = sycl::queue;
    using wait_type = sycl::event;
    template <typename... Req>
    struct scratch_t
    {
    };

    template <>
    struct scratch_t<execution_info::task_time_t>
    {
        sycl::event my_start_event;
    };

    using execution_resource_t = resource_type;
    using resource_container_t = std::vector<execution_resource_t>;

  private:
    using base_t = backend_base<sycl::queue, _ExtraResource, default_backend<sycl::queue, _ExtraResource>>;
    static inline bool is_profiling_enabled = false;
    using report_clock_type = std::chrono::steady_clock;
    using report_duration = std::chrono::milliseconds;

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
        sycl::event e_;
        std::shared_ptr<Selection> s;

      public:
        async_waiter() = default;
        async_waiter(sycl::event e, std::shared_ptr<Selection> selection) : e_(e), s(selection) {}

        sycl::event
        unwrap()
        {
            return e_;
        }

        void
        wait()
        {
            e_.wait();
        }

        void
        report() const override
        {
            if constexpr (report_value_v<Selection, execution_info::task_time_t, report_duration>)
            {
                if (s != nullptr)
                {
                    const auto time_start =
                        e_.template get_profiling_info<sycl::info::event_profiling::command_start>();
                    const auto time_end = e_.template get_profiling_info<sycl::info::event_profiling::command_end>();
                    s->report(execution_info::task_time, std::chrono::duration_cast<report_duration>(
                                                             std::chrono::nanoseconds(time_end - time_start)));
                }
            }
        }

        bool
        is_complete() const override
        {
            return e_.get_info<sycl::info::event::command_execution_status>() ==
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

      public:
        submission_group(const resource_container_t& v) : resources_(v) {}

        void
        wait()
        {
            for (auto& r : resources_)
            {
                unwrap(r).wait();
            }
        }
    };

  public:
    default_backend(const default_backend& v) = delete;
    default_backend&
    operator=(const default_backend&) = delete;

    default_backend()
    {
        initialize_default_resources();
        sgroup_ptr_ = std::make_unique<submission_group>(this->resources_);
    }

    template <typename NativeUniverseVector, typename ExtraResourceVector = oneapi::dpl::experimental::no_extra_resources>
    default_backend(const NativeUniverseVector& v, const ExtraResourceVector& r = {}) : base_t(v, r)
    {
        bool profiling = true;
        for (auto e : this->get_resources())
        {
            if (!e.template has_property<sycl::property::queue::enable_profiling>())
            {
                profiling = false;
            }
        }
        is_profiling_enabled = profiling;
        sgroup_ptr_ = std::make_unique<submission_group>(this->get_resources());
    }

    /*
    //trait to check for scratch_space 
    template<typename T, typename = void>
    struct has_scratch_space : std::false_type {};

    template<typename T>
    struct has_scratch_space<T, std::void_t<decltype(std::declval<T>().scratch_space)>>::true_type {};
*/
    template <typename SelectionHandle>
    void
    instrument_before_impl(SelectionHandle s)
    {
        if constexpr (report_value_v<SelectionHandle, execution_info::task_time_t, report_duration>)
        {
#ifdef SYCL_EXT_ONEAPI_PROFILING_TAG
            auto q = unwrap(s);
            if (!q.get_device().has(sycl::aspect::ext_oneapi_queue_profiling_tag))
            {
                std::cout << "Cannot time kernels without enabling profiling on queue\n";
                 ///TODO: THROW???
            }
            if constexpr (internal::scratch_space_member<SelectionHandle>::value)
                s.scratch_space.my_start_event =
                    sycl::ext::oneapi::experimental::submit_profiling_tag(q); //starting timestamp
#else
            std::cout << "task_time reporting not supported with this configuration " << std::endl;
#endif
        }
        if constexpr (report_info_v<SelectionHandle, execution_info::task_submission_t>)
            report(s, execution_info::task_submission);
    }

    template <typename SelectionHandle, typename WaitType>
    auto
    instrument_after_impl(SelectionHandle s, WaitType e1)
    {
        constexpr bool report_task_completion = report_info_v<SelectionHandle, execution_info::task_completion_t>;
        constexpr bool report_task_time = report_value_v<SelectionHandle, execution_info::task_time_t, report_duration>;
        if constexpr (report_task_completion || report_task_time)
        {
            async_waiter<SelectionHandle> waiter{e1, std::make_shared<SelectionHandle>(s)};
            if (report_task_time && is_profiling_enabled)
            {
                async_waiter_list.add_waiter(new async_waiter(waiter));
            }

            if (report_task_time && !is_profiling_enabled)
            {
#ifdef SYCL_EXT_ONEAPI_PROFILING_TAG
                if constexpr (internal::scratch_space_member<SelectionHandle>::value)
                {
                    auto q = unwrap(s);
                    if constexpr (oneapi::dpl::experimental::extra_resource_traits<_ExtraResource>::has_reset_v)
                    {
                        s.get_extra_resource().reset();
                    }
                    sycl::event q_end = sycl::ext::oneapi::experimental::submit_profiling_tag(q); //ending timestamp
                                                                                                  //get raw nano number
                        uint64_t time_taken_nanoseconds =
                            q_end.template get_profiling_info<sycl::info::event_profiling::command_start>() -
                            s.scratch_space.my_start_event
                                .template get_profiling_info<sycl::info::event_profiling::command_end>();
                                 //convert nanoseconds to milliseconds
                        report_duration time_taken_milliseconds = std::chrono::duration_cast<report_duration>(
                            std::chrono::nanoseconds(time_taken_nanoseconds));

                    s.report(execution_info::task_time, time_taken_milliseconds);
                }
#endif
            }
            if constexpr (report_task_completion)
                s.report(execution_info::task_completion);
            return waiter;
        }

        return async_waiter{e1, std::make_shared<SelectionHandle>(s)};
    }

    /*
    template <typename SelectionHandle>
    void
    instrument_before_impl(SelectionHandle s)
    {
#ifdef SYCL_EXT_ONEAPI_PROFILING_TAG
	auto q = unwrap(s);
         if (!q.get_device().has(sycl::aspect::ext_oneapi_queue_profiling_tag)) 
            {
                std::cout << "Cannot time kernels without enabling profiling on queue\n";
		///TODO: THROW???
            }
         if constexpr (report_value_v<SelectionHandle, execution_info::task_time_t, report_duration>) 
	 {
	     s.scratch_space.my_start_event = sycl::ext::oneapi::experimental::submit_profiling_tag(q); //starting timestamp
	 }
#endif
        if constexpr (report_info_v<SelectionHandle, execution_info::task_submission_t>)
            report(s, execution_info::task_submission);
    }


   template <typename SelectionHandle, typename WaitType>
    auto
    instrument_after_impl(SelectionHandle s, WaitType e1)
    {
        constexpr bool report_task_completion = report_info_v<SelectionHandle, execution_info::task_completion_t>;
        constexpr bool report_task_time = report_value_v<SelectionHandle, execution_info::task_time_t, report_duration>;

        auto q = unwrap(s);

        if constexpr (report_task_completion || report_task_time)
        {
            async_waiter<SelectionHandle> waiter{e1, std::make_shared<SelectionHandle>(s)};

            if constexpr (report_task_time)
            {
                if (is_profiling_enabled)
                    async_waiter_list.add_waiter(new async_waiter(waiter));
            }

            if ((report_task_time && !is_profiling_enabled) || report_task_completion)
            {
#ifdef SYCL_EXT_ONEAPI_PROFILING_TAG
	    sycl::event q_end = sycl::ext::oneapi::experimental::submit_profiling_tag(q); //ending timestamp

		//get raw nano number
                uint64_t time_taken_nanoseconds =
                q_end.template get_profiling_info<sycl::info::event_profiling::command_start>() -
                s.scratch_space.my_start_event.template get_profiling_info<sycl::info::event_profiling::command_end>();

		//convert nanoseconds to milliseconds
		report_duration time_taken_milliseconds = 
		std::chrono::duration_cast<report_duration>(std::chrono::nanoseconds(time_taken_nanoseconds));

                if constexpr (report_task_time)
                        {
                            if (!is_profiling_enabled)
                                s.report(execution_info::task_time, time_taken_milliseconds);
                        }
                        if constexpr (report_task_completion)
                            s.report(execution_info::task_completion);
#endif
            }
	    
            return waiter;
        }

        return async_waiter{e1, std::make_shared<SelectionHandle>(s)};
    }
*/
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

    void
    initialize_default_resources()
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
