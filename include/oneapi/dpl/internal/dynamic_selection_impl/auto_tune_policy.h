// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_AUTO_TUNE_POLICY_H
#define _ONEDPL_AUTO_TUNE_POLICY_H

#include <stdexcept>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <chrono>
#include <ratio>
#include <limits>
#include <vector>
#include <type_traits>
#include <tuple>
#include <unordered_map>
#include "oneapi/dpl/internal/dynamic_selection_traits.h"
#include "oneapi/dpl/internal/dynamic_selection_impl/backend_traits.h"
#if _DS_BACKEND_SYCL != 0
#    include "oneapi/dpl/internal/dynamic_selection_impl/sycl_backend.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace experimental
{

#if _DS_BACKEND_SYCL != 0
template <typename Backend = sycl_backend, typename... KeyArgs>
#else
template <typename Backend, typename... KeyArgs>
#endif
class auto_tune_policy
{

    using backend_t = Backend;
    using execution_resource_t = typename backend_t::execution_resource_t;
    using wrapped_resource_t = execution_resource_t;
    using size_type = typename std::vector<typename Backend::resource_type>::size_type;
    using timing_t = uint64_t;

    using report_clock_type = std::chrono::steady_clock;
    using report_duration = std::chrono::milliseconds;

    static constexpr timing_t never_resample = 0;
    static constexpr size_type use_best_resource = ~size_type(0);

    struct resource_with_index_t
    {
        wrapped_resource_t r_;
        size_type index_ = 0;
    };

    struct time_data_t
    {
        uint64_t num_timings_ = 0;
        timing_t value_ = 0;
    };

    struct tuner_t
    {
        std::mutex m_;

        report_clock_type::time_point t0_;

        timing_t best_timing_ = std::numeric_limits<timing_t>::max();
        resource_with_index_t best_resource_;

        const size_type max_resource_to_profile_;
        uint64_t next_resource_to_profile_ = 0; // as index in resources

        using time_by_index_t = std::unordered_map<size_type, time_data_t>;
        time_by_index_t time_by_index_;

        timing_t resample_time_ = 0;

        tuner_t(resource_with_index_t br, size_type resources_size, timing_t rt)
            : t0_(report_clock_type::now()), best_resource_(br), max_resource_to_profile_(resources_size),
              resample_time_(rt)
        {
        }

        size_type
        get_resource_to_profile()
        {
            std::lock_guard<std::mutex> l(m_);
            if (next_resource_to_profile_ < 2 * max_resource_to_profile_)
            {
                // do everything twice
                return next_resource_to_profile_++ % max_resource_to_profile_;
            }
            else if (resample_time_ == never_resample)
            {
                return use_best_resource;
            }
            else
            {
                const auto now = report_clock_type::now();
                const auto ms = std::chrono::duration_cast<report_duration>(now - t0_).count();
                if (ms < resample_time_)
                {
                    return use_best_resource;
                }
                else
                {
                    t0_ = now;
                    next_resource_to_profile_ = 0;
                    return next_resource_to_profile_++;
                }
            }
        }

        // called to add new profile info
        void
        add_new_timing(resource_with_index_t r, timing_t t)
        {
            auto index = r.index_;
            timing_t new_value = t;

            std::lock_guard<std::mutex> l(m_);

            // ignore the 1st timing to cover for JIT compilation
            auto emplace_res = time_by_index_.try_emplace(index, time_data_t{0, std::numeric_limits<timing_t>::max()});

            // emplace_res is std::pair<time_by_index_t::iterator, bool> where
            //  - emplace_res.first iterate inserted or existing element;
            //  - emplace_res.second is true if new element inserted, false if element with such key already existed.
            if (!emplace_res.second)
            {
                // get reference to time_data_t from already existing element
                auto& td = emplace_res.first->second;
                auto n = td.num_timings_;
                new_value = (n * td.value_ + t) / (n + 1);
                td.num_timings_ = n + 1;
                td.value_ = new_value;
            }
            if (new_value < best_timing_)
            {
                best_timing_ = new_value;
                best_resource_ = r;
            }
        }
    };

    class auto_tune_selection_type
    {
        using policy_t = auto_tune_policy<Backend, KeyArgs...>;
        policy_t policy_;
        resource_with_index_t resource_;
        std::shared_ptr<tuner_t> tuner_;

      public:
        auto_tune_selection_type(const policy_t& p, resource_with_index_t r, std::shared_ptr<tuner_t> t)
            : policy_(p), resource_(r), tuner_(::std::move(t))
        {
        }

        auto
        unwrap()
        {
            return ::oneapi::dpl::experimental::unwrap(resource_.r_);
        }

        policy_t
        get_policy()
        {
            return policy_;
        };

        void
        report(const execution_info::task_time_t&, report_duration v) const
        {
            tuner_->add_new_timing(resource_, v.count());
        }
    };

  public:
    // Needed by Policy Traits
    using resource_type = decltype(unwrap(std::declval<wrapped_resource_t>()));
    using wait_type = typename Backend::wait_type;
    using selection_type = auto_tune_selection_type;

    auto_tune_policy(deferred_initialization_t) {}

    auto_tune_policy(timing_t resample_time = never_resample) { initialize(resample_time); }

    auto_tune_policy(const std::vector<resource_type>& u, timing_t resample_time = never_resample)
    {
        initialize(u, resample_time);
    }

    void
    initialize(timing_t resample_time = never_resample)
    {
        if (!state_)
        {
            state_ = std::make_shared<state_t>();
            backend_ = std::make_shared<Backend>();
            initialize_impl(resample_time);
        }
    }

    void
    initialize(const std::vector<resource_type>& u, timing_t resample_time = never_resample)
    {
        if (!state_)
        {
            state_ = std::make_shared<state_t>();
            backend_ = std::make_shared<Backend>(u);
            initialize_impl(resample_time);
        }
    }

    template <typename Function, typename... Args>
    selection_type
    select(Function&& f, Args&&... args)
    {
        static_assert(sizeof...(KeyArgs) == sizeof...(Args));
        if constexpr (backend_traits::lazy_report_v<Backend>)
        {
            backend_->lazy_report();
        }
        if (state_)
        {
            std::lock_guard<std::mutex> l(state_->m_);
            auto k = make_task_key(std::forward<Function>(f), std::forward<Args>(args)...);
            auto t = state_->tuner_by_key_[k];
            auto index = t->get_resource_to_profile();
            if (index == use_best_resource)
            {
                return selection_type{*this, t->best_resource_, t};
            }
            else
            {
                auto r = state_->resources_with_index_[index];
                return selection_type{*this, r, t};
            }
        }
        else
        {
            throw std::logic_error("select called before initialization");
        }
    }

    template <typename Function, typename... Args>
    auto
    submit(selection_type e, Function&& f, Args&&... args)
    {
        static_assert(sizeof...(KeyArgs) == sizeof...(Args));
        if (backend_)
        {
            return backend_->submit(e, std::forward<Function>(f), std::forward<Args>(args)...);
        }
        else
        {
            throw std::logic_error("submit called before initialization");
        }
    }

    auto
    get_resources()
    {
        if (backend_)
        {
            return backend_->get_resources();
        }
        else
        {
            throw std::logic_error("get_resources called before initialization");
        }
    }

    auto
    get_submission_group()
    {
        if (backend_)
        {
            return backend_->get_submission_group();
        }
        else
        {
            throw std::logic_error("get_submission_group called before initialization");
        }
    }

  private:
    //
    // types
    //

    using task_key_t = std::tuple<void*, KeyArgs...>;
    using tuner_by_key_t = std::map<task_key_t, std::shared_ptr<tuner_t>>;

    //
    // member variables
    //

    timing_t resample_time_ = 0;

    struct state_t
    {
        std::mutex m_;
        std::vector<resource_with_index_t> resources_with_index_;
        tuner_by_key_t tuner_by_key_;
    };

    std::shared_ptr<Backend> backend_;
    std::shared_ptr<state_t> state_;

    //
    // private member functions
    //

    void
    initialize_impl(timing_t resample_time = never_resample)
    {
        resample_time_ = resample_time;
        auto u = get_resources();
        for (size_type i = 0; i < u.size(); ++i)
        {
            state_->resources_with_index_.push_back(resource_with_index_t{u[i], i});
        }
    }

    template <typename Function, typename... Args>
    task_key_t
    make_task_key(Function&& f, Args&&... args)
    {
        // called under lock
        task_key_t k = std::make_tuple(static_cast<void*>(&f), std::forward<Args>(args)...);
        if (state_->tuner_by_key_.count(k) == 0)
        {
            state_->tuner_by_key_[k] = std::make_shared<tuner_t>(state_->resources_with_index_[0],
                                                                 state_->resources_with_index_.size(), resample_time_);
        }
        return k;
    }
};

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_AUTO_TUNE_POLICY_H
