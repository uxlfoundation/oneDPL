// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_DYNAMIC_SELECTION_ONE_POLICY_H
#define _ONEDPL_DYNAMIC_SELECTION_ONE_POLICY_H

#include "oneapi/dpl/dynamic_selection"

enum tracing_enum
{
    t_init = 1 << 0,
    t_select = 1 << 1,
    t_try_submit_function = 1 << 2,
    t_submit_function = 1 << 3,
    t_submit_and_wait_function = 1 << 4,
    t_wait = 1 << 5
};

class no_customizations_policy_base
    : public oneapi::dpl::experimental::policy_base<no_customizations_policy_base, oneapi::dpl::identity,
                                                    oneapi::dpl::experimental::default_backend<int>>
{
    friend class oneapi::dpl::experimental::policy_base<no_customizations_policy_base, oneapi::dpl::identity,
                                                        oneapi::dpl::experimental::default_backend<int>>;
    int& trace_;

  protected:
    //required
    template <typename... Args>
    std::shared_ptr<selection_type>
    try_select_impl(Args&&...)
    {
        trace_ = (trace_ | t_select);
        return std::make_shared<selection_type>(*this);
    }

    void
    initialize_impl()
    {
        std::cout << "init\n";
        trace_ = (trace_ | t_init);
    }

  public:
    using resource_type = int;

    no_customizations_policy_base(int& t) : trace_{t} {}
};

class one_with_all_customizations
{
    int& trace_;

    class one_selection_t
    {
        one_with_all_customizations* p_;

      public:
        explicit one_selection_t(one_with_all_customizations& p) : p_(&p) {}

        // Make it copyable and movable
        one_selection_t(const one_selection_t&) = default;
        one_selection_t(one_selection_t&&) = default;
        one_selection_t&
        operator=(const one_selection_t&) = default;
        one_selection_t&
        operator=(one_selection_t&&) = default;

        auto
        unwrap()
        {
            return 1;
        }
        one_with_all_customizations&
        get_policy()
        {
            return *p_;
        }
    };

    class submission
    {
        int* trace_;

      public:
        submission(int& t) : trace_{&t} {}
        submission(const submission&) = default;
        submission(submission&&) = default;
        submission&
        operator=(const submission&) = default;
        submission&
        operator=(submission&&) = default;
        void
        wait()
        {
            *trace_ = (*trace_ | t_wait);
        }
        int
        unwrap()
        {
            return 1;
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

    using selection_type = one_selection_t;

  public:
    using resource_type = int;

    one_with_all_customizations(int& t) : trace_{t} {}

    auto
    get_resources() const
    {
        return std::vector<int>{1};
    }

    // required
    template <typename... Args>
    std::shared_ptr<selection_type>
    try_select_impl(Args&&...)
    {
        trace_ = (trace_ | t_select);
        return std::make_shared<selection_type>(*this);
    }

    // generic try_submit based on try_select_impl
    template <typename Function, typename... Args>
    auto
    try_submit(Function&&, Args&&... args)
    {
        auto e = try_select_impl(args...);
        if (!e)
        {
            return std::shared_ptr<submission>{};
        }
        trace_ = (trace_ | t_try_submit_function);
        return std::make_shared<submission>(trace_);
    }

    // required
    template <typename Function, typename... Args>
    auto
    submit(Function&& f, Args&&... args)
    {
        auto e = try_submit(f, args...);
        while (!e)
        {
            e = try_submit(f, args...);
            std::this_thread::yield();
        }
        trace_ = (trace_ | t_submit_function);
        return submission{trace_};
    }

    // optional
    template <typename Function, typename... Args>
    void
    submit_and_wait(Function&& f, Args&&... args)
    {
        submit(f, args...).wait();
        trace_ = (trace_ | t_submit_and_wait_function);
        return;
    }

    auto
    get_submission_group()
    {
        return submission_group{};
    }
};

class one_with_only_try_submit
{
    int& trace_;

    class one_selection_t
    {
        one_with_only_try_submit* p_;

      public:
        explicit one_selection_t(one_with_only_try_submit& p) : p_(&p) {}

        // Make it copyable and movable
        one_selection_t(const one_selection_t&) = default;
        one_selection_t(one_selection_t&&) = default;
        one_selection_t&
        operator=(const one_selection_t&) = default;
        one_selection_t&
        operator=(one_selection_t&&) = default;

        auto
        unwrap()
        {
            return 1;
        }
        one_with_only_try_submit&
        get_policy()
        {
            return *p_;
        }
    };

    class submission
    {
        int* trace_;

      public:
        submission(int& t) : trace_{&t} {}
        submission(const submission&) = default;
        submission(submission&&) = default;
        submission&
        operator=(const submission&) = default;
        submission&
        operator=(submission&&) = default;
        void
        wait()
        {
            *trace_ = (*trace_ | t_wait);
        }
        int
        unwrap()
        {
            return 1;
        }
    };

    using selection_type = one_selection_t;

  public:
    using resource_type = int;

    one_with_only_try_submit(int& t) : trace_{t} {}

    // Only try_submit is customized
    template <typename Function, typename... Args>
    auto
    try_submit(Function&&, Args&&... args)
    {
        // built in selection
        trace_ = (trace_ | t_try_submit_function);
        return std::make_shared<submission>(submission{trace_});
    }
};

class one_with_only_submit
{
    int& trace_;

    class one_selection_t
    {
        one_with_only_submit* p_;

      public:
        explicit one_selection_t(one_with_only_submit& p) : p_(&p) {}

        // Make it copyable and movable
        one_selection_t(const one_selection_t&) = default;
        one_selection_t(one_selection_t&&) = default;
        one_selection_t&
        operator=(const one_selection_t&) = default;
        one_selection_t&
        operator=(one_selection_t&&) = default;

        auto
        unwrap()
        {
            return 1;
        }
        one_with_only_submit&
        get_policy()
        {
            return *p_;
        }
    };

    class submission
    {
        int* trace_;

      public:
        submission(int& t) : trace_{&t} {}
        void
        wait()
        {
            *trace_ = (*trace_ | t_wait);
        }
        int
        unwrap()
        {
            return 1;
        }
    };

    using selection_type = one_selection_t;

  public:
    using resource_type = int;

    one_with_only_submit(int& t) : trace_{t} {}

    // Only submit is customized
    template <typename Function, typename... Args>
    auto
    submit(Function&&, Args&&... args)
    {
        //built in selection
        trace_ = (trace_ | t_submit_function);
        return submission{trace_};
    }
};

class one_with_only_submit_and_wait
{
    int& trace_;

    class one_selection_t
    {
        one_with_only_submit_and_wait* p_;

      public:
        explicit one_selection_t(one_with_only_submit_and_wait& p) : p_(&p) {}

        // Make it copyable and movable
        one_selection_t(const one_selection_t&) = default;
        one_selection_t(one_selection_t&&) = default;
        one_selection_t&
        operator=(const one_selection_t&) = default;
        one_selection_t&
        operator=(one_selection_t&&) = default;

        auto
        unwrap()
        {
            return 1;
        }
        one_with_only_submit_and_wait&
        get_policy()
        {
            return *p_;
        }
    };

    using selection_type = one_selection_t;

  public:
    using resource_type = int;

    one_with_only_submit_and_wait(int& t) : trace_{t} {}

    // Only submit_and_wait is customized - no selection methods
    template <typename Function, typename... Args>
    void
    submit_and_wait(Function&&, Args&&...)
    {
        //built in selection, etc.
        trace_ = (trace_ | t_submit_and_wait_function);
        return;
    }
};

class one_with_intermittent_failure
    : public oneapi::dpl::experimental::policy_base<one_with_intermittent_failure, oneapi::dpl::identity,
                                                    oneapi::dpl::experimental::default_backend<int>>
{
    friend class oneapi::dpl::experimental::policy_base<one_with_intermittent_failure, oneapi::dpl::identity,
                                                        oneapi::dpl::experimental::default_backend<int>>;

    struct state_t
    {
        std::atomic<int> attempt_count_{0};
    };
    std::shared_ptr<state_t> state_;

  protected:
    using base_t = oneapi::dpl::experimental::policy_base<one_with_intermittent_failure, oneapi::dpl::identity,
                                                          oneapi::dpl::experimental::default_backend<int>>;

    // Fails every other selection attempt
    template <typename... Args>
    std::shared_ptr<selection_type>
    try_select_impl(Args&&...)
    {
        int count = state_->attempt_count_.fetch_add(1);

        // Fail on even attempts (0, 2, 4, ...), succeed on odd attempts (1, 3, 5, ...)
        if (count % 2 == 0)
        {
            return std::shared_ptr<selection_type>{};
        }
        return std::make_shared<selection_type>(*this);
    }

    void
    initialize_impl()
    {
        if (!state_)
        {
            state_ = std::make_shared<state_t>();
        }
    }

  public:
    using resource_type = int;

    one_with_intermittent_failure() : base_t() { base_t::initialize(); }

    // Reset attempt counter for testing
    void
    reset_attempt_count()
    {
        if (state_)
        {
            state_->attempt_count_ = 0;
        }
    }

    int
    get_attempt_count() const
    {
        return state_ ? state_->attempt_count_.load() : 0;
    }
};

#endif /* _ONEDPL_DYNAMIC_SELECTION_ONE_POLICY_H */
