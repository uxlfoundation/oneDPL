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
    t_select = 1,
    t_submit_selection = 1 << 1,
    t_submit_function = 1 << 2,
    t_submit_and_wait_selection = 1 << 3,
    t_submit_and_wait_function = 1 << 4,
    t_wait = 1 << 5
};

class one_with_no_customizations
{
    int& trace_;

    class one_selection_t
    {
        one_with_no_customizations& p_;

      public:
        explicit one_selection_t(one_with_no_customizations& p) : p_(p) {}
        auto
        unwrap()
        {
            return 1;
        }
        one_with_no_customizations
        get_policy()
        {
            return p_;
        }
    };

    class submission
    {
        int& trace_;

      public:
        submission(int& t) : trace_{t} {}
        void
        wait()
        {
            trace_ = (trace_ | t_wait);
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

  public:
    using resource_type = int;
    using selection_type = one_selection_t;
    using wait_type = int;

    one_with_no_customizations(int& t) : trace_{t} {}

    auto
    get_resources() const
    {
        return std::vector<int>{1};
    }

    // required
    template <typename... Args>
    selection_type
    select(Args&&... args)
    {
        trace_ = (trace_ | t_select);
        return selection_type{*this};
    }

    // required
    template <typename Function, typename... Args>
    auto
    submit(selection_type, Function&&, Args&&...)
    {
        trace_ = (trace_ | t_submit_selection);
        return submission{trace_};
    }

    auto
    get_submission_group()
    {
        return submission_group{};
    }
};

class one_with_all_customizations
{
    int& trace_;

    class one_selection_t
    {
        one_with_all_customizations& p_;

      public:
        explicit one_selection_t(one_with_all_customizations& p) : p_(p) {}
        auto
        unwrap()
        {
            return 1;
        }
        one_with_all_customizations
        get_policy()
        {
            return p_;
        }
    };

    class submission
    {
        int& trace_;

      public:
        submission(int& t) : trace_{t} {}
        void
        wait()
        {
            trace_ = (trace_ | t_wait);
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

  public:
    using resource_type = int;
    using selection_type = one_selection_t;
    using wait_type = int;

    one_with_all_customizations(int& t) : trace_{t} {}

    auto
    get_resources() const
    {
        return std::vector<int>{1};
    }

    // required
    template <typename... Args>
    selection_type
    select(Args&&...)
    {
        trace_ = (trace_ | t_select);
        return selection_type{*this};
    }

    // required
    template <typename Function, typename... Args>
    auto
    submit(selection_type, Function&&, Args&&...)
    {
        trace_ = (trace_ | t_submit_selection);
        return submission{trace_};
    }

    // optional
    template <typename Function, typename... Args>
    auto
    submit(Function&&, Args&&...)
    {
        trace_ = (trace_ | t_submit_function);
        return submission{trace_};
    }

    // optional
    template <typename Function, typename... Args>
    void
    submit_and_wait(selection_type, Function&&, Args&&...)
    {
        trace_ = (trace_ | t_submit_and_wait_selection);
        return;
    }

    // optional
    template <typename Function, typename... Args>
    void
    submit_and_wait(Function&&, Args&&...)
    {
        trace_ = (trace_ | t_submit_and_wait_function);
        return;
    }

    auto
    get_submission_group()
    {
        return submission_group{};
    }
};

#endif /* _ONEDPL_DYNAMIC_SELECTION_ONE_POLICY_H */
