// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_INTERNAL_DYNAMIC_SELECTION_TRAITS_H
#define _ONEDPL_INTERNAL_DYNAMIC_SELECTION_TRAITS_H

#include <utility>
#include <cstdint>
#include <type_traits>
#include "oneapi/dpl/internal/dynamic_selection_impl/policy_traits.h"

namespace oneapi
{
namespace dpl
{
namespace experimental
{
namespace internal
{
template <typename T>
auto
has_unwrap_impl(...) -> std::false_type;

template <typename T>
auto
has_unwrap_impl(int) -> decltype(std::declval<T>().unwrap(), std::true_type{});

template <typename T>
struct has_unwrap : decltype(has_unwrap_impl<T>(0))
{
};

template <typename T>
auto
has_get_policy_impl(...) -> std::false_type;

template <typename T>
auto
has_get_policy_impl(int) -> decltype(std::declval<T>().get_policy(), std::true_type{});

template <typename T>
struct has_get_policy : decltype(has_get_policy_impl<T>(0))
{
};

template <typename DSPolicy, typename Function, typename... Args>
auto
has_submit_impl(...) -> std::false_type;

template <typename DSPolicy, typename Function, typename... Args>
auto
has_submit_impl(int)
    -> decltype(std::declval<DSPolicy>().submit(std::declval<Function>(), std::declval<Args>()...), std::true_type{});

template <typename DSPolicy, typename Function, typename... Args>
struct has_submit : decltype(has_submit_impl<DSPolicy, Function, Args...>(0))
{
};

template <typename DSPolicy, typename Function, typename... Args>
auto
has_submit_and_wait_impl(...) -> std::false_type;

template <typename DSPolicy, typename Function, typename... Args>
auto
has_submit_and_wait_impl(int)
    -> decltype(std::declval<DSPolicy>().submit_and_wait(std::declval<Function>(), std::declval<Args>()...),
                std::true_type{});

template <typename DSPolicy, typename Function, typename... Args>
struct has_submit_and_wait : decltype(has_submit_and_wait_impl<DSPolicy, Function, Args...>(0))
{
};

template <typename DSPolicy, typename SelectionHandle, typename Function, typename... Args>
auto
has_submit_and_wait_handle_impl(...) -> std::false_type;

template <typename DSPolicy, typename SelectionHandle, typename Function, typename... Args>
auto
has_submit_and_wait_handle_impl(int)
    -> decltype(std::declval<DSPolicy>().submit_and_wait(std::declval<SelectionHandle>(), std::declval<Function>(),
                                                         std::declval<Args>()...),
                std::true_type{});

template <typename DSPolicy, typename SelectionHandle, typename Function, typename... Args>
struct has_submit_and_wait_handle
    : decltype(has_submit_and_wait_handle_impl<DSPolicy, SelectionHandle, Function, Args...>(0))
{
};

template <typename S, typename Info>
auto
has_report_impl(...) -> std::false_type;

template <typename S, typename Info>
auto
has_report_impl(int) -> decltype(std::declval<S>().report(std::declval<Info>()), std::true_type{});

template <typename S, typename Info>
struct has_report : decltype(has_report_impl<S, Info>(0))
{
};

template <typename S, typename Info, typename ValueType>
auto
has_report_value_impl(...) -> std::false_type;

template <typename S, typename Info, typename ValueType>
auto
has_report_value_impl(int)
    -> decltype(std::declval<S>().report(std::declval<Info>(), std::declval<ValueType>()), std::true_type{});

template <typename S, typename Info, typename ValueType>
struct has_report_value : decltype(has_report_value_impl<S, Info, ValueType>(0))
{
};

} //namespace internal

struct deferred_initialization_t
{
};
inline constexpr deferred_initialization_t deferred_initialization;

// required interfaces
template <typename DSPolicy>
auto
get_resources(DSPolicy&& dp)
{
    return std::forward<DSPolicy>(dp).get_resources();
}

template <typename WaitObject>
auto
wait(WaitObject&& w)
{
    return std::forward<WaitObject>(w).wait();
}

template <typename DSPolicy>
auto
get_submission_group(DSPolicy&& dp)
{
    return std::forward<DSPolicy>(dp).get_submission_group();
}

template <typename DSPolicy, typename... Args>
typename policy_traits<DSPolicy>::selection_type
select(DSPolicy&& dp, Args&&... args)
{
    return std::forward<DSPolicy>(dp).select(std::forward<Args>(args)...);
}

// optional interfaces

template <typename T>
auto
unwrap(T&& v)
{
    if constexpr (internal::has_unwrap<T>::value)
    {
        return std::forward<T>(v).unwrap();
    }
    else
    {
        return v;
    }
}

template <typename T, typename Function, typename... Args>
auto
submit(T&& t, Function&& f, Args&&... args)
{
    if constexpr (internal::has_get_policy<T>::value)
    {
        // t is a selection
        return t.get_policy().submit(std::forward<T>(t), std::forward<Function>(f), std::forward<Args>(args)...);
    }
    else if constexpr (internal::has_submit<T, Function, Args...>::value)
    {
        // t is a policy and policy has optional submit(f, args...)
        return std::forward<T>(t).submit(std::forward<Function>(f), std::forward<Args>(args)...);
    }
    else
    {
        // t is a policy and policy does not have optional submit(f, args...)
        return std::forward<T>(t).submit(t.select(f, args...), std::forward<Function>(f), std::forward<Args>(args)...);
    }
}

template <typename T, typename Function, typename... Args>
auto
submit_and_wait(T&& t, Function&& f, Args&&... args)
{
    if constexpr (internal::has_get_policy<T>::value)
    {
        // t is a selection
        if constexpr (internal::has_submit_and_wait_handle<decltype(std::declval<T>().get_policy()), T, Function,
                                                           Args...>::value)
        {
            // policy has optional submit_and_wait(selection, f, args...)
            return t.get_policy().submit_and_wait(std::forward<T>(t), std::forward<Function>(f),
                                                  std::forward<Args>(args)...);
        }
        else
        {
            // policy does not have optional submit_and_wait for a selection
            return wait(submit(std::forward<T>(t), std::forward<Function>(f), std::forward<Args>(args)...));
        }
    }
    else
    {
        // t is a policy
        if constexpr (internal::has_submit_and_wait<T, Function, Args...>::value)
        {
            // has the optional submit_and_wait(f, args...)
            return std::forward<T>(t).submit_and_wait(std::forward<Function>(f), std::forward<Args>(args)...);
        }
        else if constexpr (internal::has_submit_and_wait_handle<T, typename std::decay_t<T>::selection_type, Function,
                                                                Args...>::value)
        {
            // has the optional submit_and_wait for a selection, so select and call
            return std::forward<T>(t).submit_and_wait(t.select(f, args...), std::forward<Function>(f),
                                                      std::forward<Args>(args)...);
        }
        else
        {
            // does not have the optional submit_and_wait(f, args...) or (s, f, args...)
            return wait(submit(std::forward<T>(t), std::forward<Function>(f), std::forward<Args>(args)...));
        }
    }
}

// support for execution info

namespace execution_info
{
struct task_time_t
{
    static constexpr bool is_execution_info_v = true;
    using value_type = uint64_t;
};
inline constexpr task_time_t task_time;

struct task_completion_t
{
    static constexpr bool is_execution_info_v = true;
    using value_type = void;
};
inline constexpr task_completion_t task_completion;

struct task_submission_t
{
    static constexpr bool is_execution_info_v = true;
    using value_type = void;
};
inline constexpr task_submission_t task_submission;
} // namespace execution_info

template <typename S, typename Info>
void
report(S&& s, const Info& i)
{
    if constexpr (internal::has_report<S, Info>::value)
    {
        std::forward<S>(s).report(i);
    }
}

template <typename S, typename Info, typename Value>
void
report(S&& s, const Info& i, const Value& v)
{
    if constexpr (internal::has_report_value<S, Info, Value>::value)
    {
        std::forward<S>(s).report(i, v);
    }
}

template <typename S, typename Info>
struct report_info
{
    static constexpr bool value = internal::has_report<S, Info>::value;
};
template <typename S, typename Info>
inline constexpr bool report_info_v = report_info<S, Info>::value;

template <typename S, typename Info, typename ValueType>
struct report_value
{
    static constexpr bool value = internal::has_report_value<S, Info, ValueType>::value;
};
template <typename S, typename Info, typename ValueType>
inline constexpr bool report_value_v = report_value<S, Info, ValueType>::value;

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif /*_ONEDPL_INTERNAL_DYNAMIC_SELECTION_TRAITS_H*/
