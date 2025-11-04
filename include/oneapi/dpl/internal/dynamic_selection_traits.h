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
#include <thread>
#include <optional>
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

template <typename SelectionHandle>
auto
has_scratch_space_member(...) -> std::false_type;

template <typename SelectionHandle>
auto
has_scratch_space_member(int) -> decltype(std::declval<SelectionHandle>().scratch_space, std::true_type{});

template <typename SelectionHandle>
struct scratch_space_member : decltype(has_scratch_space_member<SelectionHandle>(0)) //TODO: Change name?
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

template <typename T>
auto
has_wait_impl(...) -> std::false_type;

template <typename T>
auto
has_wait_impl(int) -> decltype(std::declval<T>().wait(), std::true_type{});

template <typename T>
struct has_wait : decltype(has_wait_impl<T>(0))
{
};
template <typename Policy, typename Function, typename... Args>
auto
has_try_submit_impl(...) -> std::false_type;

template <typename Policy, typename Function, typename... Args>
auto
has_try_submit_impl(int)
    -> decltype(std::declval<Policy>().try_submit(std::declval<Function>(), std::declval<Args>()...), std::true_type{});
template <typename Policy, typename Function, typename... Args>
struct has_try_submit : decltype(has_try_submit_impl<Policy, Function, Args...>(0))
{
};

template <typename Policy, typename Function, typename... Args>
inline constexpr bool has_try_submit_v = has_try_submit<Policy, Function, Args...>::value;

template <typename Policy, typename Function, typename... Args>
auto
has_submit_impl(...) -> std::false_type;

template <typename Policy, typename Function, typename... Args>
auto
has_submit_impl(int)
    -> decltype(std::declval<Policy>().submit(std::declval<Function>(), std::declval<Args>()...), std::true_type{});

template <typename Policy, typename Function, typename... Args>
struct has_submit : decltype(has_submit_impl<Policy, Function, Args...>(0))
{
};

template <typename Policy, typename Function, typename... Args>
inline constexpr bool has_submit_v = has_submit<Policy, Function, Args...>::value;

template <typename Policy, typename Function, typename... Args>
auto
has_submit_and_wait_impl(...) -> std::false_type;

template <typename Policy, typename Function, typename... Args>
auto
has_submit_and_wait_impl(int)
    -> decltype(std::declval<Policy>().submit_and_wait(std::declval<Function>(), std::declval<Args>()...),
                std::true_type{});

template <typename Policy, typename Function, typename... Args>
struct has_submit_and_wait : decltype(has_submit_and_wait_impl<Policy, Function, Args...>(0))
{
};

template <typename Policy, typename Function, typename... Args>
inline constexpr bool has_submit_and_wait_v = has_submit_and_wait<Policy, Function, Args...>::value;

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

template <typename Policy, typename Function, typename... Args>
auto
submit(Policy&& p, Function&& f, Args&&... args)
    -> std::enable_if_t<internal::has_submit_v<Policy, Function, Args...>,
                        decltype(std::declval<Policy>().submit(std::declval<Function>(), std::declval<Args>()...))>
{
    // Policy has a direct submit method
    return std::forward<Policy>(p).submit(std::forward<Function>(f), std::forward<Args>(args)...);
}

template <typename Policy, typename Function, typename... Args>
auto
submit(Policy&& p, Function&& f, Args&&... args) -> std::enable_if_t<
    !internal::has_submit_v<Policy, Function, Args...> && internal::has_try_submit_v<Policy, Function, Args...>,
    decltype(std::declval<Policy>().try_submit(std::declval<Function>(), std::declval<Args>()...).value())>
{
    // Policy has a try_submit method
    auto result = std::forward<Policy>(p).try_submit(f, args...);
    while (!result.has_value())
    {
        std::this_thread::yield();
        result = std::forward<Policy>(p).try_submit(f, args...);
    }
    return result.value();
}

template <typename Policy, typename Function, typename... Args>
auto
submit_and_wait(Policy&& p, Function&& f, Args&&... args)
{
    if constexpr (internal::has_submit_and_wait_v<Policy, Function, Args...>)
    {
        // Policy has a direct submit_and_wait method
        return std::forward<Policy>(p).submit_and_wait(std::forward<Function>(f), std::forward<Args>(args)...);
    }
    else if constexpr (internal::has_submit_v<Policy, Function, Args...> ||
                       internal::has_try_submit_v<Policy, Function, Args...>)
    {
        // Fall back to submit + wait
        auto result = submit(std::forward<Policy>(p), std::forward<Function>(f), std::forward<Args>(args)...);
        wait(result);
    }
    else
    {
        static_assert(false, "error: submit_and_wait() called on policy which does not support any submission method");
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

// Helpers to check if a type is in a parameter pack.
// Utilities for scratch space determination based upon variadic pack of the below reporting requirement structs.
template <typename T, typename... Ts>
struct contains_reporting_req : std::disjunction<std::is_same<T, Ts>...>
{
};

template <typename T, typename... Ts>
static constexpr bool contains_reporting_req_v = contains_reporting_req<T, Ts...>::value;

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
