// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_INTERNAL_BACKEND_TRAITS_H
#define _ONEDPL_INTERNAL_BACKEND_TRAITS_H

#include <utility>
#include <type_traits>

namespace oneapi
{
namespace dpl
{
namespace experimental
{

// Default trait for scratch space
template <typename... T>
struct no_scratch_t
{
};

template <typename Backend, bool has_scratch_type = false, typename... Req>
struct scratch_trait_t_impl
{
    using type = no_scratch_t<Req...>;
};

// scratch space trait if backend defines it
template <typename Backend, typename... Req>
struct scratch_trait_t_impl<Backend, true, Req...>
{
    using type = typename Backend::template scratch_t<Req...>;
};

namespace internal
{
template <typename Backend>
auto
has_lazy_report_impl(...) -> std::false_type;

template <typename Backend>
auto
has_lazy_report_impl(int) -> decltype(std::declval<Backend>().lazy_report(), std::true_type{});

template <typename Backend>
struct has_lazy_report : decltype(has_lazy_report_impl<Backend>(0))
{
};

// This is to detect if a backend has a scratch space struct defined in it with a variadic pack of template args
template <typename Backend, typename... Req>
auto
has_scratch_space_impl(...) -> std::false_type;

template <typename Backend, typename... Req>
auto
has_scratch_space_impl(int) -> decltype(std::declval<typename Backend::template scratch_t<Req...>>(), std::true_type{});

template <typename Backend, typename... Req>
struct has_scratch_space : decltype(has_scratch_space_impl<Backend>(0))
{
};

template <typename Backend>
auto
has_wait_type_impl(...) -> std::false_type;

template <typename Backend>
auto
has_wait_type_impl(int) -> decltype(std::declval<typename Backend::wait_type>(), std::true_type{});

template <typename Backend>
struct has_wait_type : decltype(has_wait_type_impl<Backend>(0))
{
};

template <typename Backend, bool = internal::has_wait_type<Backend>::value>
struct wait_trait
{
    using type = void;
};

template <typename Backend>
struct wait_trait<Backend, true>
{
    using type = typename Backend::wait_type;
};

} //namespace internal

template <typename Backend>
struct backend_traits
{

    constexpr static bool has_wait_type_v = internal::has_wait_type<std::decay_t<Backend>>::value;

    using wait_type = typename internal::wait_trait<std::decay_t<Backend>>::type;

    constexpr static bool lazy_report_v = internal::has_lazy_report<std::decay_t<Backend>>::value;

    template <typename... Req>
    constexpr static bool has_scratch_space_v = internal::has_scratch_space<std::decay_t<Backend>, Req...>::value;

    template <typename... Req>
    using selection_scratch_t =
        typename scratch_trait_t_impl<std::decay_t<Backend>, has_scratch_space_v<Req...>, Req...>::type;
};

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif /*_ONEDPL_INTERNAL_BACKEND_TRAITS_H*/
