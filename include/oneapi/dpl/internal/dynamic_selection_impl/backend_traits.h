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

//utility to check if any of a variadic pack is equal to a specific type passed first
template<typename compare_to, typename ...T>
struct any_of : std::disjunction<std::is_same<compare_to, T>...>
{
};

template<typename compare_to, typename ...T>
inline constexpr bool any_of_v = any_of<compare_to, T...>::value;

// Default trait for scratch space 
template <typename ...T>
struct no_scratch_t
{
};

template <typename Backend, bool has_scratch_type = false, typename ...Req>
struct scratch_trait_t_impl
{
    using type = no_scratch_t<Req...>;
};

// scratch space trait if backend defines it
template <typename Backend, typename ...Req>
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
template <typename Backend, typename ...Req>
auto
has_scratch_space_impl(...) -> std::false_type;

template <typename Backend, typename ...Req>
auto
has_scratch_space_impl(int) -> decltype(std::declval<typename Backend::template scratch_t<Req...>>(), std::true_type{});

template <typename Backend, typename ...Req>
struct has_scratch_space : decltype(has_scratch_space_impl<Backend>(0))
{
};

} //namespace internal


namespace backend_traits
{
template <typename S>
struct lazy_report_value
{
    static constexpr bool value = ::oneapi::dpl::experimental::internal::has_lazy_report<S>::value;
};
template <typename S>
inline constexpr bool lazy_report_v = lazy_report_value<S>::value;

template <typename Backend, typename ...Req>
inline constexpr bool scratch_space_v = internal::has_scratch_space<Backend, Req...>::value;


template <typename Backend, typename ...Req>
using selection_scratch_t = typename scratch_trait_t_impl<Backend, backend_traits::scratch_space_v<Backend, Req...>,Req...>::type;

} //namespace backend_traits


namespace internal
{
template <typename ResourceType>
struct has_initialize
{
    template <typename T>
    static auto test(int) -> decltype(std::declval<T>().initialize(), std::true_type{});

    template <typename>
    static auto test(...) -> std::false_type;

    static constexpr bool value = decltype(test<ResourceType>(0))::value;
};

template <typename ResourceType>
struct has_reset
{
    template <typename T>
    static auto test(int) -> decltype(std::declval<T>().reset(), std::true_type{});
    template <typename>
    static auto test(...) -> std::false_type;
    static constexpr bool value = decltype(test<ResourceType>(0))::value;
};

} //namespace internal

template <typename ResourceType>
struct extra_resource_traits
{
    static constexpr bool has_initialize_v = internal::has_initialize<ResourceType>::value; 

    static constexpr bool has_reset_v = internal::has_reset<ResourceType>::value;
};

} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif /*_ONEDPL_INTERNAL_BACKEND_TRAITS_H*/
