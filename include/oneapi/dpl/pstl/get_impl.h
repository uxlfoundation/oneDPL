// -*- C++ -*-
//===-- tuple_impl.h ---------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_GET_IMPL_H
#define _ONEDPL_GET_IMPL_H

#include <iterator>
#include <tuple>
#include <cassert>
#include <type_traits>

#include "utils.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{
/*
get implementations:

    std::variant - https://en.cppreference.com/w/cpp/utility/variant/get
    
        template< std::size_t I, class... Types >
        constexpr std::variant_alternative_t<I, std::variant<Types...>>&
        get(std::variant<Types...>& v );
        
        template< std::size_t I, class... Types >
        constexpr std::variant_alternative_t<I, std::variant<Types...>>&&
        get(std::variant<Types...>&& v );
        
        template< std::size_t I, class... Types >
        constexpr const std::variant_alternative_t<I, std::variant<Types...>>&
        get(const std::variant<Types...>& v );
        
        template< std::size_t I, class... Types >
        constexpr const std::variant_alternative_t<I, std::variant<Types...>>&&
        get(const std::variant<Types...>&& v );    
        
    std::pair - https://en.cppreference.com/w/cpp/utility/pair/get
    
        template< std::size_t I, class T1, class T2 >
        constexpr typename std::tuple_element<I, std::pair<T1,T2> >::type&
        get(std::pair<T1, T2>& p ) noexcept;
        
        template< std::size_t I, class T1, class T2 >
        constexpr const typename std::tuple_element<I, std::pair<T1,T2> >::type&
        get(const std::pair<T1,T2>& p ) noexcept;
        
        template< std::size_t I, class T1, class T2 >
        constexpr typename std::tuple_element<I, std::pair<T1,T2> >::type&&
        get(std::pair<T1,T2>&& p ) noexcept;
        
        template< std::size_t I, class T1, class T2 >
        constexpr const typename std::tuple_element<I, std::pair<T1,T2> >::type&&
        get(const std::pair<T1,T2>&& p ) noexcept;    
        
    std::array - https://en.cppreference.com/w/cpp/container/array/get
    
        template< std::size_t I, class T, std::size_t N >
        constexpr T& get(std::array<T,N>& a ) noexcept;

        template< std::size_t I, class T, std::size_t N >
        constexpr T&& get(std::array<T,N>&& a ) noexcept;

        template< std::size_t I, class T, std::size_t N >
        constexpr const T& get(const std::array<T,N>& a ) noexcept;

        template< std::size_t I, class T, std::size_t N >
        constexpr const T&& get(const std::array<T,N>&& a ) noexcept;
        
    oneapi::dpl::__internal::tuple

        template <size_t _Idx, typename... _Tp>
        constexpr std::tuple_element_t<_Idx, oneapi::dpl::__internal::tuple<_Tp...>>&
        get(oneapi::dpl::__internal::tuple<_Tp...>&);

        template <size_t _Idx, typename... _Tp>
        constexpr std::tuple_element_t<_Idx, oneapi::dpl::__internal::tuple<_Tp...>> const&
        get(const oneapi::dpl::__internal::tuple<_Tp...>&);

        template <size_t _Idx, typename... _Tp>
        constexpr std::tuple_element_t<_Idx, oneapi::dpl::__internal::tuple<_Tp...>>&&
        get(oneapi::dpl::__internal::tuple<_Tp...>&&);

        template <size_t _Idx, typename... _Tp>
        constexpr std::tuple_element_t<_Idx, oneapi::dpl::__internal::tuple<_Tp...>> const&&
        get(const oneapi::dpl::__internal::tuple<_Tp...>&&);

*/

    template <size_t _Idx, typename... _Args>
    constexpr auto
    __get(_Args&&... __args) -> decltype(std::get<_Idx>(std::forward<_Args>(__args)...))
    {
        using std::get;

        return get<_Idx>(std::forward<_Args>(__args)...);
    }

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_GET_IMPL_H
