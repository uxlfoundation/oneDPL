// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _ONEDPL_ZIP_VIEW_IMPL_H
#define _ONEDPL_ZIP_VIEW_IMPL_H

#if _ONEDPL_CPP20_RANGES_PRESENT

#include <ranges>
#include <vector>
#include <type_traits>

#include "tuple_impl.h"

namespace oneapi
{
namespace dpl
{

namespace ranges
{

template <bool C, typename... Views>
concept all_forward = ( std::ranges::forward_range<std::conditional_t<C, const Views, Views>> && ... );

template <bool C, typename... Views>
concept all_bidirectional = ( std::ranges::bidirectional_range<std::conditional_t<C, const Views, Views>> && ... );

template <bool C, typename... Views>
concept all_random_access = ( std::ranges::random_access_range<std::conditional_t<C, const Views, Views>> && ... );

template <typename... Rs>
concept zip_is_common =
    (sizeof...(Rs) == 1 && ( std::ranges::common_range<Rs> && ... ))
    ||
    (!(std::ranges::bidirectional_range<Rs> && ...) && (std::ranges::common_range<Rs> && ...))
    ||
    ((std::ranges::random_access_range<Rs> && ...) && (std::ranges::sized_range<Rs> && ...));

template <bool Const, typename... Views>
struct declare_iterator_category {};

template <bool Const, typename... Views>
    requires all_forward<Const, Views...>
struct declare_iterator_category<Const, Views...> {
    using iterator_category = std::input_iterator_tag;
};

template <std::ranges::input_range... Views>
    requires ((std::ranges::view<Views> && ... ) && (sizeof...(Views) > 0))
class zip_view : public std::ranges::view_interface<zip_view<Views...>> {
public:
    template <typename... Types>
    using tuple_type = oneapi::dpl::__internal::tuple<Types...>;

private:
    template <typename _ReturnAdapter, typename _F, typename _Tuple, std::size_t... _Ip>
    static decltype(auto)
    apply_to_tuple_impl(_ReturnAdapter __tr, _F __f, _Tuple& __t, std::index_sequence<_Ip...>)
    {
        return __tr(__f(oneapi::dpl::__internal::get_impl<_Ip>()(__t))...);
    }

    template <typename _ReturnAdapter, typename _F, typename _Tuple>
    static decltype(auto)
    apply_to_tuple(_ReturnAdapter __tr, _F __f, _Tuple& __t)
    {
        return apply_to_tuple_impl(__tr, __f, __t, std::make_index_sequence<sizeof...(Views)>{});
    }

    template <typename _F, typename _Tuple>
    static auto
    apply_to_tuple(_F __f, _Tuple& __t)
    {
        apply_to_tuple_impl([](auto...){}, __f, __t, std::make_index_sequence<sizeof...(Views)>{});
    }

public:
    zip_view() = default;
    constexpr zip_view(Views... views) : views_(views...) {}

    template <bool Const>
    class iterator : declare_iterator_category<Const> {
    public:
        using iterator_concept = std::conditional_t<all_random_access<Const, Views...>,
                                                    std::random_access_iterator_tag,
                                                    std::conditional_t<all_bidirectional<Const, Views...>,
                                                                       std::bidirectional_iterator_tag,
                                                                       std::conditional_t<all_forward<Const, Views...>,
                                                                                          std::forward_iterator_tag,
                                                                                          std::input_iterator_tag>>>;

        using value_type = std::conditional_t<!Const, tuple_type<std::ranges::range_value_t<Views>...>,
                                                      tuple_type<std::ranges::range_value_t<const Views>...>>;

        using difference_type = std::conditional_t<!Const, std::common_type_t<std::ranges::range_difference_t<Views>...>,
                                                           std::common_type_t<std::ranges::range_difference_t<const Views>...>>;
        using return_tuple_type = std::conditional_t<!Const, tuple_type<typename std::iterator_traits<std::ranges::iterator_t<Views>>::reference...>,
            tuple_type<typename std::iterator_traits<std::ranges::iterator_t<const Views>>::reference...>>;

        iterator() = default;

        constexpr iterator(iterator<!Const> i)
            requires Const && (std::convertible_to<std::ranges::iterator_t<Views>, std::ranges::iterator_t<const Views>> && ...)
            : current_(std::move(i.current_)) {}

    private:
        template <typename... Iterators>
        constexpr iterator(const Iterators&... iterators)
            : current_(iterators...) {}
    public:

        return_tuple_type operator*() const {

            auto __tr = [](auto&&... __args) -> decltype(auto) { return return_tuple_type(__args...);};
            return apply_to_tuple(__tr, [](auto it) -> decltype(auto) { return *it;}, current_);
        }

        constexpr return_tuple_type operator[]( difference_type n ) const
            requires all_random_access<Const, Views...>
        {
            return *(*this + n);
        }        

        constexpr iterator& operator++() {
            zip_view::apply_to_tuple([](auto& it) { return ++it; }, current_);
            return *this;
        }

        constexpr void operator++(int) {
            ++*this;
        }

        constexpr iterator operator++(int) requires all_forward<Const, Views...> {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr iterator& operator--() requires all_bidirectional<Const, Views...> {

            zip_view::apply_to_tuple([](auto& it) { return --it; }, current_);
            return *this;
        }

        constexpr iterator operator--(int) requires all_bidirectional<Const, Views...> {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        constexpr iterator& operator+=(difference_type n)
            requires all_random_access<Const, Views...>
        {
            zip_view::apply_to_tuple([n](auto& it) { return it += n; }, current_);
            return *this;
        }

        constexpr iterator& operator-=(difference_type n)
            requires all_random_access<Const, Views...>
        {
            zip_view::apply_to_tuple([n](auto& it) { return it -= n; }, current_);
            return *this;
        }

        friend constexpr bool operator==(const iterator& x, const iterator& y)
            requires ( std::equality_comparable<std::conditional_t<!Const, std::ranges::iterator_t<Views>,
                                                                           std::ranges::iterator_t<const Views>>> && ... )
        {
            if constexpr (all_bidirectional<Const, Views...>) {
                return x.current_ == y.current_;
            } else {
                return x.compare_equal(y, std::make_index_sequence<sizeof...(Views)>());
            }
        }

        friend constexpr auto operator<=>(const iterator& x, const iterator& y)
            requires all_random_access<Const, Views...>
        {
            return x.current_ <=> y.current_;
        }

        friend constexpr auto operator-(const iterator& x, const iterator& y)
            requires all_random_access<Const, Views...>
        {
            return x.distance_to_it(y.current_, std::make_index_sequence<sizeof...(Views)>());
        }

        friend constexpr iterator operator+(iterator it, difference_type n)
        {
            return it+=n;
        }

        friend constexpr iterator operator+(difference_type n, iterator it)
        {
            return it+=n;
        }

        friend constexpr iterator operator-(iterator it, difference_type n)
        {
            return it-=n;
        }

    private:
        template <std::size_t... In>
        constexpr bool compare_equal(iterator y, std::index_sequence<In...>) {
            return ((std::get<In>(current_) == std::get<In>(y.current_)) && ...);
        }

        template <typename SentinelsTuple, std::size_t... In>
        constexpr bool compare_with_sentinels(const SentinelsTuple& sentinels, std::index_sequence<In...>) {
            return ( (std::get<In>(current_) == std::get<In>(sentinels)) || ... );
        }

        template <typename SentinelsTuple, std::size_t... In>
        constexpr std::common_type_t<std::conditional_t<!Const, std::ranges::range_difference_t<Views>,
                                                                std::ranges::range_difference_t<const Views>>...>
        distance_to_sentinels(const SentinelsTuple& sentinels, std::index_sequence<0, In...>) {
            auto min = std::get<0>(current_) - std::get<0>(sentinels);

            ( (min = std::min(min, (std::get<In>(current_) - std::get<In>(sentinels)))) , ... );
            return min;
        }
        template <std::size_t... In>
        constexpr std::common_type_t<std::conditional_t<!Const, std::ranges::range_difference_t<Views>,
                                                                std::ranges::range_difference_t<const Views>>...>
        distance_to_it(const iterator it, std::index_sequence<0, In...>) const {
            auto min = std::get<0>(it.current_) - std::get<0>(current_);

            ( (min = std::min(min, (std::get<In>(it.current_) - std::get<In>(current_)))) , ... );
            return min;
        }

        friend class zip_view;

        using current_type = std::conditional_t<!Const, tuple_type<std::ranges::iterator_t<Views>...>,
                                                        tuple_type<std::ranges::iterator_t<const Views>...>>;

        current_type current_;
    }; // class iterator

    template <bool Const>
    class sentinel {
    public:
        sentinel() = default;
        constexpr sentinel(sentinel<!Const> i)
            requires Const &&
                     ( std::convertible_to<std::ranges::sentinel_t<Views>, std::ranges::sentinel_t<const Views>> && ... )
            : end_(std::move(i.end_)) {}

    private:
        template <typename... Sentinels>
        constexpr sentinel(const Sentinels&... sentinels)
            : end_(sentinels...) {}
    public:
        template <bool OtherConst>
            requires (std::sentinel_for<std::conditional_t<!Const, std::ranges::sentinel_t<Views>,
                                                                   std::ranges::sentinel_t<const Views>>,
                                        std::conditional_t<!OtherConst, std::ranges::iterator_t<Views>,
                                                                        std::ranges::iterator_t<const Views>>> && ...)
        friend constexpr bool operator==(const iterator<OtherConst>& x, const sentinel& y)
        {
            return x.compare_with_sentinels(y.end_, std::make_index_sequence<sizeof...(Views)>());
        }

        template <bool OtherConst>
            requires (std::sized_sentinel_for<std::conditional_t<!Const, std::ranges::sentinel_t<Views>,
                                                                         std::ranges::sentinel_t<const Views>>,
                                              std::conditional_t<!OtherConst, std::ranges::iterator_t<Views>,
                                                                              std::ranges::iterator_t<const Views>>> && ...)
        friend constexpr std::common_type_t<std::conditional_t<!Const, std::ranges::range_difference_t<Views>,
                                                                       std::ranges::range_difference_t<const Views>>...>
        operator-(const iterator<OtherConst>& x, const sentinel& y) {
            return x.distance_to_sentinels(y.end_, std::make_index_sequence<sizeof...(Views)>());
        }

        template <bool OtherConst>
            requires (std::sized_sentinel_for<std::conditional_t<!Const, std::ranges::sentinel_t<Views>,
                                                                         std::ranges::sentinel_t<const Views>>,
                                              std::conditional_t<!OtherConst, std::ranges::iterator_t<Views>,
                                                                              std::ranges::iterator_t<const Views>>> && ...)
        friend constexpr std::common_type_t<std::conditional_t<!Const, std::ranges::range_difference_t<Views>,
                                                                       std::ranges::range_difference_t<const Views>>...>
        operator-(const sentinel& y, const iterator<OtherConst>& x) {
            return -(x - y);
        }

    private:
        friend class zip_view;

        using end_type = std::conditional_t<!Const, tuple_type<std::ranges::sentinel_t<Views>...>,
                                                    tuple_type<std::ranges::sentinel_t<const Views>...>>;

        end_type end_;
    }; // class sentinel    

    constexpr auto begin() requires (std::ranges::range<Views> && ...) // !simple_view?
    {
        auto __tr = [](auto... __args) { return iterator<false>(__args...);};
        return apply_to_tuple(__tr, std::ranges::begin, views_);
    }

    constexpr auto begin() const requires ( std::ranges::range<const Views> && ... )
    {
        auto __tr = [](auto... __args) { return iterator<true>(__args...);};
        return apply_to_tuple(__tr, std::ranges::begin, views_);
    }

    constexpr auto end() requires (std::ranges::range<Views> && ...) // requires !simple_view {
    {
        if constexpr (!zip_is_common<Views...>)
        {
            auto __tr = [](auto... __args) { return sentinel<false>(__args...);};
            return apply_to_tuple(__tr, std::ranges::end, views_);
        }
        else if constexpr ((std::ranges::random_access_range<Views> && ...))
        {
            auto it = begin();
            it += size();
            return it;
        }
        else
        {
            auto __tr = [](auto... __args) { return iterator<false>(__args...);};
            return apply_to_tuple(__tr, std::ranges::end, views_);
        }
    }

    constexpr auto end() const requires (std::ranges::range<const Views> && ...)
    {
        if constexpr (!zip_is_common<Views...>)
        {
            auto __tr = [](auto... __args) { return sentinel<true>(__args...);};
            return apply_to_tuple(__tr, std::ranges::end, views_);
        } 
        else if constexpr ((std::ranges::random_access_range<Views> && ...))
        {
            auto it = begin();
            it += size();
            return it;
        }
        else
        {
            auto __tr = [](auto... __args) { return iterator<true>(__args...);};
            return apply_to_tuple(__tr, std::ranges::end, views_);
        }
    }

    constexpr auto size() requires (std::ranges::sized_range<Views> && ...)
    {
        auto __tr = [](auto... __args) { 
            using CT = std::make_unsigned_t<std::common_type_t<decltype(__args)...>>;
            return std::ranges::min({CT(__args)...});
            };

        return apply_to_tuple(__tr, std::ranges::size, views_);
    }

    constexpr auto size() const requires (std::ranges::sized_range<Views> && ...)
    {
        return const_cast<zip_view*>(this)->size();
    }
private:
    tuple_type<Views...> views_;
}; // class zip_view

template <typename... Rs>
zip_view(Rs&&...) -> zip_view<std::views::all_t<Rs>...>;

struct zip_fn {
    template <std::ranges::viewable_range... Rs>
    constexpr auto operator()( Rs&&... rs ) const {
        return zip_view<std::views::all_t<decltype((rs))>...>(std::forward<Rs>(rs)...);
    }
};

inline constexpr zip_fn zip{};

} // namespace ranges
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_CPP20_RANGES_PRESENT

#endif //_ONEDPL_ZIP_VIEW_IMPL_H
