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
#include "iterator_impl.h"

namespace oneapi
{
namespace dpl
{

namespace ranges
{

template <bool Const, typename T>
using __maybe_const = std::conditional_t<Const, const T, T>;

template <bool C, typename... Views>
concept all_forward = (std::ranges::forward_range<__maybe_const<C, Views>> && ...);

template <bool C, typename... Views>
concept all_bidirectional = (std::ranges::bidirectional_range<std::conditional_t<C, const Views, Views>> && ...);

template <bool C, typename... Views>
concept all_random_access = (std::ranges::random_access_range<std::conditional_t<C, const Views, Views>> && ...);

template <typename... Rs>
concept zip_is_common = (sizeof...(Rs) == 1 && (std::ranges::common_range<Rs> && ...)) ||
                        (!(std::ranges::bidirectional_range<Rs> && ...) && (std::ranges::common_range<Rs> && ...)) ||
                        ((std::ranges::random_access_range<Rs> && ...) && (std::ranges::sized_range<Rs> && ...));

template <bool Const, typename... Views>
struct declare_iterator_category
{
};

template <bool Const, typename... Views>
requires all_forward<Const, Views...> struct declare_iterator_category<Const, Views...>
{
    using iterator_category = std::input_iterator_tag;
};

template <typename _R>
    concept __simple_view =
        std::ranges::view<_R> && std::ranges::range<const _R> &&
        std::same_as<std::ranges::iterator_t<_R>, std::ranges::iterator_t<const _R>> &&
        std::same_as<std::ranges::sentinel_t<_R>, std::ranges::sentinel_t<const _R>>;

template <std::ranges::input_range... Views>
requires((std::ranges::view<Views> && ...) && (sizeof...(Views) > 0))
class zip_view: public std::ranges::view_interface<zip_view<Views...>>
{
    template <typename... Types>
    using tuple_type = oneapi::dpl::__internal::tuple<Types...>;

    template <typename _ReturnAdapter, typename _F, typename _Tuple, std::size_t... _Ip>
    static decltype(auto)
    apply_to_tuple_impl(_ReturnAdapter __tr, _F __f, _Tuple& __t, std::index_sequence<_Ip...>)
    {
        return __tr(__f(std::get<_Ip>(__t))...);
    }    
public:
    template <typename _ReturnAdapter, typename _F, typename _Tuple>
    static decltype(auto)
    apply_to_tuple(_ReturnAdapter __tr, _F __f, _Tuple& __t)
    {
        return apply_to_tuple_impl(__tr, __f, __t, std::make_index_sequence<sizeof...(Views)>{});
    }

    template <typename _F, typename _Tuple>
    static void
    apply_to_tuple(_F __f, _Tuple& __t)
    {
        apply_to_tuple([](auto&&...) {}, __f, __t);
    }

    template <typename _F, typename _Tuple1, typename _Tuple2, std::size_t... _Ip>
    static void
    bi_apply_to_tuple_impl(_F __f, _Tuple1& __t1, _Tuple2& __t2, std::index_sequence<_Ip...>)
    {
        (__f(std::get<_Ip>(__t1), std::get<_Ip>(__t2)), ...);
    }

    template <typename _F, typename _Tuple1, typename _Tuple2>
    static void
    bi_apply_to_tuple(_F __f, _Tuple1& __t1, _Tuple2& __t2)
    {
        return bi_apply_to_tuple_impl(__f, __t1, __t2, std::make_index_sequence<sizeof...(Views)>{});
    }

  public:
    zip_view() = default;
    constexpr explicit zip_view(Views... views) : views_(std::move(views)...) {}

    template <bool Const>
    class iterator : public declare_iterator_category<Const, Views...>
    {
      public:
        using iterator_concept = std::conditional_t<
            all_random_access<Const, Views...>, std::random_access_iterator_tag,
            std::conditional_t<
                all_bidirectional<Const, Views...>, std::bidirectional_iterator_tag,
                std::conditional_t<all_forward<Const, Views...>, std::forward_iterator_tag, std::input_iterator_tag>>>;

        using value_type = tuple_type<std::ranges::range_value_t<__maybe_const<Const, Views>>...>;
        using difference_type = std::common_type_t<std::ranges::range_difference_t<__maybe_const<Const, Views>>...>;
private:
        using reference_type = tuple_type<std::ranges::range_reference_t<__maybe_const<Const, Views>>...>;        
        using rvalue_reference_type = tuple_type<std::ranges::range_rvalue_reference_t<__maybe_const<Const, Views>>...>;

        using iterator_type = tuple_type<std::ranges::iterator_t<__maybe_const<Const, Views>>...>;
public:
        iterator() = default;

        constexpr iterator(iterator<!Const> i) requires Const &&
            (std::convertible_to<std::ranges::iterator_t<Views>, std::ranges::iterator_t<__maybe_const<Const, Views>>> && ...)
            : current_(std::move(i.current_))
        {
        }

      private:        
        //template <typename... Iterators>
        //constexpr explicit iterator(const Iterators&... iterators) : current_(iterators...)
        //{
        //}
        constexpr explicit iterator(iterator_type __current)
        : current_(std::move(__current)) {}

      public:
        template <typename... Iterators>
        operator oneapi::dpl::zip_iterator<Iterators...>() const
        {
            auto __tr = [](auto&&... __args) -> decltype(auto) { return oneapi::dpl::make_zip_iterator(std::forward<decltype(__args)>(__args)...); };
            return apply_to_tuple(__tr, [](auto it) -> decltype(auto) { return it; }, current_);
        }

        constexpr decltype(auto)
        operator*() const
        {
            auto __tr = [](auto&&... __args) -> decltype(auto) {
                return reference_type(std::forward<decltype(__args)>(__args)...);
            };
            return apply_to_tuple(__tr, [](auto& it) -> decltype(auto) { return *it; }, current_);
        }

        constexpr decltype(auto)
        operator[](difference_type n) const requires all_random_access<Const, Views...>
        {
            return *(*this + n);
        }

        constexpr iterator&
        operator++()
        {
            apply_to_tuple([](auto& it) -> decltype(auto) { return ++it; }, current_);
            return *this;
        }

        constexpr void
        operator++(int)
        {
            ++*this;
        }

        constexpr iterator
        operator++(int) requires all_forward<Const, Views...>
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr iterator&
        operator--() requires all_bidirectional<Const, Views...>
        {
            apply_to_tuple([](auto& it) { return --it; }, current_);
            return *this;
        }

        constexpr iterator
        operator--(int) requires all_bidirectional<Const, Views...>
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        constexpr iterator&
        operator+=(difference_type n) requires all_random_access<Const, Views...>
        {
            apply_to_tuple([n](auto& it) { return it += n; }, current_);
            return *this;
        }

        constexpr iterator&
        operator-=(difference_type n) requires all_random_access<Const, Views...>
        {
            apply_to_tuple([n](auto& it) { return it -= n; }, current_);
            return *this;
        }
        
        friend constexpr bool
        operator==(const iterator& x, const iterator& y)
            requires(std::equality_comparable<std::ranges::iterator_t<__maybe_const<Const, Views>>>&&...)
        {
            if constexpr (all_bidirectional<Const, Views...>)
            {
                return x.current_ == y.current_;
            }
            else
            {
                return x.compare_equal(y, std::make_index_sequence<sizeof...(Views)>());
            }
        }

        friend constexpr auto
        operator<=>(const iterator& x, const iterator& y) requires all_random_access<Const, Views...>
        {
            if (x.current_ < y.current_)
                return std::weak_ordering::less;
            else if (x.current_ == y.current_)
                return std::weak_ordering::equivalent;
            return std::weak_ordering::greater; //x.current > y.current_
        }
        
        friend constexpr auto
        operator-(const iterator& x, const iterator& y)
            requires(std::sized_sentinel_for<std::ranges::iterator_t<__maybe_const<Const, Views>>,
                                                               std::ranges::iterator_t<__maybe_const<Const, Views>>> && ...)
        {
            return y.distance_to_it(x, std::make_index_sequence<sizeof...(Views)>());
        }

        friend constexpr iterator
        operator+(iterator it, difference_type n) requires all_random_access<Const, Views...>
        {
            return it += n;
        }

        friend constexpr iterator
        operator+(difference_type n, iterator it) requires all_random_access<Const, Views...>
        {
            return it += n;
        }

        friend constexpr iterator
        operator-(iterator it, difference_type n) requires all_random_access<Const, Views...>
        {
            return it -= n;
        }

        friend constexpr decltype(auto) iter_move(const iterator& x) noexcept(
            (noexcept(std::ranges::iter_move(std::declval<const std::ranges::iterator_t<__maybe_const<Const, Views>>&>())) && ...) &&
            (std::is_nothrow_move_constructible_v<std::ranges::range_rvalue_reference_t<__maybe_const<Const, Views>>> && ...))
        {
            auto __tr = [](auto&&... __args) -> decltype(auto) {
                return rvalue_reference_type(std::forward<decltype(__args)>(__args)...);
            };
            return apply_to_tuple(__tr, std::ranges::iter_move, x.current_);

        }

        friend constexpr void iter_swap(const iterator& x, const iterator& y) noexcept(
            (noexcept(std::ranges::iter_swap(std::declval<const std::ranges::iterator_t<__maybe_const<Const, Views>>&>(),
                                        std::declval<const std::ranges::iterator_t<__maybe_const<Const, Views>>&>())) && ...))
          requires(std::indirectly_swappable<std::ranges::iterator_t<__maybe_const<Const, Views>>> && ...)
        {
            bi_apply_to_tuple(std::ranges::iter_swap, x.current_, y.current_);
        }

      private:
        template <std::size_t... In>
        constexpr bool
        compare_equal(iterator y, std::index_sequence<In...>) const
        {
            return ((std::get<In>(current_) == std::get<In>(y.current_)) || ...);
        }

        template <typename SentinelsTuple, std::size_t... In>
        constexpr bool
        compare_with_sentinels(const SentinelsTuple& sentinels, std::index_sequence<In...>) const
        {
            return ((std::get<In>(current_) == std::get<In>(sentinels)) || ...);
        }
        
        template <typename SentinelsTuple, std::size_t... In>
        constexpr std::common_type_t<std::ranges::range_difference_t<__maybe_const<Const, Views>>...>        
        distance_to_sentinels(const SentinelsTuple& sentinels, std::index_sequence<In...>) const
        {
            return std::ranges::min({difference_type(std::get<In>(current_) - std::get<In>(sentinels))...}, std::less{},
                                    [](auto a){ return std::abs(a);});
        }
        
        template <std::size_t... In>
        constexpr std::common_type_t<std::ranges::range_difference_t<__maybe_const<Const, Views>>...>
        distance_to_it(const iterator it, std::index_sequence<In...>) const
        {
            return std::ranges::min({difference_type(std::get<In>(it.current_) - std::get<In>(current_))...}, std::less{},
                                    [](auto a){ return std::abs(a);});
        }

        friend class zip_view;

        iterator_type current_;
    }; // class iterator

    template <bool Const>
    class sentinel
    {
        template <typename... Sentinels>
        constexpr explicit sentinel(const Sentinels&... sentinels) : end_(sentinels...)
        {
        }

      public:
        sentinel() = default;
        constexpr sentinel(sentinel<!Const> i) requires Const &&
            (std::convertible_to<std::ranges::sentinel_t<Views>, std::ranges::sentinel_t<__maybe_const<Const, Views>>> && ...)
            : end_(std::move(i.end_))
        {
        }
      
        template <bool OtherConst>
        requires(std::sentinel_for<std::ranges::sentinel_t<__maybe_const<Const, Views>>, 
                                   std::ranges::iterator_t<__maybe_const<OtherConst, Views>>>&&...)
        friend constexpr bool
        operator==(const iterator<OtherConst>& x, const sentinel& y)
        {
            return x.compare_with_sentinels(y.end_, std::make_index_sequence<sizeof...(Views)>());
        }
        
        template <bool OtherConst>
        requires(std::sized_sentinel_for<std::ranges::sentinel_t<__maybe_const<Const, Views>>, 
                                         std::ranges::iterator_t<__maybe_const<OtherConst, Views>>>&&...)
        friend constexpr std::common_type_t<std::ranges::range_difference_t<__maybe_const<Const, Views>>...>        
            operator-(const iterator<OtherConst>& x, const sentinel& y)
        {
            return x.distance_to_sentinels(y.end_, std::make_index_sequence<sizeof...(Views)>());
        }
        
        template <bool OtherConst>
        requires(std::sized_sentinel_for<std::ranges::sentinel_t<__maybe_const<Const, Views>>, 
                                         std::ranges::iterator_t<__maybe_const<OtherConst, Views>>>&&...)
        friend constexpr std::common_type_t<std::ranges::range_difference_t<__maybe_const<Const, Views>>...>
            operator-(const sentinel& y, const iterator<OtherConst>& x)
        {
            return -(x - y);
        }

      private:
        friend class zip_view;
        
        tuple_type<std::ranges::sentinel_t<__maybe_const<Const, Views>>...> end_;
    }; // class sentinel

    constexpr auto
    begin() requires(!(__simple_view<Views> && ...))
    {
        using iterator_type = tuple_type<std::ranges::iterator_t<__maybe_const<false, Views>>...>;
        auto __tr = [](auto&&... __args) {
            return iterator<false>(iterator_type(std::forward<decltype(__args)>(__args)...)); 
        };
        return apply_to_tuple(__tr, std::ranges::begin, views_);
    }

    constexpr auto
    begin() const requires(std::ranges::range<const Views>&&...)
    {
        using iterator_type = tuple_type<std::ranges::iterator_t<__maybe_const<true, Views>>...>;
        auto __tr = [](auto&&... __args) { 
            return iterator<true>(iterator_type(std::forward<decltype(__args)>(__args)...)); 
        };
        return apply_to_tuple(__tr, std::ranges::begin, views_);
    }

    constexpr auto
    end() requires(!(__simple_view<Views> && ...))
    {
        if constexpr (!zip_is_common<Views...>)
        {
            auto __tr = [](auto&&... __args) { return sentinel<false>(std::forward<decltype(__args)>(__args)...); };
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
            using iterator_type = tuple_type<std::ranges::iterator_t<__maybe_const<false, Views>>...>;
            auto __tr = [](auto&&... __args) { return iterator<false>(iterator_type(std::forward<decltype(__args)>(__args)...)); };
            return apply_to_tuple(__tr, std::ranges::end, views_);
        }
    }

    constexpr auto
    end() const requires(std::ranges::range<const Views>&&...)
    {
        if constexpr (!zip_is_common<Views...>)
        {
            auto __tr = [](auto&&... __args) { return sentinel<true>(std::forward<decltype(__args)>(__args)...); };
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
            using iterator_type = tuple_type<std::ranges::iterator_t<__maybe_const<true, Views>>...>;
            auto __tr = [](auto&&... __args) { return iterator<true>(iterator_type(__args...)); };
            return apply_to_tuple(__tr, std::ranges::end, views_);
        }
    }

    constexpr auto
    size() requires(std::ranges::sized_range<Views>&&...)
    {
        auto __tr = [](auto... __args) {
            using CT = std::make_unsigned_t<std::common_type_t<decltype(__args)...>>;
            return std::ranges::min({CT(__args)...});
        };

        return apply_to_tuple(__tr, std::ranges::size, views_);
    }

    constexpr auto
    size() const requires(std::ranges::sized_range<const Views>&&...)
    {
        auto __tr = [](auto... __args) {
            using CT = std::make_unsigned_t<std::common_type_t<decltype(__args)...>>;
            return std::ranges::min({CT(__args)...});
        };

        return apply_to_tuple(__tr, std::ranges::size, views_);
    }

  private:
    tuple_type<Views...> views_;
}; // class zip_view

template <typename... Rs>
zip_view(Rs&&...) -> zip_view<std::views::all_t<Rs>...>;

namespace __internal
{
struct zip_fn
{
    template <class... _Ranges>
    constexpr auto
    operator()(_Ranges&&... __rs) const noexcept(noexcept(oneapi::dpl::ranges::zip_view<std::views::all_t<_Ranges&&>...>(std::forward<_Ranges>(__rs)...)))
      -> decltype(oneapi::dpl::ranges::zip_view<std::views::all_t<_Ranges&&>...>(std::forward<_Ranges>(__rs)...)) {
    return oneapi::dpl::ranges::zip_view<std::views::all_t<_Ranges>...>(std::forward<_Ranges>(__rs)...);
    }
    
    constexpr auto
    operator()() const noexcept { return std::ranges::empty_view<oneapi::dpl::__internal::tuple<>>{}; }
};
} // namespace __internal

namespace views
{
inline constexpr oneapi::dpl::ranges::__internal::zip_fn zip{};
} //namespace views

} // namespace ranges

namespace views
{
using ranges::views::zip;
} //namespace views

} // namespace dpl
} // namespace oneapi

template <class... _Ranges>
inline constexpr bool std::ranges::enable_borrowed_range<oneapi::dpl::ranges::zip_view<_Ranges...>> = 
    (std::ranges::enable_borrowed_range<_Ranges> && ...);

#endif //_ONEDPL_CPP20_RANGES_PRESENT

#endif //_ONEDPL_ZIP_VIEW_IMPL_H
