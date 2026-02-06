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

#    include <ranges>
#    include <vector>
#    include <type_traits>

#    include "tuple_impl.h"
#    include "iterator_impl.h"

namespace oneapi
{
namespace dpl
{

namespace ranges
{

namespace __internal
{

template <bool _Const, typename T>
using __maybe_const = std::conditional_t<_Const, const T, T>;

template <bool C, typename... _Views>
concept __all_forward = (std::ranges::forward_range<__maybe_const<C, _Views>> && ...);

template <bool C, typename... _Views>
concept __all_bidirectional = (std::ranges::bidirectional_range<__maybe_const<C, _Views>> && ...);

template <bool C, typename... _Views>
concept __all_random_access = (std::ranges::random_access_range<__maybe_const<C, _Views>> && ...);

template <typename... Rs>
concept __zip_is_common = (sizeof...(Rs) == 1 && (std::ranges::common_range<Rs> && ...)) ||
                          (!(std::ranges::bidirectional_range<Rs> && ...) && (std::ranges::common_range<Rs> && ...)) ||
                          ((std::ranges::random_access_range<Rs> && ...) && (std::ranges::sized_range<Rs> && ...));

template <bool _Const, typename... _Views>
struct __declare_iterator_category
{
};

template <bool _Const, typename... _Views>
    requires __all_forward<_Const, _Views...>
struct __declare_iterator_category<_Const, _Views...>
{
    using iterator_category = std::input_iterator_tag;
};

template <typename _R>
concept __simple_view = std::ranges::view<_R> && std::ranges::range<const _R> &&
                        std::same_as<std::ranges::iterator_t<_R>, std::ranges::iterator_t<const _R>> &&
                        std::same_as<std::ranges::sentinel_t<_R>, std::ranges::sentinel_t<const _R>>;

template <typename _ReturnAdapter, typename _F, typename _Tuple, std::size_t... _Ip>
static decltype(auto)
__apply_to_tuple_impl(_ReturnAdapter __tr, _F __f, _Tuple& __t, std::index_sequence<_Ip...>)
{
    return __tr(__f(std::get<_Ip>(__t))...);
}

template <typename _F, typename _Tuple1, typename _Tuple2, std::size_t... _Ip>
void
__apply_to_tuples_impl(_F __f, _Tuple1& __t1, _Tuple2& __t2, std::index_sequence<_Ip...>)
{
    (__f(std::get<_Ip>(__t1), std::get<_Ip>(__t2)), ...);
}

struct __gen_lambda
{
    template <class... _Ts>
    decltype(auto) operator()(_Ts&&...) const {}
};

template <typename _F, typename _Tuple, typename _ReturnAdapter = __gen_lambda>
decltype(auto)
__apply_to_tuple(_F __f, _Tuple& __t, _ReturnAdapter __tr = {})
{
    return __apply_to_tuple_impl(__tr, __f, __t, std::make_index_sequence<std::tuple_size_v<_Tuple>>{});
}

template <typename _F, typename _Tuple1, typename _Tuple2>
void
__apply_to_tuples(_F __f, _Tuple1& __t1, _Tuple2& __t2)
{
    static_assert(std::tuple_size_v<_Tuple1> == std::tuple_size_v<_Tuple2>);

    __apply_to_tuples_impl(__f, __t1, __t2, std::make_index_sequence<std::tuple_size_v<_Tuple1>>{});
}

} //namespace __internal

template <std::ranges::input_range... _Views>
    requires((std::ranges::view<_Views> && ...) && (sizeof...(_Views) > 0))
class zip_view : public std::ranges::view_interface<zip_view<_Views...>>
{
    template <typename... Types>
    using __tuple_type = oneapi::dpl::__internal::tuple<Types...>;

  public:
    zip_view() = default;
    constexpr explicit zip_view(_Views... __views) : __views(std::move(__views)...) {}

    //forward declaration of sentinel classes 
    template <bool _Const>
    class sentinel;

    template <bool _Const>
    class iterator : public __internal::__declare_iterator_category<_Const, _Views...>
    {
      public:
        using iterator_concept = std::conditional_t<
            __internal::__all_random_access<_Const, _Views...>, std::random_access_iterator_tag,
            std::conditional_t<__internal::__all_bidirectional<_Const, _Views...>, std::bidirectional_iterator_tag,
                               std::conditional_t<__internal::__all_forward<_Const, _Views...>,
                                                  std::forward_iterator_tag, std::input_iterator_tag>>>;

        using value_type = __tuple_type<std::ranges::range_value_t<__internal::__maybe_const<_Const, _Views>>...>;
        using difference_type =
            std::common_type_t<std::ranges::range_difference_t<__internal::__maybe_const<_Const, _Views>>...>;

      private:
        using __reference_type =
            __tuple_type<std::ranges::range_reference_t<__internal::__maybe_const<_Const, _Views>>...>;
        using __rvalue_reference_type =
            __tuple_type<std::ranges::range_rvalue_reference_t<__internal::__maybe_const<_Const, _Views>>...>;

        using __iterators_type = __tuple_type<std::ranges::iterator_t<__internal::__maybe_const<_Const, _Views>>...>;

      public:
        iterator() = default;
        constexpr iterator(iterator<!_Const> i)
            requires _Const &&
                     (std::convertible_to<std::ranges::iterator_t<_Views>,
                                          std::ranges::iterator_t<__internal::__maybe_const<_Const, _Views>>> &&
                      ...)
            : __current(std::move(i.__current))
        {
        }

      private:
        constexpr explicit iterator(__iterators_type __current) : __current(std::move(__current)) {}

      public:
        template <typename... Iterators>
        operator oneapi::dpl::zip_iterator<Iterators...>() const
        {
            auto __tr = [](auto&&... __args) -> decltype(auto) {
                return oneapi::dpl::make_zip_iterator(std::forward<decltype(__args)>(__args)...);
            };
            return __internal::__apply_to_tuple([](auto it) -> decltype(auto) { return it; }, __current, __tr);
        }

        constexpr decltype(auto)
        operator*() const
        {
            auto __tr = [](auto&&... __args) -> decltype(auto) {
                return __reference_type(std::forward<decltype(__args)>(__args)...);
            };
            return __internal::__apply_to_tuple([](auto& __it) -> decltype(auto) { return *__it; }, __current, __tr);
        }

        constexpr decltype(auto)
        operator[](difference_type __n) const
            requires __internal::__all_random_access<_Const, _Views...>
        {
            return *(*this + __n);
        }

        constexpr iterator&
        operator++()
        {
            __internal::__apply_to_tuple([](auto& __it) -> decltype(auto) { return ++__it; }, __current);
            return *this;
        }

        constexpr void
        operator++(int)
        {
            ++*this;
        }

        constexpr iterator
        operator++(int)
            requires __internal::__all_forward<_Const, _Views...>
        {
            auto __tmp = *this;
            ++*this;
            return __tmp;
        }

        constexpr iterator&
        operator--()
            requires __internal::__all_bidirectional<_Const, _Views...>
        {
            __internal::__apply_to_tuple([](auto& __it) { return --__it; }, __current);
            return *this;
        }

        constexpr iterator
        operator--(int)
            requires __internal::__all_bidirectional<_Const, _Views...>
        {
            auto __tmp = *this;
            --*this;
            return __tmp;
        }

        constexpr iterator&
        operator+=(difference_type __n)
            requires __internal::__all_random_access<_Const, _Views...>
        {
            __internal::__apply_to_tuple([__n](auto& __it) { return __it += __n; }, __current);
            return *this;
        }

        constexpr iterator&
        operator-=(difference_type __n)
            requires __internal::__all_random_access<_Const, _Views...>
        {
            __internal::__apply_to_tuple([__n](auto& __it) { return __it -= __n; }, __current);
            return *this;
        }

        friend constexpr bool
        operator==(const iterator& __x, const iterator& __y)
            requires(std::equality_comparable<std::ranges::iterator_t<__internal::__maybe_const<_Const, _Views>>> &&
                     ...)
        {
            if constexpr (__internal::__all_bidirectional<_Const, _Views...>)
            {
                return __x.__current == __y.__current;
            }
            else
            {
                return __x.__compare_equal(__y, std::make_index_sequence<sizeof...(_Views)>());
            }
        }

        template <bool _OtherConst>
            requires(std::sentinel_for<std::ranges::sentinel_t<__internal::__maybe_const<_OtherConst, _Views>>,
                                       std::ranges::iterator_t<__internal::__maybe_const<_Const, _Views>>> &&
                     ...)
        friend constexpr bool
        operator==(const iterator& __x, const sentinel<_OtherConst>& __y)
        {
            return __x.__compare_with_sentinels(__y.__end, std::make_index_sequence<sizeof...(_Views)>());
        }

        friend constexpr auto
        operator<=>(const iterator& __x, const iterator& __y)
            requires __internal::__all_random_access<_Const, _Views...>
        {
            if (__x.__current < __y.__current)
                return std::weak_ordering::less;
            else if (__x.__current == __y.__current)
                return std::weak_ordering::equivalent;
            return std::weak_ordering::greater; //__x.current > __y.__current
        }

        friend constexpr auto
        operator-(const iterator& __x, const iterator& __y)
            requires(std::sized_sentinel_for<std::ranges::iterator_t<__internal::__maybe_const<_Const, _Views>>,
                                             std::ranges::iterator_t<__internal::__maybe_const<_Const, _Views>>> &&
                     ...)
        {
            auto __calc_val = [&]<std::size_t... In>(std::index_sequence<In...>) {
                return std::ranges::min({difference_type(std::get<In>(__x.__current) - std::get<In>(__y.__current))...},
                                        std::less{}, [](auto __a) { return std::abs(__a); });
            };

            return __calc_val(std::make_index_sequence<sizeof...(_Views)>());
        }

        template <bool _OtherConst>
            requires(std::sized_sentinel_for<std::ranges::sentinel_t<__internal::__maybe_const<_OtherConst, _Views>>,
                                             std::ranges::iterator_t<__internal::__maybe_const<_Const, _Views>>> &&
                     ...)
        friend constexpr difference_type
        operator-(const iterator& __x, const sentinel<_OtherConst>& __y)
        {
            auto calc_val = [&]<std::size_t... _In>(std::index_sequence<_In...>) {
                return std::ranges::min({difference_type(std::get<_In>(__x.__current) - std::get<_In>(__y.__end))...},
                                        std::less{}, [](auto __a) { return std::abs(__a); });
            };

            return calc_val(std::make_index_sequence<sizeof...(_Views)>());
        }

        template <bool _OtherConst>
            requires(std::sized_sentinel_for<std::ranges::sentinel_t<__internal::__maybe_const<_OtherConst, _Views>>,
                                             std::ranges::iterator_t<__internal::__maybe_const<_Const, _Views>>> &&
                     ...)
        friend constexpr difference_type
        operator-(const sentinel<_OtherConst>& __y, const iterator& __x)
        {
            return -(__x - __y);
        }

        friend constexpr iterator
        operator+(iterator __it, difference_type __n)
            requires __internal::__all_random_access<_Const, _Views...>
        {
            return __it += __n;
        }

        friend constexpr iterator
        operator+(difference_type __n, iterator __it)
            requires __internal::__all_random_access<_Const, _Views...>
        {
            return __it += __n;
        }

        friend constexpr iterator
        operator-(iterator __it, difference_type __n)
            requires __internal::__all_random_access<_Const, _Views...>
        {
            return __it -= __n;
        }

        friend constexpr decltype(auto)
        iter_move(const iterator& __x) noexcept(
            (noexcept(std::ranges::iter_move(
                 std::declval<const std::ranges::iterator_t<__internal::__maybe_const<_Const, _Views>>&>())) &&
             ...) &&
            (std::is_nothrow_move_constructible_v<
                 std::ranges::range_rvalue_reference_t<__internal::__maybe_const<_Const, _Views>>> &&
             ...))
        {
            auto __tr = [](auto&&... __args) -> decltype(auto) {
                return __rvalue_reference_type(std::forward<decltype(__args)>(__args)...);
            };
            return __internal::__apply_to_tuple(std::ranges::iter_move, __x.__current, __tr);
        }

        friend constexpr void
        iter_swap(const iterator& __x, const iterator& __y) noexcept(
            (noexcept(std::ranges::iter_swap(
                 std::declval<const std::ranges::iterator_t<__internal::__maybe_const<_Const, _Views>>&>(),
                 std::declval<const std::ranges::iterator_t<__internal::__maybe_const<_Const, _Views>>&>())) &&
             ...))
            requires(std::indirectly_swappable<std::ranges::iterator_t<__internal::__maybe_const<_Const, _Views>>> &&
                     ...)
        {
            __internal::__apply_to_tuples(std::ranges::iter_swap, __x.__current, __y.__current);
        }

        template <std::size_t... _In>
        constexpr bool
        __compare_equal(iterator __y, std::index_sequence<_In...>) const
        {
            return ((std::get<_In>(__current) == std::get<_In>(__y.__current)) || ...);
        }

        template <typename _SentinelsTuple, std::size_t... _In>
        constexpr bool
        __compare_with_sentinels(const _SentinelsTuple& __sentinels, std::index_sequence<_In...>) const
        {
            return ((std::get<_In>(__current) == std::get<_In>(__sentinels)) || ...);
        }

        friend class zip_view;

        __iterators_type __current;
    }; // class iterator

    template <bool _Const>
    class sentinel
    {
        using difference_type =
            std::common_type_t<std::ranges::range_difference_t<__internal::__maybe_const<_Const, _Views>>...>;

        template <typename... Sentinels>
        constexpr explicit sentinel(Sentinels... sentinels) : __end(std::move(sentinels)...)
        {
        }

      public:
        sentinel() = default;
        constexpr sentinel(sentinel<!_Const> i)
            requires _Const &&
                     (std::convertible_to<std::ranges::sentinel_t<_Views>,
                                          std::ranges::sentinel_t<__internal::__maybe_const<_Const, _Views>>> &&
                      ...)
            : __end(std::move(i.__end))
        {
        }        

      private:
        friend class zip_view;

        __tuple_type<std::ranges::sentinel_t<__internal::__maybe_const<_Const, _Views>>...> __end;
    }; // class sentinel

    constexpr auto
    begin()
        requires(!(__internal::__simple_view<_Views> && ...))
    {
        using __iterators_type = __tuple_type<std::ranges::iterator_t<__internal::__maybe_const<false, _Views>>...>;
        auto __tr = [](auto&&... __args) {
            return iterator<false>(__iterators_type(std::forward<decltype(__args)>(__args)...));
        };
        return __internal::__apply_to_tuple(std::ranges::begin, __views, __tr);
    }

    constexpr auto
    begin() const
        requires(std::ranges::range<const _Views> && ...)
    {
        using __iterators_type = __tuple_type<std::ranges::iterator_t<__internal::__maybe_const<true, _Views>>...>;
        auto __tr = [](auto&&... __args) {
            return iterator<true>(__iterators_type(std::forward<decltype(__args)>(__args)...));
        };
        return __internal::__apply_to_tuple(std::ranges::begin, __views, __tr);
    }

    constexpr auto
    end()
        requires(!(__internal::__simple_view<_Views> && ...))
    {
        if constexpr (!__internal::__zip_is_common<_Views...>)
        {
            auto __tr = [](auto&&... __args) { return sentinel<false>(std::forward<decltype(__args)>(__args)...); };
            return __internal::__apply_to_tuple(std::ranges::end, __views, __tr);
        }
        else if constexpr ((std::ranges::random_access_range<_Views> && ...))
        {
            auto __it = begin();
            __it += size();
            return __it;
        }
        else
        {
            using __iterators_type = __tuple_type<std::ranges::iterator_t<__internal::__maybe_const<false, _Views>>...>;
            auto __tr = [](auto&&... __args) {
                return iterator<false>(__iterators_type(std::forward<decltype(__args)>(__args)...));
            };
            return __internal::__apply_to_tuple(std::ranges::end, __views, __tr);
        }
    }

    constexpr auto
    end() const
        requires(std::ranges::range<const _Views> && ...)
    {
        if constexpr (!__internal::__zip_is_common<_Views...>)
        {
            auto __tr = [](auto&&... __args) { return sentinel<true>(std::forward<decltype(__args)>(__args)...); };
            return __internal::__apply_to_tuple(std::ranges::end, __views, __tr);
        }
        else if constexpr ((std::ranges::random_access_range<_Views> && ...))
        {
            auto __it = begin();
            __it += size();
            return __it;
        }
        else
        {
            using __iterators_type = __tuple_type<std::ranges::iterator_t<__internal::__maybe_const<true, _Views>>...>;
            auto __tr = [](auto&&... __args) { return iterator<true>(__iterators_type(__args...)); };
            return __internal::__apply_to_tuple(std::ranges::end, __views, __tr);
        }
    }

    constexpr auto
    size()
        requires(std::ranges::sized_range<_Views> && ...)
    {
        auto __tr = [](auto... __args) {
            using CT = std::make_unsigned_t<std::common_type_t<decltype(__args)...>>;
            return std::ranges::min({CT(__args)...});
        };

        return __internal::__apply_to_tuple(std::ranges::size, __views, __tr);
    }

    constexpr auto
    size() const
        requires(std::ranges::sized_range<const _Views> && ...)
    {
        auto __tr = [](auto... __args) {
            using CT = std::make_unsigned_t<std::common_type_t<decltype(__args)...>>;
            return std::ranges::min({CT(__args)...});
        };

        return __internal::__apply_to_tuple(std::ranges::size, __views, __tr);
    }

  private:
    __tuple_type<_Views...> __views;
}; // class zip_view

template <typename... Rs>
zip_view(Rs&&...) -> zip_view<std::views::all_t<Rs>...>;

namespace __internal
{
struct zip_fn
{
    template <class... _Ranges>
    constexpr auto
    operator()(_Ranges&&... __rs) const noexcept(
        noexcept(oneapi::dpl::ranges::zip_view<std::views::all_t<_Ranges&&>...>(std::forward<_Ranges>(__rs)...)))
        -> decltype(oneapi::dpl::ranges::zip_view<std::views::all_t<_Ranges&&>...>(std::forward<_Ranges>(__rs)...))
    {
        return oneapi::dpl::ranges::zip_view<std::views::all_t<_Ranges>...>(std::forward<_Ranges>(__rs)...);
    }

    constexpr auto
    operator()() const noexcept
    {
        return std::ranges::empty_view<oneapi::dpl::__internal::tuple<>>{};
    }
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
