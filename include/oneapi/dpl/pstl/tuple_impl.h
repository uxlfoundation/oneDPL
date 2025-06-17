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

#ifndef _ONEDPL_TUPLE_IMPL_H
#define _ONEDPL_TUPLE_IMPL_H

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
template <typename... T>
struct tuple;
} // namespace __internal
} // namespace dpl
} // namespace oneapi

namespace std
{
template <::std::size_t N, typename T, typename... Rest>
struct tuple_element<N, oneapi::dpl::__internal::tuple<T, Rest...>>
    : tuple_element<N - 1, oneapi::dpl::__internal::tuple<Rest...>>
{
};

template <typename T, typename... Rest>
struct tuple_element<0, oneapi::dpl::__internal::tuple<T, Rest...>>
{
    using type = T;
};

//forward declare new std::get<I>() functions so they can be used in impl
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

template <typename... Args>
struct tuple_size<oneapi::dpl::__internal::tuple<Args...>> : ::std::integral_constant<::std::size_t, sizeof...(Args)>
{
};
} // namespace std

//custom tuple utilities
namespace oneapi
{
namespace dpl
{
namespace __internal
{
template <typename... Size>
struct get_value_by_idx;

template <typename T1, typename... T, ::std::size_t... indices>
::std::tuple<T...>
get_tuple_tail_impl(const ::std::tuple<T1, T...>& t, const ::std::index_sequence<indices...>&)
{
    return ::std::tuple<T...>(::std::get<indices + 1>(t)...);
}

template <typename T1, typename... T>
::std::tuple<T...>
get_tuple_tail(const ::std::tuple<T1, T...>& other)
{
    return oneapi::dpl::__internal::get_tuple_tail_impl(other, ::std::make_index_sequence<sizeof...(T)>());
}

// Maps an incoming type for tuplewrapper to simplify tuple-related handling.
// as it doesn't work well with rvalue refs.
// T& -> T&, T&& -> T, T -> T
template <typename _Tp, bool = ::std::is_lvalue_reference_v<_Tp>>
struct __lvref_or_val
{
    using __type = _Tp&&;
};

template <typename _Tp>
struct __lvref_or_val<_Tp, false>
{
    using __type = ::std::remove_reference_t<_Tp>;
};

template <typename T>
struct MapValue
{
    T id;
    template <typename T1>
    auto
    operator()(const T1& t1) -> decltype(t1[id]) const
    {
        return t1[id];
    }
};

// It serves the same purpose as tuplewrapper in iterator_impl.h, but in this case we don't need
// swap, so we can simply map it to tuple adjusting the types
template <typename... T>
using tuplewrapper = oneapi::dpl::__internal::tuple<typename oneapi::dpl::__internal::__lvref_or_val<T>::__type...>;

// __internal::make_tuple
template <typename... T>
constexpr oneapi::dpl::__internal::tuple<T...>
make_tuple(T... args)
{
    return oneapi::dpl::__internal::tuple<T...>{args...};
}

// __internal::make_tuplewrapper
template <typename... T>
oneapi::dpl::__internal::tuplewrapper<T&&...>
make_tuplewrapper(T&&... t)
{
    return {::std::forward<T>(t)...};
}

template <size_t N>
struct get_impl
{
    template <typename... T>
    constexpr auto
    operator()(oneapi::dpl::__internal::tuple<T...>& t) const -> decltype(get_impl<N - 1>()(t.next))
    {
        return get_impl<N - 1>()(t.next);
    }

    template <typename... T>
    constexpr auto
    operator()(const oneapi::dpl::__internal::tuple<T...>& t) const -> decltype(get_impl<N - 1>()(t.next))
    {
        return get_impl<N - 1>()(t.next);
    }

    template <typename... T>
    constexpr auto
    operator()(oneapi::dpl::__internal::tuple<T...>&& t) const -> decltype(get_impl<N - 1>()(::std::move(t.next)))
    {
        return get_impl<N - 1>()(::std::move(t.next));
    }

    template <typename... T>
    constexpr auto
    operator()(const oneapi::dpl::__internal::tuple<T...>&& t) const -> decltype(get_impl<N - 1>()(::std::move(t.next)))
    {
        return get_impl<N - 1>()(::std::move(t.next));
    }
};

template <>
struct get_impl<0>
{
    template <typename... T>
    using ret_type = ::std::tuple_element_t<0, oneapi::dpl::__internal::tuple<T...>>;

    template <typename... T>
    constexpr ret_type<T...>&
    operator()(oneapi::dpl::__internal::tuple<T...>& t) const
    {
        return t.holder.value;
    }

    template <typename... T>
    constexpr const ret_type<T...>&
    operator()(const oneapi::dpl::__internal::tuple<T...>& t) const
    {
        return t.holder.value;
    }

    template <typename... T>
    constexpr ret_type<T...>&&
    operator()(oneapi::dpl::__internal::tuple<T...>&& t) const
    {
        return ::std::forward<ret_type<T...>&&>(t.holder.value);
    }

    template <typename... T>
    constexpr const ret_type<T...>&&
    operator()(const oneapi::dpl::__internal::tuple<T...>&& t) const
    {
        return ::std::forward<const ret_type<T...>&&>(t.holder.value);
    }
};

// __internal::map_tuple
template <size_t I, typename F, typename... T>
auto
apply_to_tuple(F f, T... in) -> decltype(f(oneapi::dpl::__internal::get_impl<I>()(in)...))
{
    return f(oneapi::dpl::__internal::get_impl<I>()(in)...);
}

struct make_inner_tuple_functor
{
    template <typename... Args>
    auto
    operator()(Args&&... args) const -> decltype(oneapi::dpl::__internal::make_tuple(::std::forward<Args>(args)...))
    {
        return oneapi::dpl::__internal::make_tuple(::std::forward<Args>(args)...);
    }
};

struct make_tuplewrapper_functor
{
    template <typename... Args>
    auto
    operator()(Args&&... args) const
        -> decltype(oneapi::dpl::__internal::make_tuplewrapper(::std::forward<Args>(args)...))
    {
        return oneapi::dpl::__internal::make_tuplewrapper(::std::forward<Args>(args)...);
    }
};

template <typename MakeTupleF, typename F, size_t... indices, typename... T>
auto
map_tuple_impl(MakeTupleF mtf, F f, ::std::index_sequence<indices...>, T... in)
    -> decltype(mtf(oneapi::dpl::__internal::apply_to_tuple<indices>(f, in...)...))
{
    return mtf(oneapi::dpl::__internal::apply_to_tuple<indices>(f, in...)...);
}

//
template <typename F, template <typename...> class TBig, typename... T, typename... RestTuples>
auto
map_tuple(F f, TBig<T...> in, RestTuples... rest)
    -> decltype(oneapi::dpl::__internal::map_tuple_impl(oneapi::dpl::__internal::make_inner_tuple_functor{}, f,
                                                        ::std::make_index_sequence<sizeof...(T)>(), in, rest...))
{
    return oneapi::dpl::__internal::map_tuple_impl(oneapi::dpl::__internal::make_inner_tuple_functor{}, f,
                                                   ::std::make_index_sequence<sizeof...(T)>(), in, rest...);
}

// Functions are needed to call get_value_by_idx: it requires to store in tuple wrapper
template <typename F, template <typename...> class TBig, typename... T, typename... RestTuples>
auto
map_tuplewrapper(F f, TBig<T...> in, RestTuples... rest)
    -> decltype(oneapi::dpl::__internal::map_tuple_impl(oneapi::dpl::__internal::make_tuplewrapper_functor{}, f,
                                                        ::std::make_index_sequence<sizeof...(T)>(), in, rest...))
{
    return oneapi::dpl::__internal::map_tuple_impl(oneapi::dpl::__internal::make_tuplewrapper_functor{}, f,
                                                   ::std::make_index_sequence<sizeof...(T)>(), in, rest...);
}

template <typename _Tp>
struct __value_holder
{
    __value_holder() = default;
    template <typename _Up>
    __value_holder(_Up&& t) : value(::std::forward<_Up>(t))
    {
    }
    _Tp value;
};

// Necessary to make tuple trivially_copy_assignable. This type decided
// if it's needed to have user-defined operator=.
template <typename _Tp, bool = ::std::is_trivially_copy_assignable_v<oneapi::dpl::__internal::__value_holder<_Tp>>>
struct __copy_assignable_holder : oneapi::dpl::__internal::__value_holder<_Tp>
{
    using oneapi::dpl::__internal::__value_holder<_Tp>::__value_holder;
};

template <typename _Tp>
struct __copy_assignable_holder<_Tp, false> : oneapi::dpl::__internal::__value_holder<_Tp>
{
    using oneapi::dpl::__internal::__value_holder<_Tp>::__value_holder;
    __copy_assignable_holder() = default;
    __copy_assignable_holder(const __copy_assignable_holder&) = default;
    __copy_assignable_holder(__copy_assignable_holder&&) = default;
    __copy_assignable_holder&
    operator=(const __copy_assignable_holder& other)
    {
        this->value = other.value;
        return *this;
    }
    __copy_assignable_holder&
    operator=(__copy_assignable_holder&& other) = default;
};

template <typename _Tuple1, typename _Tuple2, int I = 0>
constexpr bool
__equal(_Tuple1&& __lhs, _Tuple2&& __rhs)
{
    static_assert(std::tuple_size_v<std::decay_t<_Tuple1>> == std::tuple_size_v<std::decay_t<_Tuple2>>,
                  "Tuples must have the same size to be equality compared");
    if constexpr (I < std::tuple_size_v<std::decay_t<_Tuple1>>)
    {
        return std::get<I>(__lhs) == std::get<I>(__rhs) &&
               oneapi::dpl::__internal::__equal<_Tuple1, _Tuple2, I + 1>(std::forward<_Tuple1>(__lhs),
                                                                         std::forward<_Tuple2>(__rhs));
    }
    else
    {
        return true;
    }
}
template <typename _Tuple1, typename _Tuple2, int I = 0>
constexpr bool
__less(_Tuple1&& __lhs, _Tuple2&& __rhs)
{
    static_assert(std::tuple_size_v<std::decay_t<_Tuple1>> == std::tuple_size_v<std::decay_t<_Tuple2>>,
                  "Tuples must have the same size to be compared");
    if constexpr (I < std::tuple_size_v<std::decay_t<_Tuple1>>)
    {
        return std::get<I>(__lhs) < std::get<I>(__rhs) ||
               (!(std::get<I>(__rhs) < std::get<I>(__lhs)) &&
                oneapi::dpl::__internal::__less<_Tuple1, _Tuple2, I + 1>(std::forward<_Tuple1>(__lhs),
                                                                         std::forward<_Tuple2>(__rhs)));
    }
    else
    {
        return false;
    }
}

template <typename T>
struct __is_internal_tuple : std::false_type
{
};

template <typename... Ts>
struct __is_internal_tuple<oneapi::dpl::__internal::tuple<Ts...>> : std::true_type
{
};

template <typename T>
inline constexpr bool __is_internal_tuple_v = __is_internal_tuple<T>::value;

template <typename T>
struct __is_std_tuple : std::false_type
{
};

template <typename... Ts>
struct __is_std_tuple<std::tuple<Ts...>> : std::true_type
{
};

template <typename T>
inline constexpr bool __is_std_tuple_v = __is_std_tuple<T>::value;

// Either LHS = this internal tuple type + RHS = any compatible tuple type
// or     LHS = std::tuple               + RHS = this internal tuple type
// Note: We do not allow LHS to be "any compatible type" with a RHS as this internal tuple type because then we would
//       have ambiguity comparing two different compatible internal tuple types, because both instantiations would
//       create the necessary comparison. As written, the type instantiation of the type in the LHS is always
//       responsible for creating this comparison operator.
template <typename _ThisTuple, typename _U, typename _V>
inline constexpr bool __enable_comparison_op_v =
    (std::is_same_v<std::decay_t<_ThisTuple>, std::decay_t<_U>> &&                         //LHS +
                    (oneapi::dpl::__internal::__is_std_tuple_v<std::decay_t<_V>> ||        //RHS option 1
                     oneapi::dpl::__internal::__is_internal_tuple_v<std::decay_t<_V>>)) || //RHS option 2  - OR -
    (oneapi::dpl::__internal::__is_std_tuple_v<std::decay_t<_U>> &&                        //LHS +
                    std::is_same_v<std::decay_t<_ThisTuple>, std::decay_t<_V>>);           //RHS

template <typename T1, typename... T>
struct tuple<T1, T...>
{
    oneapi::dpl::__internal::__copy_assignable_holder<T1> holder;
    tuple<T...> next;

    using tuple_type = ::std::tuple<T1, T...>;

    template <::std::size_t I>
    constexpr auto
    get() & -> decltype(get_impl<I>()(*this))
    {
        return get_impl<I>()(*this);
    }

    template <::std::size_t I>
    constexpr auto
    get() const& -> decltype(get_impl<I>()(*this))
    {
        return get_impl<I>()(*this);
    }

    template <::std::size_t I>
    constexpr auto
    get() && -> decltype(get_impl<I>()(::std::move(*this)))
    {
        return get_impl<I>()(::std::move(*this));
    }

    template <::std::size_t I>
    constexpr auto
    get() const&& -> decltype(get_impl<I>()(::std::move(*this)))
    {
        return get_impl<I>()(::std::move(*this));
    }

    tuple() = default;
    tuple(const tuple& other) = default;
    tuple(tuple&& other) = default;
    template <typename _U1, typename... _U, typename = ::std::enable_if_t<(sizeof...(_U) == sizeof...(T))>>
    tuple(const tuple<_U1, _U...>& other) : holder(other.template get<0>()), next(other.next)
    {
    }

    template <typename _U1, typename... _U, typename = ::std::enable_if_t<(sizeof...(_U) == sizeof...(T))>>
    tuple(tuple<_U1, _U...>&& other) : holder(std::move(other).template get<0>()), next(std::move(other.next))
    {
    }

    template <typename _U1, typename... _U,
              typename = ::std::enable_if_t<
                  (sizeof...(_U) == sizeof...(T) &&
                   ::std::conjunction_v<::std::is_constructible<T1, _U1&&>, ::std::is_constructible<T, _U&&>...>)>>
    tuple(_U1&& _value, _U&&... _next) : holder(::std::forward<_U1>(_value)), next(::std::forward<_U>(_next)...)
    {
    }

    // required to convert ::std::tuple to inner tuple in user-provided functor
    tuple(const ::std::tuple<T1, T...>& other)
        : holder(::std::get<0>(other)), next(oneapi::dpl::__internal::get_tuple_tail(other))
    {
    }

    // conversion to ::std::tuple with the same template arguments
    operator ::std::tuple<T1, T...>() const
    {
        static constexpr ::std::size_t __tuple_size = sizeof...(T) + 1;
        return to_std_tuple(*this, ::std::make_index_sequence<__tuple_size>());
    }

    // conversion to ::std::tuple with the different template arguments
    template <typename U1, typename... U>
    operator ::std::tuple<U1, U...>() const
    {
        constexpr ::std::size_t __tuple_size = sizeof...(T) + 1;
        return to_std_tuple(static_cast<tuple<U1, U...>>(*this), ::std::make_index_sequence<__tuple_size>());
    }

    // non-const subscript operator with tuple argument
    template <typename Size1, typename... SizeRest>
    auto operator[](oneapi::dpl::__internal::tuple<Size1, SizeRest...> tuple_size)
        -> decltype(oneapi::dpl::__internal::get_value_by_idx<Size1, SizeRest...>()(*this, tuple_size))
    {
        return oneapi::dpl::__internal::get_value_by_idx<Size1, SizeRest...>()(*this, tuple_size);
    }

    // const subscript operator with tuple argument
    template <typename Size1, typename... SizeRest>
    auto operator[](const oneapi::dpl::__internal::tuple<Size1, SizeRest...> tuple_size) const
        -> decltype(oneapi::dpl::__internal::get_value_by_idx<Size1, SizeRest...>()(*this, tuple_size))
    {
        return oneapi::dpl::__internal::get_value_by_idx<Size1, SizeRest...>()(*this, tuple_size);
    }

    // non-const subscript operator with scalar argument
    template <typename Idx>
    auto operator[](Idx idx)
        -> decltype(oneapi::dpl::__internal::map_tuplewrapper(oneapi::dpl::__internal::MapValue<Idx>{idx}, *this))
    {
        return oneapi::dpl::__internal::map_tuplewrapper(oneapi::dpl::__internal::MapValue<Idx>{idx}, *this);
    }

    // const subscript operator with scalar argument
    template <typename Idx>
    auto operator[](Idx idx) const
        -> decltype(oneapi::dpl::__internal::map_tuplewrapper(oneapi::dpl::__internal::MapValue<Idx>{idx}, *this))
    {
        return oneapi::dpl::__internal::map_tuplewrapper(oneapi::dpl::__internal::MapValue<Idx>{idx}, *this);
    }

    template <typename U1, typename... U>
    tuple&
    operator=(const tuple<U1, U...>& other)
    {
        holder.value = other.holder.value;
        next = other.next;
        return *this;
    }

    // if T1 is deduced with reference, compiler generates deleted operator= and,
    // since "template operator=" is not considered as operator= overload
    // the deleted operator= has a preference during lookup
    tuple&
    operator=(const tuple<T1, T...>& other) = default;

    // for cases when we assign ::std::tuple to __internal::tuple
    template <typename U1, typename... U>
    tuple&
    operator=(const ::std::tuple<U1, U...>& other)
    {
        holder.value = ::std::get<0>(other);
        next = oneapi::dpl::__internal::get_tuple_tail(other);
        return *this;
    }

    template <typename _U, typename _V,
              std::enable_if_t<oneapi::dpl::__internal::__enable_comparison_op_v<tuple, _U, _V>, int> = 0>
    friend constexpr bool
    operator==(_U&& __lhs, _V&& __rhs)
    {
        return oneapi::dpl::__internal::__equal(std::forward<_U>(__lhs), std::forward<_V>(__rhs));
    }

    template <typename _U, typename _V,
              std::enable_if_t<oneapi::dpl::__internal::__enable_comparison_op_v<tuple, _U, _V>, int> = 0>
    friend constexpr bool
    operator!=(_U&& __lhs, _V&& __rhs)
    {
        return !oneapi::dpl::__internal::__equal(std::forward<_U>(__lhs), std::forward<_V>(__rhs));
    }

    template <typename _U, typename _V,
              std::enable_if_t<oneapi::dpl::__internal::__enable_comparison_op_v<tuple, _U, _V>, int> = 0>
    friend constexpr bool
    operator<(_U&& __lhs, _V&& __rhs)
    {
        return oneapi::dpl::__internal::__less(std::forward<_U>(__lhs), std::forward<_V>(__rhs));
    }

    template <typename _U, typename _V,
              std::enable_if_t<oneapi::dpl::__internal::__enable_comparison_op_v<tuple, _U, _V>, int> = 0>
    friend constexpr bool
    operator<=(_U&& __lhs, _V&& __rhs)
    {
        return !oneapi::dpl::__internal::__less(std::forward<_V>(__rhs), std::forward<_U>(__lhs));
    }

    template <typename _U, typename _V,
              std::enable_if_t<oneapi::dpl::__internal::__enable_comparison_op_v<tuple, _U, _V>, int> = 0>
    friend constexpr bool
    operator>(_U&& __lhs, _V&& __rhs)
    {
        return oneapi::dpl::__internal::__less(std::forward<_V>(__rhs), std::forward<_U>(__lhs));
    }

    template <typename _U, typename _V,
              std::enable_if_t<oneapi::dpl::__internal::__enable_comparison_op_v<tuple, _U, _V>, int> = 0>
    friend constexpr bool
    operator>=(_U&& __lhs, _V&& __rhs)
    {
        return !oneapi::dpl::__internal::__less(std::forward<_U>(__lhs), std::forward<_V>(__rhs));
    }

    template <typename U1, typename... U, ::std::size_t... _Ip>
    static ::std::tuple<U1, U...>
    to_std_tuple(const oneapi::dpl::__internal::tuple<U1, U...>& __t, ::std::index_sequence<_Ip...>)
    {
        return ::std::tuple<U1, U...>(oneapi::dpl::__internal::get_impl<_Ip>()(__t)...);
    }
};

template <>
struct tuple<>
{
    using tuple_type = ::std::tuple<>;
    // since compiler does not autogenerate ctors
    // if user defines its own, we have to define them too
    tuple() = default;
    tuple(const tuple& other) = default;

    tuple(const ::std::tuple<>&) {}

    tuple operator[](tuple) { return {}; }
    tuple operator[](const tuple&) const { return {}; }
    operator tuple_type() const { return tuple_type{}; }
    tuple&
    operator=(const tuple&) = default;
    tuple&
    operator=(const ::std::tuple<>& /*other*/)
    {
        return *this;
    }
    friend constexpr bool
    operator==(const tuple&, const tuple&)
    {
        return true;
    }
    friend constexpr bool
    operator!=(const tuple&, const tuple&)
    {
        return false;
    }
    friend constexpr bool
    operator<(const tuple&, const tuple&)
    {
        return false;
    }
    friend constexpr bool
    operator<=(const tuple&, const tuple&)
    {
        return true;
    }
    friend constexpr bool
    operator>(const tuple&, const tuple&)
    {
        return false;
    }
    friend constexpr bool
    operator>=(const tuple&, const tuple&)
    {
        return true;
    }
};

inline void
swap(oneapi::dpl::__internal::tuple<>& /*__x*/, oneapi::dpl::__internal::tuple<>& /*__y*/)
{
}

template <typename... _T>
void
swap(oneapi::dpl::__internal::tuple<_T...>& __x, oneapi::dpl::__internal::tuple<_T...>& __y)
{
    using ::std::swap;
    swap(__x.holder.value, __y.holder.value);
    swap(__x.next, __y.next);
}

template <typename... _T>
void
swap(oneapi::dpl::__internal::tuple<_T...>&& __x, oneapi::dpl::__internal::tuple<_T...>&& __y)
{
    using ::std::swap;
    swap(__x.holder.value, __y.holder.value);
    swap(__x.next, __y.next);
}

// Get corresponding ::std::tuple for our internal tuple(i.e. access tuple_type member
// which is ::std::tuple<Ts...> for internal::tuple<Ts...>).
// Do nothing for other types or if both operands are internal tuples.
template <class _T, class>
struct __get_tuple_type
{
    using __type = _T;
};

template <class... _Ts, class... _Us>
struct __get_tuple_type<oneapi::dpl::__internal::tuple<_Ts...>, oneapi::dpl::__internal::tuple<_Us...>>
{
    using __type = typename oneapi::dpl::__internal::tuple<_Ts...>;
};

template <class... _Ts, class _Other>
struct __get_tuple_type<oneapi::dpl::__internal::tuple<_Ts...>, _Other>
{
    using __type = typename oneapi::dpl::__internal::tuple<_Ts...>::tuple_type;
};

template <typename Size>
struct AddIndexes
{
    Size idx;

    template <typename T1>
    T1
    operator()(const T1& t1) const
    {
        return t1 + idx;
    }
};

template <typename Size>
struct SubTupleFromIndex
{
    Size idx;

    template <typename T1>
    T1
    operator()(const T1& t1) const
    {
        return idx - t1;
    }
};

template <typename Size>
struct SubIndexFromTuple
{
    Size idx;

    template <typename T1>
    T1
    operator()(const T1& t1) const
    {
        return t1 - idx;
    }
};

struct subscription_functor
{
    template <typename _A, typename _Index>
    auto
    operator()(_A __a, _Index __idx) -> decltype(__a[__idx]) const
    {
        return __a[__idx];
    }
};

template <typename... Size>
struct get_value_by_idx
{
    template <typename... Acc>
    auto
    operator()(oneapi::dpl::__internal::tuple<Acc...>& acc, oneapi::dpl::__internal::tuple<Size...>& idx)
        -> decltype(oneapi::dpl::__internal::map_tuplewrapper(oneapi::dpl::__internal::subscription_functor{}, acc,
                                                              idx))
    {
        return oneapi::dpl::__internal::map_tuplewrapper(oneapi::dpl::__internal::subscription_functor{}, acc, idx);
    }
    template <typename... Acc>
    auto
    operator()(const oneapi::dpl::__internal::tuple<Acc...>& acc,
               const oneapi::dpl::__internal::tuple<Size...>& idx) const
        -> decltype(oneapi::dpl::__internal::map_tuplewrapper(oneapi::dpl::__internal::subscription_functor{}, acc,
                                                              idx))
    {
        return oneapi::dpl::__internal::map_tuplewrapper(oneapi::dpl::__internal::subscription_functor{}, acc, idx);
    }
};

template <typename Size, typename... T1>
oneapi::dpl::__internal::tuple<T1...>
operator+(const oneapi::dpl::__internal::tuple<T1...>& tuple1, Size idx)
{
    return oneapi::dpl::__internal::map_tuple(oneapi::dpl::__internal::AddIndexes<Size>{idx}, tuple1);
}

template <typename Size, typename... T1>
oneapi::dpl::__internal::tuple<T1...>
operator+(Size idx, const oneapi::dpl::__internal::tuple<T1...>& tuple1)
{
    return oneapi::dpl::__internal::map_tuple(oneapi::dpl::__internal::AddIndexes<Size>{idx}, tuple1);
}

template <typename Size, typename... T1>
oneapi::dpl::__internal::tuple<T1...>
operator-(const oneapi::dpl::__internal::tuple<T1...>& tuple1, Size idx)
{
    return oneapi::dpl::__internal::map_tuple(oneapi::dpl::__internal::SubIndexFromTuple<Size>{idx}, tuple1);
}

// required in scan implementation for false offset calculation
template <typename Size, typename... T1>
auto
operator-(Size idx, const oneapi::dpl::__internal::tuple<T1...>& tuple1)
    -> decltype(oneapi::dpl::__internal::map_tuple(oneapi::dpl::__internal::SubTupleFromIndex<Size>{idx}, tuple1))
{
    return oneapi::dpl::__internal::map_tuple(oneapi::dpl::__internal::SubTupleFromIndex<Size>{idx}, tuple1);
}

// Decay wrapper struct with specialization to decay tuple elements
template <typename _T>
struct __decay_with_tuple_specialization
{
    using type = ::std::decay_t<_T>;
};

template <typename... _Args>
struct __decay_with_tuple_specialization<oneapi::dpl::__internal::tuple<_Args...>>
{
    using type = oneapi::dpl::__internal::tuple<::std::decay_t<_Args>...>;
};

template <typename... _Args>
struct __decay_with_tuple_specialization<::std::tuple<_Args...>>
{
    using type = ::std::tuple<::std::decay_t<_Args>...>;
};

template <typename... _Args>
using __decay_with_tuple_specialization_t = typename __decay_with_tuple_specialization<_Args...>::type;

} // namespace __internal
} // namespace dpl
} // namespace oneapi

namespace std
{
template <size_t _Idx, typename... _Tp>
constexpr ::std::tuple_element_t<_Idx, oneapi::dpl::__internal::tuple<_Tp...>>&
get(oneapi::dpl::__internal::tuple<_Tp...>& __a)
{
    return __a.template get<_Idx>();
}

template <size_t _Idx, typename... _Tp>
constexpr ::std::tuple_element_t<_Idx, oneapi::dpl::__internal::tuple<_Tp...>> const&
get(const oneapi::dpl::__internal::tuple<_Tp...>& __a)
{
    return __a.template get<_Idx>();
}
template <size_t _Idx, typename... _Tp>
constexpr ::std::tuple_element_t<_Idx, oneapi::dpl::__internal::tuple<_Tp...>>&&
get(oneapi::dpl::__internal::tuple<_Tp...>&& __a)
{
    return ::std::move(__a).template get<_Idx>();
}

template <size_t _Idx, typename... _Tp>
constexpr ::std::tuple_element_t<_Idx, oneapi::dpl::__internal::tuple<_Tp...>> const&&
get(const oneapi::dpl::__internal::tuple<_Tp...>&& __a)
{
    return ::std::move(__a).template get<_Idx>();
}

// To enable oneapi::dpl::zip_iterator to satisfy the requirements for the std::input_iterator concept in C++20,
// we must provide a basic_common_reference specialization for our internal tuple implementation. std::basic_common_reference
// is introduced in __cpp_lib_concepts, so we must ensure this feature is present.
#if _ONEDPL_CPP20_CONCEPTS_PRESENT
template <typename... TTypes, typename... UTypes, template <typename> typename TQual,
          template <typename> typename UQual>
struct basic_common_reference<oneapi::dpl::__internal::tuple<TTypes...>, oneapi::dpl::__internal::tuple<UTypes...>,
                              TQual, UQual>
{
    using type = oneapi::dpl::__internal::tuple<::std::common_reference_t<TQual<TTypes>, UQual<UTypes>>...>;
};
#endif
} // namespace std

#endif // _ONEDPL_TUPLE_IMPL_H
