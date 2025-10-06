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

#ifndef _ONEDPL_UTILS_RANGES_H
#define _ONEDPL_UTILS_RANGES_H

#include <tuple>       // std::get
#include <cstdint>     // std::uint8_t
#include <cstddef>     // std::size_t, std::ptrdiff_t
#include <utility>     // std::declval
#include <iterator>    // std::iterator_traits
#include <type_traits> // std::decay_t, std::remove_cv_t, std::remove_reference_t, std::invoke_result_t, ...

#if _ONEDPL_CPP20_RANGES_PRESENT
#    include <ranges> // std::ranges::sized_range, std::ranges::range_size_t
#endif

#include "iterator_defs.h"
#include "iterator_impl.h"

namespace oneapi
{
namespace dpl
{

namespace __internal
{

template <typename _R>
auto
get_value_type(int) -> typename ::std::decay_t<_R>::value_type;

template <typename _R>
auto
get_value_type(long) ->
    typename ::std::iterator_traits<::std::decay_t<decltype(::std::declval<_R&>().begin())>>::value_type;

template <typename _It>
auto
get_value_type(long long) -> typename ::std::iterator_traits<_It>::value_type;

template <typename _R>
auto
get_value_type(...)
{
    //static_assert should always fail when this overload is chosen, so its condition must depend on
    //the template parameter and evaluate to false
    static_assert(std::is_same_v<_R, void>,
        "error: the range has no 'value_type'; define an alias or typedef named 'value_type' in the range class");
}

template <typename _R>
using __value_t = decltype(oneapi::dpl::__internal::get_value_type<_R>(0));

template <typename _Proj, typename _R>
using __key_t = ::std::remove_cv_t<::std::remove_reference_t<::std::invoke_result_t<_Proj&, __value_t<_R>>>>;

template <typename T, typename = void>
struct __range_has_raw_ptr_iterator : ::std::false_type
{
};

template <typename T>
struct __range_has_raw_ptr_iterator<T, ::std::void_t<decltype(::std::declval<T&>().begin())>>
    : ::std::is_pointer<decltype(::std::declval<T&>().begin())>
{
};

template <typename T>
inline constexpr bool __range_has_raw_ptr_iterator_v = __range_has_raw_ptr_iterator<T>::value;

#if _ONEDPL_CPP20_RANGES_PRESENT
//The following '__range_size' type trait should be used in only the context with std::common_type
//together with a sized range.
template <typename R>
struct __range_size {
    using type = std::uint8_t;
};

template <std::ranges::sized_range R>
struct __range_size<R> {
    using type = std::ranges::range_size_t<R>;
};

template <typename _R>
using __range_size_t = typename __range_size<_R>::type;

#endif //_ONEDPL_CPP20_RANGES_PRESENT

template <typename _R>
auto
__check_size(int) -> decltype(std::declval<_R&>().size());

template <typename _R>
auto
__check_size(long) -> decltype(std::declval<_R&>().get_count());

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _R>
auto
__check_size(long long) -> decltype(std::ranges::size(std::declval<_R&>()));
#endif // _ONEDPL_CPP20_RANGES_PRESENT

template <typename _It>
auto
__check_size(...) -> typename std::iterator_traits<_It>::difference_type;

template <typename _R>
using __difference_t = std::make_signed_t<decltype(__check_size<_R>(0))>;

} //namespace __internal

#if _ONEDPL_CPP20_RANGES_PRESENT
#    if _ONEDPL_CPP26_DEFAULT_VALUE_TYPE_PRESENT
template <std::indirectly_readable I, std::indirectly_regular_unary_invocable<I> Proj>
using projected_value_t = std::projected_value_t<I, Proj>;
#    else
template <std::indirectly_readable I, std::indirectly_regular_unary_invocable<I> Proj>
using projected_value_t = std::remove_cvref_t<std::invoke_result_t<Proj&, std::iter_value_t<I>&>>;
#    endif
#endif //_ONEDPL_CPP20_RANGES_PRESENT

namespace __ranges
{

template <typename _R, typename = void>
struct __has_empty : std::false_type
{
};

template <typename _R>
struct __has_empty<_R, std::void_t<decltype(std::declval<_R>().empty())>> : std::true_type
{
};

template <typename _Range>
bool
__empty(_Range&& __rng)
{
    if constexpr (__has_empty<_Range>::value)
        return __rng.empty();
    else
        return __rng.begin() == __rng.end();
}

template <typename _R, typename = void>
struct __is_eval_size_through_size : std::false_type
{
};

template <typename _R>
struct __is_eval_size_through_size<_R, std::void_t<decltype(std::declval<_R>().size())>> : std::true_type
{
};

template <typename _Range>
std::enable_if_t<__is_eval_size_through_size<_Range>::value, decltype(std::declval<_Range>().size())>
__size(_Range&& __rng)
{
    return __rng.size();
}

#if _ONEDPL_CPP20_RANGES_PRESENT
template <typename _Range>
std::enable_if_t<!__is_eval_size_through_size<_Range>::value,
                 decltype(std::ranges::distance(std::declval<_Range>().begin(), std::declval<_Range>().end()))>
__size(_Range&& __rng)
{
    return std::ranges::distance(__rng.begin(), __rng.end());
}
#else
template <typename _Range>
std::enable_if_t<!__is_eval_size_through_size<_Range>::value,
                 decltype(std::distance(std::declval<_Range>().begin(), std::declval<_Range>().end()))>
__size(_Range&& __rng)
{
    return std::distance(__rng.begin(), __rng.end());
}
#endif

template <typename... _Rng>
using __common_size_t = std::common_type_t<std::make_unsigned_t<decltype(__size(std::declval<_Rng>()))>...>;

template <std::size_t _RngIndex>
struct __nth_range_size
{
  private:
    template <std::size_t _RngIndexCurrent, typename _Range, typename... _Ranges>
    auto
    __nth_range_size_impl(const _Range& __rng, const _Ranges&... __rngs) const
    {
        if constexpr (_RngIndexCurrent == _RngIndex)
            return __size(__rng);
        else
            return __nth_range_size_impl<_RngIndexCurrent + 1>(__rngs...);
    }

  public:
    template <typename... _Ranges>
    auto
    operator()(const _Ranges&... __rngs) const
    {
        static_assert(_RngIndex < sizeof...(_Ranges));
        return __nth_range_size_impl<0>(__rngs...);
    }
};

using __first_size_calc = __nth_range_size<0>;
using __second_size_calc = __nth_range_size<1>;

struct __min_size_calc
{
    template <typename... _Ranges>
    auto
    operator()(const _Ranges&... __rngs) const
    {
        using _Size = std::make_unsigned_t<std::common_type_t<oneapi::dpl::__internal::__difference_t<_Ranges>...>>;
        return std::min({_Size(oneapi::dpl::__ranges::__size(__rngs))...});
    }
};

// helpers to check implement "has_base"
template <typename U>
auto
test_pipeline_object(int) -> decltype(::std::declval<U>().base(), ::std::true_type{});

template <typename U>
auto
test_pipeline_object(...) -> ::std::false_type;

// has_base check definition
template <typename Range>
struct is_pipeline_object : decltype(test_pipeline_object<Range>(0))
{
};

// Recursive helper
template <typename Range, typename = void>
struct pipeline_base
{
    using type = Range;
};

template <typename Range>
struct pipeline_base<Range, ::std::enable_if_t<is_pipeline_object<Range>::value>>
{
    using type = typename pipeline_base<::std::decay_t<decltype(::std::declval<Range>().base())>>::type;
};

//pipeline_base_range
template <typename Range, typename = void>
struct pipeline_base_range
{
    Range rng;

    pipeline_base_range(Range r) : rng(r) {}
    constexpr Range
    base_range()
    {
        return rng;
    };
};

// use ::std::conditional to understand what class to inherit from
template <typename Range>
struct pipeline_base_range<Range, ::std::enable_if_t<is_pipeline_object<Range>::value>>
{
    Range rng;

    pipeline_base_range(Range r) : rng(r) {}
    constexpr auto
    base_range() -> decltype(pipeline_base_range<decltype(rng.base())>(rng.base()).base_range())
    {
        return pipeline_base_range<decltype(rng.base())>(rng.base()).base_range();
    };
};

template <typename _TupleType, typename _F, ::std::size_t... _Ip>
void
invoke(const _TupleType& __t, _F __f, ::std::index_sequence<_Ip...>)
{
    __f(::std::get<_Ip>(__t)...);
}

template <typename... _Ranges>
class zip_view
{
    static_assert(sizeof...(_Ranges) > 0, "Cannot instantiate zip_view with empty template parameter pack");

    using _tuple_ranges_t = oneapi::dpl::__internal::tuple<_Ranges...>;

    template <typename Idx, ::std::size_t... _Ip>
    auto
    make_reference(_tuple_ranges_t __t, Idx __i, ::std::index_sequence<_Ip...>) const
        -> decltype(oneapi::dpl::__internal::tuple<decltype(::std::declval<_Ranges&>().operator[](__i))...>(
            ::std::get<_Ip>(__t).operator[](__i)...))
    {
        return oneapi::dpl::__internal::tuple<decltype(::std::declval<_Ranges&>().operator[](__i))...>(
            ::std::get<_Ip>(__t).operator[](__i)...);
    }

  public:
    using value_type = oneapi::dpl::__internal::tuple<oneapi::dpl::__internal::__value_t<_Ranges>...>;
    static constexpr ::std::size_t __num_ranges = sizeof...(_Ranges);

    explicit zip_view(_Ranges... __args) : __m_ranges(__args...) {}

    auto
    size() const -> decltype(::std::get<0>(::std::declval<_tuple_ranges_t>()).size())
    {
        return ::std::get<0>(__m_ranges).size();
    }

    //TODO: C++ Standard states that the operator[] index should be the diff_type of the underlying range.
    template <typename Idx>
    constexpr auto operator[](Idx __i) const
        -> decltype(make_reference(::std::declval<_tuple_ranges_t>(), __i, ::std::make_index_sequence<__num_ranges>()))
    {
        return make_reference(__m_ranges, __i, ::std::make_index_sequence<__num_ranges>());
    }

    bool
    empty() const
    {
        return size() == 0;
    }

    _tuple_ranges_t
    tuple() const
    {
        return __m_ranges;
    }

  private:
    _tuple_ranges_t __m_ranges;
};

template <typename... _Views>
auto
make_zip_view(_Views... args)
{
    return zip_view<_Views...>(args...);
}

// a custom view, over a pair of "passed directly" iterators
template <typename _Iterator>
class guard_view
{
    using diff_type = typename ::std::iterator_traits<_Iterator>::difference_type;

  public:
    using value_type = typename ::std::iterator_traits<_Iterator>::value_type;

    guard_view(_Iterator __first = _Iterator(), diff_type __n = 0) : m_p(__first), m_count(__n) {}
    guard_view(_Iterator __first, _Iterator __last) : m_p(__first), m_count(__last - __first) {}

    _Iterator
    begin() const
    {
        return m_p;
    }

    _Iterator
    end() const
    {
        return begin() + size();
    }

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying iterator
    template <typename Idx>
    auto operator[](Idx i) const -> decltype(begin()[i])
    {
        return begin()[i];
    }

    diff_type
    size() const
    {
        return m_count;
    }
    bool
    empty() const
    {
        return size() == 0;
    }

  private:
    _Iterator m_p;     // a iterator (pointer)  to data in memory
    diff_type m_count; // size of the data
};

//It is kind of pseudo-view for reverse_view support.
template <typename _R>
struct reverse_view_simple
{
    using value_type = oneapi::dpl::__internal::__value_t<_R>;

    _R __r;

    reverse_view_simple(_R __rng) : __r(__rng) {}

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
    template <typename Idx>
    auto operator[](Idx __i) const -> decltype(__r[__i])
    {
        return __r[size() - __i - 1];
    }

    auto
    size() const -> decltype(oneapi::dpl::__ranges::__size(__r))
    {
        return oneapi::dpl::__ranges::__size(__r);
    }

    bool
    empty() const
    {
        return oneapi::dpl::__ranges::__empty(__r);
    }

    auto
    base() const -> decltype(__r)
    {
        return __r;
    }
};

//It is kind of pseudo-view for take_view support. We assume that the underlying range will not shrink
//after creation of the view to favor performance.
template <typename _R, typename _Size>
struct take_view_simple
{
    using value_type = oneapi::dpl::__internal::__value_t<_R>;

    _R __r;
    _Size __n;

    take_view_simple(_R __rng, _Size __size) : __r(__rng), __n(__size)
    {
        assert(__n >= 0 && __n <= oneapi::dpl::__ranges::__size(__r));
    }

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
    template <typename Idx>
    auto operator[](Idx __i) const -> decltype(__r[__i])
    {
        return __r[__i];
    }

    _Size
    size() const
    {
        assert(__n <= oneapi::dpl::__ranges::__size(__r));
        return __n;
    }

    bool
    empty() const
    {
        return size() == 0;
    }

    auto
    base() const -> decltype(__r)
    {
        return __r;
    }
};

//It is kind of pseudo-view for drop_view support. We assume that the underlying range will not shrink
//after creation of the view to favor performance.
template <typename _R, typename _Size>
struct drop_view_simple
{
    using value_type = oneapi::dpl::__internal::__value_t<_R>;

    _R __r;
    _Size __n;

    drop_view_simple(_R __rng, _Size __size) : __r(__rng), __n(__size)
    {
        assert(__n >= 0 && __n <= oneapi::dpl::__ranges::__size(__r));
    }

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
    template <typename Idx>
    auto operator[](Idx __i) const -> decltype(__r[__i])
    {
        return __r[__n + __i];
    }

    _Size
    size() const
    {
        assert(oneapi::dpl::__ranges::__size(__r) >= __n);
        return oneapi::dpl::__ranges::__size(__r) - __n;
    }

    bool
    empty() const
    {
        return size() == 0;
    }

    auto
    base() const -> decltype(__r)
    {
        return __r;
    }
};

//replicate_start_view_simple inserts replicates of the first element m times, then continues with the range as normal.
// For counting iterator range {0,1,2,3,4,5,...}, and __replicate_count = 3, the result is {0,0,0,0,1,2,3,4,5,...}
template <typename _R, typename _Size>
struct replicate_start_view_simple
{
    using value_type = oneapi::dpl::__internal::__value_t<_R>;

    _R __r;
    _Size __repl_count;

    replicate_start_view_simple(_R __rng, _Size __replicate_count) : __r(__rng), __repl_count(__replicate_count)
    {
        assert(__repl_count >= 0);
    }

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
    template <typename Idx>
    auto operator[](Idx __i) const -> decltype(__r[__i])
    {
        return (__i < __repl_count) ? __r[0] : __r[__i - __repl_count];
    }

    _Size
    size() const
    {
        // if base range is empty, replication does not extend the valid size
        return oneapi::dpl::__ranges::__empty(__r) ? 0 : oneapi::dpl::__ranges::__size(__r) + __repl_count;
    }

    bool
    empty() const
    {
        return size() == 0;
    }

    auto
    base() const -> decltype(__r)
    {
        return __r;
    }
};

//It is kind of pseudo-view for transfom_iterator support.
template <typename _R, typename _F>
struct transform_view_simple
{
    using value_type = ::std::decay_t<::std::invoke_result_t<_F&, decltype(::std::declval<_R&>()[0])>>;

    _R __r;
    _F __f;

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
    template <typename Idx>
    auto operator[](Idx __i) const -> decltype(__f(__r[__i]))
    {
        return __f(__r[__i]);
    }

    auto
    size() const -> decltype(oneapi::dpl::__ranges::__size(__r))
    {
        return oneapi::dpl::__ranges::__size(__r);
    }

    bool
    empty() const
    {
        return oneapi::dpl::__ranges::__empty(__r);
    }

    auto
    base() const -> decltype(__r)
    {
        return __r;
    }
};

template <typename _Map>
auto
test_map_view(int) -> decltype(::std::declval<_Map>()[0], ::std::true_type{});

template <typename _Map>
auto
test_map_view(...) -> ::std::false_type;

//pseudo-checking on viewable range concept
template <typename _Map>
struct is_map_view : decltype(test_map_view<_Map>(0))
{
};

//It is kind of pseudo-view for permutation_iterator support.
template <typename _Source, typename _M, typename = void>
struct permutation_view_simple;

//permutation view: specialization for an index map functor
//size of such view  is specified by a caller
template <typename _Source, typename _M>
struct permutation_view_simple<_Source, _M, ::std::enable_if_t<oneapi::dpl::__internal::__is_functor<_M>>>
{
    using value_type = oneapi::dpl::__internal::__value_t<_Source>;
    using _Size = oneapi::dpl::__internal::__difference_t<_Source>;

    _Source __src; //Iterator (pointer) or unreachable range
    _M __map_fn;
    _Size __size;

    permutation_view_simple(_Source __data, _M __m, _Size __s) : __src(__data), __map_fn(__m), __size(__s) {}

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
    template <typename Idx>
    decltype(auto)
    operator[](Idx __i) const
    {
        return __src[__map_fn(__i)];
    }

    auto
    size() const
    {
        return __size;
    }

    bool
    empty() const
    {
        return size() == 0;
    }

    auto
    base() const
    {
        return __src;
    }
};

//permutation view: specialization for a map view (a viewable range concept)
//size of such view  is specified by size of the map view (permutation range)
template <typename _Source, typename _M>
struct permutation_view_simple<_Source, _M, ::std::enable_if_t<is_map_view<_M>::value>>
{
    using value_type = oneapi::dpl::__internal::__value_t<_Source>;

    _Source __src; //Iterator (pointer) or unreachable range
    _M __map;      //permutation range

    permutation_view_simple(_Source __data, _M __m) : __src(__data), __map(__m) {}

    //TODO: to be consistent with C++ standard, this Idx should be changed to diff_type of underlying range
    template <typename Idx>
    decltype(auto)
    operator[](Idx __i) const
    {
        return __src[__map[__i]];
    }

    auto
    size() const
    {
        return __map.size();
    }

    bool
    empty() const
    {
        return size() == 0;
    }

    auto
    base() const
    {
        return oneapi::dpl::__internal::make_tuple(__src, __map);
    }
};

//permutation discard view:
struct permutation_discard_view
{
    using value_type = oneapi::dpl::internal::ignore_copyable;
    using difference_type = ::std::ptrdiff_t;
    difference_type m_count;

    permutation_discard_view(difference_type __n) : m_count(__n) {}

    oneapi::dpl::internal::ignore_copyable operator[](difference_type) const { return oneapi::dpl::internal::ignore; }

    difference_type
    size() const
    {
        return m_count;
    }

    bool
    empty() const
    {
        return size() == 0;
    }
};

template <typename _R, typename = void>
struct __has_subsctiption_op : std::false_type
{
};

template <typename _R>
struct __has_subsctiption_op<_R, std::void_t<decltype(std::declval<_R>().operator[](0))>> : std::true_type
{
};

template <typename _Source, typename _Base = std::decay_t<_Source>>
struct __subscription_impl_view_simple : _Base
{
    static_assert(
        !__has_subsctiption_op<_Base>::value,
        "The usage of __subscription_impl_view_simple prohibited if std::decay_t<_Source>::operator[] implemented");

    using value_type = oneapi::dpl::__internal::__value_t<_Base>;
    using index_type = oneapi::dpl::__internal::__difference_t<_Base>;

    // Define default constructors
    __subscription_impl_view_simple() = default;
    __subscription_impl_view_simple(const __subscription_impl_view_simple&) = default;
    __subscription_impl_view_simple(__subscription_impl_view_simple&&) = default;

    // Define custom constructor to forward arguments to the base class
    template <typename... _Args>
    __subscription_impl_view_simple(_Args&&... __args) : _Base(std::forward<_Args>(__args)...)
    {
    }

    // Define default operator=
    __subscription_impl_view_simple&
    operator=(const __subscription_impl_view_simple&) = default;
    __subscription_impl_view_simple&
    operator=(__subscription_impl_view_simple&&) = default;

    decltype(auto)
    operator[](index_type __i)
    {
        return *std::next(_Base::begin(), __i);
    }

    decltype(auto)
    operator[](index_type __i) const
    {
        return *std::next(_Base::begin(), __i);
    }
};

template <typename _Range>
decltype(auto)
__get_subscription_view(_Range&& __rng)
{
    if constexpr (__has_subsctiption_op<_Range>::value)
    {
        return std::forward<_Range>(__rng);
    }
    else
    {
        return __subscription_impl_view_simple<_Range>(std::forward<_Range>(__rng));
    }
}

} // namespace __ranges
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_UTILS_RANGES_H
