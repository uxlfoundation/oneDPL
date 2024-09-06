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

#ifndef _ONEDPL_UTILS_RANGES_SYCL_H
#define _ONEDPL_UTILS_RANGES_SYCL_H

#include <iterator>
#include <type_traits>

#include "../../utils_ranges.h"
#include "../../iterator_impl.h"
#include "../../glue_numeric_defs.h"
#include "sycl_iterator.h"
#include "sycl_defs.h"

namespace oneapi
{
namespace dpl
{
namespace __ranges
{

namespace __internal
{
template <typename _AccessorType, typename _BufferType, typename _DiffType>
static _AccessorType
__create_accessor(_BufferType& __buf, _DiffType __offset, _DiffType __n)
{
    auto __n_buf = __dpl_sycl::__get_buffer_size(__buf);
    auto __n_acc = (__n > 0 ? __n : __n_buf);

    assert(__offset + __n_acc <= __n_buf &&
           "The sum of accessRange and accessOffset should not exceed the range of buffer");

    return {__buf, sycl::range<1>(__n_acc), __offset};
}
} // namespace __internal

//A SYCL range over SYCL buffer
template <typename _T, sycl::access::mode _AccMode = sycl::access::mode::read,
          __dpl_sycl::__target _Target = __dpl_sycl::__target_device,
          sycl::access::placeholder _Placeholder = sycl::access::placeholder::true_t>
class all_view
{
    using __return_t = ::std::conditional_t<_AccMode == sycl::access::mode::read, const _T, _T>;
    using __diff_type = typename ::std::iterator_traits<_T*>::difference_type;
    using __accessor_t = sycl::accessor<_T, 1, _AccMode, _Target, _Placeholder>;

  public:
    using value_type = _T;

    all_view(sycl::buffer<_T, 1> __buf = sycl::buffer<_T, 1>(0), __diff_type __offset = 0, __diff_type __n = 0)
        : __m_acc(__internal::__create_accessor<__accessor_t>(__buf, __offset, __n))
    {
    }

    all_view(__accessor_t __acc) : __m_acc(__acc) {}

    __return_t*
    begin() const
    {
        return &__m_acc[0];
    } //or “honest” iterator over an accessor and a sentinel

    __return_t*
    end() const
    {
        return begin() + size();
    }
    __return_t& operator[](__diff_type i) const { return begin()[i]; }

    __diff_type
    size() const
    {
        return __dpl_sycl::__get_accessor_size(__m_acc);
    }
    bool
    empty() const
    {
        return size() == 0;
    }

    void
    require_access(sycl::handler& __cgh)
    {
        __cgh.require(__m_acc);
    }

    __accessor_t
    accessor() const
    {
        return __m_acc;
    }

  private:
    __accessor_t __m_acc;
};

template <sycl::access::mode AccMode = sycl::access::mode::read_write,
          __dpl_sycl::__target _Target = __dpl_sycl::__target_device,
          sycl::access::placeholder _Placeholder = sycl::access::placeholder::true_t>
struct all_view_fn
{
    template <typename _T>
    constexpr oneapi::dpl::__ranges::all_view<_T, AccMode, _Target, _Placeholder>
    operator()(sycl::buffer<_T, 1> __buf, typename ::std::iterator_traits<_T*>::difference_type __offset = 0,
               typename ::std::iterator_traits<_T*>::difference_type __n = 0) const
    {
        return oneapi::dpl::__ranges::all_view<_T, AccMode, _Target, _Placeholder>(__buf, __offset, __n);
    }

    template <typename _R>
    auto
    operator()(_R&& __r) const -> decltype(std::forward<_R>(__r))
    {
        return std::forward<_R>(__r);
    }
};

#if _ONEDPL_SYCL_PLACEHOLDER_HOST_ACCESSOR_DEPRECATED
struct all_host_view_fn
{
    // An overload for sycl::buffer template type
    template <typename _T>
    auto
    operator()(sycl::buffer<_T, 1> __buf, typename ::std::iterator_traits<_T*>::difference_type __offset = 0,
               typename ::std::iterator_traits<_T*>::difference_type __n = 0) const
    {
        return __internal::__create_accessor<sycl::host_accessor<_T>>(__buf, __offset, __n);
    }

    // "No operation" overload for another ranges/views
    template <typename _R>
    auto
    operator()(_R&& __r) const -> decltype(::std::forward<_R>(__r))
    {
        return ::std::forward<_R>(__r);
    }
};
#endif

namespace views
{
inline constexpr all_view_fn<sycl::access::mode::read_write, __dpl_sycl::__target_device,
                             sycl::access::placeholder::true_t>
    all;

inline constexpr all_view_fn<sycl::access::mode::read, __dpl_sycl::__target_device, sycl::access::placeholder::true_t>
    all_read;

inline constexpr all_view_fn<sycl::access::mode::write, __dpl_sycl::__target_device, sycl::access::placeholder::true_t>
    all_write;

#if _ONEDPL_SYCL_PLACEHOLDER_HOST_ACCESSOR_DEPRECATED
inline constexpr all_host_view_fn
#else
inline constexpr all_view_fn<sycl::access::mode::read_write, __dpl_sycl::__host_target,
                             sycl::access::placeholder::false_t>
#endif
    host_all;
} // namespace views

//all_view traits

template <typename Iter, typename Void = void> // for iterators that should not be passed directly
struct is_zip : ::std::false_type
{
};

template <typename Iter> // for iterators defined as direct pass
struct is_zip<Iter, ::std::enable_if_t<Iter::is_zip::value>> : ::std::true_type
{
};

template <typename Iter, typename Void = void>
struct is_permutation : ::std::false_type
{
};

template <typename Iter> // for permutation_iterators
struct is_permutation<Iter, ::std::enable_if_t<Iter::is_permutation::value>> : ::std::true_type
{
};

//is_passed_directly trait definition; specializations for the oneDPL iterators

template <typename Iter, typename Void = void>
struct is_passed_directly : ::std::is_pointer<Iter>
{
};

//support legacy "is_passed_directly" trait
template <typename Iter>
struct is_passed_directly<Iter, ::std::enable_if_t<Iter::is_passed_directly::value>> : ::std::true_type
{
};

//support std::vector::iterator with usm host / shared allocator as passed directly
template <typename Iter>
struct is_passed_directly<Iter, std::enable_if_t<oneapi::dpl::__internal::__is_known_usm_vector_iter_v<Iter>>>
    : std::true_type
{
};

template <typename Ip>
struct is_passed_directly<oneapi::dpl::counting_iterator<Ip>> : ::std::true_type
{
};

template <>
struct is_passed_directly<oneapi::dpl::discard_iterator> : ::std::true_type
{
};

template <typename Iter>
struct is_passed_directly<::std::reverse_iterator<Iter>> : is_passed_directly<Iter>
{
};

template <typename Iter, typename Unary>
struct is_passed_directly<oneapi::dpl::transform_iterator<Iter, Unary>> : is_passed_directly<Iter>
{
};

template <typename SourceIterator, typename IndexIterator>
struct is_passed_directly<oneapi::dpl::permutation_iterator<SourceIterator, IndexIterator>>
    : ::std::conjunction<
          is_passed_directly<SourceIterator>,
          is_passed_directly<typename oneapi::dpl::permutation_iterator<SourceIterator, IndexIterator>::IndexMap>>
{
};

template <typename... Iters>
struct is_passed_directly<zip_iterator<Iters...>> : ::std::conjunction<is_passed_directly<Iters>...>
{
};

template <typename Iter>
inline constexpr bool is_passed_directly_v = is_passed_directly<Iter>::value;

// A trait for checking if iterator is heterogeneous or not

template <typename Iter>
struct is_sycl_iterator : ::std::false_type
{
};

template <oneapi::dpl::access_mode Mode, typename... Types>
struct is_sycl_iterator<oneapi::dpl::__internal::sycl_iterator<Mode, Types...>> : ::std::true_type
{
};

template <typename Iter, typename Void = void>
struct is_hetero_legacy_trait : ::std::false_type
{
};

template <typename Iter>
struct is_hetero_legacy_trait<Iter, ::std::enable_if_t<Iter::is_hetero::value>> : ::std::true_type
{
};

template <typename Iter>
inline constexpr bool is_sycl_iterator_v = is_sycl_iterator<Iter>::value || is_hetero_legacy_trait<Iter>::value;

//A trait for checking if it needs to create a temporary SYCL buffer or not

template <typename _Iter, typename Void = void>
struct is_temp_buff : ::std::false_type
{
};

template <typename _Iter>
struct is_temp_buff<_Iter, ::std::enable_if_t<!is_sycl_iterator_v<_Iter> && !::std::is_pointer_v<_Iter> &&
                                              !is_passed_directly_v<_Iter>>> : ::std::true_type
{
};

template <typename _Iter>
using val_t = typename ::std::iterator_traits<_Iter>::value_type;

//range/zip_view/all_view/ variadic utilities

template <typename _Range, typename... _Ranges>
struct __get_first_range_type
{
    using type = _Range;
};

template <typename _Range, typename... _Ranges>
constexpr auto
__get_first_range_size(const _Range& __rng, const _Ranges&...) -> decltype(__rng.size())
{
    return __rng.size();
}

//forward declaration required for _require_access_args
template <typename _Range, typename... _Ranges>
void
__require_access(sycl::handler& __cgh, _Range&& __rng, _Ranges&&... __rest);

template <typename _Cgh>
struct _require_access_args
{
    _Cgh __cgh;
    template <typename... Args>
    void
    operator()(Args&&... args)
    {
        __require_access(__cgh, ::std::forward<Args>(args)...);
    }
};

template <typename... _Ranges>
void
__require_access_zip(sycl::handler& __cgh, oneapi::dpl::__ranges::zip_view<_Ranges...>& __zip)
{
    const ::std::size_t __num_ranges = sizeof...(_Ranges);
    oneapi::dpl::__ranges::invoke(__zip.tuple(), _require_access_args<decltype(__cgh)>{__cgh},
                                  ::std::make_index_sequence<__num_ranges>());
}

//__require_access utility

inline void
__require_access(sycl::handler&)
{
}

template <typename T, sycl::access::mode M>
void
__require_access_range(sycl::handler& __cgh, oneapi::dpl::__ranges::all_view<T, M>& sycl_view)
{
    sycl_view.require_access(__cgh);
}

template <typename... _Ranges>
void
__require_access_range(sycl::handler& __cgh, zip_view<_Ranges...>& zip_rng)
{
    __require_access_zip(__cgh, zip_rng);
}

template <typename... _Ranges>
void
__require_access_range(sycl::handler& __cgh, oneapi::dpl::__internal::tuple<_Ranges...>& __tuple)
{
    const ::std::size_t __num_ranges = sizeof...(_Ranges);
    oneapi::dpl::__ranges::invoke(__tuple, _require_access_args<decltype(__cgh)>{__cgh},
                                  ::std::make_index_sequence<__num_ranges>());
}

template <typename _BaseRange>
void
__require_access_range(sycl::handler&, _BaseRange&)
{
}

template <typename _Range, typename... _Ranges>
void
__require_access(sycl::handler& __cgh, _Range&& __rng, _Ranges&&... __rest)
{
    //getting an access for the all_view based range
    auto base_rng = oneapi::dpl::__ranges::pipeline_base_range<_Range>(::std::forward<_Range>(__rng)).base_range();

    __require_access_range(__cgh, base_rng);

    //getting an access for the rest ranges
    __require_access(__cgh, ::std::forward<_Ranges>(__rest)...);
}

template <typename _R>
struct __range_holder
{
    _R __r;
    constexpr _R
    all_view() const
    {
        return __r;
    }

    //TODO: The dummy conversion operators.
    //In case when a temporary buffer doesn't need we have to use a dummy type - "oneapi::dpl::internal::ignore_copyable, for example
    operator oneapi::dpl::internal::ignore_copyable() const { return oneapi::dpl::internal::ignore; }
    operator oneapi::dpl::__internal::tuple<oneapi::dpl::internal::ignore_copyable,
                                            oneapi::dpl::internal::ignore_copyable>() const
    {
        return oneapi::dpl::__internal::tuple<oneapi::dpl::internal::ignore_copyable,
                                              oneapi::dpl::internal::ignore_copyable>(oneapi::dpl::internal::ignore,
                                                                                      oneapi::dpl::internal::ignore);
    }
    operator oneapi::dpl::__internal::tuple<oneapi::dpl::internal::ignore_copyable,
                                            oneapi::dpl::internal::ignore_copyable,
                                            oneapi::dpl::internal::ignore_copyable>() const
    {
        return oneapi::dpl::__internal::tuple<oneapi::dpl::internal::ignore_copyable,
                                              oneapi::dpl::internal::ignore_copyable,
                                              oneapi::dpl::internal::ignore_copyable>(
            oneapi::dpl::internal::ignore, oneapi::dpl::internal::ignore, oneapi::dpl::internal::ignore);
    }
};

template <sycl::access::mode AccMode, typename _Iterator>
struct __get_sycl_range
{
    __get_sycl_range()
    {
        m_buffers.reserve(4); //4 - due to a number of arguments(host iterators) cannot be too big.
    }

  private:
    //We have to keep sycl buffer(s) instance here by sync reasons;
    ::std::vector<::std::unique_ptr<oneapi::dpl::__internal::__lifetime_keeper_base>> m_buffers;

    template <sycl::access::mode _LocalAccMode>
    static constexpr bool __is_copy_direct_v =
        _LocalAccMode == sycl::access::mode::read_write || _LocalAccMode == sycl::access::mode::read;
    template <sycl::access::mode _LocalAccMode>
    static constexpr bool __is_copy_back_v =
        _LocalAccMode == sycl::access::mode::read_write || _LocalAccMode == sycl::access::mode::write;

    //SFINAE iterator type checks
    template <typename It>
    static constexpr auto
    __is_addressable(int) -> decltype(std::addressof(*::std::declval<It&>()), std::true_type{});
    template <typename It>
    static constexpr std::false_type
    __is_addressable(...);
    template <typename It>
    static constexpr bool __is_addressable_v = decltype(__is_addressable<It>(0))::value;

    template <typename _F, typename _It, typename _DiffType>
    static auto
    __gen_view(_F& __f, _It __it, _DiffType __n) -> decltype(__f(__it, __it + __n))
    {
        return __f(__it, __it + __n);
    }

    template <typename _TupleType, typename _DiffType, ::std::size_t... _Ip>
    auto
    gen_zip_view(_TupleType __t, _DiffType __n, ::std::index_sequence<_Ip...>)
    {
        // Send each zipped iterator to `gen_view` which recursively calls __get_sycl_range() to process them.
        auto tmp = oneapi::dpl::__internal::make_tuple(__gen_view(*this, ::std::get<_Ip>(__t), __n)...);
        return oneapi::dpl::__ranges::make_zip_view(::std::get<_Ip>(tmp).all_view()...);
    }

    //zip iterators
    template <sycl::access::mode _LocalAccMode, typename... Iters>
    auto
    __process_input_iter(oneapi::dpl::zip_iterator<Iters...> __first, oneapi::dpl::zip_iterator<Iters...> __last)
    {
        assert(__first < __last);

        const ::std::size_t __num_it = sizeof...(Iters);
        auto rng = gen_zip_view(__first.base(), __last - __first, ::std::make_index_sequence<__num_it>());
        return __range_holder<decltype(rng)>{rng};
    }

    //specialization for transform_iterator
    template <sycl::access::mode _LocalAccMode, typename _Iter, typename _UnaryFunction>
    auto
    __process_input_iter(oneapi::dpl::transform_iterator<_Iter, _UnaryFunction> __first,
                         oneapi::dpl::transform_iterator<_Iter, _UnaryFunction> __last)
    {
        assert(__first < __last);

        auto res = __process_input_iter<_LocalAccMode>(__first.base(), __last.base());
        auto rng = oneapi::dpl::__ranges::transform_view_simple<decltype(res.all_view()), decltype(__first.functor())>{
            res.all_view(), __first.functor()};

        return __range_holder<decltype(rng)>{rng};
    }

    //specialization for std::reverse_iterator
    template <sycl::access::mode _LocalAccMode, typename _Iter>
    auto
    __process_input_iter(::std::reverse_iterator<_Iter> __first, ::std::reverse_iterator<_Iter> __last)
    {
        assert(__first < __last);

        auto __res = __process_input_iter<_LocalAccMode>(__last.base(), __first.base());
        auto __rng = oneapi::dpl::__ranges::reverse_view_simple<decltype(__res.all_view())>{__res.all_view()};

        return __range_holder<decltype(__rng)>{__rng};
    }

    template <typename _R, typename _Map, typename _Size,
              ::std::enable_if_t<oneapi::dpl::__internal::__is_functor<_Map>, int> = 0>
    static auto
    __get_permutation_view(_R __r, _Map __m, _Size __s)
    {
        return oneapi::dpl::__ranges::permutation_view_simple<_R, _Map>{__r, __m, __s};
    }

    template <typename _R, typename _Map, typename _Size,
              ::std::enable_if_t<oneapi::dpl::__internal::__is_random_access_iterator_v<_Map>, int> = 0>
    auto
    __get_permutation_view(_R __r, _Map __m, _Size __s)
    {
        //For permutation iterator, the Map iterator is always read (only)
        auto view_map = __process_input_iter<sycl::access_mode::read>(__m, __m + __s).all_view();
        return oneapi::dpl::__ranges::permutation_view_simple<_R, decltype(view_map)>{__r, view_map};
    }

    //specialization for permutation_iterator using sycl_iterator as source
    template <sycl::access::mode _LocalAccMode, typename _It, typename _Map,
              ::std::enable_if_t<is_sycl_iterator_v<_It>, int> = 0>
    auto
    __process_input_iter(oneapi::dpl::permutation_iterator<_It, _Map> __first,
                         oneapi::dpl::permutation_iterator<_It, _Map> __last)
    {
        auto __n = __last - __first;
        assert(__n > 0);

        // Types for which oneapi::dpl::__ranges::is_sycl_iterator_v = true should have both:
        //  "get_buffer()" to return the buffer they are base upon and
        //  "get_idx()" to return the buffer offset

        //  __first.base() is not guaranteed to be a sycl_iterator, it may be another type which sets the trait
        //   is_hetero = ::std::true_type.  Therefore, to make sure our types match, we use get_idx() to get the buffer
        //   offset, and use that to recurse as a sycl_iterator over the __base_buffer.
        auto __base_iter = __first.base();
        auto __base_buffer = __base_iter.get_buffer();
        auto res_src = __process_input_iter<_LocalAccMode>(oneapi::dpl::begin(__base_buffer) + __base_iter.get_idx(),
                                                           oneapi::dpl::end(__base_buffer));

        //_Map is handled by recursively calling __get_sycl_range() in __get_permutation_view.
        auto rng = __get_permutation_view(res_src.all_view(), __first.map(), __n);

        return __range_holder<decltype(rng)>{rng};
    }

    //specialization for permutation_iterator using USM pointer or direct pass object as source
    template <sycl::access::mode _LocalAccMode, typename _Iter, typename _Map,
              ::std::enable_if_t<!is_sycl_iterator_v<_Iter> && is_passed_directly_v<_Iter>, int> = 0>
    auto
    __process_input_iter(oneapi::dpl::permutation_iterator<_Iter, _Map> __first,
                         oneapi::dpl::permutation_iterator<_Iter, _Map> __last)
    {
        auto __n = __last - __first;
        assert(__n > 0);

        // The size of the source range is unknown. So, just base iterator is passing to permutation_view
        //_Map is handled by recursively calling __get_sycl_range() in __get_permutation_view.
        auto rng = __get_permutation_view(__first.base(), __first.map(), __n);

        return __range_holder<decltype(rng)>{rng};
    }

    // specialization for general case, permutation_iterator with base iterator that is not sycl_iterator or
    // passed directly.
    template <sycl::access::mode _LocalAccMode, typename _Iter, typename _Map,
              ::std::enable_if_t<!is_sycl_iterator_v<_Iter> && !is_passed_directly_v<_Iter>, int> = 0>
    auto
    __process_input_iter(oneapi::dpl::permutation_iterator<_Iter, _Map> __first,
                         oneapi::dpl::permutation_iterator<_Iter, _Map> __last)
    {
        auto __n = __last - __first;
        assert(__n > 0);

        //TODO: investigate better method of handling this specifically for fancy_iterators which are composed fully
        //      of a combination of fancy_iterators, sycl_iterators, and is_passed_directly types.
        //      Currently this relies on UB because the size of the accessor when handling sycl_iterators
        //      in recursion below this level is incorrect.
        auto res_src = this->operator()(__first.base(), __first.base() + 1 /*source size*/);

        auto rng = __get_permutation_view(res_src.all_view(), __first.map(), __n);

        return __range_holder<decltype(rng)>{rng};
    }

    //specialization for permutation discard iterator
    template <sycl::access::mode _LocalAccMode, typename _Map>
    auto
    __process_input_iter(oneapi::dpl::permutation_iterator<oneapi::dpl::discard_iterator, _Map> __first,
                         oneapi::dpl::permutation_iterator<oneapi::dpl::discard_iterator, _Map> __last)
    {
        auto __n = __last - __first;
        assert(__n > 0);

        auto rng = oneapi::dpl::__ranges::permutation_discard_view(__n);

        return __range_holder<decltype(rng)>{rng};
    }

    // for raw pointers and direct pass objects (for example, counting_iterator, iterator of USM-containers)
    template <sycl::access::mode _LocalAccMode, typename _Iter>
    ::std::enable_if_t<is_passed_directly_v<_Iter>, __range_holder<oneapi::dpl::__ranges::guard_view<_Iter>>>
    __process_input_iter(_Iter __first, _Iter __last)
    {
        assert(__first < __last);
        return __range_holder<oneapi::dpl::__ranges::guard_view<_Iter>>{
            oneapi::dpl::__ranges::guard_view<_Iter>{__first, __last - __first}};
    }

    //specialization for hetero iterator
    template <sycl::access::mode _LocalAccMode, typename _Iter>
    auto
    __process_input_iter(_Iter __first, _Iter __last)
        -> ::std::enable_if_t<is_sycl_iterator_v<_Iter>,
                              __range_holder<oneapi::dpl::__ranges::all_view<val_t<_Iter>, _LocalAccMode>>>
    {
        assert(__first < __last);
        using value_type = val_t<_Iter>;

        // Types for which oneapi::dpl::__ranges::is_sycl_iterator_v = true should have both:
        //  "get_buffer()" to return the buffer they are base upon and
        //  "get_idx()" to return the buffer offset

        //  __first is not guaranteed to be a sycl_iterator, it may be another type which sets the trait
        //   is_hetero = ::std::true_type. We use get_idx() to get the buffer offset, use get_buffer() to get the
        //   buffer and use those to create the range.
        const auto __offset = __first.get_idx();
        const auto __size = __dpl_sycl::__get_buffer_size(__first.get_buffer());
        const auto __n = ::std::min(decltype(__size)(__last - __first), __size);
        assert(__offset + __n <= __size);

        return __range_holder<oneapi::dpl::__ranges::all_view<value_type, _LocalAccMode>>{
            oneapi::dpl::__ranges::all_view<value_type, _LocalAccMode>(__first.get_buffer() /* buffer */,
                                                                       __offset /* offset*/, __n /* size*/)};
    }

    //SFINAE-overload for a contiguous host iterator
    template <sycl::access::mode _LocalAccMode, typename _Iter>
    auto
    __process_input_iter(_Iter __first, _Iter __last)
        -> ::std::enable_if_t<is_temp_buff<_Iter>::value && __is_addressable_v<_Iter> && !is_zip<_Iter>::value &&
                                  !is_permutation<_Iter>::value,
                              __range_holder<oneapi::dpl::__ranges::all_view<val_t<_Iter>, _LocalAccMode>>>
    {
        using _T = val_t<_Iter>;

        return __process_host_iter_impl<_LocalAccMode>(__first, __last, [&]() {
            if constexpr (__is_copy_direct_v<_LocalAccMode>)
            {
                //wait and copy on a buffer destructor; an exclusive access buffer, good performance
                return sycl::buffer<_T, 1>{::std::addressof(*__first), __last - __first};

                //No call to sycl::buffer::set_final_data() is required here because this sycl::buffer ctor
                // guarantees by specification that data will be written back to this host data pointer upon destruction
                // when sycl::buffer::set_write_back(true) is called
            }
            else
            {
                sycl::buffer<_T, 1> __buf(__last - __first);
                __buf.set_final_data(::std::addressof(*__first)); //wait and fast copy on a buffer destructor
                return __buf;
            }
        });
    }

    //SFINAE-overload for non-contiguous host iterator
    template <sycl::access::mode _LocalAccMode, typename _Iter>
    auto
    __process_input_iter(_Iter __first, _Iter __last)
        -> ::std::enable_if_t<is_temp_buff<_Iter>::value && !__is_addressable_v<_Iter> && !is_zip<_Iter>::value &&
                                  !is_permutation<_Iter>::value,
                              __range_holder<oneapi::dpl::__ranges::all_view<val_t<_Iter>, _LocalAccMode>>>
    {
        using _T = val_t<_Iter>;

        return __process_host_iter_impl<_LocalAccMode>(__first, __last, [&]() {
            if constexpr (__is_copy_direct_v<_LocalAccMode>)
            {
                //This constructor requires an extra host-side copy as compared to the host pointer + size constructors
                sycl::buffer<_T, 1> __buf(__first, __last); //SYCL API for non-contiguous iterators

                // Required for copy back to the host container for the above sycl::buffer ctor upon destruction
                if constexpr (__is_copy_back_v<_LocalAccMode>)
                    __buf.set_final_data(__first); //SYCL API for non-contiguous iterators
                return __buf;
            }
            else
            {
                sycl::buffer<_T, 1> __buf(__last - __first);
                __buf.set_final_data(__first); //SYCL API for non-contiguous iterators
                return __buf;
            }
        });
    }

    //implementation of operator()(_Iter __first, _Iter __last) for the host iterator types
    template <sycl::access::mode _LocalAccMode, typename _Iter, typename _GetBufferFunc>
    auto
    __process_host_iter_impl(_Iter __first, _Iter __last, _GetBufferFunc __get_buf)
    {
        static_assert(!oneapi::dpl::__internal::is_const_iterator<_Iter>::value ||
                          _LocalAccMode == sycl::access::mode::read,
                      "Should be non-const iterator for a modifying algorithm.");

        assert(__first < __last);

        auto __buf = __get_buf();
        __buf.set_write_back(__is_copy_back_v<_LocalAccMode>);

        // We have to extend sycl buffer lifetime by sync reasons in case of host iterators. SYCL runtime has sync
        // in buffer destruction and a sycl view instance keeps just placeholder accessor, not a buffer.
        using BufferType = oneapi::dpl::__internal::__lifetime_keeper<decltype(__buf)>;
        m_buffers.push_back(::std::make_unique<BufferType>(__buf));

        using _T = val_t<_Iter>;
        return __range_holder<oneapi::dpl::__ranges::all_view<_T, _LocalAccMode>>{
            oneapi::dpl::__ranges::all_view<_T, _LocalAccMode>(__buf)};
    }

  public:
    template <typename... _ArgTypes>
    auto
    operator()(_ArgTypes... __args)
    {
        //when called using operator(), use access mode provided by the struct template parameter
        return __process_input_iter<AccMode>(::std::forward<_ArgTypes>(__args)...);
    }
};

//----------------------------------------------------------
// __select_backend (for the hetero policies)
//----------------------------------------------------------

//TODO required correct implementation of this __ranges::__select_backend()
// 1. There is still not RA ranges checks
// 2. Obviously, a return tag is not necessarily oneapi::dpl::__internal::__hetero_tag
template <typename _KernelName, typename... _Ranges>
oneapi::dpl::__internal::__hetero_tag<oneapi::dpl::__internal::__device_backend_tag>
__select_backend(const execution::device_policy<_KernelName>&, _Ranges&&...)
{
    return {};
}

#if _ONEDPL_FPGA_DEVICE
//TODO required correct implementation of this __ranges::__select_backend()
// 1. There is still not RA ranges checks
// 2. Obviously, a return tag is not necessarily oneapi::dpl::__internal::__hetero_tag
template <unsigned int _Factor, typename _KernelName, typename... _Ranges>
oneapi::dpl::__internal::__hetero_tag<oneapi::dpl::__internal::__fpga_backend_tag>
__select_backend(const execution::fpga_policy<_Factor, _KernelName>&, _Ranges&&...)
{
    return {};
}
#endif

} // namespace __ranges
} // namespace dpl
} // namespace oneapi

#if _ONEDPL_CPP20_RANGES_PRESENT
//A specialization for enable_view to true because oneapi::dpl::__ranges::all_view models a view (see C++ standard)
template <typename _T, sycl::access::mode _AccMode, sycl::target _Target, sycl::access::placeholder _Placeholder>
inline constexpr bool std::ranges::enable_view<oneapi::dpl::__ranges::all_view<_T, _AccMode, _Target, _Placeholder>> = true;
#endif

#endif // _ONEDPL_UTILS_RANGES_SYCL_H
