// -*- C++ -*-
//===-- sycl_iterator.h ---------------------------------------------------===//
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

#ifndef _ONEDPL_SYCL_ITERATOR_H
#define _ONEDPL_SYCL_ITERATOR_H

#include <iterator>
#include <type_traits>
#include "../../onedpl_config.h"
#include "sycl_defs.h"

namespace oneapi
{
namespace dpl
{

using access_mode = sycl::access::mode;

namespace __internal
{
// Iterator that hides sycl::buffer to pass those to algorithms.
// SYCL iterator is a pair of sycl::buffer and integer value
template <access_mode Mode, typename T, typename Allocator = __dpl_sycl::__buffer_allocator<T>>
struct sycl_iterator
{
  private:
    using Size = ::std::size_t;
    static constexpr int dim = 1;
    sycl::buffer<T, dim, Allocator> buffer;
    Size idx;

  public:
    using value_type = T;
    using difference_type = ::std::make_signed_t<Size>;
    using pointer = T*;
    using reference = T&;
    using iterator_category = ::std::random_access_iterator_tag;
    static constexpr access_mode mode = Mode;

    // required for make_sycl_iterator
    //TODO: sycl::buffer doesn't have a default constructor (SYCL API issue), so we have to create a trivial size buffer
    sycl_iterator(sycl::buffer<T, dim, Allocator> vec = sycl::buffer<T, dim, Allocator>(0), Size index = 0)
        : buffer(vec), idx(index)
    {
    }
    // required for iter_mode
    template <access_mode inMode>
    sycl_iterator(const sycl_iterator<inMode, T, Allocator>& in) : buffer(in.get_buffer())
    {
        auto old_iter = sycl_iterator<inMode, T, Allocator>{in.get_buffer(), 0};
        idx = in - old_iter;
    }
    sycl_iterator
    operator+(difference_type forward) const
    {
        return {buffer, idx + forward};
    }
    sycl_iterator
    operator-(difference_type backward) const
    {
        return {buffer, idx - backward};
    }
    friend sycl_iterator
    operator+(difference_type forward, const sycl_iterator& it)
    {
        return it + forward;
    }
    friend sycl_iterator
    operator-(difference_type forward, const sycl_iterator& it)
    {
        return it - forward;
    }
    difference_type
    operator-(const sycl_iterator& it) const
    {
        assert(buffer == it.get_buffer());
        return idx - it.idx;
    }
    bool
    operator==(const sycl_iterator& it) const
    {
        assert(buffer == it.get_buffer());
        return *this - it == 0;
    }
    bool
    operator!=(const sycl_iterator& it) const
    {
        assert(buffer == it.get_buffer());
        return !(*this == it);
    }
    bool
    operator<(const sycl_iterator& it) const
    {
        assert(buffer == it.get_buffer());
        return *this - it < 0;
    }

    // This function is required for types for which oneapi::dpl::__ranges::is_hetero_iterator = true to ensure
    // proper handling by oneapi::dpl::__ranges::__get_sycl_range
    sycl::buffer<T, dim, Allocator>
    get_buffer() const
    {
        return buffer;
    }

    // This function is required for types for which oneapi::dpl::__ranges::is_hetero_iterator = true to ensure
    // proper handling by oneapi::dpl::__ranges::__get_sycl_range
    Size
    get_idx() const
    {
        return idx;
    }

    // While sycl_iterator cannot be "passed directly" because it is not device_copyable or a random access iterator,
    // it does represent indirectly device accessible data.
    friend std::true_type
    is_onedpl_indirectly_device_accessible(sycl_iterator)
    {
        return {}; //minimal body provided to avoid warnings of non-template-friend in g++
    }
};

// map access_mode tag to access_mode value
// TODO: consider removing the logic for discard_read_write and discard_write which are deprecated in SYCL 2020
template <typename _ModeTagT, typename _NoInitT = void>
struct __access_mode_resolver
{
};

template <typename _NoInitT>
struct __access_mode_resolver<std::decay_t<decltype(sycl::read_only)>, _NoInitT>
{
    static constexpr access_mode __value = access_mode::read;
};

template <typename _NoInitT>
struct __access_mode_resolver<std::decay_t<decltype(sycl::write_only)>, _NoInitT>
{
    static constexpr access_mode __value =
        std::is_same_v<_NoInitT, void> ? access_mode::write : access_mode::discard_write;
};

template <typename _NoInitT>
struct __access_mode_resolver<std::decay_t<decltype(sycl::read_write)>, _NoInitT>
{
    static constexpr access_mode __value =
        std::is_same_v<_NoInitT, void> ? access_mode::read_write : access_mode::discard_read_write;
};

template <typename Iter, typename ValueType = std::decay_t<typename std::iterator_traits<Iter>::value_type>>
using __default_alloc_vec_iter = typename std::vector<ValueType>::iterator;

template <typename Iter, typename ValueType = std::decay_t<typename std::iterator_traits<Iter>::value_type>>
using __usm_shared_alloc_vec_iter =
    typename std::vector<ValueType, typename sycl::usm_allocator<ValueType, sycl::usm::alloc::shared>>::iterator;

template <typename Iter, typename ValueType = std::decay_t<typename std::iterator_traits<Iter>::value_type>>
using __usm_host_alloc_vec_iter =
    typename std::vector<ValueType, typename sycl::usm_allocator<ValueType, sycl::usm::alloc::host>>::iterator;

// Evaluates to true_type if the provided type is an iterator with a value_type and if the implementation of a
// std::vector<value_type, Alloc>::iterator can be distinguished between three different allocators, the
// default, usm_shared, and usm_host. If all are distinct, it is very unlikely any non-usm based allocator
// could be confused with a usm allocator.
template <typename Iter>
using __vector_iter_distinguishes_by_allocator =
    std::conjunction<std::negation<std::is_same<__default_alloc_vec_iter<Iter>, __usm_shared_alloc_vec_iter<Iter>>>,
                     std::negation<std::is_same<__default_alloc_vec_iter<Iter>, __usm_host_alloc_vec_iter<Iter>>>,
                     std::negation<std::is_same<__usm_host_alloc_vec_iter<Iter>, __usm_shared_alloc_vec_iter<Iter>>>>;

template <typename Iter>
inline constexpr bool __vector_iter_distinguishes_by_allocator_v =
    __vector_iter_distinguishes_by_allocator<Iter>::value;

template <typename Iter>
using __is_known_usm_vector_iter =
    std::conjunction<__vector_iter_distinguishes_by_allocator<Iter>,
                     std::disjunction<std::is_same<Iter, oneapi::dpl::__internal::__usm_shared_alloc_vec_iter<Iter>>,
                                      std::is_same<Iter, oneapi::dpl::__internal::__usm_host_alloc_vec_iter<Iter>>>>;

template <typename Iter>
inline constexpr bool __is_known_usm_vector_iter_v = __is_known_usm_vector_iter<Iter>::value;

} // namespace __internal

template <typename T, typename Allocator>
__internal::sycl_iterator<access_mode::read_write, T, Allocator> begin(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_iterator<access_mode::read_write, T, Allocator>{buf, 0};
}

template <typename T, typename Allocator>
__internal::sycl_iterator<access_mode::read_write, T, Allocator> end(sycl::buffer<T, /*dim=*/1, Allocator> buf)
{
    return __internal::sycl_iterator<access_mode::read_write, T, Allocator>{buf, __dpl_sycl::__get_buffer_size(buf)};
}

// begin
template <typename T, typename Allocator, typename ModeTagT>
__internal::sycl_iterator<__internal::__access_mode_resolver<ModeTagT>::__value, T, Allocator>
begin(sycl::buffer<T, /*dim=*/1, Allocator> buf, ModeTagT)
{
    return __internal::sycl_iterator<__internal::__access_mode_resolver<ModeTagT>::__value, T, Allocator>{buf, 0};
}

template <typename T, typename Allocator, typename ModeTagT>
__internal::sycl_iterator<__internal::__access_mode_resolver<ModeTagT, __dpl_sycl::__no_init>::__value, T, Allocator>
begin(sycl::buffer<T, /*dim=*/1, Allocator> buf, ModeTagT, __dpl_sycl::__no_init)
{
    return __internal::sycl_iterator<__internal::__access_mode_resolver<ModeTagT, __dpl_sycl::__no_init>::__value, T,
                                     Allocator>{buf, 0};
}

template <typename T, typename Allocator>
__internal::sycl_iterator<access_mode::discard_read_write, T, Allocator>
    begin(sycl::buffer<T, /*dim=*/1, Allocator> buf, __dpl_sycl::__no_init)
{
    return __internal::sycl_iterator<access_mode::discard_read_write, T, Allocator>{buf, 0};
}

// end
template <typename T, typename Allocator, typename ModeTagT>
__internal::sycl_iterator<__internal::__access_mode_resolver<ModeTagT>::__value, T, Allocator>
end(sycl::buffer<T, /*dim=*/1, Allocator> buf, ModeTagT)
{
    return __internal::sycl_iterator<__internal::__access_mode_resolver<ModeTagT>::__value, T, Allocator>{
        buf, __dpl_sycl::__get_buffer_size(buf)};
}

template <typename T, typename Allocator, typename ModeTagT>
__internal::sycl_iterator<__internal::__access_mode_resolver<ModeTagT, __dpl_sycl::__no_init>::__value, T, Allocator>
end(sycl::buffer<T, /*dim=*/1, Allocator> buf, ModeTagT, __dpl_sycl::__no_init)
{
    return __internal::sycl_iterator<__internal::__access_mode_resolver<ModeTagT, __dpl_sycl::__no_init>::__value, T,
                                     Allocator>{buf, __dpl_sycl::__get_buffer_size(buf)};
}

template <typename T, typename Allocator>
__internal::sycl_iterator<access_mode::discard_read_write, T, Allocator> end(sycl::buffer<T, /*dim=*/1, Allocator> buf,
                                                                             __dpl_sycl::__no_init)
{
    return __internal::sycl_iterator<access_mode::discard_read_write, T, Allocator>{buf,
                                                                                    __dpl_sycl::__get_buffer_size(buf)};
}
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_SYCL_ITERATOR_H
