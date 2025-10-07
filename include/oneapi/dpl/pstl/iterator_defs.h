// -*- C++ -*-
//===-- iterator_defs.h ---------------------------------------------------===//
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

#ifndef _ONEDPL_ITERATOR_DEFS_H
#define _ONEDPL_ITERATOR_DEFS_H

#include <iterator>
#include <type_traits>
#include "utils.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{

// If std::iterator_traits<_IteratorType>::iterator_category is well-formed - compare it with _IteratorTag
template <typename _IteratorTag, typename _IteratorType, typename = void>
struct __is_iterator_of_category_dispatch : std::false_type
{
};

template <typename _IteratorTag, typename _IteratorType>
struct __is_iterator_of_category_dispatch<_IteratorTag, _IteratorType,
                                          std::void_t<typename std::iterator_traits<_IteratorType>::iterator_category>>
    : std::is_base_of<_IteratorTag, typename std::iterator_traits<_IteratorType>::iterator_category>
{
};

// If _IteratorType::iterator_concept is well-formed - compare it with _IteratorTag
template <typename _IteratorTag, typename _IteratorType, typename = void>
struct __is_iterator_of_concept_dispatch : __is_iterator_of_category_dispatch<_IteratorTag, _IteratorType>
{
};

template <typename _IteratorTag, typename _IteratorType>
struct __is_iterator_of_concept_dispatch<_IteratorTag, _IteratorType,
                                         std::void_t<typename _IteratorType::iterator_concept>>
    : std::is_base_of<_IteratorTag, typename _IteratorType::iterator_concept>
{
};

// Since C++20 allows user specializations for std::iterator_traits to define
// iterator_concept alias into the actual iterator category, we need to check it first
// Primary template std::iterator_traits::iterator_concept does not provide iterator_concept alias
template <typename _IteratorTag, typename _IteratorType, typename = void>
struct __is_iterator_of_traits_concept_dispatch : __is_iterator_of_concept_dispatch<_IteratorTag, _IteratorType>
{
};

template <typename _IteratorTag, typename _IteratorType>
struct __is_iterator_of_traits_concept_dispatch<
    _IteratorTag, _IteratorType, std::void_t<typename std::iterator_traits<_IteratorType>::iterator_concept>>
    : std::is_base_of<_IteratorTag, typename std::iterator_traits<_IteratorType>::iterator_concept>
{
};

// Until C++23, std::move_iterator::iterator_concept is std::input_iterator_tag and
// std::random_access_iterator<std::move_iterator<...>> returns false even if the
// base iterator is random access. Making the dispatch based on the base iterator type
// considers std::move_iterator<random-access> a random access iterator
template <typename _IteratorTag, typename _IteratorType>
struct __is_iterator_of_move_dispatch : __is_iterator_of_traits_concept_dispatch<_IteratorTag, _IteratorType>
{
};

template <typename _IteratorTag, typename _BaseIteratorType>
struct __is_iterator_of_move_dispatch<_IteratorTag, std::move_iterator<_BaseIteratorType>>
    : __is_iterator_of_traits_concept_dispatch<_IteratorTag, _BaseIteratorType>
{
};

template <typename _IteratorTag, typename... _IteratorTypes>
struct __is_iterator_of
    : std::conjunction<__is_iterator_of_move_dispatch<_IteratorTag, std::decay_t<_IteratorTypes>>...>
{
};

// Make is_random_access_iterator and is_forward_iterator not to fail with a 'hard' error when it's used in
// SFINAE with a non-iterator type by providing a default value.
template <typename... _IteratorTypes>
struct __is_random_access_iterator : __is_iterator_of<std::random_access_iterator_tag, _IteratorTypes...>
{
};

template <typename... _IteratorTypes>
struct __is_forward_iterator : __is_iterator_of<std::forward_iterator_tag, _IteratorTypes...>
{
};

template <typename... _IteratorTypes>
inline constexpr bool __is_random_access_iterator_v = __is_random_access_iterator<_IteratorTypes...>::value;

template <typename... _IteratorTypes>
inline constexpr bool __is_forward_iterator_v = __is_forward_iterator<_IteratorTypes...>::value;

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_ITERATOR_DEFS_H
