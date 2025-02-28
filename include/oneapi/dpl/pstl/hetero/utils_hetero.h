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

// This file contains SYCL specific macros and abstractions
// to support different versions of SYCL and to simplify its interfaces
//
// Include this header instead of sycl.hpp throughout the project

#ifndef _ONEDPL_UTILS_HETERO_H
#define _ONEDPL_UTILS_HETERO_H

namespace oneapi
{
namespace dpl
{
namespace __internal
{

template <typename _Pred>
struct equal_predicate
{
    _Pred __pred;

    template <typename _Value>
    bool
    operator()(const _Value& __val) const
    {
        using ::std::get;
        return !__pred(get<0>(__val), get<1>(__val));
    }
};

template <typename _Predicate>
struct adjacent_find_fn
{
    _Predicate __predicate;

    // the functor is being used instead of a lambda because
    // at this level we don't know what type we get during zip_iterator unpack
    template <typename _Pack>
    bool
    operator()(const _Pack& __packed_neighbor_values) const
    {
        using ::std::get;
        return __predicate(get<0>(__packed_neighbor_values), get<1>(__packed_neighbor_values));
    }
};

template <typename _Predicate, typename _ValueType>
struct __create_mask_unique_copy
{
    _Predicate __predicate;

    template <typename _Idx, typename _Acc>
    _ValueType
    operator()(_Idx __idx, _Acc& __acc) const
    {
        using ::std::get;

        auto __predicate_result = 1;
        if (__idx != 0)
            __predicate_result = __predicate(get<0>(__acc[__idx]), get<0>(__acc[__idx + (-1)]));

        get<1>(__acc[__idx]) = __predicate_result;
        return _ValueType{__predicate_result};
    }
};

template <typename __Queue, typename Handler, typename Event>
inline void
__depends_on(const __Queue& __queue, Handler& __cgh, const Event& __event)
{
    if (!__queue.is_in_order())
    {
        __cgh.depends_on(__event);
    }
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_UTILS_HETERO_H
