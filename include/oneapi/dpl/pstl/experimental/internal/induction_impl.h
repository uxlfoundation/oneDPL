// -*- C++ -*-
//===-- induction_impl.h --------------------------------------------------===//
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

#ifndef _ONEDPL_EXPERIMENTAL_INDUCTION_IMPL_H
#define _ONEDPL_EXPERIMENTAL_INDUCTION_IMPL_H

#include <type_traits>

namespace oneapi
{
namespace dpl
{
namespace __internal
{

// Maps _Tp according to following rules:
// _Tp& -> _Tp&
// const _Tp& -> _Tp
// _Tp -> _Tp
// We're not interested in T&& here as the rvalue-ness is stripped from T before
// construction the induction object
template <typename _Tp>
using __induction_value_type =
    ::std::conditional_t<::std::is_lvalue_reference_v<_Tp> && !::std::is_const_v<::std::remove_reference_t<_Tp>>, _Tp,
                         ::std::remove_cv_t<::std::remove_reference_t<_Tp>>>;

// Definition of induction_object structure to represent "induction" object.

template <typename _Tp, typename _Sp>
class __induction_object
{
    using __value_type = __induction_value_type<_Tp>;

    __value_type __var_;
    const _Sp __stride_;

  public:
    __induction_object(__value_type __var, _Sp __stride) : __var_(__var), __stride_(__stride) {}

    __induction_object(const __induction_object& __other) = default;

    __induction_object&
    operator=(const __induction_object& __other)
    {
        __var_ = __other.__var_;
        /* stride is always const */
        return *this;
    }

    template <typename _Index>
    ::std::remove_reference_t<__value_type>
    __get_induction_or_reduction_value(_Index __p)
    {
        return __var_ + __p * __stride_;
    }

    void
    __combine(const __induction_object&)
    {
    }

    template <typename _RangeSize>
    void
    __finalize(const _RangeSize __n)
    {
        // This value is discarded if var is not a reference
        __var_ = __n * __stride_;
    }
};

template <typename _Tp>
class __induction_object<_Tp, void>
{
    using __value_type = __induction_value_type<_Tp>;

    __value_type __var_;

  public:
    __induction_object(__value_type __var) : __var_(__var) {}

    __induction_object(const __induction_object& __other) = default;

    __induction_object&
    operator=(const __induction_object& __other)
    {
        __var_ = __other.__var_;
        return *this;
    }

    template <typename _Index>
    ::std::remove_reference_t<__value_type>
    __get_induction_or_reduction_value(_Index __p)
    {
        return __var_ + __p;
    }

    void
    __combine(const __induction_object&)
    {
    }

    template <typename _RangeSize>
    void
    __finalize(const _RangeSize __n)
    {
        // This value is discarded if var is not a reference
        __var_ = __n;
    }
};

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_EXPERIMENTAL_INDUCTION_IMPL_H
