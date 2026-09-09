// -*- C++ -*-
//===-- function.h ---------------------------------------------------------===//
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

#ifndef _ONEDPL_INTERNAL_FUNCTION_H
#define _ONEDPL_INTERNAL_FUNCTION_H

#include <utility>
#if _ONEDPL_BACKEND_SYCL
#    include "../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
#endif
#include "../functional"
#include <tuple>

namespace oneapi
{
namespace dpl
{
namespace __internal
{

// Used by: reduce_by_segment on host
template <typename Output1, typename Output2>
class scatter_and_accumulate_fun
{
  public:
    scatter_and_accumulate_fun(Output1 _result1, Output2 _result2) : result1(_result1), result2(_result2) {}

    template <typename _T>
    void
    operator()(_T&& x) const
    {
        using ::std::get;
        if (::std::get<2>(x))
        {
            result1[::std::get<1>(x)] = ::std::get<0>(x);
        }
        if (::std::get<4>(x))
        {
            result2[::std::get<1>(x)] = ::std::get<3>(x);
        }
    }

  private:
    Output1 result1;
    Output2 result2;
};

// Used by: *_by_segment algorithms
template <typename _ValueType, typename _FlagType, typename _BinaryOp>
struct __segmented_scan_fun
{
    template <typename _T1, typename _T2>
    _T1
    operator()(const _T1& __x, const _T2& __y) const
    {
        using std::get;
        using __x_t = std::tuple_element_t<0, _T1>;
        auto __new_x = get<1>(__y) ? __x_t(get<0>(__y)) : __x_t(__binary_op(get<0>(__x), get<0>(__y)));
        auto __new_y = get<1>(__x) | get<1>(__y);
        return _T1(__new_x, __new_y);
    }

    _BinaryOp __binary_op;
};

} // namespace __internal
} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL_INTERNAL_FUNCTION_H
