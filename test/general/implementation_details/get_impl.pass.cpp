// -*- C++ -*-
//===-- get_impl.pass.cpp --------------------------------------------------===//
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

// Attention: we should include this header before including get_impl.h
#include "UserDataType.h"

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(tuple)

#include <oneapi/dpl/pstl/tuple_impl.h> // for oneapi::dpl::__internal::tuple
#include <oneapi/dpl/pstl/get_impl.h>   // for oneapi::dpl::__internal::__get

#include "support/utils.h"

template <typename... _T>
static oneapi::dpl::__internal::tuple<_T...>
to_onedpl_tuple(const std::tuple<_T...>& __t)
{
    return oneapi::dpl::__internal::tuple<_T...>(__t);
}

template <typename TData, typename T1, typename T2>
void
test_get_data(TData&& data, T1 val1, T2 val2)
{
    EXPECT_EQ(val1, __dpl_internal::__get<0>(std::forward<TData>(data)), "Incorrect get data #1");
    EXPECT_EQ(val2, __dpl_internal::__get<1>(std::forward<TData>(data)), "Incorrect get data #2");
}

template <typename TData, typename T1, typename T2>
void
test_set_data(TData&& data, T1 val1, T2 val2)
{
    __dpl_internal::__get<0>(std::forward<TData>(data)) = val1;
    __dpl_internal::__get<1>(std::forward<TData>(data)) = val2;

    // Check that data is set correctly
    test_get_data(std::forward<TData>(data), val1, val2);
}

int
main()
{
    // const std::tuple - read data
    const auto std_tuple_t1 = std::make_tuple(1, 2);
    test_get_data(std_tuple_t1, 1, 2);

    // std::tuple - read + modify data
    auto std_tuple_t2 = std::make_tuple(1, 2);
    test_get_data(std_tuple_t2, 1, 2);
    test_set_data(std_tuple_t2, 3, 4);

    // const oneapi::dpl::__internal::tuple - read data
    const auto onedpl_t1 = to_onedpl_tuple(std::make_tuple(1, 2));
    test_get_data(onedpl_t1, 1, 2);

    // oneapi::dpl::__internal::tuple - read + modify data
    auto onedpl_t2 = to_onedpl_tuple(std::make_tuple(1, 2));
    test_get_data(onedpl_t2, 1, 2);
    test_set_data(onedpl_t2, 3, 4);

    // const User data type - read data
    const UserDataType udt1{1, 2};
    test_get_data(udt1, 1, 2);

    // User data type - read + modify data
    UserDataType udt2{1, 2};
    test_get_data(udt2, 1, 2);
    test_set_data(udt2, 3, 4);

    return TestUtils::done();
}
