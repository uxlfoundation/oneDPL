// -*- C++ -*-
//===-- tuple_unit.pass.cpp -----------------------------------------------===//
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
#include <cstdint>
#include "support/test_config.h"

#include _PSTL_TEST_HEADER(tuple)

#include "support/utils.h"

template <typename... _T>
static oneapi::dpl::__internal::tuple<_T...>
to_onedpl_tuple(const std::tuple<_T...>& __t)
{
    return oneapi::dpl::__internal::tuple<_T...>(__t);
}

template <typename Tuple1, typename Tuple2>
void
test_tuple(Tuple1 t1, Tuple2 t2)
{

    auto onedpl_t1 = to_onedpl_tuple(t1);
    auto onedpl_t2 = to_onedpl_tuple(t2);

    static_assert(std::is_trivially_copyable_v<decltype(onedpl_t1)>, "oneDPL tuple is not trivially copyable");

    // Test binary comparison operators for std::tuple and oneAPI::dpl::__internal::tuple
    EXPECT_EQ((t1 == t2), (onedpl_t1 == onedpl_t2), "equality comparison does not match std::tuple");
    EXPECT_EQ((t1 != t2), (onedpl_t1 != onedpl_t2), "inquality comparison does not match std::tuple");
    EXPECT_EQ((t1 < t2), (onedpl_t1 < onedpl_t2), "less than comparison does not match std::tuple");
    EXPECT_EQ((t1 <= t2), (onedpl_t1 <= onedpl_t2), "less than or equal to comparison does not match std::tuple");
    EXPECT_EQ((t1 > t2), (onedpl_t1 > onedpl_t2), "greater than comparison does not match std::tuple");
    EXPECT_EQ((t1 >= t2), (onedpl_t1 >= onedpl_t2), "greater than or equal to comparison does not match std::tuple");

    EXPECT_EQ((t1 == t2), (t1 == onedpl_t2), "equality comparison does not match std::tuple");
    EXPECT_EQ((t1 != t2), (t1 != onedpl_t2), "inquality comparison does not match std::tuple");
    EXPECT_EQ((t1 < t2), (t1 < onedpl_t2), "less than comparison does not match std::tuple");
    EXPECT_EQ((t1 <= t2), (t1 <= onedpl_t2), "less than or equal to comparison does not match std::tuple");
    EXPECT_EQ((t1 > t2), (t1 > onedpl_t2), "greater than comparison does not match std::tuple");
    EXPECT_EQ((t1 >= t2), (t1 >= onedpl_t2), "greater than or equal to comparison does not match std::tuple");

    EXPECT_EQ((t1 == t2), (onedpl_t1 == t2), "equality comparison does not match std::tuple");
    EXPECT_EQ((t1 != t2), (onedpl_t1 != t2), "inquality comparison does not match std::tuple");
    EXPECT_EQ((t1 < t2), (onedpl_t1 < t2), "less than comparison does not match std::tuple");
    EXPECT_EQ((t1 <= t2), (onedpl_t1 <= t2), "less than or equal to comparison does not match std::tuple");
    EXPECT_EQ((t1 > t2), (onedpl_t1 > t2), "greater than comparison does not match std::tuple");
    EXPECT_EQ((t1 >= t2), (onedpl_t1 >= t2), "greater than or equal to comparison does not match std::tuple");

    //checking default construction (assumes elements are default constructible)
    decltype(onedpl_t1) onedpl_t3{};
    decltype(onedpl_t2) onedpl_t4{};
    decltype(t1) t3{};
    decltype(t2) t4{};

    t4 = onedpl_t1;
    EXPECT_TRUE(t1 == t4, "assignment of oneDPL tuple to std::tuple provides incorrect results");

    t3 = onedpl_t2;
    EXPECT_TRUE(t2 == t3, "assignment of oneDPL tuple to std::tuple provides incorrect results");

    onedpl_t3 = t2;
    EXPECT_TRUE(onedpl_t2 == onedpl_t3, "assignment of oneDPL tuple from std::tuple provides incorrect results");

    onedpl_t4 = t1;
    EXPECT_TRUE(onedpl_t1 == onedpl_t4, "assignment of oneDPL tuple from std::tuple provides incorrect results");

    decltype(onedpl_t1) onedpl_t5 = onedpl_t2;
    decltype(onedpl_t1) onedpl_t6 = onedpl_t1;

    swap(onedpl_t5, onedpl_t6);
    EXPECT_TRUE(((onedpl_t1 == onedpl_t5) && (onedpl_t6 == onedpl_t2)),
                "swap of oneDPL tuple provides incorrect results");
}

int
main()
{
    test_tuple(std::tuple<int, int, int>{1, 2, 3}, std::tuple<int, int, int>{1, 2, 3});
    test_tuple(std::tuple<int, int, int>{1, 2, 3}, std::tuple<int, int, int>{1, 2, 4});
    test_tuple(std::tuple<int, int, int>{1, 2, 3}, std::tuple<int, int, int>{0, 2, 4});
    test_tuple(std::tuple<int, int, std::int64_t>{1, 2, 3}, std::tuple<int, int, std::int64_t>{1, 2, 3});
    test_tuple(std::tuple<std::uint32_t, std::uint32_t, std::uint64_t>{1, 2, 3}, std::tuple<std::uint64_t, std::uint32_t, std::uint32_t>{0, 2, 4});
    test_tuple(std::tuple<>{}, std::tuple<>{});

    return TestUtils::done();
}
