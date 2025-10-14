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

#include "support/test_config.h"

#include <oneapi/dpl/tuple>
#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

template <typename T>
typename dpl::decay<T>::type
copy(T&& x)
{
    return dpl::forward<T>(x);
}

template <typename... Args1, typename... Args2>
bool
check_tuple_cat(dpl::tuple<Args1...> t1, dpl::tuple<Args2...> t2)
{
    typedef dpl::tuple<Args1..., Args2...> concatenated;

    auto cat1 = dpl::tuple_cat(t1, t2);
    auto cat2 = dpl::tuple_cat(copy(t1), t2);
    auto cat3 = dpl::tuple_cat(t1, copy(t2));
    auto cat4 = dpl::tuple_cat(copy(t1), copy(t2));

    static_assert(dpl::is_same<decltype(cat1), concatenated>::value);
    static_assert(dpl::is_same<decltype(cat2), concatenated>::value);
    static_assert(dpl::is_same<decltype(cat3), concatenated>::value);
    static_assert(dpl::is_same<decltype(cat4), concatenated>::value);

    auto ret = (cat1 == cat2);
    ret &= (cat1 == cat3);
    ret &= (cat1 == cat4);
    return ret;
}

bool
kernel_test()
{
    int i = 0;
    dpl::tuple<> t0;
    dpl::tuple<int&> t1(i);
    dpl::tuple<int&, int> t2(i, 0);
    dpl::tuple<int const&, int, float> t3(i, 0, 0.f);

    auto ret = check_tuple_cat(t0, t0);
    ret &= check_tuple_cat(t0, t1);
    ret &= check_tuple_cat(t0, t2);
    ret &= check_tuple_cat(t0, t3);

    ret &= check_tuple_cat(t1, t0);
    ret &= check_tuple_cat(t1, t1);
    ret &= check_tuple_cat(t1, t2);
    ret &= check_tuple_cat(t1, t3);

    ret &= check_tuple_cat(t2, t0);
    ret &= check_tuple_cat(t2, t1);
    ret &= check_tuple_cat(t2, t2);
    ret &= check_tuple_cat(t2, t3);

    ret &= check_tuple_cat(t3, t0);
    ret &= check_tuple_cat(t3, t1);
    ret &= check_tuple_cat(t3, t2);
    ret &= check_tuple_cat(t3, t3);
    return ret;
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }
    EXPECT_TRUE(ret, "Wrong result of dpl::tuple_cat check in kernel_test");

    return TestUtils::done();
}
