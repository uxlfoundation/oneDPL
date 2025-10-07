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
#include <oneapi/dpl/functional>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

template <class Tuple>
bool
test1a(const Tuple& t)
{
    static_assert(dpl::tuple_size<Tuple>::value == 1);
    static_assert(dpl::is_same<typename std::tuple_element<0, Tuple>::type, int&&>::value);
    return (dpl::get<0>(t) == 1);
}

template <class Tuple>
bool
test1b(const Tuple& t)
{
    static_assert(dpl::tuple_size<Tuple>::value == 1);
    static_assert(dpl::is_same<typename std::tuple_element<0, Tuple>::type, int&>::value);
    return (dpl::get<0>(t) == 2);
}

template <class Tuple>
bool
test2a(const Tuple& t)
{
    static_assert(dpl::tuple_size<Tuple>::value == 2);
    static_assert(dpl::is_same<typename std::tuple_element<0, Tuple>::type, float&&>::value);
    static_assert(dpl::is_same<typename std::tuple_element<1, Tuple>::type, char&>::value);
    return (dpl::get<0>(t) == 2.5f && dpl::get<1>(t) == 'a');
}

class KernelForwardAsTupleTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelForwardAsTupleTest>([=]() {
            int i = 2;
            const auto tpl = dpl::forward_as_tuple();
            static_assert(std::tuple_size_v<decltype(tpl)> == 0);
            ret_access[0] = test1a(dpl::forward_as_tuple(1));
            ret_access[0] &= test1b(dpl::forward_as_tuple(i));

            char c = 'a';
            ret_access[0] &= test2a(dpl::forward_as_tuple(2.5f, c));

            const auto tpl1 = dpl::forward_as_tuple(2.5f, c);
            static_assert(std::tuple_size_v<decltype(tpl1)> == 2);
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::forward_as_tuple check");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
