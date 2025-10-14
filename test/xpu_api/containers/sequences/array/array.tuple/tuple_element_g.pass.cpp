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

#include <oneapi/dpl/array>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

bool
kernel_test()
{
    using dpl::array;
    using std::tuple_element;
    // This relies on the fact that <utility> includes <type_traits>:
    using dpl::is_same;

    const size_t len = 3;
    typedef array<int, len> array_type;

    static_assert(is_same<tuple_element<0, array_type>::type, int>::value);
    static_assert(is_same<tuple_element<1, array_type>::type, int>::value);
    static_assert(is_same<tuple_element<2, array_type>::type, int>::value);

    static_assert(is_same<tuple_element<0, const array_type>::type, const int>::value);
    static_assert(is_same<tuple_element<1, const array_type>::type, const int>::value);
    static_assert(is_same<tuple_element<2, const array_type>::type, const int>::value);

    static_assert(is_same<tuple_element<0, volatile array_type>::type, volatile int>::value);
    static_assert(is_same<tuple_element<1, volatile array_type>::type, volatile int>::value);
    static_assert(is_same<tuple_element<2, volatile array_type>::type, volatile int>::value == true);

    static_assert(is_same<tuple_element<0, const volatile array_type>::type, const volatile int>::value);
    static_assert(is_same<tuple_element<1, const volatile array_type>::type, const volatile int>::value);
    static_assert(is_same<tuple_element<2, const volatile array_type>::type, const volatile int>::value);
    return true;
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

    EXPECT_TRUE(ret, "Wrong result of work with std::tuple_element (global)");

    return TestUtils::done();
}
