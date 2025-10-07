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

#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

bool
kernel_test()
{
    bool ret = false;
    typedef dpl::integral_constant<int, 5> _5;
    static_assert(_5::value == 5);
    static_assert(dpl::is_same<_5::value_type, int>::value);
    static_assert(dpl::is_same<_5::type, _5>::value);
    static_assert(_5() == 5);
    ret = (_5() == 5);

    static_assert(_5{}() == 5);
    static_assert(dpl::true_type{}());

    static_assert(dpl::false_type::value == false);
    static_assert(dpl::is_same<dpl::false_type::value_type, bool>::value);
    static_assert(dpl::is_same<dpl::false_type::type, dpl::false_type>::value);

    static_assert(dpl::true_type::value == true);
    static_assert(dpl::is_same<dpl::true_type::value_type, bool>::value);
    static_assert(dpl::is_same<dpl::true_type::type, dpl::true_type>::value);

    dpl::false_type f1;
    dpl::false_type f2 = f1;
    ret &= (!f2);

    dpl::true_type t1;
    dpl::true_type t2 = t1;
    ret &= (t2);

    return ret;
}

class KernelTest;

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

    EXPECT_TRUE(ret, "Wrong result of work with integral constant");

    return TestUtils::done();
}
