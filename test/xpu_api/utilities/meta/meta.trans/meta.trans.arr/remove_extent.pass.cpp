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

// type_traits

// remove_extent

#include "support/test_config.h"

#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

enum Enum
{
    zero,
    one_
};

template <class T, class U>
void
test_remove_extent()
{
    ASSERT_SAME_TYPE(U, typename dpl::remove_extent<T>::type);
    ASSERT_SAME_TYPE(U, dpl::remove_extent_t<T>);
}

bool
kernel_test()
{
    test_remove_extent<int, int>();
    test_remove_extent<const Enum, const Enum>();
    test_remove_extent<int[], int>();
    test_remove_extent<const int[], const int>();
    test_remove_extent<int[3], int>();
    test_remove_extent<const int[3], const int>();
    test_remove_extent<int[][3], int[3]>();
    test_remove_extent<const int[][3], const int[3]>();
    test_remove_extent<int[2][3], int[3]>();
    test_remove_extent<const int[2][3], const int[3]>();
    test_remove_extent<int[1][2][3], int[2][3]>();
    test_remove_extent<const int[1][2][3], const int[2][3]>();

    return true;
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

    EXPECT_TRUE(ret, "Wrong result of work with dpl::remove_extent");

    return TestUtils::done();
}
