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

template <class T, class U>
void
test_is_base_of()
{
    static_assert(dpl::is_base_of<T, U>::value);
    static_assert(dpl::is_base_of<const T, U>::value);
    static_assert(dpl::is_base_of<T, const U>::value);
    static_assert(dpl::is_base_of<const T, const U>::value);
    static_assert(dpl::is_base_of_v<T, U>);
    static_assert(dpl::is_base_of_v<const T, U>);
    static_assert(dpl::is_base_of_v<T, const U>);
    static_assert(dpl::is_base_of_v<const T, const U>);
}

template <class T, class U>
void
test_is_not_base_of()
{
    static_assert(!dpl::is_base_of<T, U>::value);
}

struct B
{
};
struct B1 : B
{
};
struct B2 : B
{
};
struct D : private B1, private B2
{
};
struct I0; // incomplete

bool
kernel_test()
{
    test_is_base_of<B, D>();
    test_is_base_of<B1, D>();
    test_is_base_of<B2, D>();
    test_is_base_of<B, B1>();
    test_is_base_of<B, B2>();
    test_is_base_of<B, B>();

    test_is_not_base_of<D, B>();
    test_is_not_base_of<B&, D&>();
    test_is_not_base_of<B[3], D[3]>();
    test_is_not_base_of<int, int>();

    //  A scalar is never the base class of anything (including incomplete types)
    test_is_not_base_of<int, B>();
    test_is_not_base_of<int, B1>();
    test_is_not_base_of<int, B2>();
    test_is_not_base_of<int, D>();
    test_is_not_base_of<int, I0>();

    //  A scalar never has base classes (including incomplete types)
    test_is_not_base_of<B, int>();
    test_is_not_base_of<B1, int>();
    test_is_not_base_of<B2, int>();
    test_is_not_base_of<D, int>();
    test_is_not_base_of<I0, int>();

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

    EXPECT_TRUE(ret, "Wrong result of work with dpl::is_base_of");

    return TestUtils::done();
}
