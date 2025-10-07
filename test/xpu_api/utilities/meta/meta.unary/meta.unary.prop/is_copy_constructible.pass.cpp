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
#include "support/utils_invoke.h"

template <class T>
void
test_is_copy_constructible(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(dpl::is_copy_constructible<T>::value);
            static_assert(dpl::is_copy_constructible_v<T>);
        });
    });
}

template <class T>
void
test_is_not_copy_constructible(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!dpl::is_copy_constructible<T>::value);
            static_assert(!dpl::is_copy_constructible_v<T>);
        });
    });
}

class Empty
{
};

union Union
{
};

struct bit_zero
{
    int : 0;
};

struct A
{
    A(const A&);
};

class B
{
    B(const B&);
};

struct C
{
    C(C&); // not const
    void
    operator=(C&); // not const
};

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_is_copy_constructible<A>(deviceQueue);
    test_is_copy_constructible<int&>(deviceQueue);
    test_is_copy_constructible<Union>(deviceQueue);
    test_is_copy_constructible<Empty>(deviceQueue);
    test_is_copy_constructible<int>(deviceQueue);
    test_is_copy_constructible<int*>(deviceQueue);
    test_is_copy_constructible<const int*>(deviceQueue);
    test_is_copy_constructible<bit_zero>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_is_copy_constructible<double>(deviceQueue);
    }

    test_is_not_copy_constructible<char[3]>(deviceQueue);
    test_is_not_copy_constructible<char[]>(deviceQueue);
    test_is_not_copy_constructible<void>(deviceQueue);
    test_is_not_copy_constructible<C>(deviceQueue);
    test_is_not_copy_constructible<B>(deviceQueue);
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
