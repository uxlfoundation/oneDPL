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
test_has_nothrow_assign(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(dpl::is_nothrow_copy_assignable<T>::value);
            static_assert(dpl::is_nothrow_copy_assignable_v<T>);
        });
    });
}

template <class T>
void
test_has_not_nothrow_assign(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(!dpl::is_nothrow_copy_assignable<T>::value);
            static_assert(!dpl::is_nothrow_copy_assignable_v<T>);
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
    A&
    operator=(const A&);
};

struct ANT
{
    ANT&
    operator=(const ANT&) noexcept;
};

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_has_nothrow_assign<int&>(deviceQueue);
    test_has_nothrow_assign<Union>(deviceQueue);
    test_has_nothrow_assign<Empty>(deviceQueue);
    test_has_nothrow_assign<ANT>(deviceQueue);
    test_has_nothrow_assign<int>(deviceQueue);
    test_has_nothrow_assign<int*>(deviceQueue);
    test_has_nothrow_assign<const int*>(deviceQueue);
    test_has_nothrow_assign<bit_zero>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_has_nothrow_assign<double>(deviceQueue);
    }

    test_has_not_nothrow_assign<const int>(deviceQueue);
    test_has_not_nothrow_assign<void>(deviceQueue);
    test_has_not_nothrow_assign<A>(deviceQueue);
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
