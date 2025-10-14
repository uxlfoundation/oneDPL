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

template <class T, unsigned A>
void
test_rank(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<T>([=]() {
            static_assert(dpl::rank<T>::value == A);
            static_assert(dpl::rank<const T>::value == A);
            static_assert(dpl::rank<volatile T>::value == A);
            static_assert(dpl::rank<const volatile T>::value == A);

            static_assert(dpl::rank_v<T> == A);
            static_assert(dpl::rank_v<const T> == A);
            static_assert(dpl::rank_v<volatile T> == A);
            static_assert(dpl::rank_v<const volatile T> == A);
        });
    });
}

struct Class
{
    ~Class();
};

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test_rank<void, 0>(deviceQueue);
    test_rank<int&, 0>(deviceQueue);
    test_rank<Class, 0>(deviceQueue);
    test_rank<int*, 0>(deviceQueue);
    test_rank<const int*, 0>(deviceQueue);
    test_rank<int, 0>(deviceQueue);
    test_rank<bool, 0>(deviceQueue);
    test_rank<unsigned, 0>(deviceQueue);
    test_rank<char[3], 1>(deviceQueue);
    test_rank<char[][3], 2>(deviceQueue);
    test_rank<char[][4][3], 3>(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        test_rank<double, 0>(deviceQueue);
    }
}

int
main()
{
    kernel_test();

    return 0;
}
