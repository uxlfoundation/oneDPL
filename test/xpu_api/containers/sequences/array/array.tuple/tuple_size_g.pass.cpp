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
    using dpl::tuple_size;
    // This relies on the fact that <utility> includes <type_traits>:
    using dpl::is_same;

    {
        const size_t len = 5;
        typedef array<int, len> array_type;
        static_assert(tuple_size<array_type>::value == 5);
        static_assert(tuple_size<const array_type>::value == 5);
        static_assert(tuple_size<volatile array_type>::value == 5);
        static_assert(tuple_size<const volatile array_type>::value == 5);
    }

    {
        const size_t len = 0;
        typedef array<float, len> array_type;
        static_assert(tuple_size<array_type>::value == 0);
        static_assert(tuple_size<const array_type>::value == 0);
        static_assert(tuple_size<volatile array_type>::value == 0);
        static_assert(tuple_size<const volatile array_type>::value == 0);
    }
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

    EXPECT_TRUE(ret, "Wrong result of work with dpl::tuple_size (global)");

    return TestUtils::done();
}
