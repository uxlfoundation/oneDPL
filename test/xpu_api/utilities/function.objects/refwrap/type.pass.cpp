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

#include <oneapi/dpl/functional>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

class C
{
};

class KernelTypePassTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelTypePassTest>([=]() {
            // Static assert check...

            static_assert(dpl::is_same<dpl::reference_wrapper<C>::type, C>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<void()>::type, void()>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<int*(float*)>::type, int*(float*)>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<void (*)()>::type, void (*)()>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<int* (*)(float*)>::type, int* (*)(float*)>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<int* (C::*)(float*)>::type, int* (C::*)(float*)>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<int (C::*)(float*) const volatile>::type,
                                       int (C::*)(float*) const volatile>::value);
            // Runtime check...

            ret_access[0] = dpl::is_same<dpl::reference_wrapper<C>::type, C>::value;
            ret_access[0] &= dpl::is_same<dpl::reference_wrapper<void()>::type, void()>::value;
            ret_access[0] &= dpl::is_same<dpl::reference_wrapper<int*(float*)>::type, int*(float*)>::value;
            ret_access[0] &= dpl::is_same<dpl::reference_wrapper<void (*)()>::type, void (*)()>::value;
            ret_access[0] &= dpl::is_same<dpl::reference_wrapper<int* (*)(float*)>::type, int* (*)(float*)>::value;
            ret_access[0] &=
                dpl::is_same<dpl::reference_wrapper<int* (C::*)(float*)>::type, int* (C::*)(float*)>::value;
            ret_access[0] &= dpl::is_same<dpl::reference_wrapper<int (C::*)(float*) const volatile>::type,
                                          int (C::*)(float*) const volatile>::value;
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Error in work with dpl::reference_wrapper and type checks");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
