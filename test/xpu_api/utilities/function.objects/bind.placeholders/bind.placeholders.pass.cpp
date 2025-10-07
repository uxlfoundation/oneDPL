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

#include "support/utils.h"

class Object
{
  public:
    void
    operator()(int a, int b, int c, int& i)
    {
        i += p;
        i += a;
        i *= b;
        i -= c;
    }

  private:
    int p = 10;
};

class KernelTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::range<1> numOfItems{1};
    std::int32_t result = 10;
    {
        sycl::buffer<std::int32_t, 1> buffer1(&result, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto res_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                Object instance;
                auto bf = dpl::bind(instance, 1, 2, 3, dpl::ref(res_access[0]));
                bf();
            });
        });
    }

    EXPECT_TRUE(result == 39, "Error in work with dpl::bind");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
