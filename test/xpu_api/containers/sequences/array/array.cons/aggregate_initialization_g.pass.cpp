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

#include "support/utils.h"

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                const size_t len = 5;
                typedef dpl::array<int, len> array_type;
                array_type a = {{0, 1, 2, 3, 4}};
                array_type b = {{0, 1, 2, 3}};
                a = b;
            });
        });
    }
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
