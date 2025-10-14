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

// To fix compilation issues in case libstdc++ version 9 or 10, for details see oneAPI DPC++ Library Known Limitations.
#include <oneapi/dpl/cstddef>

#include "support/test_macros.h"
#include "support/utils.h"

class Test1;

int
main()
{
    bool ret = false;
    {
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{1});
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            auto acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<Test1>([=]() {
#ifndef NULL
#    error NULL not defined
#else
                acc[0] = true;
#endif
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result: NULL is not defined in Kernel");

    return TestUtils::done();
}
