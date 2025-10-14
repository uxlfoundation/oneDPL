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

//
// NOTE: nullptr_t emulation cannot handle a reinterpret_cast to an
// integral type
//
// typedef decltype(nullptr) nullptr_t;

#include "support/test_config.h"

#include <oneapi/dpl/cstddef>

#include "support/test_macros.h"
#include "support/utils.h"

int
main()
{
    const dpl::size_t N = 1;
    bool ret = true;
    {
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{N});
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            auto acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                dpl::ptrdiff_t i = reinterpret_cast<dpl::ptrdiff_t>(nullptr);
                acc[0] &= (i == 0);
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of work with null_ptr integral cast in Kernel");

    return TestUtils::done();
}
