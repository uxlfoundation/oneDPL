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

#ifndef offsetof
#    error offsetof not defined
#endif

struct A
{
    int x;
};

class KernelTest1;

int
main()
{
    {
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest1>([=]() { ASSERT_NOEXCEPT(offsetof(A, x)); });
        });
    }

    return TestUtils::done();
}
