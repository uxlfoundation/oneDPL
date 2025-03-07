// -*- C++ -*-
//===-- reduce.pass.cpp ---------------------------------------------------===//
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
#include "support/utils.h"

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include <iostream>

int main()
{
    sycl::queue q;
    int n = 10;
    int *ptr = sycl::malloc_shared<int>(n, q);
    q.fill(ptr, 1, n).wait();

    oneapi::dpl::execution::device_policy<class kernel> policy{q};

    auto res1 = oneapi::dpl::reduce(policy, ptr, ptr + n);
    auto res2 = oneapi::dpl::reduce(policy, ptr, ptr + n);
    auto res3 = oneapi::dpl::reduce(std::move(policy), ptr, ptr + n);

    std::cout << res1 << " " << res2 << " " << res3 << std::endl;

    return 0;
}