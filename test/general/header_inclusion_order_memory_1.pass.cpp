// -*- C++ -*-
//===-- header_inclusion_order_memory_1.pass.cpp --------------------------===//
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

#include _PSTL_TEST_HEADER(memory)
#include _PSTL_TEST_HEADER(execution)

#include "support/utils.h"

#if ONEDPL_HAS_RANGE_ALGORITHMS > 202505L
#    include <ranges>
#    include <algorithm> // std::ranges::count
#endif

int
main()
{
#if ONEDPL_HAS_RANGE_ALGORITHMS > 202505L
    constexpr int n = 10;
    std::allocator<int> alloc;
    int* raw_ptr = alloc.allocate(n);
    auto deleter = [&](int* p){ alloc.deallocate(p, n); };
    std::unique_ptr<int, decltype(deleter)> ptr(raw_ptr, deleter);
    std::ranges::subrange subrange(ptr.get(), ptr.get() + n);
    oneapi::dpl::ranges::uninitialized_fill(oneapi::dpl::execution::seq, subrange, 42);
    EXPECT_TRUE(std::ranges::count(subrange, 42) == n, "wrong results in uninitialized_fill");
    oneapi::dpl::ranges::destroy(oneapi::dpl::execution::seq, subrange);
#endif
    return TestUtils::done();
}
