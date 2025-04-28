//===-- xpu_fill.pass.cpp -------------------------------------------------===//
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

#include <oneapi/dpl/algorithm>

#include "support/utils.h"
#include "support/test_iterators.h"

#include <cassert>

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    const unsigned N = 1000;
    int ia[N] = {0};
    bool ret = true;
    sycl::range<1> itemN{N};
    {
        sycl::buffer<int, 1> buffer1(ia, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto acc_arr1 = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<Iter>([=]() { dpl::fill(Iter(&acc_arr1[0]), Iter(&acc_arr1[0] + N), 100); });
        }).wait();
    }
    assert(ret);
    for (size_t idx = 0; idx < N; ++idx)
    {
        assert(ia[idx] == 100);
    }
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test<forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<int*>>(deviceQueue);
    test<int*>(deviceQueue);
    return TestUtils::done();
}
