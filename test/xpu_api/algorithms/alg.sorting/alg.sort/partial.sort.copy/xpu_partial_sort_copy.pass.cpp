//===-- xpu_partial_sort_copy.pass.cpp ------------------------------------===//
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
#include <oneapi/dpl/functional>

#include "support/utils.h"
#include "support/test_iterators.h"

#include <cassert>

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    using T = typename std::iterator_traits<Iter>::value_type;
    bool ret = true;
    sycl::range<1> item1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<Iter>([=]() {
                {
                    T a[] = {0, 2, 33, 52, 8, 9};
                    T b[] = {-1, -1};
                    unsigned sb = sizeof(b) / sizeof(b[0]);
                    dpl::partial_sort_copy(Iter(a), Iter(a + 2), Iter(b), Iter(b + sb));
                    ret_acc[0] &= (b[0] == 0 && b[1] == 2);
                }
                {
                    T a[] = {1, 0, 34, 2, 8, 7};
                    T b[] = {-1, -1, -1, -1};
                    unsigned sb = sizeof(b) / sizeof(b[0]);
                    dpl::partial_sort_copy(Iter(a), Iter(a + 3), Iter(b), Iter(b + sb));
                    ret_acc[0] &= (b[0] == 0 && b[1] == 1 && b[2] == 34 && b[3] == -1);
                }
            });
        }).wait();
    }
    assert(ret);
}
int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test<random_access_iterator<int*>>(deviceQueue);
    test<int*>(deviceQueue);
    return TestUtils::done();
}
