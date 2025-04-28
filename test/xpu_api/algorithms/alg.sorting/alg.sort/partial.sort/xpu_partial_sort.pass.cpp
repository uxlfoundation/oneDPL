//===-- xpu_partial_sort.pass.cpp -----------------------------------------===//
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
                    unsigned sa = sizeof(a) / sizeof(a[0]);
                    dpl::partial_sort(Iter(a), Iter(a + sa), Iter(a + sa));
                    ret_acc[0] &= dpl::is_sorted(Iter(a), Iter(a + sa));
                }
                {
                    T a[] = {1, 0, 34, 2, 8, 7};
                    unsigned sa = sizeof(a) / sizeof(a[0]);
                    dpl::partial_sort(Iter(a), Iter(a + 3), Iter(a + sa));
                    ret_acc[0] &= dpl::is_sorted(Iter(a), Iter(a + 3));
                    ret_acc[0] &= (a[3] + a[4] + a[5] == 49);
                }
                {
                    T a[] = {1, 0, 0, 86, 2, 63};
                    unsigned sa = sizeof(a) / sizeof(a[0]);
                    dpl::partial_sort(Iter(a), Iter(a + 4), Iter(a + sa));
                    ret_acc[0] &= dpl::is_sorted(Iter(a), Iter(a + 4));
                    ret_acc[0] &= (a[4] + a[5] == 149);
                }
                {
                    T a[] = {0, 1, 0, 1, 12, 6, 21};
                    unsigned sa = sizeof(a) / sizeof(a[0]);
                    dpl::partial_sort(Iter(a), Iter(a + 1), Iter(a + sa));
                    ret_acc[0] &= dpl::is_sorted(Iter(a), Iter(a + 1));
                    ret_acc[0] &= (a[0] == 0);
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
