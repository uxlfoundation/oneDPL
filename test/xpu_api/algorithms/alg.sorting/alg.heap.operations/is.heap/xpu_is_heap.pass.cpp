//===-- xpu_is_heap.pass.cpp ----------------------------------------------===//
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

template <class Iter1>
void
test(sycl::queue& deviceQueue)
{
    bool ret = true;
    sycl::range<1> item1{1};
    using T = typename std::iterator_traits<Iter1>::value_type;
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<Iter1>([=]() {
                T i1[] = {0, 1};
                T i2[] = {1, 0};
                T i3[] = {0, 1, 0, 1, 0};
                T i4[] = {0, 0, 0};
                T i5[] = {0, 0, 1};
                T i6[] = {0, 1, 0};
                T i7[] = {0, 1, 1};
                T i8[] = {1, 0, 0};
                T i9[] = {1, 0, 1};
                T i10[] = {1, 1, 0};
                ret_acc[0] &= (dpl::is_heap(Iter1(i1), Iter1(i1 + 2)) ==
                               (dpl::is_heap_until(Iter1(i1), Iter1(i1 + 2)) == Iter1(i1 + 2)));
                ret_acc[0] &= (dpl::is_heap(Iter1(i2), Iter1(i2 + 2)) ==
                               (dpl::is_heap_until(Iter1(i2), Iter1(i2 + 2)) == Iter1(i2 + 2)));
                ret_acc[0] &= (dpl::is_heap(Iter1(i3), Iter1(i3 + 5)) ==
                               (dpl::is_heap_until(Iter1(i3), Iter1(i3 + 5)) == Iter1(i3 + 5)));
                ret_acc[0] &= (dpl::is_heap(Iter1(i4), Iter1(i4 + 3)) ==
                               (dpl::is_heap_until(Iter1(i4), Iter1(i4 + 3)) == Iter1(i4 + 3)));
                ret_acc[0] &= (dpl::is_heap(Iter1(i5), Iter1(i5 + 3)) ==
                               (dpl::is_heap_until(Iter1(i5), Iter1(i5 + 3)) == Iter1(i5 + 3)));
                ret_acc[0] &= (dpl::is_heap(Iter1(i6), Iter1(i6 + 3)) ==
                               (dpl::is_heap_until(Iter1(i6), Iter1(i6 + 3)) == Iter1(i6 + 3)));
                ret_acc[0] &= (dpl::is_heap(Iter1(i7), Iter1(i7 + 3)) ==
                               (dpl::is_heap_until(Iter1(i7), Iter1(i7 + 3)) == Iter1(i7 + 3)));
                ret_acc[0] &= (dpl::is_heap(Iter1(i8), Iter1(i8 + 3)) ==
                               (dpl::is_heap_until(Iter1(i8), Iter1(i8 + 3)) == Iter1(i8 + 3)));
                ret_acc[0] &= (dpl::is_heap(Iter1(i9), Iter1(i9 + 3)) ==
                               (dpl::is_heap_until(Iter1(i9), Iter1(i9 + 3)) == Iter1(i9 + 3)));
                ret_acc[0] &= (dpl::is_heap(Iter1(i10), Iter1(i10 + 3)) ==
                               (dpl::is_heap_until(Iter1(i10), Iter1(i10 + 3)) == Iter1(i10 + 3)));
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
    test<random_access_iterator<float*>>(deviceQueue);
    test<float*>(deviceQueue);
    if (deviceQueue.get_device().has(sycl::aspect::fp64))
    {
        test<random_access_iterator<double*>>(deviceQueue);
        test<double*>(deviceQueue);
    }
    return TestUtils::done();
}
