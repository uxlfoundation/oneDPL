//===-- xpu_pop_heap_cpmp.pass.cpp ----------------------------------------===//
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
                T orig[15] = {1, 1, 2, 3, 3, 8, 4, 6, 5, 5, 5, 9, 9, 7, 9};
                T work[15] = {1, 1, 2, 3, 3, 8, 4, 6, 5, 5, 5, 9, 9, 7, 9};

                ret_acc[0] &= dpl::is_heap(orig, orig + 15, dpl::greater<T>());
                for (int i = 15; i >= 1; --i)
                {
                    dpl::pop_heap(Iter(work), Iter(work + i), dpl::greater<T>());
                    ret_acc[0] &= dpl::is_heap(work, work + i - 1, dpl::greater<T>());
                }

                {
                    T input[] = {1, 2, 5, 4, 3};
                    ret_acc[0] &= dpl::is_heap(input, input + 5, dpl::greater<T>());
                    dpl::pop_heap(Iter(input), Iter(input + 5), dpl::greater<T>());
                    ret_acc[0] &= input[4] == 1;
                    dpl::pop_heap(Iter(input), Iter(input + 4), dpl::greater<T>());
                    ret_acc[0] &= input[3] == 2;
                    dpl::pop_heap(Iter(input), Iter(input + 3), dpl::greater<T>());
                    ret_acc[0] &= input[2] == 3;
                    dpl::pop_heap(Iter(input), Iter(input + 2), dpl::greater<T>());
                    ret_acc[0] &= input[1] == 4;
                    dpl::pop_heap(Iter(input), Iter(input + 1), dpl::greater<T>());
                    ret_acc[0] &= input[0] == 5;
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
