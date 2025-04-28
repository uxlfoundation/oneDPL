//===-- xpu_push_heap_comp.pass.cpp ---------------------------------------===//
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
                    T arr[15] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
                    for (int i = 1; i < 15; ++i)
                    {
                        dpl::push_heap(Iter(arr), Iter(arr + i), dpl::greater<T>());
                        ret_acc[0] &= dpl::is_heap(arr, arr + i, dpl::greater<T>());
                    }
                }

                {
                    T input[5] = {5, 3, 4, 1, 2};
                    dpl::push_heap(Iter(input), Iter(input + 1), dpl::greater<T>());
                    ret_acc[0] &= (input[0] == 5);
                    dpl::push_heap(Iter(input), Iter(input + 2), dpl::greater<T>());
                    ret_acc[0] &= (input[0] == 3);
                    dpl::push_heap(Iter(input), Iter(input + 3), dpl::greater<T>());
                    ret_acc[0] &= (input[0] == 3);
                    dpl::push_heap(Iter(input), Iter(input + 4), dpl::greater<T>());
                    ret_acc[0] &= (input[0] == 1);
                    dpl::push_heap(Iter(input), Iter(input + 5), dpl::greater<T>());
                    ret_acc[0] &= (input[0] == 1);
                    ret_acc[0] &= dpl::is_heap(input, input + 5, dpl::greater<T>());
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
