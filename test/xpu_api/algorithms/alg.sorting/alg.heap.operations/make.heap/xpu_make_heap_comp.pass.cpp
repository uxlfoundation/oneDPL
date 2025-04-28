//===-- xpu_make_heap_comp.pass.cpp ---------------------------------------===//
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
    const int N = 100;
    bool ret = true;
    sycl::range<1> item1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<Iter>([=]() {
                T input[N];
                for (int i = 0; i < N; ++i)
                    input[i] = i;
                dpl::make_heap(Iter(input), Iter(input + N), dpl::greater<T>());
                ret_acc[0] &= dpl::is_heap(input, input + N, dpl::greater<T>());
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
