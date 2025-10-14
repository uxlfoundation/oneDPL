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

// <algorithm>
// template<ForwardIterator Iter, class T, CopyConstructible Compare>
//   constexpr pair<Iter, Iter>   // constexpr after c++17
//   equal_range(Iter first, Iter last, const T& value, Compare comp);

#include "support/test_config.h"

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/functional>

#include <iostream>

#include "support/utils.h"
#include "support/test_iterators.h"
#include "support/sycl_alloc_utils.h"

//Revert the change when GPU runtime fix their issue
template <class Iter, class T>
bool
test(Iter first, Iter last, const T& value)
{
    std::pair<Iter, Iter> i = dpl::equal_range(first, last, value, dpl::greater<int>());

    for (Iter j = first; j != i.first; ++j)
        if (!(dpl::greater<int>()(*j, value)))
            return false;
    for (Iter j = first; j != i.second; ++j)
        if (dpl::greater<int>()(value, *j))
            return false;
    return true;
}

class KernelEqualRangeTest1;
class KernelEqualRangeTest2;
class KernelEqualRangeTest3;
class KernelEqualRangeTest4;

template <typename Iter, typename KC>
void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    const unsigned N = 1000;
    const unsigned M = 10;
    int host_vbuf[N];
    for (size_t i = 0; i < N; ++i)
    {
        host_vbuf[i] = i % M;
    }

    std::sort(host_vbuf, host_vbuf + N, dpl::greater<int>());

    TestUtils::usm_data_transfer<sycl::usm::alloc::device, int> dt_helper(deviceQueue, host_vbuf, N);

    deviceQueue
        .submit([&](sycl::handler& cgh) {
            int* device_vbuf = dt_helper.get_data();
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KC>([=]() {
                ret_access[0] = test(device_vbuf, device_vbuf + N, 0);
                for (int x = 1; x <= M; ++x)
                    ret_access[0] &= test(device_vbuf, device_vbuf + N, x);
            });
        })
        .wait();

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of equal_range with comparator");
}

int
main()
{
    kernel_test<forward_iterator<const int*>, KernelEqualRangeTest1>();
    kernel_test<bidirectional_iterator<const int*>, KernelEqualRangeTest2>();
    kernel_test<random_access_iterator<const int*>, KernelEqualRangeTest3>();
    kernel_test<const int*, KernelEqualRangeTest4>();

    return TestUtils::done();
}
