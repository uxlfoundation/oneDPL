//===-- xpu_for_each.pass.cpp ---------------------------------------------===//
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

template <typename _T1, typename _T2>
void
ASSERT_EQUAL(_T1&& X, _T2&& Y)
{
    if (X != Y)
        std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

template <typename _T>
struct plus1
{
    void
    operator()(_T& x)
    {
        ++x;
    }
};

template <typename _T>
struct mul2
{
    void
    operator()(_T& x)
    {
        x *= 2;
    }
};

template <typename _T>
struct div2
{
    void
    operator()(_T& x)
    {
        x /= 2;
    }
};

template <class T>
class KernelTest;

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    using VT = typename std::iterator_traits<Iter>::value_type;
    int arr[] = {1, 2, 3, 4, 5, 6};
    int ref[] = {2, 3, 4, 5, 6, 7};
    sycl::range<1> numOfItems{6};
    {
        sycl::buffer<int, 1> buffer1(arr, sycl::range<1>{6});
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto arr_acc = buffer1.get_access<sycl::access::mode::read_write>(cgh);
            cgh.single_task<KernelTest<Iter>>([=]() {
                dpl::for_each_n(Iter(&arr_acc[0]), 1, plus1<VT>());
                dpl::for_each_n(Iter(&arr_acc[1]), 1, plus1<VT>());
                dpl::for_each_n(Iter(&arr_acc[2]), 1, plus1<VT>());
                dpl::for_each_n(Iter(&arr_acc[3]), 1, plus1<VT>());
                dpl::for_each_n(Iter(&arr_acc[4]), 1, plus1<VT>());
                dpl::for_each_n(Iter(&arr_acc[5]), 1, plus1<VT>());
                dpl::for_each_n(Iter(&arr_acc[0]), 6, mul2<VT>());
                dpl::for_each_n(Iter(&arr_acc[0]), 6, div2<VT>());
            });
        });
    }

    for (size_t idx = 0; idx < 6; ++idx)
    {
        ASSERT_EQUAL(ref[idx], arr[idx]);
    }
}

void
test(sycl::queue& deviceQueue)
{
    test<input_iterator<int*>>(deviceQueue);
    test<forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<int*>>(deviceQueue);
    test<int*>(deviceQueue);
}

int
main(int, char**)
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test(deviceQueue);
    return TestUtils::done();
}
