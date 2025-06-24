// -*- C++ -*-
//===-- inner_product.pass.cpp
//--------------------------------------------===//
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

#include <oneapi/dpl/numeric>

#include <iostream>

#include "support/utils.h"
#include "support/test_iterators.h"

template <typename _T1, typename _T2>
void
ASSERT_EQUAL(_T1&& X, _T2&& Y)
{
    if (X != Y)
        std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

template <typename T1, typename T2>
class KernelName;

template <class Iter1, class Iter2>
void
test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    int input1[6] = {1, 2, 3, 4, 5, 6};
    int input2[6] = {6, 5, 4, 3, 2, 1};
    int output[8] = {};
    sycl::range<1> numOfItems1{6};
    sycl::range<1> numOfItems2{8};

    {
        sycl::buffer<int, 1> buffer1(input1, numOfItems1);
        sycl::buffer<int, 1> buffer2(input2, numOfItems1);
        sycl::buffer<int, 1> buffer3(output, numOfItems2);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto in1 = buffer1.get_access<sycl::access::mode::read>(cgh);
            auto in2 = buffer2.get_access<sycl::access::mode::read>(cgh);
            auto out = buffer3.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KernelName<Iter1, Iter2>>([=]() {
                out[0] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0]), Iter2(&in2[0]), 0);
                out[1] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0]), Iter2(&in2[0]), 10);
                out[2] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0] + 1), Iter2(&in2[0]), 0);
                out[3] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0] + 1), Iter2(&in2[0]), 10);
                out[4] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0] + 2), Iter2(&in2[0]), 0);
                out[5] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0] + 2), Iter2(&in2[0]), 10);
                out[6] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0] + 6), Iter2(&in2[0]), 0);
                out[7] = oneapi::dpl::inner_product(Iter1(&in1[0]), Iter1(&in1[0] + 6), Iter2(&in2[0]), 10);
            });
        });
    }
    const int ref[8] = {0, 10, 6, 16, 16, 26, 56, 66};
    for (int i = 0; i < 8; ++i)
    {
        ASSERT_EQUAL(ref[i], output[i]);
    }
}

int
main()
{
    test<input_iterator<const int*>, input_iterator<const int*>>();
    test<input_iterator<const int*>, forward_iterator<const int*>>();
    test<input_iterator<const int*>, bidirectional_iterator<const int*>>();
    test<input_iterator<const int*>, random_access_iterator<const int*>>();
    test<input_iterator<const int*>, const int*>();

#if !PSTL_USE_DEBUG
    // Turned out these test cases in debug mode since they consume too much time
    test<forward_iterator<const int*>, input_iterator<const int*>>();
    test<forward_iterator<const int*>, forward_iterator<const int*>>();
    test<forward_iterator<const int*>, bidirectional_iterator<const int*>>();
    test<forward_iterator<const int*>, random_access_iterator<const int*>>();
    test<forward_iterator<const int*>, const int*>();

    test<bidirectional_iterator<const int*>, input_iterator<const int*>>();
    test<bidirectional_iterator<const int*>, forward_iterator<const int*>>();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>>();
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*>>();
    test<bidirectional_iterator<const int*>, const int*>();

    test<random_access_iterator<const int*>, input_iterator<const int*>>();
    test<random_access_iterator<const int*>, forward_iterator<const int*>>();
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*>>();
    test<random_access_iterator<const int*>, random_access_iterator<const int*>>();
    test<random_access_iterator<const int*>, const int*>();

    test<const int*, input_iterator<const int*>>();
    test<const int*, forward_iterator<const int*>>();
    test<const int*, bidirectional_iterator<const int*>>();
    test<const int*, random_access_iterator<const int*>>();
    test<const int*, const int*>();
#endif

    return TestUtils::done();
}
