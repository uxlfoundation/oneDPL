// -*- C++ -*-
//===-- accumulate.pass.cpp --------------------------------------------===//
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

template <class T> class KernelTest;

template <typename _T1, typename _T2> void ASSERT_EQUAL(_T1 &&X, _T2 &&Y) {
  if (X != Y)
    std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y
              << ")" << std::endl;
}

template <class Iter> void test() {
  sycl::queue deviceQueue = TestUtils::get_test_queue();
  int input[6] = {1, 2, 3, 4, 5, 6};
  int output[7] = {};
  sycl::range<1> numOfItems1{6};
  sycl::range<1> numOfItems2{7};

  {
    sycl::buffer<int, 1> buffer1(input, numOfItems1);
    sycl::buffer<int, 1> buffer2(output, numOfItems2);
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto in = buffer1.get_access<sycl::access::mode::read>(cgh);
      auto out = buffer2.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task<KernelTest<Iter>>([=]() {
        out[0] = oneapi::dpl::accumulate(Iter(&in[0]), Iter(&in[0]), 0);
        out[1] = oneapi::dpl::accumulate(Iter(&in[0]), Iter(&in[0]), 10);
        out[2] = oneapi::dpl::accumulate(Iter(&in[0]), Iter(&in[0] + 1), 0);
        out[3] = oneapi::dpl::accumulate(Iter(&in[0]), Iter(&in[0] + 1), 10);
        out[4] = oneapi::dpl::accumulate(Iter(&in[0]), Iter(&in[0] + 2), 0);
        out[5] = oneapi::dpl::accumulate(Iter(&in[0]), Iter(&in[0] + 2), 10);
        out[6] = oneapi::dpl::accumulate(Iter(&in[0]), Iter(&in[0] + 6), 0);
      });
    });
  }
  int ref[7] = {0, 10, 1, 11, 3, 13, 21};
  // check data
  for (int i = 0; i < 7; ++i) {
    ASSERT_EQUAL(ref[i], output[i]);
  }
}

int main() {
  test<input_iterator<const int *>>();
  test<forward_iterator<const int *>>();
  test<bidirectional_iterator<const int *>>();
  test<random_access_iterator<const int *>>();
  test<const int *>();
  return TestUtils::done();
}
