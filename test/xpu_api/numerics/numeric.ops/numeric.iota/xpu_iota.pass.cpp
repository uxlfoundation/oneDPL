// -*- C++ -*-
//===-- iota.pass.cpp
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

template <class T> class KernelTest;

template <typename _T1, typename _T2> void ASSERT_EQUAL(_T1 &&X, _T2 &&Y) {
  if (X != Y)
    std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y
              << ")" << std::endl;
}

template <class InIter> void test() {
  sycl::queue deviceQueue = TestUtils::get_test_queue();
  int output[5] = {1, 2, 3, 4, 5};
  sycl::range<1> numOfItems1{5};

  {
    sycl::buffer<int, 1> buffer1(output, numOfItems1);
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto out = buffer1.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task<class KernelTest<InIter>>([=]() {
        oneapi::dpl::iota(InIter(&out[0]), InIter(&out[0] + 5), 5);
      });
    });
  }
  const int ref[5] = {5, 6, 7, 8, 9};
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQUAL(ref[i], output[i]);
  }
}

int main() {
  test<forward_iterator<int *>>();
  test<bidirectional_iterator<int *>>();
  test<random_access_iterator<int *>>();
  test<int *>();
  return TestUtils::done();
}
