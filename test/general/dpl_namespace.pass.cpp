// -*- C++ -*-
//===-- dpl_namespace.pass.cpp --------------------------------------------===//
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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/functional>

#include "support/utils.h"

#include <iostream>
#include <tuple>

class ForEach;
class Transform;
class Scan;

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    const int n = 1000;
    const int k = 1000;
    using T = std::uint64_t;

    sycl::buffer<T> key_buf{sycl::range<1>(n)};
    sycl::buffer<T> val_buf{sycl::range<1>(n)};
    sycl::buffer<T> res_buf{sycl::range<1>(k)};

    auto key_first = dpl::begin(key_buf);
    auto val_first = dpl::begin(val_buf);
    auto res_first = dpl::begin(res_buf);
    auto counting_first = dpl::counting_iterator<T>(0);
    auto zip_first = dpl::make_zip_iterator(counting_first, key_first);

    // key_buf = {0,0,...0,1,1,...,1}
    std::for_each(TestUtils::make_device_policy<ForEach>(dpl::execution::dpcpp_default),
		zip_first, zip_first + n,
        [](std::tuple<T, T> x){
            std::get<1>(x) = (2 * std::get<0>(x)) / n;
        });
    // val_buf = {0,1,2,...,n-1}
    std::transform(TestUtils::make_device_policy<Transform>(dpl::execution::dpcpp_default),
		counting_first, counting_first + n, val_first, dpl::identity());
    auto result = dpl::inclusive_scan_by_segment(
		TestUtils::make_device_policy<Scan>(dpl::execution::dpcpp_default),
		key_first, key_first + n, val_first, res_first);

    if (result - res_first != k){
        std::cout << "size of keys output is not valid: expected = "<< k <<", got: "<< result - res_first << std::endl;
    }
    else{
        std::cout << "done" << std::endl;
    }

#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
