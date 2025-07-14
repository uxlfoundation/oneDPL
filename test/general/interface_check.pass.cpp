// -*- C++ -*-
//===-- interface_check.pass.cpp --------------------------------------------===//
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

#include "support/test_config.h"

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/functional>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>
#include "support/utils.h"

#include <algorithm>
#include <iterator>
#include <vector>

using oneapi::dpl::counting_iterator;
using oneapi::dpl::discard_iterator;
using oneapi::dpl::make_zip_iterator;

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::buffer<int, 1> buf{sycl::range<1>(10)};

    auto b = oneapi::dpl::begin(buf);
    auto e = oneapi::dpl::end(buf);
#endif
    auto z = make_zip_iterator(counting_iterator<int>(), discard_iterator());
    std::get<1>(z[0]) = oneapi::dpl::identity()(*counting_iterator<int>());

    return TestUtils::done();
}
