// -*- C++ -*-
//===-- zip_iterator_copyable.pass.cpp ---------------------------------------------===//
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

#include <oneapi/dpl/iterator>
#include "support/test_config.h"
#include "support/utils.h"
#include "support/utils_device_copyable.h"

using namespace TestUtils;

#if TEST_DPCPP_BACKEND_PRESENT
void test_copyable()
{
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::zip_iterator<constant_iterator_device_copyable, int*>>,
                  "zip_iterator is not device copyable with device copyable types");

    static_assert(
        sycl::is_device_copyable_v<
            oneapi::dpl::zip_iterator<oneapi::dpl::counting_iterator<int>, constant_iterator_device_copyable, int*>>,
        "zip_iterator is not device copyable with device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::zip_iterator<constant_iterator_non_device_copyable, int*>>,
                  "zip_iterator is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::zip_iterator<int*, constant_iterator_non_device_copyable>>,
                  "zip_iterator is device copyable with non device copyable types");
}
#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_copyable();
#endif

    return done(TEST_DPCPP_BACKEND_PRESENT);
}

