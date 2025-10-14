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

// <iterator>

// move_iterator

// move_iterator();
//
//  constexpr in C++17

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/test_iterators.h"
#include "support/utils.h"

template <class It>
void
test()
{
    dpl::move_iterator<It> r;
    (void)r;
}

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    {
        deviceQueue.submit([&](sycl::handler& cgh) {
            cgh.single_task<class KernelTest>([=]() {
                test<input_iterator<char*>>();
                test<forward_iterator<char*>>();
                test<bidirectional_iterator<char*>>();
                test<random_access_iterator<char*>>();
                test<char*>();

                {
                    constexpr dpl::move_iterator<const char*> it;
                    (void)it;
                }
            });
        });
    }
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
