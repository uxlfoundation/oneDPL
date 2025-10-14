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

// struct random_access_iterator_tag : public bidirectional_iterator_tag {};

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

void
kernelTest()
{
    sycl::queue q = TestUtils::get_test_queue();
    q.submit([&](sycl::handler& cgh) {
        cgh.single_task<class IteratorTest>([=]() {
            dpl::random_access_iterator_tag tag;
            ((void)tag); // Prevent unused warning
            static_assert(dpl::is_base_of<dpl::bidirectional_iterator_tag, dpl::random_access_iterator_tag>::value);
            static_assert(!dpl::is_base_of<dpl::output_iterator_tag, dpl::random_access_iterator_tag>::value);
        });
    });
}

int
main()
{
    kernelTest();

    return TestUtils::done();
}
