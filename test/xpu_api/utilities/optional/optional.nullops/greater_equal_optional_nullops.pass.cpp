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

#include "support/test_config.h"

#include <oneapi/dpl/optional>

#include "support/test_macros.h"
#include "support/utils.h"

#if !_PSTL_TEST_COMPARISON_BROKEN
bool
kernel_test()
{
    sycl::queue q = TestUtils::get_test_queue();
    bool ret = true;
    typedef dpl::optional<int> O;
    O ia[2] = {O{}, O{1}};
    sycl::range<1> numOfItems1{1};
    sycl::range<1> numOfItems2{2};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems1);
        sycl::buffer<O, 1> buffer2(ia, numOfItems2);

        q.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto ia_acc = buffer2.get_access<sycl::access::mode::read>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                using dpl::nullopt;

                ret_access[0] &= ((nullopt >= ia_acc[0]));
                ret_access[0] &= (!(nullopt >= ia_acc[1]));
                ret_access[0] &= ((ia_acc[0] >= nullopt));
                ret_access[0] &= ((ia_acc[1] >= nullopt));
            });
        });
    }
    return ret;
}
#endif // !_PSTL_TEST_COMPARISON_BROKEN

int
main()
{
    bool bProcessed = false;

#if !_PSTL_TEST_COMPARISON_BROKEN
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of greater or equal value check");
    bProcessed = true;
#endif // !_PSTL_TEST_COMPARISON_BROKEN

    return TestUtils::done(bProcessed);
}
