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

#include <oneapi/dpl/array>
#include <oneapi/dpl/utility>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"
#include "support/move_only.h"
#include "support/test_macros.h"

#include "misc_data_structs.h"

class KernelSwapTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);

    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelSwapTest>([=]() {
            {
                int i[3] = {1, 2, 3};
                int j[3] = {4, 5, 6};
                dpl::swap(i, j);
                ret_access[0] = (i[0] == 4);
                ret_access[0] &= (i[1] == 5);
                ret_access[0] &= (i[2] == 6);
                ret_access[0] &= (j[0] == 1);
                ret_access[0] &= (j[1] == 2);
                ret_access[0] &= (j[2] == 3);
            }

            {
                int a = 1;
                int b = 2;
                int* i = &a;
                int* j = &b;
                dpl::swap(i, j);
                ret_access[0] &= (*i == 2);
                ret_access[0] &= (*j == 1);
            }

            {
                // test that the swap
                using CA = CopyOnly[42];
                using MA = NoexceptMoveOnly[42];
                using NA = NotMoveConstructible[42];
                static_assert(std::is_swappable_v<CA&>);
                static_assert(std::is_swappable_v<MA&>);
                static_assert(!std::is_swappable_v<NA&>);

                CA ca;
                MA ma;
                ASSERT_NOT_NOEXCEPT(dpl::swap(ca, ca));
                ASSERT_NOEXCEPT(dpl::swap(ma, ma));
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::swap check");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
