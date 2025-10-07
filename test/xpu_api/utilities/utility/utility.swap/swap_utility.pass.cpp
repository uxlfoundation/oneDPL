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

#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>
#include <oneapi/dpl/complex>

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
    sycl::range<1> numOfItems_acc{2};
    int acc[2] = {1, 2};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        sycl::buffer<int, 1> acc_buffer(acc, numOfItems_acc);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto acc_dev = acc_buffer.get_access<sycl::access::mode::read_write>(cgh);
            cgh.single_task<class KernelSwapTest>([=]() {
                {
                    int i = 1;
                    int j = 2;
                    dpl::swap(i, j);
                    ret_access[0] = (i == 2);
                    ret_access[0] &= (j == 1);
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
                    dpl::complex<float> c1(1.5f, 2.5f);
                    dpl::complex<float> c2(1.f, 5.5f);
                    ret_access[0] &= (c1.real() == 1.5f && c1.imag() == 2.5f);
                    ret_access[0] &= (c2.real() == 1.f && c2.imag() == 5.5f);
                    dpl::swap(c1, c2);
                    ret_access[0] &= (c2.real() == 1.5f && c2.imag() == 2.5f);
                    ret_access[0] &= (c1.real() == 1.f && c1.imag() == 5.5f);
                }

                {
                    static_assert(std::is_swappable_v<CopyOnly&>);
                    static_assert(std::is_swappable_v<MoveOnly&>);
                    static_assert(std::is_swappable_v<NoexceptMoveOnly&>);

                    static_assert(!std::is_swappable_v<NotMoveConstructible&>);
                    static_assert(!std::is_swappable_v<NotMoveAssignable&>);

                    CopyOnly c;
                    MoveOnly m;
                    NoexceptMoveOnly nm;
                    ASSERT_NOT_NOEXCEPT(dpl::swap(c, c));
                    ASSERT_NOT_NOEXCEPT(dpl::swap(m, m));
                    ASSERT_NOEXCEPT(dpl::swap(nm, nm));
                }

                {
                    ret_access[0] &= (acc_dev[0] == 1);
                    ret_access[0] &= (acc_dev[1] == 2);
                    dpl::swap(acc_dev[0], acc_dev[1]);
                }
            });
        });
    }

    EXPECT_TRUE(ret && acc[0] == 2 && acc[1] == 1, "Wrong result of dpl::swap check");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
