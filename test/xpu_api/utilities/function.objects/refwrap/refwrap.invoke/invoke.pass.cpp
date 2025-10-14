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

#include <oneapi/dpl/functional>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

struct A_void_1
{
    int* count_A_1;
    A_void_1(int* count) { count_A_1 = count; }
    void
    operator()(int i)
    {
        *count_A_1 += i;
    }

    void
    mem1()
    {
        ++(*count_A_1);
    }
};

struct A_int_1
{
    A_int_1() : data_(5) {}
    int
    operator()(int i)
    {
        return i - 1;
    }

    int
    mem1()
    {
        return 3;
    }
    int
    mem2() const
    {
        return 4;
    }
    int data_;
};

struct A_void_2
{
    A_void_2(int* c) { count = c; }
    void
    operator()(int i, int j)
    {
        *count += i + j;
    }

    int* count;
};

struct A_int_2
{
    int
    operator()(int i, int j)
    {
        return i + j;
    }

    int
    mem1(int i)
    {
        return i + 1;
    }
    int
    mem2(int i) const
    {
        return i + 2;
    }
};

class KernelInvokePassTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelInvokePassTest>([=]() {
            int count = 0;
            int save_count = count;

            {
                A_void_1 a0(&count);
                dpl::reference_wrapper<A_void_1> r1(a0);
                int i = 4;
                r1(i);
                ret_access[0] = (count == save_count + 4);
                save_count = count;
                a0.mem1();
                ret_access[0] &= (count == save_count + 1);
                save_count = count;
            }

            {
                A_int_1 a0;
                dpl::reference_wrapper<A_int_1> r1(a0);
                int i = 4;
                ret_access[0] &= (r1(i) == 3);
            }

            // member data pointer
            {
                int A_int_1::*fp = &A_int_1::data_;
                dpl::reference_wrapper<int A_int_1::*> r1(fp);
                A_int_1 a;
                ret_access[0] &= (r1(a) == 5);
                r1(a) = 6;
                ret_access[0] &= (r1(a) == 6);
                A_int_1* ap = &a;
                ret_access[0] &= (r1(ap) == 6);
                r1(ap) = 7;
                ret_access[0] &= (r1(ap) == 7);
            }

            {
                A_void_2 a0(&count);
                dpl::reference_wrapper<A_void_2> r1(a0);
                int i = 4;
                int j = 5;
                r1(i, j);
                ret_access[0] &= (count == save_count + 9);
                save_count = count;
            }

            {
                A_int_2 a0;
                dpl::reference_wrapper<A_int_2> r1(a0);
                int i = 4;
                int j = 5;
                ret_access[0] &= (r1(i, j) == i + j);
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Error in work with invoke");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
