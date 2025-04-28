//===-- xpu_is_permutation.pass.cpp ---------------------------------------===//
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

#include <oneapi/dpl/algorithm>

#include "support/utils.h"
#include "support/test_iterators.h"

#include <cassert>

struct S
{
    S(int i) : i_(i) {}
    bool
    operator==(const S& other) = delete;
    int i_;
};

struct eq
{
    bool
    operator()(const S& a, const S& b)
    {
        return a.i_ == b.i_;
    }
};

template <class Iter1>
void
kernel_test1(sycl::queue& deviceQueue)
{
    bool ret = true;
    sycl::range<1> item1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<Iter1>([=]() {
                {
                    const int ia[] = {0, 1, 2};
                    const int ib[] = {2, 0, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &= dpl::is_permutation(Iter1(ia), Iter1(ia + sa), Iter1(ib), eq());
                }
                {
                    const int ia[] = {0, 0, 1};
                    const int ib[] = {1, 0, 1};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &= !dpl::is_permutation(Iter1(ia), Iter1(ia + sa), Iter1(ib), eq());
                }
                {
                    const int ia[] = {0, 0, 1};
                    const int ib[] = {1, 0, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &= dpl::is_permutation(Iter1(ia), Iter1(ia + sa), Iter1(ib), eq());
                }
                {
                    const int ia[] = {0, 1, 2, 3, 0, 5, 6, 2, 4, 4};
                    const int ib[] = {4, 2, 3, 0, 1, 4, 0, 5, 6, 2};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &= dpl::is_permutation(Iter1(ia), Iter1(ia + sa), Iter1(ib), eq());
                }
                {
                    const int ia[] = {0, 1, 2, 3, 0, 5, 6, 2, 4, 4};
                    const int ib[] = {4, 2, 3, 0, 1, 4, 0, 5, 6, 0};
                    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                    ret_acc[0] &= !dpl::is_permutation(Iter1(ia), Iter1(ia + sa), Iter1(ib), eq());
                }
            });
        }).wait();
    }
    assert(ret);
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test1<forward_iterator<const int*>>(deviceQueue);
    kernel_test1<bidirectional_iterator<const int*>>(deviceQueue);
    kernel_test1<random_access_iterator<const int*>>(deviceQueue);
    kernel_test1<const int*>(deviceQueue);
    return TestUtils::done();
}
