//===-- xpu_unary_transform.pass.cpp --------------------------------------===//
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
#include <oneapi/dpl/functional>

#include "support/utils.h"
#include "support/test_iterators.h"

#include <cassert>

struct plusOne
{
    int
    operator()(int v)
    {
        return v + 1;
    }
};

template <class InIter, class OutIter>
class KernelTest;

template <class InIter, class OutIter>
void
test(sycl::queue& deviceQueue)
{
    int ia[] = {0, 1, 2, 3, 4};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    int ib[sa] = {1, 2, 3, 4, 5};
    bool ret = true;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{sa};
    {
        sycl::buffer<int, 1> buffer1(ia, itemN);
        sycl::buffer<int, 1> buffer2(ib, itemN);
        sycl::buffer<bool, 1> buffer3(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto acc_arr1 = buffer1.get_access<sycl::access::mode::read>(cgh);
            auto acc_arr2 = buffer2.get_access<sycl::access::mode::write>(cgh);
            auto ret_acc = buffer3.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KernelTest<InIter, OutIter>>([=]() {
                OutIter r =
                    std::transform(InIter(&acc_arr1[0]), InIter(&acc_arr1[0] + sa), OutIter(&acc_arr2[0]), plusOne());
                ret_acc[0] = (base(r) == &acc_arr2[0] + sa);
            });
        }).wait();
    }
    assert(ret);
    for (size_t idx = 0; idx < sa; ++idx)
    {
        assert(ib[idx] == 1 + idx);
    }
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test<input_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<input_iterator<const int*>, int*>(deviceQueue);

    test<forward_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<forward_iterator<const int*>, int*>(deviceQueue);

    test<bidirectional_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>, int*>(deviceQueue);

    test<random_access_iterator<const int*>, output_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, input_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, forward_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, bidirectional_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, random_access_iterator<int*>>(deviceQueue);
    test<random_access_iterator<const int*>, int*>(deviceQueue);

    test<const int*, output_iterator<int*>>(deviceQueue);
    test<const int*, input_iterator<int*>>(deviceQueue);
    test<const int*, forward_iterator<int*>>(deviceQueue);
    test<const int*, bidirectional_iterator<int*>>(deviceQueue);
    test<const int*, random_access_iterator<int*>>(deviceQueue);
    test<const int*, int*>(deviceQueue);

    return TestUtils::done();
}
