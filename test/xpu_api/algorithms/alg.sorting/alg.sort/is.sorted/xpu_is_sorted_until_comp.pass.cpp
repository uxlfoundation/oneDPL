//===-- xpu_is_sorted_comp_until.pass.cpp ---------------------------------===//
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

// In Windows, as a temporary workaround, disable vector algorithm calls to avoid calls within sycl kernels
#if defined(_MSC_VER)
#    define _USE_STD_VECTOR_ALGORITHMS 0
#endif

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/functional>

#include "support/utils.h"
#include "support/test_iterators.h"

#include <cassert>

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    using T = typename std::iterator_traits<Iter>::value_type;
    bool ret = true;
    sycl::range<1> item1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<Iter>([=]() {
                {
                    T a[] = {1, 0};
                    unsigned sa = sizeof(a) / sizeof(a[0]);
                    ret_acc[0] &= (std::is_sorted_until(Iter(a), Iter(a + sa), dpl::greater<T>()) == Iter(a + sa));
                }
                {
                    T a[] = {1, 0, 0};
                    unsigned sa = sizeof(a) / sizeof(a[0]);
                    ret_acc[0] &= (std::is_sorted_until(Iter(a), Iter(a + sa), dpl::greater<T>()) == Iter(a + sa));
                }
                {
                    T a[] = {1, 0, 1};
                    unsigned sa = sizeof(a) / sizeof(a[0]);
                    ret_acc[0] &= (std::is_sorted_until(Iter(a), Iter(a + sa), dpl::greater<T>()) == Iter(a + 2));
                }
                {
                    T a[] = {0, 0, 1, 1};
                    unsigned sa = sizeof(a) / sizeof(a[0]);
                    ret_acc[0] &= (std::is_sorted_until(Iter(a), Iter(a + sa), dpl::greater<T>()) == Iter(a + 2));
                }
                {
                    T a[] = {0, 1, 0, 0};
                    unsigned sa = sizeof(a) / sizeof(a[0]);
                    ret_acc[0] &= (std::is_sorted_until(Iter(a), Iter(a + sa), dpl::greater<T>()) == Iter(a + 1));
                }
                {
                    T a[] = {0, 1, 0, 1};
                    unsigned sa = sizeof(a) / sizeof(a[0]);
                    ret_acc[0] &= (std::is_sorted_until(Iter(a), Iter(a + sa), dpl::greater<T>()) == Iter(a + 1));
                }
            });
        });
    }
    assert(ret);
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test<forward_iterator<const int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>>(deviceQueue);
    test<random_access_iterator<const int*>>(deviceQueue);
    test<const int*>(deviceQueue);
    test<forward_iterator<const float*>>(deviceQueue);
    test<bidirectional_iterator<const float*>>(deviceQueue);
    test<random_access_iterator<const float*>>(deviceQueue);
    test<const float*>(deviceQueue);
    if (deviceQueue.get_device().has(sycl::aspect::fp64))
    {
        test<forward_iterator<const double*>>(deviceQueue);
        test<bidirectional_iterator<const double*>>(deviceQueue);
        test<random_access_iterator<const double*>>(deviceQueue);
        test<const float*>(deviceQueue);
    }
    return TestUtils::done();
}
