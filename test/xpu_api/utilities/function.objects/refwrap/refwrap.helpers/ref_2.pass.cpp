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

#include "support/utils.h"

#include "support/counting_predicates.h"
class KernelRef2PassTest;

// bool is5 ( int i ) { return i == 5; }
class is5
{
  public:
    bool
    operator()(int i) const
    {
        return i == 5;
    }
};

template <typename T>
bool
call_pred(T pred)
{
    return pred(5);
}

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelRef2PassTest>([=]() {
            {
                int i = 0;
                dpl::reference_wrapper<int> r1 = std::ref(i);
                dpl::reference_wrapper<int> r2 = std::ref(r1);
                ret_access[0] = (&r2.get() == &i);
            }

            {
                is5 is5_functor;
                unary_counting_predicate<is5, int> cp(is5_functor);
                ret_access[0] &= (!cp(6));
                ret_access[0] &= (cp.count() == 1);
                ret_access[0] &= (call_pred(cp));
                ret_access[0] &= (cp.count() == 1);
                ret_access[0] &= (call_pred(std::ref(cp)));
                ret_access[0] &= (cp.count() == 2);
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Error in work with dpl::reference_wrapper");
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}
