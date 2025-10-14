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

class MoveOnly
{
    int data_;

  public:
    MoveOnly(int data = 1) : data_(data) {}
    MoveOnly(MoveOnly&& x) : data_(x.data_) { x.data_ = 0; }
    MoveOnly(const MoveOnly&) = delete;

    MoveOnly&
    operator=(const MoveOnly&) = delete;

    MoveOnly&
    operator=(MoveOnly&& x)
    {
        data_ = x.data_;
        x.data_ = 0;
        return *this;
    }

    int
    get() const
    {
        return data_;
    }
};

template <class T>
class KernelTypePropertiesPassTest;

template <class T>
void
kernel_test(sycl::queue& deviceQueue)
{
    typedef dpl::reference_wrapper<T> Wrap;
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelTypePropertiesPassTest<T>>([=]()
        {
            // Static assert check...
            static_assert(dpl::is_copy_constructible<Wrap>::value);
            static_assert(dpl::is_copy_assignable<Wrap>::value);
            // Runtime check...
            ret_access[0] = dpl::is_copy_constructible<Wrap>::value;
            ret_access[0] &= dpl::is_copy_assignable<Wrap>::value;
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Error in work with type properties");
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test<int>(deviceQueue);
    kernel_test<MoveOnly>(deviceQueue);

    const auto device = deviceQueue.get_device();
    if (TestUtils::has_type_support<double>(device))
    {
        kernel_test<double>(deviceQueue);
    }

    return TestUtils::done();
}
