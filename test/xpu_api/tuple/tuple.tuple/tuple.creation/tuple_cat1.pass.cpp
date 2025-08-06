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

#include <oneapi/dpl/tuple>
#include <oneapi/dpl/utility>
#include <oneapi/dpl/array>

#include "support/utils.h"
#include "support/utils_invoke.h"

struct MoveOnlyData
{
    int idx_ = 0;

    MoveOnlyData(int idx) : idx_(idx) {}
    MoveOnlyData(MoveOnlyData&& x) = default;
    MoveOnlyData(const MoveOnlyData&) = delete;

    MoveOnlyData&
    operator=(MoveOnlyData&& x) = default;
    MoveOnlyData&
    operator=(const MoveOnlyData&) = delete;

    bool
    operator==(const MoveOnlyData& x) const
    {
        return idx_ == x.idx_;
    }
};

class KernelTupleCatTest1;
class KernelTupleCatTest2;

void
kernel_test1(sycl::queue& deviceQueue)
{
    bool ret = true;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelTupleCatTest1>([=]() {
            {
                dpl::tuple<> t;
                t = dpl::tuple_cat();
            }

            {
                dpl::tuple<> t1;
                [[maybe_unused]] dpl::tuple<> t2 = dpl::tuple_cat(t1);
            }

            {
                dpl::array<int, 0> empty_array;
                [[maybe_unused]] dpl::tuple<> t = dpl::tuple_cat(empty_array);
            }

            {
                dpl::tuple<int> t1(1);
                dpl::tuple<int> t = dpl::tuple_cat(t1);
                ret_access[0] &= (dpl::get<0>(t) == 1);
            }

            {
                [[maybe_unused]] constexpr dpl::tuple<> t = dpl::tuple_cat();
            }

            {
                constexpr dpl::tuple<> t1;
                [[maybe_unused]] constexpr dpl::tuple<> t2 = dpl::tuple_cat(t1);
            }

            {
                constexpr dpl::array<int, 0> empty_array = {};
                [[maybe_unused]] constexpr dpl::tuple<> t = dpl::tuple_cat(empty_array);
            }

            {
                constexpr dpl::tuple<int> t1(1);
                constexpr dpl::tuple<int> t = dpl::tuple_cat(t1);
                static_assert(dpl::get<0>(t) == 1);
            }

            {
                constexpr dpl::tuple<int> t1(1);
                constexpr dpl::tuple<int, int> t = dpl::tuple_cat(t1, t1);
                static_assert(dpl::get<0>(t) == 1);
                static_assert(dpl::get<1>(t) == 1);
            }

            {
                dpl::tuple<int, MoveOnlyData> t = dpl::tuple_cat(dpl::tuple<int, MoveOnlyData>(1, 2));
                ret_access[0] &= (dpl::get<0>(t) == 1 && dpl::get<1>(t) == 2);
            }

            {
                dpl::tuple<int, int, int> t = dpl::tuple_cat(dpl::array<int, 3>());
                ret_access[0] &= (dpl::get<0>(t) == 0 && dpl::get<1>(t) == 0 && dpl::get<2>(t) == 0);
            }

            {
                dpl::tuple<int, MoveOnlyData> t = dpl::tuple_cat(dpl::pair<int, MoveOnlyData>(2, 1));
                ret_access[0] &= (dpl::get<0>(t) == 2 && dpl::get<1>(t) == 1);
            }

            {
                dpl::tuple<> t1;
                dpl::tuple<> t2;
                [[maybe_unused]] dpl::tuple<> t3 = dpl::tuple_cat(t1, t2);
            }

            {
                dpl::tuple<> t1;
                dpl::tuple<int> t2(2);
                dpl::tuple<int> t3 = dpl::tuple_cat(t1, t2);
                ret_access[0] &= (dpl::get<0>(t3) == 2);
            }

            {
                dpl::tuple<int*> t1;
                dpl::tuple<int> t2(2);
                dpl::tuple<int*, int> t3 = dpl::tuple_cat(t1, t2);
                ret_access[0] &= (dpl::get<0>(t3) == nullptr && dpl::get<1>(t3) == 2);
            }

            {
                dpl::tuple<int*> t1;
                dpl::tuple<int> t2(2);
                dpl::tuple<int, int*> t3 = dpl::tuple_cat(t2, t1);
                ret_access[0] &= (dpl::get<0>(t3) == 2 && dpl::get<1>(t3) == nullptr);
            }

            {
                dpl::tuple<MoveOnlyData, MoveOnlyData> t1(1, 2);
                dpl::tuple<int*, MoveOnlyData> t2(nullptr, 4);
                dpl::tuple<MoveOnlyData, MoveOnlyData, int*, MoveOnlyData> t3 =
                    dpl::tuple_cat(dpl::move(t1), dpl::move(t2));
                ret_access[0] &= (dpl::get<0>(t3) == 1 && dpl::get<1>(t3) == 2 && dpl::get<2>(t3) == nullptr &&
                                  dpl::get<3>(t3) == 4);
            }

            {
                dpl::tuple<MoveOnlyData, MoveOnlyData> t1(1, 2);
                dpl::tuple<int*, MoveOnlyData> t2(nullptr, 4);
                dpl::tuple<MoveOnlyData, MoveOnlyData, int*, MoveOnlyData> t3 =
                    dpl::tuple_cat(dpl::tuple<>(), dpl::move(t1), dpl::move(t2));
                ret_access[0] &= (dpl::get<0>(t3) == 1 && dpl::get<1>(t3) == 2 && dpl::get<2>(t3) == nullptr &&
                                  dpl::get<3>(t3) == 4);
            }

            {
                dpl::tuple<MoveOnlyData, MoveOnlyData> t1(1, 2);
                dpl::tuple<int*, MoveOnlyData> t2(nullptr, 4);
                dpl::tuple<MoveOnlyData, MoveOnlyData, int*, MoveOnlyData> t3 =
                    dpl::tuple_cat(dpl::move(t1), dpl::tuple<>(), dpl::move(t2));
                ret_access[0] &= (dpl::get<0>(t3) == 1 && dpl::get<1>(t3) == 2 && dpl::get<2>(t3) == nullptr &&
                                  dpl::get<3>(t3) == 4);
            }

            {
                dpl::tuple<MoveOnlyData, MoveOnlyData> t1(1, 2);
                dpl::tuple<int*, MoveOnlyData> t2(nullptr, 4);
                dpl::tuple<MoveOnlyData, MoveOnlyData, int*, MoveOnlyData> t3 =
                    dpl::tuple_cat(dpl::move(t1), dpl::move(t2), dpl::tuple<>());
                ret_access[0] &= (dpl::get<0>(t3) == 1 && dpl::get<1>(t3) == 2 && dpl::get<2>(t3) == nullptr &&
                                  dpl::get<3>(t3) == 4);
            }

            {
                dpl::tuple<MoveOnlyData, MoveOnlyData> t1(1, 2);
                dpl::tuple<int*, MoveOnlyData> t2(nullptr, 4);
                dpl::tuple<MoveOnlyData, MoveOnlyData, int*, MoveOnlyData, int> t3 =
                    dpl::tuple_cat(dpl::move(t1), dpl::move(t2), dpl::tuple<int>(5));
                ret_access[0] &= (dpl::get<0>(t3) == 1 && dpl::get<1>(t3) == 2 && dpl::get<2>(t3) == nullptr &&
                                  dpl::get<3>(t3) == 4 && dpl::get<4>(t3) == 5);
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::tuple_cat check in kernel_test1");
}

void
kernel_test2(sycl::queue& deviceQueue)
{
    bool ret = true;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelTupleCatTest>([=]() {
            {
                dpl::tuple<int*> t1;
                dpl::tuple<int, double> t2(2, 3.5);
                dpl::tuple<int*, int, double> t3 = dpl::tuple_cat(t1, t2);
                ret_access[0] &= (dpl::get<0>(t3) == nullptr && dpl::get<1>(t3) == 2 && dpl::get<2>(t3) == 3.5);
            }

            {
                dpl::tuple<int*> t1;
                dpl::tuple<int, double> t2(2, 3.5);
                dpl::tuple<int, double, int*> t3 = dpl::tuple_cat(t2, t1);
                ret_access[0] &= (dpl::get<0>(t3) == 2 && dpl::get<1>(t3) == 3.5 && dpl::get<2>(t3) == nullptr);
            }

            {
                dpl::tuple<int*, MoveOnlyData> t1(nullptr, 1);
                dpl::tuple<int, double> t2(2, 3.5);
                dpl::tuple<int*, MoveOnlyData, int, double> t3 = dpl::tuple_cat(dpl::move(t1), t2);
                ret_access[0] &= (dpl::get<0>(t3) == nullptr && dpl::get<1>(t3) == 1 && dpl::get<2>(t3) == 2 &&
                                  dpl::get<3>(t3) == 3.5);
            }

            {
                dpl::tuple<int*, MoveOnlyData> t1(nullptr, 1);
                dpl::tuple<int, double> t2(2, 3.5);
                dpl::tuple<int, double, int*, MoveOnlyData> t3 = dpl::tuple_cat(t2, dpl::move(t1));
                ret_access[0] &= (dpl::get<0>(t3) == 2 && dpl::get<1>(t3) == 3.5 && dpl::get<2>(t3) == nullptr &&
                                  dpl::get<3>(t3) == 1);
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::tuple_cat check in kernel_test2");
}

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test1(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        kernel_test2(deviceQueue);
    }

    return TestUtils::done();
}
