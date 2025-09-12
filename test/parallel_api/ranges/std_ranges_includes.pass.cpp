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

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING
struct A1
{
    int a1;
};

struct A2
{
    int a2;
};

void test_mixed_types_host()
{
    std::vector<A1> vec_a1 = {{1}, {2}, {3}};
    std::vector<A2> vec_a2 = {{2}, {3}};

    auto proj_a1 = [](const A1& a) { return a.a1; };
    auto proj_a2 = [](const A2& a) { return a.a2; };

    bool exp_res = std::ranges::includes(vec_a1, vec_a2, std::ranges::less{}, proj_a1, proj_a2);

    bool seq_res = oneapi::dpl::ranges::includes(oneapi::dpl::execution::seq, vec_a1, vec_a2, std::ranges::less{}, proj_a1, proj_a2);
    EXPECT_EQ(seq_res, exp_res, "wrong result with seq policy");

    bool unseq_res = oneapi::dpl::ranges::includes(oneapi::dpl::execution::unseq, vec_a1, vec_a2, std::ranges::less{}, proj_a1, proj_a2);
    EXPECT_EQ(unseq_res, exp_res, "wrong result with unseq policy");

    bool par_res = oneapi::dpl::ranges::includes(oneapi::dpl::execution::par, vec_a1, vec_a2, std::ranges::less{}, proj_a1, proj_a2);
    EXPECT_EQ(par_res, exp_res, "wrong result with par policy");

    bool par_unseq_res = oneapi::dpl::ranges::includes(oneapi::dpl::execution::par_unseq, vec_a1, vec_a2, std::ranges::less{}, proj_a1, proj_a2);
    EXPECT_EQ(par_unseq_res, exp_res, "wrong result with par_unseq policy");
}

#if TEST_DPCPP_BACKEND_PRESENT
void test_mixed_types_device()
{
    auto policy = TestUtils::get_dpcpp_test_policy();
    sycl::queue q = policy.queue();
    if (q.get_device().has(sycl::aspect::usm_shared_allocations))
    {
        A1* d_a1 = sycl::malloc_shared<A1>(3, q);
        A2* d_a2 = sycl::malloc_shared<A2>(2, q);

        d_a1[0] = {1};
        d_a1[1] = {2};
        d_a1[2] = {3};

        d_a2[0] = {2};
        d_a2[1] = {3};

        std::ranges::subrange a1_range(d_a1, d_a1 + 3);
        std::ranges::subrange a2_range(d_a2, d_a2 + 2);

        auto proj_a1 = [](const A1& a) { return a.a1; };
        auto proj_a2 = [](const A2& a) { return a.a2; };

        bool exp_res = std::ranges::includes(a1_range, a2_range, std::ranges::less{}, proj_a1, proj_a2);

        bool dev_res = oneapi::dpl::ranges::includes(oneapi::dpl::execution::make_device_policy(q),
                                                     a1_range, a2_range, std::ranges::less{}, proj_a1, proj_a2);
        EXPECT_EQ(dev_res, exp_res, "wrong result with device policy");

        sycl::free(d_a1, q);
        sycl::free(d_a2, q);
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto includes_checker = TEST_PREPARE_CALLABLE(std::ranges::includes);

    test_range_algo<0,  int, data_in_in>{big_sz}(dpl_ranges::includes, includes_checker);
    test_range_algo<1,  int, data_in_in>{      }(dpl_ranges::includes, includes_checker, std::ranges::less{});
    test_range_algo<2,  int, data_in_in>{      }(dpl_ranges::includes, includes_checker, std::ranges::less{}, proj);
    test_range_algo<3 , int, data_in_in>{      }(dpl_ranges::includes, includes_checker, std::ranges::less{}, proj, proj);

    // Check with different projections,
    // but when includes returns true - to make sure that the projections are applied correctly.
    // The first sequence is [0, 3, 6, ...], the second is [0, 1, 2, ...],
    // but the second is transformed to [0, 3, 6, ...] by its projection.
    auto x1 = [](auto&& v) { return v; };
    auto x3 = [](auto&& v) { return v * 3; };
    test_range_algo<4, int, data_in_in, decltype(x3), decltype(x1)>{medium_size}(dpl_ranges::includes, includes_checker, std::ranges::less{}, x1, x3);

    test_range_algo<5, P2, data_in_in>{}(dpl_ranges::includes, includes_checker, std::ranges::less{}, &P2::x, &P2::x);
    test_range_algo<6, P2, data_in_in>{}(dpl_ranges::includes, includes_checker, std::ranges::less{}, &P2::proj, &P2::proj);

    // Check if projections are applied to the right sequences and trigger a compile-time error if not
    test_mixed_types_host();
#if TEST_DPCPP_BACKEND_PRESENT
    test_mixed_types_device();
#endif
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
