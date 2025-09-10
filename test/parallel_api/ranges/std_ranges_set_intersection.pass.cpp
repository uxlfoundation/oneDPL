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
struct A
{
    int a;
    operator int() const { return a; }
};

struct B
{
    int b;
    operator int() const { return b; }
};

void test_mixed_types_host()
{
    std::vector<A> r1 = {{1}, {2}, {3}};
    std::vector<B> r2 = {{0}, {2}, {2}, {3}};

    std::vector<int> out_expected = {2, 3};

    std::vector<int> out_seq(out_expected.size(), 0xCD);
    std::vector<int> out_par(out_expected.size(), 0xCD);
    std::vector<int> out_unseq(out_expected.size(), 0xCD);
    std::vector<int> out_par_unseq(out_expected.size(), 0xCD);

    auto proj_a = [](const A& a) { return a.a; };
    auto proj_b = [](const B& b) { return b.b; };

    oneapi::dpl::ranges::set_intersection(
        oneapi::dpl::execution::seq, r1, r2, out_seq, std::ranges::less{}, proj_a, proj_b);
    oneapi::dpl::ranges::set_intersection(
        oneapi::dpl::execution::par, r1, r2, out_par, std::ranges::less{}, proj_a, proj_b);
    oneapi::dpl::ranges::set_intersection(
        oneapi::dpl::execution::unseq, r1, r2, out_unseq, std::ranges::less{}, proj_a, proj_b);
    oneapi::dpl::ranges::set_intersection(
        oneapi::dpl::execution::par_unseq, r1, r2, out_par_unseq, std::ranges::less{}, proj_a, proj_b);

    EXPECT_EQ_RANGES(out_expected, out_seq, "wrong result with seq policy");
    EXPECT_EQ_RANGES(out_expected, out_par, "wrong result with par policy");
    EXPECT_EQ_RANGES(out_expected, out_unseq, "wrong result with unseq policy");
    EXPECT_EQ_RANGES(out_expected, out_par_unseq, "wrong result with par_unseq policy");
}

#if TEST_DPCPP_BACKEND_PRESENT
void test_mixed_types_device()
{
    auto policy = TestUtils::get_dpcpp_test_policy();
    sycl::queue q = policy.queue();
    if (q.get_device().has(sycl::aspect::usm_shared_allocations))
    {
        using r1_alloc_t = sycl::usm_allocator<A, sycl::usm::alloc::shared>;
        using r2_alloc_t = sycl::usm_allocator<B, sycl::usm::alloc::shared>;
        using r_out_alloc_t = sycl::usm_allocator<int, sycl::usm::alloc::shared>;

        std::vector<A, r1_alloc_t> v1({{1}, {2}, {3}}, r1_alloc_t(q));
        std::vector<B, r2_alloc_t> v2({{0}, {2}, {2}, {3}}, r2_alloc_t(q));
        std::vector<int> out_expected = {2, 3};

        std::vector<int, r_out_alloc_t> out(out_expected.size(), 0xCD, r_out_alloc_t(q));

        // Wrap vector with a USM allocator into the subrange because it is not device copyable
        std::ranges::subrange r1(v1.data(), v1.data() + v1.size());
        std::ranges::subrange r2(v2.data(), v2.data() + v2.size());
        std::ranges::subrange r_out(out.data(), out.data() + out.size());

        auto proj_a = [](const A& a) { return a.a; };
        auto proj_b = [](const B& b) { return b.b; };

        oneapi::dpl::ranges::set_intersection(policy, r1, r2, r_out, std::ranges::less{}, proj_a, proj_b);
        EXPECT_EQ_RANGES(out_expected, out, "wrong result with device policy");
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT
#endif // _ENABLE_STD_RANGES_TESTING

int
main()
{
    bool bProcessed = false;

#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // TODO: use data_in_in_out_lim when set_intersection supports
    // output range not-sufficiently large to hold all the processed elements

    // TODO: implement individual tests solely for seq policy
    auto set_intersection_checker = [](auto&&... args)
    {
        return oneapi::dpl::ranges::set_intersection(oneapi::dpl::execution::seq,
                                                     std::forward<decltype(args)>(args)...);
    };

    test_range_algo<0, int, data_in_in_out, mul1_t, div3_t>{big_sz}(dpl_ranges::set_intersection, set_intersection_checker);
    test_range_algo<1, int, data_in_in_out, mul1_t, div3_t>{big_sz}(dpl_ranges::set_intersection, set_intersection_checker, std::ranges::less{}, proj);

    // Testing the cut-off with the serial implementation (less than __set_algo_cut_off)
    test_range_algo<2, int, data_in_in_out, mul1_t, div3_t>{100}(dpl_ranges::set_intersection, set_intersection_checker, std::ranges::less{}, proj, proj);

    test_range_algo<3,  P2, data_in_in_out, mul1_t, div3_t>{}(dpl_ranges::set_intersection, set_intersection_checker, std::ranges::less{}, &P2::x, &P2::x);
    test_range_algo<4,  P2, data_in_in_out, mul1_t, div3_t>{}(dpl_ranges::set_intersection, set_intersection_checker, std::ranges::less{}, &P2::proj, &P2::proj);

    // Testing partial intersection less than __set_algo_cut_off
    auto medium_shift = [](auto&& v) { return v + 400; };
    using ms_t = decltype(medium_shift);
    test_range_algo<5, int, data_in_in_out, mul1_t, ms_t>{600}(dpl_ranges::set_intersection, set_intersection_checker);

    // Testing no intersection
    auto large_shift = [](auto&& v) { return v + 5000; };
    using ls_t = decltype(large_shift);
    test_range_algo<6, int, data_in_in_out, mul1_t, ls_t>{1000}(dpl_ranges::set_intersection, set_intersection_checker);
    test_range_algo<7, int, data_in_in_out, ls_t, mul1_t>{1000}(dpl_ranges::set_intersection, set_intersection_checker);

    // Check if projections are applied to the right sequences and trigger a compile-time error if not
    test_mixed_types_host();
#if TEST_DPCPP_BACKEND_PRESENT
    test_mixed_types_device();
#endif

    bProcessed = true;

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(bProcessed);
}
