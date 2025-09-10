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
namespace test_std_ranges
{
template<>
inline int out_size_with_empty_in2<std::remove_cvref_t<decltype(oneapi::dpl::ranges::set_union)>>(int in1_size)
{
    return in1_size;
}
template<>
inline int out_size_with_empty_in1<std::remove_cvref_t<decltype(oneapi::dpl::ranges::set_union)>>(int in2_size)
{
    return in2_size;
}
}

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

void test_serial_set_union()
{
    std::vector<int> v1 = {1, 2, 3, 3, 3, 4, 5};
    std::vector<int> v2 = {0, 2, 2, 3, 3, 7};
    std::vector<int> out(10, 0xCD);
    std::vector<int> out_expected = {0, 1, 2, 2, 3, 3, 3, 4, 5, 7};
    {
        // Smaller r2
        auto res = oneapi::dpl::ranges::set_union(oneapi::dpl::execution::seq, v1, v2, out);
        EXPECT_EQ_RANGES(out_expected, out, "wrong result with seq policy");
        EXPECT_EQ(std::ranges::size(v1), res.in1 - std::ranges::begin(v1), "wrong res.in1");
        EXPECT_EQ(std::ranges::size(v2), res.in2 - std::ranges::begin(v2), "wrong res.in2");
        EXPECT_EQ(std::ranges::size(out), res.out - std::ranges::begin(out), "wrong res.out");
    }
    {
        // Insufficient output range capacity, predicate
        std::ranges::fill(out, 0xCD);
        const int out_n = 5;
        auto out_subrange = std::ranges::subrange(out.data(), out.data() + out_n);
        auto out_expected_subrange = std::ranges::subrange(out_expected.data(), out_expected.data() + out_n);
        auto res = oneapi::dpl::ranges::set_union(oneapi::dpl::execution::seq, v1, v2, out_subrange, std::ranges::less{});
        EXPECT_EQ_RANGES(out_expected_subrange, out_subrange,
                         "wrong result with seq policy, case with insufficient out range capacity");
        EXPECT_EQ(3, res.in1 - std::ranges::begin(v1), "wrong res.in1");
        EXPECT_EQ(4, res.in2 - std::ranges::begin(v2), "wrong res.in2");
        EXPECT_EQ(out_n, res.out - std::ranges::begin(out_subrange), "wrong res.out");
    }
    {
        // Smaller r1, predicate + first projection
        std::ranges::fill(out, 0xCD);
        const int in1_n = 2;
        const int in2_n = 4;
        auto res = oneapi::dpl::ranges::set_union(oneapi::dpl::execution::seq,
                                                  std::ranges::take_view(v1, in1_n),
                                                  std::ranges::take_view(v2, in2_n), out,
                                                  std::ranges::less{}, [](auto x) { return x; });
        const int exp_out_n = 5;
        auto out_expected_subrange = std::ranges::subrange(out_expected.data(), out_expected.data() + exp_out_n);
        auto out_subrange = std::ranges::subrange(std::ranges::begin(out), res.out);
        EXPECT_EQ_RANGES(out_expected_subrange, out_subrange, "wrong result with seq policy, smaller r1");
    }
    {
        // Empty sequences
        std::vector<int> v1_empty;
        std::vector<int> v2_empty;
        auto res = oneapi::dpl::ranges::set_union(oneapi::dpl::execution::seq, v1_empty, v2_empty, out);
        EXPECT_EQ(0, res.in1 - std::ranges::begin(v1_empty), "wrong res.in1");
        EXPECT_EQ(0, res.in2 - std::ranges::begin(v2_empty), "wrong res.in2");
        EXPECT_EQ(0, res.out - std::ranges::begin(out), "wrong res.out");
    }
    {
        // Empty r1
        std::ranges::fill(out, 0xCD);
        std::vector<int> v1_empty;
        auto res = oneapi::dpl::ranges::set_union(oneapi::dpl::execution::seq, v1_empty, v2, out);
        auto out_subrange_expected = std::views::all(v2);
        auto out_subrange = std::ranges::subrange(std::ranges::begin(out), res.out);
        EXPECT_EQ_RANGES(out_subrange_expected, out_subrange, "wrong result with seq policy, empty r1");
        EXPECT_EQ(0, res.in1 - std::ranges::begin(v1_empty), "wrong res.in1");
        EXPECT_EQ(std::ranges::size(v2), res.in2 - std::ranges::begin(v2), "wrong res.in2");
        EXPECT_EQ(std::ranges::size(v2), res.out - std::ranges::begin(out), "wrong res.out");
    }
    {
        // Empty r2
        std::ranges::fill(out, 0xCD);
        std::vector<int> v2_empty;
        auto res = oneapi::dpl::ranges::set_union(oneapi::dpl::execution::seq, v1, v2_empty, out);
        auto out_subrange_expected = std::views::all(v1);
        auto out_subrange = std::ranges::subrange(std::ranges::begin(out), res.out);
        EXPECT_EQ_RANGES(out_subrange_expected, out_subrange, "wrong result with seq policy, empty r2");
        EXPECT_EQ(std::ranges::size(v1), res.in1 - std::ranges::begin(v1), "wrong res.in1");
        EXPECT_EQ(0, res.in2 - std::ranges::begin(v2_empty), "wrong res.in2");
        EXPECT_EQ(std::ranges::size(v1), res.out - std::ranges::begin(out), "wrong res.out");
    }
    {
        // - When a pair matches, an element from the first sequence is copied
        // - Order is preserved
        // - Predicate + two projections
        // TODO: use zip_view with c++23 onwards
        std::vector<std::pair<int, int>> kv1 = {{1, 1}, {3, 11}, {3, 12}, {4, 1}};
        std::vector<std::pair<int, int>> kv2 = {{0, 2}, {3, 2}, {3, 2}, {3, 21}};
        std::vector<std::pair<int, int>> kv_out(6, {0xCD, 0xCD});
        std::vector<int> k_out_expected = {0, 1, 3, 3, 3, 4};
        std::vector<int> v_out_expected = {2, 1, 11, 12, 21, 1};
        std::vector<int> k_out(std::ranges::size(kv_out));
        std::vector<int> v_out(std::ranges::size(kv_out));

        auto proj = [](const std::pair<int, int>& p) { return p.first; };
        oneapi::dpl::ranges::set_union(oneapi::dpl::execution::seq, kv1, kv2, kv_out, std::ranges::less{}, proj, proj);

        std::ranges::transform(kv_out, k_out.begin(), [](const auto& p) { return p.first; });
        std::ranges::transform(kv_out, v_out.begin(), [](const auto& p) { return p.second; });
        EXPECT_EQ_RANGES(k_out_expected, k_out, "wrong result with seq policy, wrong keys");
        EXPECT_EQ_RANGES(v_out_expected, v_out, "wrong result with seq policy, wrong values");
    }
    {
        // Reverse order
        std::vector<int> v3 = {3, 2, 1};
        std::vector<int> v4 = {2, 1, 0};
        std::vector<int> out2(4, 0xCD);
        std::vector<int> out2_expected = {3, 2, 1, 0};
        oneapi::dpl::ranges::set_union(oneapi::dpl::execution::seq, v3, v4, out2, std::ranges::greater{});
        EXPECT_EQ_RANGES(out2_expected, out2, "wrong result with seq policy");
    }
    // Different projections/types are tested in test_mixed_types_host

    // TODO: test type requirements
}

void test_mixed_types_host()
{
    std::vector<A> r1 = {{1}, {2}, {3}};
    std::vector<B> r2 = {{0}, {2}, {2}, {3}};

    std::vector<int> out_expected = {0, 1, 2, 2, 3};

    std::vector<int> out_seq(5, 0xCD);
    std::vector<int> out_par(5, 0xCD);
    std::vector<int> out_unseq(5, 0xCD);
    std::vector<int> out_par_unseq(5, 0xCD);

    auto proj_a = [](const A& a) { return a.a; };
    auto proj_b = [](const B& b) { return b.b; };

    oneapi::dpl::ranges::set_union(
        oneapi::dpl::execution::seq, r1, r2, out_seq, std::ranges::less{}, proj_a, proj_b);
    oneapi::dpl::ranges::set_union(
        oneapi::dpl::execution::par, r1, r2, out_par, std::ranges::less{}, proj_a, proj_b);
    oneapi::dpl::ranges::set_union(
        oneapi::dpl::execution::unseq, r1, r2, out_unseq, std::ranges::less{}, proj_a, proj_b);
    oneapi::dpl::ranges::set_union(
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
        std::vector<int, r_out_alloc_t> out(5, 0xCD, r_out_alloc_t(q));
        std::vector<int> out_expected = {0, 1, 2, 2, 3};

        // Wrap vector with a USM allocator into the subrange because it is not device copyable
        std::ranges::subrange r1(v1.data(), v1.data() + 3);
        std::ranges::subrange r2(v2.data(), v2.data() + 4);
        std::ranges::subrange r_out(out.data(), out.data() + 5);

        auto proj_a = [](const A& a) { return a.a; };
        auto proj_b = [](const B& b) { return b.b; };

        oneapi::dpl::ranges::set_union(policy, r1, r2, r_out, std::ranges::less{}, proj_a, proj_b);
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

    test_serial_set_union();
    auto set_union_checker = [](auto&&... args)
    {
        return oneapi::dpl::ranges::set_union(oneapi::dpl::execution::seq,
                                              std::forward<decltype(args)>(args)...);
    };

    // TODO: use data_in_in_out_lim when set_union supports
    // output range not-sufficiently large to hold all the processed elements

    test_range_algo<0, int, data_in_in_out, mul1_t, div3_t>{big_sz}(dpl_ranges::set_union, set_union_checker);
    test_range_algo<1, int, data_in_in_out, mul1_t, div3_t>{big_sz}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, proj);

    // Testing the cut-off with the serial implementation (less than __set_algo_cut_off)
    test_range_algo<2, int, data_in_in_out, mul1_t, div3_t>{100}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, proj, proj);

    test_range_algo<3,  P2, data_in_in_out, mul1_t, div3_t>{}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, &P2::x, &P2::x);
    test_range_algo<4,  P2, data_in_in_out, mul1_t, div3_t>{}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, &P2::proj, &P2::proj);

    test_mixed_types_host();
#if TEST_DPCPP_BACKEND_PRESENT
    test_mixed_types_device();
#endif // TEST_DPCPP_BACKEND_PRESENT

    bProcessed = true;

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(bProcessed);
}
