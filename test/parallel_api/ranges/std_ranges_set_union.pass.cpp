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
inline int out_size_with_empty_r2<std::remove_cvref_t<decltype(oneapi::dpl::ranges::set_union)>>(int r1_size)
{
    return r1_size;
}
template<>
inline int out_size_with_empty_r1<std::remove_cvref_t<decltype(oneapi::dpl::ranges::set_union)>>(int r2_size)
{
    return r2_size;
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
        A* d_r1 = sycl::malloc_shared<A>(3, q);
        B* d_r2 = sycl::malloc_shared<B>(4, q);

        d_r1[0] = {1};
        d_r1[1] = {2};
        d_r1[2] = {3};

        d_r2[0] = {0};
        d_r2[1] = {2};
        d_r2[2] = {2};
        d_r2[3] = {3};

        int* d_rout = sycl::malloc_shared<int>(5, q);
        d_rout[0] = 0xCD;
        d_rout[1] = 0xCD;
        d_rout[2] = 0xCD;
        d_rout[3] = 0xCD;
        d_rout[4] = 0xCD;

        std::vector<int> out_expected = {0, 1, 2, 2, 3};

        std::ranges::subrange r1_range(d_r1, d_r1 + 3);
        std::ranges::subrange r2_range(d_r2, d_r2 + 4);
        std::ranges::subrange rout_range(d_rout, d_rout + 5);

        auto proj_a = [](const A& a) { return a.a; };
        auto proj_b = [](const B& b) { return b.b; };

        oneapi::dpl::ranges::set_union(
            policy, r1_range, r2_range, rout_range, std::ranges::less{}, proj_a, proj_b);

        EXPECT_EQ_RANGES(out_expected, rout_range, "wrong result with device policy");

        sycl::free(d_r1, q);
        sycl::free(d_r2, q);
        sycl::free(d_rout, q);
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT
#endif // _ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto set_union_checker = [](std::ranges::random_access_range auto&& r1,
                                std::ranges::random_access_range auto&& r2,
                                std::ranges::random_access_range auto&& rout, auto&&... args)
    {
        auto res = std::ranges::set_union(std::forward<decltype(r1)>(r1), std::forward<decltype(r2)>(r2),
                                          std::ranges::begin(rout), std::forward<decltype(args)>(args)...);

        using ret_type = std::ranges::set_union_result<std::ranges::borrowed_iterator_t<decltype(r1)>,
                                                       std::ranges::borrowed_iterator_t<decltype(r2)>,
                                                       std::ranges::borrowed_iterator_t<decltype(rout)>>;
        return ret_type{res.in1, res.in2, res.out};
    };

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
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
